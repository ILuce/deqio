from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from semif_phase1 import mlx_backend


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = Path(
    os.environ.get(
        "SEMIF_MODEL",
        "models/semif-qwen3.5-4b-mlx-4bit",
    )
).resolve()

MODEL_REVISION = os.environ.get(
    "SEMIF_MODEL_REVISION",
    "local-vinci00-semif-qwen35-4b-mlx4",
)

MAX_TOKENS = int(os.environ.get("SEMIF_MAX_TOKENS", "4096"))
MLX_CACHE_MIB = int(os.environ.get("SEMIF_MLX_CACHE_MIB", "256"))

LOG_PATH = Path(
    os.environ.get("SEMIF_LOG", "logs/requests.jsonl")
)

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

model = None
tokenizer = None
metadata = None
serial_scorer = None

inference_lock = threading.Lock()
stats_lock = threading.Lock()
log_lock = threading.Lock()

started_at = time.time()

latencies = deque(maxlen=2000)
recent_requests = deque(maxlen=50)

stats = {
    "requests": 0,
    "decisions": 0,
    "errors": 0,
    "cache_hits": 0,
}


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

State = str | dict[str, Any] | list[Any]


class Option(BaseModel):
    id: str
    description: str


class DecisionRequest(BaseModel):
    state: State
    question: str
    options: list[Option]
    id: str | None = None
    mode: Literal["direct", "serial"] = "serial"


class NoulRequest(BaseModel):
    state: State
    question: str
    id: str | None = None
    mode: Literal["direct", "serial"] = "serial"


class SharedDecision(BaseModel):
    question: str
    options: list[Option]
    id: str | None = None


class SharedRequest(BaseModel):
    state: State
    decisions: list[SharedDecision]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def state_hash(state: State) -> str:
    if isinstance(state, str):
        raw = state
    else:
        raw = json.dumps(
            state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    index = round((len(ordered) - 1) * p)

    return ordered[index]


def format_result(result: dict) -> dict:
    probabilities = {
        option_id: float(probability)
        for option_id, probability in zip(
            result["option_ids"],
            result["probabilities"],
        )
    }

    logits = {
        option_id: float(logit)
        for option_id, logit in zip(
            result["option_ids"],
            result["option_logits"],
        )
    }

    decision = max(probabilities, key=probabilities.get)

    timing = {}

    for key in (
        "total_seconds",
        "forward_seconds",
        "prefill_seconds",
        "copy_seconds",
        "suffix_forward_seconds",
    ):
        if key in result:
            timing[key.replace("_seconds", "_ms")] = round(
                result[key] * 1000,
                3,
            )

    if "cache_hit" in result:
        timing["cache_hit"] = bool(result["cache_hit"])

    return {
        "id": result["id"],
        "decision": decision,
        "probabilities": probabilities,
        "top_probability": probabilities[decision],
        "option_logits": logits,
        "input_tokens": result["input_tokens"],
        "timing": timing,
        "prompt_sha256": result["prompt_sha256"],
        "probability_status": result["probability_status"],
    }


def record_event(
    *,
    request_id: str,
    mode: str,
    state: State,
    question: str,
    result: dict,
    decisions: int = 1,
):
    latency_ms = result.get("timing", {}).get("total_ms")

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "id": request_id,
        "mode": mode,
        "state_hash": state_hash(state),
        "question": question,
        "decision": result.get("decision"),
        "probabilities": result.get("probabilities"),
        "latency_ms": latency_ms,
        "input_tokens": result.get("input_tokens"),
        "cache_hit": result.get("timing", {}).get("cache_hit"),
        "decisions": decisions,
    }

    with stats_lock:
        stats["requests"] += 1
        stats["decisions"] += decisions

        if latency_ms is not None:
            latencies.append(float(latency_ms))

        if event["cache_hit"]:
            stats["cache_hits"] += 1

        recent_requests.appendleft(event)

    with log_lock:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

    probability = result.get("top_probability")

    probability_text = (
        f"{probability:.4f}"
        if probability is not None
        else "-"
    )

    print(
        f"[SemIf] {mode:<6} "
        f"id={request_id} "
        f"decision={result.get('decision')} "
        f"p={probability_text} "
        f"latency={latency_ms}ms "
        f"cache={event['cache_hit']}",
        flush=True,
    )


def run_decision(request: DecisionRequest) -> dict:
    request_id = request.id or uuid4().hex

    row = {
        "id": request_id,
        "state": request.state,
        "question": request.question,
        "options": [
            option.model_dump()
            for option in request.options
        ],
    }

    try:
        with inference_lock:
            if request.mode == "serial":
                raw = serial_scorer.score(row)
            else:
                raw = mlx_backend.score(
                    model,
                    tokenizer,
                    row,
                    metadata,
                    MAX_TOKENS,
                )

    except Exception as exc:
        with stats_lock:
            stats["errors"] += 1

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    result = format_result(raw)

    record_event(
        request_id=request_id,
        mode=request.mode,
        state=request.state,
        question=request.question,
        result=result,
    )

    return result


# ---------------------------------------------------------------------------
# Startup / model loading
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global tokenizer
    global metadata
    global serial_scorer
    global started_at

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "The current server backend requires macOS on Apple Silicon "
            "(Darwin arm64)."
        )

    if not MODEL_PATH.is_dir():
        raise RuntimeError(
            f"Model directory does not exist: {MODEL_PATH}. "
            "Download the configured model before starting the server; "
            "see README.md."
        )

    print(f"[SemIf] Loading model: {MODEL_PATH}", flush=True)

    model, tokenizer, metadata = mlx_backend.load_model(
        str(MODEL_PATH),
        MODEL_REVISION,
        bits=None,
        cache_limit_mib=MLX_CACHE_MIB,
    )

    serial_scorer = mlx_backend.SerialPrefixScorer(
        model,
        tokenizer,
        metadata,
        MAX_TOKENS,
    )

    print("[SemIf] Model loaded. Running warmup...", flush=True)

    warmup = {
        "id": "warmup",
        "state": "The system is ready.",
        "question": "Is the system ready?",
        "options": [
            {
                "id": "yes",
                "description": "Yes, the system is ready.",
            },
            {
                "id": "no",
                "description": "No, the system is not ready.",
            },
        ],
    }

    with inference_lock:
        serial_scorer.score(warmup)
        serial_scorer.score(warmup)

    started_at = time.time()

    print("[SemIf] Ready.", flush=True)

    yield


app = FastAPI(
    title="SemIf Local Decision Server",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "mlx",
        "model": str(MODEL_PATH),
        "revision": MODEL_REVISION,
        "max_tokens": MAX_TOKENS,
        "mlx_cache_mib": MLX_CACHE_MIB,
    }


@app.post("/v1/choice")
@app.post("/v1/decision")
def decision(request: DecisionRequest):
    return run_decision(request)


@app.post("/v1/noul")
def noul(request: NoulRequest):
    decision_request = DecisionRequest(
        id=request.id,
        state=request.state,
        question=request.question,
        mode=request.mode,
        options=[
            Option(
                id="yes",
                description=(
                    "Yes. The evidence supports the criterion "
                    "or question."
                ),
            ),
            Option(
                id="no",
                description=(
                    "No. The evidence does not support the "
                    "criterion or question."
                ),
            ),
        ],
    )

    return run_decision(decision_request)


@app.post("/v1/shared")
def shared(request: SharedRequest):
    if not request.decisions:
        raise HTTPException(
            status_code=400,
            detail="At least one decision is required.",
        )

    rows = []

    for item in request.decisions:
        rows.append(
            {
                "id": item.id or uuid4().hex,
                "state": request.state,
                "question": item.question,
                "options": [
                    option.model_dump()
                    for option in item.options
                ],
            }
        )

    try:
        with inference_lock:
            raw_results, timing = mlx_backend.score_shared(
                model,
                tokenizer,
                rows,
                metadata,
                MAX_TOKENS,
            )

    except Exception as exc:
        with stats_lock:
            stats["errors"] += 1

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    results = [
        format_result(raw)
        for raw in raw_results
    ]

    batch_ms = round(timing["total_seconds"] * 1000, 3)

    event = {
        "id": "shared-" + uuid4().hex[:12],
        "decision": f"{len(results)} decisions",
        "probabilities": None,
        "input_tokens": None,
        "top_probability": None,
        "timing": {
            "total_ms": batch_ms,
            "batch_size": len(results),
        },
    }

    record_event(
        request_id=event["id"],
        mode="shared",
        state=request.state,
        question=f"{len(results)} shared decisions",
        result=event,
        decisions=len(results),
    )

    return {
        "results": results,
        "shared_timing": {
            key.replace("_seconds", "_ms"):
                round(value * 1000, 3)
                if key.endswith("_seconds")
                else value
            for key, value in timing.items()
        },
    }


@app.get("/v1/stats")
def get_stats():
    with stats_lock:
        values = list(latencies)

        return {
            **stats,
            "uptime_seconds": round(
                time.time() - started_at,
                1,
            ),
            "latency_ms": {
                "average": (
                    round(sum(values) / len(values), 3)
                    if values
                    else None
                ),
                "p50": percentile(values, 0.50),
                "p95": percentile(values, 0.95),
                "samples": len(values),
            },
        }


@app.get("/v1/recent")
def get_recent():
    with stats_lock:
        return list(recent_requests)


# ---------------------------------------------------------------------------
# Tiny dashboard
# ---------------------------------------------------------------------------

DASHBOARD = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SemIf</title>
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    max-width: 1100px;
    margin: 40px auto;
    padding: 0 24px;
    background: #f6f6f6;
    color: #181818;
}
h1 { margin-bottom: 4px; }
.sub { color: #666; margin-bottom: 28px; }
.cards {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 28px;
}
.card {
    background: white;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 16px;
}
.card small {
    display:block;
    color:#777;
    margin-bottom:6px;
}
.card strong { font-size:24px; }
table {
    width:100%;
    border-collapse:collapse;
    background:white;
    border:1px solid #ddd;
}
th, td {
    text-align:left;
    padding:10px;
    border-bottom:1px solid #eee;
    font-size:13px;
}
th { background:#fafafa; }
.yes { font-weight:600; }
</style>
</head>

<body>

<h1>SemIf Local</h1>
<div class="sub">Qwen3.5-4B · MLX · native option logits</div>

<div class="cards">
  <div class="card">
    <small>Requests</small>
    <strong id="requests">-</strong>
  </div>
  <div class="card">
    <small>Decisions</small>
    <strong id="decisions">-</strong>
  </div>
  <div class="card">
    <small>P50 latency</small>
    <strong id="p50">-</strong>
  </div>
  <div class="card">
    <small>P95 latency</small>
    <strong id="p95">-</strong>
  </div>
</div>

<table>
<thead>
<tr>
  <th>Time</th>
  <th>Mode</th>
  <th>Question</th>
  <th>Decision</th>
  <th>Latency</th>
  <th>Cache</th>
</tr>
</thead>
<tbody id="rows"></tbody>
</table>

<script>
async function update() {
    const [stats, recent] = await Promise.all([
        fetch('/v1/stats').then(r => r.json()),
        fetch('/v1/recent').then(r => r.json())
    ]);

    document.getElementById('requests').textContent =
        stats.requests;

    document.getElementById('decisions').textContent =
        stats.decisions;

    document.getElementById('p50').textContent =
        stats.latency_ms.p50 == null
            ? '-'
            : stats.latency_ms.p50.toFixed(1) + ' ms';

    document.getElementById('p95').textContent =
        stats.latency_ms.p95 == null
            ? '-'
            : stats.latency_ms.p95.toFixed(1) + ' ms';

    document.getElementById('rows').innerHTML =
        recent.map(x => `
        <tr>
          <td>${new Date(x.timestamp).toLocaleTimeString()}</td>
          <td>${x.mode}</td>
          <td>${x.question}</td>
          <td>${x.decision ?? '-'}</td>
          <td>${x.latency_ms == null ? '-' : x.latency_ms + ' ms'}</td>
          <td>${x.cache_hit == null ? '-' : x.cache_hit}</td>
        </tr>
        `).join('');
}

update();
setInterval(update, 1000);
</script>

</body>
</html>
"""


@app.get("/ui", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD