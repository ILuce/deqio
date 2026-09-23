from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .backends import BackendRuntime
from .config import Settings, load_settings
from .ui import DASHBOARD


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SETTINGS: Settings = load_settings()
SETTINGS.log_path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

runtime: BackendRuntime | None = None

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
    "cache_clears": 0,
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


def _runtime() -> BackendRuntime:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model runtime is not ready")
    return runtime


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


def format_shared_timing(timing: dict) -> dict:
    return {
        key.replace("_seconds", "_ms"): (
            round(value * 1000, 3)
            if key.endswith("_seconds") and isinstance(value, (int, float))
            else value
        )
        for key, value in timing.items()
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
        "backend": SETTINGS.backend,
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
        with SETTINGS.log_path.open("a", encoding="utf-8") as f:
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
        f"[SemIf] {SETTINGS.backend:<8} {mode:<6} "
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
            raw = _runtime().score(row, request.mode)

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
    global runtime
    global started_at

    print(
        f"[SemIf] Loading backend={SETTINGS.backend} model={SETTINGS.model}",
        flush=True,
    )

    runtime = BackendRuntime.load(SETTINGS)

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
        runtime.score(warmup, "serial")
        runtime.score(warmup, "serial")

    started_at = time.time()

    print("[SemIf] Ready.", flush=True)

    try:
        yield
    finally:
        with inference_lock:
            if runtime is not None:
                runtime.close()
            runtime = None


app = FastAPI(
    title="SemIf Local Decision Server",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "ok" if runtime is not None else "starting",
        "backend": SETTINGS.backend,
        "model": SETTINGS.model,
        "revision": SETTINGS.model_revision,
        "max_tokens": SETTINGS.max_tokens,
        "mlx_cache_mib": SETTINGS.mlx_cache_mib,
        "config": str(SETTINGS.config_path),
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
            raw_results, timing = _runtime().score_shared(rows)

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

    shared_timing = format_shared_timing(timing)
    batch_ms = shared_timing.get("total_ms")

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
        "shared_timing": shared_timing,
    }


@app.post("/v1/cache/clear")
def clear_cache():
    try:
        with inference_lock:
            details = _runtime().clear_cache()
    except Exception as exc:
        with stats_lock:
            stats["errors"] += 1
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with stats_lock:
        stats["cache_clears"] += 1

    return {
        "status": "ok",
        **details,
    }


@app.get("/v1/stats")
def get_stats():
    with stats_lock:
        values = list(latencies)

        return {
            **stats,
            "backend": SETTINGS.backend,
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


@app.get("/ui", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD
