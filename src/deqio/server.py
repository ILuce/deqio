from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from . import __version__
from pydantic import BaseModel

from .backends import BackendRuntime
from .benchmark_store import list_benchmark_runs, read_benchmark_results, read_benchmark_summary
from .catalog import apply_selection, get_model, get_profile, load_catalog, public_catalog
from .config import (
    MODEL_SELECTION_ENV_VARS,
    Settings,
    load_settings,
    read_config_data,
    settings_from_data,
    write_config_data,
)
from .console import (
    info,
    log_cache_clear,
    log_request_error,
    log_request_success,
    model_switch_ok,
    model_switch_restored,
    model_switch_rollback,
    model_switch_start,
    server_ready,
    startup_header,
    warmup_ok,
)
from .installations import installed_profiles, mark_installed
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
switching_runtime = False

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


class ModelActivateRequest(BaseModel):
    model_id: str
    backend: Literal["mlx", "mps", "cuda"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runtime() -> BackendRuntime:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model runtime is not ready")
    return runtime


def _active_model() -> dict[str, Any]:
    return {
        "engine": SETTINGS.engine,
        "model_id": SETTINGS.model_id,
        "backend": SETTINGS.backend,
        "model": SETTINGS.model,
    }


def _installation_rows() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_path, config_data = read_config_data(SETTINGS.config_path)
    catalog_path = SETTINGS.model_catalog
    catalog = load_catalog(catalog_path)
    rows = installed_profiles(
        config_path=config_path,
        config_data=config_data,
        catalog=catalog,
        active_model_id=SETTINGS.model_id,
        active_backend=SETTINGS.backend,
    )
    return catalog, rows


def _reset_session_metrics() -> None:
    global started_at
    with stats_lock:
        for key in stats:
            stats[key] = 0
        latencies.clear()
        recent_requests.clear()
        started_at = time.time()


def _warmup_runtime(target: BackendRuntime, *, announce: bool) -> None:
    warmup_row = {
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
    target.score(warmup_row, "serial")
    if announce:
        warmup_ok(1, 2)
    target.score(warmup_row, "serial")
    if announce:
        warmup_ok(2, 2)


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

    raw_logits = result.get("option_logits") or []
    logits = {
        option_id: float(logit)
        for option_id, logit in zip(
            result["option_ids"],
            raw_logits,
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
    endpoint: str,
    decisions: int = 1,
    settings_snapshot: Settings | None = None,
):
    used_settings = settings_snapshot or SETTINGS
    latency_ms = result.get("timing", {}).get("total_ms")

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "id": request_id,
        "engine": used_settings.engine,
        "model_id": used_settings.model_id,
        "backend": used_settings.backend,
        "mode": mode,
        "endpoint": endpoint,
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
        with used_settings.log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

    log_request_success(
        endpoint,
        request_id=request_id,
        result=result,
        mode=mode,
        decisions=decisions,
    )


def run_decision(request: DecisionRequest, *, endpoint: str = "/v1/choice") -> dict:
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
            settings_snapshot = SETTINGS
            raw = _runtime().score(row, request.mode)

    except Exception as exc:
        with stats_lock:
            stats["errors"] += 1
        log_request_error(endpoint, request_id=request_id, error=exc, status=400)

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
        endpoint=endpoint,
        settings_snapshot=settings_snapshot,
    )

    return result


# ---------------------------------------------------------------------------
# Startup / model loading
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime
    global started_at

    startup_header(
        engine=SETTINGS.engine,
        model_id=SETTINGS.model_id,
        backend=SETTINGS.backend,
        model=SETTINGS.model,
        config=str(SETTINGS.config_path),
    )

    runtime = BackendRuntime.load(SETTINGS)

    runtime_kind = "sidecar"
    info(f"Runtime loaded: {runtime_kind}. Running warmup...")

    with inference_lock:
        _warmup_runtime(runtime, announce=True)

    mark_installed(
        SETTINGS.config_path,
        SETTINGS.model_id,
        SETTINGS.backend,
        verified=True,
        source="startup",
    )
    started_at = time.time()

    server_ready()

    try:
        yield
    finally:
        with inference_lock:
            if runtime is not None:
                runtime.close()
            runtime = None


app = FastAPI(
    title="Deqio Decision Server",
    version=__version__,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui", status_code=307)


@app.get("/health")
def health():
    status = "switching" if switching_runtime else ("ok" if runtime is not None else "starting")
    return {
        "status": status,
        "engine": SETTINGS.engine,
        "model_id": SETTINGS.model_id,
        "backend": SETTINGS.backend,
        "model": SETTINGS.model,
        "revision": SETTINGS.model_revision,
        "max_tokens": SETTINGS.max_tokens,
        "mlx_cache_mib": SETTINGS.mlx_cache_mib,
        "config": str(SETTINGS.config_path),
    }


@app.post("/v1/choice")
@app.post("/v1/decision")
def decision(payload: DecisionRequest, http_request: Request):
    return run_decision(payload, endpoint=http_request.url.path)


@app.post("/v1/noul")
def noul(payload: NoulRequest):
    request_id = payload.id or uuid4().hex
    endpoint = "/v1/noul"
    row = {
        "id": request_id,
        "state": payload.state,
        "question": payload.question,
    }
    try:
        with inference_lock:
            settings_snapshot = SETTINGS
            raw = _runtime().score_noul(row, payload.mode)
    except Exception as exc:
        with stats_lock:
            stats["errors"] += 1
        log_request_error(endpoint, request_id=request_id, error=exc, status=400)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = format_result(raw)
    record_event(
        request_id=request_id,
        mode=payload.mode,
        state=payload.state,
        question=payload.question,
        result=result,
        endpoint=endpoint,
        settings_snapshot=settings_snapshot,
    )
    return result


@app.post("/v1/shared")
def shared(payload: SharedRequest):
    endpoint = "/v1/shared"
    request_id = "shared-" + uuid4().hex[:12]
    if not payload.decisions:
        error = "At least one decision is required."
        with stats_lock:
            stats["errors"] += 1
        log_request_error(endpoint, request_id=request_id, error=error, status=400)
        raise HTTPException(status_code=400, detail=error)

    rows = []

    for item in payload.decisions:
        rows.append(
            {
                "id": item.id or uuid4().hex,
                "state": payload.state,
                "question": item.question,
                "options": [
                    option.model_dump()
                    for option in item.options
                ],
            }
        )

    try:
        with inference_lock:
            settings_snapshot = SETTINGS
            raw_results, timing = _runtime().score_shared(rows)

    except Exception as exc:
        with stats_lock:
            stats["errors"] += 1
        log_request_error(endpoint, request_id=request_id, error=exc, status=400)

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
        "id": request_id,
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
        state=payload.state,
        question=f"{len(results)} shared decisions",
        result=event,
        endpoint=endpoint,
        decisions=len(results),
        settings_snapshot=settings_snapshot,
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

    log_cache_clear(details=details)
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
            "engine": SETTINGS.engine,
            "model_id": SETTINGS.model_id,
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


@app.get("/v1/models")
def get_models():
    catalog, rows = _installation_rows()
    return {
        "active": _active_model(),
        "installed": [row for row in rows if row["installed"] and row["host_compatible"]],
        "models": public_catalog(catalog),
    }


@app.get("/v1/models/installed")
def get_installed_models():
    _, rows = _installation_rows()
    return {
        "active": _active_model(),
        "installed": [row for row in rows if row["installed"] and row["host_compatible"]],
    }


@app.post("/v1/models/activate")
def activate_model(payload: ModelActivateRequest):
    global SETTINGS
    global runtime
    global switching_runtime

    pinned = [name for name in MODEL_SELECTION_ENV_VARS if os.environ.get(name) not in (None, "")]
    if pinned:
        raise HTTPException(
            status_code=409,
            detail=(
                "Live model switching is disabled while model selection is pinned by environment variables: "
                + ", ".join(pinned)
            ),
        )

    catalog, rows = _installation_rows()
    selected = next(
        (
            row
            for row in rows
            if row["model_id"] == payload.model_id and row["backend"] == payload.backend
        ),
        None,
    )
    if selected is None:
        raise HTTPException(status_code=404, detail="Unknown model/backend profile")
    if not selected["host_compatible"]:
        raise HTTPException(status_code=409, detail="Selected backend is not compatible with this host")
    if not selected["installed"]:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Model profile is not installed. Run: uv run deqio models install {payload.model_id} "
                f"--backend {payload.backend}"
            ),
        )
    if payload.model_id == SETTINGS.model_id and payload.backend == SETTINGS.backend:
        return {
            "status": "already-active",
            "active": _active_model(),
            "load_ms": 0.0,
        }

    config_path, config_data = read_config_data(SETTINGS.config_path)
    entry = get_model(catalog, payload.model_id)
    profile = get_profile(catalog, payload.model_id, payload.backend)
    candidate_data = apply_selection(config_data, entry, profile, payload.backend)
    candidate_settings = settings_from_data(config_path, candidate_data, apply_environment=False)

    old_settings = SETTINGS
    target_name = f"{payload.model_id}/{payload.backend}"
    current_name = f"{old_settings.model_id}/{old_settings.backend}"
    started = time.perf_counter()

    with inference_lock:
        switching_runtime = True
        model_switch_start(current=current_name, target=target_name)
        old_runtime = runtime
        if old_runtime is not None:
            try:
                old_runtime.close()
            except Exception as exc:
                switching_runtime = False
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not stop the current runtime before switching: {exc}",
                ) from exc
        runtime = None

        candidate_runtime: BackendRuntime | None = None
        try:
            candidate_runtime = BackendRuntime.load(candidate_settings)
            _warmup_runtime(candidate_runtime, announce=False)
            write_config_data(config_path, candidate_data)
        except Exception as exc:
            if candidate_runtime is not None:
                try:
                    candidate_runtime.close()
                except Exception:
                    pass
            model_switch_rollback(target=target_name, error=exc)
            try:
                restored = BackendRuntime.load(old_settings)
                _warmup_runtime(restored, announce=False)
                runtime = restored
                SETTINGS = old_settings
                model_switch_restored(model_id=old_settings.model_id, backend=old_settings.backend)
            except Exception as rollback_error:
                runtime = None
                switching_runtime = False
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Failed to activate {target_name}: {exc}. "
                        f"Previous runtime could not be restored: {rollback_error}"
                    ),
                ) from exc
            switching_runtime = False
            raise HTTPException(
                status_code=500,
                detail=f"Failed to activate {target_name}: {exc}. Previous runtime was restored.",
            ) from exc

        runtime = candidate_runtime
        SETTINGS = candidate_settings
        SETTINGS.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            mark_installed(
                SETTINGS.config_path,
                SETTINGS.model_id,
                SETTINGS.backend,
                verified=True,
                source="activate",
            )
        except Exception as registry_error:
            info(f"Warning: could not update local installed-model registry: {registry_error}")
        _reset_session_metrics()
        switching_runtime = False

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    model_switch_ok(model_id=SETTINGS.model_id, backend=SETTINGS.backend, elapsed_ms=elapsed_ms)
    return {
        "status": "ok",
        "active": _active_model(),
        "load_ms": round(elapsed_ms, 3),
    }


@app.get("/v1/recent")
def get_recent():
    with stats_lock:
        return list(recent_requests)


@app.get("/v1/benchmarks")
def get_benchmarks():
    runs = list_benchmark_runs(SETTINGS.config_path)
    return {
        "available": bool(runs),
        "latest": runs[0]["id"] if runs else None,
        "runs": runs,
    }


@app.get("/v1/benchmarks/{run_id}/summary")
def get_benchmark_summary(run_id: str):
    try:
        return read_benchmark_summary(SETTINGS.config_path, run_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/v1/benchmarks/{run_id}/results")
def get_benchmark_results(run_id: str):
    try:
        rows = read_benchmark_results(SETTINGS.config_path, run_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"run_id": run_id, "results": rows}


@app.get("/ui", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD
