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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

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
from .input_contract import (
    INPUT_ATTESTATION_SCHEMA_VERSION,
    INPUT_COMPLETENESS_V1,
    DuplicateJSONKeyError,
    InputContractContext,
    UnsupportedInputContractError,
    decision_input_stub,
    parse_contract_header,
    sha256_bytes,
    unknown_input_receipt,
    validate_json_without_duplicate_keys,
)
from .ui import DASHBOARD, WATCH_DASHBOARD


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
watch_lock = threading.Lock()

started_at = time.time()

latencies = deque(maxlen=2000)
recent_requests = deque(maxlen=50)
watch_session_id = uuid4().hex
watch_started_at = time.time()
watch_session_model: dict[str, Any] = {}
watch_events: list[dict[str, Any]] = []

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


class InputPolicy(BaseModel):
    require_complete: bool = False
    overflow: Literal["reject"] = "reject"


class DecisionRequest(BaseModel):
    state: State
    question: str
    options: list[Option]
    id: str | None = None
    mode: Literal["direct", "serial"] = "serial"
    input_policy: InputPolicy | None = None


class NoulRequest(BaseModel):
    state: State
    question: str
    id: str | None = None
    mode: Literal["direct", "serial"] = "serial"
    input_policy: InputPolicy | None = None


class SharedDecision(BaseModel):
    question: str
    options: list[Option]
    id: str | None = None


class SharedRequest(BaseModel):
    state: State
    decisions: list[SharedDecision]
    input_policy: InputPolicy | None = None


class ModelActivateRequest(BaseModel):
    model_id: str
    backend: Literal["mlx", "mps", "cuda"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class InputContractHTTPError(Exception):
    def __init__(self, code: str, *, status_code: int = 422, **details: Any) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.payload = {"code": code, **details}


def _option_ids(options: list[Option]) -> list[str]:
    return [str(option.id) for option in options]


def _validate_unique_ids(values: list[str], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise InputContractHTTPError(
            "duplicate_id",
            field=label,
            inference_performed=False,
        )


async def _input_contract_context(
    http_request: Request,
    *,
    policy: InputPolicy | None,
    request_id: str | None,
    decision_ids: list[str],
    option_sets: list[list[str]],
) -> InputContractContext | None:
    try:
        version = parse_contract_header(http_request.headers.get("deqio-contract"))
    except UnsupportedInputContractError as error:
        raise InputContractHTTPError(
            "unsupported_contract",
            contract=str(error),
            supported=[INPUT_COMPLETENESS_V1],
            inference_performed=False,
        ) from error
    if version is None:
        return None

    raw_body = getattr(http_request.state, "input_contract_raw_body", None)
    request_body_sha256 = getattr(http_request.state, "input_contract_request_body_sha256", None)
    if raw_body is None or request_body_sha256 is None:
        # Fallback for direct unit calls that do not pass through ASGI middleware.
        raw_body = await http_request.body()
        request_body_sha256 = sha256_bytes(raw_body)
        try:
            validate_json_without_duplicate_keys(raw_body)
        except DuplicateJSONKeyError as error:
            raise InputContractHTTPError(
                "duplicate_json_key",
                request_id=request_id,
                request_body_sha256=request_body_sha256,
                detail=str(error),
                inference_performed=False,
            ) from error
        except ValueError as error:
            raise InputContractHTTPError(
                "invalid_request_body",
                request_id=request_id,
                request_body_sha256=request_body_sha256,
                detail=str(error),
                inference_performed=False,
            ) from error

    if policy is None:
        raise InputContractHTTPError(
            "input_policy_required",
            request_id=request_id,
            request_body_sha256=request_body_sha256,
            inference_performed=False,
        )
    if request_id is None and len(decision_ids) == 1:
        raise InputContractHTTPError(
            "request_id_required",
            request_body_sha256=request_body_sha256,
            inference_performed=False,
        )
    _validate_unique_ids(decision_ids, label="decision_ids")
    for option_ids in option_sets:
        _validate_unique_ids(option_ids, label="option_ids")

    return InputContractContext(
        version=version,
        request_body_sha256=request_body_sha256,
        require_complete=bool(policy.require_complete),
        overflow=policy.overflow,
    )


def _ensure_contract_capability(
    target: BackendRuntime,
    context: InputContractContext | None,
    *,
    request_id: str,
) -> None:
    if context is None or not context.require_complete:
        return
    capability_fn = getattr(target, "input_completeness_capability", None)
    capability = capability_fn() if callable(capability_fn) else {"status": "unavailable"}
    if not isinstance(capability, dict) or capability.get("status") != "complete":
        raise InputContractHTTPError(
            "input_completeness_unavailable",
            request_id=request_id,
            request_body_sha256=context.request_body_sha256,
            reason=(capability or {}).get("reason", "backend_model_input_not_instrumented")
            if isinstance(capability, dict)
            else "backend_model_input_not_instrumented",
            inference_performed=False,
        )


def _unknown_receipt_for_result(
    *,
    context: InputContractContext,
    runtime_identity: dict[str, Any],
    raw_result: dict[str, Any],
    settings_snapshot: Settings,
    decision_id: str,
    option_ids: list[str],
    components: list[str],
) -> dict[str, Any]:
    input_tokens = raw_result.get("input_tokens")
    return unknown_input_receipt(
        context=context,
        runtime_identity=runtime_identity,
        engine_payload_sha256=raw_result.get("engine_payload_sha256"),
        input_limit_tokens=settings_snapshot.max_tokens,
        decision_inputs=[
            decision_input_stub(
                decision_id=decision_id, option_ids=option_ids, components=components
            )
        ],
        input_tokens=int(input_tokens) if isinstance(input_tokens, int) else None,
        input_tokens_source=str(raw_result.get("input_tokens_source") or "unknown"),
    )


def _unknown_shared_receipt(
    *,
    context: InputContractContext,
    runtime_identity: dict[str, Any],
    raw_results: list[dict[str, Any]],
    settings_snapshot: Settings,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first = raw_results[0] if raw_results else {}
    tokens = first.get("input_tokens")
    return unknown_input_receipt(
        context=context,
        runtime_identity=runtime_identity,
        engine_payload_sha256=first.get("engine_payload_sha256"),
        input_limit_tokens=settings_snapshot.max_tokens,
        decision_inputs=[
            decision_input_stub(
                decision_id=str(row["id"]),
                option_ids=[str(option["id"]) for option in row["options"]],
                components=["state", "question", "options"],
            )
            for row in rows
        ],
        input_tokens=int(tokens) if isinstance(tokens, int) else None,
        input_tokens_source=str(first.get("input_tokens_source") or "unknown"),
    )


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


def _reset_watch_session() -> None:
    global watch_session_id
    global watch_started_at
    global watch_session_model
    identity = (
        _runtime_identity_snapshot(runtime, SETTINGS)
        if runtime is not None
        else {}
    )
    with watch_lock:
        watch_events.clear()
        watch_session_id = uuid4().hex
        watch_started_at = time.time()
        watch_session_model = {
            "engine": identity.get("engine", SETTINGS.engine),
            "model_id": identity.get("model_id", SETTINGS.model_id),
            "backend": identity.get("backend", SETTINGS.backend),
            "runtime_instance_id": identity.get("runtime_instance_id"),
        }


def _watch_session_token() -> str:
    with watch_lock:
        return watch_session_id


def _watch_json_snapshot(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    )


def _record_watch_event(
    *,
    expected_session_id: str,
    endpoint: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    status_code: int,
    request_id: str | None,
    mode: str,
    decisions: int,
    settings_snapshot: Settings,
) -> None:
    provenance = response_payload.get("provenance") if isinstance(response_payload, dict) else None
    runtime_identity = provenance.get("runtime", {}) if isinstance(provenance, dict) else {}
    timing = (
        response_payload.get("shared_timing", {})
        if endpoint == "/v1/shared"
        else response_payload.get("timing", {})
    ) if isinstance(response_payload, dict) else {}
    latency_ms = timing.get("total_ms") if isinstance(timing, dict) else None
    decision = response_payload.get("decision") if isinstance(response_payload, dict) else None
    top_probability = response_payload.get("top_probability") if isinstance(response_payload, dict) else None
    input_tokens = response_payload.get("input_tokens") if isinstance(response_payload, dict) else None
    if endpoint == "/v1/shared" and isinstance(response_payload, dict):
        results = response_payload.get("results")
        if isinstance(results, list):
            decision = f"{len(results)} decisions"
            token_values = [
                int(item["input_tokens"])
                for item in results
                if isinstance(item, dict) and isinstance(item.get("input_tokens"), int)
            ]
            input_tokens = sum(token_values) if token_values else None

    event = {
        "event_id": uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "method": "POST",
        "status_code": int(status_code),
        "request_id": request_id,
        "engine": runtime_identity.get("engine", settings_snapshot.engine),
        "model_id": runtime_identity.get("model_id", settings_snapshot.model_id),
        "backend": runtime_identity.get("backend", settings_snapshot.backend),
        "runtime_instance_id": runtime_identity.get("runtime_instance_id"),
        "mode": mode,
        "decisions": int(decisions),
        "decision": decision,
        "top_probability": top_probability,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "request": _watch_json_snapshot(request_payload),
        "response": _watch_json_snapshot(response_payload),
    }
    with watch_lock:
        # A model switch resets the session. A request finishing after that reset
        # must never leak its old-model payload into the new model session.
        if expected_session_id != watch_session_id:
            return
        watch_events.append(event)


def _watch_rows_snapshot() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with watch_lock:
        events = list(watch_events)
        session = {
            "id": watch_session_id,
            "started_at": datetime.fromtimestamp(watch_started_at, timezone.utc).isoformat(),
            **watch_session_model,
        }
    latencies_ms = [
        float(event["latency_ms"])
        for event in events
        if isinstance(event.get("latency_ms"), (int, float))
    ]
    session["requests"] = len(events)
    session["decisions"] = sum(int(event.get("decisions", 0)) for event in events)
    session["errors"] = sum(1 for event in events if int(event.get("status_code", 500)) >= 400)
    session["latency_ms"] = {
        "p50": percentile(latencies_ms, 0.50),
        "p95": percentile(latencies_ms, 0.95),
        "samples": len(latencies_ms),
    }
    row_keys = (
        "event_id", "timestamp", "endpoint", "method", "status_code",
        "request_id", "engine", "model_id", "backend", "runtime_instance_id",
        "mode", "decisions", "decision", "top_probability", "latency_ms", "input_tokens",
    )
    rows = [{key: event.get(key) for key in row_keys} for event in reversed(events)]
    return session, rows


def _reset_session_metrics() -> None:
    global started_at
    with stats_lock:
        for key in stats:
            stats[key] = 0
        latencies.clear()
        recent_requests.clear()
        started_at = time.time()
    _reset_watch_session()

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


def _runtime_identity_snapshot(target: BackendRuntime, settings_snapshot: Settings) -> dict[str, Any]:
    snapshot = getattr(target, "identity_snapshot", None)
    if callable(snapshot):
        value = snapshot()
        if isinstance(value, dict):
            return value
    return {
        "deqio_version": __version__,
        "runtime_instance_id": getattr(target, "runtime_instance_id", None),
        "engine": settings_snapshot.engine,
        "model_id": settings_snapshot.model_id,
        "backend": settings_snapshot.backend,
        "model": settings_snapshot.model,
        "requested_revision": settings_snapshot.model_revision,
        "artifacts": [],
        "artifact_revisions_resolved": False,
        "installation_verified_at": None,
    }


def _attestation_sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _bind_result_provenance(
    response: dict[str, Any],
    raw_result: dict[str, Any],
    runtime_identity: dict[str, Any],
    *,
    input_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = raw_result.get("score_provenance")
    if not isinstance(score, dict):
        score = {
            "kind": "unknown",
            "source": "runtime",
            "synthetic": None,
            "normalized": None,
            "transforms": [],
            "raw_logits_available": bool(response.get("option_logits")),
            "calibration": "unknown",
        }
    binding = {
        "request_id": response.get("id"),
        "prompt_sha256": response.get("prompt_sha256"),
        "decision": response.get("decision"),
        "probabilities": response.get("probabilities"),
        "runtime": runtime_identity,
        "score": score,
    }
    if input_receipt is not None:
        binding["input_receipt"] = input_receipt
    complete = bool(
        runtime_identity.get("runtime_instance_id")
        and runtime_identity.get("artifact_revisions_resolved")
        and score.get("kind") != "unknown"
        and (input_receipt is None or input_receipt.get("status") == "complete")
    )
    response["provenance"] = {
        "schema_version": INPUT_ATTESTATION_SCHEMA_VERSION if input_receipt is not None else 1,
        "runtime": runtime_identity,
        "score": score,
        "attestation": {
            "kind": (
                "deqio-local-response-v2" if input_receipt is not None else "deqio-local-response"
            ),
            "signed": False,
            "complete": complete,
            "sha256": _attestation_sha(binding),
        },
    }
    return response


def _shared_provenance(
    results: list[dict[str, Any]],
    runtime_identity: dict[str, Any],
    *,
    input_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result_hashes = [
        result.get("provenance", {}).get("attestation", {}).get("sha256")
        for result in results
    ]
    complete = bool(results) and all(
        result.get("provenance", {}).get("attestation", {}).get("complete") is True
        for result in results
    )
    score = {
        "kind": "shared_batch",
        "source": "results[*].provenance.score",
        "result_count": len(results),
    }
    binding = {
        "runtime": runtime_identity,
        "result_attestations": result_hashes,
        "score": score,
    }
    if input_receipt is not None:
        binding["input_receipt"] = input_receipt
        complete = complete and input_receipt.get("status") == "complete"
    return {
        "schema_version": INPUT_ATTESTATION_SCHEMA_VERSION if input_receipt is not None else 1,
        "runtime": runtime_identity,
        "score": score,
        "attestation": {
            "kind": (
                "deqio-local-shared-response-v2"
                if input_receipt is not None
                else "deqio-local-shared-response"
            ),
            "signed": False,
            "complete": complete,
            "sha256": _attestation_sha(binding),
        },
    }


def format_result(
    result: dict,
    *,
    runtime_identity: dict[str, Any] | None = None,
    input_receipt: dict[str, Any] | None = None,
) -> dict:
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

    response = {
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
    if input_receipt is not None:
        response["input_receipt"] = input_receipt
        response["usage"] = {
            "input_tokens": result.get("input_tokens"),
            "input_tokens_source": result.get("input_tokens_source", "unknown"),
        }
    if runtime_identity is not None:
        return _bind_result_provenance(
            response, result, runtime_identity, input_receipt=input_receipt
        )
    return response


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
        "runtime_instance_id": (result.get("provenance") or {}).get("runtime", {}).get("runtime_instance_id"),
        "score_kind": (result.get("provenance") or {}).get("score", {}).get("kind"),
        "attestation_sha256": (result.get("provenance") or {}).get("attestation", {}).get("sha256"),
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


def run_decision(
    request: DecisionRequest,
    *,
    endpoint: str = "/v1/choice",
    contract: InputContractContext | None = None,
) -> dict:
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
            runtime_snapshot = _runtime()
            runtime_identity = _runtime_identity_snapshot(runtime_snapshot, settings_snapshot)
            _ensure_contract_capability(
                runtime_snapshot, contract, request_id=request_id
            )
            raw = runtime_snapshot.score(row, request.mode)

    except InputContractHTTPError:
        raise
    except Exception as exc:
        with stats_lock:
            stats["errors"] += 1
        log_request_error(endpoint, request_id=request_id, error=exc, status=400)

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    input_receipt = (
        _unknown_receipt_for_result(
            context=contract,
            runtime_identity=runtime_identity,
            raw_result=raw,
            settings_snapshot=settings_snapshot,
            decision_id=request_id,
            option_ids=[str(option["id"]) for option in row["options"]],
            components=["state", "question", "options"],
        )
        if contract is not None
        else None
    )
    result = format_result(
        raw, runtime_identity=runtime_identity, input_receipt=input_receipt
    )

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
    _reset_watch_session()

    mark_installed(
        SETTINGS.config_path,
        SETTINGS.model_id,
        SETTINGS.backend,
        verified=True,
        source="startup",
    )
    refresh_identity = getattr(runtime, "refresh_identity", None)
    if callable(refresh_identity):
        refresh_identity()
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


_CONTRACT_ENDPOINTS = {"/v1/noul", "/v1/choice", "/v1/decision", "/v1/shared"}


@app.middleware("http")
async def capture_input_contract_body(request: Request, call_next):
    """Validate and hash negotiated request bytes before FastAPI parses the body."""
    header = request.headers.get("deqio-contract")
    if request.method == "POST" and request.url.path in _CONTRACT_ENDPOINTS and header:
        try:
            version = parse_contract_header(header)
        except UnsupportedInputContractError as error:
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "code": "unsupported_contract",
                        "contract": str(error),
                        "supported": [INPUT_COMPLETENESS_V1],
                        "inference_performed": False,
                    }
                },
            )
        if version is not None:
            raw_body = await request.body()
            request_body_sha256 = sha256_bytes(raw_body)
            try:
                validate_json_without_duplicate_keys(raw_body)
            except DuplicateJSONKeyError as error:
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": {
                            "code": "duplicate_json_key",
                            "request_body_sha256": request_body_sha256,
                            "detail": str(error),
                            "inference_performed": False,
                        }
                    },
                )
            except ValueError as error:
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": {
                            "code": "invalid_request_body",
                            "request_body_sha256": request_body_sha256,
                            "detail": str(error),
                            "inference_performed": False,
                        }
                    },
                )
            request.state.input_contract_raw_body = raw_body
            request.state.input_contract_request_body_sha256 = request_body_sha256

    return await call_next(request)


@app.exception_handler(InputContractHTTPError)
async def input_contract_error_handler(_request: Request, exc: InputContractHTTPError):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.payload})



# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui", status_code=307)


@app.get("/health")
def health():
    status = "switching" if switching_runtime else ("ok" if runtime is not None else "starting")
    runtime_identity = (
        _runtime_identity_snapshot(runtime, SETTINGS) if runtime is not None else None
    )
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
        "runtime_instance_id": (runtime_identity or {}).get("runtime_instance_id"),
        "provenance_schema_version": 1,
        "artifact_revisions_resolved": (runtime_identity or {}).get("artifact_revisions_resolved", False),
    }


@app.post("/v1/choice")
@app.post("/v1/decision")
async def decision(payload: DecisionRequest, http_request: Request):
    endpoint = http_request.url.path
    session_id = _watch_session_token()
    request_payload = payload.model_dump(mode="json")
    try:
        contract = await _input_contract_context(
            http_request,
            policy=payload.input_policy,
            request_id=payload.id,
            decision_ids=[payload.id] if payload.id is not None else ["<generated>"],
            option_sets=[_option_ids(payload.options)],
        )
        result = run_decision(payload, endpoint=endpoint, contract=contract)
    except InputContractHTTPError as exc:
        _record_watch_event(
            expected_session_id=session_id,
            endpoint=endpoint,
            request_payload=request_payload,
            response_payload={"error": exc.payload},
            status_code=exc.status_code,
            request_id=payload.id,
            mode=payload.mode,
            decisions=1,
            settings_snapshot=SETTINGS,
        )
        raise
    except HTTPException as exc:
        _record_watch_event(
            expected_session_id=session_id,
            endpoint=endpoint,
            request_payload=request_payload,
            response_payload={"detail": exc.detail},
            status_code=exc.status_code,
            request_id=payload.id,
            mode=payload.mode,
            decisions=1,
            settings_snapshot=SETTINGS,
        )
        raise

    _record_watch_event(
        expected_session_id=session_id,
        endpoint=endpoint,
        request_payload=request_payload,
        response_payload=result,
        status_code=200,
        request_id=str(result.get("id") or payload.id or ""),
        mode=payload.mode,
        decisions=1,
        settings_snapshot=SETTINGS,
    )
    return result


@app.post("/v1/noul")
async def noul(payload: NoulRequest, http_request: Request):
    request_id = payload.id or uuid4().hex
    endpoint = "/v1/noul"
    session_id = _watch_session_token()
    request_payload = payload.model_dump(mode="json")
    settings_snapshot = SETTINGS
    row = {
        "id": request_id,
        "state": payload.state,
        "question": payload.question,
    }
    try:
        contract = await _input_contract_context(
            http_request,
            policy=payload.input_policy,
            request_id=payload.id,
            decision_ids=[payload.id] if payload.id is not None else ["<generated>"],
            option_sets=[["yes", "no"]],
        )
        with inference_lock:
            settings_snapshot = SETTINGS
            runtime_snapshot = _runtime()
            runtime_identity = _runtime_identity_snapshot(runtime_snapshot, settings_snapshot)
            _ensure_contract_capability(
                runtime_snapshot, contract, request_id=request_id
            )
            raw = runtime_snapshot.score_noul(row, payload.mode)
    except InputContractHTTPError as exc:
        _record_watch_event(
            expected_session_id=session_id,
            endpoint=endpoint,
            request_payload=request_payload,
            response_payload={"error": exc.payload},
            status_code=exc.status_code,
            request_id=request_id,
            mode=payload.mode,
            decisions=1,
            settings_snapshot=settings_snapshot,
        )
        raise
    except Exception as exc:
        with stats_lock:
            stats["errors"] += 1
        log_request_error(endpoint, request_id=request_id, error=exc, status=400)
        _record_watch_event(
            expected_session_id=session_id,
            endpoint=endpoint,
            request_payload=request_payload,
            response_payload={"detail": str(exc)},
            status_code=400,
            request_id=request_id,
            mode=payload.mode,
            decisions=1,
            settings_snapshot=settings_snapshot,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    input_receipt = (
        _unknown_receipt_for_result(
            context=contract,
            runtime_identity=runtime_identity,
            raw_result=raw,
            settings_snapshot=settings_snapshot,
            decision_id=request_id,
            option_ids=["yes", "no"],
            components=["state", "question"],
        )
        if contract is not None
        else None
    )
    result = format_result(
        raw, runtime_identity=runtime_identity, input_receipt=input_receipt
    )
    record_event(
        request_id=request_id,
        mode=payload.mode,
        state=payload.state,
        question=payload.question,
        result=result,
        endpoint=endpoint,
        settings_snapshot=settings_snapshot,
    )
    _record_watch_event(
        expected_session_id=session_id,
        endpoint=endpoint,
        request_payload=request_payload,
        response_payload=result,
        status_code=200,
        request_id=request_id,
        mode=payload.mode,
        decisions=1,
        settings_snapshot=settings_snapshot,
    )
    return result


@app.post("/v1/shared")
async def shared(payload: SharedRequest, http_request: Request):
    endpoint = "/v1/shared"
    request_id = "shared-" + uuid4().hex[:12]
    session_id = _watch_session_token()
    request_payload = payload.model_dump(mode="json")
    explicit_ids = [item.id for item in payload.decisions]
    try:
        contract = await _input_contract_context(
            http_request,
            policy=payload.input_policy,
            request_id=request_id,
            decision_ids=[str(value) for value in explicit_ids if value is not None],
            option_sets=[_option_ids(item.options) for item in payload.decisions],
        )
    except InputContractHTTPError as exc:
        _record_watch_event(
            expected_session_id=session_id,
            endpoint=endpoint,
            request_payload=request_payload,
            response_payload={"error": exc.payload},
            status_code=exc.status_code,
            request_id=request_id,
            mode="shared",
            decisions=len(payload.decisions),
            settings_snapshot=SETTINGS,
        )
        raise

    if contract is not None and any(value is None for value in explicit_ids):
        exc = InputContractHTTPError(
            "decision_id_required",
            request_id=request_id,
            request_body_sha256=contract.request_body_sha256,
            inference_performed=False,
        )
        _record_watch_event(
            expected_session_id=session_id,
            endpoint=endpoint,
            request_payload=request_payload,
            response_payload={"error": exc.payload},
            status_code=exc.status_code,
            request_id=request_id,
            mode="shared",
            decisions=len(payload.decisions),
            settings_snapshot=SETTINGS,
        )
        raise exc
    if not payload.decisions:
        error = "At least one decision is required."
        with stats_lock:
            stats["errors"] += 1
        log_request_error(endpoint, request_id=request_id, error=error, status=400)
        _record_watch_event(
            expected_session_id=session_id,
            endpoint=endpoint,
            request_payload=request_payload,
            response_payload={"detail": error},
            status_code=400,
            request_id=request_id,
            mode="shared",
            decisions=0,
            settings_snapshot=SETTINGS,
        )
        raise HTTPException(status_code=400, detail=error)

    rows = [
        {
            "id": item.id or uuid4().hex,
            "state": payload.state,
            "question": item.question,
            "options": [option.model_dump() for option in item.options],
        }
        for item in payload.decisions
    ]

    settings_snapshot = SETTINGS
    try:
        with inference_lock:
            settings_snapshot = SETTINGS
            runtime_snapshot = _runtime()
            runtime_identity = _runtime_identity_snapshot(runtime_snapshot, settings_snapshot)
            _ensure_contract_capability(
                runtime_snapshot, contract, request_id=request_id
            )
            raw_results, timing = runtime_snapshot.score_shared(rows)
    except InputContractHTTPError as exc:
        _record_watch_event(
            expected_session_id=session_id,
            endpoint=endpoint,
            request_payload=request_payload,
            response_payload={"error": exc.payload},
            status_code=exc.status_code,
            request_id=request_id,
            mode="shared",
            decisions=len(rows),
            settings_snapshot=settings_snapshot,
        )
        raise
    except Exception as exc:
        with stats_lock:
            stats["errors"] += 1
        log_request_error(endpoint, request_id=request_id, error=exc, status=400)
        _record_watch_event(
            expected_session_id=session_id,
            endpoint=endpoint,
            request_payload=request_payload,
            response_payload={"detail": str(exc)},
            status_code=400,
            request_id=request_id,
            mode="shared",
            decisions=len(rows),
            settings_snapshot=settings_snapshot,
        )
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    shared_receipt = (
        _unknown_shared_receipt(
            context=contract,
            runtime_identity=runtime_identity,
            raw_results=raw_results,
            settings_snapshot=settings_snapshot,
            rows=rows,
        )
        if contract is not None
        else None
    )
    results = []
    for raw, row in zip(raw_results, rows):
        per_result_receipt = None
        if contract is not None:
            per_result_receipt = _unknown_receipt_for_result(
                context=contract,
                runtime_identity=runtime_identity,
                raw_result=raw,
                settings_snapshot=settings_snapshot,
                decision_id=str(row["id"]),
                option_ids=[str(option["id"]) for option in row["options"]],
                components=["state", "question", "options"],
            )
        results.append(
            format_result(
                raw, runtime_identity=runtime_identity, input_receipt=per_result_receipt
            )
        )
    batch_provenance = _shared_provenance(
        results, runtime_identity, input_receipt=shared_receipt
    )

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
        "provenance": batch_provenance,
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

    response = {
        "results": results,
        "shared_timing": shared_timing,
        "provenance": batch_provenance,
    }
    if shared_receipt is not None:
        response["input_receipt"] = shared_receipt
    _record_watch_event(
        expected_session_id=session_id,
        endpoint=endpoint,
        request_payload=request_payload,
        response_payload=response,
        status_code=200,
        request_id=request_id,
        mode="shared",
        decisions=len(results),
        settings_snapshot=settings_snapshot,
    )
    return response


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
                "Model profile is not installed. Run: deqio models setup"
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
    if selected.get("max_input_tokens") is not None:
        candidate_data["max_tokens"] = int(selected["max_input_tokens"])
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
            refresh_identity = getattr(runtime, "refresh_identity", None)
            if callable(refresh_identity):
                refresh_identity()
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


@app.get("/v1/watch")
def get_watch():
    session, rows = _watch_rows_snapshot()
    return {
        "session": {**session, "switching": switching_runtime},
        "events": rows,
    }


@app.get("/v1/watch/{event_id}")
def get_watch_event(event_id: str):
    with watch_lock:
        event = next((item for item in watch_events if item.get("event_id") == event_id), None)
        if event is None:
            raise HTTPException(status_code=404, detail="Watch event not found in the current model session")
        return _watch_json_snapshot(event)


@app.post("/v1/watch/clear")
def clear_watch():
    _reset_watch_session()
    session, _rows = _watch_rows_snapshot()
    return {"status": "ok", "session": session}


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


@app.get("/ui/watch", response_class=HTMLResponse)
def watch_dashboard():
    return WATCH_DASHBOARD
