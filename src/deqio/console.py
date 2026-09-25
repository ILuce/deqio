from __future__ import annotations

import re
import threading
from subprocess import Popen
from typing import Any, TextIO

_PRINT_LOCK = threading.Lock()

_ACCESS_LOG_RE = re.compile(
    r'^(?:INFO|DEBUG):\s+.*\"(?:GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\s+\S+\s+HTTP/[^\"]+\"\s+\d{3}'
)
_NOISY_SIDECAR_MARKERS = (
    "Started server process",
    "Waiting for application startup",
    "Application startup complete",
    "Uvicorn running on",
    "Shutting down",
    "Waiting for application shutdown",
    "Application shutdown complete",
    "Finished server process",
)


def _emit(prefix: str, message: str) -> None:
    with _PRINT_LOCK:
        print(f"{prefix} {message}", flush=True)


def info(message: str) -> None:
    _emit("[deqio]", message)


def request_log(message: str) -> None:
    _emit("[request]", message)


def _compact(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def startup_header(*, engine: str, model_id: str, backend: str, model: str, config: str) -> None:
    info("------------------------------------------------------------")
    info("Starting decision server")
    info(f"  engine    {engine}")
    info(f"  model     {model_id}")
    info(f"  backend   {backend}")
    info(f"  source    {model}")
    info(f"  config    {config}")
    info("------------------------------------------------------------")


def sidecar_start(*, engine: str, address: str) -> None:
    info("Starting internal inference sidecar")
    info(f"  engine    {engine}")
    info(f"  address   {address}")
    info("  role      internal inference API only")
    info(f"  logs      prefixed as [sidecar:{engine}]")


def sidecar_ready(*, engine: str, address: str) -> None:
    info(f"Internal sidecar ready: engine={engine} address={address}")


def server_ready(*, host: str = "127.0.0.1", port: int = 8787) -> None:
    base = f"http://{host}:{port}"
    info("Ready")
    info(f"  UI       {base}/ui")
    info(f"  API      {base}/docs")
    info(f"  health   {base}/health")
    info("  endpoints POST /v1/noul  /v1/choice  /v1/shared")


def warmup_ok(step: int, total: int) -> None:
    info(f"Warmup {step}/{total} OK")


def log_request_success(
    endpoint: str,
    *,
    request_id: str,
    result: dict[str, Any],
    mode: str,
    decisions: int = 1,
) -> None:
    latency_ms = result.get("timing", {}).get("total_ms")
    cache_hit = result.get("timing", {}).get("cache_hit")
    latency_text = "-" if latency_ms is None else f"{latency_ms}ms"

    if decisions > 1 or endpoint == "/v1/shared":
        request_log(
            f"POST {endpoint:<15} 200 id={request_id} mode={mode} "
            f"decisions={decisions} latency={latency_text}"
        )
        return

    probability = result.get("top_probability")
    probability_text = "-" if probability is None else f"{float(probability):.4f}"
    cache_text = "-" if cache_hit is None else ("hit" if cache_hit else "miss")
    request_log(
        f"POST {endpoint:<15} 200 id={request_id} mode={mode} "
        f"decision={result.get('decision')} p={probability_text} "
        f"latency={latency_text} cache={cache_text}"
    )


def log_request_error(endpoint: str, *, request_id: str, error: Any, status: int) -> None:
    request_log(
        f"POST {endpoint:<15} {status} id={request_id} error={_compact(error)}"
    )


def log_cache_clear(*, details: dict[str, Any]) -> None:
    request_log(
        "POST /v1/cache/clear 200 "
        f"engine={details.get('engine')} backend={details.get('backend')} "
        f"model_loaded={details.get('model_loaded')}"
    )


def is_noisy_sidecar_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _ACCESS_LOG_RE.search(stripped):
        return True
    return any(marker in stripped for marker in _NOISY_SIDECAR_MARKERS)


def sidecar_line(engine: str, line: str) -> None:
    for part in line.replace("\r", "\n").splitlines():
        stripped = part.strip()
        if not is_noisy_sidecar_line(stripped):
            _emit(f"[sidecar:{engine}]", stripped)


def start_sidecar_log_pump(process: Popen[Any], engine: str) -> threading.Thread | None:
    stream: TextIO | None = process.stdout  # type: ignore[assignment]
    if stream is None:
        return None

    def pump() -> None:
        try:
            for line in stream:
                sidecar_line(engine, line)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    thread = threading.Thread(
        target=pump,
        name=f"deqio-{engine}-sidecar-log",
        daemon=True,
    )
    thread.start()
    return thread


def model_switch_start(*, current: str, target: str) -> None:
    info(f"Switching model: {current} -> {target}")
    info("Inference requests are paused until the new runtime is ready.")


def model_switch_ok(*, model_id: str, backend: str, elapsed_ms: float) -> None:
    info(f"Model switch complete: {model_id} / {backend} ({elapsed_ms:.1f}ms)")


def model_switch_rollback(*, target: str, error: Any) -> None:
    info(f"Model switch failed for {target}: {_compact(error)}")
    info("Restoring previous runtime...")


def model_switch_restored(*, model_id: str, backend: str) -> None:
    info(f"Previous runtime restored: {model_id} / {backend}")
