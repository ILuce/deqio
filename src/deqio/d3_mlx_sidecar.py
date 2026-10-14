"""Localhost System One sidecar for the community Decision 3.0 MLX conversions (0.6.1).

The ``ahmedandaloes/d3-*-mlx`` packages ship ``d3_mlx.py`` (``D3MLX(model_dir).system_one(state,
questions)``) but no server.  This sidecar runs that runtime from the pinned snapshot directory
behind the same ``POST /v1/systemone`` / ``GET /health`` surface every Deqio sidecar exposes.  The
conversion implements ``choice`` and ``noul`` only; a ``score`` question is refused with 422 (the
catalog profile declares ``capabilities.score: false`` so the server rejects it earlier too).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any

SUPPORTED_TYPES = ("choice", "noul")


def build_app(snapshot: str, name: str):
    from fastapi import FastAPI, HTTPException

    snapshot_dir = Path(snapshot).expanduser().resolve()
    model = None
    load_lock = threading.Lock()
    inference_lock = threading.Lock()
    app = FastAPI(title="deqio Decision 3.0 MLX sidecar")

    def ensure_model():
        nonlocal model
        if model is not None:
            return model
        with load_lock:
            if model is None:
                if not (snapshot_dir / "d3_mlx.py").is_file():
                    raise HTTPException(status_code=500, detail=f"d3_mlx.py is missing from the pinned snapshot {snapshot_dir}")
                if str(snapshot_dir) not in sys.path:
                    sys.path.insert(0, str(snapshot_dir))
                try:
                    from d3_mlx import D3MLX  # the runtime shipped by the pinned community snapshot

                    model = D3MLX(str(snapshot_dir))
                except Exception as error:  # noqa: BLE001 - the cause must reach Deqio's console, not only this process
                    # 0.6.1 fix 1: a runtime environment that cannot import or load d3_mlx.py (for example an mlx-vlm
                    # without mlx_vlm.models.qwen3_5) answers with the cause instead of a bare 500.
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"Decision 3.0 MLX runtime failed to load d3_mlx.py from {snapshot_dir}: "
                            f"{type(error).__name__}: {error}. Run deqio models update for this profile."
                        ),
                    ) from error
        return model

    @app.get("/health")
    def health():
        return {
            "status": "ready" if model is not None else "process-ready",
            "model": name,
            "engine": "decision3",
            "backend": "mlx",
            "snapshot": str(snapshot_dir),
        }

    @app.post("/v1/systemone")
    def system_one(body: dict[str, Any]):
        if not isinstance(body, dict) or not isinstance(body.get("questions"), dict) or not body["questions"]:
            raise HTTPException(status_code=422, detail="questions must be a non-empty object")
        for question_id, question in body["questions"].items():
            if not isinstance(question, dict):
                raise HTTPException(status_code=422, detail=f"question {question_id!r} must be an object")
            kind = str(question.get("type", "choice"))
            if kind not in SUPPORTED_TYPES:
                raise HTTPException(
                    status_code=422,
                    detail=f"question {question_id!r}: the Decision 3.0 MLX runtime supports {', '.join(SUPPORTED_TYPES)}, not {kind!r}",
                )
        started = time.perf_counter()
        loaded = ensure_model()
        with inference_lock:
            try:
                result = loaded.system_one(body.get("state"), body["questions"])
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise HTTPException(status_code=500, detail="Decision 3.0 MLX returned an invalid result")
        response = dict(result)
        response.setdefault("model", name)
        response.setdefault("latency_ms", round((time.perf_counter() - started) * 1000, 3))
        return response

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, help="pinned Hub snapshot directory holding d3_mlx.py and the weights")
    parser.add_argument("--name", required=True, help="model name reported in responses")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        build_app(args.snapshot, args.name),
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
