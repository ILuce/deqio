from __future__ import annotations

import argparse
import threading
import time
from typing import Any


def build_app(model_id: str, revision: str | None = None):
    import laya_mlx as laya
    from fastapi import FastAPI, HTTPException

    agent = None
    load_lock = threading.Lock()
    app = FastAPI(title="deqio Laya MLX sidecar")

    def ensure_agent():
        nonlocal agent
        if agent is not None:
            return agent
        with load_lock:
            if agent is None:
                # D12: the launcher passes the catalog's pinned Hub revision.
                agent = laya.load(model_id, revision=revision) if revision else laya.load(model_id)
        return agent

    @app.get("/health")
    def health():
        return {
            "status": "ready" if agent is not None else "process-ready",
            "model": model_id,
            "engine": "laya_mlx",
        }

    @app.post("/v1/systemone")
    def system_one(body: dict[str, Any]):
        if not isinstance(body, dict) or not isinstance(body.get("questions"), dict):
            raise HTTPException(status_code=422, detail="questions must be an object")
        started = time.perf_counter()
        loaded_agent = ensure_agent()
        result = loaded_agent.predict(body.get("state"), body["questions"])
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Laya MLX returned an invalid result")
        result.setdefault("model", model_id)
        result.setdefault("latency_ms", round((time.perf_counter() - started) * 1000, 3))
        return result

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        build_app(args.model, args.revision),
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
