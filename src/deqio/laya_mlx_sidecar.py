from __future__ import annotations

import argparse
import time
from typing import Any


def build_app(model_id: str):
    import laya_mlx as laya
    from fastapi import FastAPI, HTTPException

    agent = laya.load(model_id)
    app = FastAPI(title="deqio Laya MLX sidecar")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_id, "engine": "laya_mlx"}

    @app.post("/v1/systemone")
    def system_one(body: dict[str, Any]):
        if not isinstance(body, dict) or not isinstance(body.get("questions"), dict):
            raise HTTPException(status_code=422, detail="questions must be an object")
        started = time.perf_counter()
        result = agent.predict(body.get("state"), body["questions"])
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Laya MLX returned an invalid result")
        result.setdefault("model", model_id)
        result.setdefault("latency_ms", round((time.perf_counter() - started) * 1000, 3))
        return result

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        build_app(args.model),
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
