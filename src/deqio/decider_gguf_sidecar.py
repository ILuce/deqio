from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any


def build_app(*, model_dir: Path, gguf_file: Path, max_tokens: int):
    from decider.infer import Decider
    from fastapi import FastAPI, HTTPException

    model = None
    app = FastAPI(title="Deqio Decider GGUF transport")

    def ensure_model():
        nonlocal model
        if model is None:
            model = Decider(
                str(model_dir),
                gguf_file=str(gguf_file),
                gguf_options={"n_ctx": int(max_tokens), "n_gpu_layers": -1},
            )
        return model

    @app.get("/health")
    def health():
        return {"status": "ready" if model is not None else "process-ready", "engine": "decider", "backend": "gguf"}

    @app.post("/v1/systemone")
    def system_one(body: dict[str, Any]):
        if not isinstance(body, dict) or not isinstance(body.get("questions"), dict) or not body["questions"]:
            raise HTTPException(status_code=422, detail="questions must be a non-empty object")
        started = time.perf_counter()
        try:
            result = ensure_model().system_one(body.get("state"), body["questions"])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Decider GGUF returned an invalid System One response")
        result.setdefault("model", str(model_dir))
        result.setdefault("latency_ms", round((time.perf_counter() - started) * 1000.0, 3))
        return result

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve official Decider GGUF through its native Decider.system_one runtime")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not args.gguf.is_file():
        raise RuntimeError(f"Decider GGUF weights are missing: {args.gguf}")
    import uvicorn
    uvicorn.run(build_app(model_dir=args.model_dir, gguf_file=args.gguf, max_tokens=args.max_tokens), host="127.0.0.1", port=args.port, access_log=False, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
