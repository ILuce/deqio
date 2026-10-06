from __future__ import annotations

import argparse
import threading
from typing import Any


def build_app(model_id: str, revision: str):
    """Expose the official Decision 2.0 runtime over Deqio's localhost wire.

    This module is transport only.  Inference, question validation, context
    handling, probabilities, confidence and errors all come from the model
    package's official System One runtime.
    """

    import torch
    from fastapi import FastAPI, HTTPException
    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Decision 2.0 requires the official CUDA runtime. "
            "No CPU, MPS or MLX fallback will be used."
        )

    app = FastAPI(title="deqio Decision 2.0 transport")
    model = None
    load_lock = threading.Lock()

    def ensure_model():
        nonlocal model
        if model is not None:
            return model
        with load_lock:
            if model is None:
                # Explicit cuda:0 is deliberate: upstream otherwise permits CPU
                # when CUDA is absent.  Deqio 0.5 intentionally forbids that
                # fallback for Decision 2.0.
                model = AutoModel.from_pretrained(
                    model_id,
                    revision=revision,
                    local_files_only=True,
                    trust_remote_code=True,
                    device="cuda:0",
                    dtype="auto",
                )
        return model

    @app.get("/health")
    def health():
        return {
            "status": "ready" if model is not None else "process-ready",
            "model": model_id,
            "revision": revision,
            "engine": "decision2",
            "device": "cuda:0",
        }

    @app.post("/v1/systemone")
    def system_one(body: dict[str, Any]):
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="request must be an object")
        questions = body.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise HTTPException(status_code=422, detail="questions must be a non-empty object")
        requested_model = body.get("model")
        if requested_model not in (None, model_id):
            raise HTTPException(
                status_code=409,
                detail=f"active Decision 2.0 model is {model_id!r}, not {requested_model!r}",
            )
        try:
            result = ensure_model().system_one(
                state=body.get("state"),
                questions=questions,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Decision 2.0 returned an invalid result")
        # Do not rewrite, normalize or supplement native Decision output here.
        return result

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a pinned official Decision 2.0 model on CUDA."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
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
