from __future__ import annotations

import argparse
import time
from typing import Any


def _choice_row(state: Any, qid: str, question: dict[str, Any]) -> dict[str, Any]:
    kind = str(question.get("type", ""))
    if kind == "noul":
        options = [
            {"id": "yes", "description": "Yes. The evidence supports the criterion or question."},
            {"id": "no", "description": "No. The evidence does not support the criterion or question."},
        ]
    elif kind == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"{qid}: choice criteria must be a nonempty object")
        options = [
            {"id": str(option_id), "description": "" if description is None else str(description)}
            for option_id, description in criteria.items()
        ]
    else:
        raise ValueError(f"{qid}: unsupported question type {kind!r}")
    return {
        "id": qid,
        "state": state,
        "question": str(question.get("instructions") or "Choose the best option."),
        "options": options,
    }


def _answer_from_raw(kind: str, raw: dict[str, Any]) -> dict[str, Any]:
    option_ids = [str(value) for value in raw.get("option_ids", [])]
    probabilities = [float(value) for value in raw.get("probabilities", [])]
    if not option_ids or len(option_ids) != len(probabilities):
        raise ValueError("SemIf result is missing a complete probability distribution")
    scores = dict(zip(option_ids, probabilities))
    winner = option_ids[max(range(len(probabilities)), key=probabilities.__getitem__)]
    confidence = max(probabilities)
    if kind == "noul":
        if "yes" not in scores:
            raise ValueError("SemIf Noul result is missing the yes probability")
        return {"type": "noul", "noul": float(scores["yes"]), "confidence": confidence}
    return {
        "type": "choice",
        "choice": winner,
        "probabilities": scores,
        "confidence": confidence,
    }


def build_app(
    *,
    backend: str,
    model_id: str,
    revision: str,
    max_tokens: int,
    mlx_cache_mib: int,
    torch_dtype: str,
):
    from fastapi import FastAPI, HTTPException

    if backend == "mlx":
        from semif_phase1 import mlx_backend

        model, tokenizer, metadata = mlx_backend.load_model(
            model_id,
            revision,
            bits=None,
            cache_limit_mib=mlx_cache_mib,
        )
        serial_factory = mlx_backend.SerialPrefixScorer
        direct_score = mlx_backend.score
        shared_score = mlx_backend.score_shared
    elif backend in {"mps", "cuda"}:
        from semif_phase1.core import load_causal_model
        from semif_phase1.direct import score as direct_score
        from semif_phase1.serial import SerialPrefixScorer
        from semif_phase1.shared import score_shared

        model, tokenizer, metadata = load_causal_model(model_id, revision, backend, torch_dtype)
        serial_factory = SerialPrefixScorer
    else:
        raise RuntimeError(f"Unsupported SemIf backend: {backend}")

    def new_serial():
        return serial_factory(model, tokenizer, metadata, max_tokens)

    serial = new_serial()
    app = FastAPI(title="Deqio SemIf sidecar")

    @app.get("/health")
    def health():
        return {"status": "ok", "engine": "semif", "backend": backend, "model": model_id}

    @app.post("/v1/cache/clear")
    def clear_cache():
        nonlocal serial
        import gc

        serial = new_serial()
        gc.collect()
        details: dict[str, Any] = {"engine": "semif", "backend": backend, "prefix_cache": "cleared"}
        try:
            if backend == "mlx":
                import mlx.core as mx
                before = int(mx.get_cache_memory())
                mx.clear_cache()
                details.update(allocator_cache="cleared", allocator_cache_bytes_before=before, allocator_cache_bytes_after=int(mx.get_cache_memory()))
            elif backend == "cuda":
                import torch
                before = int(torch.cuda.memory_reserved())
                torch.cuda.empty_cache()
                details.update(allocator_cache="cleared", allocator_cache_bytes_before=before, allocator_cache_bytes_after=int(torch.cuda.memory_reserved()))
            elif backend == "mps":
                import torch
                before = int(torch.mps.current_allocated_memory()) if hasattr(torch.mps, "current_allocated_memory") else None
                torch.mps.empty_cache()
                after = int(torch.mps.current_allocated_memory()) if hasattr(torch.mps, "current_allocated_memory") else None
                details.update(allocator_cache="cleared", allocator_cache_bytes_before=before, allocator_cache_bytes_after=after)
        except Exception:
            details["allocator_cache"] = "best-effort cleanup unavailable"
        return details

    @app.post("/v1/systemone")
    def systemone(payload: dict[str, Any]):
        started = time.perf_counter()
        try:
            questions = payload.get("questions")
            if not isinstance(questions, dict) or not questions:
                raise ValueError("questions must be a nonempty object")
            state = payload.get("state")
            qids = [str(qid) for qid in questions]
            kinds: dict[str, str] = {}
            rows: list[dict[str, Any]] = []
            for qid in qids:
                question = questions[qid]
                if not isinstance(question, dict):
                    raise ValueError(f"{qid}: question must be an object")
                kinds[qid] = str(question.get("type", ""))
                rows.append(_choice_row(state, qid, question))

            execution_mode = str(payload.get("execution_mode") or "serial")
            if len(rows) == 1 and execution_mode == "direct":
                raw_results = [direct_score(model, tokenizer, rows[0], metadata, max_tokens)]
            elif len(rows) == 1:
                raw_results = [serial.score(rows[0])]
            else:
                raw_results, _ = shared_score(model, tokenizer, rows, metadata, max_tokens)

            answers = {
                qid: _answer_from_raw(kinds[qid], raw)
                for qid, raw in zip(qids, raw_results)
            }
            input_tokens = max((int(raw.get("input_tokens", 0) or 0) for raw in raw_results), default=0)
            return {
                "model": model_id,
                "answers": answers,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "usage": {"input_tokens": input_tokens},
            }
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated SemIf adapter for Deqio.")
    parser.add_argument("--backend", choices=("mlx", "mps", "cuda"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--mlx-cache-mib", type=int, required=True)
    parser.add_argument("--torch-dtype", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    import uvicorn

    app = build_app(
        backend=args.backend,
        model_id=args.model,
        revision=args.revision,
        max_tokens=args.max_tokens,
        mlx_cache_mib=args.mlx_cache_mib,
        torch_dtype=args.torch_dtype,
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
