from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any


def _state_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _choice_schema(question: dict[str, Any]) -> dict[str, Any]:
    criteria = question.get("criteria") or {}
    if isinstance(criteria, dict):
        choices = [str(key) for key in criteria]
        descriptions = {str(key): str(value) for key, value in criteria.items() if value not in (None, "")}
    elif isinstance(criteria, list):
        choices = [str(value) for value in criteria]
        descriptions = {}
    else:
        raise ValueError("choice criteria must be an object or array")
    if not choices:
        raise ValueError("choice requires at least one option")
    field: dict[str, Any] = {
        "type": "enum",
        "choices": choices,
        "description": str(question.get("instructions") or "Choose the best option."),
    }
    if descriptions:
        field["choice_descriptions"] = descriptions
    return field


def _score_schema(question: dict[str, Any]) -> dict[str, Any]:
    criteria = question.get("criteria") or []
    if isinstance(criteria, dict):
        choices = [str(key) for key in criteria]
        descriptions = {str(key): str(value) for key, value in criteria.items() if value not in (None, "")}
    else:
        choices = [str(value) for value in criteria]
        descriptions = {}
    field: dict[str, Any] = {
        "type": "enum",
        "choices": choices,
        "description": str(question.get("instructions") or "Choose the best score level."),
    }
    if descriptions:
        field["choice_descriptions"] = descriptions
    return field



def _answers_from_result(
    schema: dict[str, Any], kinds: dict[str, str], result: dict[str, Any]
) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    fields = result.get("fields")
    output = result.get("output")
    if not isinstance(fields, dict) or not isinstance(output, dict):
        raise ValueError("Nimble result is missing fields/output")
    for qid, kind in kinds.items():
        field = fields.get(qid)
        if not isinstance(field, dict) or not isinstance(field.get("scores"), dict) or qid not in output:
            raise ValueError(f"Nimble result is missing field {qid!r}")
        scores = {str(key): float(value) for key, value in field["scores"].items()}
        value = output[qid]
        if kind == "noul":
            p_yes = float(scores.get("true", 1.0 if value is True else 0.0))
            answers[qid] = {
                "type": "noul",
                "noul": p_yes,
                "confidence": max(p_yes, 1.0 - p_yes),
            }
        elif kind == "choice":
            answers[qid] = {
                "type": "choice",
                "choice": str(value),
                "probabilities": scores,
                "confidence": max(scores.values()) if scores else 0.0,
            }
        else:
            choices = list(schema[qid]["choices"])
            expected = sum(index * float(scores.get(choice, 0.0)) for index, choice in enumerate(choices))
            answers[qid] = {
                "type": "score",
                "score": expected,
                "choice": str(value),
                "probabilities": scores,
                "confidence": max(scores.values()) if scores else 0.0,
            }
    return answers

def create_app(*, source_root: Path, model_config: Path, backend: str):
    sys.path.insert(0, str(source_root.resolve()))
    from fastapi import FastAPI, HTTPException

    config = json.loads(model_config.read_text(encoding="utf-8"))
    if backend == "mlx":
        from nimble.scoring.parallel_scorer import ParallelScorer

        scorer_factory = ParallelScorer
    elif backend == "cuda":
        from nimble.scoring.cuda_scorer import CudaCandidateScorer

        scorer_factory = CudaCandidateScorer
    else:
        raise RuntimeError(f"Unsupported Nimble backend: {backend}")

    scorer = None
    load_lock = threading.Lock()
    app = FastAPI(title="Deqio Nimble sidecar")

    def ensure_scorer():
        nonlocal scorer
        if scorer is not None:
            return scorer
        with load_lock:
            if scorer is None:
                scorer = scorer_factory(**config)
        return scorer

    @app.get("/health")
    def health():
        return {
            "status": "ready" if scorer is not None else "process-ready",
            "engine": "nimble",
            "backend": backend,
        }

    @app.post("/v1/systemone")
    def systemone(payload: dict[str, Any]):
        started = time.perf_counter()
        state = _state_text(payload.get("state"))
        questions = payload.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise HTTPException(status_code=422, detail="questions must be a nonempty object")
        schema: dict[str, Any] = {}
        kinds: dict[str, str] = {}
        try:
            for qid, question in questions.items():
                if not isinstance(question, dict):
                    raise ValueError(f"{qid}: question must be an object")
                kind = str(question.get("type", ""))
                kinds[str(qid)] = kind
                if kind == "noul":
                    schema[str(qid)] = {
                        "type": "boolean",
                        "description": str(question.get("instructions") or "Is this true?"),
                    }
                elif kind == "choice":
                    schema[str(qid)] = _choice_schema(question)
                elif kind == "score":
                    schema[str(qid)] = _score_schema(question)
                else:
                    raise ValueError(f"{qid}: unsupported question type {kind!r}")
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        try:
            result = ensure_scorer().score(state, schema)
            answers = _answers_from_result(schema, kinds, result)
            return {
                "model": config.get("model_id", "bespokelabs/Bespoke-Nimble-9B"),
                "answers": answers,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "usage": {"input_tokens": 0},
            }
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local Nimble System One adapter for Deqio.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--backend", choices=("mlx", "cuda"), required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    import uvicorn

    app = create_app(source_root=args.source_root, model_config=args.model_config, backend=args.backend)
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
