"""Localhost System One sidecar for the community Decision 3.0 MLX conversions (0.6.1).

The ``ahmedandaloes/d3-*-mlx`` packages ship ``d3_mlx.py`` but no server.  This sidecar runs from the
pinned snapshot directory behind the same ``POST /v1/systemone`` / ``GET /health`` surface every
Deqio sidecar exposes.  The conversion implements ``choice`` and ``noul`` only; a ``score`` question
is refused with 422 (the catalog profile declares ``capabilities.score: false`` so the server
rejects it earlier too).

0.6.1 fix 2 — Deqio owns the forward pass.  ``d3_mlx.py`` supplies the loader (``D3MLX.__init__``:
weights, readout head, tokenizer, the 255 answer codes) and the d3 prompt contract (``options``,
``render``, ``to_answer``, copied verbatim from the official ``d3_format.py``), but its own
``D3MLX.system_one`` calls every decoder layer as ``layer(h)`` with neither a cache nor a mask.  The
``qwen3_5`` full-attention layer of every released mlx-vlm (0.5.0 … 0.7.8) reads ``cache.offset``
unconditionally, so that loop fails with ``AttributeError: 'NoneType' object has no attribute
'offset'`` on every pin, and without a mask it would attend bidirectionally.  ``DecisionForward``
therefore runs the checkpoint through mlx-vlm's standard path (``LanguageModel.make_cache`` + the
model's causal forward) with the first full-attention / linear-attention layer indices taken from
the checkpoint's ``layer_types`` (the community loader hard-codes ``fa_idx = 2``, a linear layer in
d3-lite), and never calls ``D3MLX.system_one``.  The path is named in ``/health`` and in every
answer (``forward``) so benchmark results stay attributable.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

SUPPORTED_TYPES = ("choice", "noul")
D3_MLX_FORWARD = "deqio-d3-mlx-forward/1"
LAYER_TYPES = ("full_attention", "linear_attention")
REQUIRED_RUNTIME_API = ("D3MLX", "options", "render", "to_answer")


class DecisionForward:
    """Deqio's inference path for a loaded community Decision 3.0 MLX checkpoint."""

    def __init__(self, snapshot_dir: Path, runtime: Any) -> None:
        missing = [name for name in REQUIRED_RUNTIME_API if not hasattr(runtime, name)]
        if missing:
            raise RuntimeError(
                f"d3_mlx.py in {snapshot_dir} does not provide {', '.join(missing)}; Deqio's Decision 3.0 "
                "MLX forward needs the community loader and the d3 prompt contract"
            )
        self.runtime = runtime
        self.loaded = runtime.D3MLX(str(snapshot_dir))
        for attribute in ("net", "lm", "tok", "readout", "codes", "code_ids"):
            if not hasattr(self.loaded, attribute):
                raise RuntimeError(f"d3_mlx.py's D3MLX exposes no {attribute!r}; the pinned snapshot is not the expected runtime")
        self.net, self.lm = self.loaded.net, self.loaded.lm
        language_model = getattr(self.net, "language_model", None)
        if language_model is None or not callable(getattr(language_model, "make_cache", None)):
            raise RuntimeError("The loaded Decision 3.0 MLX model has no language_model.make_cache(); the mlx-vlm pin does not match")
        self.language_model = language_model
        self.layer_types = self._layer_types(snapshot_dir)
        self.lm.fa_idx = self.layer_types.index("full_attention")
        self.lm.ssm_idx = self.layer_types.index("linear_attention")

    def _layer_types(self, snapshot_dir: Path) -> list[str]:
        config_path = snapshot_dir / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Cannot read {config_path}: {error}") from error
        text_config = config.get("text_config") if isinstance(config, dict) else None
        layer_types = text_config.get("layer_types") if isinstance(text_config, dict) else None
        if not isinstance(layer_types, list) or not layer_types:
            raise RuntimeError(f"{config_path} declares no text_config.layer_types; the checkpoint layout is not the expected one")
        layer_types = [str(kind) for kind in layer_types]
        unknown = sorted(set(layer_types) - set(LAYER_TYPES))
        if unknown:
            raise RuntimeError(f"{config_path} declares unsupported layer_types {unknown}")
        layers = list(getattr(self.lm, "layers", []))
        if len(layers) != len(layer_types):
            raise RuntimeError(
                f"{config_path} declares {len(layer_types)} layer_types but the loaded model has {len(layers)} layers"
            )
        for index, (layer, kind) in enumerate(zip(layers, layer_types)):
            is_linear = getattr(layer, "is_linear", None)
            if is_linear is None or bool(is_linear) != (kind == "linear_attention"):
                raise RuntimeError(
                    f"Layer {index} is typed {kind!r} in text_config.layer_types of {config_path} but the loaded "
                    f"model marks it is_linear={is_linear!r}; refusing to run a mistyped forward"
                )
        for kind in LAYER_TYPES:
            if kind not in layer_types:
                raise RuntimeError(f"{config_path} declares no {kind} layer; the hybrid Decision 3.0 forward needs both kinds")
        return layer_types

    def hidden(self, ids: list[int]):
        """Causal forward of one token sequence: ``(1, len(ids), hidden)`` after the final norm."""
        import mlx.core as mx

        cache = self.language_model.make_cache()
        h = self.lm(mx.array([ids], dtype=mx.int32), cache=cache)
        mx.eval(h)
        return h

    def system_one(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        import mlx.core as mx

        answers: dict[str, Any] = {}
        for name, question in questions.items():
            keys, _texts = self.runtime.options(question)
            text = self.runtime.render(self.loaded.tok, state, question, self.loaded.codes)
            ids = self.loaded.tok.encode(text, add_special_tokens=False)
            h = self.hidden(list(ids))
            logits = (self.loaded.readout @ h[0, -1, :].astype(mx.float32)).astype(mx.float32)
            probabilities = mx.softmax(logits, axis=-1)
            mx.eval(probabilities)
            picked = [float(probabilities[index]) for index in self.loaded.code_ids[: len(keys)]]
            answers[name] = self.runtime.to_answer(question, picked)
        return {"answers": answers, "forward": D3_MLX_FORWARD}


def build_app(snapshot: str, name: str):
    from fastapi import FastAPI, HTTPException

    snapshot_dir = Path(snapshot).expanduser().resolve()
    model: DecisionForward | None = None
    load_lock = threading.Lock()
    inference_lock = threading.Lock()
    app = FastAPI(title="deqio Decision 3.0 MLX sidecar")

    def ensure_model() -> DecisionForward:
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
                    import d3_mlx  # the loader and prompt contract shipped by the pinned community snapshot

                    model = DecisionForward(snapshot_dir, d3_mlx)
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
            "forward": D3_MLX_FORWARD,
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
            except Exception as error:  # noqa: BLE001 - 0.6.1 fix 2: the forward's failure reaches Deqio's console with its cause
                raise HTTPException(
                    status_code=500,
                    detail=f"Decision 3.0 MLX forward ({D3_MLX_FORWARD}) failed: {type(error).__name__}: {error}",
                ) from error
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
