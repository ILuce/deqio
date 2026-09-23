from __future__ import annotations

import gc
import platform
from collections.abc import Callable
from typing import Any

from .config import Settings


class BackendRuntime:
    """One loaded SemIf backend with a stable server-facing scoring interface."""

    def __init__(
        self,
        *,
        settings: Settings,
        model: Any,
        tokenizer: Any,
        metadata: dict[str, Any],
        direct_score: Callable[..., dict],
        serial_factory: Callable[..., Any],
        shared_score: Callable[..., tuple[list[dict], dict]],
    ) -> None:
        self.settings = settings
        self.name = settings.backend
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata
        self._direct_score = direct_score
        self._serial_factory = serial_factory
        self._shared_score = shared_score
        self.serial_scorer = self._new_serial_scorer()

    @classmethod
    def load(cls, settings: Settings) -> "BackendRuntime":
        if settings.backend == "mlx":
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise RuntimeError("MLX backend requires macOS on Apple Silicon (Darwin arm64)")
            model_path = settings.model
            from pathlib import Path

            if not Path(model_path).is_dir():
                raise RuntimeError(
                    f"MLX model directory does not exist: {model_path}. "
                    "Download the configured model before starting the server."
                )
            from semif_phase1 import mlx_backend

            model, tokenizer, metadata = mlx_backend.load_model(
                model_path,
                settings.model_revision,
                bits=None,
                cache_limit_mib=settings.mlx_cache_mib,
            )
            return cls(
                settings=settings,
                model=model,
                tokenizer=tokenizer,
                metadata=metadata,
                direct_score=mlx_backend.score,
                serial_factory=mlx_backend.SerialPrefixScorer,
                shared_score=mlx_backend.score_shared,
            )

        if settings.backend == "cuda":
            from semif_phase1.core import load_causal_model
            from semif_phase1.direct import score
            from semif_phase1.serial import SerialPrefixScorer
            from semif_phase1.shared import score_shared

            model, tokenizer, metadata = load_causal_model(
                settings.model,
                settings.model_revision,
                "cuda",
                settings.torch_dtype,
            )
            return cls(
                settings=settings,
                model=model,
                tokenizer=tokenizer,
                metadata=metadata,
                direct_score=score,
                serial_factory=SerialPrefixScorer,
                shared_score=score_shared,
            )

        if settings.backend == "llamacpp":
            if not settings.llama_gguf.is_file():
                raise RuntimeError(
                    f"GGUF checkpoint does not exist: {settings.llama_gguf}. "
                    "Download it before starting the llama.cpp backend."
                )
            try:
                from semif_phase1 import llamacpp_backend
            except ImportError as error:
                raise RuntimeError(
                    "llama.cpp support is not installed. Run: "
                    "uv sync --frozen --extra llamacpp"
                ) from error

            model, tokenizer, metadata = llamacpp_backend.load_model(
                settings.model,
                settings.model_revision,
                settings.llama_gguf,
                threads=settings.llama_threads,
                context_tokens=settings.max_tokens,
            )
            return cls(
                settings=settings,
                model=model,
                tokenizer=tokenizer,
                metadata=metadata,
                direct_score=llamacpp_backend.score,
                serial_factory=llamacpp_backend.SerialPrefixScorer,
                shared_score=llamacpp_backend.score_shared,
            )

        raise RuntimeError(f"Unsupported backend: {settings.backend}")

    def _new_serial_scorer(self):
        return self._serial_factory(
            self.model,
            self.tokenizer,
            self.metadata,
            self.settings.max_tokens,
        )

    def score(self, row: dict, mode: str) -> dict:
        if mode == "serial":
            return self.serial_scorer.score(row)
        if mode == "direct":
            return self._direct_score(
                self.model,
                self.tokenizer,
                row,
                self.metadata,
                self.settings.max_tokens,
            )
        raise ValueError(f"Unsupported scoring mode: {mode}")

    def score_shared(self, rows: list[dict]) -> tuple[list[dict], dict]:
        return self._shared_score(
            self.model,
            self.tokenizer,
            rows,
            self.metadata,
            self.settings.max_tokens,
        )

    def clear_cache(self) -> dict[str, Any]:
        """Drop reusable prefix state and release backend allocator caches when available."""
        self.serial_scorer = self._new_serial_scorer()
        gc.collect()

        details: dict[str, Any] = {
            "backend": self.name,
            "prefix_cache": "cleared",
            "model_loaded": True,
        }

        if self.name == "mlx":
            import mlx.core as mx

            before = int(mx.get_cache_memory())
            mx.clear_cache()
            after = int(mx.get_cache_memory())
            details.update(
                allocator_cache="cleared",
                allocator_cache_bytes_before=before,
                allocator_cache_bytes_after=after,
            )
        elif self.name == "cuda":
            import torch

            before = int(torch.cuda.memory_reserved())
            torch.cuda.empty_cache()
            after = int(torch.cuda.memory_reserved())
            details.update(
                allocator_cache="cleared",
                allocator_cache_bytes_before=before,
                allocator_cache_bytes_after=after,
            )
        elif self.name == "llamacpp":
            self.model.engine.clear()
            details.update(context_cache="cleared")

        return details

    def close(self) -> None:
        if self.name == "llamacpp" and callable(getattr(self.model, "close", None)):
            self.model.close()
