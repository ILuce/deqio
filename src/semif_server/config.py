from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("config.json")
ENV_OVERRIDES = {
    "SEMIF_BACKEND": "backend",
    "SEMIF_MODEL": "model",
    "SEMIF_MODEL_REVISION": "model_revision",
    "SEMIF_MAX_TOKENS": "max_tokens",
    "SEMIF_MLX_CACHE_MIB": "mlx_cache_mib",
    "SEMIF_LOG": "log",
    "SEMIF_TORCH_DTYPE": "torch_dtype",
    "SEMIF_GGUF": "llama_gguf",
    "SEMIF_LLAMA_THREADS": "llama_threads",
}


@dataclass(frozen=True)
class Settings:
    config_path: Path
    backend: str
    model: str
    model_revision: str
    max_tokens: int
    mlx_cache_mib: int
    log_path: Path
    torch_dtype: str
    llama_gguf: Path
    llama_threads: int | None


def _coerce_override(key: str, value: str) -> Any:
    if key in {"max_tokens", "mlx_cache_mib", "llama_threads"}:
        return int(value)
    return value


def _resolve_model_source(value: str, base: Path, backend: str) -> str:
    path = Path(value).expanduser()
    candidate = path if path.is_absolute() else base / path

    # MLX is intentionally local in this server configuration. For Torch and
    # llama.cpp, Hugging Face repo IDs remain valid remote model sources.
    if backend == "mlx" or path.is_absolute() or value.startswith(("./", "../", "~")):
        return str(candidate.resolve())
    if candidate.exists():
        return str(candidate.resolve())
    return value


def load_settings(path: str | Path | None = None) -> Settings:
    configured = path or os.environ.get("SEMIF_CONFIG") or DEFAULT_CONFIG_PATH
    config_path = Path(configured).expanduser().resolve()
    if not config_path.is_file():
        raise RuntimeError(
            f"Configuration file does not exist: {config_path}. "
            "Create config.json or set SEMIF_CONFIG."
        )

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in {config_path}: {error}") from error

    if not isinstance(data, dict):
        raise RuntimeError("The root configuration value must be a JSON object")

    for env_name, key in ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value not in (None, ""):
            try:
                data[key] = _coerce_override(key, value)
            except ValueError as error:
                raise RuntimeError(f"Invalid value for {env_name}: {value!r}") from error

    required = {
        "backend",
        "model",
        "model_revision",
        "max_tokens",
        "mlx_cache_mib",
        "log",
        "torch_dtype",
        "llama_gguf",
        "llama_threads",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise RuntimeError(f"Configuration is missing fields: {missing}")

    backend = str(data["backend"]).lower()
    if backend not in {"mlx", "cuda", "llamacpp"}:
        raise RuntimeError("backend must be one of: mlx, cuda, llamacpp")

    try:
        max_tokens = int(data["max_tokens"])
        mlx_cache_mib = int(data["mlx_cache_mib"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("max_tokens and mlx_cache_mib must be integers") from error
    if max_tokens < 1:
        raise RuntimeError("max_tokens must be positive")
    if mlx_cache_mib < 0:
        raise RuntimeError("mlx_cache_mib must be nonnegative")

    torch_dtype = str(data["torch_dtype"])
    if torch_dtype not in {"bfloat16", "float16", "float32"}:
        raise RuntimeError("torch_dtype must be bfloat16, float16, or float32")

    threads = data["llama_threads"]
    if threads is not None:
        try:
            threads = int(threads)
        except (TypeError, ValueError) as error:
            raise RuntimeError("llama_threads must be null or a positive integer") from error
        if threads < 1:
            raise RuntimeError("llama_threads must be null or a positive integer")

    base = config_path.parent
    log_path = Path(str(data["log"])).expanduser()
    if not log_path.is_absolute():
        log_path = base / log_path
    gguf = Path(str(data["llama_gguf"])).expanduser()
    if not gguf.is_absolute():
        gguf = base / gguf

    model = str(data["model"])
    revision = str(data["model_revision"])
    if not model:
        raise RuntimeError("model must be a nonempty string")
    if not revision:
        raise RuntimeError("model_revision must be a nonempty string")

    return Settings(
        config_path=config_path,
        backend=backend,
        model=_resolve_model_source(model, base, backend),
        model_revision=revision,
        max_tokens=max_tokens,
        mlx_cache_mib=mlx_cache_mib,
        log_path=log_path.resolve(),
        torch_dtype=torch_dtype,
        llama_gguf=gguf.resolve(),
        llama_threads=threads,
    )
