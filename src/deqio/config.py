from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("config.json")
ENV_OVERRIDES = {
    "DEQIO_ENGINE": "engine",
    "DEQIO_MODEL_ID": "model_id",
    "DEQIO_BACKEND": "backend",
    "DEQIO_MODEL": "model",
    "DEQIO_MODEL_REVISION": "model_revision",
    "DEQIO_MAX_TOKENS": "max_tokens",
    "DEQIO_MLX_CACHE_MIB": "mlx_cache_mib",
    "DEQIO_LOG": "log",
    "DEQIO_TORCH_DTYPE": "torch_dtype",
    "DEQIO_RUNTIME_DIR": "runtime_dir",
    "DEQIO_MODEL_CATALOG": "model_catalog",
    "DEQIO_SIDECAR_STARTUP_SECONDS": "sidecar_startup_seconds",
}
MODEL_SELECTION_ENV_VARS = (
    "DEQIO_ENGINE",
    "DEQIO_MODEL_ID",
    "DEQIO_BACKEND",
    "DEQIO_MODEL",
    "DEQIO_MODEL_REVISION",
)


@dataclass(frozen=True)
class Settings:
    config_path: Path
    engine: str
    model_id: str
    backend: str
    model: str
    model_revision: str
    max_tokens: int
    mlx_cache_mib: int
    log_path: Path
    torch_dtype: str
    runtime_dir: Path
    model_catalog: Path
    sidecar_startup_seconds: int


def _coerce_override(key: str, value: str) -> Any:
    if key in {"max_tokens", "mlx_cache_mib", "sidecar_startup_seconds"}:
        return int(value)
    return value


def _resolve_model_source(value: str, base: Path, backend: str, engine: str) -> str:
    path = Path(value).expanduser()
    candidate = path if path.is_absolute() else base / path

    # MLX is intentionally local in this server configuration. Torch backends
    # keep Hugging Face repo IDs as remote model sources.
    if (engine == "semif" and backend == "mlx") or path.is_absolute() or value.startswith(("./", "../", "~")):
        return str(candidate.resolve())
    if candidate.exists():
        return str(candidate.resolve())
    return value


def read_config_data(path: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    configured = path or os.environ.get("DEQIO_CONFIG") or DEFAULT_CONFIG_PATH
    config_path = Path(configured).expanduser().resolve()
    if not config_path.is_file():
        raise RuntimeError(
            f"Configuration file does not exist: {config_path}. "
            "Create config.json or set DEQIO_CONFIG."
        )
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in {config_path}: {error}") from error
    if not isinstance(data, dict):
        raise RuntimeError("The root configuration value must be a JSON object")
    return config_path, data


def write_config_data(path: Path, data: dict[str, Any]) -> None:
    """Atomically persist configuration so live model switches cannot leave partial JSON."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def settings_from_data(
    config_path: Path,
    source_data: dict[str, Any],
    *,
    apply_environment: bool = True,
) -> Settings:
    data = dict(source_data)

    if apply_environment:
        for env_name, key in ENV_OVERRIDES.items():
            value = os.environ.get(env_name)
            if value not in (None, ""):
                try:
                    data[key] = _coerce_override(key, value)
                except ValueError as error:
                    raise RuntimeError(f"Invalid value for {env_name}: {value!r}") from error

    data.setdefault("engine", "semif")
    data.setdefault("model_id", "semif-qwen3.5-4b")
    data.setdefault("runtime_dir", ".model-runtimes")
    data.setdefault("model_catalog", "models.json")
    data.setdefault("sidecar_startup_seconds", 900)

    required = {
        "engine",
        "model_id",
        "backend",
        "model",
        "model_revision",
        "max_tokens",
        "mlx_cache_mib",
        "log",
        "torch_dtype",
        "sidecar_startup_seconds",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise RuntimeError(f"Configuration is missing fields: {missing}")

    engine = str(data["engine"]).lower().strip()
    if engine not in {"semif", "kev", "decider", "laya", "von"}:
        raise RuntimeError("engine must be one of: semif, kev, decider, laya, von")
    model_id = str(data["model_id"]).strip()
    if not model_id:
        raise RuntimeError("model_id must be a nonempty string")

    backend = str(data["backend"]).lower()
    if backend not in {"mlx", "mps", "cuda"}:
        raise RuntimeError("backend must be one of: mlx, mps, cuda")

    try:
        max_tokens = int(data["max_tokens"])
        mlx_cache_mib = int(data["mlx_cache_mib"])
        sidecar_startup_seconds = int(data["sidecar_startup_seconds"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("max_tokens, mlx_cache_mib and sidecar_startup_seconds must be integers") from error
    if max_tokens < 1:
        raise RuntimeError("max_tokens must be positive")
    if mlx_cache_mib < 0:
        raise RuntimeError("mlx_cache_mib must be nonnegative")
    if sidecar_startup_seconds < 1:
        raise RuntimeError("sidecar_startup_seconds must be positive")

    torch_dtype = str(data["torch_dtype"])
    if torch_dtype not in {"bfloat16", "float16", "float32"}:
        raise RuntimeError("torch_dtype must be bfloat16, float16, or float32")

    base = config_path.parent
    log_path = Path(str(data["log"])).expanduser()
    if not log_path.is_absolute():
        log_path = base / log_path
    runtime_dir = Path(str(data["runtime_dir"])).expanduser()
    if not runtime_dir.is_absolute():
        runtime_dir = base / runtime_dir
    model_catalog = Path(str(data["model_catalog"])).expanduser()
    if not model_catalog.is_absolute():
        model_catalog = base / model_catalog

    model = str(data["model"])
    revision = str(data["model_revision"])
    if not model:
        raise RuntimeError("model must be a nonempty string")
    if not revision:
        raise RuntimeError("model_revision must be a nonempty string")

    return Settings(
        config_path=config_path,
        engine=engine,
        model_id=model_id,
        backend=backend,
        model=_resolve_model_source(model, base, backend, engine),
        model_revision=revision,
        max_tokens=max_tokens,
        mlx_cache_mib=mlx_cache_mib,
        log_path=log_path.resolve(),
        torch_dtype=torch_dtype,
        runtime_dir=runtime_dir.resolve(),
        model_catalog=model_catalog.resolve(),
        sidecar_startup_seconds=sidecar_startup_seconds,
    )


def load_settings(path: str | Path | None = None) -> Settings:
    config_path, data = read_config_data(path)
    return settings_from_data(config_path, data, apply_environment=True)
