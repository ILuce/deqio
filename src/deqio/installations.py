from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import scan_cache_dir

from .catalog import SUPPORTED_BACKENDS


STATE_DIR_NAME = ".deqio"
REGISTRY_NAME = "installed-models.json"


def profile_key(model_id: str, backend: str) -> str:
    return f"{model_id}::{backend}"


def registry_path(config_path: Path) -> Path:
    return config_path.parent / STATE_DIR_NAME / REGISTRY_NAME


def load_registry(config_path: Path) -> dict[str, Any]:
    path = registry_path(config_path)
    if not path.is_file():
        return {"schema_version": 1, "profiles": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "profiles": {}}
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
        return {"schema_version": 1, "profiles": {}}
    return data


def _write_registry(config_path: Path, data: dict[str, Any]) -> None:
    path = registry_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def mark_installed(
    config_path: Path,
    model_id: str,
    backend: str,
    *,
    verified: bool = False,
    source: str = "install",
) -> None:
    data = load_registry(config_path)
    profiles = data.setdefault("profiles", {})
    key = profile_key(model_id, backend)
    now = datetime.now(timezone.utc).isoformat()
    current = profiles.get(key) if isinstance(profiles.get(key), dict) else {}
    record = dict(current)
    record.update(
        {
            "model_id": model_id,
            "backend": backend,
            "installed_at": record.get("installed_at", now),
            "source": source,
        }
    )
    if verified:
        record["verified_at"] = now
    profiles[key] = record
    _write_registry(config_path, data)


def host_backends() -> tuple[str, ...]:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine == "arm64":
        return ("mlx", "mps")
    if system in {"Linux", "Windows"}:
        return ("cuda",)
    return SUPPORTED_BACKENDS


def _runtime_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _runtime_root(config_path: Path, data: dict[str, Any]) -> Path:
    value = str(data.get("runtime_dir", ".model-runtimes"))
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _cached_hf_repos() -> set[str]:
    try:
        return {str(repo.repo_id) for repo in scan_cache_dir().repos}
    except Exception:
        return set()


def _local_download_present(config_path: Path, profile: dict[str, Any]) -> bool:
    download = profile.get("download")
    if not isinstance(download, dict) or not download.get("local_dir"):
        return False
    path = Path(str(download["local_dir"])).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    if not path.is_dir():
        return False
    try:
        return any(item.is_file() and item.name != ".gitkeep" for item in path.rglob("*"))
    except OSError:
        return False


def _looks_like_hf_repo(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith(("./", "../", "~", "/")):
        return False
    return value.count("/") == 1 and " " not in value


def installed_profiles(
    *,
    config_path: Path,
    config_data: dict[str, Any],
    catalog: dict[str, Any],
    active_model_id: str | None = None,
    active_backend: str | None = None,
) -> list[dict[str, Any]]:
    """Return local installation state without making network requests."""
    registry = load_registry(config_path)
    records = registry.get("profiles", {}) if isinstance(registry, dict) else {}
    hf_repos = _cached_hf_repos()
    runtime_root = _runtime_root(config_path, config_data)
    compatible = set(host_backends())
    rows: list[dict[str, Any]] = []

    for entry in catalog.get("models", []):
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id", ""))
        engine = str(entry.get("engine", ""))
        profiles = entry.get("backends") or {}
        if not isinstance(profiles, dict):
            continue

        for backend, profile in profiles.items():
            if not isinstance(profile, dict):
                continue
            key = profile_key(model_id, str(backend))
            record = records.get(key) if isinstance(records, dict) else None
            explicit = isinstance(record, dict)
            active = model_id == active_model_id and str(backend) == active_backend

            if engine == "semif":
                runtime_ready = True
            else:
                runtime_key = profile.get("runtime_key")
                env_dir = runtime_root / str(runtime_key) if runtime_key else runtime_root / "__missing__"
                runtime_ready = bool(runtime_key) and _runtime_python(env_dir).is_file()

            local_weights = _local_download_present(config_path, profile)
            model_source = profile.get("model")
            hf_cached = bool(_looks_like_hf_repo(model_source) and str(model_source) in hf_repos)
            weights_cached = local_weights or hf_cached

            download = profile.get("download")
            if engine == "semif" and isinstance(download, dict) and download.get("local_dir"):
                installed = local_weights
            elif engine == "semif":
                installed = explicit or weights_cached
            else:
                installed = runtime_ready and (explicit or weights_cached)
            verified = bool(isinstance(record, dict) and record.get("verified_at")) and installed

            if active and installed:
                status = "active"
            elif active:
                status = "selected"
            elif verified:
                status = "verified"
            elif installed:
                status = "installed"
            elif runtime_ready:
                status = "runtime-only"
            else:
                status = "not-installed"

            rows.append(
                {
                    "model_id": model_id,
                    "label": str(entry.get("label", model_id)),
                    "engine": engine,
                    "backend": str(backend),
                    "model": model_source,
                    "installed": installed,
                    "verified": verified,
                    "active": active,
                    "runtime_ready": runtime_ready,
                    "weights_cached": weights_cached,
                    "host_compatible": str(backend) in compatible,
                    "status": status,
                }
            )

    rows.sort(key=lambda row: (not row["active"], row["label"].lower(), row["backend"]))
    return rows
