from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import scan_cache_dir

from .hardware import detect_host, profile_compatibility


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
    artifacts: list[dict[str, Any]] | None = None,
    max_input_tokens: int | None = None,
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
    if artifacts is not None:
        record["artifacts"] = artifacts
    if max_input_tokens is not None:
        if int(max_input_tokens) < 1:
            raise ValueError("max_input_tokens must be positive")
        record["max_input_tokens"] = int(max_input_tokens)
    profiles[key] = record
    _write_registry(config_path, data)



def installation_record(config_path: Path, model_id: str, backend: str) -> dict[str, Any] | None:
    """Return a copy of one profile's local installation record, if present."""
    data = load_registry(config_path)
    profiles = data.get("profiles", {}) if isinstance(data, dict) else {}
    if not isinstance(profiles, dict):
        return None
    record = profiles.get(profile_key(model_id, backend))
    return dict(record) if isinstance(record, dict) else None



def unmark_installed(config_path: Path, model_id: str, backend: str) -> dict[str, Any] | None:
    data = load_registry(config_path)
    profiles = data.setdefault("profiles", {})
    if not isinstance(profiles, dict):
        return None
    removed = profiles.pop(profile_key(model_id, backend), None)
    _write_registry(config_path, data)
    return removed if isinstance(removed, dict) else None



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


def _declared_downloads(profile: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    download = profile.get("download")
    if isinstance(download, dict):
        result.append(download)
    downloads = profile.get("downloads")
    if isinstance(downloads, list):
        result.extend(item for item in downloads if isinstance(item, dict))
    return result


def _local_dir_present(config_path: Path, value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    if not path.is_dir():
        return False
    try:
        return any(item.is_file() and item.name != ".gitkeep" for item in path.rglob("*"))
    except OSError:
        return False


def _local_model_present(config_path: Path, profile: dict[str, Any]) -> bool:
    value = profile.get("model")
    if not isinstance(value, str) or _looks_like_hf_repo(value):
        return False
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    if not path.exists():
        return False
    if path.is_file():
        return True
    try:
        return any(item.is_file() and item.name != ".gitkeep" for item in path.rglob("*"))
    except OSError:
        return False


def _looks_like_local_model_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return value.startswith(("./", "../", "~/", "/", "models/"))


def _artifact_ready(config_path: Path, profile: dict[str, Any], hf_repos: set[str]) -> tuple[bool, bool]:
    """Validate the artifacts that provision this exact profile.

    Explicit ``download``/``downloads`` declarations are the artifact contract.
    ``profile["model"]`` is not necessarily an artifact locator: some engines use
    a runtime-facing alias there (for example ``von``).  When downloads are
    declared, validate them and only additionally validate ``model`` when it is
    an explicit local path.  Without declared downloads, ``model`` remains the
    fallback artifact locator for Hub-backed and managed-local profiles.
    """
    checks: list[bool] = []
    downloads = _declared_downloads(profile)
    for download in downloads:
        local_dir = download.get("local_dir")
        if local_dir:
            checks.append(_local_dir_present(config_path, local_dir))
        elif _looks_like_hf_repo(download.get("repo_id")):
            checks.append(str(download["repo_id"]) in hf_repos)

    model = profile.get("model")
    if downloads:
        if _looks_like_local_model_path(model):
            checks.append(_local_model_present(config_path, profile))
    elif _looks_like_hf_repo(model):
        checks.append(str(model) in hf_repos)
    elif _looks_like_local_model_path(model):
        checks.append(_local_model_present(config_path, profile))

    return (all(checks) if checks else True), bool(checks)


def _nimble_profile_matches(runtime_root: Path, profile: dict[str, Any]) -> bool:
    if profile.get("installer") != "nimble":
        return True
    expected = profile.get("repo_id")
    config_name = str(profile.get("model_config", "nimble-model.json"))
    path = runtime_root / config_name
    if not expected or not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("model_id") == expected


def _looks_like_hf_repo(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith(("./", "../", "~", "/", "models/")):
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
    """Return local installation state without making network requests.

    The registry is authoritative per ``model_id::backend``. A shared runtime
    directory or a shared Hugging Face cache entry must never make a different
    backend appear installed. Filesystem/cache checks only validate the exact
    registered profile.
    """
    registry = load_registry(config_path)
    records = registry.get("profiles", {}) if isinstance(registry, dict) else {}
    hf_repos = _cached_hf_repos()
    runtime_root = _runtime_root(config_path, config_data)
    host = detect_host()
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
            backend = str(backend)
            key = profile_key(model_id, backend)
            record = records.get(key) if isinstance(records, dict) else None
            explicit = isinstance(record, dict)
            active = model_id == active_model_id and backend == active_backend

            runtime_key = profile.get("runtime_key")
            env_dir = runtime_root / str(runtime_key) if runtime_key else runtime_root / "__missing__"
            runtime_ready = bool(runtime_key) and _runtime_python(env_dir).is_file()

            artifact_ready, has_declared_artifact = _artifact_ready(config_path, profile, hf_repos)
            artifact_ready = artifact_ready and _nimble_profile_matches(runtime_root, profile)
            weights_cached = artifact_ready if has_declared_artifact else False

            installed = explicit and runtime_ready and artifact_ready
            verified = bool(explicit and record.get("verified_at")) and installed
            compatibility = profile_compatibility(backend, profile, host=host)

            if active and installed:
                status = "active"
            elif active:
                status = "selected"
            elif verified:
                status = "verified"
            elif installed:
                status = "installed"
            elif explicit and not runtime_ready:
                status = "runtime-missing"
            elif explicit and not artifact_ready:
                status = "weights-missing"
            elif runtime_ready or weights_cached:
                status = "unregistered"
            else:
                status = "not-installed"

            rows.append(
                {
                    "model_id": model_id,
                    "label": str(entry.get("label", model_id)),
                    "engine": engine,
                    "backend": backend,
                    "model": profile.get("model"),
                    "installed": installed,
                    "registered": explicit,
                    "verified": verified,
                    "active": active,
                    "runtime_ready": runtime_ready,
                    "weights_cached": weights_cached,
                    "host_compatible": bool(compatibility["compatible"]),
                    "compatibility": compatibility,
                    "status": status,
                    "max_input_tokens": (
                        int(record["max_input_tokens"])
                        if explicit and record.get("max_input_tokens") is not None
                        else None
                    ),
                }
            )

    rows.sort(key=lambda row: (not row["active"], row["label"].lower(), row["backend"]))
    return rows
