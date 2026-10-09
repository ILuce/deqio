from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from huggingface_hub import scan_cache_dir

from .hardware import detect_host, profile_compatibility
from .process_lock import process_lock


STATE_DIR_NAME = ".deqio"
REGISTRY_NAME = "installed-models.json"
REGISTRY_LOCK_NAME = "installed-models.lock"
# Registry block written while a shared runtime is being updated (B1).
UPDATE_PENDING_KEY = "update_pending"


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
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            stream.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except OSError:
                pass
        temp.replace(path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _registry_lock_path(config_path: Path) -> Path:
    return registry_path(config_path).with_name(REGISTRY_LOCK_NAME)


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
    with process_lock(_registry_lock_path(config_path), timeout=10.0):
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
            # A real readiness probe passed: an interrupted runtime update no
            # longer applies to this profile.
            record.pop(UPDATE_PENDING_KEY, None)
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
    with process_lock(_registry_lock_path(config_path), timeout=10.0):
        data = load_registry(config_path)
        profiles = data.setdefault("profiles", {})
        if not isinstance(profiles, dict):
            return None
        removed = profiles.pop(profile_key(model_id, backend), None)
        _write_registry(config_path, data)
        return removed if isinstance(removed, dict) else None



def begin_runtime_update(config_path: Path, runtime_key: str, keys: Iterable[str]) -> list[str]:
    """Mark the registered profiles of a runtime that is about to be rebuilt.

    ``verified_at`` stays the single source of truth for "verified". It moves,
    with the previous ``source``, into an ``update_pending`` block, so a failed or
    interrupted update leaves every profile of that runtime unverified (and its
    provenance unresolved) instead of claiming a verification the rebuilt runtime
    never passed. A record that is already pending keeps its saved state.
    """
    with process_lock(_registry_lock_path(config_path), timeout=10.0):
        data = load_registry(config_path)
        profiles = data.setdefault("profiles", {})
        marked: list[str] = []
        started_at = datetime.now(timezone.utc).isoformat()
        for key in keys:
            record = profiles.get(key) if isinstance(profiles, dict) else None
            if not isinstance(record, dict):
                continue
            if not isinstance(record.get(UPDATE_PENDING_KEY), dict):
                record[UPDATE_PENDING_KEY] = {
                    "runtime_key": str(runtime_key),
                    "started_at": started_at,
                    "verified_at": record.get("verified_at"),
                    "source": record.get("source"),
                }
            record.pop("verified_at", None)
            record["source"] = "updating"
            marked.append(str(key))
        if marked:
            _write_registry(config_path, data)
        return marked


def finish_runtime_update(config_path: Path, keys: Iterable[str]) -> None:
    """Restore profiles marked by ``begin_runtime_update`` after a verified update.

    The shared runtime passed a real readiness probe and the other profiles'
    weights were not touched, so their previous verification is restored.
    Profiles re-verified meanwhile keep their fresh record.
    """
    with process_lock(_registry_lock_path(config_path), timeout=10.0):
        data = load_registry(config_path)
        profiles = data.setdefault("profiles", {})
        changed = False
        for key in keys:
            record = profiles.get(key) if isinstance(profiles, dict) else None
            if not isinstance(record, dict):
                continue
            pending = record.pop(UPDATE_PENDING_KEY, None)
            if not isinstance(pending, dict):
                continue
            changed = True
            if not record.get("verified_at") and pending.get("verified_at"):
                record["verified_at"] = pending["verified_at"]
            if record.get("source") == "updating":
                record["source"] = pending.get("source") or "install"
        if changed:
            _write_registry(config_path, data)


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


def _cached_hf_state() -> dict[str, set[str]]:
    """Return locally resolvable Hub revisions keyed by repo id.

    The set contains both immutable commit hashes and cached refs/tags.  A repo
    being present in the Hub cache is not enough for offline serving when a
    profile requests a specific revision (for example ``v1.0``): that exact ref
    must also be locally resolvable.
    """
    try:
        state: dict[str, set[str]] = {}
        for repo in scan_cache_dir().repos:
            revisions: set[str] = set()
            for revision in repo.revisions:
                revisions.add(str(revision.commit_hash))
                revisions.update(str(ref) for ref in revision.refs)
            state[str(repo.repo_id)] = revisions
        return state
    except Exception:
        return {}


def _hf_artifact_cached(repo_id: str, revision: Any, hf_state: dict[str, set[str]]) -> bool:
    cached = hf_state.get(repo_id)
    if cached is None:
        return False
    if revision is None or str(revision) in {"", "upstream-latest"}:
        return True
    return str(revision) in cached


def _declared_downloads(profile: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    download = profile.get("download")
    if isinstance(download, dict):
        result.append(download)
    downloads = profile.get("downloads")
    if isinstance(downloads, list):
        result.extend(item for item in downloads if isinstance(item, dict))
    return result


def _local_dir_path(config_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path


def _local_dir_present(config_path: Path, value: Any) -> bool:
    path = _local_dir_path(config_path, value)
    if path is None or not path.is_dir():
        return False
    try:
        return any(item.is_file() and item.name != ".gitkeep" for item in path.rglob("*"))
    except OSError:
        return False


def _local_download_present(config_path: Path, download: dict[str, Any]) -> bool:
    """Validate the concrete artifact declared inside a managed local directory.

    Multiple profiles may deliberately share one directory (for example JevK5
    4B and 9B GGUF).  Directory non-emptiness alone must therefore never make
    a sibling profile look installed.
    """
    path = _local_dir_path(config_path, download.get("local_dir"))
    if path is None or not path.is_dir():
        return False
    filename = download.get("filename")
    if isinstance(filename, str) and filename:
        return (path / filename).is_file()
    pattern = download.get("filename_pattern")
    if isinstance(pattern, str) and pattern:
        try:
            return any(item.is_file() for item in path.rglob(pattern))
        except OSError:
            return False
    return _local_dir_present(config_path, str(path))


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


def _artifact_ready(
    config_path: Path,
    profile: dict[str, Any],
    hf_state: dict[str, set[str]],
    record: dict[str, Any] | None = None,
) -> tuple[bool, bool]:
    """Validate the artifacts that provision this exact profile.

    Explicit ``download``/``downloads`` declarations are the artifact contract.
    ``profile["model"]`` is not necessarily an artifact locator: some engines use
    a runtime-facing alias there (for example ``von``).  When downloads are
    declared, validate them and only additionally validate ``model`` when it is
    an explicit local path.  Without declared downloads, ``model`` remains the
    fallback artifact locator for Hub-backed and managed-local profiles.
    """
    checks: list[bool] = []
    attestations = (
        [item for item in record.get("artifacts", []) if isinstance(item, dict)]
        if isinstance(record, dict) and isinstance(record.get("artifacts"), list)
        else []
    )

    def installed_revision(repo_id: str, role: Any, fallback: Any) -> Any:
        """Prefer the immutable revision recorded for this installation.

        Pin-on-install profiles (Basal 1.5 today, and potentially other model
        families later) intentionally replace a mutable catalog revision such
        as ``main`` with a commit SHA during installation.  The installation
        registry is therefore the source of truth when checking whether that
        exact installed profile is still locally usable.
        """
        for artifact in attestations:
            if artifact.get("source") != "huggingface":
                continue
            if str(artifact.get("repo_id", "")) != repo_id:
                continue
            artifact_role = artifact.get("role")
            if role is not None and artifact_role is not None and str(artifact_role) != str(role):
                continue
            resolved = artifact.get("resolved_revision")
            if resolved:
                return resolved
            requested = artifact.get("requested_revision")
            if requested:
                return requested
        return fallback

    downloads = _declared_downloads(profile)
    for download in downloads:
        local_dir = download.get("local_dir")
        if local_dir:
            checks.append(_local_download_present(config_path, download))
        elif _looks_like_hf_repo(download.get("repo_id")):
            repo_id = str(download["repo_id"])
            checks.append(
                _hf_artifact_cached(
                    repo_id,
                    installed_revision(repo_id, download.get("role"), download.get("revision")),
                    hf_state,
                )
            )

    model = profile.get("model")
    if downloads:
        if _looks_like_local_model_path(model):
            checks.append(_local_model_present(config_path, profile))
    elif _looks_like_hf_repo(model):
        repo_id = str(model)
        checks.append(
            _hf_artifact_cached(
                repo_id,
                installed_revision(repo_id, None, profile.get("model_revision")),
                hf_state,
            )
        )
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
    if data.get("model_id") != expected:
        return False
    expected_revision = profile.get("model_revision")
    if expected_revision and expected_revision != "upstream-latest":
        return data.get("revision") == expected_revision
    return True


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
    hf_state = _cached_hf_state()
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
            if runtime_ready and profile.get("installer") in {"llama_cpp", "jevk5_gguf"}:
                binary = env_dir / ("Scripts" if os.name == "nt" else "bin") / (
                    "llama-server.exe" if os.name == "nt" else "llama-server"
                )
                runtime_ready = binary.is_file()

            artifact_ready, has_declared_artifact = _artifact_ready(
                config_path, profile, hf_state, record if explicit else None
            )
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
                    "family": profile.get("family"),
                    "runtime": profile.get("runtime"),
                    "precision": profile.get("precision"),
                    "quantization": profile.get("quantization"),
                    "source": profile.get("source"),
                    "capabilities": dict(profile.get("capabilities") or {}),
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
