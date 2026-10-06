from __future__ import annotations

import hashlib
import json
from typing import Any

PROFILE_IDENTITY_SCHEMA_VERSION = 1


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _artifact_identity(identity: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in identity.get("artifacts", []) if isinstance(identity.get("artifacts"), list) else []:
        if not isinstance(raw, dict):
            continue
        repo_id = raw.get("repo_id")
        revision = raw.get("resolved_revision") or raw.get("requested_revision")
        if not repo_id or not revision:
            continue
        rows.append({
            "source": raw.get("source"),
            "repo_id": str(repo_id),
            "revision": str(revision),
            "role": raw.get("role"),
        })
    rows.sort(key=lambda row: (str(row.get("repo_id")), str(row.get("role")), str(row.get("revision"))))
    return rows


def build_profile_identity(
    *,
    model_id: str,
    backend: str,
    engine: str,
    runtime_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    identity = runtime_identity if isinstance(runtime_identity, dict) else {}
    runtime_metadata = identity.get("runtime_metadata")
    if not isinstance(runtime_metadata, dict):
        runtime_metadata = {}
    stable_runtime_metadata = {
        key: runtime_metadata[key]
        for key in (
            "runtime_source_commit",
            "package_schema_version",
            "format_version",
            "basal_mode",
            "basal_backend",
            "official_runtime",
        )
        if key in runtime_metadata
    }
    canonical = {
        "model_id": str(model_id),
        "backend": str(backend),
        "engine": str(engine),
        "family": identity.get("family"),
        "runtime": identity.get("runtime"),
        "runtime_version": identity.get("runtime_version"),
        "model": identity.get("model"),
        "source": identity.get("source"),
        "precision": _json_value(identity.get("precision")),
        "quantization": _json_value(identity.get("quantization")),
        "runtime_metadata": _json_value(stable_runtime_metadata),
        "artifacts": _artifact_identity(identity),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "schema_version": PROFILE_IDENTITY_SCHEMA_VERSION,
        "canonical_id": hashlib.sha256(encoded).hexdigest(),
        **canonical,
    }


def profile_identity_from_summary(model: dict[str, Any]) -> dict[str, Any] | None:
    value = model.get("profile_identity")
    if not isinstance(value, dict):
        return None
    canonical_id = value.get("canonical_id")
    if not isinstance(canonical_id, str) or not canonical_id:
        return None
    return value


def profile_label(model: dict[str, Any]) -> str:
    model_id = str(model.get("model_id", "?"))
    backend = str(model.get("backend", "?"))
    identity = profile_identity_from_summary(model) or {}
    precision = identity.get("precision")
    if isinstance(precision, str) and precision:
        return f"{model_id}:{backend}:{precision}"
    quantization = identity.get("quantization")
    if isinstance(quantization, dict):
        quant = quantization.get("quantization") or quantization.get("kind")
        if quant:
            return f"{model_id}:{backend}:{quant}"
    return f"{model_id}:{backend}"
