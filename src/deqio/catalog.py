from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CATALOG_NAME = "models.json"
SUPPORTED_BACKENDS = ("mlx", "mps", "cuda")


def load_catalog(path: str | Path) -> dict[str, Any]:
    catalog_path = Path(path).expanduser().resolve()
    if not catalog_path.is_file():
        raise RuntimeError(f"Model catalog does not exist: {catalog_path}")
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in model catalog {catalog_path}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise RuntimeError("Model catalog must contain a top-level 'models' array")
    return data


def catalog_index(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in catalog.get("models", []):
        if not isinstance(item, dict) or not item.get("id"):
            raise RuntimeError("Every model catalog entry must have an id")
        model_id = str(item["id"])
        if model_id in result:
            raise RuntimeError(f"Duplicate model id in catalog: {model_id}")
        result[model_id] = item
    return result


def get_model(catalog: dict[str, Any], model_id: str) -> dict[str, Any]:
    try:
        return catalog_index(catalog)[model_id]
    except KeyError as error:
        raise RuntimeError(f"Unknown model id: {model_id}") from error


def get_profile(catalog: dict[str, Any], model_id: str, backend: str) -> dict[str, Any]:
    model = get_model(catalog, model_id)
    profiles = model.get("backends")
    if not isinstance(profiles, dict) or backend not in profiles:
        supported = ", ".join(sorted(profiles or {})) or "none"
        raise RuntimeError(
            f"Model {model_id!r} does not support backend {backend!r}. "
            f"Supported backends: {supported}"
        )
    profile = profiles[backend]
    if not isinstance(profile, dict):
        raise RuntimeError(f"Invalid profile for {model_id}/{backend}")
    return profile


def public_catalog(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in catalog.get("models", []):
        profiles = item.get("backends") or {}
        rows.append(
            {
                "id": item.get("id"),
                "engine": item.get("engine"),
                "label": item.get("label"),
                "description": item.get("description"),
                "source": item.get("source"),
                "backends": sorted(profiles),
            }
        )
    return rows


def apply_selection(
    data: dict[str, Any],
    model_entry: dict[str, Any],
    profile: dict[str, Any],
    backend: str,
) -> dict[str, Any]:
    """Return config data updated for one catalogued model/backend profile."""
    updated = dict(data)
    updated["engine"] = str(model_entry["engine"])
    updated["model_id"] = str(model_entry["id"])
    updated["backend"] = backend
    updated["model"] = str(profile["model"])
    updated["model_revision"] = str(
        profile.get("model_revision", model_entry.get("id", "upstream"))
    )
    updated.setdefault("model_catalog", DEFAULT_CATALOG_NAME)
    updated.setdefault("runtime_dir", ".model-runtimes")
    return updated
