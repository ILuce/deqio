"""B10: one rule for the model an engine loads.

Launchers use the catalog profile; `config.json` / `DEQIO_MODEL*` are a mirror
of the selected profile and a different value is rejected before anything is
spawned, so response provenance (`profile["model"]`) can never disagree with
the weights the sidecar actually loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deqio import systemone_runtime as runtime_module
from deqio.catalog import get_profile, load_catalog
from deqio.config import read_config_data, settings_from_data
from deqio.systemone_runtime import SystemOneRuntime
from deqio.workspace import ensure_workspace


def _launcher_settings(config_path: Path, **overrides) -> SimpleNamespace:
    values = {
        "engine": "semif", "backend": "mps", "model_id": "semif-qwen3.5-4b",
        "model": "someone/other-model", "model_revision": "0000000000000000000000000000000000000000",
        "max_tokens": 4096, "mlx_cache_mib": 256, "torch_dtype": "bfloat16", "hf_offline_runtime": True,
        "config_path": config_path, "runtime_dir": config_path.parent / ".model-runtimes",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _catalog(config_path: Path) -> dict:
    return load_catalog(config_path.parent / "models.json")


def test_semif_launcher_loads_the_catalog_model_not_the_config_value(tmp_path: Path) -> None:
    config = ensure_workspace(tmp_path / "ws")
    profile = get_profile(_catalog(config), "semif-qwen3.5-4b", "mps")
    settings = _launcher_settings(config)

    command = SystemOneRuntime._command(settings, profile, tmp_path / "env", tmp_path / "env" / "bin" / "python", 9000, {})

    assert command[command.index("--model") + 1] == profile["model"]
    assert command[command.index("--revision") + 1] == str(profile["model_revision"])
    assert "someone/other-model" not in command


def test_open_jev_launcher_loads_the_catalog_checkpoint_not_the_config_value(tmp_path: Path) -> None:
    config = ensure_workspace(tmp_path / "ws")
    profile = get_profile(_catalog(config), "open-jev-2b", "cuda")
    settings = _launcher_settings(
        config, engine="open-jev", backend="cuda", model_id="open-jev-2b", model="/evil/checkpoint"
    )

    command = SystemOneRuntime._command(settings, profile, tmp_path / "env", tmp_path / "env" / "bin" / "python", 9000, {})

    checkpoint = command[command.index("--checkpoint") + 1]
    assert checkpoint == str((config.parent / profile["model"]).resolve())
    assert "/evil/checkpoint" not in command


def _runtime_for(config: Path, model_id: str, backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = get_profile(_catalog(config), model_id, backend)
    env_dir = config.parent / ".model-runtimes" / str(profile["runtime_key"])
    (env_dir / "bin").mkdir(parents=True, exist_ok=True)
    for executable in ("python", "basal-serve"):
        (env_dir / "bin" / executable).write_text("", encoding="utf-8")
    monkeypatch.setattr(SystemOneRuntime, "_validate_accelerator", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr(SystemOneRuntime, "_basal_runtime_metadata", staticmethod(lambda *args, **kwargs: {}))

    def no_spawn(*args, **kwargs):
        raise AssertionError("sidecar spawned")

    monkeypatch.setattr(runtime_module.subprocess, "Popen", no_spawn)


def _settings(config: Path, model_id: str, backend: str, **overrides):
    _, data = read_config_data(config)
    catalog = _catalog(config)
    entry = next(item for item in catalog["models"] if item["id"] == model_id)
    from deqio.catalog import apply_selection

    data = apply_selection(data, entry, get_profile(catalog, model_id, backend), backend)
    data.update(overrides)
    return settings_from_data(config, data, apply_environment=False)


@pytest.mark.parametrize(
    "override",
    [{"model": "someone/other-model"}, {"model_revision": "0000000000000000000000000000000000000000"}],
    ids=["model", "model_revision"],
)
def test_load_rejects_a_config_model_that_differs_from_the_catalog_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: dict
) -> None:
    config = ensure_workspace(tmp_path / "ws")
    _runtime_for(config, "kev-0.8b", "mps", monkeypatch)

    with pytest.raises(RuntimeError, match="catalog profile"):
        SystemOneRuntime.load(_settings(config, "kev-0.8b", "mps", **override))


def test_load_accepts_the_selected_catalog_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ensure_workspace(tmp_path / "ws")
    _runtime_for(config, "kev-0.8b", "mps", monkeypatch)

    with pytest.raises(AssertionError, match="sidecar spawned"):
        SystemOneRuntime.load(_settings(config, "kev-0.8b", "mps"))


def test_load_accepts_a_pin_on_install_revision_recorded_by_the_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin-on-install profiles legitimately run the recorded immutable SHA, not the catalog's `main`."""
    from deqio.installations import mark_installed

    config = ensure_workspace(tmp_path / "ws")
    profile = get_profile(_catalog(config), "basal-1.5-mini", "mlx")
    assert profile.get("pin_hf_revisions_at_install"), "test needs a pin-on-install profile"
    _runtime_for(config, "basal-1.5-mini", "mlx", monkeypatch)
    mark_installed(
        config, "basal-1.5-mini", "mlx", verified=True, source="install",
        artifacts=[{"source": "huggingface", "repo_id": profile["model"], "role": "basal-model",
                    "requested_revision": profile.get("model_revision"), "resolved_revision": "a" * 40}],
        max_input_tokens=4096,
    )

    with pytest.raises(AssertionError, match="sidecar spawned"):
        SystemOneRuntime.load(_settings(config, "basal-1.5-mini", "mlx"))


def test_runtime_identity_model_matches_the_catalog_profile(tmp_path: Path) -> None:
    config = ensure_workspace(tmp_path / "ws")
    profile = get_profile(_catalog(config), "semif-qwen3.5-4b", "mps")
    identity = runtime_module._runtime_identity(_launcher_settings(config), profile, "instance-1")
    assert identity["model"] == profile["model"]
    assert json.dumps(identity)  # serializable
