"""D12b: reproducible engine runtimes and an owner-side pin audit.

Every catalog profile carries an `exclude_newer` instant. Installers pass it to
every `uv pip install` that resolves from PyPI, so transitive dependencies
resolve the same way on every install (uv ignores releases uploaded later).
Commands that install one exact `torch==` pin from the PyTorch index run without
it: once a cutoff is set, uv drops every file of an index that publishes no
upload times. `python -m deqio.pin_audit` compares an installed workspace with
the catalog in one command line (the 0.5.7 handoff used shell snippets).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from deqio import model_manager as manager
from deqio.catalog import get_profile, load_catalog
from deqio.workspace import ensure_workspace

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = "2026-10-10T00:00:00Z"
INSTANT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _catalog() -> dict:
    return load_catalog(ROOT / "src" / "deqio" / "data" / "models.json")


def _pin_audit():
    return importlib.import_module("deqio.pin_audit")


def test_every_profile_declares_one_exclude_newer_instant_per_runtime() -> None:
    by_runtime: dict[str, set[str]] = {}
    missing = []
    for model in _catalog()["models"]:
        for backend, profile in model["backends"].items():
            if backend == "remote":
                # 0.6.1: a remote profile installs no packages, so it has no resolution to bound.
                continue
            value = str(profile.get("exclude_newer", ""))
            if not INSTANT.match(value):
                missing.append(f"{model['id']}::{backend}")
            by_runtime.setdefault(str(profile.get("runtime_key")), set()).add(value)
    assert missing == [], "profiles without an RFC 3339 UTC exclude_newer: " + ", ".join(missing)
    assert {key: values for key, values in by_runtime.items() if len(values) > 1} == {}
    assert set().union(*by_runtime.values()) == {CUTOFF}


def _captured_install_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str]]]:
    commands: list[list[str]] = []
    python = tmp_path / "python"
    monkeypatch.setattr(manager, "_ensure_venv", lambda *args, **kwargs: python)
    monkeypatch.setattr(manager, "_runtime_python", lambda env_dir: python)
    monkeypatch.setattr(manager, "_run", lambda command, **kwargs: commands.append(list(command)))
    monkeypatch.setattr(manager, "_checkout_nimble", lambda *args, **kwargs: tmp_path / "nimble-src")
    config = tmp_path / "config.json"
    data = {"runtime_dir": ".model-runtimes"}
    catalog = _catalog()
    captured: list[tuple[str, list[str]]] = []

    def run(model_id: str, backend: str) -> None:
        start = len(commands)
        profile = get_profile(catalog, model_id, backend)
        manager._install_runtime(config, {**data, "backend": backend}, profile, upgrade=True)
        captured.extend((profile.get("exclude_newer"), command) for command in commands[start:])

    for model_id, backend in (
        ("semif-qwen3.5-4b", "mlx"),
        ("laya-english", "mps"),
        ("d3", "cuda"),
        ("d3", "mlx"),
        ("basal-2.0-main", "gguf"),
        ("basal-2.0-main", "mlx"),
        ("nimble-9b", "mlx"),
        ("nimble-9b", "cuda"),
        ("decider-2b", "gguf"),
    ):
        run(model_id, backend)

    llama = get_profile(catalog, "kev-0.8b", "gguf")
    bin_dir = tmp_path / ".model-runtimes" / str(llama["runtime_key"]) / "bin"
    bin_dir.mkdir(parents=True)
    for tool in ("cmake", "ninja"):
        (bin_dir / tool).write_text("", encoding="utf-8")
    start = len(commands)
    with pytest.raises(RuntimeError, match="did not produce llama-server"):
        manager._install_runtime(config, {**data, "backend": "gguf"}, llama, upgrade=True)
    captured.extend((llama.get("exclude_newer"), command) for command in commands[start:])

    monkeypatch.setattr(manager, "_install_llama_cpp_runtime", lambda *args, **kwargs: tmp_path / "jevk5-gguf")
    run("jevk5-4b", "gguf")
    return [(cutoff, command) for cutoff, command in captured if command[:3] == ["uv", "pip", "install"]]


def test_installers_bound_every_pypi_resolution_by_the_profile_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installs = _captured_install_commands(tmp_path, monkeypatch)
    assert len(installs) >= 12, installs
    custom_index = [command for _, command in installs if "--index-url" in command]
    assert len(custom_index) == 1, custom_index  # Nimble CUDA torch wheel (Basal 2.0 is MLX/GGUF only; D3 torch comes from PyPI)
    for command in custom_index:
        assert "--exclude-newer" not in command, command
        specs = [token for token in command[5:] if not token.startswith("-") and not token.startswith("https://")]
        assert len(specs) == 1 and re.match(r"^torch==\d", specs[0]), command
    unbounded = [command for cutoff, command in installs if "--index-url" not in command and
                 (command.count("--exclude-newer") != 1 or command[command.index("--exclude-newer") + 1] != cutoff)]
    assert unbounded == [], "PyPI resolutions without the profile cutoff: " + repr(unbounded)


def test_install_refuses_a_profile_without_exclude_newer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(manager, "_ensure_venv", lambda *args, **kwargs: tmp_path / "python")
    monkeypatch.setattr(manager, "_run", lambda command, **kwargs: commands.append(list(command)))
    profile = {"runtime_key": "custom", "python": "3.12", "packages": ["example==1.0"]}

    with pytest.raises(RuntimeError, match="exclude_newer"):
        manager._install_runtime(tmp_path / "config.json", {"runtime_dir": ".model-runtimes"}, profile, upgrade=False)
    assert commands == []


def _site_packages(config: Path, runtime_key: str) -> Path:
    site = config.parent / ".model-runtimes" / runtime_key / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    return site


def _installed_from_git(site: Path, dist_info: str, url: str, commit: str) -> None:
    info = site / dist_info
    info.mkdir()
    (info / "direct_url.json").write_text(
        json.dumps({"url": url, "vcs_info": {"vcs": "git", "commit_id": commit}}), encoding="utf-8"
    )


def test_pin_audit_compares_installed_engine_commits_with_the_catalog(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = ensure_workspace(tmp_path / "ws")
    laya = _site_packages(config, "laya-torch")
    _installed_from_git(laya, "laya-0.3.28.dist-info", "https://github.com/NandhaKishorM/laya.git",
                        "a4a8921afebfd852bba0000475cfb6ab737a124c")
    (laya / "fastapi-0.141.1.dist-info").mkdir()
    (laya / "uvicorn-0.53.0.dist-info").mkdir()
    decider = _site_packages(config, "decider")
    _installed_from_git(decider, "decider_ai-1.8.1.dist-info", "https://github.com/Mapika/decider.git",
                        "50d0be0d7cb43d2066965ce5fa7f3fe4e489a60f")

    code = _pin_audit().main(["--config", str(config), "--offline"])

    out = capsys.readouterr().out
    rows = [line.split("\t") for line in out.splitlines() if not line.startswith("#")]
    assert ["laya-torch", "laya", "a4a8921afebfd852bba0000475cfb6ab737a124c",
            "a4a8921afebfd852bba0000475cfb6ab737a124c", "OK"] in rows
    assert ["laya-torch", "fastapi", "0.141.1", "0.141.1", "OK"] in rows
    assert ["decider", "decider-ai", "50d0be0d7cb43d2066965ce5fa7f3fe4e489a60f",
            "f8c933af759a00c1afd7ddb736ab44a7b47dac3e", "DIFF"] in rows
    assert code == 1


class _FakeHub:
    calls: list[tuple[str, object]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def model_info(self, repo_id: str, revision=None):
        _FakeHub.calls.append((repo_id, revision))
        return SimpleNamespace(sha=f"{abs(hash((repo_id, revision))) % 16**12:012x}".ljust(40, "0"))

    def list_repo_commits(self, repo_id: str):
        when = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
        return [SimpleNamespace(commit_id="bd792f44ec8e265be861bfcdf4e05967ffe0e858", created_at=when,
                                title="Upload latest checkpoint")]


def test_pin_audit_reports_hub_revisions_for_pending_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    audit = _pin_audit()
    config = ensure_workspace(tmp_path / "ws")
    installed = "1234567890abcdef1234567890abcdef12345678"
    monkeypatch.setattr(audit, "HfApi", _FakeHub)
    monkeypatch.setattr(audit, "installation_record", lambda config_path, model_id, backend: (
        {"artifacts": [{"source": "huggingface", "repo_id": "wfzyx/von", "resolved_revision": installed}]}
        if (model_id, backend) == ("von", "cuda") else None
    ))

    code = audit.main(["--config", str(config)])

    out = capsys.readouterr().out
    rows = {tuple(line.split("\t")[:2]): line.split("\t") for line in out.splitlines() if not line.startswith("#")}
    # D12c: Von is the only artifact still loaded without a catalog commit (HF_REVISION_PENDING).
    von = rows[("von::cuda", "wfzyx/von")]
    assert von[2:4] == ["-", installed] and re.match(r"^[0-9a-f]{40}$", von[4])
    assert rows[("von::mps", "wfzyx/von")][2:4] == ["-", "-"]
    assert ("kev-0.8b::gguf", "ggml-org/Kev-0.8B-GGUF") not in rows  # pinned to a commit since 0.5.9
    assert ("decider-2b::mps", "Mapika/decider-2b") not in rows
    assert ("laya-english::mps", "convaiinnovations/laya") not in rows  # already pinned to a commit
    assert ("wfzyx/von", None) in _FakeHub.calls and ("jaredpalmer/kev-0.8b", "v1.0") not in _FakeHub.calls
    assert "bd792f44ec8e265be861bfcdf4e05967ffe0e858\t2026-09-24T12:00:00+00:00\tUpload latest checkpoint" in out
    assert code == 0


def test_pin_audit_hub_errors_are_reported_and_fail_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    audit = _pin_audit()

    class BrokenHub(_FakeHub):
        def model_info(self, repo_id: str, revision=None):
            raise ConnectionError("hub unreachable")

    monkeypatch.setattr(audit, "HfApi", BrokenHub)
    config = ensure_workspace(tmp_path / "ws")

    code = audit.main(["--config", str(config)])

    out = capsys.readouterr().out
    assert "\tERROR ConnectionError: hub unreachable" in out
    assert code == 1


def test_pin_audit_lists_the_artifacts_of_the_catalog_ratchet() -> None:
    from test_catalog_pinning import _hf_artifacts

    audit = _pin_audit()
    for model in _catalog()["models"]:
        for backend, profile in model["backends"].items():
            assert audit.hf_artifacts(profile) == _hf_artifacts(profile), f"{model['id']}::{backend}"


def test_pin_audit_runs_as_one_command_line(tmp_path: Path) -> None:
    import deqio

    config = ensure_workspace(tmp_path / "ws")
    env = {**os.environ, "PYTHONPATH": str(Path(deqio.__file__).resolve().parents[1])}

    result = subprocess.run(
        [sys.executable, "-m", "deqio.pin_audit", "--offline", "--config", str(config)],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120,
    )

    assert result.returncode == 0, result.stderr
    for header in ("# runtime packages", "# nimble source", "# hub artifacts", "# result"):
        assert header in result.stdout


def test_clm_spec_uses_the_distribution_name_published_at_the_pinned_commit() -> None:
    """CLM publishes `contrastive-lm` since upstream fe9b720; uv refuses a direct URL whose metadata name differs."""
    profile = get_profile(_catalog(), "clm-8b", "cuda")
    assert profile["packages"] == [
        "contrastive-lm[serve,hf,vllm] @ git+https://github.com/Contrastive-LM/CLM.git@d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"
    ]
