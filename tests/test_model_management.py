"""Model management state machine (D9: audit items B1, B2 and B3).

Every test runs ``deqio models`` code against a disposable workspace. Package
installs, weight downloads and readiness probes are replaced at the ``_run``,
``_prefetch_declared_weights`` and ``_verify_model_ready`` seams, so no test
touches the network or builds anything.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from deqio import installations, model_manager
from deqio.workspace import ensure_workspace

TARGET = ("kev-4b", "mps")
SIBLING = ("kev-0.8b", "mps")


class _AnyRevision(set):
    def __contains__(self, item: object) -> bool:
        return True


class _EveryRepoCached(dict):
    """Stand-in for the Hugging Face cache scan: every repo and revision is present."""

    def get(self, key, default=None):
        return _AnyRevision()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = ensure_workspace(tmp_path / "ws")
    monkeypatch.setattr(installations, "_cached_hf_state", lambda: _EveryRepoCached())
    monkeypatch.setattr(model_manager, "_pin_profile_hf_revisions", lambda profile: profile)
    monkeypatch.setattr(model_manager, "_artifact_attestation", lambda *args, **kwargs: [])
    monkeypatch.setattr(model_manager, "_prefetch_declared_weights", lambda *args, **kwargs: "weights ready")
    return config


def _catalog(config: Path) -> dict:
    return model_manager.load_catalog(
        model_manager._catalog_path(config, model_manager._read_config(config))
    )


def _shared_runtime(config: Path) -> Path:
    catalog = _catalog(config)
    runtime_keys = {model_manager.get_profile(catalog, *key)["runtime_key"] for key in (TARGET, SIBLING)}
    assert runtime_keys == {"kev"}, "these tests need two profiles that share one runtime"
    env_dir = config.parent / ".model-runtimes" / "kev"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").write_text("", encoding="utf-8")
    (env_dir / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    return env_dir


def _register_verified(config: Path, *keys: tuple[str, str]) -> dict:
    for model_id, backend in keys:
        installations.mark_installed(
            config, model_id, backend, verified=True, source="install", artifacts=[], max_input_tokens=4096
        )
    return installations.load_registry(config)["profiles"]


def _rows(config: Path) -> dict:
    rows = installations.installed_profiles(
        config_path=config, config_data=model_manager._read_config(config), catalog=_catalog(config)
    )
    return {(row["model_id"], row["backend"]): row for row in rows}


def _update_target(config: Path) -> int:
    return model_manager.main([
        "--config", str(config), "update", TARGET[0], "--backend", TARGET[1],
        "--force", "--max-input-tokens", "4096",
    ])


def _recording_run(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    commands: list[list[str]] = []
    monkeypatch.setattr(model_manager, "_run", lambda command, *, env=None: commands.append(list(command)))
    return commands


def _venv_commands(commands: list[list[str]]) -> list[list[str]]:
    return [command for command in commands if command[:2] == ["uv", "venv"]]


# --- B1: updating a shared runtime is a registry state transition -------------


def test_failed_runtime_upgrade_unverifies_every_profile_sharing_the_runtime(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _shared_runtime(workspace)
    _register_verified(workspace, SIBLING, TARGET)
    before = _rows(workspace)
    assert before[TARGET]["verified"] and before[SIBLING]["verified"]

    def failing_upgrade(command: list[str], *, env=None) -> None:
        if command[:3] == ["uv", "pip", "install"] and "--upgrade" in command:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(model_manager, "_run", failing_upgrade)

    assert _update_target(workspace) == 2
    assert capsys.readouterr().err.startswith("error:")
    after = _rows(workspace)
    for key in (TARGET, SIBLING):
        assert after[key]["registered"] is True, key
        assert after[key]["installed"] is True, key
        assert after[key]["verified"] is False, key
        assert after[key]["status"] == "installed", key


def test_failed_readiness_probe_after_upgrade_keeps_the_shared_runtime_unverified(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _shared_runtime(workspace)
    _register_verified(workspace, SIBLING, TARGET)
    _recording_run(monkeypatch)

    def failing_probe(*args, **kwargs) -> None:
        raise RuntimeError("typed-decision readiness probe failed")

    monkeypatch.setattr(model_manager, "_verify_model_ready", failing_probe)

    assert _update_target(workspace) == 2
    after = _rows(workspace)
    assert after[TARGET]["verified"] is False
    assert after[SIBLING]["verified"] is False


def test_successful_update_after_a_failure_restores_the_shared_runtime_profiles(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _shared_runtime(workspace)
    original = _register_verified(workspace, SIBLING, TARGET)
    _recording_run(monkeypatch)
    probe_failures = ["typed-decision readiness probe failed"]

    def probe(*args, **kwargs) -> None:
        if probe_failures:
            raise RuntimeError(probe_failures.pop())

    monkeypatch.setattr(model_manager, "_verify_model_ready", probe)

    assert _update_target(workspace) == 2
    assert _update_target(workspace) == 0

    registry = installations.load_registry(workspace)["profiles"]
    sibling_key = installations.profile_key(*SIBLING)
    target_key = installations.profile_key(*TARGET)
    # The sibling's weights were not touched and the shared runtime passed a
    # real readiness probe on the target model: its record is restored as-is.
    assert registry[sibling_key] == original[sibling_key]
    assert set(registry[target_key]) == set(original[target_key])
    assert registry[target_key]["source"] == "update"
    assert registry[target_key]["verified_at"] >= original[target_key]["verified_at"]
    after = _rows(workspace)
    assert after[TARGET]["verified"] is True
    assert after[SIBLING]["verified"] is True


# --- B2: broken runtime directories are repaired, never deleted on a guess ----


def test_ensure_venv_recreates_a_venv_whose_interpreter_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = _recording_run(monkeypatch)
    env_dir = tmp_path / ".model-runtimes" / "kev"
    (env_dir / "lib").mkdir(parents=True)
    # An interrupted `uv venv`, or an uninstalled uv-managed Python, leaves the
    # venv metadata behind without a usable interpreter; `uv venv` then refuses
    # the existing directory unless it is told to clear it.
    (env_dir / "pyvenv.cfg").write_text("home = /gone/python/bin\n", encoding="utf-8")

    model_manager._ensure_venv(env_dir, "3.12")

    [venv] = _venv_commands(commands)
    assert venv[:3] == ["uv", "venv", str(env_dir)]
    assert "--clear" in venv


def test_ensure_venv_creates_missing_and_empty_directories_without_clearing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = _recording_run(monkeypatch)
    empty = tmp_path / "empty"
    empty.mkdir()

    model_manager._ensure_venv(tmp_path / "missing", "3.12")
    model_manager._ensure_venv(empty, "3.12")

    venvs = _venv_commands(commands)
    assert len(venvs) == 2
    assert all("--clear" not in command for command in venvs)


def test_ensure_venv_refuses_a_non_venv_directory_instead_of_deleting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = _recording_run(monkeypatch)
    env_dir = tmp_path / ".model-runtimes" / "kev"
    env_dir.mkdir(parents=True)
    (env_dir / "notes.txt").write_text("not a virtual environment", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not a virtual environment"):
        model_manager._ensure_venv(env_dir, "3.12")

    assert _venv_commands(commands) == []
    assert (env_dir / "notes.txt").is_file()


def test_default_runtime_installer_uses_the_same_venv_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = _recording_run(monkeypatch)
    config = tmp_path / "config.json"
    env_dir = tmp_path / ".model-runtimes" / "demo"
    env_dir.mkdir(parents=True)
    (env_dir / "pyvenv.cfg").write_text("home = /gone/python/bin\n", encoding="utf-8")

    model_manager._install_runtime(
        config,
        {"runtime_dir": ".model-runtimes"},
        {"runtime_key": "demo", "packages": ["demo-engine==1.0"], "python": "3.12"},
        upgrade=False,
    )

    [venv] = _venv_commands(commands)
    assert "--clear" in venv
    assert commands[-1][:3] == ["uv", "pip", "install"]


def test_llama_cpp_source_without_git_metadata_is_recloned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = _recording_run(monkeypatch)
    config = tmp_path / "config.json"
    env_dir = tmp_path / ".model-runtimes" / "llama-cpp-systemone"
    (env_dir / "bin").mkdir(parents=True)
    for name in ("python", "cmake", "ninja"):
        (env_dir / "bin" / name).write_text("", encoding="utf-8")
    (env_dir / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    source = env_dir / "llama.cpp"
    source.mkdir()
    (source / "CMakeLists.txt").write_text("half-copied checkout", encoding="utf-8")

    # Recorded commands build nothing, so the installer stops at the missing
    # binary; what matters here is how it treated the source directory.
    with pytest.raises(RuntimeError):
        model_manager._install_llama_cpp_runtime(
            config,
            {"runtime_dir": ".model-runtimes"},
            {"runtime_key": "llama-cpp-systemone", "python": "3.12", "llama_cpp_revision": "a" * 40},
            upgrade=False,
        )

    git = [command for command in commands if command[:1] == ["git"]]
    assert git, commands
    # Without its own .git, `git -C <source> fetch` would run against any
    # enclosing repository (for example a Deqio source checkout).
    assert git[0][:2] == ["git", "clone"] and git[0][-1] == str(source)
    assert not (source / "CMakeLists.txt").exists()


def test_nimble_source_without_git_metadata_is_recloned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = _recording_run(monkeypatch)
    source = tmp_path / "nimble-src"
    source.mkdir()
    (source / "README.md").write_text("partial", encoding="utf-8")

    pinned = "dcfdbd9a64f0d869f658d7a72f1beaee32737773"
    # The recorded clone lands on the pinned commit (D12), so no fetch/checkout follows.
    monkeypatch.setattr(model_manager, "_git_head", lambda source_dir: pinned)
    model_manager._checkout_nimble(tmp_path, source_key="nimble-src", revision=pinned)

    assert [command[:2] for command in commands] == [["git", "clone"]]
    assert commands[0][-1] == str(source)
    assert not (source / "README.md").exists()


# --- B3: delete unregisters only after the artifacts are gone -----------------


def test_delete_keeps_the_registration_when_cleanup_fails(
    workspace: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    external = tmp_path / "external-disk" / "kev"
    (external / "bin").mkdir(parents=True)
    (external / "bin" / "python").write_text("", encoding="utf-8")
    (external / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    runtime_root = workspace.parent / ".model-runtimes"
    runtime_root.mkdir()
    # shutil.rmtree refuses symlinks: a real OSError from the real cleanup path.
    (runtime_root / "kev").symlink_to(external, target_is_directory=True)
    _register_verified(workspace, SIBLING)

    result = model_manager.main([
        "--config", str(workspace), "delete", SIBLING[0], "--backend", SIBLING[1], "--yes",
    ])

    assert result == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "still registered" in err
    assert installations.profile_key(*SIBLING) in installations.load_registry(workspace)["profiles"]
    assert (external / "bin" / "python").is_file()


def test_delete_drops_the_registration_only_after_cleanup(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_dir = _shared_runtime(workspace)
    _register_verified(workspace, SIBLING)
    runtime_present_at_unmark: list[bool] = []
    real_unmark = model_manager.unmark_installed

    def spy(*args, **kwargs):
        runtime_present_at_unmark.append(env_dir.exists())
        return real_unmark(*args, **kwargs)

    monkeypatch.setattr(model_manager, "unmark_installed", spy)

    assert model_manager.main([
        "--config", str(workspace), "delete", SIBLING[0], "--backend", SIBLING[1], "--yes",
    ]) == 0
    assert runtime_present_at_unmark == [False]
    assert installations.profile_key(*SIBLING) not in installations.load_registry(workspace)["profiles"]


def test_model_management_reports_os_errors_without_a_traceback(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing_uv(command: list[str], *, env=None) -> None:
        raise FileNotFoundError(2, "No such file or directory", command[0])

    monkeypatch.setattr(model_manager, "_run", missing_uv)

    result = model_manager.main([
        "--config", str(workspace), "install", SIBLING[0], "--backend", SIBLING[1],
        "--force", "--max-input-tokens", "4096",
    ])

    assert result == 2
    assert capsys.readouterr().err.startswith("error:")
