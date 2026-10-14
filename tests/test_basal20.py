"""0.6.1: Basal 2.0 (MLX 8-bit and GGUF Q8_0) replaces the Basal 1.5 catalog family.

Runtime: the official ``rkinas/basal`` v2.0.0 tag (commit af1ce43a...) through ``basal-serve``;
weights: ``Remek/basal-2.0-{mini,4.5B,max}-MLX-8bit`` and ``-GGUF`` (Q8_0) at the commits audited on
2026-10-14.  Basal 2.0 ships no evidence head; ``facts``/``soam``/``multi``/``act`` stay native.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from deqio import model_manager as manager
from deqio.catalog import get_profile, load_catalog
from deqio.systemone_runtime import SystemOneCapabilityError, SystemOneRuntime

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "src" / "deqio" / "data" / "models.json"
BASAL_RUNTIME_COMMIT = "af1ce43a1770dd4fdfc1611e3fe720b93d42fc73"  # rkinas/basal tag v2.0.0
# tier -> (upstream repo stem, base commit, MLX-8bit commit, GGUF commit)
BASAL20 = {
    "mini": ("basal-2.0-mini", "cb81b32ce89b0b87e62fcdad28f0435e25c953f3", "f26d1e2df8002665a54583f74d954569fe99ec96", "594d115f7d5e9f7c7858b39d023c78574c2fe77a"),
    "main": ("basal-2.0-4.5B", "e5ba9149760d99b73b1963d8e0ccf948b58afacb", "77d3b723e5e2e7835dcea372045fe916097bb13c", "237294ca8fec0ae697f44ae0093995817c8ad578"),
    "max": ("basal-2.0-max", "2fffc5cc41e5cf7a2f752f38f17e7b4d9a4a8252", "8d4dc59784524004583d19bb2ac74f81f506b293", "8af9c1e127f0247cbdc326be04c2feeb273b8e26"),
}
MAX_INPUT_TOKENS = {"mini": 8192, "main": 32768, "max": 32768}


def _catalog() -> dict:
    return load_catalog(CATALOG)


def test_basal20_replaces_basal15_in_the_catalog() -> None:
    catalog = _catalog()
    ids = [model["id"] for model in catalog["models"]]
    assert not any(model_id.startswith("basal-1.5") for model_id in ids)
    assert [model_id for model_id in ids if model_id.startswith("basal-")] == ["basal-2.0-mini", "basal-2.0-main", "basal-2.0-max"]
    assert (ROOT / "models.json").read_bytes() == CATALOG.read_bytes()
    for source in (ROOT / "src" / "deqio" / "systemone_runtime.py", Path(manager.__file__)):
        text = source.read_text(encoding="utf-8")
        assert "basal1.5" not in text and "Basal 1.5" not in text, source.name


@pytest.mark.parametrize("tier", sorted(BASAL20))
def test_basal20_profiles_pin_official_runtime_and_audited_artifacts(tier: str) -> None:
    repo, base, mlx_sha, gguf_sha = BASAL20[tier]
    catalog = _catalog()
    model_id = f"basal-2.0-{tier}"
    entry = next(model for model in catalog["models"] if model["id"] == model_id)
    assert entry["engine"] == "basal" and entry["source"] == "https://github.com/rkinas/basal"
    assert set(entry["backends"]) == {"mlx", "gguf"}

    for backend in ("mlx", "gguf"):
        profile = get_profile(catalog, model_id, backend)
        assert profile["installer"] == "basal" and profile["basal_runtime_version"] == "2.0.0"
        assert profile["basal_package"] == f"basal[{backend}] @ https://github.com/rkinas/basal/archive/{BASAL_RUNTIME_COMMIT}.tar.gz"
        assert profile["basal_mode"] == backend and profile["basal_backend"] == backend and profile["basal_soam"] is True
        assert profile["family"] == "basal2.0" and profile["runtime"] == f"basal-2.0-{backend}" and profile["source"] == "official"
        assert profile["runtime_key"] == f"basal20-{tier}-{backend}" and profile["exclude_newer"] == "2026-10-10T00:00:00Z"
        assert profile["pin_hf_revisions_at_install"] is True and profile["python"] == "3.12"
        assert profile["max_input_tokens"] == MAX_INPUT_TOKENS[tier] and profile["default_max_input_tokens"] == 4096
        capabilities = profile["capabilities"]
        assert capabilities["evidence"] is False, "Basal 2.0 ships no evidence head"
        for feature in ("choice", "noul", "score", "multi", "act", "facts", "soam", "multi_question", "option_keys", "structured_state", "systemone"):
            assert capabilities[feature] is True, feature

    mlx = get_profile(catalog, model_id, "mlx")
    assert mlx["model"] == f"Remek/{repo}-MLX-8bit" and mlx["model_revision"] == mlx_sha
    assert mlx["download"] == {"type": "snapshot", "repo_id": f"Remek/{repo}-MLX-8bit", "revision": mlx_sha, "role": "basal-model"}
    assert mlx["precision"] == "mlx-8bit" and mlx["systems"] == ["Darwin"]
    assert mlx["quantization"] == {"kind": "mlx-affine", "bits": 8, "group_size": 64, "applied": "checkpoint", "source_model": f"Remek/{repo}"}

    gguf = get_profile(catalog, model_id, "gguf")
    assert gguf["model"] == f"Remek/{repo}" and gguf["model_revision"] == base
    assert gguf["downloads"] == [
        {"type": "snapshot", "repo_id": f"Remek/{repo}", "revision": base, "role": "basal-metadata", "ignore_patterns": ["*.safetensors", "*.bin", "*.pt"]},
        {"type": "file", "repo_id": f"Remek/{repo}-GGUF", "revision": gguf_sha, "local_dir": f"models/{repo}-GGUF", "role": "basal-gguf-q8_0", "filename_pattern": f"{repo}-Q8_0.gguf"},
    ]
    assert gguf["basal_gguf_dir"] == f"models/{repo}-GGUF" and gguf["basal_gguf_pattern"] == f"{repo}-Q8_0.gguf"
    assert gguf["precision"] == "q8_0" and set(gguf["systems"]) == {"Darwin", "Linux"}
    assert gguf["quantization"] == {"kind": "gguf", "quantization": "Q8_0", "bits": 8, "block_size": 32, "applied": "checkpoint", "source_model": f"Remek/{repo}"}


def test_basal_installer_accepts_only_the_v200_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    python = tmp_path / "env" / "bin" / "python"
    monkeypatch.setattr(manager, "_ensure_venv", lambda *args, **kwargs: python)
    monkeypatch.setattr(manager, "_run", lambda command, **kwargs: commands.append(list(command)))
    config = tmp_path / "config.json"
    catalog = _catalog()

    for backend in ("mlx", "gguf"):
        commands.clear()
        profile = get_profile(catalog, "basal-2.0-main", backend)
        env_dir = manager._install_runtime(config, {"runtime_dir": ".model-runtimes", "backend": backend}, profile, upgrade=False)
        assert env_dir == (tmp_path / ".model-runtimes" / f"basal20-main-{backend}").resolve()
        assert commands == [[
            "uv", "pip", "install", "--python", str(python), "--exclude-newer", "2026-10-10T00:00:00Z",
            f"basal[{backend}] @ https://github.com/rkinas/basal/archive/{BASAL_RUNTIME_COMMIT}.tar.gz",
        ]]

    stale = {**get_profile(catalog, "basal-2.0-mini", "mlx"), "basal_runtime_version": "1.5.0"}
    with pytest.raises(RuntimeError, match="2.0.0"):
        manager._install_runtime(config, {"runtime_dir": ".model-runtimes", "backend": "mlx"}, stale, upgrade=False)


def test_basal_runtime_metadata_requires_the_installed_v200(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import subprocess

    profile = get_profile(_catalog(), "basal-2.0-mini", "mlx")

    def fake_run(command, **kwargs):
        return SimpleNamespace(stdout='{"runtime_version": "2.0.0"}', returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    metadata = SystemOneRuntime._basal_runtime_metadata(profile, tmp_path / "python")
    assert metadata["runtime_version"] == "2.0.0" and metadata["basal_mode"] == "mlx" and metadata["official_runtime"] is True

    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: SimpleNamespace(stdout='{"runtime_version": "1.5.0"}', returncode=0))
    with pytest.raises(RuntimeError, match="expected 2.0.0"):
        SystemOneRuntime._basal_runtime_metadata(profile, tmp_path / "python")


def test_basal20_launchers_use_the_official_server_with_the_pinned_revision(tmp_path: Path) -> None:
    catalog = _catalog()
    env_dir = tmp_path / "runtime"
    executable = env_dir / "bin" / "basal-serve"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    python = env_dir / "bin" / "python"
    python.write_text("", encoding="utf-8")

    mlx = get_profile(catalog, "basal-2.0-max", "mlx")
    command = SystemOneRuntime._command(
        SimpleNamespace(engine="basal", backend="mlx", model_revision=mlx["model_revision"], config_path=tmp_path / "config.json"),
        mlx, env_dir, python, 9441, {},
    )
    assert command == [
        str(executable), "--model", "Remek/basal-2.0-max-MLX-8bit", "--revision", BASAL20["max"][2],
        "--mode", "mlx", "--soam", "on", "--context-overflow", "refuse", "--port", "9441",
    ]

    gguf = get_profile(catalog, "basal-2.0-mini", "gguf")
    gguf_dir = tmp_path / "models" / "basal-2.0-mini-GGUF"
    gguf_dir.mkdir(parents=True)
    weights = gguf_dir / "basal-2.0-mini-Q8_0.gguf"
    weights.write_bytes(b"gguf")
    (gguf_dir / "basal-2.0-mini-Q5_K_M.gguf").write_bytes(b"gguf")
    command = SystemOneRuntime._command(
        SimpleNamespace(engine="basal", backend="gguf", model_revision=gguf["model_revision"], config_path=tmp_path / "config.json"),
        gguf, env_dir, python, 9442, {},
    )
    assert command == [
        str(executable), "--model", "Remek/basal-2.0-mini", "--revision", BASAL20["mini"][1],
        "--mode", "gguf", "--gguf", str(weights.resolve()), "--soam", "on", "--context-overflow", "refuse", "--port", "9442",
    ]


def test_basal20_native_system_one_is_capability_aware_and_refuses_evidence() -> None:
    catalog = _catalog()
    sent: list[tuple] = []
    runtime = object.__new__(SystemOneRuntime)
    runtime.engine = "basal"
    runtime.settings = SimpleNamespace(backend="mlx", model="Remek/basal-2.0-mini-MLX-8bit")
    runtime.profile = get_profile(catalog, "basal-2.0-mini", "mlx")
    runtime._request = lambda state, questions, execution_mode=None, request_options=None: sent.append((state, questions, request_options)) or (
        {"state": state, "questions": questions}, {"answers": {qid: {"type": q["type"]} for qid, q in questions.items()}, "usage": {"input_tokens": 12}}, 3.0
    )
    response, timing = runtime.system_one(
        "state", {"a": {"type": "choice", "instructions": "x", "criteria": {"k": "K"}}, "b": {"type": "act", "instructions": "y", "criteria": {"k": "K"}, "costs": {"wrong": 1, "defer": 0.2}}},
        {"facts": "auto"},
    )
    assert set(response["answers"]) == {"a", "b"} and sent[0][2] == {"facts": "auto"} and timing["engine"] == "basal"
    with pytest.raises(SystemOneCapabilityError, match="[Ee]vidence"):
        runtime.system_one("state", {"a": {"type": "choice", "instructions": "x", "criteria": {"k": "K"}, "evidence": True}})

    legacy = object.__new__(SystemOneRuntime)
    legacy.engine = "basal"
    legacy.settings = SimpleNamespace(backend="mlx")
    legacy.profile = {"family": "basal1.5", "capabilities": {"choice": True, "systemone": True}}
    with pytest.raises(RuntimeError, match="Basal 2.0"):
        legacy.system_one("state", {"a": {"type": "choice", "instructions": "x", "criteria": {"k": "K"}}})

# --- generic Basal installer/launcher/pin logic, carried over from the 1.5 tests with synthetic 2.0 profiles
# (the catalog ships MLX and GGUF profiles only; the mps/cuda branches of the installer and launcher stay testable)

def test_basal20_pin_resolves_exact_q8_artifact_without_filename_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    import deqio.model_manager as manager  # noqa: F811 (module-level import exists)

    class FakeApi:
        def model_info(self, repo_id, revision=None):
            suffix = "1" if repo_id.endswith("4.5B") else "2"
            return SimpleNamespace(sha=suffix * 40)

        def list_repo_files(self, repo_id, revision=None):
            assert revision == "2" * 40
            return ["README.md", "official-basal-main-Q8_0.gguf", "official-basal-main-Q4_K_M.gguf"]

    monkeypatch.setattr(manager, "HfApi", lambda: FakeApi())
    profile = {
        "pin_hf_revisions_at_install": True,
        "model": "Remek/basal-2.0-4.5B",
        "model_revision": "main",
        "downloads": [
            {
                "type": "snapshot",
                "repo_id": "Remek/basal-2.0-4.5B",
                "revision": "main",
                "role": "basal-metadata",
            },
            {
                "type": "file",
                "repo_id": "Remek/basal-2.0-4.5B-GGUF",
                "revision": "main",
                "role": "basal-gguf-q8_0",
                "filename_pattern": "*Q8_0.gguf",
                "local_dir": "models/basal-2.0-4.5B-GGUF",
            },
        ],
    }
    pinned = manager._pin_profile_hf_revisions(profile)
    assert pinned["model_revision"] == "1" * 40
    assert pinned["downloads"][0]["revision"] == "1" * 40
    assert pinned["downloads"][1]["revision"] == "2" * 40
    assert pinned["downloads"][1]["filename"] == "official-basal-main-Q8_0.gguf"
    assert "filename" not in profile["downloads"][1]


def test_basal20_profile_pin_prefers_selected_exact_revision_before_install_record() -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import _profile_with_installed_pin

    revision = "c" * 40
    settings = SimpleNamespace(
        model_revision=revision,
        config_path=Path("/tmp/config.json"),
        model_id="basal-2.0-main",
        backend="mlx",
    )
    profile = {
        "model": "Remek/basal-2.0-4.5B-MLX-8bit",
        "model_revision": "main",
        "pin_hf_revisions_at_install": True,
    }
    pinned = _profile_with_installed_pin(settings, profile)
    assert pinned["model_revision"] == revision
    assert profile["model_revision"] == "main"


def test_basal20_installer_uses_official_backend_specific_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.model_manager as manager  # noqa: F811 (module-level import exists)

    commands: list[list[str]] = []
    python = tmp_path / ".model-runtimes" / "basal20" / "bin" / "python"
    monkeypatch.setattr(manager, "_ensure_venv", lambda *args, **kwargs: python)
    monkeypatch.setattr(manager, "_run", lambda command, **kwargs: commands.append(list(command)))

    for backend, package in (
        ("mlx", "basal[mlx] @ https://github.com/rkinas/basal/archive/af1ce43a1770dd4fdfc1611e3fe720b93d42fc73.tar.gz"),
        ("mps", "basal @ https://github.com/rkinas/basal/archive/af1ce43a1770dd4fdfc1611e3fe720b93d42fc73.tar.gz"),
        ("cuda", "basal @ https://github.com/rkinas/basal/archive/af1ce43a1770dd4fdfc1611e3fe720b93d42fc73.tar.gz"),
    ):
        commands.clear()
        manager._install_basal(
            tmp_path / "config.json",
            {"backend": backend, "runtime_dir": ".model-runtimes"},
            {
                "runtime_key": "basal20",
                "exclude_newer": "2026-10-10T00:00:00Z",
                "python": "3.12",
                "basal_backend": backend,
                "basal_runtime_version": "2.0.0",
                "basal_package": package,
            },
            upgrade=False,
        )
        if backend == "cuda":
            assert len(commands) == 2
            assert "torch==2.11.0" in commands[0]
            assert commands[1][-1] == package
        else:
            assert len(commands) == 1
            assert commands[0][-1] == package


def test_basal20_cuda_installer_leaves_torch_resolution_to_official_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.model_manager as manager  # noqa: F811 (module-level import exists)

    commands: list[list[str]] = []
    python = tmp_path / ".model-runtimes" / "basal20-main-cuda" / "bin" / "python"
    monkeypatch.setattr(manager, "_ensure_venv", lambda *args, **kwargs: python)
    monkeypatch.setattr(manager, "_run", lambda command, **kwargs: commands.append(list(command)))

    manager._install_basal(
        tmp_path / "config.json",
        {"backend": "cuda", "runtime_dir": ".model-runtimes"},
        {
            "runtime_key": "basal20-main-cuda",
            "exclude_newer": "2026-10-10T00:00:00Z",
            "python": "3.12",
            "basal_backend": "cuda",
            "basal_runtime_version": "2.0.0",
            "basal_mode": "fast-nocompile",
            "basal_package": "basal @ https://github.com/rkinas/basal/archive/af1ce43a1770dd4fdfc1611e3fe720b93d42fc73.tar.gz",
        },
        upgrade=False,
    )
    assert len(commands) == 2
    assert "torch==2.11.0" in commands[0]
    assert "https://download.pytorch.org/whl/cu128" in commands[0]
    assert commands[1][-1].endswith("basal/archive/af1ce43a1770dd4fdfc1611e3fe720b93d42fc73.tar.gz")


def test_basal20_main_cuda_launcher_uses_official_fast_nocompile_mode(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneRuntime

    env_dir = tmp_path / "runtime"
    executable = env_dir / "bin" / "basal-serve"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    python = env_dir / "bin" / "python"
    python.write_text("", encoding="utf-8")

    command = SystemOneRuntime._command(
        SimpleNamespace(engine="basal", backend="cuda", model_revision="main", config_path=tmp_path / "config.json"),
        {"model": "Remek/basal-2.0-4.5B", "model_revision": "main", "basal_mode": "fast-nocompile", "basal_soam": True},
        env_dir, python, 9033, {},
    )
    assert command == [
        str(executable), "--model", "Remek/basal-2.0-4.5B",
        "--mode", "fast-nocompile", "--soam", "on", "--port", "9033",
    ]


def test_basal20_launchers_use_official_server_for_mlx_and_mps_modes(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneRuntime

    env_dir = tmp_path / "runtime"
    executable = env_dir / "bin" / "basal-serve"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    python = env_dir / "bin" / "python"
    python.write_text("", encoding="utf-8")

    for backend, model, mode in (
        ("mlx", "Remek/basal-2.0-mini-MLX-8bit", "mlx"),
        ("mps", "Remek/basal-2.0-mini", "mps"),
    ):
        command = SystemOneRuntime._command(
            SimpleNamespace(engine="basal", backend=backend, model_revision="main", config_path=tmp_path / "config.json"),
            {"model": model, "model_revision": "main", "basal_mode": mode, "basal_soam": True},
            env_dir, python, 9020, {},
        )
        assert command == [
            str(executable), "--model", model, "--mode", mode, "--soam", "on", "--port", "9020"
        ]


# --- 0.6.1 fix 1: memory guardrails as in 1.5, explicit context overflow ------------------------------
# tier -> backend -> (min_memory_gib, recommended_memory_gib): the Basal 1.5 values of 0.5.9 for the same
# Bielik-v3 weights and the same 8-bit / Q8_0 quantizations (0.6.1 shipped higher numbers; max became unavailable
# on a 16 GiB Mac that ran basal-1.5-max).
BASAL15_MEMORY = {
    "mini": {"mlx": (1.8, 4), "gguf": (1.7, 4)},
    "main": {"mlx": (5.2, 8), "gguf": (5.1, 8)},
    "max": {"mlx": (12.0, 20), "gguf": (12.0, 20)},
}


@pytest.mark.parametrize("tier", sorted(BASAL15_MEMORY))
def test_basal20_memory_guardrails_match_basal15_for_the_same_weights(tier: str) -> None:
    catalog = _catalog()
    for backend, (minimum, recommended) in BASAL15_MEMORY[tier].items():
        profile = get_profile(catalog, f"basal-2.0-{tier}", backend)
        assert profile["min_memory_gib"] == pytest.approx(minimum), (tier, backend, profile["min_memory_gib"])
        assert profile["recommended_memory_gib"] == pytest.approx(recommended), (tier, backend)


def test_basal20_context_is_the_models_and_overflow_is_refused_explicitly(tmp_path: Path) -> None:
    """basal-serve 2.0 takes the context from the model's config.json (mini 8192, 4.5B/max 32768) and by default
    truncates the middle of a longer state (`--context-overflow truncate`, marked only in usage.truncated).
    Deqio never truncates silently: every Basal 2.0 profile declares `basal_context_overflow: refuse` and the
    launcher passes it, so an over-long prompt is a 422 from the sidecar and a 422 for the client."""
    catalog = _catalog()
    for tier in ("mini", "main", "max"):
        for backend in ("mlx", "gguf"):
            profile = get_profile(catalog, f"basal-2.0-{tier}", backend)
            assert profile["basal_context_overflow"] == "refuse", (tier, backend)
            assert profile["max_input_tokens"] == MAX_INPUT_TOKENS[tier]

    env_dir = tmp_path / "runtime"
    executable = env_dir / "bin" / "basal-serve"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    python = env_dir / "bin" / "python"
    python.write_text("", encoding="utf-8")
    profile = get_profile(catalog, "basal-2.0-mini", "mlx")
    command = SystemOneRuntime._command(
        SimpleNamespace(engine="basal", backend="mlx", model_revision=profile["model_revision"], config_path=tmp_path / "config.json"),
        profile, env_dir, python, 9450, {},
    )
    assert command[command.index("--context-overflow") + 1] == "refuse"
    assert command.index("--context-overflow") < command.index("--port")
    with pytest.raises(RuntimeError, match="basal_context_overflow"):
        SystemOneRuntime._command(
            SimpleNamespace(engine="basal", backend="mlx", model_revision=profile["model_revision"], config_path=tmp_path / "config.json"),
            {**profile, "basal_context_overflow": "drop"}, env_dir, python, 9450, {},
        )
