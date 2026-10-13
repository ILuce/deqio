"""D12c: Hugging Face artifacts pinned to the commits of the owner's ``pin_audit`` (0.5.8, hub=online).

Every entry below comes from ``pin_audit.txt`` (``python -m deqio.pin_audit``, 2026-10-12): the ``hub_revision``
column resolved for the catalog's own reference (``v1.0`` for Kev, ``main`` elsewhere). A revision is pinned only
where the launcher makes the engine load it (AGENTS.md); the launcher changes are tested here as well.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from deqio import clm_sidecar, model_manager
from deqio import systemone_runtime as runtime_module
from deqio.catalog import get_profile, load_catalog
from deqio.systemone_runtime import SystemOneRuntime
from deqio.workspace import ensure_workspace

SHA40 = re.compile(r"^[0-9a-f]{40}$")
ROOT = Path(__file__).resolve().parents[1]
# [packaged catalog, repository catalog] -- the same two files tests/test_catalog_pinning.py guards.
CATALOG_PATHS = (ROOT / "src" / "deqio" / "data" / "models.json", ROOT / "models.json")

KEV = {
    "0.8b": "bf75a6a8848ea6960ff2ed108d9ed44c2941174f",
    "4b": "6cfce5c2fa4b4bd64026336ab649c5ca78857d52",
    "9b": "db029f08b290afd9fee4aa4bbcd9ae48602d1eb0",
}
BASAL = {
    # tier -> (model repo suffix, MLX commit, base commit, GGUF commit)
    "mini": ("mini", "6ea29c486e39e1a217932ab56b222c0d1bc4ba82", "1978d0705ce09cd2d7d8d3e87b304468121e7255",
             "9671696b53ccf4adbb30adc24f4ae8372e02dffc"),
    "main": ("4.5B", "79f5a7cbbd028758c506f9aca5fec1ef431ee893", "784a683bfadcc8865238fc0fc74a83b4000269bc",
             "6943511d883b0cd7029d94abd0e837482d7a7f86"),
    "max": ("max", "fafde24a328b3ea092b12ee12c17024b6a467468", "be1b5ee7e7a9755a931262fa7fab4f59be0fd03c",
            "88dafb2b58fe82ec3751c81c2431acf442e9038c"),
}
DECIDER = {
    "0.8b": "a0a01d6f8135298f400a8c856b355793012ae971",
    "2b": "ff99089b6810691bd79134ec6febb5d9afa6e619",
    "4b": "99d7fd5bfcc4d11ba37239dafe0ed653803819d3",
}
JEVK5_GGUF = "ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a"
CLM_ENCODER = "b968826d9c46dd6066d109eabc6255188de91218"
CLM_CHECKPOINT = "e939398d4556fcd9400c76fa8c5a513202f42b0a"

# (model_id, backend) -> {repo_id: commit}; every Hub artifact of the profile must carry exactly these commits.
PINS: dict[tuple[str, str], dict[str, str]] = {
    **{(f"kev-{size}", backend): {f"jaredpalmer/kev-{size}": sha}
       for size, sha in KEV.items() for backend in ("mlx", "cuda", "mps")},
    ("kev-27b", "cuda"): {"jaredpalmer/kev-27b": "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf"},
    ("kev-0.8b", "gguf"): {"ggml-org/Kev-0.8B-GGUF": "e551e319d483ff57e1ff208924b349d397cffc1c"},
    ("kev-4b", "gguf"): {"ggml-org/Kev-4B-GGUF": "d924f2e2c3872da8b8aaf3eb4453b4126deceb79"},
    ("kev-9b", "gguf"): {"ggml-org/Kev-9B-GGUF": "ec2bbfe6620218aee2e01cc93bb78dee2a96ed58"},
    ("jevk5-4b", "cuda"): {"alibiserikbay/JevK5": "c4f7fdb3aeab5582336406e78d3bef11bf98833d"},
    ("jevk5-9b", "cuda"): {"alibiserikbay/JevK5-9B": "d6521a18a86999190e9d775c915af3d6d6772fc4"},
    ("jevk5-4b", "gguf"): {"alibiserikbay/JevK5-GGUF": JEVK5_GGUF},
    ("jevk5-9b", "gguf"): {"alibiserikbay/JevK5-GGUF": JEVK5_GGUF},
    ("open-jev-27b-v1.1", "cuda"): {
        "ZefanCai/Open-Jev-27B-v1.1": "28cf73067d5b337860bbef3c85b8b82ba8730956",
        "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    },
    ("clm-8b", "cuda"): {"Qwen/Qwen3-8B": CLM_ENCODER, "Contrastive-LM/CLM-v0.1-8B": CLM_CHECKPOINT},
    **{(f"basal-1.5-{tier}", "mlx"): {f"Remek/basal-1.5-{repo}-MLX-8bit": mlx}
       for tier, (repo, mlx, _, _) in BASAL.items()},
    **{(f"basal-1.5-{tier}", backend): {f"Remek/basal-1.5-{repo}": base}
       for tier, (repo, _, base, _) in BASAL.items() for backend in ("mps", "cuda")},
    **{(f"basal-1.5-{tier}", "gguf"): {f"Remek/basal-1.5-{repo}": base, f"Remek/basal-1.5-{repo}-GGUF": gguf}
       for tier, (repo, _, base, gguf) in BASAL.items()},
    ("clef-flash", "gguf"): {"ggml-org/Clef-Flash-GGUF": "6b4fe4c59098479652d94f0fe11a40facbebd1ec"},
    ("clef", "gguf"): {"ggml-org/Clef-GGUF": "d2aa75cb425ae922b5c3c15ef3157d417b7f7641"},
    **{(f"decider-{size}", backend): {f"Mapika/decider-{size}": sha}
       for size, sha in DECIDER.items() for backend in ("cuda", "mps")},
    ("decider-2b", "gguf"): {"Mapika/decider-2b-GGUF": "ff2e5e687327eda9ac34e9a3ca84d3f400672c87"},
    ("decider-4b", "gguf"): {"Mapika/decider-4b-GGUF": "b79f09d9ba7837f1b744295ea267b55d08e958ec"},
    ("laya-english", "gguf"): {"ggml-org/Laya-GGUF": "22265007700297ba9e128297e82540cf28c5d7d4"},
}


def _catalog() -> dict:
    return load_catalog(CATALOG_PATHS[1])


def _artifacts(profile: dict) -> dict[str, object]:
    """repo_id -> revision for every Hub artifact a profile declares (downloads first, else the model repo)."""
    rows: dict[str, object] = {}
    downloads = model_manager._declared_downloads(profile)
    for item in downloads:
        rows[str(item["repo_id"])] = item.get("revision") or profile.get("model_revision")
    model = str(profile.get("model", ""))
    if model_manager._looks_like_hf_repo(model.partition("@")[0]):
        rows.setdefault(model.partition("@")[0], profile.get("model_revision"))
    return rows


@pytest.mark.parametrize("model_id,backend", sorted(PINS), ids=[f"{m}::{b}" for m, b in sorted(PINS)])
def test_catalog_pins_the_audited_hub_commits(model_id: str, backend: str) -> None:
    profile = get_profile(_catalog(), model_id, backend)
    expected = PINS[(model_id, backend)]
    assert _artifacts(profile) == expected
    assert profile.get("model_revision") in expected.values()
    if model_id.startswith("kev-") and backend != "gguf":
        # kev.serve --run REPO@REV: kev.checkpoint.resolve_run partitions on "@" and passes the revision
        # to snapshot_download (HUB_ID accepts [\w.-]+ after "@"); the catalog must name the commit there.
        repo, sha = next(iter(expected.items()))
        assert profile["model"] == f"{repo}@{sha}"


def test_pinned_hf_revisions_are_commits_in_both_catalog_copies() -> None:
    """Guard: the packaged catalog is the workspace catalog (D12c edits both files)."""
    assert CATALOG_PATHS[0].read_bytes() == CATALOG_PATHS[1].read_bytes()
    for (model_id, backend), expected in PINS.items():
        for sha in expected.values():
            assert SHA40.match(sha), f"{model_id}::{backend}: {sha}"


def _launcher_settings(config_path: Path, **overrides) -> SimpleNamespace:
    values = {
        "engine": "decider", "backend": "cuda", "model_id": "decider-2b",
        "model": "Mapika/decider-2b", "model_revision": DECIDER["2b"],
        "max_tokens": 4096, "mlx_cache_mib": 256, "torch_dtype": "bfloat16", "hf_offline_runtime": True,
        "config_path": config_path, "runtime_dir": config_path.parent / ".model-runtimes",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _command(settings: SimpleNamespace, profile: dict, tmp_path: Path, env: dict[str, str]) -> list[str]:
    env_dir = tmp_path / "env"
    python = env_dir / "bin" / "python"
    return SystemOneRuntime._command(settings, profile, env_dir, python, 9000, env)


def _fake_snapshot(tmp_path: Path, calls: list[dict]):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(exist_ok=True)

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    return snapshot, snapshot_download


def test_decider_launcher_loads_the_pinned_snapshot_by_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DECIDER_MODEL takes a repo id or a path; only a path can name the pinned commit, so the launcher
    resolves the cached snapshot of the catalog commit (offline) and refuses to start without it."""
    config = ensure_workspace(tmp_path / "ws")
    profile = get_profile(_catalog(), "decider-2b", "cuda")
    calls: list[dict] = []
    snapshot, fake = _fake_snapshot(tmp_path, calls)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake)
    env: dict[str, str] = {}
    _command(_launcher_settings(config), profile, tmp_path, env)
    assert env["DECIDER_MODEL"] == str(snapshot)
    assert calls == [{"repo_id": "Mapika/decider-2b", "revision": DECIDER["2b"], "local_files_only": True}]


def test_decider_launcher_refuses_an_uncached_pinned_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ensure_workspace(tmp_path / "ws")
    profile = get_profile(_catalog(), "decider-2b", "cuda")

    def missing(**kwargs):
        raise FileNotFoundError("not in cache")

    monkeypatch.setattr("huggingface_hub.snapshot_download", missing)
    env: dict[str, str] = {}
    with pytest.raises(RuntimeError, match=r"deqio models (setup|update)"):
        _command(_launcher_settings(config), profile, tmp_path, env)
    assert "DECIDER_MODEL" not in env, "the launcher must not fall back to the floating repo id"


def test_jevk5_launcher_loads_the_pinned_snapshot_by_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """jevk5.server --model is passed straight to transformers from_pretrained (local folder or repo id)."""
    config = ensure_workspace(tmp_path / "ws")
    profile = get_profile(_catalog(), "jevk5-4b", "cuda")
    calls: list[dict] = []
    snapshot, fake = _fake_snapshot(tmp_path, calls)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake)
    settings = _launcher_settings(
        config, engine="jevk5", model_id="jevk5-4b", model="alibiserikbay/JevK5", model_revision=PINS[("jevk5-4b", "cuda")]["alibiserikbay/JevK5"]
    )
    command = _command(settings, profile, tmp_path, {})
    assert command[command.index("--model") + 1] == str(snapshot)
    assert calls == [{"repo_id": "alibiserikbay/JevK5", "revision": PINS[("jevk5-4b", "cuda")]["alibiserikbay/JevK5"], "local_files_only": True}]


def test_clm_launcher_passes_the_pinned_encoder_revision(tmp_path: Path) -> None:
    config = ensure_workspace(tmp_path / "ws")
    profile = dict(get_profile(_catalog(), "clm-8b", "cuda"))
    checkpoint = config.parent / "models" / "clm-8b" / "CLM_v0.1-8B.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"ckpt")
    settings = _launcher_settings(config, engine="clm", model_id="clm-8b", model="Qwen/Qwen3-8B", model_revision=CLM_ENCODER)
    command = _command(settings, profile, tmp_path, {})
    assert command[command.index("--revision") + 1] == CLM_ENCODER


def test_clm_sidecar_passes_the_revision_to_vllm_serve() -> None:
    """The sidecar's `vllm serve` call is the only place the encoder revision can be enforced."""
    builder = getattr(clm_sidecar, "_encoder_command", None)
    assert builder is not None, "clm_sidecar has no _encoder_command: vllm serve runs without --revision"
    args = SimpleNamespace(
        encoder_model="Qwen/Qwen3-8B", embedding_model="qwen3-8b", max_tokens=2048,
        gpu_memory_utilization=0.35, revision=CLM_ENCODER,
    )
    command = builder(args, Path("/rt/bin/vllm"), 5000)
    assert command[:3] == ["/rt/bin/vllm", "serve", "Qwen/Qwen3-8B"]
    assert command[command.index("--revision") + 1] == CLM_ENCODER
    assert "--port" in command and command[command.index("--port") + 1] == "5000"


def test_installed_pin_must_match_a_catalog_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin-on-install profiles whose catalog now names the commit: the catalog wins, and an installation
    made at another commit is refused with the repair command instead of silently loading the old files."""
    config = ensure_workspace(tmp_path / "ws")
    profile = get_profile(_catalog(), "basal-1.5-mini", "mlx")
    catalog_sha = BASAL["mini"][1]
    settings = _launcher_settings(
        config, engine="basal", backend="mlx", model_id="basal-1.5-mini", model=profile["model"], model_revision="main"
    )
    monkeypatch.setattr(runtime_module, "installation_record", lambda *args: None)
    assert runtime_module._profile_with_installed_pin(settings, profile)["model_revision"] == catalog_sha

    record = {"artifacts": [{"source": "huggingface", "repo_id": profile["model"], "resolved_revision": catalog_sha}]}
    monkeypatch.setattr(runtime_module, "installation_record", lambda *args: record)
    assert runtime_module._profile_with_installed_pin(settings, profile)["model_revision"] == catalog_sha

    stale = {"artifacts": [{"source": "huggingface", "repo_id": profile["model"], "resolved_revision": "f" * 40}]}
    monkeypatch.setattr(runtime_module, "installation_record", lambda *args: stale)
    with pytest.raises(RuntimeError, match="deqio models update"):
        runtime_module._profile_with_installed_pin(settings, profile)


def test_decision2_triton_pin_matches_the_installed_torch() -> None:
    """The installer puts torch==2.12.0 from the PyTorch index into every Decision 2.0 runtime; torch 2.12.0's
    own metadata requires triton==3.7.0 (PyPI requires_dist, read 2026-10-12), so a profile pin of another
    triton leaves torch's dependency unmet after `uv pip install`."""
    source = Path(model_manager.__file__).read_text(encoding="utf-8")
    torch_version = re.search(r'torch_command\.append\("torch==([\d.]+)"\)', source).group(1)
    expected = {"2.12.0": "3.7.0", "2.12.1": "3.7.1"}[torch_version]
    for model in _catalog()["models"]:
        if model["engine"] != "decision2":
            continue
        for backend, profile in model["backends"].items():
            triton = [spec for spec in profile["packages"] if str(spec).startswith("triton==")]
            assert triton in ([], [f"triton=={expected}"]), f"{model['id']}::{backend}: {triton} with torch {torch_version}"
