"""0.6.1: Decision 3.0 (`decision3`) replaces Decision 2.0; 0.6.2: the MLX backend of Decision 3.0 is removed.

Official ``vllm-sr`` packages ship their own System One server (``d3_server.py``); Deqio runs it
from the pinned snapshot on CUDA.  GGUF Q8_0 artifacts by ``ggml-org`` run through the pinned
llama.cpp (one bump for every GGUF profile to the commit that fixes the D3 multi-sequence case,
ggml-org/llama.cpp#30408).  The community BF16 MLX conversions (``ahmedandaloes``) and Deqio's own
MLX forward (``deqio.d3_mlx_sidecar``, 0.6.1 fix 2 / 0.6.2 fix 3) are gone: on the owner's Mac the
MLX path scored 41.9 % / 56.8 % decision accuracy against 90.3 % / 90.8 % for the same model on GGUF
(runs ``small-pl`` / ``small-eng``, 2026-10-17), so Decision 3.0 runs on CUDA and GGUF only.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from deqio import model_manager as manager
from deqio import systemone_runtime
from deqio.catalog import SUPPORTED_ENGINES, get_profile, load_catalog
from deqio.systemone_runtime import SystemOneCapabilityError, SystemOneRuntime

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "src" / "deqio" / "data" / "models.json"
LLAMA_CPP_D3 = "cbe28bb3cb0758f0c566a9a3561766821aeb0748"  # model: fix vllm d3 multi seq case (#30408), 2026-10-13
OFFICIAL = {
    "d3": "093cffbaa7be6f5dfe14252ae7e80c396c06dbd9",
    "d3-flash": "f6075bff3c0f7d5704e353d0fb2ebf94888d1691",
    "d3-mini": "0e78bbdbe3cf480ed1461c185c9e0c4fa9258011",
    "d3-nano": "9d9b937f5f117f78c030ea7ba677c0b3d328ec58",
    "d3-lite": "3c49f5c0bd48a4b28e29b8ec1f38f56e199f3d61",
}
GGUF = {
    "d3": "de9670136f5ddd83e677c89ce041d5b9bb3b0dcb",
    "d3-flash": "3d53d13778bf8035e6e1e534ce489ed052ee5663",
    "d3-mini": "c03ac9c3bdc5ae949111acb945338933f396e4d7",
    "d3-nano": "67baf946c0042cd74ca7106929ed57db699a5f4b",
    "d3-lite": "a1f9fa124e8d60ac02d1a2884b1dec5a3ec2fd21",
}
MODEL_SHA256 = {
    "d3": "0e439bebbbbf80d9e1b601c56c55a6fc3a6502b4e396cbbb5f3c3f3acc1d607b",
    "d3-flash": "74b71d370f3d9e5731c603dafcb8af4e9134fd2c769c89c5169b031a0418799b",
    "d3-mini": "9ac186bb30b30c129d7aae03d73932600424963bd4644c2041a9d7120f029af5",
    "d3-nano": "a2a6dce5e40298a43f904dfbffe799e09e675d964df7477683ee2bc9db04cdd8",
    "d3-lite": "fda5cfc7bc9063fbb7cd40b50fa9a0963d83a4790a3d6a11a356833fd1171992",
}
MLX_LEFTOVERS = ("d3_mlx", "d3-mlx", "mlx-vlm", "mlx_vlm", "ahmedandaloes", "D3_MLX_FORWARD", "deqio-d3-mlx-forward")


def _catalog() -> dict:
    return load_catalog(CATALOG)


def test_decision3_replaces_decision2_in_engines_catalog_and_code() -> None:
    assert "decision3" in SUPPORTED_ENGINES and "decision2" not in SUPPORTED_ENGINES
    catalog = _catalog()
    ids = [model["id"] for model in catalog["models"]]
    assert not any(model_id.startswith("decision-2.0") for model_id in ids)
    assert [model_id for model_id in ids if model_id.startswith("d3")] == list(OFFICIAL)
    assert not (ROOT / "src" / "deqio" / "decision2_sidecar.py").exists()
    source = (ROOT / "src" / "deqio" / "systemone_runtime.py").read_text(encoding="utf-8")
    assert "decision2" not in source and "Decision 2.0" not in source
    assert "decision2" not in Path(manager.__file__).read_text(encoding="utf-8")
    assert (ROOT / "models.json").read_bytes() == CATALOG.read_bytes()


@pytest.mark.parametrize("model_id", sorted(OFFICIAL))
def test_decision3_catalog_entry_pins_cuda_and_gguf_only(model_id: str) -> None:
    catalog = _catalog()
    entry = next(model for model in catalog["models"] if model["id"] == model_id)
    assert entry["engine"] == "decision3" and entry["source"] == f"https://huggingface.co/vllm-sr/{model_id}"
    assert set(entry["backends"]) == {"cuda", "gguf"}  # 0.6.2: the community MLX conversion was removed
    assert "MLX" not in entry["description"] and "community" not in entry["description"], entry["description"]

    cuda = get_profile(catalog, model_id, "cuda")
    assert cuda["model"] == f"vllm-sr/{model_id}" and cuda["model_revision"] == OFFICIAL[model_id]
    assert cuda["download"] == {"type": "snapshot", "repo_id": f"vllm-sr/{model_id}", "revision": OFFICIAL[model_id]}
    assert cuda["installer"] == "decision3" and cuda["launcher"] == "d3_server" and cuda["runtime_key"] == "decision3-cuda"
    assert cuda["source"] == "official" and cuda["systems"] == ["Linux"] and cuda["platform"] == "cuda"
    assert cuda["family"] == "decision3" and cuda["d3_model_sha256"] == MODEL_SHA256[model_id]
    assert cuda["exclude_newer"] == "2026-10-10T00:00:00Z" and cuda["python"] == "3.12"
    assert "torch==2.13.0" in cuda["packages"] and "transformers==5.17.0" in cuda["packages"]
    assert "flash-linear-attention==0.5.2" in cuda["packages"] and "torchvision==0.28.0" in cuda["packages"]
    assert cuda["capabilities"] == {
        "systemone": True, "choice": True, "noul": True, "score": True, "multi_question": True,
        "multi": False, "act": False, "facts": False, "evidence": False,
    }
    assert cuda["precision"] == {"policy": "upstream-managed", "gpu_backbone": "bf16", "readout": "fp32"}
    assert cuda["default_max_input_tokens"] == 4096 and cuda["min_memory_gib"] >= 2

    gguf = get_profile(catalog, model_id, "gguf")
    assert gguf["model"] == f"ggml-org/vllm-{model_id}-GGUF" and gguf["model_revision"] == GGUF[model_id]
    assert gguf["installer"] == "llama_cpp" and gguf["launcher"] == "llama_cpp" and gguf["runtime_key"] == "llama-cpp-systemone"
    assert gguf["llama_cpp_revision"] == LLAMA_CPP_D3
    assert gguf["download"] == {
        "type": "file", "repo_id": f"ggml-org/vllm-{model_id}-GGUF", "revision": GGUF[model_id],
        "local_dir": f"models/{model_id}-GGUF", "role": "systemone-gguf-q8_0", "filename_pattern": f"vllm-{model_id}-Q8_0.gguf",
    }
    assert gguf["llama_cpp_gguf_dir"] == f"models/{model_id}-GGUF" and gguf["llama_cpp_gguf_pattern"] == f"vllm-{model_id}-Q8_0.gguf"
    assert gguf["quantization"] == {"kind": "gguf", "quantization": "Q8_0", "bits": 8, "block_size": 32, "applied": "checkpoint"}
    assert gguf["source"] == "official" and gguf["precision"] == "q8_0" and gguf["pin_hf_revisions_at_install"] is True
    assert gguf["d3_source_sha"] == OFFICIAL[model_id]
    assert set(gguf["systems"]) == {"Darwin", "Linux"} and gguf["capabilities"]["score"] is True


def test_every_gguf_profile_uses_the_post_30408_llama_cpp_commit() -> None:
    revisions = {
        f"{model['id']}::{backend}": profile.get("llama_cpp_revision")
        for model in _catalog()["models"]
        for backend, profile in model["backends"].items()
        if profile.get("llama_cpp_revision")
    }
    assert revisions, "no llama.cpp profiles"
    assert set(revisions.values()) == {LLAMA_CPP_D3}, revisions
    assert "kev-0.8b::gguf" in revisions and "clef::gguf" in revisions and "d3-mini::gguf" in revisions


def test_decision3_cuda_launcher_runs_the_official_server_from_the_pinned_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deqio import systemone_runtime

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "d3_server.py").write_text("", encoding="utf-8")
    (snapshot / "MODEL_MANIFEST.json").write_text("{}", encoding="utf-8")
    requested: list[tuple] = []
    monkeypatch.setattr(
        systemone_runtime, "_pinned_snapshot_path",
        lambda repo_id, revision, *, model_id, backend: requested.append((repo_id, revision, model_id, backend)) or str(snapshot),
    )
    profile = get_profile(_catalog(), "d3-mini", "cuda")
    settings = SimpleNamespace(
        engine="decision3", backend="cuda", model_id="d3-mini", model=profile["model"], model_revision=profile["model_revision"],
        max_tokens=4096, config_path=tmp_path / "config.json", runtime_dir=tmp_path / ".model-runtimes",
    )
    python = tmp_path / "env" / "bin" / "python"
    env: dict[str, str] = {}

    command = SystemOneRuntime._command(settings, profile, tmp_path / "env", python, 9100, env)

    assert requested == [("vllm-sr/d3-mini", OFFICIAL["d3-mini"], "d3-mini", "cuda")]
    assert command == [
        str(python), str(snapshot / "d3_server.py"),
        "--model", str(snapshot),
        "--device", "cuda:0",
        "--verify", "fast",
        "--name", "vllm-sr/d3-mini",
        "--host", "127.0.0.1",
        "--port", "9100",
    ]
    with pytest.raises(RuntimeError, match="CUDA"):
        SystemOneRuntime._command(SimpleNamespace(**{**vars(settings), "backend": "mps"}), profile, tmp_path / "env", python, 9100, {})


def test_decision3_cuda_launcher_refuses_a_snapshot_without_the_official_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from deqio import systemone_runtime

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr(systemone_runtime, "_pinned_snapshot_path", lambda *args, **kwargs: str(snapshot))
    profile = get_profile(_catalog(), "d3-lite", "cuda")
    settings = SimpleNamespace(
        engine="decision3", backend="cuda", model_id="d3-lite", model=profile["model"], model_revision=profile["model_revision"],
        max_tokens=4096, config_path=tmp_path / "config.json", runtime_dir=tmp_path / ".model-runtimes",
    )
    with pytest.raises(RuntimeError, match="d3_server.py"):
        SystemOneRuntime._command(settings, profile, tmp_path / "env", tmp_path / "python", 9100, {})


def test_decision3_installer_pins_torch_first_then_the_official_requirements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    python = tmp_path / "env" / "bin" / "python"
    monkeypatch.setattr(manager, "_ensure_venv", lambda *args, **kwargs: python)
    monkeypatch.setattr(manager, "_run", lambda command, **kwargs: commands.append(list(command)))
    catalog = _catalog()
    config = tmp_path / "config.json"

    env_dir = manager._install_runtime(config, {"runtime_dir": ".model-runtimes", "backend": "cuda"}, get_profile(catalog, "d3-mini", "cuda"), upgrade=False)

    assert env_dir == (tmp_path / ".model-runtimes" / "decision3-cuda").resolve()
    assert len(commands) == 2
    assert commands[0][:5] == ["uv", "pip", "install", "--python", str(python)]
    assert "--exclude-newer" in commands[0] and commands[0][-1] == "torch==2.13.0"
    assert commands[1][:3] == ["uv", "pip", "install"] and "transformers==5.17.0" in commands[1]
    assert "flash-linear-attention==0.5.2" in commands[1] and "torchvision==0.28.0" in commands[1]
    assert all(spec.count("==") == 1 for spec in commands[1] if not spec.startswith("-") and spec not in {"uv", "pip", "install", str(python), "2026-10-10T00:00:00Z"})
    with pytest.raises(RuntimeError, match="CUDA"):
        manager._install_runtime(config, {"runtime_dir": ".model-runtimes", "backend": "mps"}, {**get_profile(catalog, "d3-mini", "cuda"), "platform": "mps"}, upgrade=False)



def test_decision3_mlx_backend_is_removed_everywhere() -> None:
    """0.6.2: no catalog profile, installer, launcher, sidecar module, runtime metadata or test helper for Decision 3.0
    on MLX remains; the launcher refuses the backend explicitly (no fallback to another backend)."""
    catalog = _catalog()
    for model in catalog["models"]:
        if model["engine"] == "decision3":
            assert "mlx" not in model["backends"], model["id"]
            for profile in model["backends"].values():
                assert profile.get("installer") != "d3_mlx" and profile.get("launcher") != "d3_mlx", model["id"]
                assert profile.get("runtime_key") != "d3-mlx" and "mlx-vlm==0.7.6" not in (profile.get("packages") or [])
    package_dir = Path(systemone_runtime.__file__).parent
    assert not (package_dir / "d3_mlx_sidecar.py").exists()
    assert not (ROOT / "tests" / "test_decision3_mlx_runtime.py").exists()
    assert not (ROOT / "tests" / "_d3_mlx_synthetic.py").exists()
    for module in (systemone_runtime.__file__, manager.__file__, package_dir / "benchmark_tables.py", package_dir / "catalog.py"):
        source = Path(module).read_text(encoding="utf-8")
        for leftover in MLX_LEFTOVERS:
            assert leftover not in source, (module, leftover)
    assert (ROOT / "models.json").read_text(encoding="utf-8") == CATALOG.read_text(encoding="utf-8")  # the workspace template mirrors the package


def test_decision3_launcher_refuses_mlx_explicitly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale `config.json` or registry row that still names decision3 on mlx is refused by name, never launched."""
    profile = {**get_profile(_catalog(), "d3-lite", "cuda"), "platform": "mlx"}
    settings = SimpleNamespace(
        engine="decision3", backend="mlx", model_id="d3-lite", model="community/d3-lite-mlx", model_revision="main",
        max_tokens=4096, config_path=tmp_path / "config.json", runtime_dir=tmp_path / ".model-runtimes",
    )
    monkeypatch.setattr(systemone_runtime, "_pinned_snapshot_path", lambda *args, **kwargs: str(tmp_path / "snapshot"))
    with pytest.raises(RuntimeError) as refused:
        SystemOneRuntime._command(settings, profile, tmp_path / "env", tmp_path / "python", 9100, {})
    message = str(refused.value)
    assert "CUDA" in message and "GGUF" in message and "'mlx'" in message and "no fallback" in message
    assert "sidecar" not in message and "community" not in message  # no MLX path is offered any more


def test_decision3_runtime_metadata_has_no_forward_marker() -> None:
    catalog = _catalog()
    gguf = SystemOneRuntime._decision3_runtime_metadata(SimpleNamespace(engine="decision3", backend="gguf", model_id="d3-lite"), get_profile(catalog, "d3-lite", "gguf"))
    assert gguf == {"family": "decision3", "official_runtime": False, "launcher": "llama_cpp", "d3_source_sha": OFFICIAL["d3-lite"]}
    assert "forward" not in gguf


def test_decision3_native_system_one_honors_backend_capabilities() -> None:
    """A profile that lacks a native capability refuses the request before any inference (no silent degradation)."""
    catalog = _catalog()
    limited = object.__new__(SystemOneRuntime)
    limited.engine = "decision3"
    limited.settings = SimpleNamespace(backend="gguf", model="ggml-org/vllm-d3-mini-GGUF")
    gguf = get_profile(catalog, "d3-mini", "gguf")
    limited.profile = {**gguf, "capabilities": {**gguf["capabilities"], "score": False, "facts": False}}
    with pytest.raises(SystemOneCapabilityError, match="score"):
        limited.system_one("state", {"s": {"type": "score", "instructions": "how?", "criteria": ["a", "b"]}})
    with pytest.raises(SystemOneCapabilityError, match="facts"):
        limited.system_one("state", {"q": {"type": "noul", "instructions": "ok?"}}, {"facts": "auto"})

    sent: list[tuple] = []
    cuda = object.__new__(SystemOneRuntime)
    cuda.engine = "decision3"
    cuda.settings = SimpleNamespace(backend="cuda", model="vllm-sr/d3-mini")
    cuda.profile = get_profile(catalog, "d3-mini", "cuda")
    cuda._request = lambda state, questions, execution_mode=None, request_options=None: sent.append((state, questions)) or (
        {"state": state, "questions": questions}, {"answers": {qid: {"type": q["type"]} for qid, q in questions.items()}, "usage": {"input_tokens": 12}}, 3.0
    )
    response, timing = cuda.system_one("state", {"s": {"type": "score", "instructions": "how?", "criteria": ["a", "b"]}, "n": {"type": "noul"}})
    assert set(response["answers"]) == {"s", "n"} and timing["engine"] == "decision3" and len(sent) == 1


def test_decision3_choice_probabilities_are_preserved_without_renormalization() -> None:
    from deqio.systemone_runtime import _choice_raw

    row = {"id": "r", "state": "s", "question": "q?", "options": [{"id": "a"}, {"id": "b"}]}
    answer = {"type": "choice", "choice": "a", "probabilities": {"a": 0.62, "b": 0.3}, "confidence": 0.62}
    raw = _choice_raw(row=row, answer=answer, response={"answers": {"decision": answer}}, payload={}, latency_ms=1.0, preserve_native="decision3")
    assert raw["probabilities"] == [0.62, 0.3]
    assert "Decision 3.0" in raw["probability_status"] and raw["score_provenance"]["normalized"] is False
