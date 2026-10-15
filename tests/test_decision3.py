"""0.6.1: Decision 3.0 (`decision3`) replaces Decision 2.0.

Official ``vllm-sr`` packages ship their own System One server (``d3_server.py``); Deqio runs it
from the pinned snapshot on CUDA.  GGUF Q8_0 artifacts by ``ggml-org`` run through the pinned
llama.cpp (one bump for every GGUF profile to the commit that fixes the D3 multi-sequence case,
ggml-org/llama.cpp#30408).  MLX uses the community BF16 conversions by ``ahmedandaloes`` through
``deqio.d3_mlx_sidecar`` (choice and noul only: ``d3_mlx.py`` implements no score).
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from deqio import model_manager as manager
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
MLX = {
    "d3": "cbc374ff4bc8e2ee17cb78df4725231f97dc1e17",
    "d3-flash": "77adb3bea9abde92d1f32158743a0cbbb0bf7aa3",
    "d3-mini": "c645173f5729051d473916283f9e37208b5113bb",
    "d3-nano": "b503613cd2bdbbaef1cfbd09c6e348a5758b255b",
    "d3-lite": "235b1fb6ed8d20ab911e10801a4f4cec9cadc24a",
}
MODEL_SHA256 = {
    "d3": "0e439bebbbbf80d9e1b601c56c55a6fc3a6502b4e396cbbb5f3c3f3acc1d607b",
    "d3-flash": "74b71d370f3d9e5731c603dafcb8af4e9134fd2c769c89c5169b031a0418799b",
    "d3-mini": "9ac186bb30b30c129d7aae03d73932600424963bd4644c2041a9d7120f029af5",
    "d3-nano": "a2a6dce5e40298a43f904dfbffe799e09e675d964df7477683ee2bc9db04cdd8",
    "d3-lite": "fda5cfc7bc9063fbb7cd40b50fa9a0963d83a4790a3d6a11a356833fd1171992",
}


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
def test_decision3_catalog_entry_pins_the_three_backends(model_id: str) -> None:
    catalog = _catalog()
    entry = next(model for model in catalog["models"] if model["id"] == model_id)
    assert entry["engine"] == "decision3" and entry["source"] == f"https://huggingface.co/vllm-sr/{model_id}"
    assert set(entry["backends"]) == {"cuda", "gguf", "mlx"}

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

    mlx = get_profile(catalog, model_id, "mlx")
    assert mlx["model"] == f"ahmedandaloes/{model_id}-mlx" and mlx["model_revision"] == MLX[model_id]
    assert mlx["download"] == {"type": "snapshot", "repo_id": f"ahmedandaloes/{model_id}-mlx", "revision": MLX[model_id]}
    assert mlx["installer"] == "d3_mlx" and mlx["launcher"] == "d3_mlx" and mlx["runtime_key"] == "d3-mlx"
    assert mlx["source"] == "community" and mlx["source_url"] == f"https://huggingface.co/ahmedandaloes/{model_id}-mlx"
    assert mlx["precision"] == "bf16" and "bits" not in mlx and "quantization" not in mlx
    assert mlx["systems"] == ["Darwin"] and mlx["platform"] == "mlx"
    assert "mlx-vlm==0.7.6" in mlx["packages"] and "mlx==0.32.3" in mlx["packages"] and "transformers==5.17.0" in mlx["packages"]
    assert mlx["capabilities"]["score"] is False and mlx["capabilities"]["choice"] is True and mlx["capabilities"]["noul"] is True
    assert mlx["capabilities"]["systemone"] is True and mlx["capabilities"]["multi_question"] is True


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


def test_decision3_mlx_launcher_uses_the_deqio_sidecar_on_the_pinned_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from deqio import systemone_runtime

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "d3_mlx.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(systemone_runtime, "_pinned_snapshot_path", lambda *args, **kwargs: str(snapshot))
    profile = get_profile(_catalog(), "d3-flash", "mlx")
    settings = SimpleNamespace(
        engine="decision3", backend="mlx", model_id="d3-flash", model=profile["model"], model_revision=profile["model_revision"],
        max_tokens=4096, config_path=tmp_path / "config.json", runtime_dir=tmp_path / ".model-runtimes",
    )
    python = tmp_path / "env" / "bin" / "python"

    command = SystemOneRuntime._command(settings, profile, tmp_path / "env", python, 9101, {})

    sidecar = Path(systemone_runtime.__file__).with_name("d3_mlx_sidecar.py").resolve()
    assert command == [
        str(python), str(sidecar),
        "--snapshot", str(snapshot),
        "--name", "ahmedandaloes/d3-flash-mlx",
        "--port", "9101",
    ]


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

    commands.clear()
    manager._install_runtime(config, {"runtime_dir": ".model-runtimes", "backend": "mlx"}, get_profile(catalog, "d3-mini", "mlx"), upgrade=False)
    assert len(commands) == 1 and "mlx-vlm==0.7.6" in commands[0] and "fastapi==0.141.1" in commands[0]
    assert "--exclude-newer" in commands[0]


REGULAR_LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2


class _FakeArray:
    """Enough of ``mlx.core.array`` for the sidecar's forward: indexing, astype, matmul, float()."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        value = self.data
        for k in keys:
            value = value[k] if not isinstance(k, slice) else value[k]
        return _FakeArray(value) if isinstance(value, list) else value

    def astype(self, _dtype):
        return self

    def __matmul__(self, other):
        return _FakeArray([sum(a * b for a, b in zip(row, other.data)) for row in self.data])

    def __truediv__(self, divisor):
        return _FakeArray([value / divisor for value in self.data])

    @property
    def shape(self):
        shape, value = [], self.data
        while isinstance(value, list):
            shape.append(len(value))
            value = value[0] if value else None
        return tuple(shape)


def _fake_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    import math

    core = types.ModuleType("mlx.core")
    core.int32, core.float32 = "int32", "float32"
    core.array = lambda data, dtype=None: _FakeArray(data)
    core.eval = lambda *args: None

    def softmax(x, axis=-1):
        peak = max(x.data)
        weights = [math.exp(v - peak) for v in x.data]
        return _FakeArray([w / sum(weights) for w in weights])

    core.softmax = softmax
    package = types.ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)


FAKE_CODES = ["A", "B", "C", "D"] + [f"X{i}" for i in range(251)]
FAKE_TOKEN_IDS = list(range(32, 32 + 255))  # like the real tokenizer: "A" is token 32, "B" 33 … — never readout rows


def _fake_community_runtime(
    snapshot: Path,
    layer_types: list[str],
    *,
    model_is_linear: list[bool] | None = None,
    hidden: int = 40,
    attention_mode: str = "noncausal_full_attention",
    decision_config: bool = True,
    token_ids: list[int] | None = None,
):
    """A d3_mlx.py stand-in: loader + d3 contract, and a ``system_one`` that fails like the real one does.

    The fake last hidden state has logit 10 at readout row 0 (answer code ``A``) and logit 5 at row 33 (the
    tokenizer id of ``B``): a forward that reads the readout by option position answers ``A``; one that reads
    ``probs[code_ids[i]]`` (the community line copied by 0.6.1 fix 2) answers ``B``.
    """
    (snapshot / "d3_mlx.py").write_text("", encoding="utf-8")  # the sidecar checks the runtime file before importing
    (snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {"layer_types": layer_types}}), encoding="utf-8")
    code_ids = list(FAKE_TOKEN_IDS if token_ids is None else token_ids)
    if decision_config:
        (snapshot / "decision_config.json").write_text(json.dumps({
            "format_version": 1, "format_id": "d3-code-readout-v1", "prompt": "d3", "codes": FAKE_CODES,
            "token_ids": FAKE_TOKEN_IDS, "temperature": 1.0, "attention_mode": attention_mode, "pooling": "last",
            "max_length": None, "readout_dtype": "float32",
        }), encoding="utf-8")
    is_linear = model_is_linear if model_is_linear is not None else [kind == "linear_attention" for kind in layer_types]
    record: dict = {"loaded": [], "forwards": [], "layer_calls": [], "community_calls": 0}

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [len(word) for word in text.split()]

    def last_hidden(length: int) -> _FakeArray:
        vector = [0.0] * hidden
        vector[0] = 10.0  # readout row 0 = answer code A (option position 0)
        vector[33] = 5.0  # readout row 33 = what probs[token id of "B"] reads
        return _FakeArray([[[0.0] * hidden] * (length - 1) + [vector]])

    class Layer:
        def __init__(self, index: int, linear: bool) -> None:
            self.index, self.is_linear = index, linear

        def __call__(self, h, mask=None, cache=None, **_kwargs):
            record["layer_calls"].append({"layer": self.index, "is_linear": self.is_linear, "mask": mask, "cache": cache})
            return h

    class LanguageModelModel:
        def __init__(self):
            self.layers = [Layer(index, flag) for index, flag in enumerate(is_linear)]
            self.fa_idx = 2  # the community loader's hard-coded value
            self.ssm_idx = 0

        def embed_tokens(self, ids):
            record["forwards"].append({"ids": ids.data, "fa_idx": self.fa_idx, "ssm_idx": self.ssm_idx, "path": "layers"})
            return last_hidden(len(ids.data[0]))

        def norm(self, h):
            return h

        def __call__(self, ids, cache=None):  # mlx-vlm's standard (causal) path, the one 0.6.1 fix 2 called
            record["forwards"].append({"ids": ids.data, "cache": cache, "fa_idx": self.fa_idx, "ssm_idx": self.ssm_idx, "path": "standard"})
            return last_hidden(len(ids.data[0]))

    class LanguageModel:
        def __init__(self, model):
            self.model = model
            self.caches = []

        def make_cache(self):
            cache = [object() for _ in self.model.layers]
            self.caches.append(cache)
            return cache

    class D3MLX:
        def __init__(self, model_dir):
            record["loaded"].append(str(model_dir))
            self.lm = LanguageModelModel()
            self.net = SimpleNamespace(language_model=LanguageModel(self.lm))
            self.tok = Tokenizer()
            self.readout = _FakeArray([[1.0 if row == column else 0.0 for column in range(hidden)] for row in range(255)])
            self.codes = list(FAKE_CODES)
            self.code_ids = code_ids

        def system_one(self, state, questions, batch_size=8):
            record["community_calls"] += 1
            raise AttributeError("'NoneType' object has no attribute 'offset'")

    def options(question):
        if question["type"] == "choice":
            keys = list(question["criteria"])
            return keys, [k if v is None else f"{k}: {v}" for k, v in question["criteria"].items()]
        if question["type"] == "noul":
            return ["false", "true"], ["No / false", "Yes / true"]
        raise ValueError(f"unsupported question type {question['type']!r}")

    def render(tok, state, question, codes):
        keys, texts = options(question)
        return f"system user {state} " + " ".join(f"{code}: {text}" for code, text in zip(codes, texts)) + " assistant"

    def to_answer(question, probabilities):
        keys, _ = options(question)
        total = sum(probabilities)
        values = [p / total for p in probabilities]
        if question["type"] == "noul":
            return {"type": "noul", "noul": values[1]}
        best = max(range(len(values)), key=values.__getitem__)
        return {"type": "choice", "choice": keys[best], "probabilities": dict(zip(keys, values))}

    fake = types.ModuleType("d3_mlx")
    fake.D3MLX, fake.options, fake.render, fake.to_answer = D3MLX, options, render, to_answer
    return fake, record


def _sidecar_client(app):
    async def post(payload: dict) -> httpx.Response:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
            return await client.post("/v1/systemone", json=payload)

    async def health() -> httpx.Response:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
            return await client.get("/health")

    return post, health


def test_decision3_mlx_sidecar_serves_choice_and_noul_and_refuses_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """0.6.1 fix 2: the sidecar answers through Deqio's own forward (fresh mlx-vlm cache, the checkpoint's
    layer_types) and never through the community ``D3MLX.system_one`` loop, which raises on every mlx-vlm.
    0.6.2 fix 3: the forward is the official one — readout rows by option position, softmax over the first n."""
    from deqio import d3_mlx_sidecar

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    fake, record = _fake_community_runtime(snapshot, REGULAR_LAYER_TYPES)
    monkeypatch.setitem(sys.modules, "d3_mlx", fake)
    _fake_mlx(monkeypatch)

    app = d3_mlx_sidecar.build_app(str(snapshot), "ahmedandaloes/d3-mini-mlx")
    post, health = _sidecar_client(app)

    first_health = asyncio.run(health()).json()
    assert record["loaded"] == [] and first_health["status"] == "process-ready"
    questions = {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": None}}, "n": {"type": "noul"}}
    response = asyncio.run(post({"state": "s", "questions": questions}))
    assert response.status_code == 200, response.text  # 0.6.1 fix 1 answered 500 here: the community loop raised
    body = json.loads(response.text)
    assert body["answers"]["q"]["choice"] == "a" and body["answers"]["q"]["probabilities"]["a"] > 0.99  # fix 2 answered "b"
    assert body["answers"]["n"]["noul"] < 0.01  # row 1 ("true") has logit 0 against row 0's 10; fix 2 answered 1.0
    assert first_health["forward"] == d3_mlx_sidecar.D3_MLX_FORWARD == "deqio-d3-mlx-forward/2"
    assert body["forward"] == "deqio-d3-mlx-forward/2"
    assert body["model"] == "ahmedandaloes/d3-mini-mlx" and body["latency_ms"] >= 0
    assert record["loaded"] == [str(snapshot)] and record["community_calls"] == 0
    assert [forward["path"] for forward in record["forwards"]] == ["layers", "layers"]
    for forward in record["forwards"]:
        assert forward["fa_idx"] == 3 and forward["ssm_idx"] == 0  # from layer_types, not the loader's fa_idx = 2
    assert len(record["layer_calls"]) == 2 * len(REGULAR_LAYER_TYPES)
    first_caches = [call["cache"] for call in record["layer_calls"][: len(REGULAR_LAYER_TYPES)]]
    second_caches = [call["cache"] for call in record["layer_calls"][len(REGULAR_LAYER_TYPES):]]
    assert all(entry is not None for entry in first_caches) and not set(map(id, first_caches)) & set(map(id, second_caches))  # a fresh cache per question
    ready_health = asyncio.run(health()).json()
    assert ready_health["status"] == "ready" and ready_health["attention_mode"] == "noncausal_full_attention"
    refused = asyncio.run(post({"state": "s", "questions": {"q": {"type": "score", "criteria": ["a", "b"]}}}))
    assert refused.status_code == 422 and "score" in refused.text
    assert len(record["forwards"]) == 2


def test_decision3_mlx_forward_indexes_the_readout_by_option_position_not_token_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """0.6.2 fix 3: the official ``d3_runtime`` reads readout row ``i`` for option ``i`` and takes the softmax over the
    first ``n`` rows; ``probs[code_ids[i]]`` (community line, copied by fix 2) reads rows 32… of the 255-way softmax, so
    on the owner's Mac d3-lite answered the second option / "yes" on 60/60 cases regardless of the state."""
    from deqio import d3_mlx_sidecar

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    fake, _record = _fake_community_runtime(snapshot, REGULAR_LAYER_TYPES)
    monkeypatch.setitem(sys.modules, "d3_mlx", fake)
    _fake_mlx(monkeypatch)
    forward = d3_mlx_sidecar.DecisionForward(snapshot, fake)
    questions = {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B", "c": "C"}}, "n": {"type": "noul"}}
    answers = forward.system_one("s", questions)["answers"]
    assert answers["q"]["choice"] == "a"
    probabilities = answers["q"]["probabilities"]
    assert probabilities["a"] > 0.99 and abs(probabilities["b"] - probabilities["c"]) < 1e-9  # rows 1 and 2 both logit 0
    assert abs(sum(probabilities.values()) - 1.0) < 1e-9  # softmax over the first n rows, not a slice of the 255-way one
    assert answers["n"]["noul"] < 0.01
    assert forward.temperature == 1.0 and forward.attention_mode == "noncausal_full_attention"


def test_decision3_mlx_forward_applies_the_checkpoints_attention_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``decision_config.json`` names the attention mode: ``noncausal_full_attention`` (every published Decision 3.0
    checkpoint) runs the full-attention layers with no mask, ``causal`` with the causal mask; the Gated DeltaNet
    layers never get a mask (their recurrence is causal by construction); every question starts a fresh cache."""
    from deqio import d3_mlx_sidecar

    def run(attention_mode: str):
        snapshot = tmp_path / attention_mode
        snapshot.mkdir()
        fake, record = _fake_community_runtime(snapshot, REGULAR_LAYER_TYPES, attention_mode=attention_mode)
        monkeypatch.setitem(sys.modules, "d3_mlx", fake)
        _fake_mlx(monkeypatch)
        forward = d3_mlx_sidecar.DecisionForward(snapshot, fake)
        forward.system_one("s", {"n": {"type": "noul"}})
        return forward, record

    forward, record = run("noncausal_full_attention")
    assert forward.attention_mode == "noncausal_full_attention"
    assert len(record["layer_calls"]) == len(REGULAR_LAYER_TYPES) and record["forwards"][0]["path"] == "layers"
    assert [call["mask"] for call in record["layer_calls"]] == [None] * len(REGULAR_LAYER_TYPES)
    assert all(call["cache"] is not None for call in record["layer_calls"])

    forward, record = run("causal")
    assert forward.attention_mode == "causal"
    masks = [(call["is_linear"], call["mask"]) for call in record["layer_calls"]]
    assert masks == [(True, None), (True, None), (True, None), (False, "causal")] * 2


def test_decision3_mlx_forward_requires_a_valid_decision_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No decision_config.json, an unknown attention mode or codes that disagree with the loader are refused with the
    cause in the sidecar reply (never a guessed mode, never a silently different readout)."""
    from deqio import d3_mlx_sidecar

    cases = [
        ("missing", {"decision_config": False}, "decision_config.json"),
        ("unknown", {"attention_mode": "bidirectional"}, "attention_mode"),
        ("codes", {"token_ids": list(range(255))}, "answer codes"),
    ]
    for name, kwargs, needle in cases:
        snapshot = tmp_path / name
        snapshot.mkdir()
        fake, record = _fake_community_runtime(snapshot, REGULAR_LAYER_TYPES, **kwargs)
        monkeypatch.setitem(sys.modules, "d3_mlx", fake)
        _fake_mlx(monkeypatch)
        post, _health = _sidecar_client(d3_mlx_sidecar.build_app(str(snapshot), "ahmedandaloes/d3-nano-mlx"))
        response = asyncio.run(post({"state": "s", "questions": {"n": {"type": "noul"}}}))
        assert response.status_code == 500, (name, response.text)
        assert needle in response.text and "d3_mlx" in response.text, (name, response.text)
        assert record["forwards"] == [] and record["layer_calls"] == [], name


def test_decision3_mlx_sidecar_refuses_a_checkpoint_whose_layers_do_not_match_layer_types(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A loaded model typed differently from the checkpoint's layer_types is refused explicitly (no mistyped forward)."""
    from deqio import d3_mlx_sidecar

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    mistyped = [True] * len(REGULAR_LAYER_TYPES)  # every layer linear, although layer_types names full attention
    fake, record = _fake_community_runtime(snapshot, REGULAR_LAYER_TYPES, model_is_linear=mistyped)
    monkeypatch.setitem(sys.modules, "d3_mlx", fake)
    _fake_mlx(monkeypatch)
    post, _health = _sidecar_client(d3_mlx_sidecar.build_app(str(snapshot), "ahmedandaloes/d3-nano-mlx"))

    response = asyncio.run(post({"state": "s", "questions": {"n": {"type": "noul"}}}))
    assert response.status_code == 500
    assert "layer_types" in response.text and "is_linear" in response.text and "Layer 3" in response.text, response.text
    assert record["forwards"] == [] and record["community_calls"] == 0

    only_linear = tmp_path / "linear"
    only_linear.mkdir()
    fake2, record2 = _fake_community_runtime(only_linear, ["linear_attention"] * 4)
    monkeypatch.setitem(sys.modules, "d3_mlx", fake2)
    post2, _ = _sidecar_client(d3_mlx_sidecar.build_app(str(only_linear), "ahmedandaloes/d3-nano-mlx"))
    response2 = asyncio.run(post2({"state": "s", "questions": {"n": {"type": "noul"}}}))
    assert response2.status_code == 500 and "full_attention" in response2.text, response2.text
    assert record2["forwards"] == []


def test_decision3_mlx_runtime_metadata_names_the_deqio_forward() -> None:
    from deqio.d3_mlx_sidecar import D3_MLX_FORWARD

    profile = get_profile(_catalog(), "d3-lite", "mlx")
    settings = SimpleNamespace(engine="decision3", backend="mlx", model_id="d3-lite")
    metadata = SystemOneRuntime._decision3_runtime_metadata(settings, profile)
    assert metadata["family"] == "decision3" and metadata["official_runtime"] is False
    assert metadata["forward"] == D3_MLX_FORWARD == "deqio-d3-mlx-forward/2"  # fix 3 changed the semantics: new marker
    gguf = SystemOneRuntime._decision3_runtime_metadata(SimpleNamespace(engine="decision3", backend="gguf", model_id="d3-lite"), get_profile(_catalog(), "d3-lite", "gguf"))
    assert "forward" not in gguf


def test_decision3_native_system_one_honors_backend_capabilities() -> None:
    catalog = _catalog()
    runtime = object.__new__(SystemOneRuntime)
    runtime.engine = "decision3"
    runtime.settings = SimpleNamespace(backend="mlx", model="ahmedandaloes/d3-mini-mlx")
    runtime.profile = get_profile(catalog, "d3-mini", "mlx")
    with pytest.raises(SystemOneCapabilityError, match="score"):
        runtime.system_one("state", {"s": {"type": "score", "instructions": "how?", "criteria": ["a", "b"]}})
    with pytest.raises(SystemOneCapabilityError, match="facts"):
        runtime.system_one("state", {"q": {"type": "noul", "instructions": "ok?"}}, {"facts": "auto"})

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


# --- 0.6.1 fix 1: the MLX runtime pin must carry mlx_vlm.models.qwen3_5 -------------------------------
MLX_VLM_PIN = "mlx-vlm==0.7.6"  # 2026-10-05, the last release before the catalog cutoff; qwen3_5 (GatedDeltaNet) since v0.7.x
MLX_PIN = "mlx==0.32.3"  # the mlx that mlx-vlm 0.7.6 requires (macOS 14+ wheels)


def test_decision3_mlx_runtime_pin_carries_the_qwen3_5_model() -> None:
    """The community d3_mlx.py imports `mlx_vlm.models.qwen3_5`; mlx-vlm 0.3.9 (the pin 0.6.1 shipped, resolved by
    uv for a macOS 13 target) has no such module and the install-time probe failed with ImportError on the owner's
    Mac. Every D3 MLX profile pins the release that has it and the matching mlx, without mlx-lm."""
    for model_id in sorted(OFFICIAL):
        packages = get_profile(_catalog(), model_id, "mlx")["packages"]
        assert MLX_VLM_PIN in packages and MLX_PIN in packages, (model_id, packages)
        assert not any(spec.startswith("mlx-lm") or spec.startswith("mlx-vlm==0.3") for spec in packages), packages
        pins = {spec.split("==")[0]: spec for spec in packages}
        assert pins["transformers"] == "transformers==5.17.0"
        assert pins["fastapi"] == "fastapi==0.141.1" and pins["uvicorn[standard]"] == "uvicorn[standard]==0.53.0"


def test_decision3_mlx_sidecar_reports_a_runtime_import_failure_in_the_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken runtime environment must surface its cause through the HTTP reply Deqio prints, not only in the
    sidecar's traceback (0.6.1 on the owner's Mac printed `HTTP 500: Internal Server Error` for the ImportError)."""
    import importlib.abc
    import importlib.machinery

    from deqio import d3_mlx_sidecar

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "d3_mlx.py").write_text("", encoding="utf-8")

    class Broken(importlib.abc.Loader):
        def create_module(self, spec):
            raise ImportError("cannot import name 'qwen3_5' from 'mlx_vlm.models'")

        def exec_module(self, module):  # pragma: no cover - create_module raises first
            raise AssertionError

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "d3_mlx":
                return importlib.machinery.ModuleSpec("d3_mlx", Broken())
            return None

    monkeypatch.delitem(sys.modules, "d3_mlx", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Finder(), *sys.meta_path])
    app = d3_mlx_sidecar.build_app(str(snapshot), "ahmedandaloes/d3-lite-mlx")

    async def call() -> httpx.Response:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
            return await client.post("/v1/systemone", json={"state": "s", "questions": {"q": {"type": "noul"}}})

    response = asyncio.run(call())
    assert response.status_code == 500
    assert "qwen3_5" in response.text and "d3_mlx" in response.text, response.text
    assert response.headers.get("content-type", "").startswith("application/json")
