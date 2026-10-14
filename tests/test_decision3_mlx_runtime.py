"""Opt-in runtime tests for Deqio's Decision 3.0 MLX forward (0.6.1 fix 2) on a real mlx-vlm.

They need the Decision 3.0 MLX runtime packages (``mlx``, ``mlx_vlm``, ``transformers``; on Linux the
``mlx-cpu`` backend works) and the community ``d3_mlx.py`` from a pinned ``ahmedandaloes/d3-*-mlx``
snapshot, named by ``DEQIO_D3_MLX_SNAPSHOT`` (its ``d3_mlx.py`` is copied next to a synthetic
checkpoint; the real weights are not needed).  Without either they are skipped, so ``ci.yml`` is
unaffected.  From the owner's checkout::

    uv pip install --python .model-runtimes/d3-mlx/bin/python pytest httpx
    DEQIO_D3_MLX_SNAPSHOT=~/.cache/huggingface/hub/models--ahmedandaloes--d3-lite-mlx/snapshots/<commit> \\
      PYTHONPATH=src .model-runtimes/d3-mlx/bin/python -m pytest -q tests/test_decision3_mlx_runtime.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import httpx
import pytest

mlx_vlm = pytest.importorskip("mlx_vlm")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from deqio import d3_mlx_sidecar  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _d3_mlx_synthetic as synthetic  # noqa: E402

QUESTIONS = {
    "route": {"type": "choice", "instructions": "Pick the route.", "criteria": {"a": "left", "b": "right"}},
    "ok": {"type": "noul"},
}


def _runtime_file() -> Path:
    value = os.environ.get("DEQIO_D3_MLX_SNAPSHOT")
    if not value:
        pytest.skip("DEQIO_D3_MLX_SNAPSHOT (a pinned ahmedandaloes/d3-*-mlx snapshot with d3_mlx.py) is not set")
    runtime = Path(value).expanduser() / "d3_mlx.py"
    if not runtime.is_file():
        pytest.skip(f"{runtime} does not exist")
    return runtime


@pytest.fixture(scope="module")
def regular_snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    snapshot = synthetic.build_checkpoint(tmp_path_factory.mktemp("d3-regular"), layer_types=synthetic.REGULAR_LAYER_TYPES)
    shutil.copy(_runtime_file(), snapshot / "d3_mlx.py")
    return snapshot


@pytest.fixture(scope="module")
def irregular_snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    snapshot = synthetic.build_checkpoint(tmp_path_factory.mktemp("d3-irregular"), layer_types=synthetic.IRREGULAR_LAYER_TYPES, seed=11)
    shutil.copy(_runtime_file(), snapshot / "d3_mlx.py")
    return snapshot


def _import_runtime(snapshot: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(sys.modules, "d3_mlx", raising=False)
    monkeypatch.syspath_prepend(str(snapshot))
    import d3_mlx

    assert Path(d3_mlx.__file__).resolve() == (snapshot / "d3_mlx.py").resolve()
    return d3_mlx


def _post(app, payload: dict) -> httpx.Response:
    async def call() -> httpx.Response:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
            return await client.post("/v1/systemone", json=payload)

    return asyncio.run(call())


def test_community_system_one_still_needs_a_cache(regular_snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Documents the upstream defect Deqio's forward works around.  When this test starts failing, the pinned
    d3_mlx.py (or mlx-vlm) changed: re-evaluate whether Deqio still needs its own forward (0.6.1 fix 2)."""
    d3_mlx = _import_runtime(regular_snapshot, monkeypatch)
    loaded = d3_mlx.D3MLX(str(regular_snapshot))
    assert len(loaded.codes) == 255
    with pytest.raises(AttributeError, match="offset"):
        loaded.system_one("the state x y z", QUESTIONS)


def test_sidecar_answers_choice_and_noul_through_the_deqio_forward(regular_snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "d3_mlx", raising=False)
    app = d3_mlx_sidecar.build_app(str(regular_snapshot), "ahmedandaloes/d3-lite-mlx")
    response = _post(app, {"state": "the state x y z", "questions": QUESTIONS})
    assert response.status_code == 200, response.text
    body = json.loads(response.text)
    assert body["forward"] == d3_mlx_sidecar.D3_MLX_FORWARD
    route, ok = body["answers"]["route"], body["answers"]["ok"]
    assert route["type"] == "choice" and route["choice"] in {"a", "b"}
    assert abs(sum(route["probabilities"].values()) - 1.0) < 1e-6
    assert ok["type"] == "noul" and 0.0 <= ok["noul"] <= 1.0
    again = json.loads(_post(app, {"state": "the state x y z", "questions": QUESTIONS}).text)
    assert again["answers"] == body["answers"]  # deterministic
    refused = _post(app, {"state": "s", "questions": {"s": {"type": "score", "criteria": ["a", "b"]}}})
    assert refused.status_code == 422


def test_forward_is_causal_and_uses_the_checkpoints_layer_types(regular_snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mlx.core as mx

    d3_mlx = _import_runtime(regular_snapshot, monkeypatch)
    forward = d3_mlx_sidecar.DecisionForward(regular_snapshot, d3_mlx)
    assert forward.lm.fa_idx == 3 and forward.lm.ssm_idx == 0  # the community loader hard-codes fa_idx = 2
    text = d3_mlx.render(forward.loaded.tok, "the state x y z", QUESTIONS["route"], forward.loaded.codes)
    ids = list(forward.loaded.tok.encode(text, add_special_tokens=False))
    assert len(ids) > 12
    cut = len(ids) - 6
    full = forward.hidden(ids)
    prefix = forward.hidden(ids[:cut])
    assert full.shape == (1, len(ids), 64) and prefix.shape == (1, cut, 64)
    # a causal forward: the hidden state at a position does not depend on the tokens after it
    assert float(mx.abs(full[0, cut - 1, :] - prefix[0, cut - 1, :]).max()) < 1e-4
    # the fresh cache is typed per layer kind
    cache = forward.language_model.make_cache()
    kinds = [type(entry).__name__ for entry in cache]
    assert kinds == ["ArraysCache", "ArraysCache", "ArraysCache", "KVCache"] * 2


def test_irregular_layer_types_take_the_first_full_attention_layer_from_the_checkpoint(irregular_snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d3_mlx = _import_runtime(irregular_snapshot, monkeypatch)
    forward = d3_mlx_sidecar.DecisionForward(irregular_snapshot, d3_mlx)
    assert forward.layer_types == synthetic.IRREGULAR_LAYER_TYPES
    assert forward.lm.fa_idx == 0 and forward.lm.ssm_idx == 1
    result = forward.system_one("state", QUESTIONS)
    assert result["forward"] == d3_mlx_sidecar.D3_MLX_FORWARD
    assert result["answers"]["route"]["choice"] in {"a", "b"} and 0.0 <= result["answers"]["ok"]["noul"] <= 1.0
