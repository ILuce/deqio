"""Opt-in runtime tests for Deqio's Decision 3.0 MLX forward (0.6.1 fix 2, 0.6.2 fix 3) on a real mlx-vlm.

They need the Decision 3.0 MLX runtime packages (``mlx``, ``mlx_vlm``, ``transformers``; on Linux the
``mlx-cpu`` backend works) and the community ``d3_mlx.py`` from a pinned ``ahmedandaloes/d3-*-mlx``
snapshot, named by ``DEQIO_D3_MLX_SNAPSHOT`` (its ``d3_mlx.py`` is copied next to a synthetic
checkpoint; the real weights are not needed).  Without either they are skipped, so ``ci.yml`` is
unaffected.  From the owner's checkout::

    uv pip install --python .model-runtimes/d3-mlx/bin/python pytest httpx
    DEQIO_D3_MLX_SNAPSHOT=~/.cache/huggingface/hub/models--ahmedandaloes--d3-lite-mlx/snapshots/<commit> \\
      PYTHONPATH=src .model-runtimes/d3-mlx/bin/python -m pytest -q tests/test_decision3_mlx_runtime.py

0.6.2 fix 3: the official Decision 3.0 runtime (``vllm-sr/d3-*`` ``d3_runtime.py`` + ``decision_config.json``)
reads the readout by option position (row ``i`` = the ``i``-th answer code), takes the softmax over the first
``n`` codes and runs the full-attention layers without a causal mask (``attention_mode:
noncausal_full_attention``) while the Gated DeltaNet layers stay causal.  The community ``d3_mlx.py`` indexes
the softmax with the codes' tokenizer ids instead (rows 32… of a 255-row head), and 0.6.1 fix 2 copied that
line and ran a causal forward: on the owner's Mac ``d3-lite`` answered the second option / "yes" on 60/60
benchmark cases.  These tests pin the official semantics on a synthetic checkpoint.
"""

from __future__ import annotations

import asyncio
import json
import math
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


def _snapshot(tmp_path_factory: pytest.TempPathFactory, name: str, **kwargs) -> Path:
    snapshot = synthetic.build_checkpoint(tmp_path_factory.mktemp(name), **kwargs)
    shutil.copy(_runtime_file(), snapshot / "d3_mlx.py")
    return snapshot


@pytest.fixture(scope="module")
def regular_snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _snapshot(tmp_path_factory, "d3-regular", layer_types=synthetic.REGULAR_LAYER_TYPES)


@pytest.fixture(scope="module")
def causal_snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _snapshot(tmp_path_factory, "d3-causal", layer_types=synthetic.REGULAR_LAYER_TYPES, attention_mode="causal")


@pytest.fixture(scope="module")
def recurrent_only_snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Non-causal attention mode, but every full-attention ``o_proj`` is zero: only the causal recurrence carries information."""
    return _snapshot(tmp_path_factory, "d3-recurrent", layer_types=synthetic.REGULAR_LAYER_TYPES, zero_attention_output=True)


@pytest.fixture(scope="module")
def irregular_snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _snapshot(tmp_path_factory, "d3-irregular", layer_types=synthetic.IRREGULAR_LAYER_TYPES, seed=11)


def _import_runtime(snapshot: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(sys.modules, "d3_mlx", raising=False)
    monkeypatch.syspath_prepend(str(snapshot))
    import d3_mlx

    assert Path(d3_mlx.__file__).resolve() == (snapshot / "d3_mlx.py").resolve()
    return d3_mlx


def _forward(snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> d3_mlx_sidecar.DecisionForward:
    return d3_mlx_sidecar.DecisionForward(snapshot, _import_runtime(snapshot, monkeypatch))


def _post(app, payload: dict) -> httpx.Response:
    async def call() -> httpx.Response:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
            return await client.post("/v1/systemone", json=payload)

    return asyncio.run(call())


def _prompt_ids(forward: d3_mlx_sidecar.DecisionForward, d3_mlx, state: str, question: dict) -> list[int]:
    text = d3_mlx.render(forward.loaded.tok, state, question, forward.loaded.codes)
    return list(forward.loaded.tok.encode(text, add_special_tokens=False))


def _positional_probabilities(forward: d3_mlx_sidecar.DecisionForward, ids: list[int], count: int) -> list[float]:
    """The official readout rule, computed independently of the sidecar: softmax over the first ``count`` rows."""
    import mlx.core as mx

    h = forward.hidden(ids)
    logits = (forward.loaded.readout @ h[0, -1, :].astype(mx.float32)).tolist()[:count]
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    return [weight / sum(weights) for weight in weights]


def _max_delta(forward: d3_mlx_sidecar.DecisionForward, ids: list[int], cut: int) -> float:
    """How much the hidden state at position ``cut - 1`` changes when the tokens after it are removed."""
    import mlx.core as mx

    full = forward.hidden(ids)
    prefix = forward.hidden(ids[:cut])
    assert full.shape == (1, len(ids), 64) and prefix.shape == (1, cut, 64)
    return float(mx.abs(full[0, cut - 1, :] - prefix[0, cut - 1, :]).max())


def test_community_system_one_still_needs_a_cache(regular_snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Documents the upstream defect Deqio's forward works around.  When this test starts failing, the pinned
    d3_mlx.py (or mlx-vlm) changed: re-evaluate whether Deqio still needs its own forward (0.6.1 fix 2)."""
    d3_mlx = _import_runtime(regular_snapshot, monkeypatch)
    loaded = d3_mlx.D3MLX(str(regular_snapshot))
    assert len(loaded.codes) == 255
    with pytest.raises(AttributeError, match="offset"):
        loaded.system_one("the state x y z", QUESTIONS)


def test_community_loader_indexes_the_readout_with_token_ids(regular_snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard for the second upstream defect (0.6.2 fix 3): the community ``code_ids`` are tokenizer ids, not readout
    rows, so ``probs[code_ids[i]]`` reads the wrong row.  When this starts failing the pinned file changed."""
    d3_mlx = _import_runtime(regular_snapshot, monkeypatch)
    loaded = d3_mlx.D3MLX(str(regular_snapshot))
    config = json.loads((regular_snapshot / "decision_config.json").read_text(encoding="utf-8"))
    assert loaded.code_ids == config["token_ids"] and loaded.codes == config["codes"]
    assert loaded.code_ids[:2] != [0, 1]
    assert "self.code_ids[: len(keys)]" in (regular_snapshot / "d3_mlx.py").read_text(encoding="utf-8")


def test_sidecar_answers_choice_and_noul_through_the_deqio_forward(regular_snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "d3_mlx", raising=False)
    app = d3_mlx_sidecar.build_app(str(regular_snapshot), "ahmedandaloes/d3-lite-mlx")
    response = _post(app, {"state": "the state x y z", "questions": QUESTIONS})
    assert response.status_code == 200, response.text
    body = json.loads(response.text)
    assert body["forward"] == d3_mlx_sidecar.D3_MLX_FORWARD == "deqio-d3-mlx-forward/2"
    route, ok = body["answers"]["route"], body["answers"]["ok"]
    assert route["type"] == "choice" and route["choice"] in {"a", "b"}
    assert abs(sum(route["probabilities"].values()) - 1.0) < 1e-6
    assert ok["type"] == "noul" and 0.0 <= ok["noul"] <= 1.0
    again = json.loads(_post(app, {"state": "the state x y z", "questions": QUESTIONS}).text)
    assert again["answers"] == body["answers"]  # deterministic
    refused = _post(app, {"state": "s", "questions": {"s": {"type": "score", "criteria": ["a", "b"]}}})
    assert refused.status_code == 422


def test_readout_probabilities_are_indexed_by_option_position(regular_snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """0.6.2 fix 3: option ``i`` is readout row ``i`` and the softmax runs over the first ``n`` rows (the official
    ``d3_runtime`` rule), not over all 255 rows read at the codes' tokenizer ids."""
    d3_mlx = _import_runtime(regular_snapshot, monkeypatch)
    forward = d3_mlx_sidecar.DecisionForward(regular_snapshot, d3_mlx)
    answers = forward.system_one("the state x y z", QUESTIONS)["answers"]
    for name, question, keys in (("route", QUESTIONS["route"], ["a", "b"]), ("ok", QUESTIONS["ok"], ["false", "true"])):
        expected = _positional_probabilities(forward, _prompt_ids(forward, d3_mlx, "the state x y z", question), len(keys))
        if name == "route":
            got = [answers[name]["probabilities"][key] for key in keys]
        else:
            got = [1.0 - answers[name]["noul"], answers[name]["noul"]]
        assert got == pytest.approx(expected, abs=1e-5), (name, got, expected)
    assert forward.attention_mode == "noncausal_full_attention" and forward.temperature == 1.0


def test_full_attention_layers_read_the_whole_prompt_and_the_recurrence_stays_causal(
    regular_snapshot: Path, recurrent_only_snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``attention_mode: noncausal_full_attention``: a full-attention layer sees the tokens after a position, so an
    earlier hidden state changes when the suffix changes; with the attention outputs zeroed only the Gated DeltaNet
    recurrence remains and the forward is causal again — the non-causal mask touches the attention layers alone."""
    d3_mlx = _import_runtime(regular_snapshot, monkeypatch)
    forward = d3_mlx_sidecar.DecisionForward(regular_snapshot, d3_mlx)
    assert forward.attention_mode == "noncausal_full_attention"
    assert forward.lm.fa_idx == 3 and forward.lm.ssm_idx == 0  # still corrected from layer_types (the loader hard-codes 2)
    ids = _prompt_ids(forward, d3_mlx, "the state x y z", QUESTIONS["route"])
    assert len(ids) > 12
    assert _max_delta(forward, ids, len(ids) - 6) > 1e-3

    recurrent = _forward(recurrent_only_snapshot, monkeypatch)
    assert recurrent.attention_mode == "noncausal_full_attention"
    ids = _prompt_ids(recurrent, sys.modules["d3_mlx"], "the state x y z", QUESTIONS["route"])
    assert _max_delta(recurrent, ids, len(ids) - 6) < 1e-5


def test_causal_mode_matches_the_standard_mlx_vlm_path(causal_snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``attention_mode: causal`` keeps 0.6.1's behaviour: the per-layer loop equals mlx-vlm's own forward and is causal."""
    import mlx.core as mx

    d3_mlx = _import_runtime(causal_snapshot, monkeypatch)
    forward = d3_mlx_sidecar.DecisionForward(causal_snapshot, d3_mlx)
    assert forward.attention_mode == "causal"
    ids = _prompt_ids(forward, d3_mlx, "the state x y z", QUESTIONS["route"])
    assert _max_delta(forward, ids, len(ids) - 6) < 1e-5
    reference = forward.lm(mx.array([ids], dtype=mx.int32), cache=forward.language_model.make_cache())
    mx.eval(reference)
    assert float(mx.abs(forward.hidden(ids) - reference).max()) < 1e-5
    cache = forward.language_model.make_cache()
    assert [type(entry).__name__ for entry in cache] == ["ArraysCache", "ArraysCache", "ArraysCache", "KVCache"] * 2


def test_irregular_layer_types_take_the_first_full_attention_layer_from_the_checkpoint(irregular_snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d3_mlx = _import_runtime(irregular_snapshot, monkeypatch)
    forward = d3_mlx_sidecar.DecisionForward(irregular_snapshot, d3_mlx)
    assert forward.layer_types == synthetic.IRREGULAR_LAYER_TYPES
    assert forward.lm.fa_idx == 0 and forward.lm.ssm_idx == 1
    result = forward.system_one("state", QUESTIONS)
    assert result["forward"] == d3_mlx_sidecar.D3_MLX_FORWARD
    assert result["answers"]["route"]["choice"] in {"a", "b"} and 0.0 <= result["answers"]["ok"]["noul"] <= 1.0


def test_decision_config_is_required_and_validated(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkpoint's ``decision_config.json`` decides the attention mode and names the codes; it is never guessed."""
    missing = _snapshot(tmp_path_factory, "d3-no-config", decision_config=False)
    with pytest.raises(RuntimeError, match="decision_config.json"):
        _forward(missing, monkeypatch)

    unknown = _snapshot(tmp_path_factory, "d3-unknown-mode", attention_mode="bidirectional_everything")
    with pytest.raises(RuntimeError, match="attention_mode"):
        _forward(unknown, monkeypatch)

    mismatched = _snapshot(tmp_path_factory, "d3-codes")
    config = json.loads((mismatched / "decision_config.json").read_text(encoding="utf-8"))
    config["token_ids"] = list(reversed(config["token_ids"]))
    (mismatched / "decision_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="answer codes"):
        _forward(mismatched, monkeypatch)
