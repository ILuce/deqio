"""D11b: remote decision providers speak the documented wires, not the guessed provider wire v1.

OpenAI Decisions (``POST https://api.openai.com/v1/decisions``) takes ``input`` plus a ``questions``
array (``name``, ``choices``/``levels``, ``predicate`` for yes/no) and answers with an ``answers``
array and ``usage``.  TypeSafe (``POST https://api.typesafe.ai/v1/systemone``) is the native System One
wire: ``state``, ``model`` and questions keyed by id, answers keyed by id, ``usage``.  ``deqio.providers``
maps both to the System One shape the rest of Deqio consumes; the cassettes under
``tests/fixtures/providers`` are real exchanges recorded by the owner with ``python -m deqio.provider_record``
(2026-10-15: OpenAI ``gpt-6-luna`` HTTP 200, 147 input tokens, request id from ``x-request-id``; TypeSafe
``jev-1.13.0`` HTTP 200, 351 input tokens) and the replay test takes every expectation from the recording itself —
never from numbers typed into the test.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from deqio import server
from deqio.config import load_settings
from deqio.workspace import ensure_workspace

from _remote_fakes import FakeOpenAIDecisionsAPI, FakeTypeSafeAPI

try:
    from deqio import providers
    from deqio.remote_runtime import RemoteProviderRuntime
except ImportError as error:  # a Deqio without the provider layer: every test below fails on it
    providers = None  # type: ignore[assignment]
    RemoteProviderRuntime = None  # type: ignore[assignment,misc]
    PROVIDER_IMPORT_ERROR: Exception | None = error
else:
    PROVIDER_IMPORT_ERROR = None


@pytest.fixture(autouse=True)
def _require_provider_layer() -> None:
    if PROVIDER_IMPORT_ERROR is not None:
        pytest.fail(f"decision provider layer is missing: {PROVIDER_IMPORT_ERROR}")


ROOT = Path(__file__).resolve().parents[1]
CASSETTES = ROOT / "tests" / "fixtures" / "providers"
RATE = {"openai-decisions": 0.10, "typesafe-jev": 0.042}
ENDPOINT = {
    "openai-decisions": "https://api.openai.com/v1/decisions",
    "typesafe-jev": "https://api.typesafe.ai/v1/systemone",
}
WIRE = {"openai-decisions": "openai-decisions-v1", "typesafe-jev": "typesafe-systemone-v1"}
DEFAULT_MODEL = {"openai-decisions": "gpt-6-luna", "typesafe-jev": "jev-latest"}
KEY_ENV = {"openai-decisions": "OPENAI_API_KEY", "typesafe-jev": "TYPESAFE_API_KEY"}

CHOICE_ROW = {
    "id": "choice-01",
    "state": "A customer reports that the API returns HTTP 500 after the latest deployment.",
    "question": "Classify the report.",
    "options": [
        {"id": "bug", "description": "An existing function no longer works"},
        {"id": "feature", "description": "A request for new functionality"},
        {"id": "support", "description": "A question about how to use the product"},
    ],
}
QUESTIONS = {
    "route": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Money", "support": None}},
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "severity": {"type": "score", "instructions": "How severe?", "criteria": ["low", "medium", "high"]},
}


def _cassette(name: str) -> dict:
    return json.loads((CASSETTES / name / "choice.json").read_text(encoding="utf-8"))


def _provider(name: str, *, endpoint: str = "", model: str = "") -> "providers.RemoteDecisionProvider":
    return providers.RemoteDecisionProvider(providers.provider_spec(name), api_key="k", endpoint=endpoint, model=model)


def _remote_settings(tmp_path: Path, name: str):
    config = ensure_workspace(tmp_path / "ws")
    settings = load_settings(config)
    return replace(settings, engine=name, model_id=name, backend="remote", model=DEFAULT_MODEL[name], model_revision="remote")


def test_provider_specs_document_the_real_endpoints_and_have_no_flags() -> None:
    for name in sorted(ENDPOINT):
        spec = providers.provider_spec(name)
        assert spec.kind == "remote"
        assert spec.wire == WIRE[name]
        assert spec.default_endpoint == ENDPOINT[name]
        assert spec.default_model == DEFAULT_MODEL[name]
        assert spec.api_key_env == KEY_ENV[name]
        assert spec.usd_per_million_input_tokens == RATE[name]
        assert not hasattr(spec, "flag_env")
    assert not hasattr(providers, "ProviderDisabledError")
    assert providers.provider_spec("local").kind == "local"
    with pytest.raises(providers.ProviderError, match="Unknown decision provider"):
        providers.provider_spec("nope")


def test_remote_provider_needs_an_api_key_and_uses_the_documented_endpoint() -> None:
    spec = providers.provider_spec("typesafe-jev")
    with pytest.raises(providers.ProviderError, match="API key"):
        providers.RemoteDecisionProvider(spec, api_key="")
    provider = providers.RemoteDecisionProvider(spec, api_key="ts-secret-key-123")
    assert provider.endpoint == ENDPOINT["typesafe-jev"] and provider.model == "jev-latest"
    described = provider.describe()
    assert described["endpoint"] == ENDPOINT["typesafe-jev"] and described["wire"] == WIRE["typesafe-jev"]
    assert "ts-secret-key-123" not in json.dumps(described)


def test_openai_build_request_maps_system_one_questions_to_the_decisions_wire() -> None:
    provider = _provider("openai-decisions")
    url, headers, payload = provider.build_request("The payout failed twice.", QUESTIONS)
    assert url == ENDPOINT["openai-decisions"]
    assert headers["authorization"] == "Bearer k" and headers["content-type"] == "application/json"
    assert set(payload) == {"model", "input", "questions"}
    assert payload["model"] == "gpt-6-luna" and payload["input"] == "The payout failed twice."
    assert payload["questions"] == [
        {
            "type": "choice",
            "name": "route",
            "instructions": "Which team?",
            "choices": [{"value": "billing", "description": "Money"}, {"value": "support", "description": ""}],
        },
        {"type": "predicate", "name": "urgent", "instructions": "Is it urgent?"},
        {
            "type": "score",
            "name": "severity",
            "instructions": "How severe?",
            "levels": [
                {"label": "0", "description": "low"},
                {"label": "1", "description": "medium"},
                {"label": "2", "description": "high"},
            ],
        },
    ]
    # A structured state is sent as JSON text: the Decisions API takes text/image input only.
    _, _, structured = provider.build_request({"ticket": 7, "text": "x"}, {"q": {"type": "noul", "instructions": "ok?"}})
    assert json.loads(structured["input"]) == {"ticket": 7, "text": "x"}


def test_openai_parse_response_normalizes_answers_to_the_system_one_shape() -> None:
    provider = _provider("openai-decisions")
    body = {
        "answers": [
            {"type": "choice", "name": "route", "choice": "billing", "confidence": 0.93,
             "probabilities": [{"value": "billing", "probability": 0.7}, {"value": "support", "probability": 0.3}]},
            {"type": "predicate", "name": "urgent", "probability": 0.8},
            {"type": "score", "name": "severity", "score": 1.0, "confidence": 0.5,
             "probabilities": [{"value": 0, "label": "0", "probability": 0.25}, {"value": 1, "label": "1", "probability": 0.5},
                               {"value": 2, "label": "2", "probability": 0.25}]},
        ],
        "model": "gpt-6-luna",
        "usage": {"input_tokens": 412, "output_tokens": 0, "total_tokens": 412},
    }
    parsed = provider.parse_response(200, {"x-request-id": "req_abc"}, body, questions=QUESTIONS)
    assert parsed.answers["route"] == {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.7, "support": 0.3}, "confidence": 0.93}
    assert parsed.answers["urgent"] == {"type": "noul", "noul": 0.8}
    assert parsed.answers["severity"]["type"] == "score" and parsed.answers["severity"]["score"] == 1.0
    assert parsed.answers["severity"]["probabilities"] == {"0": 0.25, "1": 0.5, "2": 0.25}
    assert parsed.answers["severity"]["legend"] == {"0": "low", "1": "medium", "2": "high"}
    assert parsed.input_tokens == 412 and parsed.remote_request_id == "req_abc" and parsed.model == "gpt-6-luna"


def test_openai_refusal_or_missing_answer_is_an_explicit_error() -> None:
    provider = _provider("openai-decisions")
    questions = {"route": QUESTIONS["route"]}
    with pytest.raises(providers.ProviderResponseError, match="refus"):
        provider.parse_response(200, {}, {"answers": [{"type": "refusal", "name": "route"}], "model": "gpt-6-luna"}, questions=questions)
    with pytest.raises(providers.ProviderResponseError, match="route"):
        provider.parse_response(200, {}, {"answers": [], "model": "gpt-6-luna"}, questions=questions)


def test_typesafe_build_request_is_the_native_system_one_wire() -> None:
    provider = _provider("typesafe-jev", model="jev-1.13.0")
    url, headers, payload = provider.build_request({"ticket": 7}, QUESTIONS)
    assert url == ENDPOINT["typesafe-jev"] and headers["authorization"] == "Bearer k"
    assert payload == {"state": {"ticket": 7}, "model": "jev-1.13.0", "questions": QUESTIONS}


def test_typesafe_parse_response_keeps_answers_and_reports_the_versioned_model() -> None:
    provider = _provider("typesafe-jev")
    body = {
        "model": "jev-1.13.0",
        "answers": {"route": {"type": "choice", "choice": "billing", "confidence": 1.0, "probabilities": {"billing": 1.0, "support": 0.0}},
                    "urgent": {"type": "noul", "noul": 0.8}},
        "usage": {"input_tokens": 412, "output_tokens": 0},
    }
    parsed = provider.parse_response(200, {}, body)
    assert parsed.answers == body["answers"] and parsed.model == "jev-1.13.0"
    assert parsed.input_tokens == 412 and parsed.remote_request_id is None
    with_header = provider.parse_response(200, {"X-Request-Id": "ts_1"}, body)
    assert with_header.remote_request_id == "ts_1"


def _recorded_expectations(name: str, cassette: dict) -> dict:
    """What the recorded reply says, read directly in the vendor's own shape (not through ``parse_response``)."""
    reply = cassette["response"]["json"]
    if name == "openai-decisions":
        answer = next(item for item in reply["answers"] if item["name"] == "decision")
        probabilities = {item["value"]: item["probability"] for item in answer["probabilities"]}
        request_id = cassette["response"]["headers"].get("x-request-id")
    else:
        answer = reply["answers"]["decision"]
        probabilities = dict(answer["probabilities"])
        request_id = cassette["response"]["headers"].get("x-request-id")
    return {"choice": answer["choice"], "probabilities": probabilities, "input_tokens": reply["usage"]["input_tokens"],
            "model": reply["model"], "remote_request_id": request_id}


@pytest.mark.parametrize("name", sorted(ENDPOINT))
def test_recorded_cassettes_replay_through_the_provider_mapping(name: str) -> None:
    cassette = _cassette(name)
    spec = providers.provider_spec(name)
    assert cassette["provider"] == name and cassette["wire"] == spec.wire
    assert cassette.get("synthetic") is False, "the shipped cassettes are the owner's real recordings (0.6.2)"
    assert cassette["recorded_latency_ms"] > 0
    provider = _provider(name, model=cassette["request"]["body"]["model"])
    system_one = cassette["request"]["system_one"]
    url, _headers, body = provider.build_request(system_one["state"], system_one["questions"])
    assert url == cassette["request"]["url"] == spec.default_endpoint
    assert body == cassette["request"]["body"]  # the request Deqio builds is byte-for-byte what was sent to the real API
    parsed = provider.parse_response(
        cassette["response"]["status"], cassette["response"]["headers"], cassette["response"]["json"], questions=system_one["questions"]
    )
    expected = _recorded_expectations(name, cassette)
    assert parsed.answers["decision"]["choice"] == expected["choice"] == "bug"
    assert parsed.answers["decision"]["probabilities"] == pytest.approx(expected["probabilities"])
    assert set(parsed.answers["decision"]["probabilities"]) == set(system_one["questions"]["decision"]["criteria"])
    assert abs(sum(parsed.answers["decision"]["probabilities"].values()) - 1.0) < 1e-6
    assert parsed.input_tokens == expected["input_tokens"] > 0
    assert parsed.model == expected["model"]
    assert parsed.remote_request_id == expected["remote_request_id"]
    if name == "openai-decisions":
        assert parsed.remote_request_id.startswith("req_")  # from the x-request-id header; the body has no id
    else:
        assert parsed.remote_request_id is None  # TypeSafe returns no request id (documented)
    estimate = providers.cost_estimate(spec, parsed.input_tokens)
    assert estimate["currency"] == "USD" and estimate["estimate"] is True
    assert estimate["amount"] == pytest.approx(expected["input_tokens"] * RATE[name] / 1_000_000)


def test_cost_estimate_without_reported_usage_has_no_amount() -> None:
    estimate = providers.cost_estimate(providers.provider_spec("openai-decisions"), None)
    assert estimate["amount"] is None and estimate["basis"] == "input_tokens_unavailable"


def test_parse_response_rejects_http_errors_and_non_decisions() -> None:
    provider = _provider("openai-decisions")
    with pytest.raises(providers.ProviderResponseError, match="HTTP 429") as info:
        provider.parse_response(429, {}, {"error": {"message": "slow down"}})
    assert info.value.status == 429
    with pytest.raises(providers.ProviderResponseError, match="no answers"):
        provider.parse_response(200, {}, {"output": "free text"})
    typesafe = _provider("typesafe-jev")
    with pytest.raises(providers.ProviderResponseError, match="HTTP 529"):
        typesafe.parse_response(529, {}, {"detail": "overloaded"})


def test_remote_runtime_scores_over_the_openai_wire_with_provider_provenance(tmp_path: Path) -> None:
    with FakeOpenAIDecisionsAPI() as api:
        provider = _provider("openai-decisions", endpoint=api.endpoint)
        runtime = RemoteProviderRuntime(_remote_settings(tmp_path, "openai-decisions"), provider)
        identity = runtime.identity_snapshot()
        assert identity["backend"] == "remote" and identity["engine"] == "openai-decisions"
        assert identity["provider"]["model"] == "gpt-6-luna" and "authorization" not in json.dumps(identity).lower()

        raw = runtime.score(CHOICE_ROW, "serial")
        assert raw["provider"]["name"] == "openai-decisions" and raw["provider"]["kind"] == "remote"
        assert raw["remote_request_id"] == "req_0001"
        assert raw["cost_estimate"]["amount"] == pytest.approx(raw["cost_estimate"]["input_tokens"] * 0.10 / 1_000_000)
        assert raw["probabilities"][0] == pytest.approx(0.7)
        sent = api.requests[0]
        assert sent["path"] == "/v1/decisions" and sent["headers"]["Authorization"] == "Bearer k"
        assert sent["body"]["input"] == CHOICE_ROW["state"]
        assert sent["body"]["questions"][0]["name"] == "decision" and sent["body"]["questions"][0]["type"] == "choice"
        assert [choice["value"] for choice in sent["body"]["questions"][0]["choices"]] == ["bug", "feature", "support"]

        noul = runtime.score_noul({"id": "n1", "state": "s", "question": "ok?"}, "serial")
        assert noul["probabilities"][0] == pytest.approx(0.8) and noul["remote_request_id"] == "req_0002"
        assert api.requests[1]["body"]["questions"][0]["type"] == "predicate"

        shared, timing = runtime.score_shared([CHOICE_ROW, dict(CHOICE_ROW, id="choice-02")])
        assert [r["remote_request_id"] for r in shared] == ["req_0003", "req_0003"]
        assert shared[0]["cost_estimate"]["scope"] == "batch" and timing["remote_request_id"] == "req_0003"
        assert shared[0]["input_tokens"] is None and shared[0]["batch_input_tokens"] > 0
        assert [question["name"] for question in api.requests[2]["body"]["questions"]] == ["q0", "q1"]

        native, native_timing = runtime.system_one("s", {"q": {"type": "noul", "instructions": "ok?"}})
        assert native["answers"]["q"] == {"type": "noul", "noul": 0.8} and native_timing["remote_request_id"] == "req_0004"
        assert native["model"] == "gpt-6-luna"
        runtime.close()


def test_remote_runtime_scores_over_the_typesafe_wire(tmp_path: Path) -> None:
    with FakeTypeSafeAPI() as api:
        provider = _provider("typesafe-jev", endpoint=api.endpoint, model="jev-1.13.0")
        runtime = RemoteProviderRuntime(_remote_settings(tmp_path, "typesafe-jev"), provider)
        raw = runtime.score(CHOICE_ROW, "serial")
        assert raw["probabilities"][0] == pytest.approx(0.7) and raw["provider"]["name"] == "typesafe-jev"
        sent = api.requests[0]
        assert sent["path"] == "/v1/systemone" and sent["body"]["model"] == "jev-1.13.0"
        assert sent["body"]["state"] == CHOICE_ROW["state"]
        assert sent["body"]["questions"]["decision"]["criteria"] == {
            "bug": "An existing function no longer works",
            "feature": "A request for new functionality",
            "support": "A question about how to use the product",
        }
        native, timing = runtime.system_one("s", {"q": {"type": "score", "instructions": "how?", "criteria": ["a", "b"]}})
        assert native["answers"]["q"]["type"] == "score" and native["model"] == "jev-1.13.0"
        assert timing["cost_estimate"]["amount"] == pytest.approx(100 * 0.042 / 1_000_000)
        runtime.close()


def test_remote_runtime_surfaces_api_errors_instead_of_falling_back(tmp_path: Path) -> None:
    with FakeOpenAIDecisionsAPI(status=503, body={"error": {"message": "overloaded"}}) as api:
        runtime = RemoteProviderRuntime(_remote_settings(tmp_path, "openai-decisions"), _provider("openai-decisions", endpoint=api.endpoint))
        with pytest.raises(providers.ProviderResponseError, match="HTTP 503") as info:
            runtime.score(CHOICE_ROW, "serial")
        assert info.value.status == 503


def test_local_results_carry_the_provider_fields_in_provenance() -> None:
    identity = {"runtime_instance_id": "r1", "artifact_revisions_resolved": True, "provider": {"name": "local", "kind": "local", "engine": "kev"}}
    raw = {"score_provenance": {"kind": "engine_probability"}, "provider": {"name": "local", "kind": "local", "engine": "kev"},
           "remote_request_id": None, "cost_estimate": None}
    response = server._bind_result_provenance({"id": "x", "decision": "a", "probabilities": {"a": 1.0}}, raw, identity)
    provenance = response["provenance"]
    assert provenance["provider"] == {"name": "local", "kind": "local", "engine": "kev"}
    assert provenance["remote_request_id"] is None and provenance["cost_estimate"] is None
    assert provenance["attestation"]["kind"] == "deqio-local-response"

    remote_raw = dict(raw, provider={"name": "openai-decisions", "kind": "remote", "model": "gpt-6-luna"},
                      remote_request_id="dec_1", cost_estimate={"currency": "USD", "amount": 1e-5})
    remote = server._bind_result_provenance({"id": "x", "decision": "a", "probabilities": {"a": 1.0}}, remote_raw, dict(identity, artifact_revisions_resolved=False))
    assert remote["provenance"]["attestation"]["kind"] == "deqio-remote-response"
    assert remote["provenance"]["remote_request_id"] == "dec_1" and remote["provenance"]["cost_estimate"]["amount"] == 1e-5
