"""D11: DecisionProvider — local sidecars and remote decision APIs behind one contract.

Remote providers (openai-decisions, typesafe-jev) are opt-in (flag + key), answer over HTTP in the
provider wire v1, and every response carries ``provider``, ``remote_request_id`` and
``cost_estimate`` in provenance.  The contract tests replay the cassettes in
``tests/fixtures/providers``; a cassette recorded from the real API by
``python -m deqio.provider_record`` replaces the synthetic one without changing these tests.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from deqio import benchmark, server
from deqio.backends import BackendRuntime
from deqio.config import load_settings, settings_from_data
from deqio.workspace import ensure_workspace

try:
    from deqio import providers
    from deqio.remote_runtime import RemoteProviderRuntime
except ImportError as error:  # a Deqio without the provider layer (<= 0.5.9): every test below fails on it
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
FLAG = {"openai-decisions": "DEQIO_PROVIDER_OPENAI_DECISIONS", "typesafe-jev": "DEQIO_PROVIDER_TYPESAFE_JEV"}
KEY = {"openai-decisions": "OPENAI_API_KEY", "typesafe-jev": "TYPESAFE_JEV_API_KEY"}
RATE = {"openai-decisions": 0.10, "typesafe-jev": 0.042}

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


def _cassette(name: str) -> dict:
    return json.loads((CASSETTES / name / "choice.json").read_text(encoding="utf-8"))


class _FakeDecisionsAPI:
    """Minimal remote decision API answering any System One request (wire v1)."""

    def __init__(self, *, status: int = 200, body: dict | None = None) -> None:
        self.requests: list[dict] = []
        self.status = status
        self.body = body
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                api.requests.append({"path": self.path, "headers": dict(self.headers), "body": payload})
                if api.body is not None:
                    response = api.body
                else:
                    answers = {}
                    for qid, question in payload["questions"].items():
                        if question.get("type") == "noul":
                            answers[qid] = {"noul": 0.8}
                        else:
                            ids = list(question["criteria"])
                            answers[qid] = {
                                "choice": ids[0],
                                "probabilities": {oid: (0.7 if oid == ids[0] else 0.3 / max(len(ids) - 1, 1)) for oid in ids},
                            }
                    response = {"id": f"dec_{len(api.requests):04d}", "answers": answers, "usage": {"input_tokens": 100 * len(payload["questions"])}}
                raw = json.dumps(response).encode("utf-8")
                self.send_response(api.status)
                self.send_header("content-type", "application/json")
                self.send_header("x-request-id", f"req_{len(api.requests):04d}")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args) -> None:
                return None

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_FakeDecisionsAPI":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"


def _remote_settings(tmp_path: Path, name: str, base_url: str):
    config = ensure_workspace(tmp_path / "ws")
    settings = load_settings(config)
    return replace(settings, provider=name, provider_model="", provider_base_url=base_url)


def test_provider_registry_names_local_and_the_two_remote_providers() -> None:
    assert providers.provider_spec("local").kind == "local"
    assert providers.provider_spec("openai-decisions").kind == "remote"
    assert providers.provider_spec("typesafe-jev").kind == "remote"
    with pytest.raises(providers.ProviderError, match="Unknown decision provider"):
        providers.provider_spec("nope")


@pytest.mark.parametrize("name", sorted(FLAG))
def test_remote_provider_is_behind_a_flag_and_needs_a_key(name: str) -> None:
    spec = providers.provider_spec(name)
    with pytest.raises(providers.ProviderDisabledError, match=FLAG[name]):
        providers.RemoteDecisionProvider(spec, environ={})
    with pytest.raises(providers.ProviderError, match=KEY[name]):
        providers.RemoteDecisionProvider(spec, base_url="http://127.0.0.1:1", environ={FLAG[name]: "1"})


def test_typesafe_jev_has_no_default_endpoint() -> None:
    spec = providers.provider_spec("typesafe-jev")
    with pytest.raises(providers.ProviderError, match="TYPESAFE_JEV_BASE_URL"):
        providers.RemoteDecisionProvider(spec, environ={FLAG["typesafe-jev"]: "1", KEY["typesafe-jev"]: "k"})


@pytest.mark.parametrize("name", sorted(FLAG))
def test_recorded_cassette_parses_into_a_system_one_answer_with_id_and_cost(name: str) -> None:
    cassette = _cassette(name)
    assert cassette["wire"] == providers.WIRE_VERSION
    spec = providers.provider_spec(name)
    provider = providers.RemoteDecisionProvider(
        spec, base_url="http://127.0.0.1:1", environ={FLAG[name]: "1", KEY[name]: "k"}
    )
    url, headers, body = provider.build_request(cassette["request"]["body"]["state"], cassette["request"]["body"]["questions"])
    assert body == cassette["request"]["body"] and url.endswith(spec.path)
    assert headers["authorization"] == "Bearer k"
    parsed = provider.parse_response(cassette["response"]["status"], cassette["response"]["headers"], cassette["response"]["json"])
    assert parsed.answers["decision"]["choice"] == "bug"
    assert parsed.remote_request_id == cassette["response"]["json"]["id"]
    assert parsed.input_tokens == 412
    estimate = providers.cost_estimate(spec, parsed.input_tokens)
    assert estimate["currency"] == "USD" and estimate["estimate"] is True
    assert estimate["amount"] == pytest.approx(412 * RATE[name] / 1_000_000)


def test_cost_estimate_without_reported_usage_has_no_amount() -> None:
    estimate = providers.cost_estimate(providers.provider_spec("openai-decisions"), None)
    assert estimate["amount"] is None and estimate["basis"] == "input_tokens_unavailable"


def test_parse_response_rejects_non_decisions() -> None:
    provider = providers.RemoteDecisionProvider(
        providers.provider_spec("openai-decisions"), environ={FLAG["openai-decisions"]: "1", KEY["openai-decisions"]: "k"}
    )
    with pytest.raises(providers.ProviderResponseError, match="HTTP 429"):
        provider.parse_response(429, {}, {"error": {"message": "slow down"}})
    with pytest.raises(providers.ProviderResponseError, match="no answers"):
        provider.parse_response(200, {}, {"id": "x", "output": "free text"})


def test_remote_runtime_scores_over_http_with_provider_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FLAG["openai-decisions"], "1")
    monkeypatch.setenv(KEY["openai-decisions"], "test-key")
    with _FakeDecisionsAPI() as api:
        runtime = BackendRuntime.load(_remote_settings(tmp_path, "openai-decisions", api.base_url))
        assert isinstance(runtime, RemoteProviderRuntime)
        identity = runtime.identity_snapshot()
        assert identity["backend"] == "remote" and identity["engine"] == "openai-decisions"
        assert identity["provider"]["model"] == "gpt-6-luna" and "authorization" not in json.dumps(identity).lower()

        raw = runtime.score(CHOICE_ROW, "serial")
        assert raw["provider"]["name"] == "openai-decisions" and raw["provider"]["kind"] == "remote"
        assert raw["remote_request_id"] == "dec_0001"
        assert raw["cost_estimate"]["amount"] == pytest.approx(100 * 0.10 / 1_000_000)
        assert raw["probabilities"][0] == pytest.approx(0.7)
        sent = api.requests[0]
        assert sent["path"] == "/v1/decisions"
        assert sent["headers"]["Authorization"] == "Bearer test-key"
        assert sent["body"]["model"] == "gpt-6-luna" and sent["body"]["questions"]["decision"]["type"] == "choice"

        noul = runtime.score_noul({"id": "n1", "state": "s", "question": "ok?"}, "serial")
        assert noul["probabilities"][0] == pytest.approx(0.8) and noul["remote_request_id"] == "dec_0002"

        shared, timing = runtime.score_shared([CHOICE_ROW, dict(CHOICE_ROW, id="choice-02")])
        assert [r["remote_request_id"] for r in shared] == ["dec_0003", "dec_0003"]
        assert shared[0]["cost_estimate"]["scope"] == "batch" and timing["remote_request_id"] == "dec_0003"
        assert shared[0]["input_tokens"] is None and shared[0]["batch_input_tokens"] == 200

        native, native_timing = runtime.system_one("s", {"q": {"type": "noul", "instructions": "ok?"}})
        assert native["answers"]["q"]["noul"] == 0.8 and native_timing["remote_request_id"] == "dec_0004"
        runtime.close()


def test_remote_runtime_surfaces_api_errors_instead_of_falling_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FLAG["openai-decisions"], "1")
    monkeypatch.setenv(KEY["openai-decisions"], "test-key")
    with _FakeDecisionsAPI(status=503, body={"error": {"message": "overloaded"}}) as api:
        runtime = BackendRuntime.load(_remote_settings(tmp_path, "openai-decisions", api.base_url))
        with pytest.raises(providers.ProviderResponseError, match="HTTP 503") as info:
            runtime.score(CHOICE_ROW, "serial")
        assert info.value.status == 503


def test_remote_runtime_refuses_without_the_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FLAG["openai-decisions"], raising=False)
    monkeypatch.setenv(KEY["openai-decisions"], "test-key")
    with pytest.raises(providers.ProviderDisabledError, match=FLAG["openai-decisions"]):
        BackendRuntime.load(_remote_settings(tmp_path, "openai-decisions", "http://127.0.0.1:1"))


def test_settings_carry_the_provider_and_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ensure_workspace(tmp_path / "ws")
    data = json.loads(config.read_text(encoding="utf-8"))
    settings = settings_from_data(config, data, apply_environment=False)
    assert settings.provider == "local" and settings.provider_model == "" and settings.provider_base_url == ""
    monkeypatch.setenv("DEQIO_PROVIDER", "typesafe-jev")
    monkeypatch.setenv("DEQIO_PROVIDER_MODEL", "jev-pro")
    monkeypatch.setenv("DEQIO_PROVIDER_BASE_URL", "https://jev.example.invalid")
    overridden = settings_from_data(config, data, apply_environment=True)
    assert (overridden.provider, overridden.provider_model, overridden.provider_base_url) == (
        "typesafe-jev", "jev-pro", "https://jev.example.invalid"
    )


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


def test_benchmark_provider_option_selects_the_remote_provider() -> None:
    parser = benchmark.build_parser()
    args = parser.parse_args(["--provider", "openai-decisions", "--provider-model", "gpt-6-luna"])
    assert benchmark._provider_row(args) == {
        "provider": "openai-decisions", "model_id": "gpt-6-luna", "backend": "remote", "engine": "openai-decisions", "provider_base_url": ""
    }
    assert benchmark._provider_row(parser.parse_args(["--all"])) is None
    with pytest.raises(RuntimeError, match="remote provider"):
        benchmark._provider_row(parser.parse_args(["--provider", "local"]))


def test_benchmark_runs_a_suite_through_a_remote_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The summary.json of a provider run carries the remote identity and the cost per decision."""
    monkeypatch.setenv(FLAG["openai-decisions"], "1")
    monkeypatch.setenv(KEY["openai-decisions"], "test-key")
    config = ensure_workspace(tmp_path / "ws")
    suite = {
        "schema_version": 1, "name": "mini", "label": "mini", "description": "three cases",
        "cases": [
            {"id": "noul-01", "type": "noul", "state": "All checks passed.", "question": "Ready?", "expected": "yes"},
            {"id": "choice-01", "type": "choice", "state": CHOICE_ROW["state"], "question": CHOICE_ROW["question"], "options": CHOICE_ROW["options"], "expected": "bug"},
            {"id": "shared-01", "type": "shared", "state": CHOICE_ROW["state"], "decisions": [
                {"id": "d1", "question": CHOICE_ROW["question"], "options": CHOICE_ROW["options"], "expected": "bug"},
                {"id": "d2", "question": "Severity?", "options": [{"id": "high", "description": "h"}, {"id": "low", "description": "l"}], "expected": "high"},
            ]},
        ],
    }
    suite_path = tmp_path / "mini.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    output = tmp_path / "out"
    with _FakeDecisionsAPI() as api:
        code = benchmark.main([
            "--config", str(config), "--suite", str(suite_path), "--output", str(output),
            "--provider", "openai-decisions", "--provider-base-url", api.base_url,
        ])
    assert code == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    model = summary["models"][0]
    assert (model["engine"], model["model_id"], model["backend"]) == ("openai-decisions", "gpt-6-luna", "remote")
    assert model.get("load_error") in (None, "")
    assert model["runtime_identity"]["provider"]["name"] == "openai-decisions"
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and all(row.get("provider_name") == "openai-decisions" for row in rows)
    assert all(row.get("remote_request_id") for row in rows)
    assert any(isinstance(row.get("cost_estimate"), dict) for row in rows)
