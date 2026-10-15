"""D13 (0.6.2): the OpenAPI 3.1 document of the decision routes.

``GET /openapi.json`` (and ``deqio openapi``) describe the eight decision routes and the ``/v1/decision`` alias
with their response contracts — the portable result with ``provenance`` (``provider``, ``remote_request_id``,
``cost_estimate``), the shared batch, the native System One reply with its ``deqio`` block — and the error shapes.
``docs/openapi.json`` is the committed copy; it must equal the generated document.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "docs" / "openapi.json"
DECISION_PATHS = ["/v1/noul", "/v1/choice", "/v1/decision", "/v1/shared", "/v1/score", "/v1/multi", "/v1/act", "/v1/soam", "/v1/systemone"]
PROVENANCE_FIELDS = ("provider", "remote_request_id", "cost_estimate")


def _resolve(document: dict, schema: dict) -> dict:
    while "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        schema = document["components"]["schemas"][name]
    return schema


def _property(document: dict, schema: dict, name: str) -> dict:
    schema = _resolve(document, schema)
    assert name in schema["properties"], (name, sorted(schema["properties"]))
    target = schema["properties"][name]
    if "anyOf" in target:  # nullable: {"anyOf": [{"$ref": ...}, {"type": "null"}]}
        target = next(entry for entry in target["anyOf"] if entry.get("type") != "null")
    return _resolve(document, target)


def test_openapi_document_is_3_1_and_documents_every_decision_route_with_provenance() -> None:
    from deqio.openapi import DECISION_ROUTES, build_openapi
    from deqio.server import app

    document = build_openapi(app)
    assert document["openapi"] == "3.1.0"
    assert document["info"]["version"] == app.version
    assert [route[0] for route in DECISION_ROUTES] == DECISION_PATHS
    operation_ids = []
    for path in DECISION_PATHS:
        operation = document["paths"][path]["post"]
        operation_ids.append(operation["operationId"])
        assert operation["tags"] == ["decisions"] and operation["summary"]
        assert {"200", "400", "422", "502", "503"} <= set(operation["responses"])
        assert "requestBody" in operation  # the pydantic request model FastAPI always documented
        success = _resolve(document, operation["responses"]["200"]["content"]["application/json"]["schema"])
        if path in ("/v1/noul", "/v1/choice", "/v1/decision"):
            assert {"id", "decision", "probabilities", "top_probability", "option_logits", "input_tokens", "timing",
                    "prompt_sha256", "probability_status", "provenance"} <= set(success["required"])
            provenance = _property(document, success, "provenance")
        elif path == "/v1/shared":
            assert {"results", "shared_timing", "provenance"} <= set(success["required"])
            results = _property(document, success, "results")
            assert _resolve(document, results["items"])["title"] == "DecisionResponse"
            provenance = _property(document, success, "provenance")
        else:
            assert {"answers", "deqio"} <= set(success["required"])
            provenance = _property(document, success, "deqio")
        for field in PROVENANCE_FIELDS:
            assert field in provenance["properties"], (path, field)
            assert field in provenance["required"], (path, field)
        provider = _property(document, provenance, "provider")
        assert provider["properties"]["kind"]["enum"] == ["local", "remote"]
        cost = _property(document, provenance, "cost_estimate")
        assert {"amount", "basis", "scope", "estimate"} <= set(cost["properties"])
        runtime = _property(document, provenance, "runtime")
        assert {"runtime_instance_id", "engine", "model_id", "backend"} <= set(runtime["required"])
    assert len(set(operation_ids)) == len(operation_ids) == 9
    assert all(operation_id.startswith("decide_") for operation_id in operation_ids)
    for path in ("/v1/noul", "/v1/choice", "/v1/decision", "/v1/shared"):
        parameters = document["paths"][path]["post"]["parameters"]
        header = next(parameter for parameter in parameters if parameter["name"] == "Deqio-Contract")
        assert header["in"] == "header" and header["schema"]["enum"] == ["input-completeness-v1"]
        assert "oneOf" in document["paths"][path]["post"]["responses"]["422"]["content"]["application/json"]["schema"]
    # The other public routes are still described; nothing was removed from FastAPI's own document.
    for path in ("/health", "/v1/models", "/v1/models/installed", "/v1/models/activate", "/v1/stats", "/v1/recent", "/v1/watch"):
        assert path in document["paths"], path
    assert {"HTTPErrorResponse", "ContractErrorResponse"} <= set(document["components"]["schemas"])


def test_served_openapi_json_equals_the_generated_document_and_the_committed_copy() -> None:
    from deqio.openapi import build_openapi, render
    from deqio.server import app

    async def fetch() -> dict:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://deqio") as client:
            response = await client.get("/openapi.json")
            assert response.status_code == 200
            return response.json()

    generated = build_openapi(app)
    assert asyncio.run(fetch()) == generated
    assert json.loads(render(app)) == generated
    assert COMMITTED.is_file(), "docs/openapi.json is missing; run `uv run deqio openapi --out docs/openapi.json`"
    assert json.loads(COMMITTED.read_text(encoding="utf-8")) == generated, (
        "docs/openapi.json is stale; run `uv run deqio openapi --out docs/openapi.json`"
    )


def test_documented_response_models_validate_real_shapes_and_reject_undocumented_fields() -> None:
    from pydantic import ValidationError

    from deqio.openapi import validate_response
    from deqio.server import _bind_result_provenance, format_result

    raw = {
        "id": "choice-1", "option_ids": ["a", "b"], "probabilities": [0.7, 0.3], "input_tokens": 21,
        "total_seconds": 0.003, "prompt_sha256": "ab" * 32, "probability_status": "engine probabilities",
        "score_provenance": {"kind": "engine_probability", "source": "engine.probabilities", "synthetic": False,
                             "normalized": False, "transforms": [], "raw_logits_available": False, "calibration": "unspecified"},
        "remote_request_id": "req_1", "cost_estimate": {"currency": "USD", "amount": 2e-05, "basis": "input_tokens",
                                                          "scope": "decision", "input_tokens": 200,
                                                          "usd_per_million_input_tokens": 0.1, "price_source": "page", "estimate": True},
        "provider": {"name": "openai-decisions", "kind": "remote", "model": "gpt-6-luna", "endpoint": "https://api.openai.com/v1/decisions"},
    }
    identity = {"deqio_version": "0.6.2", "runtime_instance_id": "remote-1", "engine": "openai-decisions", "model_id": "openai-decisions",
                "backend": "remote", "model": "gpt-6-luna", "requested_revision": None, "artifacts": [],
                "artifact_revisions_resolved": False, "installation_verified_at": None}
    response = format_result(dict(raw), runtime_identity=identity)
    validated = validate_response("/v1/choice", response)
    assert validated.provenance.provider.kind == "remote" and validated.provenance.cost_estimate.amount == 2e-05
    assert validated.provenance.remote_request_id == "req_1"
    assert _bind_result_provenance  # the binding that wrote the fields the document names

    undocumented = dict(response)
    undocumented["surprise"] = 1
    with pytest.raises(ValidationError):
        validate_response("/v1/choice", undocumented)
    with pytest.raises(ValidationError):
        validate_response("/v1/systemone", {"answers": {"q": {"type": "noul", "noul": 0.9}}})  # no deqio block


def test_deqio_openapi_cli_prints_or_writes_the_document(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from deqio.cli import main
    from deqio.openapi import build_openapi
    from deqio.server import app

    out = tmp_path / "nested" / "openapi.json"
    assert main(["openapi", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == build_openapi(app)
    assert str(out) in capsys.readouterr().out
    assert main(["openapi"]) == 0
    assert json.loads(capsys.readouterr().out) == build_openapi(app)
