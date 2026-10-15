"""OpenAPI 3.1 document for the Deqio decision routes (0.6.2, D13).

FastAPI already describes every request body from the pydantic models in ``deqio.server``; the decision handlers
return plain dicts, so their responses were undocumented.  This module adds the response contracts of the eight
decision routes (``/v1/decision`` is the ``/v1/choice`` alias) as components — the portable result with its
Decision Provenance / Attestation block (``provenance.provider``, ``provenance.remote_request_id``,
``provenance.cost_estimate``, additive since 0.6.0.dev0/0.6.1), the shared batch, and the native System One result
with its ``deqio`` block — and the error shapes every route can answer with.

The models below document and validate; they are never used as FastAPI ``response_model``s, so the server keeps
serialising the dicts it always did (nothing is filtered, renamed or defaulted).  ``validate_response`` checks a
real response against the documented shape (top-level fields are closed, engine-specific nested blocks stay open);
the live smoke test runs it over every decision route.  ``build_openapi(app)`` returns the document, ``install(app)``
makes ``GET /openapi.json`` and ``/docs`` serve it, ``deqio openapi`` writes it, and ``docs/openapi.json`` is the
committed copy (a test keeps it equal to the generated one).
"""

from __future__ import annotations

import copy
import json
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, ConfigDict, Field

OPENAPI_VERSION = "3.1.0"
CONTRACT_HEADER = "Deqio-Contract"
CONTRACT_VALUE = "input-completeness-v1"

# (path, response schema, summary, description) — the public decision routes in AGENTS.md order.
DECISION_ROUTES: tuple[tuple[str, str, str, str], ...] = (
    ("/v1/noul", "DecisionResponse", "Noul — yes / no", "One yes/no decision over the state. `decision` is `yes` or `no`."),
    ("/v1/choice", "DecisionResponse", "Choice — exactly one option", "One option out of `options` (2 … 128)."),
    ("/v1/decision", "DecisionResponse", "Choice (compatibility alias)", "Alias of `/v1/choice` with the same request and response."),
    ("/v1/shared", "SharedResponse", "Shared — several decisions over one state",
     "Up to 64 choice decisions answered over one shared state; one result per decision plus a batch-level provenance."),
    ("/v1/score", "SystemOneResponse", "Score — ordered levels (native)",
     "Convenience wrapper that injects `type: score` into one native System One question."),
    ("/v1/multi", "SystemOneResponse", "Multi — independent labels (native)",
     "Convenience wrapper that injects `type: multi` into one native System One question."),
    ("/v1/act", "SystemOneResponse", "Act — cost-aware action (native)",
     "Convenience wrapper that injects `type: act` into one native System One question."),
    ("/v1/soam", "SystemOneResponse", "SOAM — State Once, Ask Many (native)",
     "Several typed questions over one state in one native request; same execution path as `/v1/systemone`."),
    ("/v1/systemone", "SystemOneResponse", "Raw native System One", "The canonical native System One contract of the active profile."),
)
CONTRACT_ROUTES = ("/v1/noul", "/v1/choice", "/v1/decision", "/v1/shared")
RESPONSE_SCHEMAS = ("DecisionResponse", "SharedResponse", "SystemOneResponse")


# ---------------------------------------------------------------------------
# Response models (documentation + validation; never FastAPI response_model)
# ---------------------------------------------------------------------------


class _Open(BaseModel):
    """Engine- or provider-specific block: the documented fields are guaranteed, more may follow."""

    model_config = ConfigDict(extra="allow")


class _Closed(BaseModel):
    """Top-level response: every field is documented here (a new field must be added to the document)."""

    model_config = ConfigDict(extra="forbid")


class ProviderDescriptor(_Open):
    """Who answered: the local sidecar of the active profile, or the remote decision API behind a `remote` profile."""

    name: str = Field(description="`local`, or the remote profile's provider (`openai-decisions`, `typesafe-jev`).")
    kind: Literal["local", "remote"]
    engine: str | None = Field(default=None, description="Local engine name (local providers).")
    model: str | None = Field(default=None, description="Remote model identifier typed at `deqio models setup` (remote providers).")
    endpoint: str | None = Field(default=None, description="Remote endpoint URL (remote providers).")
    wire: str | None = Field(default=None, description="Wire mapping identifier, e.g. `openai-decisions-v1`.")
    usd_per_million_input_tokens: float | None = None
    price_source: str | None = None


class CostEstimate(_Open):
    """Estimated price of one remote call from the API's reported input tokens and the published input price. Never a bill."""

    currency: str = "USD"
    amount: float | None = Field(description="`null` when the API did not report usage.")
    basis: str = Field(description="What the amount is computed from, e.g. `input_tokens`.")
    scope: Literal["decision", "batch"] = Field(description="`decision` for one result, `batch` for a whole `/v1/shared` call.")
    input_tokens: int | None = None
    usd_per_million_input_tokens: float | None = None
    price_source: str | None = None
    estimate: bool = True


class RuntimeIdentity(_Open):
    """The loaded runtime captured under the same inference lock as the score."""

    deqio_version: str
    runtime_instance_id: str | None
    engine: str
    model_id: str
    backend: Literal["mlx", "mps", "gguf", "cuda", "remote"]
    model: str | None = None
    requested_revision: str | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    artifact_revisions_resolved: bool = False
    installation_verified_at: str | None = None
    provider: ProviderDescriptor | None = None
    capabilities: dict[str, Any] | None = None
    max_input_tokens: int | None = None
    source: str | None = None


class ScoreProvenance(_Open):
    """How the probabilities were produced: engine-reported, a Deqio transform, or a synthetic fallback — never hidden."""

    kind: str
    source: str
    synthetic: bool | None = None
    normalized: bool | None = None
    transforms: list[Any] = Field(default_factory=list)
    raw_logits_available: bool | None = None
    calibration: str | None = None


class Attestation(_Closed):
    """Local, unsigned response binding (Attestation v1/v2): not a cryptographic signature or a remote trust proof."""

    kind: str
    signed: bool
    complete: bool
    sha256: str


class DecisionProvenance(_Closed):
    """Decision Provenance / Attestation for one portable result."""

    schema_version: int = Field(description="1, or 2 when the `input-completeness-v1` contract was negotiated.")
    runtime: RuntimeIdentity
    score: ScoreProvenance
    attestation: Attestation
    provider: ProviderDescriptor
    remote_request_id: str | None = Field(description="The remote API's request id, `null` for local engines or when not reported.")
    cost_estimate: CostEstimate | None = Field(description="`null` for local engines.")


class UsageBlock(_Open):
    input_tokens: int | None = None
    input_tokens_source: str | None = None


class DecisionResponse(_Closed):
    """`/v1/noul`, `/v1/choice`, `/v1/decision`: one decision with probabilities over the option ids (`yes`/`no` for noul)."""

    id: str
    decision: str
    probabilities: dict[str, float]
    top_probability: float
    option_logits: dict[str, float] = Field(description="Raw logits when the engine exposes them, otherwise empty.")
    input_tokens: int | None
    timing: dict[str, Any]
    prompt_sha256: str
    probability_status: str
    provenance: DecisionProvenance
    input_receipt: dict[str, Any] | None = Field(default=None, description="Present only for negotiated `Deqio-Contract` requests.")
    usage: UsageBlock | None = Field(default=None, description="Present only for negotiated `Deqio-Contract` requests.")


class SharedScoreProvenance(_Open):
    kind: str
    source: str
    result_count: int


class SharedProvenance(_Closed):
    """Batch-level provenance of `/v1/shared`: the same runtime instance bound to every result."""

    schema_version: int
    runtime: RuntimeIdentity
    score: SharedScoreProvenance
    attestation: Attestation
    provider: ProviderDescriptor
    remote_request_id: str | None
    cost_estimate: CostEstimate | None = Field(description="Batch-scoped (`scope: batch`) when a remote API answered the whole batch.")


class SharedResponse(_Closed):
    results: list[DecisionResponse]
    shared_timing: dict[str, Any]
    provenance: SharedProvenance
    input_receipt: dict[str, Any] | None = None


class NativeTiming(_Open):
    total_ms: float
    decisions: int


class DeqioNativeBlock(_Closed):
    """Deqio's block on native responses: runtime identity, timing and the provider."""

    provenance_schema_version: int
    runtime: RuntimeIdentity
    timing: NativeTiming
    provider: ProviderDescriptor
    remote_request_id: str | None
    cost_estimate: CostEstimate | None


class SystemOneAnswer(_Open):
    """One native answer. `choice`/`probabilities` for choice, `noul` for noul, `score`/`legend` for score, and the
    engine's own fields for multi/act — preserved unchanged from the runtime."""

    type: str


class SystemOneResponse(_Open):
    """`/v1/score`, `/v1/multi`, `/v1/act`, `/v1/soam`, `/v1/systemone`: the native System One reply plus `deqio`.
    Upstream fields other than `model`, `answers` and `usage` are preserved as the runtime returned them."""

    answers: dict[str, SystemOneAnswer]
    model: str | None = None
    usage: dict[str, Any] | None = None
    deqio: DeqioNativeBlock


class HTTPErrorResponse(_Closed):
    """Rejections raised by a handler: validation before inference, a capability the active profile lacks (422),
    a runtime that is loading or suspended (503), a protocol failure (502)."""

    detail: Any = Field(description="A message, or FastAPI's list of body-validation errors.")


class ContractErrorResponse(_Closed):
    """Rejections of the negotiated `Deqio-Contract` body (unknown version, duplicate keys, missing policy)."""

    error: dict[str, Any] = Field(description="`code` and `message`, plus contract-specific fields.")


_COMPONENT_MODELS: tuple[type[BaseModel], ...] = (
    DecisionResponse, SharedResponse, SystemOneResponse, HTTPErrorResponse, ContractErrorResponse,
)
_ERROR_RESPONSES = {
    "400": "Rejected before inference (blank question, missing or duplicate ids, too many options or decisions).",
    "422": "Validation error, or a request the active profile cannot serve (capability, context overflow, unsupported question type).",
    "502": "The engine answered with a protocol error.",
    "503": "The runtime is still loading, or inference is suspended while a benchmark owns it.",
}
_CONTRACT_ERROR_RESPONSE = "The negotiated `Deqio-Contract` body was rejected before inference."


def _component_schemas() -> dict[str, Any]:
    """JSON Schemas (2020-12, as OpenAPI 3.1 uses) of every documented response model, flattened into components."""
    components: dict[str, Any] = {}
    for model in _COMPONENT_MODELS:
        schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
        for name, definition in (schema.pop("$defs", None) or {}).items():
            components.setdefault(name, definition)
        components[model.__name__] = schema
    return components


def _json_response(schema_name: str, description: str) -> dict[str, Any]:
    return {"description": description, "content": {"application/json": {"schema": {"$ref": f"#/components/schemas/{schema_name}"}}}}


def build_openapi(app: FastAPI) -> dict[str, Any]:
    """The OpenAPI 3.1 document of ``app`` with the decision routes' response contracts added."""
    document = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=OPENAPI_VERSION,
        description=(
            "Deqio — decision in, probabilities out. Portable routes (`/v1/noul`, `/v1/choice`, `/v1/shared`) return the "
            "cross-model Deqio contract with Decision Provenance / Attestation (`provenance`, including `provider`, "
            "`remote_request_id` and `cost_estimate`); native routes (`/v1/score`, `/v1/multi`, `/v1/act`, `/v1/soam`, "
            "`/v1/systemone`) return the active profile's native System One reply plus a `deqio` block. "
            "`/v1/decision` is the `/v1/choice` alias. The Python SDK `deqio-client` and `deqio decide` call these routes."
        ),
        routes=app.routes,
    )
    components = document.setdefault("components", {}).setdefault("schemas", {})
    for name, schema in _component_schemas().items():
        components[name] = schema
    paths = document.get("paths", {})
    for path, schema_name, summary, description in DECISION_ROUTES:
        operation = paths.get(path, {}).get("post")
        if operation is None:
            raise RuntimeError(f"{path} is not a POST route of the app; the decision routes changed")
        operation["summary"] = summary
        operation["description"] = description
        operation["operationId"] = "decide_" + path.rsplit("/", 1)[-1]
        operation["tags"] = ["decisions"]
        responses = operation.setdefault("responses", {})
        responses["200"] = _json_response(schema_name, "The decision.")
        for status, text in _ERROR_RESPONSES.items():
            responses[status] = _json_response("HTTPErrorResponse", text)
        if path in CONTRACT_ROUTES:
            responses["422"] = {
                "description": _ERROR_RESPONSES["422"] + " " + _CONTRACT_ERROR_RESPONSE,
                "content": {"application/json": {"schema": {"oneOf": [
                    {"$ref": "#/components/schemas/HTTPErrorResponse"},
                    {"$ref": "#/components/schemas/ContractErrorResponse"},
                ]}}},
            }
            parameters = [parameter for parameter in operation.get("parameters", []) if parameter.get("name") != CONTRACT_HEADER]
            parameters.append({
                "name": CONTRACT_HEADER,
                "in": "header",
                "required": False,
                "schema": {"type": "string", "enum": [CONTRACT_VALUE]},
                "description": (
                    "Negotiates the input-completeness contract: the exact body bytes are hashed, duplicate JSON keys are "
                    "rejected, explicit stable ids and an `input_policy` are required, and the response carries "
                    "`input_receipt`/`usage` with attestation schema v2. Unknown versions are rejected with 422."
                ),
            })
            operation["parameters"] = parameters
        operation["responses"] = {status: responses[status] for status in sorted(responses, key=lambda code: (len(code), code))}
    document.setdefault("tags", [])
    if not any(tag.get("name") == "decisions" for tag in document["tags"]):
        document["tags"].append({"name": "decisions", "description": "The eight decision routes and the `/v1/decision` alias."})
    return document


def install(app: FastAPI) -> None:
    """Serve the augmented document from ``GET /openapi.json`` (and therefore ``/docs``)."""

    def openapi() -> dict[str, Any]:
        if not app.openapi_schema:
            app.openapi_schema = build_openapi(app)
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]


def render(app: FastAPI) -> str:
    """The document as the JSON text ``deqio openapi`` writes and ``docs/openapi.json`` holds."""
    return json.dumps(build_openapi(app), indent=2, ensure_ascii=False) + "\n"


def response_schema_for(path: str) -> str:
    for route, schema_name, _summary, _description in DECISION_ROUTES:
        if route == path:
            return schema_name
    raise KeyError(f"{path} is not a documented decision route")


def validate_response(path: str, payload: dict[str, Any]) -> BaseModel:
    """Check a real 200 response of ``path`` against the documented shape; raises ``pydantic.ValidationError``."""
    model = {"DecisionResponse": DecisionResponse, "SharedResponse": SharedResponse, "SystemOneResponse": SystemOneResponse}[
        response_schema_for(path)
    ]
    return model.model_validate(copy.deepcopy(payload))
