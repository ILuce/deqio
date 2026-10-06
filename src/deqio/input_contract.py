from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


INPUT_COMPLETENESS_V1 = "input-completeness-v1"
INPUT_RECEIPT_SCHEMA = "deqio.input.v1"
INPUT_ATTESTATION_SCHEMA_VERSION = 2
ENGINE_PAYLOAD_BUILDER_REVISION = "deqio-systemone-payload-v1"


class DuplicateJSONKeyError(ValueError):
    pass


class UnsupportedInputContractError(ValueError):
    pass


@dataclass(frozen=True)
class InputContractContext:
    version: str
    request_body_sha256: str
    require_complete: bool
    overflow: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_json_without_duplicate_keys(raw_body: bytes) -> None:
    """Reject duplicate object keys anywhere in a negotiated JSON request."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateJSONKeyError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        json.loads(raw_body.decode("utf-8"), object_pairs_hook=object_pairs)
    except UnicodeDecodeError as error:
        raise ValueError("request body must be UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON request body: {error.msg}") from error


def parse_contract_header(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    normalized = value.strip().lower()
    if normalized != INPUT_COMPLETENESS_V1:
        raise UnsupportedInputContractError(normalized)
    return normalized


def decision_input_stub(*, decision_id: str, option_ids: list[str], components: list[str]) -> dict[str, Any]:
    return {
        "decision_id": decision_id,
        "option_ids": option_ids,
        "components": components,
        "rendered_input_sha256": None,
        "prepared_tokens": None,
        "consumed_tokens": None,
        "prepared_token_ids_sha256": None,
        "consumed_token_ids_sha256": None,
        "masked_input_tokens": None,
        "cache_reused": None,
    }


def unknown_input_receipt(
    *,
    context: InputContractContext,
    runtime_identity: dict[str, Any],
    engine_payload_sha256: str | None,
    input_limit_tokens: int,
    decision_inputs: list[dict[str, Any]],
    input_tokens: int | None,
    input_tokens_source: str,
) -> dict[str, Any]:
    return {
        "schema": INPUT_RECEIPT_SCHEMA,
        "status": "unknown",
        "verification_point": "deqio_engine_boundary",
        "request_body_sha256": context.request_body_sha256,
        "engine_payload_sha256": engine_payload_sha256,
        "runtime_instance_id": runtime_identity.get("runtime_instance_id"),
        "builder_revision": ENGINE_PAYLOAD_BUILDER_REVISION,
        "tokenizer_sha256": None,
        "input_limit_tokens": input_limit_tokens,
        "reserved_tokens": None,
        "overflow_policy": context.overflow,
        "truncated": None,
        "dropped_components": [],
        "decision_inputs": decision_inputs,
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_source": input_tokens_source,
        },
        "reason": "backend_model_input_not_instrumented",
    }
