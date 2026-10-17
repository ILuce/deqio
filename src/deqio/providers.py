"""Remote decision providers (D11, wires corrected in D11b / 0.6.1).

A remote provider answers a System One request (``state`` + keyed ``questions``) over HTTPS and
names itself in provenance (``provider``, ``remote_request_id``, ``cost_estimate``).  Each provider
has its own documented wire; the mapping to and from Deqio's native System One shape lives only
here so a recorded exchange (``python -m deqio.provider_record``) can be replayed by the contract
tests in ``tests/test_providers.py``:

* ``openai-decisions`` (``openai-decisions-v1``): ``POST https://api.openai.com/v1/decisions`` with
  ``{"model", "input", "questions": [{"type", "name", "instructions", "choices" | "levels"}]}``;
  yes/no questions are ``predicate``; the reply is ``{"answers": [...], "model", "usage"}`` where a
  choice answer carries ``choice``, ``probabilities: [{"value", "probability"}]`` and
  ``confidence``, a predicate ``probability``, a score ``score`` and labelled probabilities, and a
  ``refusal`` is an error for Deqio (never a silent empty answer).
* ``typesafe-jev`` (``typesafe-systemone-v1``): ``POST https://api.typesafe.ai/v1/systemone`` with
  ``{"state", "model", "questions": {id: {"type", "instructions", "criteria"}}}`` -- the native
  System One wire -- answered by ``{"model", "answers": {id: ...}, "usage"}``.

Keys are never read here: the caller (``deqio.remote_runtime``) takes them from the OS keychain
(``deqio.credentials``).  Cost is an estimate from the reported ``usage.input_tokens`` and the
published input price, never a bill; ``amount`` stays ``None`` without usage.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import __version__

LOCAL_PROVIDER = "local"
OPENAI_WIRE = "openai-decisions-v1"
TYPESAFE_WIRE = "typesafe-systemone-v1"
QUESTION_TYPES = ("choice", "noul", "score")


class ProviderError(RuntimeError):
    """A provider could not be used; never answered with another provider instead."""


class ProviderResponseError(ProviderError):
    """The remote API answered, but not with a usable decision."""

    def __init__(self, status: int | None, detail: str) -> None:
        self.status = status
        super().__init__(detail)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    kind: str  # "local" | "remote"
    wire: str = ""
    default_endpoint: str = ""
    default_model: str = ""
    # The vendor SDKs' conventional variable; only `python -m deqio.provider_record --key env` reads it.
    api_key_env: str = ""
    usd_per_million_input_tokens: float | None = None
    price_source: str = ""
    docs_url: str = ""


PROVIDERS: dict[str, ProviderSpec] = {
    LOCAL_PROVIDER: ProviderSpec(name=LOCAL_PROVIDER, kind="local"),
    "openai-decisions": ProviderSpec(
        name="openai-decisions",
        kind="remote",
        wire=OPENAI_WIRE,
        default_endpoint="https://api.openai.com/v1/decisions",
        default_model="gpt-6-luna",
        api_key_env="OPENAI_API_KEY",
        usd_per_million_input_tokens=0.10,
        price_source="OpenAI pricing page, gpt-6-luna standard tier short context, read 2026-10-14 ($0.10 per 1M input tokens; long context $0.20)",
        docs_url="https://developers.openai.com/api/docs/guides/decisions",
    ),
    "typesafe-jev": ProviderSpec(
        name="typesafe-jev",
        kind="remote",
        wire=TYPESAFE_WIRE,
        default_endpoint="https://api.typesafe.ai/v1/systemone",
        default_model="jev-latest",
        api_key_env="TYPESAFE_API_KEY",
        usd_per_million_input_tokens=0.042,
        price_source="TypeSafe models page, read 2026-10-14 ($0.042 per 1M input tokens; output free)",
        docs_url="https://docs.typesafe.ai/api",
    ),
}


def provider_spec(name: str) -> ProviderSpec:
    key = str(name or LOCAL_PROVIDER).strip().lower()
    spec = PROVIDERS.get(key)
    if spec is None:
        raise ProviderError(
            f"Unknown decision provider {name!r}; known providers: " + ", ".join(sorted(PROVIDERS))
        )
    return spec


def cost_estimate(spec: ProviderSpec, input_tokens: int | None, *, scope: str = "decision") -> dict[str, Any]:
    """Estimated input cost of one call; ``amount`` is None when the API reported no token usage."""
    rate = spec.usd_per_million_input_tokens
    estimate: dict[str, Any] = {
        "currency": "USD",
        "amount": None,
        "basis": "input_tokens",
        "scope": scope,
        "input_tokens": input_tokens,
        "usd_per_million_input_tokens": rate,
        "price_source": spec.price_source,
        "estimate": True,
    }
    if rate is not None and isinstance(input_tokens, int) and input_tokens > 0:
        estimate["amount"] = round(input_tokens * rate / 1_000_000.0, 9)
    else:
        estimate["basis"] = "input_tokens_unavailable" if rate is not None else "price_unknown"
    return estimate


@dataclass
class ProviderResponse:
    answers: dict[str, Any]
    remote_request_id: str | None
    input_tokens: int | None
    latency_ms: float
    raw: dict[str, Any]
    status: int
    model: str | None = None


Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, dict[str, str], bytes]]


def _urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, dict[str, str], bytes]:
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - provider URL from the catalog
            return int(response.status), {k.lower(): v for k, v in response.headers.items()}, response.read()
    except HTTPError as error:
        payload = error.read() if hasattr(error, "read") else b""
        return int(error.code), {k.lower(): v for k, v in (error.headers.items() if error.headers else [])}, payload
    except URLError as error:
        raise ProviderError(f"Provider request to {url} failed: {error.reason}") from error


# --- System One question helpers ------------------------------------------------------------


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _question_type(question_id: str, question: Any) -> str:
    if not isinstance(question, dict):
        raise ProviderError(f"Question {question_id!r} must be an object")
    kind = str(question.get("type", "choice"))
    if kind not in QUESTION_TYPES:
        raise ProviderError(f"Question {question_id!r}: remote providers support {', '.join(QUESTION_TYPES)}, not {kind!r}")
    return kind


def _choice_criteria(question_id: str, question: dict[str, Any]) -> list[tuple[str, Any]]:
    criteria = question.get("criteria")
    if isinstance(criteria, dict) and criteria:
        return [(str(key), value) for key, value in criteria.items()]
    if isinstance(criteria, list) and criteria:
        return [(str(key), None) for key in criteria]
    raise ProviderError(f"Question {question_id!r}: a choice needs a criteria mapping of option keys to descriptions")


def _score_levels(question_id: str, question: dict[str, Any]) -> list[Any]:
    criteria = question.get("criteria")
    if isinstance(criteria, list) and len(criteria) >= 2:
        return list(criteria)
    raise ProviderError(f"Question {question_id!r}: a score needs an ordered criteria list of at least two levels")


# --- the two wires ----------------------------------------------------------------------------


def _openai_payload(model: str, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
    wire_questions: list[dict[str, Any]] = []
    for question_id, question in questions.items():
        kind = _question_type(question_id, question)
        instructions = _text(question.get("instructions"))
        item: dict[str, Any] = {"type": kind, "name": str(question_id), "instructions": instructions}
        if kind == "choice":
            item["choices"] = [{"value": key, "description": _text(value)} for key, value in _choice_criteria(question_id, question)]
        elif kind == "noul":
            item["type"] = "predicate"
            criteria = question.get("criteria")
            if isinstance(criteria, dict) and criteria:
                # The Decisions API has no yes/no descriptions; keep them visible in the instructions.
                notes = "; ".join(f"{key}: {_text(value)}" for key, value in criteria.items() if _text(value))
                if notes:
                    item["instructions"] = (instructions + " (" + notes + ")").strip()
        else:
            item["levels"] = [{"label": str(index), "description": _text(level)} for index, level in enumerate(_score_levels(question_id, question))]
        wire_questions.append(item)
    return {"model": model, "input": state if isinstance(state, str) else json.dumps(state, ensure_ascii=False), "questions": wire_questions}


def _openai_answers(spec_name: str, data: dict[str, Any], questions: Mapping[str, Any] | None) -> dict[str, Any]:
    wire_answers = data.get("answers")
    if not isinstance(wire_answers, list):
        raise ProviderResponseError(200, f"{spec_name} response has no answers array ({OPENAI_WIRE})")
    by_name: dict[str, dict[str, Any]] = {}
    for item in wire_answers:
        if isinstance(item, dict) and item.get("name") not in (None, ""):
            by_name[str(item["name"])] = item
    expected = list(questions) if questions else list(by_name)
    answers: dict[str, Any] = {}
    for question_id in expected:
        item = by_name.get(str(question_id))
        if item is None:
            raise ProviderResponseError(200, f"{spec_name} returned no answer for question {question_id!r}")
        kind = str(item.get("type", ""))
        if kind == "refusal":
            raise ProviderResponseError(200, f"{spec_name} refused question {question_id!r} (type refusal); Deqio does not answer it otherwise")
        if kind == "predicate":
            answers[question_id] = {"type": "noul", "noul": _probability(spec_name, item.get("probability"), question_id)}
        elif kind == "choice":
            probabilities = {
                str(entry.get("value")): _probability(spec_name, entry.get("probability"), question_id)
                for entry in item.get("probabilities") or []
                if isinstance(entry, dict)
            }
            answer: dict[str, Any] = {"type": "choice", "choice": item.get("choice"), "probabilities": probabilities}
            if item.get("confidence") is not None:
                answer["confidence"] = _probability(spec_name, item.get("confidence"), question_id)
            answers[question_id] = answer
        elif kind == "score":
            probabilities = {}
            for entry in item.get("probabilities") or []:
                if not isinstance(entry, dict):
                    continue
                label = entry.get("label", entry.get("value"))
                probabilities[str(label)] = _probability(spec_name, entry.get("probability"), question_id)
            answer = {"type": "score", "score": item.get("score"), "probabilities": probabilities}
            if item.get("confidence") is not None:
                answer["confidence"] = _probability(spec_name, item.get("confidence"), question_id)
            question = questions.get(question_id) if questions else None
            if isinstance(question, dict) and isinstance(question.get("criteria"), list):
                answer["legend"] = {str(index): _text(level) for index, level in enumerate(question["criteria"])}
            answers[question_id] = answer
        else:
            raise ProviderResponseError(200, f"{spec_name} returned an unknown answer type {kind!r} for {question_id!r}")
    return answers


def _probability(spec_name: str, value: Any, question_id: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ProviderResponseError(200, f"{spec_name} returned a non-numeric probability for {question_id!r}") from error
    if not 0.0 <= number <= 1.0 or number != number:
        raise ProviderResponseError(200, f"{spec_name} returned a probability outside [0, 1] for {question_id!r}")
    return number


def _typesafe_payload(model: str, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
    for question_id, question in questions.items():
        _question_type(question_id, question)
    return {"state": state, "model": model, "questions": questions}


def _typesafe_answers(spec_name: str, data: dict[str, Any], questions: Mapping[str, Any] | None) -> dict[str, Any]:
    answers = data.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise ProviderResponseError(200, f"{spec_name} response has no answers object ({TYPESAFE_WIRE})")
    for question_id in (questions or answers):
        if not isinstance(answers.get(str(question_id)), dict):
            raise ProviderResponseError(200, f"{spec_name} returned no answer for question {question_id!r}")
    return dict(answers)


class RemoteDecisionProvider:
    """HTTP adapter for one remote decision API in its documented wire."""

    def __init__(
        self,
        spec: ProviderSpec,
        *,
        api_key: str | None,
        model: str = "",
        endpoint: str = "",
        timeout: float = 60.0,
        transport: Transport | None = None,
    ) -> None:
        if spec.kind != "remote":
            raise ProviderError(f"Provider {spec.name!r} is not a remote provider")
        if spec.wire not in {OPENAI_WIRE, TYPESAFE_WIRE}:
            raise ProviderError(f"Provider {spec.name!r} has no wire mapping")
        if not str(api_key or "").strip():
            raise ProviderError(f"Provider {spec.name!r} needs an API key; run deqio models setup for its remote profile")
        self.spec = spec
        self.model = str(model or spec.default_model)
        self._endpoint = str(endpoint or spec.default_endpoint).strip()
        if not self._endpoint.startswith(("https://", "http://")):
            raise ProviderError(f"Provider {spec.name!r} endpoint must be an http(s) URL, got {self._endpoint!r}")
        self._api_key = str(api_key)
        self.timeout = float(timeout)
        self._transport = transport or _urllib_transport

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def describe(self) -> dict[str, Any]:
        """Identity for provenance: never the key."""
        return {
            "name": self.spec.name,
            "kind": "remote",
            "model": self.model,
            "endpoint": self.endpoint,
            "wire": self.spec.wire,
            "usd_per_million_input_tokens": self.spec.usd_per_million_input_tokens,
            "price_source": self.spec.price_source,
        }

    def build_request(
        self, state: Any, questions: dict[str, Any], request_options: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        if not isinstance(questions, dict) or not questions:
            raise ProviderError("questions must be a non-empty object")
        if request_options:
            raise ProviderError(
                f"Provider {self.spec.name!r} takes no System One request options; got: " + ", ".join(sorted(request_options))
            )
        if self.spec.wire == OPENAI_WIRE:
            payload = _openai_payload(self.model, state, questions)
        else:
            payload = _typesafe_payload(self.model, state, questions)
        headers = {
            "content-type": "application/json",
            "accept": "application/json",
            "authorization": f"Bearer {self._api_key}",
            "user-agent": f"deqio/{__version__} ({self.spec.wire})",
        }
        return self.endpoint, headers, payload

    def parse_response(
        self,
        status: int,
        headers: Mapping[str, str],
        body: bytes | str | dict[str, Any],
        *,
        latency_ms: float = 0.0,
        questions: Mapping[str, Any] | None = None,
    ) -> ProviderResponse:
        if isinstance(body, dict):
            data: Any = body
        else:
            text = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
            try:
                data = json.loads(text) if text.strip() else {}
            except json.JSONDecodeError as error:
                raise ProviderResponseError(status, f"{self.spec.name} returned non-JSON HTTP {status}: {text[:200]!r}") from error
        lower_headers = {str(k).lower(): str(v) for k, v in dict(headers).items()}
        request_id = lower_headers.get("x-request-id") or None
        if request_id is None and isinstance(data, dict) and data.get("id") not in (None, ""):
            request_id = str(data["id"])
        if status != 200:
            detail = data.get("error", data.get("detail", data)) if isinstance(data, dict) else data
            raise ProviderResponseError(status, f"{self.spec.name} returned HTTP {status}: {json.dumps(detail)[:300]}")
        if not isinstance(data, dict):
            raise ProviderResponseError(status, f"{self.spec.name} response is not a JSON object")
        if self.spec.wire == OPENAI_WIRE:
            if not isinstance(data.get("answers"), list):
                raise ProviderResponseError(status, f"{self.spec.name} response has no answers array ({OPENAI_WIRE})")
            answers = _openai_answers(self.spec.name, data, questions)
        else:
            if not isinstance(data.get("answers"), dict):
                raise ProviderResponseError(status, f"{self.spec.name} response has no answers object ({TYPESAFE_WIRE})")
            answers = _typesafe_answers(self.spec.name, data, questions)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        input_tokens: int | None = None
        try:
            if usage.get("input_tokens") is not None and int(usage["input_tokens"]) > 0:
                input_tokens = int(usage["input_tokens"])
        except (TypeError, ValueError):
            input_tokens = None
        model = str(data["model"]) if data.get("model") not in (None, "") else None
        return ProviderResponse(
            answers=answers,
            remote_request_id=request_id,
            input_tokens=input_tokens,
            latency_ms=float(latency_ms),
            raw=data,
            status=status,
            model=model,
        )

    def request(
        self, state: Any, questions: dict[str, Any], request_options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], ProviderResponse]:
        url, headers, payload = self.build_request(state, questions, request_options)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        status, response_headers, response_body = self._transport(url, headers, body, self.timeout)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return payload, self.parse_response(status, response_headers, response_body, latency_ms=latency_ms, questions=questions)


def cassette_from_exchange(
    provider: RemoteDecisionProvider,
    state: Any,
    questions: dict[str, Any],
    payload: dict[str, Any],
    status: int,
    headers: Mapping[str, str],
    body: bytes,
    *,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Recording of one exchange for the contract tests; secrets are never written."""
    safe_headers = {
        str(k).lower(): str(v)
        for k, v in dict(headers).items()
        if str(k).lower() in {"content-type", "x-request-id", "openai-processing-ms"}
    }
    try:
        decoded: Any = json.loads(body.decode("utf-8")) if body else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        decoded = body.decode("utf-8", "replace")
    return {
        "provider": provider.spec.name,
        "wire": provider.spec.wire,
        "synthetic": synthetic,
        "request": {"url": provider.endpoint, "system_one": {"state": state, "questions": questions}, "body": payload},
        "response": {"status": int(status), "headers": safe_headers, "json": decoded},
    }
