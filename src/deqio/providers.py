"""Decision providers (D11): one contract for the local sidecars and remote decision APIs.

A provider answers a System One request (``state`` + ``questions``) and reports who answered.
``local`` is the sidecar runtime selected from the catalog (``SystemOneRuntime``).  Remote
providers are HTTP APIs behind an opt-in flag; their wire mapping lives in one place here so a
recorded response (``python -m deqio.provider_record``) can be replayed by the contract tests.

Deqio provider wire v1 (the shape both remote adapters send and expect until recordings show
otherwise -- see docs/providers.md): request ``{"model", "state", "questions"}``; response
``{"id", "answers": {question_id: {...}}, "usage": {"input_tokens": n}}``.  The request id is the
response ``id`` or the ``x-request-id`` header; cost is estimated from ``usage.input_tokens`` and
the provider's published input price and is reported as an estimate, never as a bill.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

WIRE_VERSION = "deqio-provider-wire-v1"
LOCAL_PROVIDER = "local"
TRUE_VALUES = {"1", "true", "yes", "on"}


class ProviderError(RuntimeError):
    """A provider could not be used; never answered with another provider instead."""


class ProviderDisabledError(ProviderError):
    """The provider is behind a flag that is not set."""


class ProviderResponseError(ProviderError):
    """The remote API answered, but not with a usable decision."""

    def __init__(self, status: int | None, detail: str) -> None:
        self.status = status
        super().__init__(detail)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    kind: str  # "local" | "remote"
    default_base_url: str | None = None
    path: str = "/v1/decisions"
    default_model: str = ""
    api_key_env: str = ""
    flag_env: str = ""
    base_url_env: str = ""
    usd_per_million_input_tokens: float | None = None
    price_source: str = ""
    extra_headers: Mapping[str, str] = field(default_factory=dict)


PROVIDERS: dict[str, ProviderSpec] = {
    LOCAL_PROVIDER: ProviderSpec(name=LOCAL_PROVIDER, kind="local"),
    "openai-decisions": ProviderSpec(
        name="openai-decisions",
        kind="remote",
        default_base_url="https://api.openai.com",
        path="/v1/decisions",
        default_model="gpt-6-luna",
        api_key_env="OPENAI_API_KEY",
        flag_env="DEQIO_PROVIDER_OPENAI_DECISIONS",
        base_url_env="OPENAI_BASE_URL",
        usd_per_million_input_tokens=0.10,
        price_source="OpenAI Decisions list price, roadmap 2026-10-08 ($0.10 per 1M input tokens)",
    ),
    "typesafe-jev": ProviderSpec(
        name="typesafe-jev",
        kind="remote",
        default_base_url=None,  # no public default: TYPESAFE_JEV_BASE_URL is required
        path="/v1/decisions",
        default_model="jev",
        api_key_env="TYPESAFE_JEV_API_KEY",
        flag_env="DEQIO_PROVIDER_TYPESAFE_JEV",
        base_url_env="TYPESAFE_JEV_BASE_URL",
        usd_per_million_input_tokens=0.042,
        price_source="TypeSafe Jev list price, roadmap 2026-10-08 ($0.042 per 1M input tokens)",
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


def flag_enabled(spec: ProviderSpec, environ: Mapping[str, str] | None = None) -> bool:
    if spec.kind == "local":
        return True
    env = os.environ if environ is None else environ
    return str(env.get(spec.flag_env, "")).strip().lower() in TRUE_VALUES


def require_enabled(spec: ProviderSpec, environ: Mapping[str, str] | None = None) -> None:
    if not flag_enabled(spec, environ):
        raise ProviderDisabledError(
            f"Provider {spec.name!r} is behind the {spec.flag_env}=1 flag; set it to use this provider"
        )


def local_provider_descriptor(engine: str | None = None) -> dict[str, Any]:
    descriptor: dict[str, Any] = {"name": LOCAL_PROVIDER, "kind": "local"}
    if engine:
        descriptor["engine"] = str(engine)
    return descriptor


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


Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, dict[str, str], bytes]]


def _urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, dict[str, str], bytes]:
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - provider URL from configuration
            return int(response.status), {k.lower(): v for k, v in response.headers.items()}, response.read()
    except HTTPError as error:
        payload = error.read() if hasattr(error, "read") else b""
        return int(error.code), {k.lower(): v for k, v in (error.headers.items() if error.headers else [])}, payload
    except URLError as error:
        raise ProviderError(f"Provider request to {url} failed: {error.reason}") from error


class RemoteDecisionProvider:
    """HTTP adapter for a remote decision API using the provider wire v1."""

    def __init__(
        self,
        spec: ProviderSpec,
        *,
        model: str = "",
        base_url: str = "",
        api_key: str | None = None,
        timeout: float = 60.0,
        transport: Transport | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if spec.kind != "remote":
            raise ProviderError(f"Provider {spec.name!r} is not a remote provider")
        env = os.environ if environ is None else environ
        require_enabled(spec, env)
        self.spec = spec
        self.model = str(model or spec.default_model)
        resolved_base = str(base_url or env.get(spec.base_url_env, "") or spec.default_base_url or "").rstrip("/")
        if not resolved_base:
            raise ProviderError(
                f"Provider {spec.name!r} has no endpoint; set {spec.base_url_env} to its base URL"
            )
        self.base_url = resolved_base
        key = api_key if api_key is not None else env.get(spec.api_key_env, "")
        if not key:
            raise ProviderError(f"Provider {spec.name!r} needs {spec.api_key_env} in the environment")
        self._api_key = str(key)
        self.timeout = float(timeout)
        self._transport = transport or _urllib_transport

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}{self.spec.path}"

    def describe(self) -> dict[str, Any]:
        """Identity for provenance: never the key."""
        return {
            "name": self.spec.name,
            "kind": "remote",
            "model": self.model,
            "endpoint": self.endpoint,
            "wire": WIRE_VERSION,
            "usd_per_million_input_tokens": self.spec.usd_per_million_input_tokens,
            "price_source": self.spec.price_source,
        }

    def build_request(
        self, state: Any, questions: dict[str, Any], request_options: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        payload: dict[str, Any] = {"model": self.model, "state": state, "questions": questions}
        if request_options:
            reserved = {"model", "state", "questions"}
            overlap = reserved.intersection(request_options)
            if overlap:
                raise ProviderError("Provider request options may not override: " + ", ".join(sorted(overlap)))
            payload.update(request_options)
        headers = {
            "content-type": "application/json",
            "accept": "application/json",
            "authorization": f"Bearer {self._api_key}",
            "user-agent": "deqio-provider/" + WIRE_VERSION,
        }
        headers.update(dict(self.spec.extra_headers))
        return self.endpoint, headers, payload

    def parse_response(
        self, status: int, headers: Mapping[str, str], body: bytes | str | dict[str, Any], *, latency_ms: float = 0.0
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
        request_id = None
        if isinstance(data, dict) and data.get("id") not in (None, ""):
            request_id = str(data["id"])
        elif lower_headers.get("x-request-id"):
            request_id = lower_headers["x-request-id"]
        if status != 200:
            detail = data.get("error") if isinstance(data, dict) else data
            raise ProviderResponseError(status, f"{self.spec.name} returned HTTP {status}: {json.dumps(detail)[:300]}")
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict) or not data["answers"]:
            raise ProviderResponseError(status, f"{self.spec.name} response has no answers object ({WIRE_VERSION})")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        input_tokens: int | None = None
        try:
            if usage.get("input_tokens") is not None and int(usage["input_tokens"]) > 0:
                input_tokens = int(usage["input_tokens"])
        except (TypeError, ValueError):
            input_tokens = None
        return ProviderResponse(
            answers=dict(data["answers"]),
            remote_request_id=request_id,
            input_tokens=input_tokens,
            latency_ms=float(latency_ms),
            raw=data,
            status=status,
        )

    def request(
        self, state: Any, questions: dict[str, Any], request_options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], ProviderResponse]:
        url, headers, payload = self.build_request(state, questions, request_options)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        status, response_headers, response_body = self._transport(url, headers, body, self.timeout)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return payload, self.parse_response(status, response_headers, response_body, latency_ms=latency_ms)


def cassette_from_exchange(
    provider: RemoteDecisionProvider,
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
        "wire": WIRE_VERSION,
        "synthetic": synthetic,
        "request": {"url": provider.endpoint, "body": payload},
        "response": {"status": int(status), "headers": safe_headers, "json": decoded},
    }
