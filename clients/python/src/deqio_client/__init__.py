"""deqio-client — a dependency-free Python SDK for the Deqio decision server (0.6.2, D13).

One class, ``DeqioClient``, covers the public decision routes of a running ``deqio serve``::

    from deqio_client import DeqioClient

    client = DeqioClient("http://127.0.0.1:8787")
    answer = client.noul("Payment latency is elevated for some customers.", "Is checkout degraded?")
    answer["decision"], answer["probabilities"], answer["provenance"]["provider"]

Portable routes: ``noul`` (``POST /v1/noul``), ``choice`` (``/v1/choice``; ``decision`` is the ``/v1/decision``
alias) and ``shared`` (``/v1/shared``).  Native typed routes: ``score``, ``multi``, ``act`` (``/v1/score``,
``/v1/multi``, ``/v1/act``) and the State-Once-Ask-Many routes ``soam`` (``/v1/soam``) and ``systemone``
(``/v1/systemone``).  Every method returns the server's JSON response unchanged — the same fields the OpenAPI
document (``GET /openapi.json``, ``deqio openapi``) describes, including ``provenance`` (``provider``,
``remote_request_id``, ``cost_estimate``) — and raises ``DeqioError`` on any non-2xx answer with the status and
the server's ``detail``.  Nothing is retried, cached or defaulted silently; a server that cannot be reached raises
``DeqioConnectionError``.

The package has no dependencies (``urllib`` only) and does not import ``deqio``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Sequence

__version__ = "0.6.2"
__all__ = [
    "CONTRACT_INPUT_COMPLETENESS_V1",
    "DECISION_ROUTES",
    "DeqioClient",
    "DeqioConnectionError",
    "DeqioError",
    "__version__",
    "normalize_options",
]

CONTRACT_INPUT_COMPLETENESS_V1 = "input-completeness-v1"
USER_AGENT = f"deqio-client/{__version__}"

# The decision routes this SDK covers, in the server's public order (``/v1/decision`` is the ``/v1/choice`` alias).
DECISION_ROUTES = (
    "/v1/noul",
    "/v1/choice",
    "/v1/decision",
    "/v1/shared",
    "/v1/score",
    "/v1/multi",
    "/v1/act",
    "/v1/soam",
    "/v1/systemone",
)

State = Any  # str | dict | list — sent as given; the server renders structured state deterministically
Transport = Callable[[str, str, Mapping[str, str], "bytes | None", float], "tuple[int, Mapping[str, str], bytes]"]


class DeqioError(Exception):
    """The server answered with a non-2xx status.  ``status`` and ``detail`` carry the server's own rejection."""

    def __init__(self, status: int, detail: Any, *, path: str, body: Any = None) -> None:
        self.status = status
        self.detail = detail
        self.path = path
        self.body = body
        super().__init__(f"HTTP {status} from {path}: {_describe_detail(detail)}")


class DeqioConnectionError(Exception):
    """The server could not be reached (connection refused, DNS, timeout)."""


def _describe_detail(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    try:
        return json.dumps(detail, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(detail)


def normalize_options(options: Mapping[str, str | None] | Sequence[Mapping[str, Any]] | Sequence[str]) -> list[dict[str, str]]:
    """Return ``[{"id": ..., "description": ...}, ...]`` from a mapping ``{"id": "description"}``, a sequence of
    such objects, or a sequence of ids (the description then repeats the id).  Nothing else is accepted."""
    if isinstance(options, Mapping):
        return [{"id": str(key), "description": str(key if value is None else value)} for key, value in options.items()]
    if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
        raise ValueError("options must be a mapping {id: description} or a sequence of {'id', 'description'} objects or ids")
    normalized: list[dict[str, str]] = []
    for option in options:
        if isinstance(option, Mapping):
            if "id" not in option:
                raise ValueError(f"option {option!r} has no 'id'")
            normalized.append({"id": str(option["id"]), "description": str(option.get("description", option["id"]))})
        elif isinstance(option, str):
            normalized.append({"id": option, "description": option})
        else:
            raise ValueError(f"option {option!r} is neither a mapping nor an id")
    return normalized


def _urllib_transport(method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> tuple[int, Mapping[str, str], bytes]:
    request = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), dict(error.headers.items()), error.read()
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        raise DeqioConnectionError(f"cannot reach {url}: {error}") from error


class DeqioClient:
    """HTTP client for one Deqio server.

    ``base_url`` is the server's root (``deqio serve`` prints it; default ``http://127.0.0.1:8787``).  ``timeout``
    is the per-request socket timeout in seconds.  ``transport`` replaces the ``urllib`` call (tests inject a
    recorder); it receives ``(method, url, headers, body_bytes, timeout)`` and returns ``(status, headers, body_bytes)``.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8787", *, timeout: float = 60.0, transport: Transport | None = None) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self._transport = transport or _urllib_transport

    # ---- transport ----

    def request(self, method: str, path: str, payload: Any = None, *, headers: Mapping[str, str] | None = None) -> Any:
        """One HTTP exchange; returns the decoded JSON body of a 2xx answer, raises ``DeqioError`` otherwise."""
        body = None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        sent = {"accept": "application/json", "user-agent": USER_AGENT}
        if body is not None:
            sent["content-type"] = "application/json"
        sent.update(headers or {})
        status, _response_headers, raw = self._transport(method, self.base_url + path, sent, body, self.timeout)
        text = raw.decode("utf-8", errors="replace") if raw else ""
        try:
            decoded = json.loads(text) if text else None
        except ValueError:
            decoded = text
        if not 200 <= status < 300:
            detail = decoded.get("detail", decoded.get("error", decoded)) if isinstance(decoded, dict) else decoded
            raise DeqioError(status, detail, path=path, body=decoded)
        return decoded

    def post(self, path: str, payload: Any, *, contract: bool = False) -> Any:
        headers = {"Deqio-Contract": CONTRACT_INPUT_COMPLETENESS_V1} if contract else {}
        return self.request("POST", path, payload, headers=headers)

    def get(self, path: str) -> Any:
        return self.request("GET", path)

    # ---- server state ----

    def health(self) -> dict[str, Any]:
        """``GET /health`` — the active engine/model/backend and runtime instance."""
        return self.get("/health")

    # ---- portable Deqio routes ----

    def noul(self, state: State, question: str, *, id: str | None = None, mode: str | None = None,
             input_policy: Mapping[str, Any] | None = None, contract: bool = False) -> dict[str, Any]:
        """``POST /v1/noul`` — yes / no over one state."""
        return self.post("/v1/noul", _portable_payload(state, question, None, id, mode, input_policy), contract=contract)

    def choice(self, state: State, question: str, options: Mapping[str, str | None] | Sequence[Any], *, id: str | None = None,
               mode: str | None = None, input_policy: Mapping[str, Any] | None = None, contract: bool = False) -> dict[str, Any]:
        """``POST /v1/choice`` — exactly one of the options."""
        return self.post("/v1/choice", _portable_payload(state, question, normalize_options(options), id, mode, input_policy), contract=contract)

    def decision(self, state: State, question: str, options: Mapping[str, str | None] | Sequence[Any], *, id: str | None = None,
                 mode: str | None = None, input_policy: Mapping[str, Any] | None = None, contract: bool = False) -> dict[str, Any]:
        """``POST /v1/decision`` — the compatibility alias of ``/v1/choice``."""
        return self.post("/v1/decision", _portable_payload(state, question, normalize_options(options), id, mode, input_policy), contract=contract)

    def shared(self, state: State, decisions: Sequence[Mapping[str, Any]], *, input_policy: Mapping[str, Any] | None = None,
               contract: bool = False) -> dict[str, Any]:
        """``POST /v1/shared`` — several choice decisions over one shared state.

        Each decision is ``{"question": ..., "options": ..., "id": optional}``; ``options`` takes every form
        ``normalize_options`` accepts.
        """
        payload: dict[str, Any] = {"state": state, "decisions": [_shared_decision(decision) for decision in decisions]}
        if input_policy is not None:
            payload["input_policy"] = dict(input_policy)
        return self.post("/v1/shared", payload, contract=contract)

    # ---- native typed routes ----

    def score(self, state: State, question: Mapping[str, Any], *, name: str = "decision", facts: str | None = None) -> dict[str, Any]:
        """``POST /v1/score`` — one native ordered-level question (``{"instructions", "criteria": [...]}``)."""
        return self.post("/v1/score", _typed_payload(state, question, name, facts))

    def multi(self, state: State, question: Mapping[str, Any], *, name: str = "decision", facts: str | None = None) -> dict[str, Any]:
        """``POST /v1/multi`` — one native multi-label question."""
        return self.post("/v1/multi", _typed_payload(state, question, name, facts))

    def act(self, state: State, question: Mapping[str, Any], *, name: str = "decision", facts: str | None = None) -> dict[str, Any]:
        """``POST /v1/act`` — one native cost-aware action question."""
        return self.post("/v1/act", _typed_payload(state, question, name, facts))

    def soam(self, state: State, questions: Mapping[str, Mapping[str, Any]], *, facts: str | None = None) -> dict[str, Any]:
        """``POST /v1/soam`` — State Once, Ask Many: several typed questions over one state in one native request."""
        return self.post("/v1/soam", _system_one_payload(state, questions, facts))

    def systemone(self, state: State, questions: Mapping[str, Mapping[str, Any]], *, facts: str | None = None) -> dict[str, Any]:
        """``POST /v1/systemone`` — the raw native System One contract."""
        return self.post("/v1/systemone", _system_one_payload(state, questions, facts))


def _portable_payload(state: State, question: str, options: list[dict[str, str]] | None, id: str | None, mode: str | None,
                      input_policy: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    payload: dict[str, Any] = {"state": state, "question": question}
    if options is not None:
        if not options:
            raise ValueError("options must not be empty")
        payload["options"] = options
    if id is not None:
        payload["id"] = id
    if mode is not None:
        if mode not in ("direct", "serial"):
            raise ValueError("mode must be 'direct' or 'serial'")
        payload["mode"] = mode
    if input_policy is not None:
        payload["input_policy"] = dict(input_policy)
    return payload


def _shared_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(decision, Mapping) or "question" not in decision or "options" not in decision:
        raise ValueError("each shared decision needs 'question' and 'options'")
    item: dict[str, Any] = {"question": str(decision["question"]), "options": normalize_options(decision["options"])}
    if decision.get("id") is not None:
        item["id"] = str(decision["id"])
    return item


def _typed_payload(state: State, question: Mapping[str, Any], name: str, facts: str | None) -> dict[str, Any]:
    if not isinstance(question, Mapping) or not question:
        raise ValueError("question must be a non-empty object (instructions, criteria, ...)")
    payload: dict[str, Any] = {"state": state, "question": dict(question), "name": name}
    if facts is not None:
        payload["facts"] = facts
    return payload


def _system_one_payload(state: State, questions: Mapping[str, Mapping[str, Any]], facts: str | None) -> dict[str, Any]:
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("questions must be a non-empty object {name: question}")
    payload: dict[str, Any] = {"state": state, "questions": {str(key): dict(value) for key, value in questions.items()}}
    if facts is not None:
        payload["facts"] = facts
    return payload
