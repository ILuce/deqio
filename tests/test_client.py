"""D13 (0.6.2): the Python SDK ``deqio-client`` (``clients/python``, package ``deqio_client``).

The SDK is dependency-free, never imports ``deqio``, covers the eight decision routes and the ``/v1/decision``
alias, sends exactly the request shapes the server's pydantic models accept (checked here by parsing every
recorded body with those models), returns the server's JSON unchanged and raises ``DeqioError`` with the
server's status and ``detail`` on any rejection.  The requests travel over a real localhost HTTP server.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIENT_SRC = ROOT / "clients" / "python" / "src"
sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(CLIENT_SRC))

from _fake_deqio_http import FakeDeqioServer  # noqa: E402

STATE = "A customer reports that the API returns HTTP 500 after the latest deployment."
OPTIONS = {"bug": "An existing function no longer works", "feature": "A request for new functionality"}
NATIVE = {"instructions": "Rate the operational severity.", "criteria": ["low", "medium", "high"]}
QUESTIONS = {"ok": {"type": "noul", "instructions": "Is it fine?"}, "route": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}


def _client(server: FakeDeqioServer):
    from deqio_client import DeqioClient

    return DeqioClient(server.base_url, timeout=5)


def test_sdk_is_dependency_free_and_versioned_with_deqio() -> None:
    import deqio_client

    assert Path(deqio_client.__file__).resolve() == (CLIENT_SRC / "deqio_client" / "__init__.py").resolve()
    from deqio import __version__

    assert deqio_client.__version__ == __version__
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    project = tomllib.loads((ROOT / "clients" / "python" / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "deqio-client" and project["version"] == __version__
    assert project["dependencies"] == [] and project["requires-python"] == ">=3.10"
    root = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "clients/python" in root["tool"]["uv"]["workspace"]["members"]
    assert any(dependency.startswith("deqio-client>=") for dependency in root["project"]["dependencies"])
    assert deqio_client.DECISION_ROUTES == (
        "/v1/noul", "/v1/choice", "/v1/decision", "/v1/shared", "/v1/score", "/v1/multi", "/v1/act", "/v1/soam", "/v1/systemone",
    )


def test_sdk_does_not_import_deqio() -> None:
    import subprocess

    code = "import sys, deqio_client; assert 'deqio' not in sys.modules, sorted(m for m in sys.modules if m.startswith('deqio')); print('ok')"
    result = subprocess.run([sys.executable, "-c", code], env={"PYTHONPATH": str(CLIENT_SRC), "PATH": "/usr/bin:/bin"},
                            text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "ok"


def test_every_decision_route_sends_the_servers_request_shape() -> None:
    from deqio.server import DecisionRequest, NoulRequest, SharedRequest, SystemOneRequest, TypedSystemOneRequest

    with FakeDeqioServer() as server:
        client = _client(server)
        calls = [
            ("noul", lambda: client.noul(STATE, "Is it a bug?"), "/v1/noul", NoulRequest,
             {"state": STATE, "question": "Is it a bug?"}),
            ("choice", lambda: client.choice(STATE, "Classify.", OPTIONS), "/v1/choice", DecisionRequest,
             {"state": STATE, "question": "Classify.", "options": [{"id": "bug", "description": OPTIONS["bug"]},
                                                                     {"id": "feature", "description": OPTIONS["feature"]}]}),
            ("decision", lambda: client.decision(STATE, "Classify.", ["bug", "feature"], id="d-1", mode="direct"), "/v1/decision",
             DecisionRequest, {"state": STATE, "question": "Classify.", "options": [{"id": "bug", "description": "bug"},
                                                                                    {"id": "feature", "description": "feature"}],
                               "id": "d-1", "mode": "direct"}),
            ("shared", lambda: client.shared({"text": STATE}, [{"id": "k", "question": "Kind?", "options": OPTIONS},
                                                               {"question": "Urgent?", "options": [{"id": "yes", "description": "Y"}]}]),
             "/v1/shared", SharedRequest,
             {"state": {"text": STATE}, "decisions": [
                 {"question": "Kind?", "options": [{"id": "bug", "description": OPTIONS["bug"]}, {"id": "feature", "description": OPTIONS["feature"]}], "id": "k"},
                 {"question": "Urgent?", "options": [{"id": "yes", "description": "Y"}]}]}),
            ("score", lambda: client.score(STATE, NATIVE), "/v1/score", TypedSystemOneRequest,
             {"state": STATE, "question": NATIVE, "name": "decision"}),
            ("multi", lambda: client.multi(STATE, {"criteria": {"a": "A"}}, name="labels", facts="auto"), "/v1/multi", TypedSystemOneRequest,
             {"state": STATE, "question": {"criteria": {"a": "A"}}, "name": "labels", "facts": "auto"}),
            ("act", lambda: client.act(STATE, {"criteria": {"t": "T"}, "costs": {"go": {"t": 0}}}), "/v1/act", TypedSystemOneRequest,
             {"state": STATE, "question": {"criteria": {"t": "T"}, "costs": {"go": {"t": 0}}}, "name": "decision"}),
            ("soam", lambda: client.soam(STATE, QUESTIONS), "/v1/soam", SystemOneRequest, {"state": STATE, "questions": QUESTIONS}),
            ("systemone", lambda: client.systemone(STATE, QUESTIONS, facts="off"), "/v1/systemone", SystemOneRequest,
             {"state": STATE, "questions": QUESTIONS, "facts": "off"}),
        ]
        for name, call, path, model, expected_body in calls:
            server.answer(path, {"route": path, "answers": {}})
            response = call()
            assert response == {"route": path, "answers": {}}, name  # the server's JSON, unchanged
            sent = server.last
            assert sent["method"] == "POST" and sent["path"] == path, name
            assert sent["headers"]["content-type"] == "application/json", name
            assert sent["headers"]["user-agent"].startswith("deqio-client/"), name
            assert "deqio-contract" not in sent["headers"], name
            assert sent["body"] == expected_body, (name, sent["body"])
            model.model_validate(sent["body"])  # exactly what the server's handler accepts (extra fields forbidden on native routes)
        assert client.health() == {"ok": True}
        assert server.last["method"] == "GET" and server.last["path"] == "/health"


def test_contract_negotiation_sends_the_header_the_policy_and_the_id() -> None:
    with FakeDeqioServer() as server:
        client = _client(server)
        server.answer("/v1/noul", {"input_receipt": {"status": "unknown"}})
        client.noul("s", "Is it?", id="n-1", input_policy={"require_complete": False}, contract=True)
        sent = server.last
        assert sent["headers"]["deqio-contract"] == "input-completeness-v1"
        assert sent["body"] == {"state": "s", "question": "Is it?", "id": "n-1", "input_policy": {"require_complete": False}}
        assert sent["raw"] == b'{"state": "s", "question": "Is it?", "id": "n-1", "input_policy": {"require_complete": false}}'


def test_rejections_raise_deqio_error_with_the_servers_status_and_detail() -> None:
    from deqio_client import DeqioConnectionError, DeqioError

    with FakeDeqioServer() as server:
        client = _client(server)
        server.answer("/v1/choice", {"detail": "options contains 0 options"}, status=400)
        with pytest.raises(DeqioError) as refused:
            client.choice("s", "Which?", [{"id": "a", "description": "A"}])
        assert refused.value.status == 400 and refused.value.detail == "options contains 0 options"
        assert refused.value.path == "/v1/choice" and "HTTP 400" in str(refused.value)
        server.answer("/v1/noul", {"error": {"code": "unsupported_contract", "message": "no"}}, status=422)
        with pytest.raises(DeqioError) as contract:
            client.noul("s", "Is it?")
        assert contract.value.status == 422 and contract.value.detail == {"code": "unsupported_contract", "message": "no"}
        server.answer("/v1/score", {"detail": "runtime suspended"}, status=503)
        with pytest.raises(DeqioError) as suspended:
            client.score("s", NATIVE)
        assert suspended.value.status == 503
        base = server.base_url
    with socket.socket() as probe:  # the port is closed again: a connection error, never a retry or a fallback
        probe.settimeout(1)
        assert probe.connect_ex(("127.0.0.1", int(base.rsplit(":", 1)[1]))) != 0
    from deqio_client import DeqioClient

    with pytest.raises(DeqioConnectionError):
        DeqioClient(base, timeout=2).noul("s", "Is it?")


def test_sdk_refuses_invalid_shapes_before_sending() -> None:
    from deqio_client import DeqioClient, normalize_options

    assert normalize_options({"a": "A", "b": None}) == [{"id": "a", "description": "A"}, {"id": "b", "description": "b"}]
    assert normalize_options(["x", {"id": "y", "description": "Y"}]) == [{"id": "x", "description": "x"}, {"id": "y", "description": "Y"}]
    with pytest.raises(ValueError):
        normalize_options("ab")
    with pytest.raises(ValueError):
        normalize_options([{"description": "no id"}])

    recorded: list = []

    def transport(method, url, headers, body, timeout):
        recorded.append((method, url))
        return 200, {}, b"{}"

    client = DeqioClient("http://127.0.0.1:1", transport=transport)
    with pytest.raises(ValueError):
        client.choice("s", "Which?", [])
    with pytest.raises(ValueError):
        client.noul("s", "   ")
    with pytest.raises(ValueError):
        client.noul("s", "Is it?", mode="fast")
    with pytest.raises(ValueError):
        client.score("s", {})
    with pytest.raises(ValueError):
        client.soam("s", {})
    with pytest.raises(ValueError):
        client.shared("s", [{"question": "no options"}])
    with pytest.raises(ValueError):
        DeqioClient("127.0.0.1:8787")
    assert recorded == []  # nothing was sent
    assert client.noul("s", "Is it?") == {} and recorded == [("POST", "http://127.0.0.1:1/v1/noul")]
