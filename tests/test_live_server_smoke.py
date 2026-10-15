"""Boot the real ``deqio serve`` path with a stub engine and hit every route over HTTP.

The ASGI-level tests in ``test_server.py`` exercise handlers in-process. This
test runs uvicorn in a child process exactly like ``deqio serve`` does
(workspace control file, lifespan, worker threads, contract middleware) and
checks the public contract that AGENTS.md lists for serving changes:
``/health`` plus the nine decision routes, with a real network round-trip.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

STUB_SERVER = r'''
import hashlib, json, os, sys
from deqio.backends import BackendRuntime


class StubRuntime:
    """Deterministic engine stand-in with the raw-result shape SystemOneRuntime produces."""

    runtime_instance_id = "stub-runtime-1"
    engine = "stub"

    def identity_snapshot(self):
        return {
            "deqio_version": "test", "runtime_instance_id": self.runtime_instance_id,
            "engine": "stub", "model_id": "stub-model", "backend": "gguf", "model": "stub/model",
            "requested_revision": "main", "artifacts": [], "artifact_revisions_resolved": False,
            "installation_verified_at": None,
        }

    def refresh_identity(self):
        return None

    def input_completeness_capability(self):
        return {"status": "unavailable", "reason": "stub"}

    def _raw(self, row, option_ids, probabilities, *, batch=None):
        payload = json.dumps({"state": row["state"], "question": row["question"]}, sort_keys=True).encode()
        raw = {
            "id": row["id"], "option_ids": option_ids, "probabilities": probabilities,
            "input_tokens": 21, "input_tokens_source": "engine_reported",
            "engine_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "total_seconds": 0.003, "prompt_sha256": hashlib.sha256(payload).hexdigest(),
            "probability_status": "stub probabilities",
            "score_provenance": {"kind": "engine_probability", "source": "engine.probabilities",
                                 "synthetic": False, "normalized": False, "transforms": [],
                                 "raw_logits_available": False, "calibration": "unspecified"},
        }
        if batch is not None:
            raw.update(input_tokens=None, input_tokens_source="unknown",
                       batch_input_tokens=batch, batch_input_tokens_source="engine_reported")
        return raw

    def score(self, row, mode):
        ids = [str(o["id"]) for o in row["options"]]
        probs = [0.7] + [0.3 / max(1, len(ids) - 1)] * (len(ids) - 1)
        return self._raw(row, ids, probs)

    def score_noul(self, row, mode):
        return self._raw(row, ["yes", "no"], [0.8, 0.2])

    def score_shared(self, rows):
        results = []
        for row in rows:
            ids = [str(o["id"]) for o in row["options"]]
            probs = [0.6] + [0.4 / max(1, len(ids) - 1)] * (len(ids) - 1)
            results.append(self._raw(row, ids, probs, batch=120 if len(rows) > 1 else None))
        return results, {"total_seconds": 0.005, "batch_size": len(rows), "engine": "stub"}

    def system_one(self, state, questions, options=None):
        answers = {}
        for qid, question in questions.items():
            kind = question.get("type")
            if kind == "noul":
                answers[qid] = {"type": "noul", "noul": 0.9}
            elif kind in ("choice", "score", "multi", "act"):
                criteria = question.get("criteria") or {}
                keys = list(criteria) if isinstance(criteria, dict) else [str(c) for c in criteria]
                answers[qid] = {"type": kind, "choice": keys[0] if keys else None,
                                "probabilities": {k: 1.0 / len(keys) for k in keys} if keys else {}}
            else:
                raise ValueError(f"unsupported question type {kind!r}")
        return {"model": "stub/model", "answers": answers, "usage": {"input_tokens": 33}}, {
            "total_seconds": 0.004, "batch_size": len(questions), "engine": "stub"}

    def clear_cache(self):
        return {"engine": "stub", "backend": "gguf", "model_loaded": True}

    def close(self):
        return None


BackendRuntime.load = classmethod(lambda cls, settings: StubRuntime())

from deqio.cli import main
sys.exit(main(["serve", "--host", os.environ.get("SMOKE_HOST", "127.0.0.1"), "--port", os.environ["SMOKE_PORT"]]))
'''


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _call(base: str, path: str, body: dict | None = None, headers: dict | None = None) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        base + path, data=data, method="POST" if body is not None else "GET",
        headers={"content-type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


@pytest.mark.skipif(os.name == "nt", reason="deqio serve is validated on POSIX hosts only in 0.5")
def test_live_server_serves_every_decision_route_over_http(tmp_path: Path) -> None:
    from deqio.workspace import ensure_workspace

    workspace = tmp_path / "workspace"
    config_path = ensure_workspace(workspace)
    port = _free_port()
    env = {**os.environ, "DEQIO_CONFIG": str(config_path), "SMOKE_PORT": str(port)}
    log = (tmp_path / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-c", STUB_SERVER], cwd=workspace, env=env,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 30
        health: dict = {}
        while time.time() < deadline:
            if process.poll() is not None:
                raise AssertionError("server exited early:\n" + (tmp_path / "server.log").read_text())
            try:
                status, health = _call(base, "/health")
                if status == 200 and health.get("status") == "ok":
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.2)
        assert health.get("status") == "ok", health
        assert health["runtime_instance_id"] == "stub-runtime-1"
        assert (workspace / ".deqio" / "server-control.json").is_file()
        banner = (tmp_path / "server.log").read_text()
        assert f"http://127.0.0.1:{port}/ui" in banner  # the banner names the real bind address

        options = [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}]
        question = {"instructions": "Rate it", "criteria": {"low": "Low", "high": "High"}}
        successes = [
            ("/v1/noul", {"state": "s", "question": "Is it?"}),
            ("/v1/choice", {"state": "s", "question": "Which?", "options": options}),
            ("/v1/decision", {"state": "s", "question": "Which?", "options": options}),
            ("/v1/shared", {"state": "s", "decisions": [
                {"id": "d1", "question": "A?", "options": options},
                {"id": "d2", "question": "B?", "options": options},
            ]}),
            ("/v1/score", {"state": "s", "question": question}),
            ("/v1/multi", {"state": "s", "question": question}),
            ("/v1/act", {"state": "s", "question": question}),
            ("/v1/soam", {"state": "s", "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}),
            ("/v1/systemone", {"state": "s", "questions": {"q": {"type": "choice", **question}}}),
        ]
        from deqio.openapi import validate_response

        for path, body in successes:
            status, payload = _call(base, path, body)
            assert status == 200, (path, payload)
            validate_response(path, payload)  # D13: every documented route answers in its documented shape
            if path in ("/v1/noul", "/v1/choice", "/v1/decision"):
                assert payload["provenance"]["runtime"]["runtime_instance_id"] == "stub-runtime-1"
                assert payload["input_tokens"] == 21
            elif path == "/v1/shared":
                assert [r["input_tokens"] for r in payload["results"]] == [None, None]
                assert payload["provenance"]["attestation"]["kind"] == "deqio-local-shared-response"
            else:
                assert payload["deqio"]["runtime"]["runtime_instance_id"] == "stub-runtime-1"
                assert payload["usage"]["input_tokens"] == 33

        # Negotiated contract over the real middleware: body hash + receipt.
        status, payload = _call(
            base, "/v1/noul",
            {"state": "s", "question": "Is it?", "id": "n-1", "input_policy": {"require_complete": False}},
            {"deqio-contract": "input-completeness-v1"},
        )
        assert status == 200, payload
        assert payload["input_receipt"]["status"] == "unknown"
        assert payload["provenance"]["schema_version"] == 2
        validate_response("/v1/noul", payload)

        # D13: the SDK and `deqio decide` over the same real server, through their own urllib transport.
        from deqio_client import DeqioClient, DeqioError

        sdk = DeqioClient(base, timeout=10)
        assert sdk.health()["runtime_instance_id"] == "stub-runtime-1"
        sdk_calls = [
            ("/v1/noul", lambda: sdk.noul("s", "Is it?")),
            ("/v1/choice", lambda: sdk.choice("s", "Which?", {"a": "A", "b": "B"})),
            ("/v1/decision", lambda: sdk.decision("s", "Which?", options)),
            ("/v1/shared", lambda: sdk.shared("s", [{"id": "d1", "question": "A?", "options": options}, {"id": "d2", "question": "B?", "options": options}])),
            ("/v1/score", lambda: sdk.score("s", question)),
            ("/v1/multi", lambda: sdk.multi("s", question)),
            ("/v1/act", lambda: sdk.act("s", question)),
            ("/v1/soam", lambda: sdk.soam("s", {"q": {"type": "noul", "instructions": "Is it?"}})),
            ("/v1/systemone", lambda: sdk.systemone("s", {"q": {"type": "choice", **question}})),
        ]
        for path, call in sdk_calls:
            payload = call()
            validate_response(path, payload)
            block = payload["provenance"] if path in ("/v1/noul", "/v1/choice", "/v1/decision", "/v1/shared") else payload["deqio"]
            assert block["runtime"]["runtime_instance_id"] == "stub-runtime-1" and block["provider"]["name"] == "local", path
            assert block["remote_request_id"] is None and block["cost_estimate"] is None, path
        negotiated = sdk.noul("s", "Is it?", id="n-3", input_policy={"require_complete": False}, contract=True)
        assert negotiated["input_receipt"]["status"] == "unknown" and negotiated["provenance"]["schema_version"] == 2
        try:
            sdk.choice("s", "Which?", [{"id": "x", "description": "X"}, {"id": "x", "description": "X"}])
        except DeqioError as rejected:
            assert rejected.status == 400, rejected.detail  # the server's own rejection, surfaced unchanged
        else:
            raise AssertionError("duplicate option ids must be rejected by the server")

        import io
        from contextlib import redirect_stderr, redirect_stdout

        from deqio.cli import main as deqio_main

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = deqio_main(["decide", "choice", "--url", base, "--state", "s", "--question", "Which?",
                               "--option", "a=A", "--option", "b=B", "--json"])
        assert code == 0, err.getvalue()
        decided = json.loads(out.getvalue())
        validate_response("/v1/choice", decided)
        assert decided["decision"] == "a" and decided["provenance"]["runtime"]["runtime_instance_id"] == "stub-runtime-1"
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = deqio_main(["decide", "soam", "--url", base, "--state", "s", "--questions-json", '{"q": {"type": "noul"}}'])
        assert code == 0 and "q: " in out.getvalue() and "provider=local" in out.getvalue(), err.getvalue()
        sdk_requests = len(sdk_calls) + 1 + 1 + 2  # + negotiated noul + rejected choice + two `deqio decide` calls

        # Rejections that reach a handler: counted once each.
        rejections = [
            ("/v1/choice", {"state": "s", "question": "Which?", "options": []}, {}, 400),
            ("/v1/noul", {"state": "s", "question": "   "}, {}, 400),
            ("/v1/shared", {"state": "s", "decisions": [
                {"id": "x", "question": "A?", "options": options},
                {"id": "x", "question": "B?", "options": options},
            ]}, {}, 400),
            ("/v1/score", {"state": "s", "name": " ", "question": question}, {}, 422),
            ("/v1/systemone", {"state": "s", "questions": {}}, {}, 422),
            ("/v1/noul", {"state": "s", "question": "Is it?", "id": "n-2"},
             {"deqio-contract": "input-completeness-v1"}, 422),  # policy missing
        ]
        for path, body, headers, expected in rejections:
            status, payload = _call(base, path, body, headers)
            assert status == expected, (path, payload)

        # Rejected before any handler: not decision traffic.
        status, payload = _call(base, "/v1/choice", {"state": "s"})  # pydantic: missing fields
        assert status == 422
        status, payload = _call(base, "/v1/noul", {"state": "s", "question": "q"}, {"deqio-contract": "no-such-v9"})
        assert status == 422 and payload["error"]["code"] == "unsupported_contract"

        status, stats = _call(base, "/v1/stats")
        assert status == 200
        assert stats["requests"] == len(successes) + 1 + sdk_requests - 1  # the SDK's rejected choice is an error, not a request
        assert stats["errors"] == len(rejections) + 1
        assert stats["decisions"] == len(successes) + 1 + 1 + (sdk_requests - 1) + 1  # shared carries two decisions (twice)

        status, watch = _call(base, "/v1/watch?limit=100")
        assert status == 200
        assert len(watch["events"]) == len(successes) + 1 + len(rejections) + sdk_requests
        shared_rows = [row for row in watch["events"] if row["endpoint"] == "/v1/shared" and row["status_code"] == 200]
        assert shared_rows and shared_rows[0]["input_tokens"] == 120

        status, recent = _call(base, "/v1/recent")
        assert status == 200
        assert isinstance(recent, list) and len(recent) == stats["requests"]  # every successful decision request (50 max)
    finally:
        if process.poll() is None:
            # Ctrl-C equivalent: step 6 of the audit acceptance protocol. (A
            # SIGTERM also shuts down cleanly, but uvicorn re-raises the signal
            # afterwards, so the exit status would be -15 rather than 0.)
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        log.close()

    assert process.returncode == 0, (tmp_path / "server.log").read_text()
    assert not (workspace / ".deqio" / "server-control.json").exists()
    assert "Traceback" not in (tmp_path / "server.log").read_text()


@pytest.mark.skipif(os.name == "nt", reason="deqio serve is validated on POSIX hosts only in 0.5")
def test_live_server_on_a_non_loopback_bind_protects_watch_with_a_token(tmp_path: Path) -> None:
    """D10: `deqio serve --host 0.0.0.0` keeps decisions open but puts Watch behind a token."""
    import http.client

    from deqio.workspace import ensure_workspace

    workspace = tmp_path / "workspace"
    config_path = ensure_workspace(workspace)
    port = _free_port()
    token = "smoke-watch-token-0123456789abcdef"
    env = {
        **os.environ, "DEQIO_CONFIG": str(config_path), "SMOKE_PORT": str(port),
        "SMOKE_HOST": "0.0.0.0", "DEQIO_WATCH_TOKEN": token,
    }
    log = (tmp_path / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-c", STUB_SERVER], cwd=workspace, env=env,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 30
        health: dict = {}
        while time.time() < deadline:
            if process.poll() is not None:
                raise AssertionError("server exited early:\n" + (tmp_path / "server.log").read_text())
            try:
                status, health = _call(base, "/health")
                if status == 200 and health.get("status") == "ok":
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.2)
        assert health.get("status") == "ok", health

        status, _ = _call(base, "/v1/noul", {"state": "s", "question": "Is it?"})
        assert status == 200
        status, body = _call(base, "/v1/watch")
        assert status == 401, body
        status, body = _call(base, "/v1/watch", headers={"X-Deqio-Watch-Token": token})
        assert status == 200 and body["pagination"]["total"] >= 1

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            connection.request("GET", f"/ui/watch?token={token}")
            exchanged = connection.getresponse()
            exchanged.read()
            assert exchanged.status == 303
            assert "HttpOnly" in (exchanged.getheader("set-cookie") or "")
            connection.request("GET", "/ui/watch")
            anonymous = connection.getresponse()
            anonymous.read()
            assert anonymous.status == 401
        finally:
            connection.close()

        banner = (tmp_path / "server.log").read_text()
        assert "Watch token required" in banner
        assert token not in banner  # a configured token is never echoed
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        log.close()

    assert process.returncode == 0, (tmp_path / "server.log").read_text()
    assert "Traceback" not in (tmp_path / "server.log").read_text()
