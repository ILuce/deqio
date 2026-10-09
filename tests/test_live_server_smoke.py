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
sys.exit(main(["serve", "--port", os.environ["SMOKE_PORT"]]))
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
        for path, body in successes:
            status, payload = _call(base, path, body)
            assert status == 200, (path, payload)
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
        assert stats["requests"] == len(successes) + 1
        assert stats["errors"] == len(rejections)
        assert stats["decisions"] == len(successes) + 1 + 1  # shared carries two decisions

        status, watch = _call(base, "/v1/watch?limit=100")
        assert status == 200
        assert len(watch["events"]) == len(successes) + 1 + len(rejections)
        shared_rows = [row for row in watch["events"] if row["endpoint"] == "/v1/shared" and row["status_code"] == 200]
        assert shared_rows and shared_rows[0]["input_tokens"] == 120

        status, recent = _call(base, "/v1/recent")
        assert status == 200
        assert len(recent) == len(successes) + 1 if isinstance(recent, list) else True
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
