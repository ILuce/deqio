"""Fake remote decision APIs speaking the documented wires (0.6.1 / D11b).

``FakeOpenAIDecisionsAPI`` implements ``POST /v1/decisions`` as documented at
https://developers.openai.com/api/docs/guides/decisions (questions as an array with ``name``,
``choices``/``levels``; answers as an array; ``usage.input_tokens``).  ``FakeTypeSafeAPI``
implements ``POST /v1/systemone`` as documented at https://docs.typesafe.ai/api (questions keyed
by id, answers keyed by id, ``usage``).  Both record every request so tests can assert the exact
bytes Deqio sends.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


class _FakeAPI:
    path = "/"

    def __init__(self, *, status: int = 200, body: dict | None = None, request_id_header: bool = True) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = status
        self.body = body
        self.request_id_header = request_id_header
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", "0"))
                raw_body = self.rfile.read(length)
                try:
                    payload = json.loads(raw_body.decode("utf-8"))
                except json.JSONDecodeError:
                    payload = None
                api.requests.append({"path": self.path, "headers": dict(self.headers), "body": payload})
                status = api.status
                if api.body is not None:
                    response = api.body
                elif self.path != api.path or not isinstance(payload, dict):
                    status = 404
                    response = {"error": {"message": f"nothing is served at {self.path}"}}
                else:
                    try:
                        response = api.answer(payload)
                    except ValueError as error:
                        status = 422
                        response = {"error": {"message": str(error)}}
                raw = json.dumps(response).encode("utf-8")
                self.send_response(status)
                self.send_header("content-type", "application/json")
                if api.request_id_header:
                    self.send_header("x-request-id", f"req_{len(api.requests):04d}")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args) -> None:
                return None

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def answer(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    @property
    def endpoint(self) -> str:
        return self.base_url + self.path


class FakeOpenAIDecisionsAPI(_FakeAPI):
    """``POST /v1/decisions``: ``{"model", "input", "questions": [...]}`` -> ``{"answers": [...], "model", "usage"}``."""

    path = "/v1/decisions"

    def answer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("model") != "gpt-6-luna":
            raise ValueError("unknown model")
        if not isinstance(payload.get("input"), (str, list)):
            raise ValueError("input must be a string or an array of messages")
        questions = payload.get("questions")
        if not isinstance(questions, list) or not questions:
            raise ValueError("questions must be a non-empty array")
        answers = []
        for question in questions:
            kind = question.get("type")
            name = question.get("name")
            if not name:
                raise ValueError("every question needs a name")
            if kind == "choice":
                values = [choice["value"] for choice in question.get("choices", [])]
                if not values:
                    raise ValueError("choice needs choices")
                probabilities = [
                    {"value": value, "probability": (0.7 if index == 0 else round(0.3 / max(len(values) - 1, 1), 6))}
                    for index, value in enumerate(values)
                ]
                answers.append({"type": "choice", "name": name, "choice": values[0], "probabilities": probabilities, "confidence": 0.93})
            elif kind == "predicate":
                answers.append({"type": "predicate", "name": name, "probability": 0.8})
            elif kind == "score":
                levels = question.get("levels", [])
                if len(levels) < 2:
                    raise ValueError("score needs levels")
                share = round(1.0 / len(levels), 6)
                answers.append({
                    "type": "score",
                    "name": name,
                    "score": (len(levels) - 1) / 2.0,
                    "probabilities": [{"value": index, "label": level["label"], "probability": share} for index, level in enumerate(levels)],
                    "confidence": 0.5,
                })
            else:
                raise ValueError(f"unsupported question type {kind!r}")
        text = payload["input"] if isinstance(payload["input"], str) else json.dumps(payload["input"])
        return {
            "answers": answers,
            "model": "gpt-6-luna",
            "usage": {"input_tokens": 100 * len(questions) + len(text) // 100, "output_tokens": 0, "total_tokens": 100 * len(questions)},
        }


class FakeTypeSafeAPI(_FakeAPI):
    """``POST /v1/systemone``: ``{"state", "model", "questions": {id: ...}}`` -> ``{"model", "answers": {id: ...}, "usage"}``."""

    path = "/v1/systemone"

    def answer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "state" not in payload or not payload.get("model"):
            raise ValueError("state and model are required")
        questions = payload.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty object")
        answers: dict[str, Any] = {}
        for qid, question in questions.items():
            kind = question.get("type")
            if kind == "choice":
                keys = list(question.get("criteria") or {})
                if not keys:
                    raise ValueError("choice needs criteria")
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[0],
                    "confidence": 0.93,
                    "probabilities": {key: (0.7 if index == 0 else round(0.3 / max(len(keys) - 1, 1), 6)) for index, key in enumerate(keys)},
                }
            elif kind == "noul":
                answers[qid] = {"type": "noul", "noul": 0.8}
            elif kind == "score":
                levels = question.get("criteria") or []
                if len(levels) < 2:
                    raise ValueError("score needs at least two levels")
                share = round(1.0 / len(levels), 6)
                answers[qid] = {
                    "type": "score",
                    "score": (len(levels) - 1) / 2.0,
                    "confidence": 0.5,
                    "legend": {str(index): level for index, level in enumerate(levels)},
                    "probabilities": {str(index): share for index in range(len(levels))},
                }
            else:
                raise ValueError(f"unsupported question type {kind!r}")
        return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 100 * len(questions), "output_tokens": 0}}


class FakeKeyring:
    """In-memory stand-in for the ``keyring`` module (set/get/delete_password)."""

    def __init__(self, *, available: bool = True) -> None:
        self.store: dict[tuple[str, str], str] = {}
        self.available = available

    def _check(self) -> None:
        if not self.available:
            raise RuntimeError("No recommended backend was available (keyring.errors.NoKeyringError)")

    def set_password(self, service: str, username: str, password: str) -> None:
        self._check()
        self.store[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        self._check()
        return self.store.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        self._check()
        if (service, username) not in self.store:
            raise RuntimeError("password not found (keyring.errors.PasswordDeleteError)")
        del self.store[(service, username)]

    def get_keyring(self):
        return type("Backend", (), {"name": "FakeKeyring"})()
