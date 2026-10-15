"""A real localhost HTTP stand-in for ``deqio serve`` used by the SDK and ``deqio decide`` tests (D13).

It records every request (method, path, headers, decoded JSON body) and answers with the canned JSON configured
per path (status + body), so the client code under test runs its real ``urllib`` transport end to end.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class FakeDeqioServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses: dict[str, tuple[int, Any]] = {}
        self.default_response: tuple[int, Any] = (200, {"ok": True})
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:  # keep pytest output clean
                return None

            def _serve(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw.decode("utf-8")) if raw else None
                server.requests.append({
                    "method": self.command, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body, "raw": raw,
                })
                status, payload = server.responses.get(self.path, server.default_response)
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = _serve
            do_POST = _serve

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeDeqioServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def answer(self, path: str, body: Any, *, status: int = 200) -> None:
        self.responses[path] = (status, body)

    @property
    def last(self) -> dict[str, Any]:
        assert self.requests, "no request reached the fake server"
        return self.requests[-1]
