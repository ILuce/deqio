from __future__ import annotations

import argparse
import atexit
from contextlib import asynccontextmanager
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _wait(url: str, process: subprocess.Popen[str], timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during JevK5 GGUF startup ({process.returncode})")
        try:
            with urllib.request.urlopen(url + "/health", timeout=1.0) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("Timed out waiting for the pinned llama-server used by JevK5 GGUF")


def build_app(*, llama_server: Path, gguf: Path, max_tokens: int, temperature: float, knockout_temperature: float):
    from fastapi import FastAPI, HTTPException
    from jevk5 import JevK5GGUF

    inner_port = _free_port()
    command = [
        str(llama_server), "-m", str(gguf), "-c", str(max_tokens), "-ngl", "99",
        "--host", "127.0.0.1", "--port", str(inner_port), "--log-disable",
    ]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    atexit.register(_terminate, process)
    url = f"http://127.0.0.1:{inner_port}"
    _wait(url, process)
    model = JevK5GGUF(url, temperature=temperature, knockout_temperature=knockout_temperature)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            _terminate(process)

    app = FastAPI(title="Deqio JevK5 GGUF transport", lifespan=lifespan)

    @app.get("/health")
    def health():
        if process.poll() is not None:
            raise HTTPException(status_code=503, detail="llama-server is not running")
        return {"status": "ready", "engine": "jevk5", "backend": "gguf"}

    @app.post("/v1/systemone")
    def system_one(body: dict[str, Any]):
        questions = body.get("questions") if isinstance(body, dict) else None
        if not isinstance(questions, dict) or not questions:
            raise HTTPException(status_code=422, detail="questions must be a non-empty object")
        for qid, question in questions.items():
            if not isinstance(question, dict):
                raise HTTPException(status_code=422, detail=f"{qid}: question must be an object")
        started = time.perf_counter()
        answers: dict[str, Any] = {}
        input_tokens = 0
        try:
            for qid, question in questions.items():
                answer = model.decide(body.get("state"), question)
                input_tokens += int(answer.pop("input_tokens", 0) or 0)
                answers[str(qid)] = answer
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "model": gguf.name,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "questions": len(questions), "output_tokens": 0},
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve official JevK5 Q8 GGUF using the upstream JevK5GGUF readout")
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--knockout-temperature", type=float, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not args.llama_server.is_file():
        raise RuntimeError(f"Pinned llama-server is missing: {args.llama_server}")
    if not args.gguf.is_file():
        raise RuntimeError(f"JevK5 GGUF weights are missing: {args.gguf}")
    import uvicorn
    uvicorn.run(
        build_app(
            llama_server=args.llama_server,
            gguf=args.gguf,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            knockout_temperature=args.knockout_temperature,
        ),
        host="127.0.0.1", port=args.port, access_log=False, log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
