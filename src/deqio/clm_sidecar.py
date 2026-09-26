from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _runtime_executable(name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return Path(sys.executable).resolve().parent / f"{name}{suffix}"


def _wait_for_http(port: int, process: subprocess.Popen[Any], *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"CLM encoder exited during startup with code {process.returncode}")
        try:
            with urlopen(url, timeout=1.0) as response:
                if 200 <= int(response.status) < 500:
                    return
        except (OSError, URLError, TimeoutError):
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for the CLM encoder on port {port}")


def _terminate(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run(args: argparse.Namespace) -> int:
    vllm = _runtime_executable("vllm")
    clm_serve = _runtime_executable("clm-serve")
    if not vllm.is_file():
        raise RuntimeError(f"vLLM executable was not installed: {vllm}")
    if not clm_serve.is_file():
        raise RuntimeError(f"clm-serve executable was not installed: {clm_serve}")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"CLM checkpoint does not exist: {checkpoint}")

    encoder_port = _free_port()
    encoder: subprocess.Popen[Any] | None = None
    api: subprocess.Popen[Any] | None = None
    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        _terminate(api)
        _terminate(encoder)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    env = os.environ.copy()
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    try:
        encoder = subprocess.Popen(
            [
                str(vllm), "serve", args.encoder_model,
                "--served-model-name", args.embedding_model,
                "--runner", "pooling",
                "--enable-prefix-caching",
                "--max-model-len", str(args.max_tokens),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--host", "127.0.0.1",
                "--port", str(encoder_port),
            ],
            env=env,
        )
        _wait_for_http(encoder_port, encoder, timeout=float(args.encoder_startup_timeout))

        api = subprocess.Popen(
            [
                str(clm_serve),
                "--port", str(args.port),
                "--emb-url", f"http://127.0.0.1:{encoder_port}/v1/embeddings",
                "--emb-model", args.embedding_model,
                "--max-tokens", str(args.max_tokens),
                "--ckpt", str(checkpoint),
                "--device", "cuda",
                "--no-ui",
            ],
            env=env,
        )

        while not stopping:
            encoder_code = encoder.poll()
            api_code = api.poll()
            if encoder_code is not None:
                return int(encoder_code or 1)
            if api_code is not None:
                return int(api_code)
            time.sleep(0.5)
        return 0
    finally:
        _terminate(api)
        _terminate(encoder)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the two-process CLM runtime as one Deqio sidecar.")
    parser.add_argument("--encoder-model", required=True)
    parser.add_argument("--embedding-model", default="qwen3-8b")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--encoder-startup-timeout", type=float, default=900.0)
    parser.add_argument("--port", type=int, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
