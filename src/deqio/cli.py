from __future__ import annotations

import argparse
import sys

import uvicorn

from . import benchmark as benchmark_runner
from . import model_manager


def _serve(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="deqio serve",
        description="Start the Deqio HTTP API and browser UI.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", default=8787, type=int, help="Bind port (default: 8787)")
    args = parser.parse_args(argv)

    uvicorn.run(
        "deqio.server:app",
        host=args.host,
        port=args.port,
        workers=1,
        access_log=False,
        log_level="warning",
    )
    return 0


def _print_help() -> None:
    print(
        """Deqio — Decisions in. Probabilities out.

Usage:
  deqio serve [--host HOST] [--port PORT]
  deqio models <command> [options]
  deqio benchmark [--all | --model MODEL_ID:BACKEND]
  deqio status

Commands:
  serve    Start the API and browser UI
  models    Install, inspect, select, and update decision models
  benchmark Run the editable local benchmark suite
  status    Show the currently selected model/runtime

Examples:
  deqio serve
  deqio models setup
  deqio models installed
  deqio models use
  deqio benchmark --all
  deqio status
"""
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0

    command, rest = args[0], args[1:]
    if command == "serve":
        return _serve(rest)
    if command == "models":
        return model_manager.main(rest)
    if command == "benchmark":
        return benchmark_runner.main(rest)
    if command == "status":
        return model_manager.main(["status", *rest])

    print(f"error: unknown command {command!r}\n", file=sys.stderr)
    _print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
