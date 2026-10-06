from __future__ import annotations

import argparse
import os
import sys

import uvicorn

from . import __version__
from . import benchmark as benchmark_runner
from . import model_manager
from .config import read_config_data
from .runtime_control import discover_server


def _serve(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="deqio serve",
        description="Start the Deqio HTTP API and browser UI.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", default=8787, type=int, help="Bind port (default: 8787)")
    args = parser.parse_args(argv)

    try:
        config_path, _ = read_config_data()
        existing = discover_server(config_path)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if existing is not None and int(existing.get("pid", 0)) != os.getpid():
        print(
            f"error: another Deqio server is already running for this workspace "
            f"(pid={existing.get('pid')}, {existing.get('base_url')}).",
            file=sys.stderr,
        )
        return 2

    previous = {
        name: os.environ.get(name)
        for name in ("DEQIO_SERVER_CONTROL", "DEQIO_SERVER_HOST", "DEQIO_SERVER_PORT")
    }
    os.environ["DEQIO_SERVER_CONTROL"] = "1"
    os.environ["DEQIO_SERVER_HOST"] = str(args.host)
    os.environ["DEQIO_SERVER_PORT"] = str(args.port)
    try:
        uvicorn.run(
            "deqio.server:app",
            host=args.host,
            port=args.port,
            workers=1,
            access_log=False,
            log_level="warning",
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return 0


def _print_help() -> None:
    print(
        """Deqio — Decision in, probabilities out.

Usage:
  deqio serve [--host HOST] [--port PORT]
  deqio models <command> [options]
  deqio benchmark [--all | --model MODEL_ID:BACKEND]
  deqio benchmark compare [--left RUN_ID --right RUN_ID] [--json]
  deqio status
  deqio version
  deqio --version

Commands:
  serve      Start the API and browser UI
  models     Install, inspect, select, and update decision models
  benchmark  Run benchmark suites or compare completed runs
  status     Show the currently selected model/runtime
  version    Show the installed Deqio version

Examples:
  deqio serve
  deqio models setup
  deqio models list --compatible
  deqio models installed
  deqio models use
  deqio models delete
  deqio benchmark --all
  deqio benchmark compare
  deqio status
  deqio --version
"""
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0
    if args[0] in {"-V", "--version", "version"}:
        print(f"deqio {__version__}")
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
