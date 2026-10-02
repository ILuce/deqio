from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _snapshot(model: str, revision: str) -> Path:
    """Resolve the already-prefetched Clef snapshot without network access."""
    from huggingface_hub import snapshot_download

    kwargs: dict[str, object] = {
        "repo_id": model,
        "local_files_only": True,
    }
    if revision and revision != "upstream-latest":
        kwargs["revision"] = revision
    return Path(snapshot_download(**kwargs)).resolve()


def _serve_command(
    *,
    python: str,
    snapshot: Path,
    name: str,
    max_length: int,
    port: int,
) -> list[str]:
    script = snapshot / "clef_mlx.py"
    if not script.is_file():
        raise RuntimeError(
            f"Clef MLX loader is missing from cached snapshot: {script}. "
            "Run: deqio models setup"
        )
    return [
        python,
        str(script),
        "serve",
        "--model",
        str(snapshot),
        "--name",
        name,
        "--max-length",
        str(max_length),
        "--no-truncate",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--quiet",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the pinned Clef MLX SystemOne server from the local HF cache."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    snapshot = _snapshot(args.model, args.revision)
    command = _serve_command(
        python=sys.executable,
        snapshot=snapshot,
        name=args.name,
        max_length=args.max_length,
        port=args.port,
    )
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
