from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import requires
from importlib.resources import files
from pathlib import Path


def run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    import deqio

    metadata_requirements = requires("deqio") or []
    forbidden = [
        requirement
        for requirement in metadata_requirements
        if "git+" in requirement.lower() or " @ http" in requirement.lower()
    ]
    if forbidden:
        raise RuntimeError(f"published metadata contains direct/VCS dependencies: {forbidden}")

    package_root = files("deqio")
    required = [
        "data/config.json",
        "data/models.json",
        "data/benchmarks/basic.json",
    ]
    for relative in required:
        node = package_root
        for part in relative.split("/"):
            node = node.joinpath(part)
        if not node.is_file():
            raise RuntimeError(f"wheel is missing packaged asset: {relative}")

    with tempfile.TemporaryDirectory(prefix="deqio-wheel-smoke-") as raw:
        work = Path(raw)
        env = os.environ.copy()
        env.pop("DEQIO_CONFIG", None)

        help_result = subprocess.run(
            ["deqio", "--help"], cwd=work, env=env, check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if "Deqio" not in help_result.stdout:
            raise RuntimeError("deqio --help did not execute from the installed distribution")

        subprocess.run(
            ["deqio", "models", "list"], cwd=work, env=env, check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        subprocess.run(
            ["deqio", "serve", "--help"], cwd=work, env=env, check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        subprocess.run(
            ["deqio", "benchmark", "--help"], cwd=work, env=env, check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        for relative in ("config.json", "models.json", "benchmarks/basic.json"):
            if not (work / relative).is_file():
                raise RuntimeError(f"workspace bootstrap did not create {relative}")

        suite = json.loads((work / "benchmarks/basic.json").read_text(encoding="utf-8"))
        counts = {kind: 0 for kind in ("noul", "choice", "shared")}
        for case in suite["cases"]:
            counts[case["type"]] += 1
        if counts != {"noul": 50, "choice": 50, "shared": 50}:
            raise RuntimeError(f"unexpected packaged benchmark counts: {counts}")

        subprocess.run(
            [
                sys.executable,
                "-c",
                "from deqio.server import app; "
                "paths={r.path for r in app.routes}; "
                "assert '/ui' in paths and '/v1/noul' in paths and '/v1/benchmarks' in paths",
            ],
            cwd=work,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    print(f"package smoke OK: deqio {deqio.__version__} from {deqio.__file__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
