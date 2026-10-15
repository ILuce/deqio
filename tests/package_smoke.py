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
    import deqio_client

    metadata_requirements = requires("deqio") or []
    forbidden = [
        requirement
        for requirement in metadata_requirements
        if "git+" in requirement.lower() or " @ http" in requirement.lower()
    ]
    if forbidden:
        raise RuntimeError(f"published metadata contains direct/VCS dependencies: {forbidden}")
    if not any(requirement.startswith("deqio-client") for requirement in metadata_requirements):
        raise RuntimeError(f"deqio metadata does not require deqio-client: {metadata_requirements}")
    if requires("deqio-client"):
        raise RuntimeError(f"deqio-client must stay dependency-free: {requires('deqio-client')}")
    if deqio_client.__version__ != deqio.__version__:
        raise RuntimeError(f"deqio-client {deqio_client.__version__} does not match deqio {deqio.__version__}")

    package_root = files("deqio")
    required = [
        "data/config.json",
        "data/models.json",
        "data/benchmarks/basic.json",
        "data/benchmarks/pl.json",
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

        version_result = subprocess.run(
            ["deqio", "--version"], cwd=work, env=env, check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if version_result.stdout.strip() != f"deqio {deqio.__version__}":
            raise RuntimeError("deqio --version did not report the installed package version")

        subprocess.run(
            ["deqio", "models", "list"], cwd=work, env=env, check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        compatible = subprocess.run(
            ["deqio", "models", "list", "--compatible", "--json"],
            cwd=work, env=env, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        compatible_payload = json.loads(compatible.stdout)
        if not isinstance(compatible_payload.get("models"), list):
            raise RuntimeError("deqio models list --compatible --json returned an invalid payload")
        subprocess.run(
            ["deqio", "models", "delete", "--help"], cwd=work, env=env, check=True,
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
        subprocess.run(
            ["deqio", "benchmark", "compare", "--help"], cwd=work, env=env, check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        decide_help = subprocess.run(
            ["deqio", "decide", "--help"], cwd=work, env=env, check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if "--option" not in decide_help.stdout:
            raise RuntimeError("deqio decide --help did not execute from the installed distribution")
        unreachable = subprocess.run(
            ["deqio", "decide", "noul", "--url", "http://127.0.0.1:9", "--timeout", "2", "--state", "s", "--question", "Is it?"],
            cwd=work, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if unreachable.returncode != 1 or "cannot reach" not in unreachable.stderr:
            raise RuntimeError(f"deqio decide against a closed port must exit 1: {unreachable.returncode} {unreachable.stderr}")
        openapi = subprocess.run(
            ["deqio", "openapi"], cwd=work, env=env, check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        document = json.loads(openapi.stdout)
        if document.get("openapi") != "3.1.0" or "/v1/systemone" not in document.get("paths", {}):
            raise RuntimeError("deqio openapi did not print the OpenAPI 3.1 document")
        if document["info"]["version"] != deqio.__version__:
            raise RuntimeError("deqio openapi reports another version than the installed package")

        for relative in ("config.json", "models.json", "benchmarks/basic.json", "benchmarks/pl.json"):
            if not (work / relative).is_file():
                raise RuntimeError(f"workspace bootstrap did not create {relative}")

        suite = json.loads((work / "benchmarks/basic.json").read_text(encoding="utf-8"))
        counts = {kind: 0 for kind in ("noul", "choice", "shared")}
        for case in suite["cases"]:
            counts[case["type"]] += 1
        if counts != {"noul": 50, "choice": 50, "shared": 50}:
            raise RuntimeError(f"unexpected packaged benchmark counts: {counts}")

        pl_suite = json.loads((work / "benchmarks/pl.json").read_text(encoding="utf-8"))
        pl_counts = {kind: 0 for kind in ("noul", "choice", "shared")}
        for case in pl_suite["cases"]:
            pl_counts[case["type"]] += 1
        if pl_counts != {"noul": 20, "choice": 20, "shared": 20}:
            raise RuntimeError(f"unexpected packaged Polish benchmark counts: {pl_counts}")

        subprocess.run(
            [
                sys.executable,
                "-c",
                "from deqio.server import app; "
                "paths={r.path for r in app.routes}; "
                "assert '/ui' in paths and '/ui/watch' in paths and '/v1/noul' in paths and '/v1/watch' in paths and '/v1/watch/settings' in paths and '/v1/benchmarks' in paths and '/v1/benchmarks/compare' in paths",
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
