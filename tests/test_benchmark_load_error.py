"""B12: a profile whose runtime fails to load is reported as such everywhere.

Previously such a profile showed up as "0 cases, 0.0 %, 0 errors" in the CLI
table, `load_error` was displayed nowhere, and `compare_documents` filed it as
a "legacy profile without canonical identity".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from deqio import benchmark
from deqio.backends import BackendRuntime
from deqio.benchmark_compare import compare_documents
from deqio.workspace import ensure_workspace

LOAD_ERROR = "Runtime for kev-0.8b is not installed. Run: deqio models setup"


def test_benchmark_run_reports_a_runtime_that_failed_to_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = ensure_workspace(tmp_path / "ws")
    suite = config.parent / "benchmarks" / "basic.json"
    if not suite.is_file():
        suite = Path(benchmark.__file__).with_name("data") / "benchmarks" / "basic.json"
    row = {
        "model_id": "kev-0.8b", "backend": "mps", "engine": "kev", "label": "Kev 0.8B",
        "installed": True, "host_compatible": True, "max_input_tokens": 4096,
    }
    monkeypatch.setattr(benchmark, "_installed", lambda *args, **kwargs: [row])

    def failing_load(cls, settings):
        raise RuntimeError(LOAD_ERROR)

    monkeypatch.setattr(BackendRuntime, "load", classmethod(failing_load))
    output = tmp_path / "run"
    args = benchmark.build_parser().parse_args(
        ["--config", str(config), "--suite", str(suite), "--output", str(output), "--all"]
    )

    assert benchmark.run(args) == 0

    out = capsys.readouterr().out
    assert "load error" in out and LOAD_ERROR in out
    # No fake "0 cases / 0.0 %" statistics rows for a profile that never ran.
    assert re.search(r"kev-0\.8b:mps\s+all\s+0\s+0\.0%", out) is None
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["models"][0]["load_error"] == LOAD_ERROR


def _failed_model() -> dict:
    return {
        "model_id": "kev-0.8b", "backend": "mps", "engine": "kev", "load_ms": 12.0,
        "load_error": LOAD_ERROR, "profile_identity": None, "runtime_identity": None,
        "summary": benchmark.summarize([]),
    }


def _legacy_model() -> dict:
    return {
        "model_id": "old-model", "backend": "mlx", "engine": "old", "load_ms": 100.0,
        "load_error": None, "profile_identity": None, "runtime_identity": None,
        "summary": benchmark.summarize([]),
    }


def test_compare_separates_load_failures_from_legacy_profiles() -> None:
    left = {"run_id": "a", "suite_name": "basic", "models": [_failed_model()]}
    right = {"run_id": "b", "suite_name": "basic", "models": [_legacy_model()]}

    result = compare_documents(left, right, left_id="a", right_id="b")

    assert [row["model_id"] for row in result["load_failed_left"]] == ["kev-0.8b"]
    assert result["load_failed_left"][0]["load_error"] == LOAD_ERROR
    assert result["load_failed_right"] == []
    assert result["legacy_unverified_left"] == []
    assert [row["model_id"] for row in result["legacy_unverified_right"]] == ["old-model"]
    assert result["common_profiles"] == []
    assert any("failed to load" in warning for warning in result["warnings"])
    json.dumps(result)
