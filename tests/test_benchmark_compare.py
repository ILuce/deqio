import json
from pathlib import Path

import pytest

from deqio.benchmark import BenchResult, summarize
from deqio.benchmark_compare import compare_documents, main as compare_main
from deqio.benchmark_identity import build_profile_identity
from deqio.runtime_control import discover_server, register_server, unregister_server


def _identity(model_id: str, backend: str, *, precision: str, revision: str = "abc") -> dict:
    return build_profile_identity(
        model_id=model_id,
        backend=backend,
        engine="demo",
        runtime_identity={
            "family": "demo",
            "runtime": "demo-native",
            "runtime_version": "1.0",
            "model": f"upstream/{model_id}",
            "source": "official",
            "precision": precision,
            "artifacts": [
                {
                    "source": "huggingface",
                    "repo_id": f"upstream/{model_id}",
                    "resolved_revision": revision,
                    "role": "model",
                }
            ],
        },
    )


def _model(model_id: str, backend: str, *, accuracy: float, median: float, precision: str = "fp16", p95: float | None = None, throughput: float | None = None) -> dict:
    identity = _identity(model_id, backend, precision=precision)
    overall = {
        "cases": 10,
        "case_accuracy": accuracy,
        "decision_accuracy": accuracy,
        "median_ms": median,
        "p95_ms": p95 if p95 is not None else median * 1.5,
        "throughput_decisions_per_s": throughput if throughput is not None else 1000.0 / median,
    }
    return {
        "model_id": model_id,
        "backend": backend,
        "engine": "demo",
        "profile_identity": identity,
        "summary": {
            "overall": overall,
            "choice": dict(overall),
        },
    }


def _document(models: list[dict], *, suite: str) -> dict:
    return {"schema_version": 2, "created_at": "2026-10-05T00:00:00+00:00", "suite_name": suite, "models": models}


def test_benchmark_comparison_intersection_deltas_rank_and_only_sides() -> None:
    left = _document(
        [
            _model("model-a", "mlx", accuracy=0.95, median=20),
            _model("model-b", "mlx", accuracy=0.80, median=30),
            _model("model-c", "gguf", accuracy=0.70, median=10, precision="q8_0"),
        ],
        suite="ENG",
    )
    right = _document(
        [
            _model("model-b", "mlx", accuracy=0.90, median=25),
            _model("model-c", "gguf", accuracy=0.60, median=12, precision="q8_0"),
            _model("model-d", "mps", accuracy=0.99, median=40),
        ],
        suite="PL",
    )

    comparison = compare_documents(left, right, left_id="run-a", right_id="run-b")

    assert [row["model_id"] for row in comparison["common_profiles"]] == ["model-b", "model-c"]
    assert [row["model_id"] for row in comparison["only_in_left"]] == ["model-a"]
    assert [row["model_id"] for row in comparison["only_in_right"]] == ["model-d"]
    model_b = comparison["common_profiles"][0]
    assert model_b["metrics"]["case_accuracy"]["absolute"] == pytest.approx(0.10)
    assert model_b["metrics"]["median_ms"]["absolute"] == pytest.approx(-5.0)
    assert model_b["rank"]["left"] == 2
    assert model_b["rank"]["right"] == 2
    assert model_b["rank"]["shift"] == 0
    assert "choice" in model_b["breakdown"]
    assert comparison["same_suite"] is False
    assert comparison["warnings"]


def test_same_model_different_backend_or_quantization_is_not_common() -> None:
    left = _document([_model("same", "mlx", accuracy=0.8, median=20, precision="mlx-8bit")], suite="ENG")
    right_backend = _document([_model("same", "gguf", accuracy=0.8, median=20, precision="q8_0")], suite="ENG")
    right_precision = _document([_model("same", "mlx", accuracy=0.8, median=20, precision="mlx-4bit")], suite="ENG")

    assert compare_documents(left, right_backend, left_id="a", right_id="b")["common_profiles"] == []
    assert compare_documents(left, right_precision, left_id="a", right_id="b")["common_profiles"] == []


def test_missing_metrics_are_not_invented_and_sort_is_deterministic() -> None:
    left_model = _model("b", "mlx", accuracy=0.8, median=20)
    right_model = _model("b", "mlx", accuracy=0.9, median=30)
    del left_model["summary"]["overall"]["p95_ms"]
    del right_model["summary"]["overall"]["p95_ms"]
    comparison = compare_documents(
        _document([left_model], suite="ENG"),
        _document([right_model], suite="ENG"),
        left_id="a",
        right_id="b",
    )
    metrics = comparison["common_profiles"][0]["metrics"]
    assert "p95_ms" not in metrics
    assert comparison["warnings"] == []


def test_legacy_summary_profiles_are_not_silently_matched() -> None:
    legacy = {"model_id": "legacy", "backend": "mlx", "engine": "demo", "summary": {"overall": {"case_accuracy": 1.0}}}
    comparison = compare_documents(_document([legacy], suite="ENG"), _document([legacy], suite="ENG"), left_id="a", right_id="b")
    assert comparison["common_profiles"] == []
    assert len(comparison["legacy_unverified_left"]) == 1
    assert len(comparison["legacy_unverified_right"]) == 1
    assert any("legacy" in warning.lower() for warning in comparison["warnings"])


def test_summarize_reports_runtime_errors_and_throughput() -> None:
    rows = [
        BenchResult("m", "mlx", "e", "1", "choice", True, "a", "a", 100.0, 1, 1),
        BenchResult("m", "mlx", "e", "2", "choice", False, "a", None, 50.0, 1, 0, error="oom"),
    ]
    overall = summarize(rows)["overall"]
    assert overall["runtime_errors"] == 1
    assert overall["completed_assertions"] == 1
    assert overall["throughput_decisions_per_s"] == pytest.approx(10.0)
    assert overall["case_accuracy"] == pytest.approx(0.5)


def test_compare_cli_json_and_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"engine": "demo"}), encoding="utf-8")
    root = tmp_path / ".deqio" / "benchmarks"
    for run_id, accuracy in (("run-a", 0.8), ("run-b", 0.9)):
        directory = root / run_id
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(
            json.dumps(_document([_model("m", "mlx", accuracy=accuracy, median=20)], suite="ENG")),
            encoding="utf-8",
        )
    output = tmp_path / "comparison.json"
    assert compare_main(["--config", str(tmp_path / "config.json"), "--left", "run-a", "--right", "run-b", "--json", "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["common_profiles"][0]["model_id"] == "m"
    assert json.loads(output.read_text())["right"]["run_id"] == "run-b"


def test_runtime_control_registration_is_workspace_scoped(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    registration = register_server(config, host="0.0.0.0", port=9876, pid=__import__("os").getpid())
    try:
        found = discover_server(config)
        assert found is not None
        assert found["pid"] == registration["pid"]
        assert found["base_url"] == "http://127.0.0.1:9876"
    finally:
        unregister_server(config, pid=registration["pid"])
    assert discover_server(config) is None


def test_compare_allows_different_case_counts() -> None:
    left_model = _model("m", "mlx", accuracy=0.8, median=20)
    right_model = _model("m", "mlx", accuracy=0.9, median=25)
    left_model["summary"]["overall"]["cases"] = 10
    right_model["summary"]["overall"]["cases"] = 25
    comparison = compare_documents(
        _document([left_model], suite="ENG"),
        _document([right_model], suite="ENG"),
        left_id="a",
        right_id="b",
    )
    assert comparison["common_profiles"][0]["metrics"]["case_accuracy"]["absolute"] == pytest.approx(0.1)


def test_compare_cli_interactive_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"engine": "demo"}), encoding="utf-8")
    root = tmp_path / ".deqio" / "benchmarks"
    for run_id, accuracy in (("run-a", 0.8), ("run-b", 0.9)):
        directory = root / run_id
        directory.mkdir(parents=True)
        document = _document([_model("m", "mlx", accuracy=accuracy, median=20)], suite="ENG")
        document["created_at"] = "2026-10-05T00:00:00+00:00" if run_id == "run-a" else "2026-10-06T00:00:00+00:00"
        (directory / "summary.json").write_text(json.dumps(document), encoding="utf-8")
    answers = iter(["1", "1"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert compare_main(["--config", str(config)]) == 0
    output = capsys.readouterr().out
    assert "Common profiles" in output
    assert "Breakdown by decision type" in output
    assert "Only in A" in output
    assert "Only in B" in output


def test_benchmark_workspace_lock_rejects_live_owner_and_reaps_stale(tmp_path: Path) -> None:
    import os
    from deqio.benchmark import _benchmark_workspace_lock

    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    lock_path = tmp_path / ".deqio" / "benchmark.lock"
    lock_path.mkdir(parents=True)
    (lock_path / "owner.json").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="already running"):
        with _benchmark_workspace_lock(config):
            pass

    (lock_path / "owner.json").write_text(json.dumps({"pid": 99999999}), encoding="utf-8")
    with _benchmark_workspace_lock(config):
        assert lock_path.is_file()
    # Kernel lock files intentionally persist; ownership itself is released by
    # the OS and the next benchmark can acquire the same inode safely.
    assert lock_path.is_file()


def test_same_model_same_backend_different_pinned_revision_is_not_common() -> None:
    left = _model("same", "mlx", accuracy=0.8, median=20, precision="mlx-8bit")
    right = _model("same", "mlx", accuracy=0.9, median=20, precision="mlx-8bit")
    right["profile_identity"] = _identity("same", "mlx", precision="mlx-8bit", revision="def")
    comparison = compare_documents(
        _document([left], suite="ENG"),
        _document([right], suite="ENG"),
        left_id="a",
        right_id="b",
    )
    assert comparison["common_profiles"] == []


def test_metric_delta_contains_absolute_and_relative_change() -> None:
    comparison = compare_documents(
        _document([_model("m", "mlx", accuracy=0.8, median=20)], suite="ENG"),
        _document([_model("m", "mlx", accuracy=0.9, median=10)], suite="ENG"),
        left_id="a",
        right_id="b",
    )
    row = comparison["common_profiles"][0]
    assert row["metrics"]["case_accuracy"]["absolute"] == pytest.approx(0.1)
    assert row["metrics"]["case_accuracy"]["percent"] == pytest.approx(0.125)
    assert row["metrics"]["median_ms"]["absolute"] == pytest.approx(-10.0)
    assert row["metrics"]["median_ms"]["percent"] == pytest.approx(-0.5)


def test_runtime_control_refuses_second_live_server_same_workspace(tmp_path: Path) -> None:
    import os

    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    first = register_server(config, host="127.0.0.1", port=8787, pid=os.getpid())
    try:
        assert discover_server(config)["pid"] == os.getpid()
        with pytest.raises(RuntimeError, match="Another Deqio server is already running"):
            register_server(config, host="127.0.0.1", port=8788, pid=os.getpid() + 100000)
        assert discover_server(config)["token"] == first["token"]
    finally:
        unregister_server(config, pid=os.getpid())
