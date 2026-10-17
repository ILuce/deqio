"""0.6.2: ``deqio benchmark tables`` renders the README / ``benchmarks/summary_*.md`` tables from completed runs.

The generator reads a run's ``summary.json`` (schema 2) and ``results.jsonl``, keeps every number as the run
reported it, sorts by decision accuracy, computes the estimated cost per 1 000 scored decisions for remote
profiles from the per-row estimates, lists profiles that did not load, and excludes Decision 3.0 MLX rows produced
by another Deqio forward than the current one (their numbers are not comparable).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _model(model_id: str, backend: str, *, decision_accuracy: float, case_accuracy: float, median: float, p95: float,
           throughput: float, identity: dict | None = None, runtime_metadata: dict | None = None, provider: dict | None = None,
           load_error: str | None = None, per_type: bool = True) -> dict:
    block = lambda cases, correct, assertions: {  # noqa: E731 - compact fixture builder
        "cases": cases, "passed_cases": int(cases * case_accuracy), "case_accuracy": case_accuracy, "assertions": assertions,
        "correct": correct, "decision_accuracy": decision_accuracy, "runtime_errors": 0, "completed_assertions": assertions,
        "mean_ms": median, "median_ms": median, "p95_ms": p95, "throughput_decisions_per_s": throughput,
    }
    summary = {"overall": block(60, int(93 * decision_accuracy), 93)}
    if per_type:
        summary.update({"noul": block(20, 15, 20), "choice": block(20, 16, 20), "shared": block(20, 10, 53)})
    return {
        "model_id": model_id, "backend": backend, "engine": model_id.split("-")[0], "load_ms": 10.0, "load_error": load_error,
        "profile_identity": {"schema_version": 1, "canonical_id": f"{model_id}-{backend}-id", "model_id": model_id, "backend": backend,
                             "engine": model_id.split("-")[0], "model": f"org/{model_id}", "source": "official",
                             "artifacts": [{"source": "huggingface", "repo_id": f"org/{model_id}", "revision": "0123456789abcdef" * 2 + "01234567"}],
                             **(identity or {})},
        "runtime_identity": {"deqio_version": "0.6.2", "runtime_instance_id": "r-1", "engine": model_id.split("-")[0], "model_id": model_id,
                             "backend": backend, "runtime_metadata": runtime_metadata or {}, **({"provider": provider} if provider else {})},
        "summary": summary,
    }


def _write_run(directory: Path, models: list[dict], results: list[dict] | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps({
        "schema_version": 2, "run_id": "20261015T120300Z", "created_at": "2026-10-15T12:05:34+00:00",
        "suite": "/somewhere/benchmarks/pl.json", "suite_name": "deqio-pl-60", "suite_cases": 60, "models": models,
    }), encoding="utf-8")
    if results is not None:
        (directory / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in results), encoding="utf-8")
    return directory


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    from deqio.d3_mlx_sidecar import D3_MLX_FORWARD

    models = [
        _model("kev-0.8b", "mlx", decision_accuracy=0.785, case_accuracy=0.667, median=54.0, p95=84.2, throughput=25.13),
        _model("decider-0.8b", "mps", decision_accuracy=0.839, case_accuracy=0.75, median=330.7, p95=395.7, throughput=4.9),
        _model("basal-2.0-main", "gguf", decision_accuracy=0.9, case_accuracy=0.85, median=800.0, p95=900.0, throughput=1.2,
               identity={"precision": "gguf-q8_0", "quantization": {"quantization": "q8_0", "kind": "gguf"}, "runtime": "llama-cpp", "runtime_version": "cbe28bb3"}),
        _model("openai-decisions", "remote", decision_accuracy=0.989, case_accuracy=0.983, median=314.1, p95=487.7, throughput=4.56,
               identity={"artifacts": [], "model": "gpt-6-luna"},
               provider={"name": "openai-decisions", "kind": "remote", "model": "gpt-6-luna", "endpoint": "https://api.openai.com/v1/decisions"}),
        _model("typesafe-jev", "remote", decision_accuracy=1.0, case_accuracy=1.0, median=317.7, p95=424.8, throughput=4.64,
               identity={"artifacts": [], "model": "jev-latest"},
               provider={"name": "typesafe-jev", "kind": "remote", "model": "jev-latest", "endpoint": "https://api.typesafe.ai/v1/systemone"}),
        _model("d3-lite", "mlx", decision_accuracy=0.548, case_accuracy=0.367, median=118.2, p95=352.3, throughput=8.76,
               identity={"precision": "bf16", "source": "community", "runtime": "d3-mlx-community"}, runtime_metadata={"forward": "deqio-d3-mlx-forward/1"}),
        _model("d3-nano", "mlx", decision_accuracy=0.7, case_accuracy=0.6, median=200.0, p95=300.0, throughput=5.0,
               identity={"precision": "bf16", "source": "community", "runtime": "d3-mlx-community"}, runtime_metadata={"forward": D3_MLX_FORWARD}),
        _model("laya-english", "mlx", decision_accuracy=0.0, case_accuracy=0.0, median=0.0, p95=0.0, throughput=0.0,
               load_error="sidecar did not become ready", per_type=False),
    ]
    results = []
    for index in range(93):
        results.append({"model_id": "openai-decisions", "backend": "remote", "case_id": f"c{index}", "kind": "choice", "passed": True,
                        "assertions": 1, "correct": 1, "latency_ms": 300.0, "provider_name": "openai-decisions",
                        "remote_request_id": f"req_{index}", "cost_estimate": {"currency": "USD", "amount": 0.00002, "basis": "input_tokens",
                                                                              "scope": "decision", "input_tokens": 200, "estimate": True}})
    results.append({"model_id": "typesafe-jev", "backend": "remote", "case_id": "c0", "kind": "choice", "passed": True, "assertions": 1,
                    "correct": 1, "latency_ms": 300.0, "provider_name": "typesafe-jev", "remote_request_id": None,
                    "cost_estimate": {"currency": "USD", "amount": None, "basis": "input_tokens_unavailable", "scope": "decision", "estimate": True}})
    return _write_run(tmp_path / "run", models, results)


def test_tables_keep_the_runs_numbers_sort_by_decision_accuracy_and_exclude_incomparable_rows(run_dir: Path) -> None:
    from deqio.benchmark_tables import build_tables, load_run
    from deqio.catalog import load_catalog

    catalog = load_catalog(ROOT / "src" / "deqio" / "data" / "models.json")
    tables = build_tables(load_run(run_dir), catalog=catalog)
    assert tables["run_id"] == "20261015T120300Z" and tables["suite_name"] == "deqio-pl-60" and tables["suite_cases"] == 60
    assert tables["scored_decisions"] == 93 and tables["deqio_version"] == "0.6.2"
    assert [row["profile"] for row in tables["rows"]] == [
        "typesafe-jev:remote", "openai-decisions:remote", "basal-2.0-main:gguf", "decider-0.8b:mps", "kev-0.8b:mlx", "d3-nano:mlx",
    ]
    assert tables["excluded"] == [{"profile": "d3-lite:mlx", "reason": "Decision 3.0 MLX forward deqio-d3-mlx-forward/1; only deqio-d3-mlx-forward/2 runs are reported"}]
    assert tables["not_loaded"] == [{"profile": "laya-english:mlx", "load_error": "sidecar did not become ready"}]
    by_profile = {row["profile"]: row for row in tables["rows"]}
    kev = by_profile["kev-0.8b:mlx"]
    assert kev["label"] == "Kev 0.8B" and kev["backend_label"] == "MLX" and kev["decision_accuracy"] == 0.785 and kev["median_ms"] == 54.0
    assert kev["per_type"]["shared"] == {"case_accuracy": 0.667, "decision_accuracy": 0.785} and kev["cost"] is None
    assert kev["revision"] == "01234567" and kev["source"] == "official"
    assert by_profile["basal-2.0-main:gguf"]["backend_label"] == "GGUF Q8_0" and by_profile["basal-2.0-main:gguf"]["runtime"] == "llama-cpp"
    openai = by_profile["openai-decisions:remote"]
    assert openai["cost"]["usd_per_1k_decisions"] == pytest.approx(0.00002 * 93 / 93 * 1000) and openai["cost"]["decisions"] == 93
    assert openai["endpoint"] == "https://api.openai.com/v1/decisions" and openai["model"] == "gpt-6-luna" and openai["backend_label"] == "remote"
    assert by_profile["typesafe-jev:remote"]["cost"]["usd_per_1k_decisions"] is None  # one row without a reported amount → no number
    assert by_profile["d3-nano:mlx"]["d3_mlx_forward"] == "deqio-d3-mlx-forward/2"


def test_markdown_tables_name_the_run_the_exclusions_and_every_metric(run_dir: Path) -> None:
    from deqio.benchmark_tables import build_tables, load_run, render_markdown

    text = render_markdown(build_tables(load_run(run_dir)), title="Models up to 1B")
    assert text.startswith("### Models up to 1B\n\n#### Run\n")
    assert "- `20261015T120300Z` · suite `deqio-pl-60` — 60 cases / 93 scored decisions · Deqio 0.6.2 · 2026-10-15" in text
    assert "- excluded: `d3-lite:mlx` — Decision 3.0 MLX forward deqio-d3-mlx-forward/1" in text
    assert "- did not load: `laya-english:mlx` — sidecar did not become ready" in text
    assert "| Model | Backend | Accuracy | Decision accuracy | Noul | Choice | Shared | Shared decision accuracy |" in text
    assert "| typesafe-jev | remote | 100.0% | 100.0% |" in text  # no catalog → the model id is the label
    assert "| kev-0.8b | MLX | 66.7% | 78.5% | 66.7% | 66.7% | 66.7% | 78.5% |" in text
    assert "| Model | Backend | Median | P95 | Throughput | Cost / 1k decisions |" in text
    assert "| openai-decisions | remote | 314.1 ms | 487.7 ms | 4.56/s | $0.0200 |" in text
    assert "| typesafe-jev | remote | 317.7 ms | 424.8 ms | 4.64/s | n/a |" in text
    assert "| kev-0.8b | MLX | 54.0 ms | 84.2 ms | 25.13/s | — |" in text
    assert "| basal-2.0-main | GGUF Q8_0 | basal | llama-cpp cbe28bb3 | gguf-q8_0 | official | `org/basal-2.0-main` | 01234567 |" in text
    assert "`gpt-6-luna` · `https://api.openai.com/v1/decisions`" in text


def test_tables_cli_filters_profiles_prints_json_and_refuses_bad_input(run_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from deqio.cli import main

    assert main(["benchmark", "tables", "--summary", str(run_dir), "--models", "kev-0.8b:mlx", "decider-0.8b", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert [row["profile"] for row in document["rows"]] == ["decider-0.8b:mps", "kev-0.8b:mlx"]
    assert document["rows"][0]["label"] == "Decider 0.8B" and document["excluded"] == [] and document["not_loaded"] == []
    out = tmp_path / "tables.md"
    assert main(["benchmark", "tables", "--summary", str(run_dir / "summary.json"), "--title", "PL", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("### PL\n") and "Kev 0.8B" in out.read_text(encoding="utf-8")
    assert main(["benchmark", "tables"]) == 2
    assert "--run" in capsys.readouterr().err
    assert main(["benchmark", "tables", "--summary", str(tmp_path / "missing" / "summary.json")]) == 2
    (tmp_path / "old.json").write_text(json.dumps({"schema_version": 1, "models": []}), encoding="utf-8")
    assert main(["benchmark", "tables", "--summary", str(tmp_path / "old.json")]) == 2
    assert "schema_version" in capsys.readouterr().err
    assert main(["benchmark", "tables", "--summary", str(run_dir), "--models", ":mlx"]) == 2


def test_tables_resolve_run_ids_from_the_workspace_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from deqio.cli import main
    from deqio.workspace import ensure_workspace

    config_path = ensure_workspace(tmp_path / "ws")
    store = tmp_path / "ws" / ".deqio" / "benchmarks" / "20261015T120300Z"
    _write_run(store, [_model("kev-0.8b", "mlx", decision_accuracy=0.785, case_accuracy=0.667, median=54.0, p95=84.2, throughput=25.13)])
    assert main(["benchmark", "tables", "--config", str(config_path), "--run", "20261015T120300Z"]) == 0
    assert "Kev 0.8B" in capsys.readouterr().out
    assert main(["benchmark", "tables", "--config", str(config_path), "--run", "nope"]) == 2
    assert "benchmark run not found" in capsys.readouterr().err
