from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends import BackendRuntime
from .catalog import apply_selection, get_model, get_profile, load_catalog
from .config import read_config_data, settings_from_data
from .installations import installed_profiles


DEFAULT_SUITE = Path("benchmarks/basic.json")


@dataclass
class BenchResult:
    model_id: str
    backend: str
    engine: str
    case_id: str
    kind: str
    passed: bool
    expected: Any
    actual: Any
    latency_ms: float | None
    assertions: int
    correct: int
    top_probability: float | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def load_suite(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise RuntimeError("Benchmark suite must contain a top-level cases array")
    if int(data.get("schema_version", 0)) != 1:
        raise RuntimeError("Unsupported benchmark schema_version")
    seen: set[str] = set()
    counts = {"noul": 0, "choice": 0, "shared": 0}
    for case in data["cases"]:
        if not isinstance(case, dict):
            raise RuntimeError("Every benchmark case must be an object")
        case_id = str(case.get("id", ""))
        kind = str(case.get("type", ""))
        if not case_id or case_id in seen:
            raise RuntimeError(f"Benchmark case id is missing or duplicated: {case_id!r}")
        if kind not in counts:
            raise RuntimeError(f"Unsupported benchmark case type: {kind!r}")
        seen.add(case_id)
        counts[kind] += 1
    return data


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * p)
    return ordered[index]


def _choice(raw: dict[str, Any]) -> tuple[str, float]:
    ids = [str(value) for value in raw["option_ids"]]
    probs = [float(value) for value in raw["probabilities"]]
    index = max(range(len(probs)), key=probs.__getitem__)
    return ids[index], probs[index]


def _choice_row(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(case["id"]),
        "state": case["state"],
        "question": str(case["question"]),
        "options": list(case["options"]),
    }


def _run_case(runtime: BackendRuntime, model: dict[str, str], case: dict[str, Any]) -> BenchResult:
    kind = str(case["type"])
    started = time.perf_counter()
    try:
        if kind == "noul":
            row = {"id": str(case["id"]), "state": case["state"], "question": str(case["question"])}
            raw = runtime.score_noul(row, "serial")
            actual, top = _choice(raw)
            expected = str(case["expected"])
            latency_ms = float(raw.get("total_seconds", time.perf_counter() - started)) * 1000.0
            return BenchResult(**model, case_id=str(case["id"]), kind=kind, passed=actual == expected,
                               expected=expected, actual=actual, latency_ms=latency_ms,
                               assertions=1, correct=int(actual == expected), top_probability=top)

        if kind == "choice":
            raw = runtime.score(_choice_row(case), "serial")
            actual, top = _choice(raw)
            expected = str(case["expected"])
            latency_ms = float(raw.get("total_seconds", time.perf_counter() - started)) * 1000.0
            return BenchResult(**model, case_id=str(case["id"]), kind=kind, passed=actual == expected,
                               expected=expected, actual=actual, latency_ms=latency_ms,
                               assertions=1, correct=int(actual == expected), top_probability=top)

        rows = []
        expected: list[str] = []
        for decision in case["decisions"]:
            rows.append({
                "id": str(decision["id"]),
                "state": case["state"],
                "question": str(decision["question"]),
                "options": list(decision["options"]),
            })
            expected.append(str(decision["expected"]))
        raw_results, timing = runtime.score_shared(rows)
        actual = [_choice(raw)[0] for raw in raw_results]
        correct = sum(a == e for a, e in zip(actual, expected))
        latency_ms = float(timing.get("total_seconds", time.perf_counter() - started)) * 1000.0
        return BenchResult(**model, case_id=str(case["id"]), kind=kind, passed=correct == len(expected),
                           expected=expected, actual=actual, latency_ms=latency_ms,
                           assertions=len(expected), correct=correct)
    except Exception as error:
        return BenchResult(**model, case_id=str(case["id"]), kind=kind, passed=False,
                           expected=case.get("expected") or [d.get("expected") for d in case.get("decisions", [])],
                           actual=None, latency_ms=(time.perf_counter() - started) * 1000.0,
                           assertions=max(1, len(case.get("decisions", []))), correct=0, error=str(error))


def summarize(results: list[BenchResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    groups: dict[str, list[BenchResult]] = {}
    for result in results:
        groups.setdefault(result.kind, []).append(result)
    all_latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    assertions = sum(r.assertions for r in results)
    correct = sum(r.correct for r in results)
    summary["overall"] = {
        "cases": len(results),
        "passed_cases": sum(r.passed for r in results),
        "case_accuracy": (sum(r.passed for r in results) / len(results)) if results else 0.0,
        "assertions": assertions,
        "correct": correct,
        "decision_accuracy": (correct / assertions) if assertions else 0.0,
        "mean_ms": statistics.fmean(all_latencies) if all_latencies else None,
        "median_ms": statistics.median(all_latencies) if all_latencies else None,
        "p95_ms": _percentile(all_latencies, 0.95),
    }
    for kind, rows in groups.items():
        latencies = [r.latency_ms for r in rows if r.latency_ms is not None]
        kind_assertions = sum(r.assertions for r in rows)
        kind_correct = sum(r.correct for r in rows)
        summary[kind] = {
            "cases": len(rows),
            "passed_cases": sum(r.passed for r in rows),
            "case_accuracy": sum(r.passed for r in rows) / len(rows),
            "assertions": kind_assertions,
            "correct": kind_correct,
            "decision_accuracy": kind_correct / kind_assertions if kind_assertions else 0.0,
            "mean_ms": statistics.fmean(latencies) if latencies else None,
            "median_ms": statistics.median(latencies) if latencies else None,
            "p95_ms": _percentile(latencies, 0.95),
        }
    return summary


def _installed(config_path: Path, config_data: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in installed_profiles(
        config_path=config_path,
        config_data=config_data,
        catalog=catalog,
        active_model_id=str(config_data.get("model_id", "")),
        active_backend=str(config_data.get("backend", "")),
    ) if row["installed"] and row["host_compatible"]]


def _select_profiles(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not rows:
        raise RuntimeError("No installed model profiles are available on this host")
    if args.all:
        return rows
    if args.model:
        selected = []
        for value in args.model:
            if ":" not in value:
                raise RuntimeError("--model must use MODEL_ID:BACKEND, for example decider-0.8b:mps")
            model_id, backend = value.rsplit(":", 1)
            match = next((row for row in rows if row["model_id"] == model_id and row["backend"] == backend), None)
            if match is None:
                raise RuntimeError(f"Installed profile not found: {value}")
            selected.append(match)
        return selected

    print("Installed model profiles:")
    for index, row in enumerate(rows, start=1):
        print(f"  {index}. {row['label']} — {row['backend']} ({row['engine']})")
    print("  A. all installed profiles")
    value = input("Select models [A or comma-separated numbers]: ").strip().lower()
    if value in {"", "a", "all"}:
        return rows
    indexes = []
    for token in value.split(","):
        index = int(token.strip()) - 1
        if index not in range(len(rows)):
            raise RuntimeError("Invalid benchmark model selection")
        indexes.append(index)
    return [rows[index] for index in dict.fromkeys(indexes)]


def _runtime_settings(config_path: Path, config_data: dict[str, Any], catalog: dict[str, Any], row: dict[str, Any]):
    entry = get_model(catalog, str(row["model_id"]))
    profile = get_profile(catalog, str(row["model_id"]), str(row["backend"]))
    selected = apply_selection(config_data, entry, profile, str(row["backend"]))
    return settings_from_data(config_path, selected, apply_environment=False)


def _warmup(runtime: BackendRuntime) -> None:
    row = {
        "id": "benchmark-warmup",
        "state": "The service is ready and healthy.",
        "question": "What is the service state?",
        "options": [
            {"id": "ready", "description": "The service is ready."},
            {"id": "not_ready", "description": "The service is not ready."},
        ],
    }
    runtime.score(row, "serial")
    runtime.score(row, "serial")


def _fmt_ms(value: Any) -> str:
    return "-" if value is None else f"{float(value):.1f}"


def run(args: argparse.Namespace) -> int:
    config_path, config_data = read_config_data(args.config)
    catalog_path = Path(str(config_data.get("model_catalog", "models.json")))
    if not catalog_path.is_absolute():
        catalog_path = config_path.parent / catalog_path
    catalog = load_catalog(catalog_path)
    suite_path = Path(args.suite).expanduser().resolve()
    suite = load_suite(suite_path)
    profiles = _select_profiles(_installed(config_path, config_data, catalog), args)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output).expanduser().resolve() if args.output else (config_path.parent / ".deqio" / "benchmarks" / stamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    print(f"[bench] suite={suite.get('name', suite_path.name)} cases={len(suite['cases'])}")
    print(f"[bench] models={len(profiles)} output={output_dir}")

    all_summaries: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8") as log:
        for profile_row in profiles:
            settings = _runtime_settings(config_path, config_data, catalog, profile_row)
            label = f"{settings.model_id}/{settings.backend}"
            print(f"\n[bench] Loading {label} ({settings.engine})")
            runtime = None
            model_results: list[BenchResult] = []
            load_started = time.perf_counter()
            try:
                runtime = BackendRuntime.load(settings)
                load_ms = (time.perf_counter() - load_started) * 1000.0
                _warmup(runtime)
                print(f"[bench] Ready {label} load={load_ms:.1f}ms; warmup complete")
                totals = {kind: sum(c["type"] == kind for c in suite["cases"]) for kind in ("noul", "choice", "shared")}
                seen = {"noul": 0, "choice": 0, "shared": 0}
                for case in suite["cases"]:
                    kind = str(case["type"])
                    seen[kind] += 1
                    result = _run_case(runtime, {"model_id": settings.model_id, "backend": settings.backend, "engine": settings.engine}, case)
                    model_results.append(result)
                    log.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")
                    log.flush()
                    status = "PASS" if result.passed else "FAIL"
                    detail = f"{result.correct}/{result.assertions}" if kind == "shared" else f"actual={result.actual} expected={result.expected}"
                    error = f" error={result.error}" if result.error else ""
                    print(f"[bench] {label:30} {kind:6} {seen[kind]:02}/{totals[kind]:02} {status:4} {detail} {_fmt_ms(result.latency_ms)}ms{error}")
            except Exception as error:
                load_ms = (time.perf_counter() - load_started) * 1000.0
                print(f"[bench] ERROR loading {label}: {error}")
            finally:
                if runtime is not None:
                    runtime.close()

            model_summary = summarize(model_results)
            all_summaries.append({
                "model_id": settings.model_id,
                "backend": settings.backend,
                "engine": settings.engine,
                "load_ms": load_ms,
                "summary": model_summary,
            })

    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": str(suite_path),
        "suite_name": suite.get("name"),
        "models": all_summaries,
    }
    summary_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nBenchmark summary")
    print(f"{'MODEL':30} {'TYPE':7} {'CASES':>7} {'ACC':>8} {'DEC ACC':>8} {'MEDIAN':>9} {'P95':>9}")
    print("-" * 91)
    for item in all_summaries:
        model = f"{item['model_id']}:{item['backend']}"
        for kind in ("overall", "noul", "choice", "shared"):
            stats = item["summary"].get(kind, {})
            cases = int(stats.get("cases", 0))
            acc = float(stats.get("case_accuracy", 0.0)) * 100.0
            dacc = float(stats.get("decision_accuracy", 0.0)) * 100.0
            label = "all" if kind == "overall" else kind
            print(
                f"{model:30} {label:7} {cases:7d} {acc:7.1f}% {dacc:7.1f}% "
                f"{_fmt_ms(stats.get('median_ms')):>7}ms {_fmt_ms(stats.get('p95_ms')):>7}ms"
            )
    print(f"\n[bench] results: {results_path}")
    print(f"[bench] summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deqio benchmark", description="Benchmark installed Deqio model profiles.")
    parser.add_argument("--config", help="Path to config.json (default: ./config.json)")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE), help="Benchmark suite JSON file")
    parser.add_argument("--output", help="Output directory (default: .deqio/benchmarks/<timestamp>)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Benchmark every installed host-compatible profile")
    group.add_argument("--model", action="append", help="Benchmark MODEL_ID:BACKEND; repeat to select several")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
