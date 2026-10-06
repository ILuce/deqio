from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .benchmark_identity import profile_identity_from_summary, profile_label
from .benchmark_store import list_benchmark_runs, read_benchmark_summary
from .config import read_config_data

COMPARISON_SCHEMA_VERSION = 1
_METRICS = (
    "case_accuracy",
    "decision_accuracy",
    "median_ms",
    "p95_ms",
    "throughput_decisions_per_s",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _metric_delta(left: Any, right: Any) -> dict[str, float | None] | None:
    a = _number(left)
    b = _number(right)
    if a is None or b is None:
        return None
    absolute = b - a
    percent = None if a == 0 else absolute / abs(a)
    return {"left": a, "right": b, "absolute": absolute, "percent": percent}


def _overall(model: dict[str, Any]) -> dict[str, Any]:
    summary = model.get("summary")
    if not isinstance(summary, dict):
        return {}
    value = summary.get("overall")
    return value if isinstance(value, dict) else {}


def _rank_map(models: list[dict[str, Any]]) -> dict[str, int]:
    ranked: list[tuple[tuple[Any, ...], str]] = []
    for model in models:
        identity = profile_identity_from_summary(model)
        if identity is None:
            continue
        stats = _overall(model)
        decision_accuracy = _number(stats.get("decision_accuracy"))
        case_accuracy = _number(stats.get("case_accuracy"))
        median_ms = _number(stats.get("median_ms"))
        ranked.append((
            (
                -(decision_accuracy if decision_accuracy is not None else -1.0),
                -(case_accuracy if case_accuracy is not None else -1.0),
                median_ms if median_ms is not None else float("inf"),
                str(identity["canonical_id"]),
            ),
            str(identity["canonical_id"]),
        ))
    ranked.sort(key=lambda item: item[0])
    return {canonical_id: index for index, (_, canonical_id) in enumerate(ranked, start=1)}


def _summary_types(model: dict[str, Any]) -> set[str]:
    summary = model.get("summary")
    if not isinstance(summary, dict):
        return set()
    return {str(key) for key, value in summary.items() if key != "overall" and isinstance(value, dict)}


def _compare_stats(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in _METRICS:
        delta = _metric_delta(left.get(metric), right.get(metric))
        if delta is not None:
            metrics[metric] = delta
    return metrics


def _profile_record(model: dict[str, Any]) -> dict[str, Any]:
    identity = profile_identity_from_summary(model)
    return {
        "model_id": model.get("model_id"),
        "backend": model.get("backend"),
        "engine": model.get("engine"),
        "label": profile_label(model),
        "canonical_id": identity.get("canonical_id") if identity else None,
        "identity": identity,
    }


def compare_documents(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_id: str,
    right_id: str,
) -> dict[str, Any]:
    left_models = [row for row in left.get("models", []) if isinstance(row, dict)]
    right_models = [row for row in right.get("models", []) if isinstance(row, dict)]
    left_by_id = {
        str(identity["canonical_id"]): row
        for row in left_models
        if (identity := profile_identity_from_summary(row)) is not None
    }
    right_by_id = {
        str(identity["canonical_id"]): row
        for row in right_models
        if (identity := profile_identity_from_summary(row)) is not None
    }
    common_ids = sorted(set(left_by_id) & set(right_by_id))
    left_ranks = _rank_map(left_models)
    right_ranks = _rank_map(right_models)

    common: list[dict[str, Any]] = []
    for canonical_id in common_ids:
        left_model = left_by_id[canonical_id]
        right_model = right_by_id[canonical_id]
        left_stats = _overall(left_model)
        right_stats = _overall(right_model)
        type_names = sorted(_summary_types(left_model) & _summary_types(right_model))
        breakdown: dict[str, Any] = {}
        for kind in type_names:
            left_kind = left_model.get("summary", {}).get(kind, {})
            right_kind = right_model.get("summary", {}).get(kind, {})
            metrics = _compare_stats(left_kind, right_kind)
            if metrics:
                breakdown[kind] = metrics
        left_rank = left_ranks.get(canonical_id)
        right_rank = right_ranks.get(canonical_id)
        common.append({
            **_profile_record(right_model),
            "metrics": _compare_stats(left_stats, right_stats),
            "rank": {
                "left": left_rank,
                "right": right_rank,
                "shift": (left_rank - right_rank) if left_rank is not None and right_rank is not None else None,
            },
            "breakdown": breakdown,
        })

    common.sort(key=lambda row: (str(row.get("model_id")), str(row.get("backend")), str(row.get("canonical_id"))))
    only_left = [_profile_record(left_by_id[key]) for key in sorted(set(left_by_id) - set(right_by_id))]
    only_right = [_profile_record(right_by_id[key]) for key in sorted(set(right_by_id) - set(left_by_id))]
    legacy_left = [_profile_record(row) for row in left_models if profile_identity_from_summary(row) is None]
    legacy_right = [_profile_record(row) for row in right_models if profile_identity_from_summary(row) is None]

    left_suite = left.get("suite_name")
    right_suite = right.get("suite_name")
    warnings: list[str] = []
    if left_suite != right_suite:
        warnings.append(
            "Benchmark suites differ; metric deltas are factual differences between runs and are not interpreted as pure quality/language deltas."
        )
    if legacy_left or legacy_right:
        warnings.append(
            "One or both runs contain legacy profiles without canonical runtime identity; those profiles are not treated as common."
        )

    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "left": {
            "run_id": left_id,
            "created_at": left.get("created_at"),
            "suite_name": left_suite,
            "schema_version": left.get("schema_version"),
        },
        "right": {
            "run_id": right_id,
            "created_at": right.get("created_at"),
            "suite_name": right_suite,
            "schema_version": right.get("schema_version"),
        },
        "same_suite": left_suite == right_suite,
        "common_profiles": common,
        "only_in_left": only_left,
        "only_in_right": only_right,
        "legacy_unverified_left": legacy_left,
        "legacy_unverified_right": legacy_right,
        "warnings": warnings,
    }


def compare_runs(config_path: Path, left_id: str, right_id: str) -> dict[str, Any]:
    return compare_documents(
        read_benchmark_summary(config_path, left_id),
        read_benchmark_summary(config_path, right_id),
        left_id=left_id,
        right_id=right_id,
    )


def _choose_run(runs: list[dict[str, Any]], label: str, *, exclude: str | None = None) -> str | None:
    available = [run for run in runs if run.get("has_summary") and run.get("id") != exclude]
    if not available:
        raise RuntimeError("At least two benchmark runs with summary.json are required")
    print(f"Available benchmark runs for {label}:")
    for index, run in enumerate(available, start=1):
        print(
            f"  {index}. {run['id']} — {run.get('suite_name') or '?'} — "
            f"{run.get('models') if run.get('models') is not None else '?'} profiles"
        )
    print("  Q. quit")
    value = input(f"Select benchmark {label} [1-{len(available)}, q]: ").strip().lower()
    if value in {"q", "quit", "exit"}:
        return None
    try:
        index = int(value) - 1
    except ValueError as error:
        raise RuntimeError("Invalid benchmark selection") from error
    if index not in range(len(available)):
        raise RuntimeError("Invalid benchmark selection")
    return str(available[index]["id"])


def _fmt_percent(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100.0:.1f}%"


def _fmt_ms(value: Any) -> str:
    return "-" if value is None else f"{float(value):.1f}ms"


def _fmt_throughput(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f}/s"


def _fmt_metric_delta(metric: dict[str, Any] | None, kind: str) -> str:
    if not metric:
        return "-"
    absolute = metric.get("absolute")
    percent = metric.get("percent")
    if kind in {"case_accuracy", "decision_accuracy"}:
        absolute_text = "-" if absolute is None else f"{float(absolute) * 100.0:+.1f} pp"
    elif kind in {"median_ms", "p95_ms"}:
        absolute_text = "-" if absolute is None else f"{float(absolute):+.1f}ms"
    elif kind == "throughput_decisions_per_s":
        absolute_text = "-" if absolute is None else f"{float(absolute):+.2f}/s"
    else:
        absolute_text = "-" if absolute is None else f"{float(absolute):+.3f}"
    if percent is None:
        return absolute_text
    return f"{absolute_text} ({float(percent) * 100.0:+.1f}%)"


def _print_comparison(document: dict[str, Any]) -> None:
    left = document["left"]
    right = document["right"]
    print(f"Benchmark A: {left['run_id']} ({left.get('suite_name') or '?'})")
    print(f"Benchmark B: {right['run_id']} ({right.get('suite_name') or '?'})")
    for warning in document.get("warnings", []):
        print(f"WARNING: {warning}")
    print("\nCommon profiles")
    rows = document.get("common_profiles", [])
    if not rows:
        print("  none")
    else:
        print(
            f"{'PROFILE':34} {'ACC A':>7} {'ACC B':>7} {'DACC A':>7} {'DACC B':>7} "
            f"{'MED A':>9} {'MED B':>9} {'P95 A':>9} {'P95 B':>9} {'THR A':>9} {'THR B':>9} {'RANK':>11}"
        )
        print("-" * 145)
        for row in rows:
            metrics = row.get("metrics", {})
            acc = metrics.get("case_accuracy", {})
            decision = metrics.get("decision_accuracy", {})
            median = metrics.get("median_ms", {})
            p95 = metrics.get("p95_ms", {})
            throughput = metrics.get("throughput_decisions_per_s", {})
            rank = row.get("rank", {})
            shift = rank.get("shift")
            rank_text = "-" if rank.get("left") is None else f"{rank.get('left')}->{rank.get('right')} ({shift:+d})"
            print(
                f"{str(row.get('label'))[:34]:34} {_fmt_percent(acc.get('left')):>7} {_fmt_percent(acc.get('right')):>7} "
                f"{_fmt_percent(decision.get('left')):>7} {_fmt_percent(decision.get('right')):>7} "
                f"{_fmt_ms(median.get('left')):>9} {_fmt_ms(median.get('right')):>9} "
                f"{_fmt_ms(p95.get('left')):>9} {_fmt_ms(p95.get('right')):>9} "
                f"{_fmt_throughput(throughput.get('left')):>9} {_fmt_throughput(throughput.get('right')):>9} {rank_text:>11}"
            )
            deltas = []
            for name, label in (
                ("case_accuracy", "acc"),
                ("decision_accuracy", "decision"),
                ("median_ms", "median"),
                ("p95_ms", "p95"),
                ("throughput_decisions_per_s", "throughput"),
            ):
                metric = metrics.get(name)
                if metric:
                    deltas.append(f"{label} Δ={_fmt_metric_delta(metric, name)}")
            if deltas:
                print(f"  deltas: {', '.join(deltas)}; rank shift={shift if shift is not None else '-'}")

        print("\nBreakdown by decision type")
        any_breakdown = False
        for row in rows:
            for kind, metrics in sorted((row.get("breakdown") or {}).items()):
                any_breakdown = True
                acc = metrics.get("case_accuracy", {})
                decision = metrics.get("decision_accuracy", {})
                median = metrics.get("median_ms", {})
                print(
                    f"  {row.get('label')} / {kind}: "
                    f"acc {_fmt_percent(acc.get('left'))} -> {_fmt_percent(acc.get('right'))} "
                    f"(Δ {_fmt_percent(acc.get('absolute'))}), "
                    f"decision {_fmt_percent(decision.get('left'))} -> {_fmt_percent(decision.get('right'))}, "
                    f"median {_fmt_ms(median.get('left'))} -> {_fmt_ms(median.get('right'))}"
                )
        if not any_breakdown:
            print("  none")
    for key, title in (("only_in_left", "Only in A"), ("only_in_right", "Only in B")):
        print(f"\n{title}")
        values = document.get(key, [])
        if not values:
            print("  none")
        else:
            for row in values:
                print(f"  - {row['label']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deqio benchmark compare", description="Compare two completed Deqio benchmark runs.")
    parser.add_argument("--config", help="Path to config.json (default: ./config.json)")
    parser.add_argument("--left", help="Benchmark run ID for side A")
    parser.add_argument("--right", help="Benchmark run ID for side B")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--output", help="Write comparison JSON to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config_path, _ = read_config_data(args.config)
        if bool(args.left) != bool(args.right):
            raise RuntimeError("--left and --right must be provided together")
        left_id = args.left
        right_id = args.right
        if not left_id:
            runs = list_benchmark_runs(config_path)
            left_id = _choose_run(runs, "A")
            if left_id is None:
                print("Cancelled.")
                return 0
            right_id = _choose_run(runs, "B", exclude=left_id)
            if right_id is None:
                print("Cancelled.")
                return 0
        document = compare_runs(config_path, str(left_id), str(right_id))
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(document, ensure_ascii=False, indent=2))
        else:
            _print_comparison(document)
            if args.output:
                print(f"\nComparison JSON: {Path(args.output).expanduser().resolve()}")
        return 0
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        return 2
