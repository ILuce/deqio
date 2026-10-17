"""``deqio benchmark tables`` — README / ``benchmarks/summary_*.md`` tables from completed benchmark runs (0.6.2).

    deqio benchmark tables --run 20261015T120300Z --models decider-0.8b:mps kev-0.8b:mlx [--out docs/table.md]
    deqio benchmark tables --summary path/to/summary.json --json

Reads the ``summary.json`` (schema 2) a run wrote — and its ``results.jsonl`` when present, for the cost
estimates — and renders, per run: the run line (id, suite, cases, scored decisions, Deqio version, date), the
quality table (case accuracy, decision accuracy, per-type accuracy), the performance table (median, P95,
throughput, estimated cost per 1 000 decisions for remote profiles) and the identity table (engine, runtime,
precision, source, model, pinned revision).  Rows are sorted by decision accuracy.  Nothing is interpolated:
a profile whose runtime failed to load is listed under "did not load", a missing metric prints ``—``, and a
Decision 3.0 MLX row produced by another Deqio forward than the current ``deqio-d3-mlx-forward/N`` is excluded
and named (its numbers are not comparable).  ``--json`` prints the same data as JSON (an additive automation
contract, like ``deqio benchmark compare --json``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .benchmark_store import benchmark_root
from .catalog import catalog_index, load_catalog
from .config import read_config_data

SCHEMA_VERSION = 1
D3_MLX_FORWARD_PREFIX = "deqio-d3-mlx-forward/"
TYPE_ORDER = ("noul", "choice", "shared", "score", "multi", "act", "systemone")


def _current_d3_forward() -> str:
    from .d3_mlx_sidecar import D3_MLX_FORWARD

    return D3_MLX_FORWARD


def load_run(source: str | Path) -> dict[str, Any]:
    """A run from its ``summary.json`` path or its directory (``results.jsonl`` next to it is optional)."""
    path = Path(source).expanduser()
    summary_path = path / "summary.json" if path.is_dir() else path
    if not summary_path.is_file():
        raise RuntimeError(f"benchmark summary not found: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON in {summary_path}: {error}") from error
    if not isinstance(summary, dict) or not isinstance(summary.get("models"), list):
        raise RuntimeError(f"{summary_path} is not a benchmark summary (no models array)")
    if summary.get("schema_version") != 2:
        raise RuntimeError(f"{summary_path}: schema_version {summary.get('schema_version')!r} is not supported (expected 2)")
    results: list[dict[str, Any]] = []
    results_path = summary_path.with_name("results.jsonl")
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                results.append(json.loads(line))
    return {"summary": summary, "results": results, "path": str(summary_path)}


def _parse_selector(value: str) -> tuple[str, str | None]:
    model_id, separator, backend = value.partition(":")
    if not model_id:
        raise RuntimeError(f"invalid model selector {value!r}; use MODEL_ID or MODEL_ID:BACKEND")
    return model_id, (backend or None) if separator else None


def _selected(model: dict[str, Any], selectors: list[tuple[str, str | None]] | None) -> bool:
    if not selectors:
        return True
    return any(model.get("model_id") == model_id and (backend is None or model.get("backend") == backend) for model_id, backend in selectors)


def _backend_label(model: dict[str, Any]) -> str:
    backend = str(model.get("backend", "?"))
    identity = model.get("profile_identity") if isinstance(model.get("profile_identity"), dict) else {}
    precision = identity.get("precision")
    quantization = identity.get("quantization") if isinstance(identity.get("quantization"), dict) else {}
    if backend == "remote":
        return "remote"
    if isinstance(precision, str) and precision.startswith("mlx-") and precision.endswith("bit"):
        return f"MLX {precision[4:-3]}-bit"
    quant = quantization.get("quantization") or quantization.get("kind")
    if backend == "gguf" and isinstance(quant, str):
        return f"GGUF {quant.upper()}" if quant.lower().startswith("q") else "GGUF"
    if isinstance(precision, str) and precision and precision != backend:
        return f"{backend.upper()} {precision.upper()}"
    return backend.upper()


def _cost_per_1k(model: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimated USD per 1 000 scored decisions from the per-row cost estimates; ``null`` when any row lacks one."""
    rows = [row for row in results if row.get("model_id") == model.get("model_id") and row.get("backend") == model.get("backend")]
    if not rows:
        return {"usd_per_1k_decisions": None, "basis": "no results.jsonl rows", "decisions": 0}
    amounts: list[float] = []
    decisions = 0
    for row in rows:
        estimate = row.get("cost_estimate")
        if not isinstance(estimate, dict) or estimate.get("amount") is None:
            return {"usd_per_1k_decisions": None, "basis": "cost estimate not reported for every decision", "decisions": 0}
        amounts.append(float(estimate["amount"]))
        decisions += int(row.get("assertions") or 1)
    if decisions == 0:
        return {"usd_per_1k_decisions": None, "basis": "no scored decisions", "decisions": 0}
    return {"usd_per_1k_decisions": sum(amounts) / decisions * 1000.0, "basis": "sum of per-decision estimates / scored decisions", "decisions": decisions}


def _revision(model: dict[str, Any]) -> str | None:
    identity = model.get("profile_identity") if isinstance(model.get("profile_identity"), dict) else {}
    artifacts = identity.get("artifacts") if isinstance(identity.get("artifacts"), list) else []
    revisions = [str(item.get("revision"))[:8] for item in artifacts if isinstance(item, dict) and item.get("revision")]
    return ", ".join(revisions) if revisions else None


def build_tables(run: dict[str, Any], *, catalog: dict[str, Any] | None = None, models: list[str] | None = None,
                 d3_mlx_forward: str | None = None) -> dict[str, Any]:
    summary, results = run["summary"], run["results"]
    labels = {model_id: str(entry.get("label") or model_id) for model_id, entry in catalog_index(catalog).items()} if catalog else {}
    selectors = [_parse_selector(value) for value in models] if models else None
    current_forward = d3_mlx_forward or _current_d3_forward()
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    not_loaded: list[dict[str, Any]] = []
    scored_decisions = 0
    for model in summary["models"]:
        if not isinstance(model, dict) or not _selected(model, selectors):
            continue
        key = f"{model.get('model_id')}:{model.get('backend')}"
        if model.get("load_error"):
            not_loaded.append({"profile": key, "load_error": str(model["load_error"])})
            continue
        runtime_identity = model.get("runtime_identity") if isinstance(model.get("runtime_identity"), dict) else {}
        metadata = runtime_identity.get("runtime_metadata") if isinstance(runtime_identity.get("runtime_metadata"), dict) else {}
        forward = metadata.get("forward")
        if isinstance(forward, str) and forward.startswith(D3_MLX_FORWARD_PREFIX) and forward != current_forward:
            excluded.append({"profile": key, "reason": f"Decision 3.0 MLX forward {forward}; only {current_forward} runs are reported"})
            continue
        overall = (model.get("summary") or {}).get("overall") or {}
        per_type = {kind: (model.get("summary") or {}).get(kind) for kind in TYPE_ORDER if isinstance((model.get("summary") or {}).get(kind), dict)}
        identity = model.get("profile_identity") if isinstance(model.get("profile_identity"), dict) else {}
        provider = runtime_identity.get("provider") if isinstance(runtime_identity.get("provider"), dict) else {}
        scored_decisions = max(scored_decisions, int(overall.get("assertions") or 0))
        rows.append({
            "profile": key,
            "model_id": model.get("model_id"),
            "backend": model.get("backend"),
            "label": labels.get(str(model.get("model_id")), str(model.get("model_id"))),
            "backend_label": _backend_label(model),
            "engine": model.get("engine"),
            "runtime": identity.get("runtime"),
            "runtime_version": identity.get("runtime_version"),
            "precision": identity.get("precision"),
            "source": identity.get("source"),
            "model": identity.get("model") if model.get("backend") != "remote" else (provider.get("model") or identity.get("model")),
            "revision": _revision(model),
            "endpoint": provider.get("endpoint") if model.get("backend") == "remote" else None,
            "canonical_id": identity.get("canonical_id"),
            "cases": overall.get("cases"),
            "case_accuracy": overall.get("case_accuracy"),
            "decisions": overall.get("assertions"),
            "decision_accuracy": overall.get("decision_accuracy"),
            "runtime_errors": overall.get("runtime_errors"),
            "median_ms": overall.get("median_ms"),
            "p95_ms": overall.get("p95_ms"),
            "throughput_decisions_per_s": overall.get("throughput_decisions_per_s"),
            "per_type": {kind: {"case_accuracy": value.get("case_accuracy"), "decision_accuracy": value.get("decision_accuracy")} for kind, value in per_type.items()},
            "cost": _cost_per_1k(model, results) if model.get("backend") == "remote" else None,
            "d3_mlx_forward": forward if isinstance(forward, str) else None,
        })
    rows.sort(key=lambda row: (-(row["decision_accuracy"] or 0.0), -(row["case_accuracy"] or 0.0), row["label"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": summary.get("run_id"),
        "suite_name": summary.get("suite_name"),
        "suite_cases": summary.get("suite_cases"),
        "scored_decisions": scored_decisions,
        "created_at": summary.get("created_at"),
        "deqio_version": next((str(m["runtime_identity"]["deqio_version"]) for m in summary["models"]
                               if isinstance(m, dict) and isinstance(m.get("runtime_identity"), dict) and m["runtime_identity"].get("deqio_version")), None),
        "rows": rows,
        "excluded": excluded,
        "not_loaded": not_loaded,
    }


def _pct(value: Any) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "—"


def _ms(value: Any) -> str:
    return f"{float(value):,.1f} ms" if isinstance(value, (int, float)) else "—"


def _rate(value: Any) -> str:
    return f"{float(value):.2f}/s" if isinstance(value, (int, float)) else "—"


def _cost(cost: dict[str, Any] | None) -> str:
    if cost is None:
        return "—"
    value = cost.get("usd_per_1k_decisions")
    return f"${value:.4f}" if isinstance(value, (int, float)) else "n/a"


def render_markdown(tables: dict[str, Any], *, title: str | None = None) -> str:
    lines: list[str] = []
    if title:
        lines += [f"### {title}", ""]
    date = str(tables.get("created_at") or "")[:10]
    lines += [
        "#### Run",
        "",
        f"- `{tables.get('run_id')}` · suite `{tables.get('suite_name')}` — {tables.get('suite_cases')} cases / "
        f"{tables.get('scored_decisions')} scored decisions · Deqio {tables.get('deqio_version')} · {date}",
    ]
    for item in tables["excluded"]:
        lines.append(f"- excluded: `{item['profile']}` — {item['reason']}")
    for item in tables["not_loaded"]:
        lines.append(f"- did not load: `{item['profile']}` — {item['load_error']}")
    errors = sum(int(row.get("runtime_errors") or 0) for row in tables["rows"])
    lines.append(f"- runtime errors across the listed profiles: {errors}")
    kinds = [kind for kind in TYPE_ORDER if any(kind in row["per_type"] for row in tables["rows"])]
    lines += ["", "#### Quality", "", "`Accuracy` is whole-case accuracy; `Decision accuracy` scores every decision/assertion "
              "(a Shared case holds several).", ""]
    header = ["Model", "Backend", "Accuracy", "Decision accuracy"] + [kind.capitalize() for kind in kinds] + (["Shared decision accuracy"] if "shared" in kinds else [])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| --- | :---: | " + " | ".join(["---:"] * (len(header) - 2)) + " |")
    for row in tables["rows"]:
        cells = [row["label"], row["backend_label"], _pct(row["case_accuracy"]), _pct(row["decision_accuracy"])]
        cells += [_pct(row["per_type"].get(kind, {}).get("case_accuracy")) for kind in kinds]
        if "shared" in kinds:
            cells.append(_pct(row["per_type"].get("shared", {}).get("decision_accuracy")))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "#### Performance", "", "Median and P95 are per-request latencies; throughput is scored decisions per second over the "
              "whole run; cost is the estimated USD per 1 000 scored decisions from the API's reported input tokens and its "
              "published input price (remote profiles only; an estimate, not a bill).", "",
              "| Model | Backend | Median | P95 | Throughput | Cost / 1k decisions |", "| --- | :---: | ---: | ---: | ---: | ---: |"]
    for row in tables["rows"]:
        lines.append(f"| {row['label']} | {row['backend_label']} | {_ms(row['median_ms'])} | {_ms(row['p95_ms'])} | "
                     f"{_rate(row['throughput_decisions_per_s'])} | {_cost(row['cost'])} |")
    lines += ["", "#### Identity", "", "| Model | Backend | Engine | Runtime | Precision | Source | Model / endpoint | Revision |",
              "| --- | :---: | --- | --- | --- | --- | --- | --- |"]
    for row in tables["rows"]:
        runtime = row["runtime"] or "—"
        if row["runtime_version"]:
            runtime = f"{runtime} {row['runtime_version']}"
        target = f"`{row['model']}`" if row["model"] else "—"
        if row["endpoint"]:
            target = f"{target} · `{row['endpoint']}`"
        lines.append(f"| {row['label']} | {row['backend_label']} | {row['engine'] or '—'} | {runtime} | {row['precision'] or '—'} | "
                     f"{row['source'] or '—'} | {target} | {row['revision'] or '—'} |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deqio benchmark tables", description="Render README tables from completed benchmark runs.")
    parser.add_argument("--config", help="Path to config.json (default: ./config.json); resolves --run ids in its workspace")
    parser.add_argument("--run", action="append", default=[], help="run id under .deqio/benchmarks (repeat for several runs)")
    parser.add_argument("--summary", action="append", default=[], help="path to a summary.json or a run directory (repeat)")
    parser.add_argument("--models", nargs="+", metavar="MODEL_ID[:BACKEND]", help="only these profiles (default: every profile of the run)")
    parser.add_argument("--title", help="heading printed above the tables")
    parser.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    parser.add_argument("--out", help="write to this file instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.run and not args.summary:
            raise RuntimeError("give at least one --run RUN_ID or --summary PATH")
        sources: list[Path] = []
        if args.run:
            config_path, _ = read_config_data(args.config)
            root = benchmark_root(config_path)
            for run_id in args.run:
                if not run_id or Path(run_id).name != run_id or not (root / run_id).is_dir():
                    raise RuntimeError(f"benchmark run not found: {run_id} (under {root})")
                sources.append(root / run_id)
        sources += [Path(value) for value in args.summary]
        catalog = load_catalog(Path(__file__).with_name("data") / "models.json")
        documents = [build_tables(load_run(source), catalog=catalog, models=args.models) for source in sources]
        if args.json:
            text = json.dumps(documents if len(documents) > 1 else documents[0], ensure_ascii=False, indent=2) + "\n"
        else:
            text = "\n".join(render_markdown(document, title=args.title) for document in documents)
        if args.out:
            out = Path(args.out).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"wrote {out}")
        else:
            sys.stdout.write(text)
        return 0
    except (RuntimeError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
