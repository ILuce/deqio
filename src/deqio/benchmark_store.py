from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def benchmark_root(config_path: Path) -> Path:
    return config_path.parent / ".deqio" / "benchmarks"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in benchmark file {path}: {error}") from error
    if not isinstance(data, dict):
        raise RuntimeError(f"Benchmark file must contain a JSON object: {path}")
    return data


def _run_dir(config_path: Path, run_id: str) -> Path:
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise FileNotFoundError(f"Benchmark run not found: {run_id}")
    root = benchmark_root(config_path)
    candidate = root / run_id
    if not candidate.is_dir():
        raise FileNotFoundError(f"Benchmark run not found: {run_id}")
    return candidate


def _file_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def list_benchmark_runs(config_path: Path) -> list[dict[str, Any]]:
    root = benchmark_root(config_path)
    if not root.is_dir():
        return []

    runs: list[dict[str, Any]] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        summary_path = directory / "summary.json"
        results_path = directory / "results.jsonl"
        if not summary_path.is_file() and not results_path.is_file():
            continue

        summary: dict[str, Any] | None = None
        summary_error: str | None = None
        if summary_path.is_file():
            try:
                summary = _read_json(summary_path)
            except RuntimeError as error:
                summary_error = str(error)

        created_at = None
        suite_name = None
        model_count = None
        if summary is not None:
            created_at = summary.get("created_at")
            suite_name = summary.get("suite_name")
            models = summary.get("models")
            if isinstance(models, list):
                model_count = len(models)

        fallback_path = summary_path if summary_path.is_file() else results_path
        runs.append({
            "id": directory.name,
            "created_at": str(created_at or _file_iso(fallback_path)),
            "suite_name": suite_name,
            "models": model_count,
            "results": _count_jsonl(results_path),
            "has_summary": summary_path.is_file(),
            "has_results": results_path.is_file(),
            "summary_error": summary_error,
        })

    runs.sort(key=lambda row: (str(row.get("created_at", "")), str(row["id"])), reverse=True)
    return runs


def read_benchmark_summary(config_path: Path, run_id: str) -> dict[str, Any]:
    path = _run_dir(config_path, run_id) / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"summary.json is missing for benchmark run: {run_id}")
    return _read_json(path)


def read_benchmark_results(config_path: Path, run_id: str) -> list[dict[str, Any]]:
    path = _run_dir(config_path, run_id) / "results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"results.jsonl is missing for benchmark run: {run_id}")

    results: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSONL in benchmark run {run_id} at line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"Benchmark result at line {line_number} must be a JSON object"
                )
            results.append(row)
    return results
