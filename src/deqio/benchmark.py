from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from uuid import uuid4
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends import BackendRuntime
from .benchmark_compare import main as compare_main
from .benchmark_identity import build_profile_identity
from .catalog import apply_selection, get_model, get_profile, load_catalog
from .config import read_config_data, settings_from_data
from .installations import installed_profiles
from .watch_store import WatchStore
from .process_lock import process_lock
from .runtime_control import control_request, discover_server


DEFAULT_BENCHMARK_DIR = Path("benchmarks")


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


def discover_suites(directory: Path) -> list[dict[str, Any]]:
    """Discover valid benchmark suites from one editable workspace directory."""
    if not directory.is_dir():
        return []
    suites: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"), key=lambda item: item.name.lower()):
        try:
            suite = load_suite(path)
        except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as error:
            print(f"[bench] Skipping invalid suite {path.name}: {error}", file=sys.stderr)
            continue
        suites.append({
            "path": path.resolve(),
            "label": str(suite.get("label") or suite.get("name") or path.stem),
            "name": str(suite.get("name") or path.stem),
            "description": str(suite.get("description") or ""),
            "cases": len(suite["cases"]),
        })
    return suites


def _select_suite(config_path: Path, args: argparse.Namespace) -> Path | None:
    if args.suite:
        return Path(args.suite).expanduser().resolve()

    directory = (config_path.parent / DEFAULT_BENCHMARK_DIR).resolve()
    suites = discover_suites(directory)
    if not suites:
        raise RuntimeError(f"No valid benchmark suites found in {directory}")

    print("Available benchmark suites:")
    for index, suite in enumerate(suites, start=1):
        print(f"  {index}. {suite['label']} — {suite['cases']} cases")
    print("  Q. quit")
    value = input(f"Select benchmark [1-{len(suites)}, q]: ").strip().lower()
    if value in {"q", "quit", "exit"}:
        return None
    try:
        index = int(value) - 1
    except ValueError as error:
        raise RuntimeError("Invalid benchmark selection") from error
    if index not in range(len(suites)):
        raise RuntimeError("Invalid benchmark selection")
    return Path(suites[index]["path"])


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


def _runtime_identity(runtime: BackendRuntime, model: dict[str, str]) -> dict[str, Any]:
    identity = getattr(runtime, "identity_snapshot", None)
    if callable(identity):
        value = identity()
        if isinstance(value, dict):
            return value
    return {
        "engine": model["engine"],
        "model_id": model["model_id"],
        "backend": model["backend"],
        "runtime_instance_id": getattr(runtime, "runtime_instance_id", None),
    }


def _probability_map(raw: dict[str, Any]) -> dict[str, float]:
    return {
        str(option_id): float(probability)
        for option_id, probability in zip(raw.get("option_ids", []), raw.get("probabilities", []))
    }


def _watch_response(raw: dict[str, Any], *, identity: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    decision, top = _choice(raw)
    response = {
        "id": raw.get("id"),
        "decision": decision,
        "probabilities": _probability_map(raw),
        "top_probability": top,
        "input_tokens": raw.get("input_tokens"),
        "input_tokens_source": raw.get("input_tokens_source", "unknown"),
        "timing": {"total_ms": latency_ms},
        "prompt_sha256": raw.get("prompt_sha256"),
        "probability_status": raw.get("probability_status"),
        "provenance": {"runtime": identity},
    }
    score = raw.get("score_provenance")
    if isinstance(score, dict):
        response["score_provenance"] = score
    return response


def _append_benchmark_watch(
    watch: WatchStore | None,
    *,
    endpoint: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    status_code: int,
    request_id: str,
    mode: str,
    decisions: int,
    model: dict[str, str],
    identity: dict[str, Any],
    latency_ms: float,
    expected_session_id: str | None = None,
) -> None:
    if watch is None or expected_session_id is None:
        return
    response_payload = dict(response_payload)
    provenance = response_payload.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    provenance.setdefault("runtime", identity)
    response_payload["provenance"] = provenance
    try:
        watch.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "benchmark",
            "endpoint": endpoint,
            "method": "INTERNAL",
            "status_code": int(status_code),
            "request_id": request_id,
            "engine": identity.get("engine", model["engine"]),
            "model_id": identity.get("model_id", model["model_id"]),
            "backend": identity.get("backend", model["backend"]),
            "runtime_instance_id": identity.get("runtime_instance_id"),
            "mode": mode,
            "decisions": int(decisions),
            "decision": response_payload.get("decision"),
            "top_probability": response_payload.get("top_probability"),
            "latency_ms": float(latency_ms),
            "input_tokens": response_payload.get("input_tokens"),
            "request": request_payload,
            "response": response_payload,
        }, expected_session_id=expected_session_id)
    except Exception as error:
        print(f"[bench] Watch event write failed: {error}", file=sys.stderr)


def _benchmark_watch_session_token(watch: WatchStore | None) -> str | None:
    if watch is None:
        return None
    try:
        return watch.session_token()
    except Exception as error:
        print(f"[bench] Watch session lookup failed: {error}", file=sys.stderr)
        return None


def _run_case(
    runtime: BackendRuntime,
    model: dict[str, str],
    case: dict[str, Any],
    *,
    watch: WatchStore | None = None,
) -> BenchResult:
    kind = str(case["type"])
    case_id = str(case["id"])
    identity = _runtime_identity(runtime, model)
    watch_session_id = _benchmark_watch_session_token(watch)
    started = time.perf_counter()
    endpoint = f"/v1/{kind}"
    request_payload: dict[str, Any]
    decisions = 1
    try:
        if kind == "noul":
            row = {"id": case_id, "state": case["state"], "question": str(case["question"])}
            request_payload = dict(row)
            request_payload["mode"] = "serial"
            raw = runtime.score_noul(row, "serial")
            actual, top = _choice(raw)
            expected = str(case["expected"])
            latency_ms = float(raw.get("total_seconds", time.perf_counter() - started)) * 1000.0
            response_payload = _watch_response(raw, identity=identity, latency_ms=latency_ms)
            _append_benchmark_watch(
                watch, endpoint=endpoint, request_payload=request_payload,
                response_payload=response_payload, status_code=200, request_id=case_id,
                mode="serial", decisions=1, model=model, identity=identity, latency_ms=latency_ms,
                expected_session_id=watch_session_id,
            )
            return BenchResult(**model, case_id=case_id, kind=kind, passed=actual == expected,
                               expected=expected, actual=actual, latency_ms=latency_ms,
                               assertions=1, correct=int(actual == expected), top_probability=top)

        if kind == "choice":
            row = _choice_row(case)
            request_payload = {**row, "mode": "serial"}
            raw = runtime.score(row, "serial")
            actual, top = _choice(raw)
            expected = str(case["expected"])
            latency_ms = float(raw.get("total_seconds", time.perf_counter() - started)) * 1000.0
            response_payload = _watch_response(raw, identity=identity, latency_ms=latency_ms)
            _append_benchmark_watch(
                watch, endpoint=endpoint, request_payload=request_payload,
                response_payload=response_payload, status_code=200, request_id=case_id,
                mode="serial", decisions=1, model=model, identity=identity, latency_ms=latency_ms,
                expected_session_id=watch_session_id,
            )
            return BenchResult(**model, case_id=case_id, kind=kind, passed=actual == expected,
                               expected=expected, actual=actual, latency_ms=latency_ms,
                               assertions=1, correct=int(actual == expected), top_probability=top)

        rows = []
        expected: list[str] = []
        request_decisions: list[dict[str, Any]] = []
        for decision in case["decisions"]:
            row = {
                "id": str(decision["id"]),
                "state": case["state"],
                "question": str(decision["question"]),
                "options": list(decision["options"]),
            }
            rows.append(row)
            request_decisions.append({
                "id": row["id"],
                "question": row["question"],
                "options": row["options"],
            })
            expected.append(str(decision["expected"]))
        decisions = len(rows)
        request_payload = {"id": case_id, "state": case["state"], "decisions": request_decisions}
        raw_results, timing = runtime.score_shared(rows)
        actual = [_choice(raw)[0] for raw in raw_results]
        correct = sum(a == e for a, e in zip(actual, expected))
        latency_ms = float(timing.get("total_seconds", time.perf_counter() - started)) * 1000.0
        response_results = [
            _watch_response(raw, identity=identity, latency_ms=float(raw.get("total_seconds", 0.0)) * 1000.0)
            for raw in raw_results
        ]
        response_payload = {
            "results": response_results,
            "shared_timing": {"total_ms": latency_ms, "batch_size": len(response_results)},
            "provenance": {"runtime": identity},
            "decision": f"{len(response_results)} decisions",
        }
        token_values = [
            int(item["input_tokens"])
            for item in response_results
            if isinstance(item.get("input_tokens"), int)
        ]
        if token_values:
            response_payload["input_tokens"] = sum(token_values)
        _append_benchmark_watch(
            watch, endpoint=endpoint, request_payload=request_payload,
            response_payload=response_payload, status_code=200, request_id=case_id,
            mode="shared", decisions=decisions, model=model, identity=identity, latency_ms=latency_ms,
            expected_session_id=watch_session_id,
        )
        return BenchResult(**model, case_id=case_id, kind=kind, passed=correct == len(expected),
                           expected=expected, actual=actual, latency_ms=latency_ms,
                           assertions=len(expected), correct=correct)
    except Exception as error:
        latency_ms = (time.perf_counter() - started) * 1000.0
        if kind == "shared":
            request_payload = {
                "id": case_id,
                "state": case.get("state"),
                "decisions": list(case.get("decisions", [])),
            }
            decisions = max(1, len(case.get("decisions", [])))
        else:
            request_payload = {
                "id": case_id,
                "state": case.get("state"),
                "question": case.get("question"),
                "options": case.get("options") if kind == "choice" else None,
                "mode": "serial",
            }
        _append_benchmark_watch(
            watch, endpoint=endpoint, request_payload=request_payload,
            response_payload={"detail": str(error), "provenance": {"runtime": identity}},
            status_code=500, request_id=case_id, mode="shared" if kind == "shared" else "serial",
            decisions=decisions, model=model, identity=identity, latency_ms=latency_ms,
            expected_session_id=watch_session_id,
        )
        return BenchResult(**model, case_id=case_id, kind=kind, passed=False,
                           expected=case.get("expected") or [d.get("expected") for d in case.get("decisions", [])],
                           actual=None, latency_ms=latency_ms,
                           assertions=max(1, len(case.get("decisions", []))), correct=0, error=str(error))


def summarize(results: list[BenchResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    groups: dict[str, list[BenchResult]] = {}
    for result in results:
        groups.setdefault(result.kind, []).append(result)

    def stats_for(rows: list[BenchResult]) -> dict[str, Any]:
        latencies = [r.latency_ms for r in rows if r.latency_ms is not None]
        successful_latencies = [r.latency_ms for r in rows if r.latency_ms is not None and r.error is None]
        assertions = sum(r.assertions for r in rows)
        correct = sum(r.correct for r in rows)
        completed_assertions = sum(r.assertions for r in rows if r.error is None)
        successful_seconds = sum(float(value) for value in successful_latencies) / 1000.0
        runtime_errors = sum(r.error is not None for r in rows)
        return {
            "cases": len(rows),
            "passed_cases": sum(r.passed for r in rows),
            "case_accuracy": (sum(r.passed for r in rows) / len(rows)) if rows else 0.0,
            "assertions": assertions,
            "correct": correct,
            "decision_accuracy": (correct / assertions) if assertions else 0.0,
            "runtime_errors": runtime_errors,
            "completed_assertions": completed_assertions,
            "mean_ms": statistics.fmean(latencies) if latencies else None,
            "median_ms": statistics.median(latencies) if latencies else None,
            "p95_ms": _percentile(latencies, 0.95),
            "throughput_decisions_per_s": (
                completed_assertions / successful_seconds
                if completed_assertions and successful_seconds > 0
                else None
            ),
        }

    summary["overall"] = stats_for(results)
    for kind, rows in groups.items():
        summary[kind] = stats_for(rows)
    return summary


def _installed(config_path: Path, config_data: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in installed_profiles(
        config_path=config_path,
        config_data=config_data,
        catalog=catalog,
        active_model_id=str(config_data.get("model_id", "")),
        active_backend=str(config_data.get("backend", "")),
    ) if row["installed"] and row["host_compatible"]]


def _select_profiles(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]] | None:
    if not rows:
        raise RuntimeError("No installed model profiles are available on this host")
    if args.all:
        return rows
    if args.model:
        selected = []
        seen: set[tuple[str, str]] = set()
        for value in args.model:
            if ":" not in value:
                raise RuntimeError("--model must use MODEL_ID:BACKEND, for example decider-0.8b:mps")
            model_id, backend = value.rsplit(":", 1)
            match = next((row for row in rows if row["model_id"] == model_id and row["backend"] == backend), None)
            if match is None:
                raise RuntimeError(f"Installed profile not found: {value}")
            key = (str(match["model_id"]), str(match["backend"]))
            if key in seen:
                continue
            seen.add(key)
            selected.append(match)
        return selected

    print("Installed model profiles:")
    for index, row in enumerate(rows, start=1):
        print(f"  {index}. {row['label']} — {row['backend']} ({row['engine']})")
    print("  A. all installed profiles")
    print("  Q. quit")
    value = input("Select models [A, comma-separated numbers, or q]: ").strip().lower()
    if value in {"q", "quit", "exit"}:
        return None
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


@contextmanager
def _benchmark_workspace_lock(config_path: Path):
    root = config_path.parent / ".deqio"
    lock_dir = root / "benchmark.lock"
    model_lock = root / "model-management.lock"
    root.mkdir(parents=True, exist_ok=True)
    message = "Another Deqio benchmark is already running in this workspace (pid={pid})"
    model_message = "A Deqio model-management operation is already running in this workspace (pid={pid})"
    # Benchmarks execute installed runtimes directly, so runtime/model files
    # must not be replaced or deleted underneath a benchmark workload.
    with process_lock(model_lock, timeout=2.0, live_owner_error=model_message):
        with process_lock(lock_dir, timeout=2.0, live_owner_error=message) as owner:
            yield owner


def _suspend_live_server(config_path: Path, *, timeout: float) -> dict[str, Any] | None:
    server = discover_server(config_path)
    if server is None:
        return None
    lease_id = uuid4().hex
    print(f"[bench] Active Deqio server detected: pid={server['pid']} {server['base_url']}")
    print("[bench] Suspending server inference runtime; HTTP/UI/Watch remain available")
    response = control_request(
        server,
        "suspend",
        {
            "lease_id": lease_id,
            "owner_pid": os.getpid(),
            "reason": "benchmark",
        },
        timeout=timeout,
    )
    previous = response.get("previous_profile") or {}
    if isinstance(previous, dict) and previous.get("model_id"):
        print(f"[bench] Server runtime suspended: {previous.get('model_id')}/{previous.get('backend')}")
    else:
        print("[bench] Server inference runtime suspended")
    return {"server": server, "lease_id": lease_id, "previous_profile": previous}


def _resume_live_server(lease: dict[str, Any], *, timeout: float) -> None:
    previous = lease.get("previous_profile") or {}
    if isinstance(previous, dict) and previous.get("model_id"):
        print(f"[bench] Restoring server runtime: {previous.get('model_id')}/{previous.get('backend')}")
    else:
        print("[bench] Releasing server inference suspension")
    response = control_request(
        lease["server"],
        "resume",
        {"lease_id": lease["lease_id"]},
        timeout=timeout,
    )
    active = response.get("active") or {}
    if isinstance(active, dict) and active.get("model_id"):
        print(f"[bench] Server runtime restored: {active.get('model_id')}/{active.get('backend')}")
    else:
        print("[bench] Server runtime suspension released")


def run(args: argparse.Namespace) -> int:
    config_path, config_data = read_config_data(args.config)
    catalog_path = Path(str(config_data.get("model_catalog", "models.json")))
    if not catalog_path.is_absolute():
        catalog_path = config_path.parent / catalog_path
    catalog = load_catalog(catalog_path)
    watch = WatchStore(config_path)
    if _benchmark_watch_session_token(watch) is None:
        watch = None
    suite_path = _select_suite(config_path, args)
    if suite_path is None:
        print("Cancelled.")
        return 0
    suite = load_suite(suite_path)
    profiles = _select_profiles(_installed(config_path, config_data, catalog), args)
    if profiles is None:
        print("Cancelled.")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output).expanduser().resolve() if args.output else (config_path.parent / ".deqio" / "benchmarks" / stamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    print(f"[bench] suite={suite.get('name', suite_path.name)} cases={len(suite['cases'])}")
    print(f"[bench] models={len(profiles)} output={output_dir}")

    sidecar_timeout = float(config_data.get("sidecar_startup_seconds", 1800))
    lease: dict[str, Any] | None = None
    all_summaries: list[dict[str, Any]] = []
    with _benchmark_workspace_lock(config_path):
        try:
            lease = _suspend_live_server(
                config_path,
                timeout=min(max(sidecar_timeout, 60.0), 1800.0),
            )
            with results_path.open("w", encoding="utf-8") as log:
                for profile_row in profiles:
                    settings = _runtime_settings(config_path, config_data, catalog, profile_row)
                    label = f"{settings.model_id}/{settings.backend}"
                    print(f"\n[bench] Loading {label} ({settings.engine})")
                    runtime = None
                    model_results: list[BenchResult] = []
                    load_started = time.perf_counter()
                    runtime_identity: dict[str, Any] | None = None
                    profile_identity: dict[str, Any] | None = None
                    load_error: str | None = None
                    try:
                        runtime = BackendRuntime.load(settings)
                        load_ms = (time.perf_counter() - load_started) * 1000.0
                        _warmup(runtime)
                        runtime_identity = _runtime_identity(
                            runtime,
                            {"model_id": settings.model_id, "backend": settings.backend, "engine": settings.engine},
                        )
                        profile_identity = build_profile_identity(
                            model_id=settings.model_id,
                            backend=settings.backend,
                            engine=settings.engine,
                            runtime_identity=runtime_identity,
                        )
                        print(f"[bench] Ready {label} load={load_ms:.1f}ms; warmup complete")
                        totals = {
                            kind: sum(c["type"] == kind for c in suite["cases"])
                            for kind in ("noul", "choice", "shared")
                        }
                        seen = {"noul": 0, "choice": 0, "shared": 0}
                        for case in suite["cases"]:
                            kind = str(case["type"])
                            seen[kind] += 1
                            result = _run_case(
                                runtime,
                                {
                                    "model_id": settings.model_id,
                                    "backend": settings.backend,
                                    "engine": settings.engine,
                                },
                                case,
                                watch=watch,
                            )
                            model_results.append(result)
                            row = result.as_dict()
                            row["profile_canonical_id"] = profile_identity["canonical_id"]
                            log.write(json.dumps(row, ensure_ascii=False) + "\n")
                            log.flush()
                            status = "PASS" if result.passed else "FAIL"
                            detail = (
                                f"{result.correct}/{result.assertions}"
                                if kind == "shared"
                                else f"actual={result.actual} expected={result.expected}"
                            )
                            error = f" error={result.error}" if result.error else ""
                            print(
                                f"[bench] {label:30} {kind:6} {seen[kind]:02}/{totals[kind]:02} "
                                f"{status:4} {detail} {_fmt_ms(result.latency_ms)}ms{error}"
                            )
                    except Exception as error:
                        load_ms = (time.perf_counter() - load_started) * 1000.0
                        load_error = str(error)
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
                        "load_error": load_error,
                        "profile_identity": profile_identity,
                        "runtime_identity": runtime_identity,
                        "summary": model_summary,
                    })
        finally:
            if lease is not None:
                try:
                    _resume_live_server(lease, timeout=max(sidecar_timeout + 60.0, 120.0))
                except Exception as error:
                    print(
                        "[bench] WARNING: benchmark completed but the server runtime could not be "
                        f"restored automatically: {error}",
                        file=sys.stderr,
                    )

    document = {
        "schema_version": 2,
        "run_id": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": str(suite_path),
        "suite_name": suite.get("name"),
        "suite_cases": len(suite["cases"]),
        "models": all_summaries,
    }
    summary_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nBenchmark summary")
    print(
        f"{'MODEL':30} {'TYPE':7} {'CASES':>7} {'ACC':>8} {'DEC ACC':>8} "
        f"{'ERR':>5} {'MEDIAN':>9} {'P95':>9} {'THR/S':>9}"
    )
    print("-" * 109)
    for item in all_summaries:
        model = f"{item['model_id']}:{item['backend']}"
        for kind in ("overall", "noul", "choice", "shared"):
            stats = item["summary"].get(kind, {})
            cases = int(stats.get("cases", 0))
            acc = float(stats.get("case_accuracy", 0.0)) * 100.0
            dacc = float(stats.get("decision_accuracy", 0.0)) * 100.0
            errors = int(stats.get("runtime_errors", 0))
            throughput = stats.get("throughput_decisions_per_s")
            throughput_text = "-" if throughput is None else f"{float(throughput):.2f}"
            label = "all" if kind == "overall" else kind
            print(
                f"{model:30} {label:7} {cases:7d} {acc:7.1f}% {dacc:7.1f}% {errors:5d} "
                f"{_fmt_ms(stats.get('median_ms')):>7}ms {_fmt_ms(stats.get('p95_ms')):>7}ms "
                f"{throughput_text:>9}"
            )
    print(f"\n[bench] results: {results_path}")
    print(f"[bench] summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deqio benchmark", description="Benchmark installed Deqio model profiles.")
    parser.add_argument("--config", help="Path to config.json (default: ./config.json)")
    parser.add_argument(
        "--suite",
        help="Benchmark suite JSON file; omit to choose from ./benchmarks/*.json",
    )
    parser.add_argument("--output", help="Output directory (default: .deqio/benchmarks/<timestamp>)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Benchmark every installed host-compatible profile")
    group.add_argument("--model", action="append", help="Benchmark MODEL_ID:BACKEND; repeat to select several")
    return parser


def main(argv: list[str] | None = None) -> int:
    values = list(argv or [])
    if values and values[0] == "compare":
        return compare_main(values[1:])
    parser = build_parser()
    args = parser.parse_args(values if argv is not None else None)
    try:
        return run(args)
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
