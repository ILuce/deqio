import json
from pathlib import Path

import pytest

from deqio.config import load_settings
from deqio.server import app, format_result, format_shared_timing, percentile, state_hash
from deqio.ui import DASHBOARD


def test_public_routes_are_registered() -> None:
    routes = {route.path for route in app.routes}

    assert "/" in routes
    assert "/health" in routes
    assert "/ui" in routes
    assert "/v1/stats" in routes
    assert "/v1/recent" in routes
    assert "/v1/noul" in routes
    assert "/v1/choice" in routes
    assert "/v1/decision" in routes
    assert "/v1/shared" in routes
    assert "/v1/cache/clear" in routes
    assert "/v1/models" in routes


def test_percentile() -> None:
    assert percentile([], 0.5) is None
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0


def test_state_hash_is_stable_for_dict_key_order() -> None:
    assert state_hash({"b": 2, "a": 1}) == state_hash({"a": 1, "b": 2})


def test_format_result() -> None:
    result = format_result(
        {
            "id": "example",
            "option_ids": ["yes", "no"],
            "probabilities": [0.8, 0.2],
            "option_logits": [2.0, 1.0],
            "input_tokens": 42,
            "total_seconds": 0.125,
            "forward_seconds": 0.100,
            "cache_hit": True,
            "prompt_sha256": "abc",
            "probability_status": "conditional",
        }
    )

    assert result["decision"] == "yes"
    assert result["probabilities"] == {"yes": 0.8, "no": 0.2}
    assert result["option_logits"] == {"yes": 2.0, "no": 1.0}
    assert result["timing"]["total_ms"] == 125.0
    assert result["timing"]["forward_ms"] == 100.0
    assert result["timing"]["cache_hit"] is True


def test_format_shared_timing() -> None:
    assert format_shared_timing(
        {"total_seconds": 0.2, "batch_size": 3, "prefix_tokens": 20}
    ) == {"total_ms": 200.0, "batch_size": 3, "prefix_tokens": 20}


def test_config_json_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for env in (
        "DEQIO_BACKEND",
        "DEQIO_MODEL",
        "DEQIO_MODEL_REVISION",
        "DEQIO_MAX_TOKENS",
        "DEQIO_MLX_CACHE_MIB",
        "DEQIO_LOG",
        "DEQIO_TORCH_DTYPE",
    ):
        monkeypatch.delenv(env, raising=False)

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "mlx",
                "model": "models/mlx",
                "model_revision": "local-test",
                "max_tokens": 4096,
                "mlx_cache_mib": 256,
                "log": "logs/requests.jsonl",
                "torch_dtype": "bfloat16",
            }
        )
    )

    settings = load_settings(config_path)

    assert settings.backend == "mlx"
    assert settings.model == str((tmp_path / "models/mlx").resolve())
    assert settings.log_path == (tmp_path / "logs/requests.jsonl").resolve()
    assert settings.max_tokens == 4096
    assert settings.mlx_cache_mib == 256


def test_config_environment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "mlx",
                "model": "models/mlx",
                "model_revision": "local-test",
                "max_tokens": 4096,
                "mlx_cache_mib": 256,
                "log": "logs/requests.jsonl",
                "torch_dtype": "bfloat16",
            }
        )
    )

    monkeypatch.setenv("DEQIO_BACKEND", "cuda")
    monkeypatch.setenv("DEQIO_MODEL", "Qwen/Qwen3.5-4B")
    monkeypatch.setenv("DEQIO_MAX_TOKENS", "8192")

    settings = load_settings(config_path)

    assert settings.backend == "cuda"
    assert settings.model == "Qwen/Qwen3.5-4B"
    assert settings.max_tokens == 8192


def test_ui_contains_all_playground_modes_and_cache_action() -> None:
    assert 'data-endpoint="noul"' in DASHBOARD
    assert 'data-endpoint="choice"' in DASHBOARD
    assert 'data-endpoint="shared"' in DASHBOARD
    assert "/v1/cache/clear" in DASHBOARD
    assert "Generated request JSON" in DASHBOARD


def test_model_catalog_contains_multi_engine_profiles() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    engines = {entry["engine"] for entry in catalog["models"]}

    assert {"semif", "kev", "decider", "laya", "von"}.issubset(engines)
    assert get_profile(catalog, "semif-qwen3.5-4b", "mps")["model"] == "Qwen/Qwen3.5-4B"
    assert get_profile(catalog, "kev-4b", "mlx")["model"] == "jaredpalmer/kev-4b"
    assert get_profile(catalog, "kev-4b", "mps")["model"] == "jaredpalmer/kev-4b"
    assert get_profile(catalog, "decider-2b", "mps")["model"] == "Mapika/decider-2b"
    assert get_profile(catalog, "decider-2b", "cuda")["model"] == "Mapika/decider-2b"
    with pytest.raises(RuntimeError, match="does not support backend"):
        get_profile(catalog, "decider-2b", "mlx")
    assert get_profile(catalog, "laya-multilingual", "mlx")["model"] == "aac6fef/laya-multilingual-mlx"
    assert get_profile(catalog, "laya-multilingual", "mps")["model"] == "convaiinnovations/laya-multilingual"
    assert get_profile(catalog, "von", "mps")["wire_model"] == "von-1.2.0"
    assert get_profile(catalog, "von", "cuda")["wire_model"] == "von-1.2.0"


def test_catalog_rejects_unsupported_backend() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    with pytest.raises(RuntimeError, match="does not support backend"):
        get_profile(catalog, "von", "mlx")




def test_model_manager_exposes_mlx_and_mps_on_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.model_manager as model_manager

    monkeypatch.setattr(model_manager.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(model_manager.platform, "machine", lambda: "arm64")

    assert model_manager._host_backends() == ("mlx", "mps")


def test_supported_backends_are_accelerator_only() -> None:
    from deqio.catalog import SUPPORTED_BACKENDS, load_catalog

    assert SUPPORTED_BACKENDS == ("mlx", "mps", "cuda")
    catalog = load_catalog(Path("models.json"))
    for entry in catalog["models"]:
        assert set(entry["backends"]) <= set(SUPPORTED_BACKENDS)


def test_config_defaults_new_multi_engine_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEQIO_ENGINE", raising=False)
    monkeypatch.delenv("DEQIO_MODEL_ID", raising=False)
    monkeypatch.delenv("DEQIO_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("DEQIO_MODEL_CATALOG", raising=False)
    monkeypatch.delenv("DEQIO_SIDECAR_STARTUP_SECONDS", raising=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "cuda",
                "model": "Qwen/Qwen3.5-4B",
                "model_revision": "test",
                "max_tokens": 4096,
                "mlx_cache_mib": 256,
                "log": "logs/requests.jsonl",
                "torch_dtype": "bfloat16",
            }
        )
    )

    settings = load_settings(config_path)

    assert settings.engine == "semif"
    assert settings.model_id == "semif-qwen3.5-4b"
    assert settings.runtime_dir == (tmp_path / ".model-runtimes").resolve()
    assert settings.model_catalog == (tmp_path / "models.json").resolve()
    assert settings.sidecar_startup_seconds == 900


def test_external_choice_response_normalization() -> None:
    from deqio.systemone_runtime import _choice_raw

    row = {
        "id": "route",
        "state": "charged twice",
        "question": "Which team?",
        "options": [
            {"id": "billing", "description": "payments"},
            {"id": "technical", "description": "bugs"},
        ],
    }
    payload = {"state": row["state"], "questions": {}}
    raw = _choice_raw(
        row=row,
        answer={"choice": "billing", "probabilities": {"billing": 0.8, "technical": 0.2}},
        response={"usage": {"input_tokens": 31}},
        payload=payload,
        latency_ms=12.5,
    )

    assert raw["option_ids"] == ["billing", "technical"]
    assert raw["probabilities"] == [0.8, 0.2]
    assert raw["input_tokens"] == 31
    assert raw["total_seconds"] == 0.0125
    assert "option_logits" not in raw


def test_external_noul_response_normalization() -> None:
    from deqio.systemone_runtime import _noul_raw

    row = {"id": "safe", "state": "validated", "question": "Is it safe?"}
    raw = _noul_raw(
        row=row,
        answer={"noul": 0.75},
        response={"usage": {"input_tokens": 10}},
        payload={"state": "validated", "questions": {}},
        latency_ms=5.0,
    )

    assert raw["option_ids"] == ["yes", "no"]
    assert raw["probabilities"] == [0.75, 0.25]


def test_ui_shows_multi_engine_runtime() -> None:
    assert "health.engine" in DASHBOARD
    assert "health.model_id" in DASHBOARD

def test_sidecar_log_filter_hides_internal_http_noise() -> None:
    from deqio.console import is_noisy_sidecar_line

    assert is_noisy_sidecar_line(
        'INFO:     127.0.0.1:54650 - "POST /v1/systemone HTTP/1.1" 200 OK'
    )
    assert is_noisy_sidecar_line("INFO:     Uvicorn running on http://127.0.0.1:54114")
    assert not is_noisy_sidecar_line(
        '[serve] ready {"model": "decider-0.8b-v1", "device": "mps"}'
    )


def test_request_console_log_is_endpoint_specific(capsys: pytest.CaptureFixture[str]) -> None:
    from deqio.console import log_request_success

    log_request_success(
        "/v1/noul",
        request_id="req-1",
        result={
            "decision": "yes",
            "top_probability": 0.875,
            "timing": {"total_ms": 12.5, "cache_hit": True},
        },
        mode="serial",
    )

    output = capsys.readouterr().out
    assert "[request] POST /v1/noul" in output
    assert "decision=yes" in output
    assert "p=0.8750" in output
    assert "latency=12.5ms" in output
    assert "cache=hit" in output


def test_shared_console_log_is_compact(capsys: pytest.CaptureFixture[str]) -> None:
    from deqio.console import log_request_success

    log_request_success(
        "/v1/shared",
        request_id="shared-1",
        result={"timing": {"total_ms": 44.2}},
        mode="shared",
        decisions=4,
    )

    output = capsys.readouterr().out
    assert "[request] POST /v1/shared" in output
    assert "decisions=4" in output
    assert "latency=44.2ms" in output


def test_ui_examples_are_endpoint_specific() -> None:
    assert "Should tests be run before considering the task complete?" in DASHBOARD
    assert "Which team should handle this request?" in DASHBOARD
    assert "A customer was charged twice for the same subscription renewal." in DASHBOARD
    assert "What should be the next validation action?" in DASHBOARD
    assert "A patch changed an internal authentication module." in DASHBOARD


def test_ui_shared_example_contains_multiple_decisions() -> None:
    assert "run_integration_tests" in DASHBOARD
    assert "continue_validation" in DASHBOARD
    assert "ready_for_release" in DASHBOARD


def test_model_control_routes_are_registered() -> None:
    routes = {route.path for route in app.routes}
    assert "/v1/models/installed" in routes
    assert "/v1/models/activate" in routes


def test_installation_registry_tracks_specific_profile(tmp_path: Path) -> None:
    from deqio.installations import load_registry, mark_installed, profile_key

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    mark_installed(config_path, "decider-0.8b", "mps", verified=False)
    registry = load_registry(config_path)
    record = registry["profiles"][profile_key("decider-0.8b", "mps")]
    assert record["model_id"] == "decider-0.8b"
    assert record["backend"] == "mps"
    assert "verified_at" not in record

    mark_installed(config_path, "decider-0.8b", "mps", verified=True, source="startup")
    registry = load_registry(config_path)
    record = registry["profiles"][profile_key("decider-0.8b", "mps")]
    assert "verified_at" in record


def test_installed_profiles_require_registry_or_local_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.installations as installations

    config_path = tmp_path / "config.json"
    config_data = {"runtime_dir": ".model-runtimes"}
    runtime_python = tmp_path / ".model-runtimes" / "decider" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("")
    catalog = {
        "models": [
            {
                "id": "decider-0.8b",
                "engine": "decider",
                "label": "Decider 0.8B",
                "backends": {
                    "mps": {
                        "model": "Mapika/decider-0.8b",
                        "runtime_key": "decider",
                    }
                },
            }
        ]
    }
    monkeypatch.setattr(installations, "_cached_hf_repos", lambda: set())
    monkeypatch.setattr(installations, "host_backends", lambda: ("mps",))

    rows = installations.installed_profiles(
        config_path=config_path,
        config_data=config_data,
        catalog=catalog,
        active_model_id="decider-0.8b",
        active_backend="mps",
    )
    assert rows[0]["installed"] is False
    assert rows[0]["status"] == "selected"

    installations.mark_installed(config_path, "decider-0.8b", "mps")
    rows = installations.installed_profiles(
        config_path=config_path,
        config_data=config_data,
        catalog=catalog,
        active_model_id="decider-0.8b",
        active_backend="mps",
    )
    assert rows[0]["installed"] is True
    assert rows[0]["status"] == "active"


def test_ui_contains_installed_model_selector_and_live_activate_endpoint() -> None:
    assert 'id="modelSelect"' in DASHBOARD
    assert 'id="activateModel"' in DASHBOARD
    assert "/v1/models/installed" in DASHBOARD
    assert "/v1/models/activate" in DASHBOARD


def test_live_model_activation_persists_selection_and_swaps_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.server as server
    from deqio.config import settings_from_data

    config_path = tmp_path / "config.json"
    base_data = {
        "engine": "decider",
        "model_id": "decider-0.8b",
        "backend": "mps",
        "model": "Mapika/decider-0.8b",
        "model_revision": "decider-0.8b",
        "model_catalog": "models.json",
        "runtime_dir": ".model-runtimes",
        "max_tokens": 4096,
        "mlx_cache_mib": 256,
        "log": "logs/requests.jsonl",
        "torch_dtype": "bfloat16",
        "sidecar_startup_seconds": 900,
    }
    config_path.write_text(json.dumps(base_data))
    old_settings = settings_from_data(config_path, base_data, apply_environment=False)
    catalog = {
        "models": [
            {
                "id": "decider-0.8b",
                "engine": "decider",
                "label": "Decider 0.8B",
                "backends": {"mps": {"model": "Mapika/decider-0.8b"}},
            },
            {
                "id": "decider-2b",
                "engine": "decider",
                "label": "Decider 2B",
                "backends": {"mps": {"model": "Mapika/decider-2b"}},
            },
        ]
    }
    rows = [
        {
            "model_id": "decider-0.8b",
            "backend": "mps",
            "engine": "decider",
            "installed": True,
            "host_compatible": True,
        },
        {
            "model_id": "decider-2b",
            "backend": "mps",
            "engine": "decider",
            "installed": True,
            "host_compatible": True,
        },
    ]

    class FakeRuntime:
        def __init__(self, settings):
            self.settings = settings
            self.closed = False

        def score(self, row, mode):
            return {
                "id": row["id"],
                "option_ids": ["yes", "no"],
                "probabilities": [0.9, 0.1],
                "input_tokens": 1,
                "total_seconds": 0.001,
                "prompt_sha256": "warmup",
                "probability_status": "test",
            }

        def close(self):
            self.closed = True

    class FakeBackendRuntime:
        @staticmethod
        def load(settings):
            return FakeRuntime(settings)

    old_runtime = FakeRuntime(old_settings)
    monkeypatch.setattr(server, "SETTINGS", old_settings)
    monkeypatch.setattr(server, "runtime", old_runtime)
    monkeypatch.setattr(server, "BackendRuntime", FakeBackendRuntime)
    monkeypatch.setattr(server, "_installation_rows", lambda: (catalog, rows))
    monkeypatch.setattr(server, "mark_installed", lambda *args, **kwargs: None)
    for env_name in server.MODEL_SELECTION_ENV_VARS:
        monkeypatch.delenv(env_name, raising=False)

    response = server.activate_model(server.ModelActivateRequest(model_id="decider-2b", backend="mps"))

    assert response["status"] == "ok"
    assert response["active"]["model_id"] == "decider-2b"
    assert server.SETTINGS.model_id == "decider-2b"
    assert old_runtime.closed is True
    persisted = json.loads(config_path.read_text())
    assert persisted["model_id"] == "decider-2b"
    assert persisted["backend"] == "mps"

def test_release_version_is_consistent() -> None:
    import tomllib

    from deqio import __version__
    from deqio.server import app

    project = tomllib.loads(Path("pyproject.toml").read_text())

    assert __version__ == "0.1.0"
    assert project["project"]["version"] == __version__
    assert app.version == __version__


def test_native_runtime_close_releases_model_references() -> None:
    from types import SimpleNamespace

    from deqio.backends import BackendRuntime

    runtime = BackendRuntime(
        settings=SimpleNamespace(backend="test", max_tokens=4096),
        model=object(),
        tokenizer=object(),
        metadata={"test": True},
        direct_score=lambda *args, **kwargs: {},
        serial_factory=lambda *args, **kwargs: object(),
        shared_score=lambda *args, **kwargs: ([], {}),
    )

    runtime.close()

    assert runtime.model is None
    assert runtime.tokenizer is None
    assert runtime.serial_scorer is None
    assert runtime.metadata == {}


def test_catalog_contains_nimble_mlx_and_cuda_only() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    mlx = get_profile(catalog, "nimble-9b", "mlx")
    cuda = get_profile(catalog, "nimble-9b", "cuda")
    assert mlx["installer"] == "nimble"
    assert mlx["nimble_backend"] == "mlx"
    assert cuda["nimble_backend"] == "cuda"
    with pytest.raises(RuntimeError, match="does not support backend"):
        get_profile(catalog, "nimble-9b", "mps")


def test_benchmark_basic_suite_has_fifty_cases_per_request_type() -> None:
    from deqio.benchmark import load_suite

    suite = load_suite(Path("benchmarks/basic.json"))
    counts = {kind: 0 for kind in ("noul", "choice", "shared")}
    for case in suite["cases"]:
        counts[case["type"]] += 1
    assert counts == {"noul": 50, "choice": 50, "shared": 50}


def test_benchmark_summary_tracks_case_and_decision_accuracy() -> None:
    from deqio.benchmark import BenchResult, summarize

    rows = [
        BenchResult("a", "mlx", "x", "n1", "noul", True, "yes", "yes", 10.0, 1, 1, 0.9),
        BenchResult("a", "mlx", "x", "c1", "choice", False, "a", "b", 20.0, 1, 0, 0.6),
        BenchResult("a", "mlx", "x", "s1", "shared", False, ["a", "b"], ["a", "c"], 30.0, 2, 1),
    ]
    summary = summarize(rows)
    assert summary["overall"]["cases"] == 3
    assert summary["overall"]["passed_cases"] == 1
    assert summary["overall"]["assertions"] == 4
    assert summary["overall"]["correct"] == 2
    assert summary["overall"]["decision_accuracy"] == 0.5
    assert summary["overall"]["median_ms"] == 20.0


def test_nimble_sidecar_builds_choice_schema_without_importing_runtime() -> None:
    from deqio.nimble_sidecar import _choice_schema

    field = _choice_schema({
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "Payments", "technical": "Bugs"},
    })
    assert field["type"] == "enum"
    assert field["choices"] == ["billing", "technical"]
    assert field["choice_descriptions"]["billing"] == "Payments"


def test_nimble_sidecar_normalizes_noul_and_choice_results() -> None:
    from deqio.nimble_sidecar import _answers_from_result

    schema = {
        "binary": {"type": "boolean", "description": "Is it valid?"},
        "route": {"type": "enum", "choices": ["billing", "technical"], "description": "Route it."},
    }
    kinds = {"binary": "noul", "route": "choice"}
    result = {
        "output": {"binary": True, "route": "billing"},
        "fields": {
            "binary": {"scores": {"false": 0.2, "true": 0.8}},
            "route": {"scores": {"billing": 0.75, "technical": 0.25}},
        },
    }
    answers = _answers_from_result(schema, kinds, result)
    assert answers["binary"]["noul"] == pytest.approx(0.8)
    assert answers["binary"]["confidence"] == pytest.approx(0.8)
    assert answers["route"]["choice"] == "billing"
    assert answers["route"]["probabilities"] == {"billing": 0.75, "technical": 0.25}



def test_benchmark_store_lists_and_reads_completed_runs(tmp_path: Path) -> None:
    import json

    from deqio.benchmark_store import list_benchmark_runs, read_benchmark_results, read_benchmark_summary

    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / ".deqio" / "benchmarks" / "20260925T120000Z"
    run_dir.mkdir(parents=True)
    summary = {
        "schema_version": 1,
        "created_at": "2026-09-25T12:00:00+00:00",
        "suite_name": "deqio-basic-150",
        "models": [{"model_id": "test", "backend": "mlx", "engine": "x", "load_ms": 12.0, "summary": {}}],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = [
        {"model_id": "test", "backend": "mlx", "engine": "x", "case_id": "noul-01", "kind": "noul", "passed": True},
        {"model_id": "test", "backend": "mlx", "engine": "x", "case_id": "choice-01", "kind": "choice", "passed": False},
    ]
    (run_dir / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    runs = list_benchmark_runs(config_path)
    assert len(runs) == 1
    assert runs[0]["id"] == "20260925T120000Z"
    assert runs[0]["suite_name"] == "deqio-basic-150"
    assert runs[0]["models"] == 1
    assert runs[0]["results"] == 2
    assert runs[0]["has_summary"] is True
    assert runs[0]["has_results"] is True
    assert read_benchmark_summary(config_path, runs[0]["id"])["suite_name"] == "deqio-basic-150"
    assert len(read_benchmark_results(config_path, runs[0]["id"])) == 2


def test_benchmark_store_rejects_path_traversal(tmp_path: Path) -> None:
    from deqio.benchmark_store import read_benchmark_summary

    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        read_benchmark_summary(config_path, "../outside")


def test_benchmark_routes_and_ui_are_exposed() -> None:
    from deqio.server import app
    from deqio.ui import DASHBOARD

    routes = {route.path for route in app.routes}
    assert "/v1/benchmarks" in routes
    assert "/v1/benchmarks/{run_id}/summary" in routes
    assert "/v1/benchmarks/{run_id}/results" in routes
    assert 'id="benchmarkRunSelect"' in DASHBOARD
    assert 'id="benchmarkSummaryRows"' in DASHBOARD
    assert 'id="benchmarkResultRows"' in DASHBOARD
    assert "Accuracy" in DASHBOARD
    assert "Median latency" in DASHBOARD
    assert "PASS + FAIL" in DASHBOARD


def test_default_workspace_bootstraps_from_packaged_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from deqio.config import read_config_data
    from deqio.benchmark import load_suite
    from deqio.catalog import load_catalog

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEQIO_CONFIG", raising=False)

    config_path, data = read_config_data()

    assert config_path == (tmp_path / "config.json").resolve()
    assert config_path.is_file()
    assert (tmp_path / "models.json").is_file()
    assert (tmp_path / "benchmarks" / "basic.json").is_file()
    assert data["model_catalog"] == "models.json"
    assert len(load_catalog(tmp_path / "models.json")["models"]) >= 1
    suite = load_suite(tmp_path / "benchmarks" / "basic.json")
    counts = {kind: 0 for kind in ("noul", "choice", "shared")}
    for case in suite["cases"]:
        counts[case["type"]] += 1
    assert counts == {"noul": 50, "choice": 50, "shared": 50}


def test_workspace_bootstrap_never_overwrites_existing_files(tmp_path: Path) -> None:
    from deqio.workspace import ensure_workspace

    custom = tmp_path / "models.json"
    custom.write_text('{"custom": true}\n', encoding="utf-8")
    ensure_workspace(tmp_path)
    assert custom.read_text(encoding="utf-8") == '{"custom": true}\n'


def test_semif_profiles_use_isolated_runtime() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    for backend in ("mlx", "mps", "cuda"):
        profile = get_profile(catalog, "semif-qwen3.5-4b", backend)
        assert profile["runtime_key"] == "semif"
        assert any("github.com/TheoLeeCJ/SemIf.git" in package for package in profile["packages"])


def test_backend_loader_routes_semif_through_systemone(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from deqio.backends import BackendRuntime
    from deqio.systemone_runtime import SystemOneRuntime

    sentinel = object()
    monkeypatch.setattr(SystemOneRuntime, "load", classmethod(lambda cls, settings: sentinel))
    assert BackendRuntime.load(SimpleNamespace(engine="semif")) is sentinel


def test_semif_sidecar_normalizes_noul_and_choice_without_runtime_import() -> None:
    from deqio.semif_sidecar import _answer_from_raw

    noul = _answer_from_raw(
        "noul",
        {"option_ids": ["yes", "no"], "probabilities": [0.75, 0.25]},
    )
    assert noul["noul"] == pytest.approx(0.75)
    assert noul["confidence"] == pytest.approx(0.75)

    choice = _answer_from_raw(
        "choice",
        {"option_ids": ["a", "b"], "probabilities": [0.2, 0.8]},
    )
    assert choice["choice"] == "b"
    assert choice["probabilities"] == {"a": 0.2, "b": 0.8}


def test_packaged_workspace_templates_are_current_and_self_consistent(tmp_path: Path) -> None:
    import json
    from importlib.resources import files

    from deqio.workspace import ensure_workspace

    package_data = files("deqio.data")
    packaged_models = json.loads(package_data.joinpath("models.json").read_text(encoding="utf-8"))
    repository_models = json.loads(Path("models.json").read_text(encoding="utf-8"))
    assert packaged_models == repository_models

    packaged_benchmark = json.loads(
        package_data.joinpath("benchmarks").joinpath("basic.json").read_text(encoding="utf-8")
    )
    repository_benchmark = json.loads(Path("benchmarks/basic.json").read_text(encoding="utf-8"))
    assert packaged_benchmark == repository_benchmark

    # The repository-root config.json is local mutable state and is intentionally
    # ignored by Git. It changes whenever the active model changes, so it must not
    # be used as the golden source for the immutable PyPI workspace template.
    packaged_config = json.loads(package_data.joinpath("config.json").read_text(encoding="utf-8"))
    model_by_id = {str(item["id"]): item for item in packaged_models["models"]}
    selected = model_by_id[packaged_config["model_id"]]
    profile = selected["backends"][packaged_config["backend"]]

    assert packaged_config["engine"] == selected["engine"]
    assert packaged_config["model"] == profile["model"]
    assert packaged_config["model_revision"] == profile.get("model_revision", selected["id"])

    ensure_workspace(tmp_path)
    bootstrapped_config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert bootstrapped_config == packaged_config
