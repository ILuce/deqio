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
