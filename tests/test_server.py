import asyncio
import json
from pathlib import Path

import pytest

from deqio.config import load_settings
from deqio.server import app, format_result, format_shared_timing, percentile, state_hash
from deqio.ui import DASHBOARD, WATCH_DASHBOARD


def _asgi_post_json(path: str, body: bytes | str, headers: dict[str, str]) -> tuple[int, dict]:
    raw = body.encode("utf-8") if isinstance(body, str) else body

    async def invoke() -> tuple[int, dict]:
        sent: list[dict] = []
        delivered = False

        async def receive() -> dict:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": raw, "more_body": False}

        async def send(message: dict) -> None:
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in headers.items()],
            "client": ("testclient", 123),
            "server": ("testserver", 80),
            "root_path": "",
        }
        await app(scope, receive, send)
        status = next(message["status"] for message in sent if message["type"] == "http.response.start")
        response_body = b"".join(
            message.get("body", b"") for message in sent if message["type"] == "http.response.body"
        )
        return int(status), json.loads(response_body.decode("utf-8"))

    return asyncio.run(invoke())


def test_public_routes_are_registered() -> None:
    routes = {route.path for route in app.routes}

    assert "/" in routes
    assert "/health" in routes
    assert "/ui" in routes
    assert "/ui/watch" in routes
    assert "/v1/stats" in routes
    assert "/v1/recent" in routes
    assert "/v1/watch" in routes
    assert "/v1/watch/settings" in routes
    assert "/v1/watch/{event_id}" in routes
    assert "/v1/watch/clear" in routes
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
    assert get_profile(catalog, "kev-4b", "mlx")["model"] == "jaredpalmer/kev-4b@v1.0"
    assert get_profile(catalog, "kev-4b", "mps")["model"] == "jaredpalmer/kev-4b@v1.0"
    assert get_profile(catalog, "decider-2b", "mps")["model"] == "Mapika/decider-2b"
    assert get_profile(catalog, "decider-2b", "cuda")["model"] == "Mapika/decider-2b"
    with pytest.raises(RuntimeError, match="does not support backend"):
        get_profile(catalog, "decider-2b", "mlx")
    assert get_profile(catalog, "laya-multilingual", "mlx")["model"] == "aac6fef/laya-multilingual-mlx"
    assert get_profile(catalog, "laya-multilingual", "mps")["model"] == "convaiinnovations/laya-multilingual"
    assert get_profile(catalog, "von", "mps")["wire_model"] == "von-1.2.0"
    assert get_profile(catalog, "von", "cuda")["wire_model"] == "von-1.2.0"


def test_config_engine_validation_covers_every_catalog_engine() -> None:
    from deqio.catalog import SUPPORTED_ENGINES, apply_selection, load_catalog
    from deqio.config import settings_from_data

    config_path = Path("config.json").resolve()
    base_data = json.loads(config_path.read_text(encoding="utf-8"))
    catalog = load_catalog(Path("models.json"))
    catalog_engines = {str(entry["engine"]) for entry in catalog["models"]}

    assert catalog_engines == set(SUPPORTED_ENGINES)

    validated_engines: set[str] = set()
    for entry in catalog["models"]:
        engine = str(entry["engine"])
        if engine in validated_engines:
            continue
        backend, profile = next(iter(entry["backends"].items()))
        selected = apply_selection(base_data, entry, profile, str(backend))
        settings = settings_from_data(config_path, selected, apply_environment=False)
        assert settings.engine == engine
        validated_engines.add(engine)

    assert validated_engines == catalog_engines


def test_catalog_rejects_unsupported_backend() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    with pytest.raises(RuntimeError, match="does not support backend"):
        get_profile(catalog, "von", "mlx")




def test_model_manager_exposes_mlx_and_mps_on_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.hardware as hardware

    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")

    assert hardware.host_backends() == ("mlx", "mps")


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
    assert settings.sidecar_startup_seconds == 1800
    assert settings.sidecar_process_ready_seconds == 300
    assert settings.hf_offline_runtime is True


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
    from deqio.hardware import HostCapabilities

    monkeypatch.setattr(installations, "_cached_hf_repos", lambda: {"Mapika/decider-0.8b"})
    monkeypatch.setattr(
        installations,
        "detect_host",
        lambda: HostCapabilities("Darwin", "arm64", ("mlx", "mps"), 32.0, None),
    )

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



def test_installed_profiles_accept_managed_local_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.installations as installations
    from deqio.hardware import HostCapabilities

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    runtime_python = tmp_path / ".model-runtimes" / "semif" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("")
    model_dir = tmp_path / "models" / "semif-qwen3.5-4b-mlx-4bit"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")

    catalog = {
        "models": [
            {
                "id": "semif-qwen3.5-4b",
                "engine": "semif",
                "label": "SemIf / Qwen3.5 4B",
                "backends": {
                    "mlx": {
                        "model": "models/semif-qwen3.5-4b-mlx-4bit",
                        "runtime_key": "semif",
                        "download": {
                            "type": "snapshot",
                            "repo_id": "vinci00/semif-qwen3.5-4b-mlx-4bit",
                            "local_dir": "models/semif-qwen3.5-4b-mlx-4bit",
                        },
                    }
                },
            }
        ]
    }
    monkeypatch.setattr(installations, "_cached_hf_repos", lambda: set())
    monkeypatch.setattr(
        installations,
        "detect_host",
        lambda: HostCapabilities("Darwin", "arm64", ("mlx", "mps"), 16.0, None),
    )

    installations.mark_installed(config_path, "semif-qwen3.5-4b", "mlx", verified=True)
    row = installations.installed_profiles(
        config_path=config_path,
        config_data={"runtime_dir": ".model-runtimes"},
        catalog=catalog,
        active_model_id="semif-qwen3.5-4b",
        active_backend="mlx",
    )[0]

    assert row["installed"] is True
    assert row["verified"] is True
    assert row["status"] == "active"
    assert row["weights_cached"] is True


def test_installed_profiles_use_declared_downloads_as_artifact_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.installations as installations
    from deqio.hardware import HostCapabilities

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    runtime_python = tmp_path / ".model-runtimes" / "example" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("")
    catalog = {
        "models": [
            {
                "id": "runtime-alias-model",
                "engine": "example",
                "label": "Runtime Alias Model",
                "backends": {
                    "mps": {
                        "model": "runtime-alias",
                        "runtime_key": "example",
                        "download": {
                            "type": "file",
                            "repo_id": "example/model-artifacts",
                            "filename": "marker.pt",
                        },
                    }
                },
            }
        ]
    }
    monkeypatch.setattr(installations, "_cached_hf_repos", lambda: {"example/model-artifacts"})
    monkeypatch.setattr(
        installations,
        "detect_host",
        lambda: HostCapabilities("Darwin", "arm64", ("mlx", "mps"), 16.0, None),
    )

    installations.mark_installed(config_path, "runtime-alias-model", "mps", verified=True)
    row = installations.installed_profiles(
        config_path=config_path,
        config_data={"runtime_dir": ".model-runtimes"},
        catalog=catalog,
        active_model_id=None,
        active_backend=None,
    )[0]

    assert row["installed"] is True
    assert row["verified"] is True
    assert row["weights_cached"] is True
    assert row["status"] == "verified"


def test_models_setup_marks_installed_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import deqio.model_manager as manager
    from deqio.hardware import HostCapabilities

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model_catalog": "models.json", "runtime_dir": ".model-runtimes"}))
    (tmp_path / "models.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "semif-qwen3.5-4b",
                        "engine": "semif",
                        "label": "SemIf / Qwen3.5 4B",
                        "backends": {"mlx": {"model": "models/semif", "runtime_key": "semif"}},
                    },
                    {
                        "id": "kev-0.8b",
                        "engine": "kev",
                        "label": "Kev 0.8B",
                        "backends": {"mlx": {"model": "jaredpalmer/kev-0.8b", "runtime_key": "kev"}},
                    },
                ]
            }
        )
    )
    host = HostCapabilities("Darwin", "arm64", ("mlx", "mps"), 16.0, None)
    monkeypatch.setattr(manager, "detect_host", lambda: host)
    monkeypatch.setattr(manager, "host_backends", lambda: ["mlx"])
    monkeypatch.setattr(
        manager,
        "profile_compatibility",
        lambda backend, profile, host=None: {
            "compatible": True,
            "warning": None,
            "minimum_memory_gib": 6.0,
            "available_memory_gib": 12.0,
            "reason": "compatible",
        },
    )
    monkeypatch.setattr(
        manager,
        "_installation_rows",
        lambda *args, **kwargs: [
            {"model_id": "semif-qwen3.5-4b", "backend": "mlx", "installed": True}
        ],
    )
    monkeypatch.setattr(manager, "_install_profile", lambda **kwargs: None)
    monkeypatch.setattr(manager, "_write_config", lambda *args, **kwargs: None)
    answers = iter(["1", "1", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    assert manager.cmd_setup(type("Args", (), {"config": str(config_path)})()) == 0
    output = capsys.readouterr().out

    assert "1. [x] semif-qwen3.5-4b" in output
    assert "2. [ ] kev-0.8b" in output
    assert "Maximum input tokens for semif-qwen3.5-4b / mlx:" in output
    assert "4096 tokens" in output
    assert "Select max input tokens" not in output  # prompt text is supplied to input(), not printed by the test stub


def test_models_use_lists_only_installed_host_compatible_profiles_and_persists_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from types import SimpleNamespace

    import deqio.model_manager as manager

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "engine": "kev",
                "model_id": "kev-0.8b",
                "backend": "mlx",
                "model": "jaredpalmer/kev-0.8b",
                "model_revision": "kev-0.8b",
                "max_tokens": 4096,
                "mlx_cache_mib": 256,
                "log": "logs/requests.jsonl",
                "torch_dtype": "bfloat16",
                "runtime_dir": ".model-runtimes",
                "model_catalog": "models.json",
                "sidecar_startup_seconds": 1800,
                "sidecar_process_ready_seconds": 300,
                "hf_offline_runtime": True,
            }
        ),
        encoding="utf-8",
    )
    catalog = {
        "models": [
            {
                "id": "kev-0.8b",
                "engine": "kev",
                "label": "Kev 0.8B",
                "backends": {
                    "mlx": {
                        "model": "jaredpalmer/kev-0.8b",
                        "model_revision": "kev-0.8b",
                    }
                },
            },
            {
                "id": "basal-1.5b",
                "engine": "basal",
                "label": "Basal 1.5B",
                "backends": {
                    "mlx": {
                        "model": "Remek/basal-1.0-1.5B",
                        "model_revision": "basal-1.5b",
                    }
                },
            },
            {
                "id": "hidden-uninstalled",
                "engine": "kev",
                "label": "Hidden Uninstalled",
                "backends": {"mlx": {"model": "example/uninstalled"}},
            },
            {
                "id": "hidden-incompatible",
                "engine": "kev",
                "label": "Hidden Incompatible",
                "backends": {"mlx": {"model": "example/incompatible"}},
            },
        ]
    }
    (tmp_path / "models.json").write_text(json.dumps(catalog), encoding="utf-8")

    rows = [
        {
            "model_id": "kev-0.8b",
            "backend": "mlx",
            "engine": "kev",
            "label": "Kev 0.8B",
            "installed": True,
            "host_compatible": True,
            "active": True,
            "max_input_tokens": 4096,
        },
        {
            "model_id": "basal-1.5b",
            "backend": "mlx",
            "engine": "basal",
            "label": "Basal 1.5B",
            "installed": True,
            "host_compatible": True,
            "active": False,
            "max_input_tokens": 8192,
        },
        {
            "model_id": "hidden-uninstalled",
            "backend": "mlx",
            "engine": "kev",
            "label": "Hidden Uninstalled",
            "installed": False,
            "host_compatible": True,
            "active": False,
            "max_input_tokens": 4096,
        },
        {
            "model_id": "hidden-incompatible",
            "backend": "mlx",
            "engine": "kev",
            "label": "Hidden Incompatible",
            "installed": True,
            "host_compatible": False,
            "active": False,
            "max_input_tokens": 4096,
        },
    ]
    monkeypatch.setattr(manager, "_installation_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(manager, "_prompt_index", lambda label, count: 1)

    result = manager.cmd_use(SimpleNamespace(config=str(config_path), model_id=None, backend=None))

    assert result == 0
    output = capsys.readouterr().out
    assert "Kev 0.8B — mlx (kev) [active]" in output
    assert "Basal 1.5B — mlx (basal)" in output
    assert "Hidden Uninstalled" not in output
    assert "Hidden Incompatible" not in output
    assert "Selected installed profile basal-1.5b / mlx" in output

    selected = json.loads(config_path.read_text(encoding="utf-8"))
    assert selected["engine"] == "basal"
    assert selected["model_id"] == "basal-1.5b"
    assert selected["backend"] == "mlx"
    assert selected["model"] == "Remek/basal-1.0-1.5B"
    assert selected["model_revision"] == "basal-1.5b"
    assert selected["max_tokens"] == 8192


def test_ui_contains_installed_model_selector_and_live_activate_endpoint() -> None:
    assert 'id="modelSelect"' in DASHBOARD
    assert 'id="activateModel"' in DASHBOARD
    assert "/v1/models/installed" in DASHBOARD
    assert "/v1/models/activate" in DASHBOARD
    assert 'href="/ui/watch"' in DASHBOARD


def test_watch_ui_uses_disk_backed_session_endpoints() -> None:
    assert "Deqio Watch" in WATCH_DASHBOARD
    assert "fetch(`/v1/watch?limit=${pageSize}&offset=${offset}`)" in WATCH_DASHBOARD
    assert "/v1/watch/${encodeURIComponent(eventId)}" in WATCH_DASHBOARD
    assert "/v1/watch/settings" in WATCH_DASHBOARD
    assert "/v1/watch/clear" in WATCH_DASHBOARD
    assert 'id="sourceFilter"' in WATCH_DASHBOARD
    assert 'id="modelFilter"' in WATCH_DASHBOARD
    assert 'id="watchAutoClear"' in WATCH_DASHBOARD
    assert 'href="/ui"' in WATCH_DASHBOARD
    assert 'id="watchAutoClear"' in DASHBOARD


def test_watch_session_records_full_request_response_on_disk_and_rejects_stale_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.server as server
    from deqio.config import settings_from_data

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    catalog_path = tmp_path / "models.json"
    catalog_path.write_text('{"models": []}')
    settings = settings_from_data(
        config_path,
        {
            "engine": "kev",
            "model_id": "kev-0.8b",
            "backend": "mlx",
            "model": "jaredpalmer/kev-0.8b",
            "model_revision": "upstream-latest",
            "model_catalog": str(catalog_path),
            "runtime_dir": str(tmp_path / ".model-runtimes"),
            "max_tokens": 4096,
            "mlx_cache_mib": 256,
            "log": str(tmp_path / "logs" / "requests.jsonl"),
            "torch_dtype": "bfloat16",
            "sidecar_startup_seconds": 900,
            "hf_offline_runtime": True,
        },
        apply_environment=False,
    )
    monkeypatch.setattr(server, "SETTINGS", settings)

    server._reset_watch_session(reason="test-start")
    session_id = server._watch_session_token()
    request_payload = {"id": "req-1", "state": "s", "question": "q"}
    response_payload = {
        "id": "req-1",
        "decision": "yes",
        "top_probability": 0.8,
        "input_tokens": 12,
        "timing": {"total_ms": 5.0},
        "provenance": {"runtime": {"runtime_instance_id": "runtime-1"}},
    }
    event_id = server._record_watch_event(
        expected_session_id=session_id,
        endpoint="/v1/noul",
        request_payload=request_payload,
        response_payload=response_payload,
        status_code=200,
        request_id="req-1",
        mode="serial",
        decisions=1,
        settings_snapshot=settings,
    )

    session, rows = server._watch_rows_snapshot()
    assert session["requests"] == 1
    assert session["decisions"] == 1
    assert session["errors"] == 0
    assert session["storage"]["kind"] == "temporary-jsonl"
    assert session["storage"]["max_lines_per_file"] == 10_000
    assert rows[0]["request_id"] == "req-1"
    assert rows[0]["source"] == "api"
    stored = server._watch_store().get_event(str(event_id))
    assert stored is not None
    assert stored["request"] == request_payload
    assert stored["response"]["decision"] == "yes"
    event_files = list((tmp_path / ".deqio" / "watch").glob("events-*.jsonl"))
    assert len(event_files) == 1
    assert len(event_files[0].read_text(encoding="utf-8").splitlines()) == 1

    server._reset_watch_session(reason="manual-clear")
    assert list((tmp_path / ".deqio" / "watch").glob("events-*.jsonl")) == []
    server._record_watch_event(
        expected_session_id=session_id,
        endpoint="/v1/noul",
        request_payload=request_payload,
        response_payload=response_payload,
        status_code=200,
        request_id="req-old",
        mode="serial",
        decisions=1,
        settings_snapshot=settings,
    )
    reset_session, reset_rows = server._watch_rows_snapshot()
    assert reset_session["id"] != session_id
    assert reset_session["requests"] == 0
    assert reset_rows == []


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
    server._reset_watch_session()
    previous_watch_session = server._watch_session_token()
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
    watch_session, watch_rows = server._watch_rows_snapshot()
    assert watch_session["id"] == previous_watch_session
    assert watch_rows == []

def test_release_version_is_consistent() -> None:
    import tomllib

    from deqio import __version__
    from deqio.server import app

    project = tomllib.loads(Path("pyproject.toml").read_text())

    assert __version__ == "0.3.0"
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
    from deqio.ui import DASHBOARD, WATCH_DASHBOARD


def _asgi_post_json(path: str, body: bytes | str, headers: dict[str, str]) -> tuple[int, dict]:
    raw = body.encode("utf-8") if isinstance(body, str) else body

    async def invoke() -> tuple[int, dict]:
        sent: list[dict] = []
        delivered = False

        async def receive() -> dict:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": raw, "more_body": False}

        async def send(message: dict) -> None:
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in headers.items()],
            "client": ("testclient", 123),
            "server": ("testserver", 80),
            "root_path": "",
        }
        await app(scope, receive, send)
        status = next(message["status"] for message in sent if message["type"] == "http.response.start")
        response_body = b"".join(
            message.get("body", b"") for message in sent if message["type"] == "http.response.body"
        )
        return int(status), json.loads(response_body.decode("utf-8"))

    return asyncio.run(invoke())

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
        assert any("github.com/TheoLeeCJ/SemIf-OpenJev.git" in package for package in profile["packages"])


def test_semif_mlx_profile_uses_canonical_8bit_runtime_quantization() -> None:
    from deqio.catalog import get_profile, load_catalog

    profile = get_profile(load_catalog(Path("models.json")), "semif-qwen3.5-4b", "mlx")

    assert profile["model"] == "Qwen/Qwen3.5-4B"
    assert profile["model_revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert "download" not in profile
    assert profile["semif_mlx_bits"] == 8
    assert profile["quantization"] == {
        "kind": "mlx-affine",
        "bits": 8,
        "group_size": 64,
        "applied": "load-time",
        "source_precision": "bf16",
    }


def test_semif_mlx_command_passes_quantization_bits(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneRuntime

    settings = SimpleNamespace(
        engine="semif",
        backend="mlx",
        model="Qwen/Qwen3.5-4B",
        model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        max_tokens=8192,
        mlx_cache_mib=256,
        torch_dtype="bfloat16",
    )
    command = SystemOneRuntime._command(
        settings,
        {"model": settings.model, "semif_mlx_bits": 8},
        tmp_path / "runtime",
        tmp_path / "python",
        9009,
        {},
    )

    assert command[command.index("--mlx-bits") + 1] == "8"
    assert command[command.index("--model") + 1] == "Qwen/Qwen3.5-4B"


def test_kev_profiles_are_pinned_to_release_1_0_and_9b_guardrail_stays_bf16() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    for model_id in ("kev-0.8b", "kev-4b", "kev-9b", "kev-27b"):
        entry = next(item for item in catalog["models"] if item["id"] == model_id)
        for profile in entry["backends"].values():
            assert profile["model"].endswith("@v1.0")
            assert profile["model_revision"] == "v1.0"
            assert profile["download"]["revision"] == "v1.0"
            assert profile["packages"] == [
                "kev[serve] @ git+https://github.com/jaredpalmer/kev.git@kev-1.0"
            ]

    kev9 = get_profile(catalog, "kev-9b", "mlx")
    assert kev9["min_memory_gib"] == 22
    assert kev9["recommended_memory_gib"] == 32
    assert "quantization" not in kev9


def test_runtime_identity_binds_catalog_quantization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import deqio.systemone_runtime as runtime_module
    from types import SimpleNamespace

    monkeypatch.setattr(runtime_module, "installation_record", lambda *a, **k: {
        "artifacts": [{"source": "huggingface", "resolved_revision": "abc"}],
        "verified_at": "now",
        "max_input_tokens": 8192,
    })
    settings = SimpleNamespace(
        config_path=tmp_path / "config.json", model_id="demo", backend="mlx",
        engine="semif", model="owner/model", model_revision="rev", max_tokens=8192,
    )
    quantization = {"kind": "mlx-affine", "bits": 8, "group_size": 64}

    identity = runtime_module._runtime_identity(
        settings, {"model": "owner/model", "model_revision": "rev", "quantization": quantization}, "instance"
    )

    assert identity["quantization"] == quantization
    assert identity["quantization"] is not quantization


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


def test_hardware_memory_preflight_blocks_kev_9b_on_16_gib_mac() -> None:
    from deqio.hardware import HostCapabilities, profile_compatibility

    profile = {"min_memory_gib": 22, "recommended_memory_gib": 32}
    small = HostCapabilities("Darwin", "arm64", ("mlx", "mps"), 16.0, None)
    large = HostCapabilities("Darwin", "arm64", ("mlx", "mps"), 32.0, None)

    blocked = profile_compatibility("mlx", profile, host=small)
    allowed = profile_compatibility("mlx", profile, host=large)

    assert blocked["compatible"] is False
    assert "22.0 GiB" in blocked["reason"]
    assert allowed["compatible"] is True


def test_kev_4b_mlx_is_compatible_with_16_gib_apple_silicon() -> None:
    from deqio.hardware import HostCapabilities, profile_compatibility
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    profile = get_profile(catalog, "kev-4b", "mlx")
    host = HostCapabilities("Darwin", "arm64", ("mlx", "mps"), 16.0, None)

    result = profile_compatibility("mlx", profile, host=host)

    assert profile["min_memory_gib"] == 12
    assert profile["recommended_memory_gib"] == 16
    assert result["available_memory_gib"] == 12.0
    assert result["compatible"] is True


def test_installed_registry_does_not_leak_shared_kev_runtime_across_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.installations as installations
    from deqio.hardware import HostCapabilities

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    runtime_python = tmp_path / ".model-runtimes" / "kev" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("")
    catalog = {
        "models": [
            {
                "id": "kev-0.8b",
                "engine": "kev",
                "label": "Kev 0.8B",
                "backends": {
                    "mlx": {"model": "jaredpalmer/kev-0.8b", "runtime_key": "kev"},
                    "mps": {"model": "jaredpalmer/kev-0.8b", "runtime_key": "kev"},
                },
            }
        ]
    }
    monkeypatch.setattr(installations, "_cached_hf_repos", lambda: {"jaredpalmer/kev-0.8b"})
    monkeypatch.setattr(
        installations,
        "detect_host",
        lambda: HostCapabilities("Darwin", "arm64", ("mlx", "mps"), 32.0, None),
    )

    installations.mark_installed(config_path, "kev-0.8b", "mlx", verified=True)
    rows = installations.installed_profiles(
        config_path=config_path,
        config_data={"runtime_dir": ".model-runtimes"},
        catalog=catalog,
        active_model_id="kev-0.8b",
        active_backend="mlx",
    )
    by_backend = {row["backend"]: row for row in rows}

    assert by_backend["mlx"]["installed"] is True
    assert by_backend["mps"]["installed"] is False
    assert by_backend["mps"]["status"] == "unregistered"


def test_install_profile_registers_only_after_model_ready_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.model_manager as manager
    from deqio.installations import load_registry, profile_key

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    catalog = {
        "models": [
            {
                "id": "demo",
                "engine": "kev",
                "label": "Demo",
                "backends": {
                    "mlx": {
                        "model": "owner/demo",
                        "runtime_key": "demo-runtime",
                        "packages": ["demo"],
                    }
                },
            }
        ]
    }
    calls: list[str] = []
    runtime = tmp_path / ".model-runtimes" / "demo-runtime"
    monkeypatch.setattr(manager, "_preflight_profile", lambda *a, **k: {"compatible": True})
    monkeypatch.setattr(manager, "_install_runtime", lambda *a, **k: runtime)
    monkeypatch.setattr(manager, "_prefetch_declared_weights", lambda *a, **k: "cached")
    monkeypatch.setattr(manager, "_verify_model_ready", lambda *a, **k: calls.append("ready"))
    monkeypatch.setattr(
        manager,
        "_artifact_attestation",
        lambda *a, **k: [{"source": "huggingface", "repo_id": "owner/demo", "resolved_revision": "a" * 40}],
    )

    manager._install_profile(
        config_path=config_path,
        data={"runtime_dir": ".model-runtimes"},
        catalog=catalog,
        model_id="demo",
        backend="mlx",
        upgrade=False,
        force=False,
    )

    assert calls == ["ready"]
    record = load_registry(config_path)["profiles"][profile_key("demo", "mlx")]
    assert "verified_at" in record
    assert record["artifacts"][0]["resolved_revision"] == "a" * 40


def test_systemone_load_separates_process_ready_and_model_ready_and_uses_offline_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.systemone_runtime as runtime_module
    from deqio.config import settings_from_data

    catalog_path = tmp_path / "models.json"
    catalog_path.write_text(json.dumps({
        "models": [{
            "id": "demo",
            "engine": "kev",
            "backends": {"mlx": {"model": "owner/demo", "runtime_key": "kev", "wire_model": "demo"}},
        }]
    }))
    python = tmp_path / ".model-runtimes" / "kev" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    settings = settings_from_data(
        tmp_path / "config.json",
        {
            "engine": "kev",
            "model_id": "demo",
            "backend": "mlx",
            "model": "owner/demo",
            "model_revision": "demo",
            "model_catalog": str(catalog_path),
            "runtime_dir": str(tmp_path / ".model-runtimes"),
            "max_tokens": 4096,
            "mlx_cache_mib": 256,
            "log": str(tmp_path / "requests.jsonl"),
            "torch_dtype": "bfloat16",
            "sidecar_process_ready_seconds": 17,
            "sidecar_startup_seconds": 321,
            "hf_offline_runtime": True,
        },
        apply_environment=False,
    )

    class DummyProcess:
        stdout = None
        returncode = None
        def poll(self): return None
        def terminate(self): self.returncode = 0
        def wait(self, timeout=None): self.returncode = 0; return 0
        def kill(self): self.returncode = -9

    observed: dict[str, object] = {}
    monkeypatch.setattr(runtime_module.SystemOneRuntime, "_validate_accelerator", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(runtime_module.SystemOneRuntime, "_command", staticmethod(lambda *a, **k: ["demo"]))
    monkeypatch.setattr(runtime_module, "_free_port", lambda: 12345)
    monkeypatch.setattr(runtime_module, "_wait_for_port", lambda port, process, timeout: observed.update(port_timeout=timeout))
    monkeypatch.setattr(runtime_module, "start_sidecar_log_pump", lambda *a, **k: None)
    monkeypatch.setattr(
        runtime_module.subprocess,
        "Popen",
        lambda *a, **k: (observed.update(env=k["env"]) or DummyProcess()),
    )
    monkeypatch.setattr(
        runtime_module.SystemOneRuntime,
        "_probe_model_ready",
        lambda self, timeout: observed.update(model_timeout=timeout),
    )

    runtime = runtime_module.SystemOneRuntime.load(settings)
    runtime.close()

    assert observed["port_timeout"] == 17.0
    assert observed["model_timeout"] == 321.0
    env = observed["env"]
    assert isinstance(env, dict)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_model_manager_parser_exposes_delete_and_force_install() -> None:
    from deqio.model_manager import build_parser

    parser = build_parser()
    delete = parser.parse_args(["delete", "kev-4b", "--backend", "mlx", "--yes"])
    install = parser.parse_args(["install", "kev-9b", "--backend", "mlx", "--force"])

    assert delete.command == "delete"
    assert delete.model_id == "kev-4b"
    assert delete.yes is True
    assert install.command == "install"
    assert install.force is True


def test_public_catalog_exposes_profile_memory_guardrails() -> None:
    from deqio.catalog import public_catalog

    rows = public_catalog({
        "models": [{
            "id": "demo",
            "engine": "kev",
            "label": "Demo",
            "backends": {
                "mlx": {
                    "model": "owner/demo",
                    "min_memory_gib": 12,
                    "recommended_memory_gib": 16,
                }
            },
        }]
    })

    assert rows[0]["backends"] == ["mlx"]
    assert rows[0]["profiles"]["mlx"] == {
        "min_memory_gib": 12,
        "recommended_memory_gib": 16,
    }


def test_systemone_timeout_reports_model_ready_context(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.systemone_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "urlopen", lambda *a, **k: (_ for _ in ()).throw(TimeoutError()))

    with pytest.raises(RuntimeError, match=r"Timed out waiting for model response.*12\.5s"):
        runtime_module._post_json(
            "http://127.0.0.1:12345/v1/systemone",
            {"state": "probe"},
            timeout=12.5,
        )


def test_delete_cleanup_preserves_shared_runtime_until_last_profile(tmp_path: Path) -> None:
    import deqio.model_manager as manager

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    data = {"runtime_dir": ".model-runtimes"}
    runtime_dir = tmp_path / ".model-runtimes" / "kev"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "marker").write_text("shared")

    catalog = {
        "models": [
            {
                "id": "kev-a",
                "engine": "kev",
                "backends": {"mlx": {"model": "owner/kev-a", "runtime_key": "kev"}},
            },
            {
                "id": "kev-b",
                "engine": "kev",
                "backends": {"mlx": {"model": "owner/kev-b", "runtime_key": "kev"}},
            },
        ]
    }
    first = catalog["models"][0]["backends"]["mlx"]
    second = catalog["models"][1]["backends"]["mlx"]

    manager._cleanup_profile_artifacts(
        config_path=config_path,
        data=data,
        catalog=catalog,
        profile=first,
        remaining_keys={"kev-b::mlx"},
        purge_cache=False,
    )
    assert runtime_dir.is_dir()

    manager._cleanup_profile_artifacts(
        config_path=config_path,
        data=data,
        catalog=catalog,
        profile=second,
        remaining_keys=set(),
        purge_cache=False,
    )
    assert not runtime_dir.exists()


def test_catalog_contains_current_decision_families_and_nimble() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))

    assert get_profile(catalog, "kev-27b", "cuda")["model"] == "jaredpalmer/kev-27b@v1.0"
    assert get_profile(catalog, "jevk5-4b", "cuda")["model"] == "alibiserikbay/JevK5"
    assert get_profile(catalog, "jevk5-9b", "cuda")["model"] == "alibiserikbay/JevK5-9B"
    assert get_profile(catalog, "open-jev-2b", "cuda")["model"].endswith("open-jev-2b/package/checkpoint")
    assert get_profile(catalog, "open-jev-9b", "cuda")["model"].endswith("open-jev-9b/package/checkpoint")
    assert get_profile(catalog, "open-jev-27b-v1.1", "cuda")["model"].endswith(
        "open-jev-27b-v1.1/package/checkpoint"
    )
    assert get_profile(catalog, "clm-8b", "cuda")["clm_checkpoint"] == "models/clm-8b/CLM_v0.1-8B.pt"
    assert get_profile(catalog, "clm-8b", "cuda")["packages"] == [
        "clm[serve,hf,vllm] @ git+https://github.com/Contrastive-LM/CLM.git"
    ]
    assert get_profile(catalog, "nimble-9b", "mlx")["repo_id"] == "bespokelabs/Bespoke-Nimble-9B"
    assert get_profile(catalog, "nimble-9b", "cuda")["repo_id"] == "bespokelabs/Bespoke-Nimble-9B"

    for model_id in ("kev-27b", "jevk5-4b", "jevk5-9b", "open-jev-2b", "open-jev-9b", "open-jev-27b-v1.1", "clm-8b"):
        with pytest.raises(RuntimeError, match="does not support backend"):
            get_profile(catalog, model_id, "mps")
        with pytest.raises(RuntimeError, match="does not support backend"):
            get_profile(catalog, model_id, "mlx")


def test_prefetch_declared_weights_fetches_every_declared_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.model_manager as manager

    calls: list[tuple[str, dict[str, object]]] = []

    def fake_snapshot(**kwargs):
        calls.append(("snapshot", dict(kwargs)))
        return "/cache/snapshot"

    def fake_file(*, filename, **kwargs):
        calls.append(("file", {"filename": filename, **kwargs}))
        return "/cache/file"

    monkeypatch.setattr(manager, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(manager, "hf_hub_download", fake_file)

    profile = {
        "model": "Qwen/Qwen3-8B",
        "model_revision": "upstream-latest",
        "downloads": [
            {"type": "snapshot", "repo_id": "Qwen/Qwen3-8B"},
            {
                "type": "file",
                "repo_id": "Contrastive-LM/CLM-v0.1-8B",
                "filename": "CLM_v0.1-8B.pt",
                "local_dir": "models/clm-8b",
            },
        ],
    }

    message = manager._prefetch_declared_weights(tmp_path / "config.json", profile)

    assert [kind for kind, _ in calls] == ["snapshot", "file"]
    assert calls[1][1]["local_dir"] == str((tmp_path / "models" / "clm-8b").resolve())
    assert "Qwen/Qwen3-8B" in message
    assert "CLM_v0.1-8B.pt" in message



def test_nimble_installer_passes_audited_revision_to_preparer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.model_manager as manager

    source_dir = tmp_path / "nimble-src"
    requirements = source_dir / "requirements"
    requirements.mkdir(parents=True)
    (requirements / "mlx.txt").write_text("", encoding="utf-8")
    (requirements / "training.txt").write_text("", encoding="utf-8")

    runtime_root = tmp_path / ".model-runtimes"
    for runtime_name in ("nimble-mlx", "nimble-prep"):
        python_path = runtime_root / runtime_name / "bin" / "python"
        python_path.parent.mkdir(parents=True)
        python_path.write_text("", encoding="utf-8")

    commands: list[list[str]] = []
    monkeypatch.setattr(manager, "_checkout_nimble", lambda *a, **k: source_dir)
    monkeypatch.setattr(manager, "_run", lambda command: commands.append(list(command)))

    revision = "bd792f44ec8e265be861bfcdf4e05967ffe0e858"
    manager._install_nimble(
        tmp_path / "config.json",
        {"runtime_dir": ".model-runtimes", "backend": "mlx"},
        {
            "model": "models/nimble-9b",
            "repo_id": "bespokelabs/Bespoke-Nimble-9B",
            "model_revision": revision,
            "runtime_key": "nimble-mlx",
            "model_config": "nimble-model.json",
            "nimble_backend": "mlx",
            "python": "3.12",
        },
        upgrade=False,
    )

    prepare = next(command for command in commands if "nimble_prepare.py" in " ".join(command))
    assert prepare[prepare.index("--repo-id") + 1] == "bespokelabs/Bespoke-Nimble-9B"
    assert prepare[prepare.index("--revision") + 1] == revision

def test_stale_nimble_preparation_is_not_reported_as_latest_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.installations as installations
    from deqio.hardware import HostCapabilities

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    runtime = tmp_path / ".model-runtimes" / "nimble-mlx"
    runtime_python = runtime / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("")
    model_dir = tmp_path / "models" / "nimble-9b"
    model_dir.mkdir(parents=True)
    (model_dir / "READY.json").write_text("{}")
    nimble_config = tmp_path / ".model-runtimes" / "nimble-model.json"
    nimble_config.write_text(json.dumps({"model_id": "bespokelabs/Bespoke-Nimble-9B-v2", "revision": "old"}))

    profile = {
        "model": "models/nimble-9b",
        "runtime_key": "nimble-mlx",
        "model_config": "nimble-model.json",
        "repo_id": "bespokelabs/Bespoke-Nimble-9B",
        "model_revision": "bd792f44ec8e265be861bfcdf4e05967ffe0e858",
        "installer": "nimble",
    }
    catalog = {"models": [{"id": "nimble-9b", "engine": "nimble", "backends": {"mlx": profile}}]}
    monkeypatch.setattr(installations, "_cached_hf_repos", lambda: set())
    monkeypatch.setattr(
        installations,
        "detect_host",
        lambda: HostCapabilities("Darwin", "arm64", ("mlx", "mps"), 32.0, None),
    )
    installations.mark_installed(config_path, "nimble-9b", "mlx", verified=True)

    row = installations.installed_profiles(
        config_path=config_path,
        config_data={"runtime_dir": ".model-runtimes"},
        catalog=catalog,
    )[0]

    assert row["installed"] is False
    assert row["status"] == "weights-missing"


def test_nimble_preparation_requires_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.installations as installations
    from deqio.hardware import HostCapabilities

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    runtime = tmp_path / ".model-runtimes" / "nimble-mlx"
    runtime_python = runtime / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("")
    model_dir = tmp_path / "models" / "nimble-9b"
    model_dir.mkdir(parents=True)
    (model_dir / "READY.json").write_text("{}")
    model_config = tmp_path / ".model-runtimes" / "nimble-model.json"
    model_config.write_text(json.dumps({
        "model_id": "bespokelabs/Bespoke-Nimble-9B",
        "revision": "older-revision",
    }))
    profile = {
        "model": "models/nimble-9b",
        "runtime_key": "nimble-mlx",
        "model_config": "nimble-model.json",
        "repo_id": "bespokelabs/Bespoke-Nimble-9B",
        "model_revision": "bd792f44ec8e265be861bfcdf4e05967ffe0e858",
        "installer": "nimble",
        "min_memory_gib": 20,
    }
    catalog = {"models": [{"id": "nimble-9b", "engine": "nimble", "backends": {"mlx": profile}}]}
    monkeypatch.setattr(installations, "_cached_hf_repos", lambda: set())
    monkeypatch.setattr(
        installations, "detect_host",
        lambda: HostCapabilities("Darwin", "arm64", ("mlx", "mps"), 32.0, None),
    )
    installations.mark_installed(config_path, "nimble-9b", "mlx", verified=True)

    row = installations.installed_profiles(
        config_path=config_path, config_data={"runtime_dir": ".model-runtimes"}, catalog=catalog
    )[0]

    assert row["installed"] is False
    assert row["status"] == "weights-missing"


def test_systemone_commands_for_new_native_sidecars(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneRuntime

    python = tmp_path / "python"
    env_dir = tmp_path / "runtime"
    env: dict[str, str] = {}

    jevk5 = SimpleNamespace(engine="jevk5", backend="cuda")
    command = SystemOneRuntime._command(
        jevk5, {"model": "alibiserikbay/JevK5"}, env_dir, python, 9010, env
    )
    assert command[-4:] == ["--host", "127.0.0.1", "--port", "9010"]
    assert "jevk5.server" in command

    open_model = tmp_path / "models" / "open-jev" / "package" / "checkpoint"
    open_model.mkdir(parents=True)
    openjev = SimpleNamespace(engine="open-jev", backend="cuda", model=str(open_model))
    command = SystemOneRuntime._command(
        openjev,
        {"model": "./models/open-jev/package/checkpoint", "open_jev_max_length": 4096},
        env_dir,
        python,
        9011,
        {},
    )
    assert "jev.server" in command
    assert command[command.index("--checkpoint") + 1] == str(open_model)
    assert command[command.index("--device") + 1] == "cuda:0"

    clm_checkpoint = tmp_path / "models" / "clm-8b" / "CLM_v0.1-8B.pt"
    clm_checkpoint.parent.mkdir(parents=True)
    clm_checkpoint.write_bytes(b"head")
    clm = SimpleNamespace(
        engine="clm",
        backend="cuda",
        config_path=tmp_path / "config.json",
        model_id="clm-8b",
    )
    command = SystemOneRuntime._command(
        clm,
        {
            "model": "Qwen/Qwen3-8B",
            "clm_checkpoint": "models/clm-8b/CLM_v0.1-8B.pt",
            "clm_embedding_model": "qwen3-8b",
            "clm_max_tokens": 2048,
            "clm_gpu_memory_utilization": 0.35,
        },
        env_dir,
        python,
        9012,
        {},
    )
    assert command[1].endswith("clm_sidecar.py")
    assert command[command.index("--checkpoint") + 1] == str(clm_checkpoint.resolve())
    assert command[command.index("--encoder-model") + 1] == "Qwen/Qwen3-8B"


def test_cuda_only_profiles_can_restrict_the_supported_operating_system() -> None:
    from deqio.hardware import HostCapabilities, profile_compatibility

    profile = {"systems": ["Linux"], "min_memory_gib": 1}
    windows = HostCapabilities("Windows", "AMD64", ("cuda",), 32.0, 24.0)
    linux = HostCapabilities("Linux", "x86_64", ("cuda",), 32.0, 24.0)

    assert profile_compatibility("cuda", profile, host=windows)["compatible"] is False
    assert profile_compatibility("cuda", profile, host=linux)["compatible"] is True


def test_cli_version_commands_report_release_version(capsys: pytest.CaptureFixture[str]) -> None:
    from deqio import __version__
    from deqio.cli import main

    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"deqio {__version__}"
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == f"deqio {__version__}"


def test_model_manager_interactive_prompt_accepts_q(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.model_manager as manager

    monkeypatch.setattr("builtins.input", lambda prompt="": "q")
    assert manager._prompt_index("Select model", 3) is None


def test_benchmark_interactive_selection_accepts_q(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import deqio.benchmark as benchmark

    rows = [{"label": "Demo", "backend": "mlx", "engine": "demo"}]
    monkeypatch.setattr("builtins.input", lambda prompt="": "q")
    selected = benchmark._select_profiles(rows, argparse.Namespace(all=False, model=None))
    assert selected is None


def test_external_choice_score_provenance_marks_synthetic_one_hot() -> None:
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
    raw = _choice_raw(
        row=row,
        answer={"choice": "billing"},
        response={},
        payload={"state": row["state"], "questions": {}},
        latency_ms=1.0,
    )

    assert raw["probabilities"] == [1.0, 0.0]
    assert raw["score_provenance"]["kind"] == "synthetic_one_hot"
    assert raw["score_provenance"]["source"] == "engine.choice"
    assert raw["score_provenance"]["synthetic"] is True
    assert raw["score_provenance"]["transforms"] == ["one_hot_fallback"]
    assert "not a model confidence score" in raw["probability_status"]


def test_external_choice_score_provenance_records_renormalization() -> None:
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
    raw = _choice_raw(
        row=row,
        answer={"choice": "billing", "probabilities": {"billing": 8.0, "technical": 2.0}},
        response={},
        payload={"state": row["state"], "questions": {}},
        latency_ms=1.0,
    )

    assert raw["probabilities"] == [0.8, 0.2]
    assert raw["score_provenance"]["kind"] == "engine_probability"
    assert raw["score_provenance"]["normalized"] is True
    assert raw["score_provenance"]["transforms"] == ["renormalized"]


def test_format_result_binds_decision_provenance_and_local_attestation() -> None:
    from deqio.server import format_result

    identity = {
        "deqio_version": "0.2.1",
        "runtime_instance_id": "runtime-123",
        "engine": "decider",
        "model_id": "decider-4b",
        "backend": "mps",
        "model": "Mapika/decider-4b",
        "requested_revision": None,
        "artifacts": [{
            "source": "huggingface",
            "repo_id": "Mapika/decider-4b",
            "resolved_revision": "a" * 40,
        }],
        "artifact_revisions_resolved": True,
        "installation_verified_at": "2026-09-29T00:00:00+00:00",
    }
    raw = {
        "id": "req-1",
        "option_ids": ["yes", "no"],
        "probabilities": [0.8, 0.2],
        "input_tokens": 42,
        "total_seconds": 0.1,
        "prompt_sha256": "prompt",
        "probability_status": "native",
        "score_provenance": {
            "kind": "engine_probability",
            "source": "engine.probabilities",
            "synthetic": False,
            "normalized": False,
            "transforms": [],
            "raw_logits_available": False,
            "calibration": "unspecified",
        },
    }

    result = format_result(raw, runtime_identity=identity)

    assert result["provenance"]["schema_version"] == 1
    assert result["provenance"]["runtime"]["runtime_instance_id"] == "runtime-123"
    assert result["provenance"]["score"]["source"] == "engine.probabilities"
    assert result["provenance"]["attestation"]["signed"] is False
    assert result["provenance"]["attestation"]["complete"] is True
    assert len(result["provenance"]["attestation"]["sha256"]) == 64


def test_runtime_identity_reads_resolved_installation_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deqio.config import settings_from_data
    from deqio.installations import mark_installed
    from deqio.systemone_runtime import _runtime_identity

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    catalog_path = tmp_path / "models.json"
    catalog_path.write_text('{"models": []}')
    settings = settings_from_data(
        config_path,
        {
            "engine": "kev",
            "model_id": "kev-0.8b",
            "backend": "mlx",
            "model": "jaredpalmer/kev-0.8b",
            "model_revision": "kev-0.8b",
            "model_catalog": str(catalog_path),
            "runtime_dir": str(tmp_path / ".model-runtimes"),
            "max_tokens": 4096,
            "mlx_cache_mib": 256,
            "log": str(tmp_path / "requests.jsonl"),
            "torch_dtype": "bfloat16",
            "sidecar_startup_seconds": 900,
            "hf_offline_runtime": True,
        },
        apply_environment=False,
    )
    artifacts = [{
        "source": "huggingface",
        "repo_id": "jaredpalmer/kev-0.8b",
        "requested_revision": None,
        "resolved_revision": "b" * 40,
    }]
    mark_installed(config_path, "kev-0.8b", "mlx", verified=True, artifacts=artifacts)

    identity = _runtime_identity(settings, {"model": "jaredpalmer/kev-0.8b"}, "runtime-abc")

    assert identity["runtime_instance_id"] == "runtime-abc"
    assert identity["artifacts"] == artifacts
    assert identity["artifact_revisions_resolved"] is True
    assert identity["installation_verified_at"] is not None


def test_decision_provenance_uses_captured_runtime_identity_not_later_global_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.server as server
    from deqio.config import settings_from_data

    def make_settings(model_id: str, model: str):
        return settings_from_data(
            tmp_path / "config.json",
            {
                "engine": "kev",
                "model_id": model_id,
                "backend": "mlx",
                "model": model,
                "model_revision": model_id,
                "model_catalog": str(tmp_path / "models.json"),
                "runtime_dir": str(tmp_path / ".model-runtimes"),
                "max_tokens": 4096,
                "mlx_cache_mib": 256,
                "log": str(tmp_path / "requests.jsonl"),
                "torch_dtype": "bfloat16",
                "sidecar_startup_seconds": 900,
                "hf_offline_runtime": True,
            },
            apply_environment=False,
        )

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "models.json").write_text('{"models": []}')
    old_settings = make_settings("kev-0.8b", "jaredpalmer/kev-0.8b")
    new_settings = make_settings("kev-4b", "jaredpalmer/kev-4b")

    class FakeRuntime:
        runtime_instance_id = "old-runtime"

        def identity_snapshot(self):
            return {
                "deqio_version": "0.2.1",
                "runtime_instance_id": "old-runtime",
                "engine": "kev",
                "model_id": "kev-0.8b",
                "backend": "mlx",
                "model": "jaredpalmer/kev-0.8b",
                "requested_revision": None,
                "artifacts": [{"source": "huggingface", "resolved_revision": "c" * 40}],
                "artifact_revisions_resolved": True,
                "installation_verified_at": "2026-09-29T00:00:00+00:00",
            }

        def score(self, row, mode):
            server.SETTINGS = new_settings
            return {
                "id": row["id"],
                "option_ids": ["a", "b"],
                "probabilities": [0.75, 0.25],
                "input_tokens": 1,
                "total_seconds": 0.001,
                "prompt_sha256": "prompt",
                "probability_status": "native",
                "score_provenance": {
                    "kind": "engine_probability",
                    "source": "engine.probabilities",
                    "synthetic": False,
                    "normalized": False,
                    "transforms": [],
                    "raw_logits_available": False,
                    "calibration": "unspecified",
                },
            }

    monkeypatch.setattr(server, "SETTINGS", old_settings)
    monkeypatch.setattr(server, "runtime", FakeRuntime())
    monkeypatch.setattr(server, "log_request_success", lambda *a, **k: None)

    result = server.run_decision(server.DecisionRequest(
        id="req-atomic",
        state="state",
        question="choose",
        options=[server.Option(id="a", description="A"), server.Option(id="b", description="B")],
    ))

    assert server.SETTINGS.model_id == "kev-4b"
    assert result["provenance"]["runtime"]["model_id"] == "kev-0.8b"
    assert result["provenance"]["runtime"]["runtime_instance_id"] == "old-runtime"


def test_missing_runtime_error_points_to_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deqio.config import settings_from_data
    from deqio.systemone_runtime import SystemOneRuntime

    catalog_path = tmp_path / "models.json"
    catalog_path.write_text(json.dumps({
        "models": [{
            "id": "demo",
            "engine": "kev",
            "backends": {"mlx": {"model": "owner/demo", "runtime_key": "kev"}},
        }]
    }))
    settings = settings_from_data(
        tmp_path / "config.json",
        {
            "engine": "kev",
            "model_id": "demo",
            "backend": "mlx",
            "model": "owner/demo",
            "model_revision": "demo",
            "model_catalog": str(catalog_path),
            "runtime_dir": str(tmp_path / ".model-runtimes"),
            "max_tokens": 4096,
            "mlx_cache_mib": 256,
            "log": str(tmp_path / "requests.jsonl"),
            "torch_dtype": "bfloat16",
            "sidecar_startup_seconds": 900,
            "hf_offline_runtime": True,
        },
        apply_environment=False,
    )

    with pytest.raises(RuntimeError, match="deqio models setup") as error:
        SystemOneRuntime.load(settings)
    assert "uv run" not in str(error.value)


def test_shared_provenance_binds_per_result_attestations() -> None:
    from deqio.server import _shared_provenance

    runtime_identity = {
        "runtime_instance_id": "runtime-shared",
        "artifact_revisions_resolved": True,
    }
    results = [
        {"provenance": {"attestation": {"complete": True, "sha256": "a" * 64}}},
        {"provenance": {"attestation": {"complete": True, "sha256": "b" * 64}}},
    ]

    provenance = _shared_provenance(results, runtime_identity)

    assert provenance["runtime"]["runtime_instance_id"] == "runtime-shared"
    assert provenance["score"] == {
        "kind": "shared_batch",
        "source": "results[*].provenance.score",
        "result_count": 2,
    }
    assert provenance["attestation"]["complete"] is True
    assert len(provenance["attestation"]["sha256"]) == 64



def test_token_budget_options_apply_profile_and_host_guardrails() -> None:
    import deqio.model_manager as manager

    profile = {"min_memory_gib": 12}
    compatibility = {"minimum_memory_gib": 12.0, "available_memory_gib": 12.0}
    available, blocked = manager._token_budget_options(
        profile, compatibility, default=4096
    )

    assert available == [4096]
    blocked_values = {value for value, _reason in blocked}
    assert {8192, 12288, 16384, 32768}.issubset(blocked_values)


def test_token_budget_options_honor_engine_specific_hard_limit() -> None:
    import deqio.model_manager as manager

    profile = {"min_memory_gib": 8, "open_jev_max_length": 4096}
    compatibility = {"minimum_memory_gib": 8.0, "available_memory_gib": 24.0}
    available, blocked = manager._token_budget_options(
        profile, compatibility, default=4096
    )

    assert available == [4096]
    assert any(value == 8192 and "profile limit" in reason for value, reason in blocked)


def test_installation_registry_records_max_input_tokens(tmp_path: Path) -> None:
    from deqio.installations import load_registry, mark_installed, profile_key

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    mark_installed(
        config_path,
        "example",
        "mlx",
        verified=True,
        max_input_tokens=8192,
    )
    record = load_registry(config_path)["profiles"][profile_key("example", "mlx")]
    assert record["max_input_tokens"] == 8192


def test_input_contract_rejects_duplicate_json_keys() -> None:
    from deqio.input_contract import DuplicateJSONKeyError, validate_json_without_duplicate_keys

    with pytest.raises(DuplicateJSONKeyError):
        validate_json_without_duplicate_keys(b'{"id":"a","id":"b"}')


def test_negotiated_input_receipt_keeps_unknown_distinct_from_complete() -> None:
    from deqio.input_contract import InputContractContext, unknown_input_receipt

    receipt = unknown_input_receipt(
        context=InputContractContext(
            version="input-completeness-v1",
            request_body_sha256="a" * 64,
            require_complete=False,
            overflow="reject",
        ),
        runtime_identity={"runtime_instance_id": "runtime-1"},
        engine_payload_sha256="b" * 64,
        input_limit_tokens=4096,
        decision_inputs=[],
        input_tokens=None,
        input_tokens_source="unknown",
    )
    assert receipt["status"] == "unknown"
    assert receipt["usage"] == {"input_tokens": None, "input_tokens_source": "unknown"}
    assert receipt["input_limit_tokens"] == 4096


def test_attestation_v2_binds_negotiated_input_receipt() -> None:
    from deqio.server import format_result

    runtime_identity = {
        "runtime_instance_id": "runtime-1",
        "artifact_revisions_resolved": True,
    }
    receipt = {"schema": "deqio.input.v1", "status": "unknown"}
    result = format_result(
        {
            "id": "req-1",
            "option_ids": ["yes", "no"],
            "probabilities": [0.75, 0.25],
            "input_tokens": None,
            "input_tokens_source": "unknown",
            "prompt_sha256": "c" * 64,
            "probability_status": "native",
            "score_provenance": {"kind": "engine_probability"},
        },
        runtime_identity=runtime_identity,
        input_receipt=receipt,
    )
    assert result["input_receipt"] is receipt
    assert result["usage"]["input_tokens"] is None
    assert result["provenance"]["schema_version"] == 2
    assert result["provenance"]["attestation"]["complete"] is False


def test_strict_contract_fails_closed_for_uninstrumented_runtime() -> None:
    from deqio.input_contract import InputContractContext
    from deqio.server import InputContractHTTPError, _ensure_contract_capability

    class Runtime:
        def input_completeness_capability(self):
            return {"status": "unavailable", "reason": "backend_model_input_not_instrumented"}

    with pytest.raises(InputContractHTTPError) as error:
        _ensure_contract_capability(
            Runtime(),
            InputContractContext(
                version="input-completeness-v1",
                request_body_sha256="d" * 64,
                require_complete=True,
                overflow="reject",
            ),
            request_id="req-1",
        )
    assert error.value.payload["code"] == "input_completeness_unavailable"
    assert error.value.payload["inference_performed"] is False


def test_input_contract_duplicate_keys_rejected_before_endpoint_parsing() -> None:
    status, body = _asgi_post_json(
        "/v1/choice",
        (
            '{"id":"req-1","id":"req-2","state":"s","question":"q",'
            '"options":[{"id":"a","description":"A"}],'
            '"input_policy":{"require_complete":false,"overflow":"reject"}}'
        ),
        {
            "content-type": "application/json",
            "Deqio-Contract": "input-completeness-v1",
        },
    )

    assert status == 422
    assert body["error"]["code"] == "duplicate_json_key"
    assert len(body["error"]["request_body_sha256"]) == 64
    assert body["error"]["inference_performed"] is False


def test_unknown_input_contract_rejected_before_endpoint_parsing() -> None:
    status, body = _asgi_post_json(
        "/v1/choice",
        b"not-even-json",
        {
            "content-type": "application/json",
            "Deqio-Contract": "input-completeness-v999",
        },
    )

    assert status == 422
    assert body["error"]["code"] == "unsupported_contract"
    assert body["error"]["supported"] == ["input-completeness-v1"]
    assert body["error"]["inference_performed"] is False


def test_basal_15b_catalog_profiles_are_mlx_and_cuda_only() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    basal = next(entry for entry in catalog["models"] if entry["id"] == "basal-1.5b")

    assert basal["engine"] == "basal"
    assert set(basal["backends"]) == {"mlx", "cuda"}
    mlx = get_profile(catalog, "basal-1.5b", "mlx")
    cuda = get_profile(catalog, "basal-1.5b", "cuda")
    assert mlx["model"] == "pawelkiszczak/basal-1.0-1.5B-MLX-8bit"
    assert mlx["installer"] == "basal"
    assert mlx["basal_mode"] == "mlx"
    assert mlx["quantization"]["bits"] == 8
    assert mlx["model_revision"] == "097465843096292389d039e22265de8b972d2b7b"
    assert cuda["model_revision"] == "81a74acc6e7f7604008697b2daa83b3652d85b68"
    assert mlx["min_memory_gib"] == 3
    assert mlx["recommended_memory_gib"] == 4
    assert mlx["systems"] == ["Darwin"]
    assert cuda["model"] == "Remek/basal-1.0-1.5B"
    assert cuda["installer"] == "basal"
    assert cuda["systems"] == ["Linux"]
    assert cuda["basal_mode"] == "fast-nocompile"
    with pytest.raises(RuntimeError, match="does not support backend"):
        get_profile(catalog, "basal-1.5b", "mps")


def test_basal_installer_uses_backend_specific_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.model_manager as manager

    commands: list[list[str]] = []
    monkeypatch.setattr(manager, "_run", lambda command: commands.append(list(command)))

    mlx_env = tmp_path / ".model-runtimes" / "basal-mlx"
    mlx_python = mlx_env / "bin" / "python"
    mlx_python.parent.mkdir(parents=True)
    mlx_python.write_text("", encoding="utf-8")
    manager._install_basal(
        tmp_path / "config.json",
        {"backend": "mlx", "runtime_dir": ".model-runtimes"},
        {
            "runtime_key": "basal-mlx",
            "python": "3.12",
            "basal_backend": "mlx",
            "basal_package": "basal[mlx] @ git+https://github.com/pawelkiszczak/basal.git@main",
        },
        upgrade=False,
    )
    assert commands[0][-1] == "mlx>=0.32,<0.33"
    assert commands[1][-1] == "basal[mlx] @ git+https://github.com/pawelkiszczak/basal.git@main"

    commands.clear()
    cuda_env = tmp_path / ".model-runtimes" / "basal-cuda"
    cuda_python = cuda_env / "bin" / "python"
    cuda_python.parent.mkdir(parents=True)
    cuda_python.write_text("", encoding="utf-8")
    manager._install_basal(
        tmp_path / "config.json",
        {"backend": "cuda", "runtime_dir": ".model-runtimes"},
        {
            "runtime_key": "basal-cuda",
            "python": "3.12",
            "basal_backend": "cuda",
            "basal_package": "basal[fp8] @ https://github.com/rkinas/basal/archive/refs/tags/v1.0.1.tar.gz",
        },
        upgrade=False,
    )
    assert "torch==2.11.0" in commands[0]
    assert "https://download.pytorch.org/whl/cu128" in commands[0]
    assert commands[1][-1].endswith("basal/archive/refs/tags/v1.0.1.tar.gz")


def test_basal_systemone_commands_use_native_server(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneRuntime

    env_dir = tmp_path / "runtime"
    executable = env_dir / "bin" / "basal-serve"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    python = env_dir / "bin" / "python"
    python.write_text("", encoding="utf-8")

    mlx = SimpleNamespace(engine="basal", backend="mlx")
    command = SystemOneRuntime._command(
        mlx,
        {
            "model": "pawelkiszczak/basal-1.0-1.5B-MLX-8bit",
            "basal_mode": "mlx",
        },
        env_dir,
        python,
        9020,
        {},
    )
    assert command == [
        str(executable),
        "--model", "pawelkiszczak/basal-1.0-1.5B-MLX-8bit",
        "--mode", "mlx",
        "--port", "9020",
    ]

    cuda = SimpleNamespace(engine="basal", backend="cuda")
    command = SystemOneRuntime._command(
        cuda,
        {
            "model": "Remek/basal-1.0-1.5B",
            "basal_mode": "fast-nocompile",
        },
        env_dir,
        python,
        9021,
        {},
    )
    assert command == [
        str(executable),
        "--model", "Remek/basal-1.0-1.5B",
        "--mode", "fast-nocompile",
        "--port", "9021",
    ]


def test_patch3_catalog_adds_validated_quantized_profiles_only() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    ids = {str(entry["id"]) for entry in catalog["models"]}

    flash = get_profile(catalog, "clef-flash", "mlx")
    assert flash["model"] == "mlx-community/clef-flash-8bit"
    assert flash["model_revision"] == "dfa0993decb4f8507a0eae01afd1b2d33a4bb734"
    assert flash["wire_model"] == "clef-flash"
    assert flash["runtime_key"] == "clef-mlx"
    assert flash["quantization"] == {
        "kind": "mlx-affine",
        "bits": 8,
        "group_size": 64,
        "applied": "checkpoint",
        "source_model": "Cloudflare/clef-flash",
        "joint_head_precision": "bf16",
    }
    assert flash["min_memory_gib"] == pytest.approx(11.6)
    assert flash["recommended_memory_gib"] == 20
    assert flash["max_input_tokens"] == 16384
    assert flash["default_max_input_tokens"] == 4096

    clef = get_profile(catalog, "clef", "mlx")
    assert clef["model"] == "mlx-community/clef-8bit"
    assert clef["model_revision"] == "ffcdb6132b3cc94e523860321dd9ed58bc1ee9a1"
    assert clef["wire_model"] == "clef"
    assert clef["quantization"]["bits"] == 8
    assert clef["quantization"]["joint_head_precision"] == "bf16"
    assert clef["min_memory_gib"] == pytest.approx(30.8)
    assert clef["recommended_memory_gib"] == 44

    basal = get_profile(catalog, "basal-4.5b", "mlx")
    assert basal["model"] == "pawelkiszczak/basal-1.0-4.5B-MLX-8bit"
    assert basal["model_revision"] == "7c8291bce44c9096da1ca52ba2136a9722975c3e"
    assert basal["runtime_key"] == "basal-mlx-45"
    assert basal["basal_mode"] == "mlx"
    assert basal["quantization"]["bits"] == 8
    assert basal["min_memory_gib"] == 6
    assert basal["recommended_memory_gib"] == 8

    basal_cuda = get_profile(catalog, "basal-4.5b", "cuda")
    assert basal_cuda["model"] == "Remek/basal-1.0-4.5B-FP8"
    assert basal_cuda["model_revision"] == "6a4718943f85d80612242be688eb57e3037ec88e"
    assert basal_cuda["runtime_key"] == "basal-cuda-vllm"
    assert basal_cuda["basal_mode"] == "vllm"
    assert basal_cuda["min_cuda_compute_capability"] == 9.0
    assert basal_cuda["quantization"]["kind"] == "fp8"
    assert basal_cuda["quantization"]["output_head_precision"] == "bf16"

    for model_id in ("clef-flash", "clef"):
        with pytest.raises(RuntimeError, match="does not support backend"):
            get_profile(catalog, model_id, "cuda")
        with pytest.raises(RuntimeError, match="does not support backend"):
            get_profile(catalog, model_id, "mps")

    with pytest.raises(RuntimeError, match="does not support backend"):
        get_profile(catalog, "basal-4.5b", "mps")

    # Solar Decide is currently a hosted API product, not downloadable local
    # weights. Keep it out of the local MLX/MPS/CUDA catalog until Deqio grows
    # an explicit remote-provider backend and credential contract.
    assert "solar-decide" not in ids


def test_clef_systemone_command_uses_pinned_mlx_sidecar(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from deqio.systemone_runtime import SystemOneRuntime

    env_dir = tmp_path / "runtime"
    python = env_dir / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    settings = SimpleNamespace(engine="clef", backend="mlx", max_tokens=4096)
    profile = {
        "model": "mlx-community/clef-flash-8bit",
        "model_revision": "dfa0993decb4f8507a0eae01afd1b2d33a4bb734",
        "wire_model": "clef-flash",
        "max_input_tokens": 16384,
    }
    command = SystemOneRuntime._command(settings, profile, env_dir, python, 9030, {})

    assert command[0] == str(python)
    assert command[1].endswith("/deqio/clef_sidecar.py")
    assert command[2:] == [
        "--model", "mlx-community/clef-flash-8bit",
        "--revision", "dfa0993decb4f8507a0eae01afd1b2d33a4bb734",
        "--name", "clef-flash",
        "--max-length", "4096",
        "--port", "9030",
    ]

    with pytest.raises(RuntimeError, match="require MLX"):
        SystemOneRuntime._command(
            SimpleNamespace(engine="clef", backend="cuda", max_tokens=4096),
            profile,
            env_dir,
            python,
            9031,
            {},
        )


def test_clef_sidecar_executes_cached_upstream_server_without_truncation(tmp_path: Path) -> None:
    from deqio.clef_sidecar import _serve_command

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    script = snapshot / "clef_mlx.py"
    script.write_text("# upstream launcher\n", encoding="utf-8")

    command = _serve_command(
        python="/runtime/python",
        snapshot=snapshot,
        name="clef-flash",
        max_length=4096,
        port=9032,
    )
    assert command == [
        "/runtime/python",
        str(script),
        "serve",
        "--model", str(snapshot),
        "--name", "clef-flash",
        "--max-length", "4096",
        "--no-truncate",
        "--host", "127.0.0.1",
        "--port", "9032",
        "--quiet",
    ]


def test_basal_vllm_installer_leaves_torch_resolution_to_vllm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.model_manager as manager

    commands: list[list[str]] = []
    monkeypatch.setattr(manager, "_run", lambda command: commands.append(list(command)))

    env = tmp_path / ".model-runtimes" / "basal-cuda-vllm"
    python = env / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    manager._install_basal(
        tmp_path / "config.json",
        {"backend": "cuda", "runtime_dir": ".model-runtimes"},
        {
            "runtime_key": "basal-cuda-vllm",
            "python": "3.12",
            "basal_backend": "cuda",
            "basal_mode": "vllm",
            "basal_package": "basal[vllm] @ https://github.com/rkinas/basal/archive/refs/tags/v1.0.1.tar.gz",
        },
        upgrade=False,
    )

    assert len(commands) == 1
    assert commands[0][-1].endswith("basal/archive/refs/tags/v1.0.1.tar.gz")
    assert all("torch==" not in part for part in commands[0])


def test_basal_45b_systemone_cuda_uses_vllm_mode(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from deqio.systemone_runtime import SystemOneRuntime

    env_dir = tmp_path / "runtime"
    executable = env_dir / "bin" / "basal-serve"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    python = env_dir / "bin" / "python"
    python.write_text("", encoding="utf-8")

    command = SystemOneRuntime._command(
        SimpleNamespace(engine="basal", backend="cuda"),
        {
            "model": "Remek/basal-1.0-4.5B-FP8",
            "basal_mode": "vllm",
        },
        env_dir,
        python,
        9033,
        {},
    )
    assert command == [
        str(executable),
        "--model", "Remek/basal-1.0-4.5B-FP8",
        "--mode", "vllm",
        "--port", "9033",
    ]


def test_cuda_compute_capability_guard_is_fail_closed_for_fp8_profiles() -> None:
    from deqio.hardware import HostCapabilities, profile_compatibility

    profile = {
        "systems": ["Linux"],
        "min_memory_gib": 12,
        "recommended_memory_gib": 16,
        "min_cuda_compute_capability": 9.0,
    }
    unknown = HostCapabilities("Linux", "x86_64", ("cuda",), 64.0, 24.0)
    ada = HostCapabilities("Linux", "x86_64", ("cuda",), 64.0, 24.0, 8.9)
    hopper = HostCapabilities("Linux", "x86_64", ("cuda",), 64.0, 24.0, 9.0)

    unknown_result = profile_compatibility("cuda", profile, host=unknown)
    assert unknown_result["compatible"] is False
    assert "could not be detected" in unknown_result["reason"]

    ada_result = profile_compatibility("cuda", profile, host=ada)
    assert ada_result["compatible"] is False
    assert "host reports 8.9" in ada_result["reason"]

    hopper_result = profile_compatibility("cuda", profile, host=hopper)
    assert hopper_result["compatible"] is True


def test_clef_flash_8bit_is_short_context_compatible_on_16gib_apple_host() -> None:
    import deqio.model_manager as manager
    from deqio.catalog import get_profile, load_catalog
    from deqio.hardware import HostCapabilities, profile_compatibility

    profile = get_profile(load_catalog(Path("models.json")), "clef-flash", "mlx")
    host = HostCapabilities("Darwin", "arm64", ("mlx", "mps"), 16.0, None)
    compatibility = profile_compatibility("mlx", profile, host=host)

    assert compatibility["compatible"] is True
    assert compatibility["available_memory_gib"] == 12.0
    assert compatibility["warning"] is not None

    available, blocked = manager._token_budget_options(
        profile,
        compatibility,
        default=profile["default_max_input_tokens"],
    )
    assert available == [4096]
    assert any(value == 8192 for value, _reason in blocked)
    assert any(value == 32768 and "profile limit is 16384" in reason for value, reason in blocked)
