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
    assert "/v1/score" in routes
    assert "/v1/multi" in routes
    assert "/v1/act" in routes
    assert "/v1/soam" in routes
    assert "/v1/systemone" in routes
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

    # The root config.json is a developer-local, git-ignored file; a clean
    # checkout (CI) only has the packaged template.
    config_path = Path("src/deqio/data/config.json").resolve()
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




def test_model_manager_exposes_mlx_mps_and_gguf_on_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.hardware as hardware

    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")

    assert hardware.host_backends() == ("mlx", "mps", "gguf")


def test_supported_backends_include_official_basal_gguf() -> None:
    from deqio.catalog import SUPPORTED_BACKENDS, load_catalog

    assert SUPPORTED_BACKENDS == ("mlx", "mps", "gguf", "cuda")
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

    monkeypatch.setattr(installations, "_cached_hf_state", lambda: {"Mapika/decider-0.8b": set()})
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
    monkeypatch.setattr(installations, "_cached_hf_state", lambda: {})
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
    monkeypatch.setattr(installations, "_cached_hf_state", lambda: {"example/model-artifacts": set()})
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


def test_installed_profiles_require_exact_cached_hf_revision(
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
                    "mlx": {
                        "model": "jaredpalmer/kev-0.8b@v1.0",
                        "model_revision": "v1.0",
                        "runtime_key": "kev",
                        "download": {
                            "type": "snapshot",
                            "repo_id": "jaredpalmer/kev-0.8b",
                            "revision": "v1.0",
                        },
                    }
                },
            }
        ]
    }
    state = {"jaredpalmer/kev-0.8b": {"old-tag", "a" * 40}}
    monkeypatch.setattr(installations, "_cached_hf_state", lambda: state)
    monkeypatch.setattr(
        installations,
        "detect_host",
        lambda: HostCapabilities("Darwin", "arm64", ("mlx", "mps"), 32.0, None),
    )
    installations.mark_installed(config_path, "kev-0.8b", "mlx", verified=True)

    row = installations.installed_profiles(
        config_path=config_path,
        config_data={"runtime_dir": ".model-runtimes"},
        catalog=catalog,
    )[0]
    assert row["installed"] is False
    assert row["verified"] is False
    assert row["weights_cached"] is False
    assert row["status"] == "weights-missing"

    state["jaredpalmer/kev-0.8b"].add("v1.0")
    row = installations.installed_profiles(
        config_path=config_path,
        config_data={"runtime_dir": ".model-runtimes"},
        catalog=catalog,
    )[0]
    assert row["installed"] is True
    assert row["verified"] is True
    assert row["weights_cached"] is True
    assert row["status"] == "verified"


def test_installed_profiles_require_model_revision_for_direct_hf_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.installations as installations
    from deqio.hardware import HostCapabilities

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    runtime_python = tmp_path / ".model-runtimes" / "demo" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("")
    revision = "b" * 40
    catalog = {
        "models": [{
            "id": "demo",
            "engine": "demo",
            "label": "Demo",
            "backends": {"mlx": {
                "model": "owner/demo",
                "model_revision": revision,
                "runtime_key": "demo",
            }},
        }]
    }
    state = {"owner/demo": {"a" * 40}}
    monkeypatch.setattr(installations, "_cached_hf_state", lambda: state)
    monkeypatch.setattr(
        installations,
        "detect_host",
        lambda: HostCapabilities("Darwin", "arm64", ("mlx",), 32.0, None),
    )
    installations.mark_installed(config_path, "demo", "mlx", verified=True)

    row = installations.installed_profiles(
        config_path=config_path,
        config_data={"runtime_dir": ".model-runtimes"},
        catalog=catalog,
    )[0]
    assert row["status"] == "weights-missing"

    state["owner/demo"].add(revision)
    row = installations.installed_profiles(
        config_path=config_path,
        config_data={"runtime_dir": ".model-runtimes"},
        catalog=catalog,
    )[0]
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
                "id": "basal-1.5-mini",
                "engine": "basal",
                "label": "Basal 1.5 Mini",
                "backends": {
                    "mlx": {
                        "model": "Remek/basal-1.5-mini-MLX-8bit",
                        "model_revision": "main",
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
            "model_id": "basal-1.5-mini",
            "backend": "mlx",
            "engine": "basal",
            "label": "Basal 1.5 Mini",
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
    assert "Basal 1.5 Mini — mlx (basal)" in output
    assert "Hidden Uninstalled" not in output
    assert "Hidden Incompatible" not in output
    assert "Selected installed profile basal-1.5-mini / mlx" in output

    selected = json.loads(config_path.read_text(encoding="utf-8"))
    assert selected["engine"] == "basal"
    assert selected["model_id"] == "basal-1.5-mini"
    assert selected["backend"] == "mlx"
    assert selected["model"] == "Remek/basal-1.5-mini-MLX-8bit"
    assert selected["model_revision"] == "main"
    assert selected["max_tokens"] == 8192


def test_ui_contains_installed_model_selector_and_live_activate_endpoint() -> None:
    assert 'id="modelSelect"' in DASHBOARD
    assert 'id="activateModel"' in DASHBOARD
    assert "/v1/models/installed" in DASHBOARD
    assert "/v1/models/activate" in DASHBOARD
    assert 'href="/ui/watch"' in DASHBOARD


def test_watch_ui_uses_disk_backed_session_endpoints() -> None:
    assert "Deqio Watch" in WATCH_DASHBOARD
    assert "/v1/watch?limit=" in WATCH_DASHBOARD
    assert "/v1/watch/${encodeURIComponent(eventId)}" in WATCH_DASHBOARD
    assert "/v1/watch/settings" in WATCH_DASHBOARD
    assert "/v1/watch/clear" in WATCH_DASHBOARD
    assert 'id="sourceFilter"' in WATCH_DASHBOARD
    assert 'id="modelFilter"' in WATCH_DASHBOARD
    assert 'id="watchAutoClear"' in WATCH_DASHBOARD
    assert 'href="/ui"' in WATCH_DASHBOARD
    assert 'href="/ui#benchmarkPanel"' not in WATCH_DASHBOARD
    assert 'Benchmark comparison</a>' not in WATCH_DASHBOARD
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

    watch_snapshot = server._watch_store().list_events()
    session, rows = watch_snapshot["session"], watch_snapshot["events"]
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
    reset_snapshot = server._watch_store().list_events()
    reset_session, reset_rows = reset_snapshot["session"], reset_snapshot["events"]
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
    watch_snapshot = server._watch_store().list_events()
    watch_session, watch_rows = watch_snapshot["session"], watch_snapshot["events"]
    assert watch_session["id"] == previous_watch_session
    assert watch_rows == []

def test_release_version_is_consistent() -> None:
    import sys

    from deqio import __version__
    from deqio.server import app

    if sys.version_info >= (3, 11):
        import tomllib
    else:  # Python 3.10: the dev extra installs the API-compatible backport.
        import tomli as tomllib

    project = tomllib.loads(Path("pyproject.toml").read_text())

    assert __version__ == "0.5.7"
    assert project["project"]["version"] == __version__
    assert app.version == __version__


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
    assert suite["label"] == "ENG Bench"
    counts = {kind: 0 for kind in ("noul", "choice", "shared")}
    for case in suite["cases"]:
        counts[case["type"]] += 1
    assert counts == {"noul": 50, "choice": 50, "shared": 50}


def test_benchmark_polish_suite_is_dedicated_and_balanced() -> None:
    from deqio.benchmark import load_suite

    suite = load_suite(Path("benchmarks/pl.json"))
    assert suite["label"] == "PL Bench"
    assert "Polski benchmark" in suite["description"]
    counts = {kind: 0 for kind in ("noul", "choice", "shared")}
    for case in suite["cases"]:
        counts[case["type"]] += 1
    assert counts == {"noul": 20, "choice": 20, "shared": 20}
    encoded = json.dumps(suite, ensure_ascii=False)
    assert "Czy" in encoded
    assert "ż" in encoded or "ł" in encoded


def test_benchmark_discovers_every_valid_workspace_suite(tmp_path: Path) -> None:
    from deqio.benchmark import discover_suites

    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    minimal = {
        "schema_version": 1,
        "name": "custom-suite",
        "label": "Custom Bench",
        "cases": [{"id": "n1", "type": "noul", "state": "x", "question": "y", "expected": "yes"}],
    }
    (benchmarks / "custom.json").write_text(json.dumps(minimal), encoding="utf-8")
    (benchmarks / "broken.json").write_text("{", encoding="utf-8")

    suites = discover_suites(benchmarks)
    assert [(suite["label"], suite["cases"]) for suite in suites] == [("Custom Bench", 1)]


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
    assert 'id="detailRuntime" class="runtime-id"' in WATCH_DASHBOARD
    assert "navigator.clipboard.writeText" in WATCH_DASHBOARD
    assert "text-overflow:ellipsis" in WATCH_DASHBOARD


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
    assert (tmp_path / "benchmarks" / "pl.json").is_file()
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


def test_kev_native_profiles_stay_pinned_and_official_q8_gguf_is_separate() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    for model_id in ("kev-0.8b", "kev-4b", "kev-9b", "kev-27b"):
        entry = next(item for item in catalog["models"] if item["id"] == model_id)
        for backend, profile in entry["backends"].items():
            if backend == "gguf":
                continue
            assert profile["model"].endswith("@v1.0")
            assert profile["model_revision"] == "v1.0"
            assert profile["download"]["revision"] == "v1.0"
            assert profile["packages"] == [
                "kev[serve] @ git+https://github.com/jaredpalmer/kev.git@6b719c3c3f367295f6ef336f4f751cf5ff970abc"
            ]

    for model_id in ("kev-0.8b", "kev-4b", "kev-9b"):
        gguf = get_profile(catalog, model_id, "gguf")
        assert gguf["source"] == "official"
        assert gguf["precision"] == "q8_0"
        assert gguf["launcher"] == "llama_cpp"
        assert gguf["download"]["filename_pattern"] == "*Q8_0.gguf"
    with pytest.raises(RuntimeError, match="does not support backend"):
        get_profile(catalog, "kev-27b", "gguf")

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
    packaged_pl_benchmark = json.loads(
        package_data.joinpath("benchmarks").joinpath("pl.json").read_text(encoding="utf-8")
    )
    repository_pl_benchmark = json.loads(Path("benchmarks/pl.json").read_text(encoding="utf-8"))
    assert packaged_pl_benchmark == repository_pl_benchmark

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
    monkeypatch.setattr(installations, "_cached_hf_state", lambda: {"jaredpalmer/kev-0.8b": set()})
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
        def wait(self, timeout=None):
            self.returncode = 0
            return 0
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
        lambda *a, **k: (
            observed.update(
                command=list(a[0]),
                env=k["env"],
                start_new_session=k.get("start_new_session"),
                stdin=k.get("stdin"),
            )
            or DummyProcess()
        ),
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
    import os

    if os.name != "nt":
        assert observed["start_new_session"] is True
    command = observed["command"]
    assert isinstance(command, list)
    assert Path(command[1]).name == "sidecar_guard.py"
    assert command[2] == "--parent-pid"
    assert command[4:6] == ["--control-stdin", "--engine-pid-file"]
    assert command[6].endswith(".pid")
    assert command[7:9] == ["--", "demo"]
    # close() reaps the published engine PID file together with the supervisor.
    assert not Path(command[6]).exists()
    assert observed["stdin"] is runtime_module.subprocess.PIPE
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


def test_delete_cleanup_never_removes_external_user_model_path(tmp_path: Path) -> None:
    import deqio.model_manager as manager

    config_path = tmp_path / "workspace" / "config.json"
    config_path.parent.mkdir()
    config_path.write_text("{}")
    external = tmp_path / "user-owned-model"
    external.mkdir()
    (external / "weights.bin").write_bytes(b"keep")

    profile = {"model": str(external)}
    manager._cleanup_profile_artifacts(
        config_path=config_path,
        data={"runtime_dir": ".model-runtimes"},
        catalog={"models": []},
        profile=profile,
        remaining_keys=set(),
        purge_cache=False,
    )

    assert external.is_dir()
    assert (external / "weights.bin").read_bytes() == b"keep"


def test_catalog_rejects_runtime_key_path_traversal() -> None:
    from deqio.catalog import get_profile

    catalog = {
        "models": [
            {
                "id": "unsafe",
                "engine": "kev",
                "backends": {
                    "mlx": {
                        "model": "owner/model",
                        "runtime_key": "../../outside",
                    }
                },
            }
        ]
    }

    with pytest.raises(RuntimeError, match="Invalid runtime_key"):
        get_profile(catalog, "unsafe", "mlx")


def test_catalog_rejects_nimble_support_path_traversal() -> None:
    from deqio.catalog import get_profile

    for field, value in (("source_key", "../outside"), ("model_config", "/tmp/outside.json")):
        catalog = {
            "models": [
                {
                    "id": "unsafe-nimble",
                    "engine": "nimble",
                    "backends": {
                        "mlx": {
                            "model": "owner/model",
                            "installer": "nimble",
                            field: value,
                        }
                    },
                }
            ]
        }
        with pytest.raises(RuntimeError, match=f"Invalid {field}"):
            get_profile(catalog, "unsafe-nimble", "mlx")


def test_nimble_install_refuses_destructive_external_model_output(tmp_path: Path) -> None:
    import deqio.model_manager as manager

    config_path = tmp_path / "workspace" / "config.json"
    config_path.parent.mkdir()
    config_path.write_text("{}\n", encoding="utf-8")
    external = tmp_path / "user-owned-model"
    external.mkdir()
    (external / "weights.bin").write_bytes(b"keep")

    with pytest.raises(RuntimeError, match="must be a child of the workspace models directory"):
        manager._install_nimble(
            config_path,
            {"runtime_dir": ".model-runtimes", "backend": "mlx"},
            {
                "installer": "nimble",
                "nimble_backend": "mlx",
                "runtime_key": "nimble-mlx",
                "source_key": "nimble-src",
                "model_config": "nimble-model.json",
                "model": str(external),
            },
            upgrade=True,
        )

    assert (external / "weights.bin").read_bytes() == b"keep"


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
        "clm[serve,hf,vllm] @ git+https://github.com/Contrastive-LM/CLM.git@d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"
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
            "nimble_source_revision": "dcfdbd9a64f0d869f658d7a72f1beaee32737773",
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
    monkeypatch.setattr(installations, "_cached_hf_state", lambda: {})
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
    monkeypatch.setattr(installations, "_cached_hf_state", lambda: {})
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
    openjev = SimpleNamespace(
        engine="open-jev", backend="cuda", model=str(open_model), config_path=tmp_path / "config.json"
    )
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


def test_windows_is_reported_as_unsupported_instead_of_offering_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deqio 0.5 is validated on macOS (Apple Silicon) and Linux only.

    Windows must get an explicit reason from the install preflight rather than
    CUDA/GGUF profiles that were never exercised on that platform.
    """
    import deqio.hardware as hardware

    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "C:/nvidia-smi.exe")
    assert hardware.host_backends() == ()

    windows = hardware.HostCapabilities("Windows", "amd64", ("cuda", "gguf"), 64.0, 24.0)
    for backend in ("cuda", "gguf"):
        verdict = hardware.profile_compatibility(backend, {"min_memory_gib": 1}, host=windows)
        assert verdict["compatible"] is False
        assert "Windows is not supported in Deqio 0.5" in verdict["reason"]

    linux = hardware.HostCapabilities("Linux", "x86_64", ("cuda", "gguf"), 64.0, 24.0)
    assert hardware.profile_compatibility("cuda", {"min_memory_gib": 1}, host=linux)["compatible"] is True


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


def test_benchmark_interactive_suite_selection_lists_workspace_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse
    import deqio.benchmark as benchmark

    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    for filename, label in (("basic.json", "ENG Bench"), ("pl.json", "PL Bench")):
        payload = {
            "schema_version": 1,
            "name": filename.removesuffix(".json"),
            "label": label,
            "cases": [{"id": filename, "type": "noul", "state": "x", "question": "y", "expected": "yes"}],
        }
        (benchmarks / filename).write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda prompt="": "2")
    selected = benchmark._select_suite(tmp_path / "config.json", argparse.Namespace(suite=None))
    assert selected == (benchmarks / "pl.json").resolve()
    output = capsys.readouterr().out
    assert "ENG Bench" in output
    assert "PL Bench" in output


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


def test_external_choice_rejects_missing_or_invalid_probabilities() -> None:
    from deqio.systemone_runtime import _choice_raw

    row = {
        "id": "route",
        "state": "state",
        "question": "Which?",
        "options": [
            {"id": "a", "description": "A"},
            {"id": "b", "description": "B"},
        ],
    }
    common = {"row": row, "response": {}, "payload": {"state": "state", "questions": {}}, "latency_ms": 1.0}

    with pytest.raises(RuntimeError, match="missing option"):
        _choice_raw(answer={"choice": "a", "probabilities": {"a": 1.0}}, **common)
    with pytest.raises(RuntimeError, match="invalid choice probabilities"):
        _choice_raw(answer={"choice": "a", "probabilities": {"a": float("nan"), "b": 0.0}}, **common)
    with pytest.raises(RuntimeError, match="invalid choice probabilities"):
        _choice_raw(answer={"choice": "a", "probabilities": {"a": "not-a-number", "b": 0.0}}, **common)


def test_native_probabilities_reject_nonfinite_or_out_of_range_values() -> None:
    from deqio.systemone_runtime import _choice_raw, _noul_raw

    row = {
        "id": "route",
        "state": "state",
        "question": "Which?",
        "options": [
            {"id": "a", "description": "A"},
            {"id": "b", "description": "B"},
        ],
    }
    with pytest.raises(RuntimeError, match="invalid native choice probabilities"):
        _choice_raw(
            row=row,
            answer={"choice": "a", "probabilities": {"a": 1.2, "b": -0.2}},
            response={},
            payload={"state": "state", "questions": {}},
            latency_ms=1.0,
            preserve_native=True,
        )
    with pytest.raises(RuntimeError, match="invalid Noul probability"):
        _noul_raw(
            row={"id": "n", "state": "state", "question": "True?"},
            answer={"noul": float("nan")},
            response={},
            payload={"state": "state", "questions": {}},
            latency_ms=1.0,
        )
    with pytest.raises(RuntimeError, match="invalid Noul probability"):
        _noul_raw(
            row={"id": "n", "state": "state", "question": "True?"},
            answer={"noul": "not-a-number"},
            response={},
            payload={"state": "state", "questions": {}},
            latency_ms=1.0,
        )


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


def test_legacy_basal10_catalog_profiles_are_removed() -> None:
    from deqio.catalog import load_catalog

    catalog = load_catalog(Path("models.json"))
    ids = {entry["id"] for entry in catalog["models"]}
    assert "basal-1.5b" not in ids
    assert "basal-4.5b" not in ids
    assert {"basal-1.5-mini", "basal-1.5-main", "basal-1.5-max"}.issubset(ids)

def test_basal15_installer_uses_official_backend_specific_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.model_manager as manager

    commands: list[list[str]] = []
    python = tmp_path / ".model-runtimes" / "basal15" / "bin" / "python"
    monkeypatch.setattr(manager, "_ensure_venv", lambda *args, **kwargs: python)
    monkeypatch.setattr(manager, "_run", lambda command, **kwargs: commands.append(list(command)))

    for backend, package in (
        ("mlx", "basal[mlx] @ https://github.com/rkinas/basal/archive/refs/tags/v1.5.0.tar.gz"),
        ("mps", "basal @ https://github.com/rkinas/basal/archive/refs/tags/v1.5.0.tar.gz"),
        ("cuda", "basal @ https://github.com/rkinas/basal/archive/refs/tags/v1.5.0.tar.gz"),
    ):
        commands.clear()
        manager._install_basal(
            tmp_path / "config.json",
            {"backend": backend, "runtime_dir": ".model-runtimes"},
            {
                "runtime_key": "basal15",
                "python": "3.12",
                "basal_backend": backend,
                "basal_runtime_version": "1.5.0",
                "basal_package": package,
            },
            upgrade=False,
        )
        if backend == "cuda":
            assert len(commands) == 2
            assert "torch==2.11.0" in commands[0]
            assert commands[1][-1] == package
        else:
            assert len(commands) == 1
            assert commands[0][-1] == package

def test_basal15_systemone_commands_use_official_server_for_mlx_and_mps(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneRuntime

    env_dir = tmp_path / "runtime"
    executable = env_dir / "bin" / "basal-serve"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    python = env_dir / "bin" / "python"
    python.write_text("", encoding="utf-8")

    for backend, model, mode in (
        ("mlx", "Remek/basal-1.5-mini-MLX-8bit", "mlx"),
        ("mps", "Remek/basal-1.5-mini", "mps"),
    ):
        command = SystemOneRuntime._command(
            SimpleNamespace(engine="basal", backend=backend, model_revision="main", config_path=tmp_path / "config.json"),
            {"model": model, "model_revision": "main", "basal_mode": mode, "basal_soam": True},
            env_dir, python, 9020, {},
        )
        assert command == [
            str(executable), "--model", model, "--mode", mode, "--soam", "on", "--port", "9020"
        ]

def test_patch3_catalog_adds_validated_quantized_profiles_only() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    ids = {str(entry["id"]) for entry in catalog["models"]}
    assert {"clef-flash", "clef", "basal-1.5-main"} <= ids

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

    basal = get_profile(catalog, "basal-1.5-main", "mlx")
    assert basal["model"] == "Remek/basal-1.5-4.5B-MLX-8bit"
    assert basal["runtime_key"] == "basal15-main-mlx"
    assert basal["basal_mode"] == "mlx"
    assert basal["quantization"]["bits"] == 8
    assert basal["min_memory_gib"] == pytest.approx(5.2)
    assert basal["recommended_memory_gib"] == 8

    basal_cuda = get_profile(catalog, "basal-1.5-main", "cuda")
    assert basal_cuda["model"] == "Remek/basal-1.5-4.5B"
    assert basal_cuda["runtime_key"] == "basal15-main-cuda"
    assert basal_cuda["basal_mode"] == "fast-nocompile"
    assert basal_cuda["basal_runtime_version"] == "1.5.0"

    basal_mps = get_profile(catalog, "basal-1.5-main", "mps")
    assert basal_mps["basal_mode"] == "mps"
    assert basal_mps["capabilities"]["evidence"] is True

    for model_id in ("clef-flash", "clef"):
        with pytest.raises(RuntimeError, match="does not support backend"):
            get_profile(catalog, model_id, "cuda")
        with pytest.raises(RuntimeError, match="does not support backend"):
            get_profile(catalog, model_id, "mps")



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


def test_basal15_cuda_installer_leaves_torch_resolution_to_official_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.model_manager as manager

    commands: list[list[str]] = []
    python = tmp_path / ".model-runtimes" / "basal15-main-cuda" / "bin" / "python"
    monkeypatch.setattr(manager, "_ensure_venv", lambda *args, **kwargs: python)
    monkeypatch.setattr(manager, "_run", lambda command, **kwargs: commands.append(list(command)))

    manager._install_basal(
        tmp_path / "config.json",
        {"backend": "cuda", "runtime_dir": ".model-runtimes"},
        {
            "runtime_key": "basal15-main-cuda",
            "python": "3.12",
            "basal_backend": "cuda",
            "basal_runtime_version": "1.5.0",
            "basal_mode": "fast-nocompile",
            "basal_package": "basal @ https://github.com/rkinas/basal/archive/refs/tags/v1.5.0.tar.gz",
        },
        upgrade=False,
    )
    assert len(commands) == 2
    assert "torch==2.11.0" in commands[0]
    assert "https://download.pytorch.org/whl/cu128" in commands[0]
    assert commands[1][-1].endswith("basal/archive/refs/tags/v1.5.0.tar.gz")

def test_basal15_main_cuda_uses_official_fast_nocompile_mode(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneRuntime

    env_dir = tmp_path / "runtime"
    executable = env_dir / "bin" / "basal-serve"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    python = env_dir / "bin" / "python"
    python.write_text("", encoding="utf-8")

    command = SystemOneRuntime._command(
        SimpleNamespace(engine="basal", backend="cuda", model_revision="main", config_path=tmp_path / "config.json"),
        {"model": "Remek/basal-1.5-4.5B", "model_revision": "main", "basal_mode": "fast-nocompile", "basal_soam": True},
        env_dir, python, 9033, {},
    )
    assert command == [
        str(executable), "--model", "Remek/basal-1.5-4.5B",
        "--mode", "fast-nocompile", "--soam", "on", "--port", "9033",
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


def test_patch1_decision2_catalog_is_official_cuda_only() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    expected = {
        "decision-2.0-kai": ("vllm-sr/Decision-2.0-Kai-0.6B", 8192),
        "decision-2.0-eos": ("vllm-sr/Decision-2.0-Eos-0.8B", 16384),
        "decision-2.0-sol": ("vllm-sr/Decision-2.0-Sol-2B", 16384),
        "decision-2.0-nox": ("vllm-sr/Decision-2.0-Nox-4B", 16384),
        "decision-2.0-lux": ("vllm-sr/Decision-2.0-Lux-9B", 16384),
        "decision-2.0-vega": ("vllm-sr/Decision-2.0-Vega-27B", 32768),
    }

    for model_id, (repo_id, max_tokens) in expected.items():
        entry = next(item for item in catalog["models"] if item["id"] == model_id)
        assert entry["engine"] == "decision2"
        assert set(entry["backends"]) == {"cuda"}
        profile = get_profile(catalog, model_id, "cuda")
        assert profile["model"] == repo_id
        assert len(profile["model_revision"]) == 40
        assert profile["family"] == "decision2"
        assert profile["runtime"] == "decision2-native"
        assert profile["platform"] == "cuda"
        assert profile["source"] == "official"
        assert profile["systems"] == ["Linux"]
        assert profile["max_input_tokens"] == max_tokens
        assert profile["capabilities"] == {
            "systemone": True,
            "choice": True,
            "noul": True,
            "score": True,
            "multi_question": True,
            "multi": False,
            "act": False,
            "facts": False,
            "evidence": False,
        }
        for unsupported in ("mlx", "mps"):
            with pytest.raises(RuntimeError):
                get_profile(catalog, model_id, unsupported)

    vega = get_profile(catalog, "decision-2.0-vega", "cuda")
    assert vega["downloads"][1] == {
        "type": "snapshot",
        "repo_id": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "role": "pinned-base",
    }
    kai = get_profile(catalog, "decision-2.0-kai", "cuda")
    assert "flash-linear-attention==0.5.2" not in kai["packages"]
    eos = get_profile(catalog, "decision-2.0-eos", "cuda")
    assert "flash-linear-attention==0.5.2" in eos["packages"]
    assert "causal-conv1d==1.7.0" in eos["packages"]
    assert "peft==0.21.0" not in eos["packages"]
    assert "peft==0.21.0" in vega["packages"]
    assert "huggingface-hub==1.31.0" in vega["packages"]


def test_patch1_vega_attestation_preserves_package_and_pinned_base_roles(tmp_path: Path) -> None:
    import deqio.model_manager as manager
    from deqio.catalog import get_profile, load_catalog

    profile = get_profile(load_catalog(Path("models.json")), "decision-2.0-vega", "cuda")
    artifacts = manager._artifact_attestation(
        tmp_path / "config.json", {"runtime_dir": ".model-runtimes"}, profile
    )

    assert [(item["role"], item["repo_id"], item["resolved_revision"]) for item in artifacts] == [
        (
            "decision2-package",
            "vllm-sr/Decision-2.0-Vega-27B",
            "7aec49ae11a18741706da549ab626b9052795fe7",
        ),
        (
            "pinned-base",
            "Qwen/Qwen3.8-27B",
            "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        ),
    ]


def test_patch1_decision2_installer_pins_official_runtime_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.model_manager as manager

    env = tmp_path / ".model-runtimes" / "decision2-cuda"
    python = env / "bin" / "python"
    commands: list[list[str]] = []
    monkeypatch.setattr(manager, "_ensure_venv", lambda *a, **k: python)
    monkeypatch.setattr(manager, "_run", lambda command: commands.append(list(command)))

    result = manager._install_decision2(
        tmp_path / "config.json",
        {"runtime_dir": ".model-runtimes"},
        {
            "runtime_key": "decision2-cuda",
            "python": "3.12.13",
            "platform": "cuda",
            "packages": [
                "transformers==5.17.0",
                "safetensors==0.8.0",
                "tokenizers==0.23.2",
                "triton==3.7.1",
                "causal-conv1d==1.7.0",
                "flash-linear-attention==0.5.2",
                "huggingface-hub==1.31.0",
                "peft==0.21.0",
            ],
        },
        upgrade=False,
    )

    assert result == env
    assert commands[0][-1] == "torch==2.12.0"
    assert "transformers==5.17.0" in commands[1]
    assert "safetensors==0.8.0" in commands[1]
    assert "flash-linear-attention==0.5.2" in commands[1]
    assert "causal-conv1d==1.7.0" in commands[1]
    assert "peft==0.21.0" in commands[1]
    assert not any("mlx" in part.lower() or "mps" in part.lower() for command in commands for part in command)


def test_patch1_decision2_command_is_transport_to_pinned_official_model(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneRuntime

    python = tmp_path / "python"
    command = SystemOneRuntime._command(
        SimpleNamespace(
            engine="decision2",
            backend="cuda",
            model_revision="a" * 40,
        ),
        {
            "model": "vllm-sr/Decision-2.0-Kai-0.6B",
            "model_revision": "b" * 40,
        },
        tmp_path,
        python,
        9123,
        {},
    )

    assert command[0] == str(python)
    assert command[1].endswith("decision2_sidecar.py")
    assert command[2:] == [
        "--model",
        "vllm-sr/Decision-2.0-Kai-0.6B",
        "--revision",
        "b" * 40,
        "--port",
        "9123",
    ]


def test_patch1_decision2_accelerator_fails_closed_without_cuda_on_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deqio.systemone_runtime import SystemOneRuntime
    from types import SimpleNamespace

    monkeypatch.setattr("deqio.systemone_runtime.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "deqio.systemone_runtime.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    with pytest.raises(RuntimeError, match="CUDA is unavailable") as exc:
        SystemOneRuntime._validate_accelerator(
            SimpleNamespace(engine="decision2", backend="cuda"), Path("/runtime/python")
        )
    assert "No CPU, MPS or MLX fallback" in str(exc.value)


def test_patch1_decision2_accelerator_never_falls_back_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deqio.systemone_runtime import SystemOneRuntime
    from types import SimpleNamespace

    monkeypatch.setattr("deqio.systemone_runtime.platform.system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="CUDA runtime") as exc:
        SystemOneRuntime._validate_accelerator(
            SimpleNamespace(engine="decision2", backend="cuda"), Path("/missing/python")
        )
    message = str(exc.value)
    assert "macOS/Apple Silicon" in message
    assert "No CPU, MPS or MLX fallback" in message


def test_patch1_decision2_choice_probabilities_are_not_renormalized() -> None:
    from deqio.systemone_runtime import _choice_raw

    result = _choice_raw(
        row={
            "id": "q",
            "options": [
                {"id": "a", "description": "A"},
                {"id": "b", "description": "B"},
            ],
        },
        answer={"choice": "a", "probabilities": {"a": 0.51, "b": 0.48}},
        response={"usage": {"input_tokens": 12}},
        payload={"state": "s", "questions": {}},
        latency_ms=1.0,
        preserve_native=True,
    )

    assert result["probabilities"] == [0.51, 0.48]
    assert result["score_provenance"]["normalized"] is False
    assert result["score_provenance"]["transforms"] == []
    assert "preserved unchanged" in result["probability_status"]


def test_patch1_decision2_native_systemone_keeps_mixed_questions_in_one_request() -> None:
    from deqio.systemone_runtime import SystemOneRuntime

    runtime = object.__new__(SystemOneRuntime)
    runtime.engine = "decision2"
    runtime.profile = {
        "capabilities": {
            "choice": True,
            "noul": True,
            "score": True,
            "multi_question": True,
        }
    }
    calls: list[tuple[object, object, object]] = []

    def fake_request(state, questions, execution_mode=None):
        calls.append((state, questions, execution_mode))
        return (
            {"model": "decision2", "state": state, "questions": questions},
            {
                "model": "vllm-sr/Decision-2.0-Kai-0.6B",
                "answers": {
                    "route": {"choice": "support", "probabilities": {"support": 0.8, "sales": 0.2}},
                    "urgent": {"noul": 0.7},
                    "severity": {"score": 2.4, "probabilities": {"1": 0.1, "2": 0.4, "3": 0.5}},
                },
                "usage": {"input_tokens": 42, "output_tokens": 0},
            },
            3.5,
        )

    runtime._request = fake_request
    questions = {
        "route": {"type": "choice", "instructions": "Route", "criteria": {"support": "S", "sales": "P"}},
        "urgent": {"type": "noul", "instructions": "Urgent?"},
        "severity": {"type": "score", "instructions": "Severity", "criteria": ["low", "mid", "high"]},
    }
    response, timing = runtime.system_one("state", questions)

    assert len(calls) == 1
    assert calls[0] == ("state", questions, None)
    assert response["answers"]["urgent"]["noul"] == 0.7
    assert response["answers"]["severity"]["score"] == 2.4
    assert timing["batch_size"] == 3


def test_patch1_decision2_systemone_rejects_non_decision2_capability() -> None:
    from deqio.systemone_runtime import SystemOneRuntime

    runtime = object.__new__(SystemOneRuntime)
    runtime.engine = "decision2"
    runtime.profile = {"capabilities": {"choice": True, "noul": True, "score": True, "multi_question": True}}
    with pytest.raises(RuntimeError, match="does not support question type"):
        runtime.system_one(
            "state",
            {"labels": {"type": "multi", "instructions": "Select all"}},
        )


def test_patch1_systemone_api_preserves_native_response_and_adds_deqio_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deqio.server as server

    class FakeRuntime:
        def identity_snapshot(self):
            return {
                "runtime_instance_id": "decision2-runtime",
                "engine": "decision2",
                "model_id": "decision-2.0-kai",
                "backend": "cuda",
                "requested_revision": "d" * 40,
                "artifact_revisions_resolved": True,
            }

        def system_one(self, state, questions):
            return (
                {
                    "model": "vllm-sr/Decision-2.0-Kai-0.6B",
                    "answers": {
                        "pick": {
                            "choice": "a",
                            "probabilities": {"a": 0.51, "b": 0.48},
                        }
                    },
                    "usage": {"input_tokens": 9, "output_tokens": 0},
                },
                {"total_seconds": 0.002, "batch_size": 1, "engine": "decision2"},
            )

    monkeypatch.setattr(server, "runtime", FakeRuntime())
    monkeypatch.setattr(server, "record_event", lambda **kwargs: None)
    monkeypatch.setattr(server, "_record_watch_event", lambda **kwargs: None)
    monkeypatch.setattr(server, "log_request_error", lambda *args, **kwargs: None)

    result = asyncio.run(server.system_one(server.SystemOneRequest(
        state="state",
        questions={
            "pick": {
                "type": "choice",
                "instructions": "Pick",
                "criteria": {"a": "A", "b": "B"},
            }
        },
    )))

    assert result["model"] == "vllm-sr/Decision-2.0-Kai-0.6B"
    assert result["answers"]["pick"]["probabilities"] == {"a": 0.51, "b": 0.48}
    assert result["usage"] == {"input_tokens": 9, "output_tokens": 0}
    assert result["deqio"]["runtime"]["runtime_instance_id"] == "decision2-runtime"
    assert result["deqio"]["timing"]["total_ms"] == 2.0



def test_patch2_basal15_catalog_has_full_official_family_and_capabilities() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    expected = {
        "basal-1.5-mini": {
            "mlx": "Remek/basal-1.5-mini-MLX-8bit",
            "mps": "Remek/basal-1.5-mini",
            "gguf": "Remek/basal-1.5-mini",
            "cuda": "Remek/basal-1.5-mini",
        },
        "basal-1.5-main": {
            "mlx": "Remek/basal-1.5-4.5B-MLX-8bit",
            "mps": "Remek/basal-1.5-4.5B",
            "gguf": "Remek/basal-1.5-4.5B",
            "cuda": "Remek/basal-1.5-4.5B",
        },
        "basal-1.5-max": {
            "mlx": "Remek/basal-1.5-max-MLX-8bit",
            "mps": "Remek/basal-1.5-max",
            "gguf": "Remek/basal-1.5-max",
            "cuda": "Remek/basal-1.5-max",
        },
    }
    for model_id, backends in expected.items():
        entry = next(item for item in catalog["models"] if item["id"] == model_id)
        assert entry["engine"] == "basal"
        assert set(entry["backends"]) == {"mlx", "mps", "gguf", "cuda"}
        for backend, repo in backends.items():
            profile = get_profile(catalog, model_id, backend)
            assert profile["family"] == "basal1.5"
            assert profile["source"] == "official"
            assert profile["model"] == repo
            assert profile["basal_runtime_version"] == "1.5.0"
            assert profile["pin_hf_revisions_at_install"] is True
            caps = profile["capabilities"]
            for capability in ("choice", "noul", "score", "multi", "act", "facts", "soam", "option_keys"):
                assert caps[capability] is True
            assert caps["evidence"] is (backend in {"mps", "cuda"})


def test_patch2_basal15_mlx_and_gguf_use_quality_profiles() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    for model_id in ("basal-1.5-mini", "basal-1.5-main", "basal-1.5-max"):
        mlx = get_profile(catalog, model_id, "mlx")
        gguf = get_profile(catalog, model_id, "gguf")
        assert mlx["precision"] == "mlx-8bit"
        assert mlx["quantization"]["bits"] == 8
        assert mlx["basal_mode"] == "mlx"
        assert "basal[mlx]" in mlx["basal_package"]
        assert gguf["precision"] == "q8_0"
        assert gguf["quantization"]["quantization"] == "Q8_0"
        assert gguf["basal_mode"] == "gguf"
        assert gguf["basal_gguf_pattern"] == "*Q8_0.gguf"
        q8_download = next(item for item in gguf["downloads"] if item.get("role") == "basal-gguf-q8_0")
        assert q8_download["filename_pattern"] == "*Q8_0.gguf"
        assert "filename" not in q8_download
        assert "basal[gguf]" in gguf["basal_package"]


def test_patch2_basal15_installer_uses_official_v150_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.model_manager as manager

    commands: list[list[str]] = []
    python = tmp_path / ".model-runtimes" / "basal15" / "bin" / "python"
    monkeypatch.setattr(manager, "_ensure_venv", lambda *args, **kwargs: python)
    monkeypatch.setattr(manager, "_run", lambda command: commands.append(list(command)))

    manager._install_basal(
        tmp_path / "config.json",
        {"backend": "mlx", "runtime_dir": ".model-runtimes"},
        {
            "runtime_key": "basal15",
            "basal_backend": "mlx",
            "basal_runtime_version": "1.5.0",
            "basal_package": "basal[mlx] @ https://github.com/rkinas/basal/archive/refs/tags/v1.5.0.tar.gz",
        },
        upgrade=False,
    )
    assert len(commands) == 1
    assert commands[0][-1].endswith("basal/archive/refs/tags/v1.5.0.tar.gz")

    commands.clear()
    manager._install_basal(
        tmp_path / "config.json",
        {"backend": "gguf", "runtime_dir": ".model-runtimes"},
        {
            "runtime_key": "basal15",
            "basal_backend": "gguf",
            "basal_runtime_version": "1.5.0",
            "basal_package": "basal[gguf] @ https://github.com/rkinas/basal/archive/refs/tags/v1.5.0.tar.gz",
        },
        upgrade=False,
    )
    assert len(commands) == 1
    assert "basal[gguf]" in commands[0][-1]


def test_patch2_basal15_command_preserves_official_mlx_soam_and_revision(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneRuntime

    env_dir = tmp_path / "runtime"
    executable = env_dir / "bin" / "basal-serve"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    python = env_dir / "bin" / "python"
    python.write_text("", encoding="utf-8")
    revision = "a" * 40
    command = SystemOneRuntime._command(
        SimpleNamespace(engine="basal", backend="mlx", model_revision=revision, config_path=tmp_path / "config.json"),
        {
            "model": "Remek/basal-1.5-4.5B-MLX-8bit",
            "model_revision": revision,
            "basal_mode": "mlx",
            "basal_soam": True,
        },
        env_dir, python, 9441, {},
    )
    assert command == [
        str(executable),
        "--model", "Remek/basal-1.5-4.5B-MLX-8bit",
        "--revision", revision,
        "--mode", "mlx",
        "--soam", "on",
        "--port", "9441",
    ]


def test_patch2_basal15_gguf_command_uses_local_q8_and_official_metadata(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneRuntime

    env_dir = tmp_path / "runtime"
    executable = env_dir / "bin" / "basal-serve"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    python = env_dir / "bin" / "python"
    python.write_text("", encoding="utf-8")
    gguf = tmp_path / "models" / "main-Q8_0.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_text("weights", encoding="utf-8")
    revision = "b" * 40
    command = SystemOneRuntime._command(
        SimpleNamespace(engine="basal", backend="gguf", model_revision=revision, config_path=tmp_path / "config.json"),
        {
            "model": "Remek/basal-1.5-4.5B",
            "model_revision": revision,
            "basal_mode": "gguf",
            "basal_soam": True,
            "basal_gguf_dir": "models",
            "basal_gguf_pattern": "*Q8_0.gguf",
        },
        env_dir, python, 9442, {},
    )
    assert "--gguf" in command
    assert command[command.index("--gguf") + 1] == str(gguf.resolve())
    assert command[command.index("--revision") + 1] == revision
    assert command[command.index("--mode") + 1] == "gguf"


def test_patch2_basal15_systemone_preserves_mixed_native_response_and_one_request() -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneRuntime

    runtime = object.__new__(SystemOneRuntime)
    runtime.engine = "basal"
    runtime.settings = SimpleNamespace(backend="mlx")
    runtime.profile = {
        "family": "basal1.5",
        "capabilities": {
            "choice": True, "noul": True, "score": True, "multi": True, "act": True,
            "facts": True, "soam": True, "option_keys": True, "evidence": False,
            "early_exit": False,
        },
    }
    calls: list[tuple[object, object, object]] = []
    native = {
        "model": "Remek/basal-1.5-4.5B-MLX-8bit",
        "answers": {
            "route": {"choice": "support", "probabilities": {"support": 0.61, "sales": 0.39}},
            "risk": {"noul": 0.73},
            "severity": {"score": 2.2, "probabilities": {"low": 0.1, "mid": 0.6, "high": 0.3}},
            "labels": {"probabilities": {"a": 0.8, "b": 0.7}, "selected": ["a", "b"], "threshold": 0.5, "set_confidence": 0.74},
            "action": {"action": "defer", "expected_costs": {"approve": 3.1, "defer": 1.2}},
        },
        "facts": {"vat": 23},
        "usage": {"input_tokens": 77},
    }

    def fake_request(state, questions, execution_mode=None, request_options=None):
        calls.append((state, questions, request_options))
        return ({"state": state, "questions": questions, **(request_options or {})}, native, 4.0)

    runtime._request = fake_request
    questions = {
        "route": {"type": "choice", "instructions": "Route", "criteria": {"support": "S", "sales": "P"}, "option_keys": "show"},
        "risk": {"type": "noul", "instructions": "Risk?"},
        "severity": {"type": "score", "instructions": "Severity", "criteria": ["low", "mid", "high"]},
        "labels": {"type": "multi", "instructions": "Labels", "criteria": {"a": "A", "b": "B"}, "threshold": 0.5},
        "action": {
            "type": "act",
            "instructions": "Act",
            "criteria": {"true": "Approve", "false": "Reject"},
            "costs": {
                "approve": {"true": 0, "false": 5},
                "defer": {"true": 1, "false": 1},
            },
        },
    }
    response, timing = runtime.system_one("state", questions, {"facts": "auto"})

    assert len(calls) == 1
    assert calls[0] == ("state", questions, {"facts": "auto"})
    assert response is native
    assert response["answers"]["labels"]["probabilities"] == {"a": 0.8, "b": 0.7}
    assert sum(response["answers"]["labels"]["probabilities"].values()) == pytest.approx(1.5)
    assert response["answers"]["action"]["expected_costs"] == {"approve": 3.1, "defer": 1.2}
    assert timing["batch_size"] == 5


def test_patch2_basal15_evidence_is_capability_aware() -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneCapabilityError, SystemOneRuntime

    runtime = object.__new__(SystemOneRuntime)
    runtime.engine = "basal"
    runtime.settings = SimpleNamespace(backend="mlx")
    runtime.profile = {
        "family": "basal1.5",
        "capabilities": {"choice": True, "soam": True, "evidence": False},
    }
    with pytest.raises(SystemOneCapabilityError, match="Evidence is not available") as exc:
        runtime.system_one(
            "The receipt is valid.",
            {"proof": {"type": "choice", "instructions": "Valid?", "criteria": {"yes": "Yes", "no": "No"}, "evidence": True}},
        )
    assert exc.value.capability == "evidence"

    runtime.settings = SimpleNamespace(backend="mps")
    runtime.profile["capabilities"]["evidence"] = True
    native = {
        "answers": {
            "proof": {
                "choice": "yes",
                "probabilities": {"yes": 0.9, "no": 0.1},
                "evidence": [{"text": "receipt", "start": 4, "end": 11, "probability": 0.88}],
            }
        }
    }
    runtime._request = lambda state, questions, execution_mode=None, request_options=None: ({}, native, 1.0)
    response, _timing = runtime.system_one(
        "The receipt is valid.",
        {"proof": {"type": "choice", "instructions": "Valid?", "criteria": {"yes": "Yes", "no": "No"}, "evidence": True}},
    )
    span = response["answers"]["proof"]["evidence"][0]
    assert "The receipt is valid."[span["start"]:span["end"]] == span["text"]


def test_patch2_basal15_rejects_evidence_for_multi_even_on_native_backend() -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneCapabilityError, SystemOneRuntime

    runtime = object.__new__(SystemOneRuntime)
    runtime.engine = "basal"
    runtime.settings = SimpleNamespace(backend="mps")
    runtime.profile = {
        "family": "basal1.5",
        "capabilities": {"multi": True, "soam": True, "evidence": True},
    }
    with pytest.raises(SystemOneCapabilityError, match="not supported for multi"):
        runtime.system_one(
            "state",
            {"labels": {"type": "multi", "instructions": "Labels", "criteria": {"a": "A"}, "evidence": True}},
        )


def test_patch2_basal15_api_passes_facts_auto_and_preserves_native_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deqio.server as server

    calls: list[tuple[object, object, object]] = []

    class FakeRuntime:
        def identity_snapshot(self):
            return {"runtime_instance_id": "basal15-runtime", "engine": "basal", "backend": "mlx"}

        def system_one(self, state, questions, options=None):
            calls.append((state, questions, options))
            return (
                {
                    "answers": {"labels": {"probabilities": {"a": 0.8, "b": 0.7}, "selected": ["a", "b"]}},
                    "facts": {"vat": 23},
                    "calibration": {"source": "basal"},
                },
                {"total_seconds": 0.003, "batch_size": 1, "engine": "basal"},
            )

    monkeypatch.setattr(server, "runtime", FakeRuntime())
    monkeypatch.setattr(server, "record_event", lambda **kwargs: None)
    monkeypatch.setattr(server, "_record_watch_event", lambda **kwargs: None)
    monkeypatch.setattr(server, "log_request_error", lambda *args, **kwargs: None)

    result = asyncio.run(server.system_one(server.SystemOneRequest(
        state="VAT 23%",
        facts="auto",
        questions={"labels": {"type": "multi", "instructions": "Labels", "criteria": {"a": "A", "b": "B"}}},
    )))
    assert calls[0][2] == {"facts": "auto"}
    assert result["answers"]["labels"]["probabilities"] == {"a": 0.8, "b": 0.7}
    assert result["facts"] == {"vat": 23}
    assert result["calibration"] == {"source": "basal"}
    assert result["deqio"]["runtime"]["runtime_instance_id"] == "basal15-runtime"


def test_patch2_basal15_api_returns_422_for_unsupported_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deqio.server as server
    from deqio.systemone_runtime import SystemOneCapabilityError

    class FakeRuntime:
        def identity_snapshot(self):
            return {"runtime_instance_id": "basal15-runtime", "engine": "basal", "backend": "mlx"}

        def system_one(self, state, questions, options=None):
            raise SystemOneCapabilityError("evidence", "Evidence is not available on the active Basal mlx backend")

    monkeypatch.setattr(server, "runtime", FakeRuntime())
    monkeypatch.setattr(server, "record_event", lambda **kwargs: None)
    monkeypatch.setattr(server, "_record_watch_event", lambda **kwargs: None)
    monkeypatch.setattr(server, "log_request_error", lambda *args, **kwargs: None)

    with pytest.raises(Exception) as exc:
        asyncio.run(server.system_one(server.SystemOneRequest(
            state="state",
            questions={"proof": {"type": "choice", "evidence": True}},
        )))
    assert getattr(exc.value, "status_code", None) == 422
    assert "Evidence is not available" in str(getattr(exc.value, "detail", ""))


def test_patch2_basal15_profile_pin_prefers_selected_exact_revision_before_install_record() -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import _profile_with_installed_pin

    revision = "c" * 40
    settings = SimpleNamespace(
        model_revision=revision,
        config_path=Path("/tmp/config.json"),
        model_id="basal-1.5-main",
        backend="mlx",
    )
    profile = {
        "model": "Remek/basal-1.5-4.5B-MLX-8bit",
        "model_revision": "main",
        "pin_hf_revisions_at_install": True,
    }
    pinned = _profile_with_installed_pin(settings, profile)
    assert pinned["model_revision"] == revision
    assert profile["model_revision"] == "main"


def test_patch2_basal15_runtime_metadata_requires_official_v150(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneRuntime

    monkeypatch.setattr(
        "deqio.systemone_runtime.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout='{"runtime_version":"1.5.0"}'),
    )
    metadata = SystemOneRuntime._basal_runtime_metadata(
        {"basal_runtime_version": "1.5.0", "basal_mode": "mlx", "basal_backend": "mlx"},
        tmp_path / "python",
    )
    assert metadata == {
        "runtime_version": "1.5.0",
        "basal_mode": "mlx",
        "basal_backend": "mlx",
        "official_runtime": True,
    }


def test_patch2_systemone_request_rejects_unknown_top_level_options() -> None:
    from pydantic import ValidationError
    import deqio.server as server

    with pytest.raises(ValidationError):
        server.SystemOneRequest(
            state="state",
            questions={"q": {"type": "noul", "instructions": "Q?"}},
            silently_ignored_feature=True,
        )


def test_patch2_basal15_pin_resolves_exact_q8_artifact_without_filename_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    import deqio.model_manager as manager

    class FakeApi:
        def model_info(self, repo_id, revision=None):
            suffix = "1" if repo_id.endswith("4.5B") else "2"
            return SimpleNamespace(sha=suffix * 40)

        def list_repo_files(self, repo_id, revision=None):
            assert revision == "2" * 40
            return ["README.md", "official-basal-main-Q8_0.gguf", "official-basal-main-Q4_K_M.gguf"]

    monkeypatch.setattr(manager, "HfApi", lambda: FakeApi())
    profile = {
        "pin_hf_revisions_at_install": True,
        "model": "Remek/basal-1.5-4.5B",
        "model_revision": "main",
        "downloads": [
            {
                "type": "snapshot",
                "repo_id": "Remek/basal-1.5-4.5B",
                "revision": "main",
                "role": "basal-metadata",
            },
            {
                "type": "file",
                "repo_id": "Remek/basal-1.5-4.5B-GGUF",
                "revision": "main",
                "role": "basal-gguf-q8_0",
                "filename_pattern": "*Q8_0.gguf",
                "local_dir": "models/basal-1.5-4.5B-GGUF",
            },
        ],
    }
    pinned = manager._pin_profile_hf_revisions(profile)
    assert pinned["model_revision"] == "1" * 40
    assert pinned["downloads"][0]["revision"] == "1" * 40
    assert pinned["downloads"][1]["revision"] == "2" * 40
    assert pinned["downloads"][1]["filename"] == "official-basal-main-Q8_0.gguf"
    assert "filename" not in profile["downloads"][1]


def test_patch4_adds_native_convenience_endpoints_without_inventing_modifier_endpoints() -> None:
    routes = {route.path for route in app.routes}
    for path in ("/v1/systemone", "/v1/soam", "/v1/score", "/v1/multi", "/v1/act"):
        assert path in routes
    # Facts, evidence and option_keys remain native request/question modifiers.
    assert "/v1/evidence" not in routes
    assert "/v1/facts" not in routes
    assert "/v1/option-key" not in routes
    assert "/v1/optionKey" not in routes


def test_fix01_installed_profiles_use_attested_resolved_revision_for_pin_on_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.installations as installations
    from deqio.hardware import HostCapabilities

    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    runtime_python = tmp_path / ".model-runtimes" / "basal15-mini-mlx" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("", encoding="utf-8")
    resolved = "6ea29c486e39e1a217932ab56b222c0d1bc4ba82"
    catalog = {
        "models": [{
            "id": "basal-1.5-mini",
            "engine": "basal",
            "label": "Basal 1.5 Mini — 1.5B",
            "backends": {"mlx": {
                "model": "Remek/basal-1.5-mini-MLX-8bit",
                "model_revision": "main",
                "runtime_key": "basal15-mini-mlx",
                "family": "basal1.5",
                "runtime": "basal-1.5-mlx",
                "source": "official",
                "precision": "mlx-8bit",
                "capabilities": {"systemone": True, "choice": True},
                "download": {
                    "type": "snapshot",
                    "repo_id": "Remek/basal-1.5-mini-MLX-8bit",
                    "revision": "main",
                    "role": "basal-model",
                },
            }},
        }]
    }
    monkeypatch.setattr(
        installations,
        "_cached_hf_state",
        lambda: {"Remek/basal-1.5-mini-MLX-8bit": {resolved}},
    )
    monkeypatch.setattr(
        installations,
        "detect_host",
        lambda: HostCapabilities("Darwin", "arm64", ("mlx", "mps", "gguf"), 16.0, None),
    )
    installations.mark_installed(
        config_path,
        "basal-1.5-mini",
        "mlx",
        verified=True,
        artifacts=[{
            "source": "huggingface",
            "repo_id": "Remek/basal-1.5-mini-MLX-8bit",
            "requested_revision": resolved,
            "resolved_revision": resolved,
            "role": "basal-model",
        }],
        max_input_tokens=8192,
    )

    row = installations.installed_profiles(
        config_path=config_path,
        config_data={"runtime_dir": ".model-runtimes"},
        catalog=catalog,
        active_model_id="basal-1.5-mini",
        active_backend="mlx",
    )[0]
    assert row["installed"] is True
    assert row["verified"] is True
    assert row["status"] == "active"
    assert row["capabilities"] == {"systemone": True, "choice": True}
    assert row["precision"] == "mlx-8bit"
    assert row["source"] == "official"


def test_patch4_ui_exposes_native_endpoints_and_capability_matrix() -> None:
    assert 'id="systemOneTab"' in DASHBOARD
    assert 'SOAM · /v1/soam' in DASHBOARD
    assert 'data-endpoint="score"' in DASHBOARD
    assert 'data-endpoint="multi"' in DASHBOARD
    assert 'data-endpoint="act"' in DASHBOARD
    assert 'raw /v1/systemone' in DASHBOARD
    assert 'id="systemOneQuestions"' in DASHBOARD
    assert 'id="factsMode"' in DASHBOARD
    assert 'id="nativeQuestionInput"' in DASHBOARD
    for label in ("Multi", "Act", "Evidence modifier", "Facts modifier", "SOAM", "Option keys modifier"):
        assert label in DASHBOARD
    assert "refreshSystemOneCapabilities" in DASHBOARD
    assert "activeProfile()?.capabilities" in DASHBOARD


def test_fix01_generic_systemone_profiles_are_exposed_without_schema_rewrite() -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneRuntime

    runtime = object.__new__(SystemOneRuntime)
    runtime.engine = "kev"
    runtime.settings = SimpleNamespace(backend="gguf")
    runtime.profile = {
        "capabilities": {
            "systemone": True,
            "choice": True,
            "noul": True,
            "score": True,
            "multi_question": True,
            "multi": False,
            "act": False,
            "facts": False,
            "evidence": False,
        }
    }
    native = {
        "answers": {
            "route": {"type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.3}},
            "risk": {"type": "noul", "noul": 0.4},
        },
        "usage": {"input_tokens": 33, "output_tokens": 0},
    }
    calls = []
    runtime._request = lambda state, questions, execution_mode=None, request_options=None: (
        calls.append((state, questions, request_options)) or {}, native, 3.0
    )
    questions = {
        "route": {"type": "choice", "instructions": "Route", "criteria": {"a": "A", "b": "B"}},
        "risk": {"type": "noul", "instructions": "Risk?"},
    }
    result, timing = runtime.system_one("state", questions)
    assert result is native
    assert calls == [("state", questions, None)]
    assert timing["batch_size"] == 2


def test_fix01_catalog_adds_only_audited_official_q8_gguf_profiles() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    expected = {
        "kev-0.8b": "ggml-org/Kev-0.8B-GGUF",
        "kev-4b": "ggml-org/Kev-4B-GGUF",
        "kev-9b": "ggml-org/Kev-9B-GGUF",
        "jevk5-4b": "alibiserikbay/JevK5-GGUF",
        "jevk5-9b": "alibiserikbay/JevK5-GGUF",
        "decider-2b": "Mapika/decider-2b-GGUF",
        "decider-4b": "Mapika/decider-4b-GGUF",
        "laya-english": "ggml-org/Laya-GGUF",
        "clef-flash": "ggml-org/Clef-Flash-GGUF",
        "clef": "ggml-org/Clef-GGUF",
    }
    for model_id, repo_id in expected.items():
        profile = get_profile(catalog, model_id, "gguf")
        assert profile["model"] == repo_id
        assert profile["source"] == "official"
        assert profile["precision"] == "q8_0"
        assert profile["quantization"]["quantization"] == "Q8_0"
        assert profile["capabilities"]["systemone"] is True
        downloads = profile.get("downloads") or [profile.get("download")]
        q8 = [item for item in downloads if isinstance(item, dict) and "q8" in str(item.get("role", "")).lower()]
        assert q8
        artifact = q8[-1]
        assert artifact.get("filename", "").endswith("Q8_0.gguf") or artifact.get("filename_pattern") == "*Q8_0.gguf"

    # Deliberately absent: no exact official Q8 integration for these current Deqio profiles.
    for model_id in ("kev-27b", "decider-0.8b", "laya-multilingual", "laya-typed-decisions", "nimble-9b"):
        with pytest.raises(RuntimeError, match="does not support backend"):
            get_profile(catalog, model_id, "gguf")
    for model_id in (
        "decision-2.0-kai", "decision-2.0-eos", "decision-2.0-sol",
        "decision-2.0-nox", "decision-2.0-lux", "decision-2.0-vega",
    ):
        with pytest.raises(RuntimeError, match="does not support backend"):
            get_profile(catalog, model_id, "gguf")


def test_fix01_official_gguf_launchers_use_local_q8_and_native_readout(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneRuntime

    config_path = tmp_path / "config.json"
    runtime = tmp_path / "runtime"
    python = runtime / "bin" / "python"
    llama_server = runtime / "bin" / "llama-server"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    llama_server.write_text("", encoding="utf-8")

    kev_dir = tmp_path / "models" / "kev"
    kev_dir.mkdir(parents=True)
    kev = kev_dir / "Kev-4B-Q8_0.gguf"
    kev.write_text("weights", encoding="utf-8")
    command = SystemOneRuntime._command(
        SimpleNamespace(engine="kev", backend="gguf", max_tokens=8192, config_path=config_path),
        {
            "model": "ggml-org/Kev-4B-GGUF", "launcher": "llama_cpp",
            "llama_cpp_gguf_dir": "models/kev", "llama_cpp_gguf_pattern": "*Q8_0.gguf",
        },
        runtime, python, 9001, {},
    )
    assert command[0] == str(llama_server)
    assert command[command.index("-m") + 1] == str(kev.resolve())

    decider_dir = tmp_path / "models" / "decider"
    decider_dir.mkdir(parents=True)
    decider = decider_dir / "decider-2b-v11-Q8_0.gguf"
    decider.write_text("weights", encoding="utf-8")
    command = SystemOneRuntime._command(
        SimpleNamespace(engine="decider", backend="gguf", max_tokens=8192, config_path=config_path),
        {
            "model": "Mapika/decider-2b-GGUF", "launcher": "decider_gguf",
            "decider_gguf_dir": "models/decider", "decider_gguf_pattern": "*Q8_0.gguf",
        },
        runtime, python, 9002, {},
    )
    assert Path(command[1]).name == "decider_gguf_sidecar.py"
    assert command[command.index("--gguf") + 1] == str(decider.resolve())

    jev_dir = tmp_path / "models" / "jev"
    jev_dir.mkdir(parents=True)
    jev = jev_dir / "jevk5-9b-v0.3.3-Q8_0.gguf"
    jev.write_text("weights", encoding="utf-8")
    command = SystemOneRuntime._command(
        SimpleNamespace(engine="jevk5", backend="gguf", max_tokens=8192, config_path=config_path),
        {
            "model": "alibiserikbay/JevK5-GGUF", "launcher": "jevk5_gguf",
            "jevk5_gguf_dir": "models/jev", "jevk5_gguf_pattern": "*Q8_0.gguf",
            "jevk5_temperature": 1.316, "jevk5_knockout_temperature": 1.05,
        },
        runtime, python, 9003, {},
    )
    assert Path(command[1]).name == "jevk5_gguf_sidecar.py"
    assert command[command.index("--gguf") + 1] == str(jev.resolve())
    assert command[command.index("--temperature") + 1] == "1.316"


@pytest.mark.parametrize("model_id", ["basal-1.5-mini", "basal-1.5-main", "basal-1.5-max"])
def test_fix01_basal15_native_mps_family_preserves_exact_evidence_spans(model_id: str, tmp_path: Path) -> None:
    from types import SimpleNamespace
    from deqio.catalog import get_profile, load_catalog
    from deqio.systemone_runtime import SystemOneRuntime

    profile = get_profile(load_catalog(Path("models.json")), model_id, "mps")
    assert profile["basal_mode"] == "mps"
    assert profile["capabilities"]["evidence"] is True

    env_dir = tmp_path / "runtime"
    executable = env_dir / "bin" / "basal-serve"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    python = env_dir / "bin" / "python"
    python.write_text("", encoding="utf-8")
    command = SystemOneRuntime._command(
        SimpleNamespace(engine="basal", backend="mps", model_revision="main", config_path=tmp_path / "config.json"),
        profile,
        env_dir, python, 9330, {},
    )
    assert command[command.index("--mode") + 1] == "mps"

    runtime = object.__new__(SystemOneRuntime)
    runtime.engine = "basal"
    runtime.settings = SimpleNamespace(backend="mps")
    runtime.profile = profile
    state = "The damaged laptop arrived yesterday."
    start = state.index("damaged laptop")
    text = "damaged laptop"
    end = start + len(text)
    native = {
        "answers": {
            "proof": {
                "type": "noul",
                "noul": 0.93,
                "probabilities": {"true": 0.93, "false": 0.07},
                "evidence": [{"text": text, "start": start, "end": end, "probability": 0.91}],
            }
        }
    }
    runtime._request = lambda state, questions, execution_mode=None, request_options=None: ({}, native, 1.0)
    response, _ = runtime.system_one(
        state,
        {"proof": {"type": "noul", "instructions": "Was it damaged?", "evidence": True}},
    )
    span = response["answers"]["proof"]["evidence"][0]
    assert state[span["start"]:span["end"]] == span["text"]


def test_fix01_von_and_clm_expose_native_systemone_capabilities() -> None:
    from deqio.catalog import get_profile, load_catalog

    catalog = load_catalog(Path("models.json"))
    for model_id, backend in (("von", "mps"), ("von", "cuda"), ("clm-8b", "cuda")):
        capabilities = get_profile(catalog, model_id, backend)["capabilities"]
        assert capabilities == {
            "systemone": True,
            "choice": True,
            "noul": True,
            "score": True,
            "multi_question": True,
            "multi": False,
            "act": False,
            "facts": False,
            "evidence": False,
        }


def test_fix01_shared_local_gguf_directory_requires_exact_profile_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.installations as installations

    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    shared = tmp_path / "models" / "jevk5-GGUF"
    shared.mkdir(parents=True)
    (shared / "jevk5-4b-v0.3-Q8_0.gguf").write_text("4b", encoding="utf-8")

    profile_4b = {
        "model": "alibiserikbay/JevK5-GGUF",
        "download": {
            "type": "file",
            "repo_id": "alibiserikbay/JevK5-GGUF",
            "local_dir": "models/jevk5-GGUF",
            "filename": "jevk5-4b-v0.3-Q8_0.gguf",
        },
    }
    profile_9b = {
        "model": "alibiserikbay/JevK5-GGUF",
        "download": {
            "type": "file",
            "repo_id": "alibiserikbay/JevK5-GGUF",
            "local_dir": "models/jevk5-GGUF",
            "filename": "jevk5-9b-v0.3.3-Q8_0.gguf",
        },
    }
    monkeypatch.setattr(installations, "_cached_hf_state", lambda: {})
    ready_4b, _ = installations._artifact_ready(config_path, profile_4b, {})
    ready_9b, _ = installations._artifact_ready(config_path, profile_9b, {})
    assert ready_4b is True
    assert ready_9b is False


def test_fix01_jevk5_gguf_launcher_selects_exact_file_when_both_sizes_are_installed(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from deqio.systemone_runtime import SystemOneRuntime

    runtime = tmp_path / "runtime"
    python = runtime / "bin" / "python"
    llama_server = runtime / "bin" / "llama-server"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    llama_server.write_text("", encoding="utf-8")
    directory = tmp_path / "models" / "jevk5-GGUF"
    directory.mkdir(parents=True)
    file4 = directory / "jevk5-4b-v0.3-Q8_0.gguf"
    file9 = directory / "jevk5-9b-v0.3.3-Q8_0.gguf"
    file4.write_text("4b", encoding="utf-8")
    file9.write_text("9b", encoding="utf-8")

    profile = {
        "model": "alibiserikbay/JevK5-GGUF",
        "launcher": "jevk5_gguf",
        "jevk5_gguf_dir": "models/jevk5-GGUF",
        "jevk5_gguf_filename": file9.name,
        "jevk5_temperature": 1.316,
        "jevk5_knockout_temperature": 1.05,
    }
    command = SystemOneRuntime._command(
        SimpleNamespace(engine="jevk5", backend="gguf", max_tokens=8192, config_path=tmp_path / "config.json"),
        profile, runtime, python, 9003, {},
    )
    assert command[command.index("--gguf") + 1] == str(file9.resolve())


def test_fix02_pin_on_install_accepts_non_basal_artifact_roles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from types import SimpleNamespace
    import deqio.systemone_runtime as runtime_module

    revision = "d" * 40
    monkeypatch.setattr(
        runtime_module,
        "installation_record",
        lambda *args, **kwargs: {
            "artifacts": [
                {
                    "source": "huggingface",
                    "repo_id": "Mapika/decider-2b-GGUF",
                    "role": "decider-gguf-q8_0",
                    "requested_revision": "main",
                    "resolved_revision": revision,
                }
            ]
        },
    )
    settings = SimpleNamespace(
        model_revision="main",
        config_path=tmp_path / "config.json",
        model_id="decider-2b",
        backend="gguf",
    )
    profile = {
        "model": "Mapika/decider-2b-GGUF",
        "model_revision": "main",
        "pin_hf_revisions_at_install": True,
    }

    pinned = runtime_module._profile_with_installed_pin(settings, profile)

    assert pinned["model_revision"] == revision
    assert profile["model_revision"] == "main"


def test_fix02_llama_cpp_build_exposes_isolated_ninja_to_cmake(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os
    import deqio.model_manager as manager

    config_path = tmp_path / "config.json"
    runtime_root = tmp_path / ".model-runtimes"
    env_dir = runtime_root / "llama-cpp-systemone"
    bin_dir = env_dir / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    python = bin_dir / ("python.exe" if os.name == "nt" else "python")
    cmake = bin_dir / ("cmake.exe" if os.name == "nt" else "cmake")
    ninja = bin_dir / ("ninja.exe" if os.name == "nt" else "ninja")
    for executable in (python, cmake, ninja):
        executable.write_text("", encoding="utf-8")

    monkeypatch.setattr(manager, "_ensure_venv", lambda *args, **kwargs: python)
    monkeypatch.setattr(manager.sys, "platform", "darwin")
    monkeypatch.setattr(
        manager.shutil,
        "which",
        lambda name, path=None: "/usr/bin/xcrun" if name == "xcrun" else None,
    )

    class ProbeResult:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    monkeypatch.setattr(
        manager.subprocess,
        "run",
        lambda command, **kwargs: ProbeResult(
            "/usr/bin/clang++\n" if command[-1] == "clang++" else "/usr/bin/clang\n"
        ),
    )
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_run(command: list[str], *, env: dict[str, str] | None = None) -> None:
        calls.append((list(command), env))
        if command[:2] == ["git", "clone"]:
            Path(command[-1]).mkdir(parents=True, exist_ok=True)
        if "--build" in command:
            binary = env_dir / "llama-build" / "bin" / "llama-server"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text("server", encoding="utf-8")

    monkeypatch.setattr(manager, "_run", fake_run)
    profile = {
        "runtime_key": "llama-cpp-systemone",
        "python": "3.12",
        "llama_cpp_revision": "1" * 40,
    }

    result = manager._install_llama_cpp_runtime(
        config_path,
        {"runtime_dir": str(runtime_root)},
        profile,
        upgrade=False,
    )

    configure = next((command, env) for command, env in calls if "-G" in command and "Ninja" in command)
    command, env = configure
    assert f"-DCMAKE_MAKE_PROGRAM={ninja}" in command
    assert "-DGGML_METAL=ON" in command
    assert "-DCMAKE_C_COMPILER=/usr/bin/clang" in command
    assert "-DCMAKE_CXX_COMPILER=/usr/bin/clang++" in command
    assert env is not None
    assert env["PATH"].split(os.pathsep)[0] == str(bin_dir)
    build_command, build_env = next((command, env) for command, env in calls if "--build" in command)
    assert build_env is not None
    assert build_env["PATH"].split(os.pathsep)[0] == str(bin_dir)
    assert result == env_dir
    assert (bin_dir / ("llama-server.exe" if os.name == "nt" else "llama-server")).is_file()


def test_fix03_model_readiness_retries_transient_http_503(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    import deqio.systemone_runtime as runtime_module

    runtime = object.__new__(runtime_module.SystemOneRuntime)
    runtime.base_url = "http://127.0.0.1:9999"
    runtime.profile = {"wire_model": "kev-latest"}
    runtime.settings = SimpleNamespace(model="ggml-org/Kev-0.8B-GGUF")
    runtime.process = SimpleNamespace(poll=lambda: None, returncode=None)

    calls = {"count": 0}

    def fake_post_json(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise runtime_module.SystemOneSidecarHTTPError(
                503,
                '{"error":{"message":"Loading model","type":"unavailable_error","code":503}}',
            )
        return {"answers": {"ready": {"type": "choice", "choice": "ready"}}}

    monkeypatch.setattr(runtime_module, "_post_json", fake_post_json)
    monkeypatch.setattr(runtime_module.time, "sleep", lambda *_: None)

    runtime._probe_model_ready(timeout=5.0)

    assert calls["count"] == 3


def test_fix03_model_readiness_does_not_hide_non_503_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    import deqio.systemone_runtime as runtime_module

    runtime = object.__new__(runtime_module.SystemOneRuntime)
    runtime.base_url = "http://127.0.0.1:9999"
    runtime.profile = {"wire_model": "kev-latest"}
    runtime.settings = SimpleNamespace(model="ggml-org/Kev-0.8B-GGUF")
    runtime.process = SimpleNamespace(poll=lambda: None, returncode=None)

    calls = {"count": 0}

    def fake_post_json(*args, **kwargs):
        calls["count"] += 1
        raise runtime_module.SystemOneSidecarHTTPError(422, '{"detail":"invalid request"}')

    monkeypatch.setattr(runtime_module, "_post_json", fake_post_json)

    with pytest.raises(RuntimeError, match=r"HTTP 422"):
        runtime._probe_model_ready(timeout=5.0)

    assert calls["count"] == 1


def test_systemone_runtime_close_terminates_posix_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import signal
    import deqio.systemone_runtime as runtime_module

    if os.name == "nt":
        pytest.skip("POSIX process-group lifecycle test")

    class DummyProcess:
        pid = 4242
        returncode = None
        stdout = None
        terminate_called = False
        kill_called = False
        wait_calls = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminate_called = True
            self.returncode = 0

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise runtime_module.subprocess.TimeoutExpired("guard", timeout)
            self.returncode = 0
            return 0

        def kill(self):
            self.kill_called = True
            self.returncode = -9

    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        runtime_module.os, "killpg", lambda pgid, sig: signals.append((pgid, sig))
    )

    runtime = object.__new__(runtime_module.SystemOneRuntime)
    runtime.process = DummyProcess()
    runtime.engine = "jevk5"
    runtime._log_thread = None
    runtime._engine_pid_file = None

    runtime.close()

    assert signals == [(4242, signal.SIGTERM)]
    assert runtime.process.terminate_called is False
    assert runtime.process.kill_called is False


def test_jevk5_inner_llama_server_cleanup_waits_for_exit() -> None:
    from deqio.jevk5_gguf_sidecar import _terminate

    class DummyProcess:
        returncode = None
        terminate_called = False
        waited: list[int | None] = []

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminate_called = True

        def wait(self, timeout=None):
            self.waited.append(timeout)
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = -9

    process = DummyProcess()
    _terminate(process)

    assert process.terminate_called is True
    assert process.waited == [10]


def test_load_warmed_runtime_closes_runtime_when_warmup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    class FakeRuntime:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake = FakeRuntime()

    class FakeBackendRuntime:
        @staticmethod
        def load(settings):
            return fake

    monkeypatch.setattr(server, "BackendRuntime", FakeBackendRuntime)
    monkeypatch.setattr(
        server,
        "_warmup_runtime",
        lambda target, announce: (_ for _ in ()).throw(RuntimeError("warmup failed")),
    )

    with pytest.raises(RuntimeError, match="warmup failed"):
        server._load_warmed_runtime(server.SETTINGS, announce=False)

    assert fake.closed is True


def test_rejected_second_server_does_not_clear_watch_session(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    events: list[str] = []
    monkeypatch.setenv("DEQIO_SERVER_CONTROL", "1")
    monkeypatch.setattr(server, "startup_header", lambda **kwargs: None)
    monkeypatch.setattr(
        server,
        "register_server",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("already running")),
    )
    monkeypatch.setattr(
        server, "_reset_watch_session", lambda **kwargs: events.append("watch-reset") or {}
    )

    async def invoke() -> None:
        with pytest.raises(RuntimeError, match="already running"):
            async with server.lifespan(server.app):
                pass

    asyncio.run(invoke())
    assert events == []


def test_failed_startup_unregisters_control_before_propagating(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    events: list[str] = []

    monkeypatch.setenv("DEQIO_SERVER_CONTROL", "1")
    monkeypatch.setenv("DEQIO_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("DEQIO_SERVER_PORT", "8787")
    monkeypatch.setattr(server, "_reset_watch_session", lambda **kwargs: {})
    monkeypatch.setattr(server, "startup_header", lambda **kwargs: None)
    monkeypatch.setattr(
        server,
        "register_server",
        lambda *args, **kwargs: events.append("register") or {"pid": 123, "token": "token"},
    )
    monkeypatch.setattr(
        server,
        "unregister_server",
        lambda *args, **kwargs: events.append("unregister"),
    )
    monkeypatch.setattr(
        server,
        "_load_warmed_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("startup failed")),
    )

    async def invoke() -> None:
        with pytest.raises(RuntimeError, match="startup failed"):
            async with server.lifespan(server.app):
                pass

    asyncio.run(invoke())

    assert events == ["register", "unregister"]
    assert server.runtime_control_token is None


def test_choice_rejects_empty_or_duplicate_option_ids_before_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("inference must not run")

    monkeypatch.setattr(server, "run_decision", fail_if_called)

    for options in (
        [],
        [{"id": "same", "description": "A"}, {"id": "same", "description": "B"}],
        [{"id": "   ", "description": "blank"}],
    ):
        status, payload = _asgi_post_json(
            "/v1/choice",
            json.dumps({"state": "state", "question": "Choose", "options": options}),
            {"content-type": "application/json"},
        )
        assert status == 400
        assert "option" in str(payload["detail"]).lower()

    assert called is False


def test_shared_rejects_duplicate_decision_or_option_ids_before_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    class NoInferenceRuntime:
        def score_shared(self, rows):
            raise AssertionError("inference must not run")

    monkeypatch.setattr(server, "runtime", NoInferenceRuntime())

    duplicate_decisions = {
        "state": "state",
        "decisions": [
            {"id": "same", "question": "A?", "options": [{"id": "a", "description": "A"}]},
            {"id": "same", "question": "B?", "options": [{"id": "b", "description": "B"}]},
        ],
    }
    status, payload = _asgi_post_json(
        "/v1/shared", json.dumps(duplicate_decisions), {"content-type": "application/json"}
    )
    assert status == 400
    assert "duplicate decision ids" in str(payload["detail"])

    duplicate_options = {
        "state": "state",
        "decisions": [
            {
                "question": "A?",
                "options": [
                    {"id": "same", "description": "A"},
                    {"id": "same", "description": "B"},
                ],
            }
        ],
    }
    status, payload = _asgi_post_json(
        "/v1/shared", json.dumps(duplicate_options), {"content-type": "application/json"}
    )
    assert status == 400
    assert "duplicate option ids" in str(payload["detail"])


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/v1/choice",
            {
                "state": "state",
                "question": "Choose",
                "options": [
                    {"id": "a", "description": "A"},
                    {"id": "b", "description": "B"},
                ],
            },
        ),
        ("/v1/noul", {"state": "state", "question": "Is this true?"}),
        (
            "/v1/shared",
            {
                "state": "state",
                "decisions": [
                    {
                        "question": "Choose",
                        "options": [
                            {"id": "a", "description": "A"},
                            {"id": "b", "description": "B"},
                        ],
                    }
                ],
            },
        ),
    ],
)
def test_portable_endpoints_preserve_503_while_runtime_is_suspended(
    path: str, body: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.server as server

    monkeypatch.setattr(server, "runtime", None)
    monkeypatch.setattr(
        server,
        "runtime_suspension",
        {
            "owner": "benchmark",
            "reason": "benchmark",
            "lease_id": "lease-http",
            "owner_pid": 12345,
            "started_at": "2026-10-06T00:00:00+00:00",
            "previous_profile": server._active_model(),
            "restore_error": None,
        },
    )

    status, payload = _asgi_post_json(
        path,
        json.dumps(body),
        {"content-type": "application/json"},
    )

    assert status == 503
    assert payload["detail"]["code"] == "runtime_suspended_for_benchmark"


def test_runtime_suspension_keeps_server_state_but_unloads_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    class FakeRuntime:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    active = FakeRuntime()
    monkeypatch.setattr(server, "runtime", active)
    monkeypatch.setattr(server, "runtime_suspension", None)
    monkeypatch.setattr(server, "switching_runtime", False)

    response = server._suspend_runtime_for_benchmark(
        lease_id="lease-1",
        owner_pid=12345,
        reason="benchmark",
    )

    assert response["status"] == "suspended"
    assert active.closed is True
    assert server.runtime is None
    assert server.runtime_suspension["lease_id"] == "lease-1"
    with pytest.raises(server.HTTPException) as error:
        server._runtime()
    assert error.value.status_code == 503
    assert error.value.detail["code"] == "runtime_suspended_for_benchmark"


def test_runtime_resume_restores_previous_server_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    class FakeRuntime:
        def close(self):
            pass

        def refresh_identity(self):
            pass

    restored = FakeRuntime()

    class FakeBackendRuntime:
        @staticmethod
        def load(settings):
            assert settings is server.SETTINGS
            return restored

    monkeypatch.setattr(server, "runtime", None)
    monkeypatch.setattr(server, "runtime_suspension", {
        "owner": "benchmark",
        "reason": "benchmark",
        "lease_id": "lease-2",
        "owner_pid": 12345,
        "started_at": "2026-10-05T00:00:00+00:00",
        "previous_profile": server._active_model(),
        "restore_error": None,
    })
    monkeypatch.setattr(server, "BackendRuntime", FakeBackendRuntime)
    monkeypatch.setattr(server, "_warmup_runtime", lambda runtime, announce=False: None)
    monkeypatch.setattr(server, "mark_installed", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_reset_session_metrics", lambda: None)

    response = server._resume_runtime_after_benchmark(lease_id="lease-2")

    assert response["status"] == "restored"
    assert server.runtime is restored
    assert server.runtime_suspension is None


def test_model_switch_is_blocked_while_benchmark_owns_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    monkeypatch.setattr(server, "runtime_suspension", {
        "owner": "benchmark",
        "reason": "benchmark",
        "lease_id": "lease-3",
        "owner_pid": 12345,
        "started_at": "2026-10-05T00:00:00+00:00",
        "previous_profile": server._active_model(),
        "restore_error": None,
    })
    with pytest.raises(server.HTTPException) as error:
        server.activate_model(server.ModelActivateRequest(model_id="whatever", backend="mlx"))
    assert error.value.status_code == 409
    assert error.value.detail["code"] == "runtime_suspended_for_benchmark"


def test_model_switch_rechecks_benchmark_suspension_after_waiting_for_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.server as server
    from deqio.catalog import load_catalog
    from deqio.config import settings_from_data

    catalog_path = Path("models.json").resolve()
    config_path = tmp_path / "config.json"
    data = {
        "engine": "laya",
        "model_id": "laya-typed-decisions",
        "backend": "mlx",
        "model": "aac6fef/laya-typed-decisions-mlx",
        "model_revision": "laya-typed-decisions",
        "model_catalog": str(catalog_path),
        "runtime_dir": str(tmp_path / ".model-runtimes"),
        "max_tokens": 4096,
        "mlx_cache_mib": 256,
        "log": str(tmp_path / "requests.jsonl"),
        "torch_dtype": "bfloat16",
        "sidecar_startup_seconds": 900,
    }
    config_path.write_text(json.dumps(data), encoding="utf-8")
    old_settings = settings_from_data(config_path, data, apply_environment=False)
    catalog = load_catalog(catalog_path)

    class ExistingRuntime:
        closed = False

        def close(self) -> None:
            self.closed = True

    existing = ExistingRuntime()
    suspension = {
        "owner": "benchmark",
        "reason": "benchmark",
        "lease_id": "race-lease",
        "owner_pid": 12345,
        "started_at": "2026-10-06T00:00:00+00:00",
        "previous_profile": server._active_model(),
        "restore_error": None,
    }

    class RaceLock:
        def __enter__(self):
            # Simulate the benchmark acquiring the lock after activate_model's
            # initial fast-path check but before model switching starts.
            monkeypatch.setattr(server, "runtime_suspension", suspension)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    row = {
        "model_id": "decider-2b",
        "backend": "mps",
        "engine": "decider",
        "installed": True,
        "host_compatible": True,
        "max_input_tokens": 4096,
    }
    monkeypatch.setattr(server, "SETTINGS", old_settings)
    monkeypatch.setattr(server, "runtime", existing)
    monkeypatch.setattr(server, "runtime_suspension", None)
    monkeypatch.setattr(server, "inference_lock", RaceLock())
    monkeypatch.setattr(server, "_installation_rows", lambda: (catalog, [row]))
    for env_name in server.MODEL_SELECTION_ENV_VARS:
        monkeypatch.delenv(env_name, raising=False)

    with pytest.raises(server.HTTPException) as error:
        server.activate_model(server.ModelActivateRequest(model_id="decider-2b", backend="mps"))

    assert error.value.status_code == 409
    assert error.value.detail["code"] == "runtime_suspended_for_benchmark"
    assert existing.closed is False


def test_same_profile_activation_reloads_when_runtime_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.server as server
    from deqio.catalog import load_catalog
    from deqio.config import settings_from_data

    catalog_path = Path("models.json").resolve()
    config_path = tmp_path / "config.json"
    data = {
        "engine": "laya",
        "model_id": "laya-typed-decisions",
        "backend": "mlx",
        "model": "aac6fef/laya-typed-decisions-mlx",
        "model_revision": "laya-typed-decisions",
        "model_catalog": str(catalog_path),
        "runtime_dir": str(tmp_path / ".model-runtimes"),
        "max_tokens": 4096,
        "mlx_cache_mib": 256,
        "log": str(tmp_path / "requests.jsonl"),
        "torch_dtype": "bfloat16",
        "sidecar_startup_seconds": 900,
    }
    config_path.write_text(json.dumps(data), encoding="utf-8")
    settings = settings_from_data(config_path, data, apply_environment=False)
    catalog = load_catalog(catalog_path)
    row = {
        "model_id": settings.model_id,
        "backend": settings.backend,
        "engine": settings.engine,
        "installed": True,
        "host_compatible": True,
        "max_input_tokens": 4096,
    }

    class FakeRuntime:
        def refresh_identity(self):
            pass

        def close(self):
            pass

    fake = FakeRuntime()

    class FakeBackendRuntime:
        @staticmethod
        def load(candidate_settings):
            assert candidate_settings.model_id == settings.model_id
            return fake

    monkeypatch.setattr(server, "SETTINGS", settings)
    monkeypatch.setattr(server, "runtime", None)
    monkeypatch.setattr(server, "runtime_suspension", None)
    monkeypatch.setattr(server, "BackendRuntime", FakeBackendRuntime)
    monkeypatch.setattr(server, "_warmup_runtime", lambda runtime, announce=False: None)
    monkeypatch.setattr(server, "_installation_rows", lambda: (catalog, [row]))
    monkeypatch.setattr(server, "mark_installed", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_reset_session_metrics", lambda: None)
    for env_name in server.MODEL_SELECTION_ENV_VARS:
        monkeypatch.delenv(env_name, raising=False)

    response = server.activate_model(
        server.ModelActivateRequest(model_id=settings.model_id, backend=settings.backend)
    )

    assert response["status"] == "ok"
    assert server.runtime is fake


def test_failed_benchmark_suspend_never_leaves_half_closed_runtime_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deqio.server as server

    class BrokenRuntime:
        def close(self):
            raise RuntimeError("close failed")

    monkeypatch.setattr(server, "runtime", BrokenRuntime())
    monkeypatch.setattr(server, "runtime_suspension", None)
    monkeypatch.setattr(server, "switching_runtime", False)

    with pytest.raises(RuntimeError, match="close failed"):
        server._suspend_runtime_for_benchmark(
            lease_id="lease-close-failure", owner_pid=12345, reason="benchmark"
        )

    # Keep the reference when close itself fails. Dropping it would make a
    # still-live sidecar impossible to retry/clean during shutdown and is the
    # exact shape that can produce an orphaned model process.
    assert isinstance(server.runtime, BrokenRuntime)
    assert server.runtime_suspension is None
    assert server.switching_runtime is False


def test_patch3_routes_and_ui_are_exposed() -> None:
    routes = {route.path for route in app.routes}
    assert "/v1/benchmarks/compare" in routes
    assert "/v1/internal/runtime/suspend" in routes
    assert "/v1/internal/runtime/resume" in routes
    assert 'data-benchmark-view="compare"' in DASHBOARD
    assert "benchmarkCompareLeft" in DASHBOARD
    assert "benchmarkCompareRight" in DASHBOARD
    assert "runtime_suspension" in WATCH_DASHBOARD


@pytest.mark.parametrize("endpoint_name,question_type", [("score_native", "score"), ("multi_native", "multi"), ("act_native", "act")])
def test_patch4_native_single_endpoint_wrappers_inject_type_and_preserve_response(
    endpoint_name: str, question_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.server as server

    calls: list[tuple[object, dict, dict | None]] = []

    class FakeRuntime:
        def identity_snapshot(self):
            return {
                "runtime_instance_id": "native-runtime", "engine": "basal",
                "model_id": "basal-1.5-mini", "backend": "mlx",
                "artifact_revisions_resolved": True,
            }

        def system_one(self, state, questions, options=None):
            calls.append((state, questions, options))
            name, question = next(iter(questions.items()))
            return (
                {
                    "model": "native-model",
                    "answers": {name: {"type": question["type"], "probabilities": {"x": 1.0}}},
                    "usage": {"input_tokens": 3, "output_tokens": 0},
                },
                {"total_seconds": 0.001},
            )

    monkeypatch.setattr(server, "runtime", FakeRuntime())
    monkeypatch.setattr(server, "record_event", lambda **kwargs: None)
    monkeypatch.setattr(server, "_record_watch_event", lambda **kwargs: None)
    monkeypatch.setattr(server, "log_request_error", lambda *args, **kwargs: None)

    handler = getattr(server, endpoint_name)
    result = asyncio.run(handler(server.TypedSystemOneRequest(
        state="state", name="check", question={"instructions": "Evaluate", "criteria": ["x"]}, facts="auto"
    )))

    assert calls == [("state", {"check": {"instructions": "Evaluate", "criteria": ["x"], "type": question_type}}, {"facts": "auto"})]
    assert result["answers"]["check"]["type"] == question_type
    assert result["deqio"]["timing"]["decisions"] == 1


def test_patch4_soam_alias_uses_same_native_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    calls = []

    class FakeRuntime:
        def identity_snapshot(self):
            return {"runtime_instance_id": "r", "engine": "kev", "model_id": "kev-0.8b", "backend": "mlx"}

        def system_one(self, state, questions):
            calls.append((state, questions))
            return ({"model": "kev", "answers": {"a": {"type": "noul", "noul": 0.8}}}, {"total_seconds": 0.002})

    monkeypatch.setattr(server, "runtime", FakeRuntime())
    monkeypatch.setattr(server, "record_event", lambda **kwargs: None)
    monkeypatch.setattr(server, "_record_watch_event", lambda **kwargs: None)
    monkeypatch.setattr(server, "log_request_error", lambda *args, **kwargs: None)

    payload = server.SystemOneRequest(state="state", questions={"a": {"type": "noul", "instructions": "Allowed?"}})
    result = asyncio.run(server.soam(payload))
    assert calls == [("state", {"a": {"type": "noul", "instructions": "Allowed?"}})]
    assert result["answers"]["a"]["noul"] == 0.8


def test_patch4_ui_reloads_native_example_when_active_profile_changes() -> None:
    assert "activeModelKey !== systemOneExampleModelKey" in DASHBOARD
    assert "endpointInitialized.delete(nativeEndpoint)" in DASHBOARD
    assert "loadEndpointExample(endpoint)" in DASHBOARD


def test_patch4_watch_filter_lists_new_native_endpoints() -> None:
    for path in ("/v1/score", "/v1/multi", "/v1/act", "/v1/soam", "/v1/systemone"):
        assert f'<option value="{path}">{path}</option>' in WATCH_DASHBOARD


def test_patch4_benchmark_compare_uses_responsive_cards_not_wide_metric_table() -> None:
    assert 'id="benchmarkCompareCards"' in DASHBOARD
    assert 'class="compare-card"' in DASHBOARD
    assert 'class="compare-metrics"' in DASHBOARD
    assert 'id="benchmarkCompareRows"' not in DASHBOARD


def test_patch4_readme_is_simplified_and_documents_public_api_and_benchmarks() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert len(readme.splitlines()) < 700
    for heading in (
        "## API at a glance",
        "### Small models (up to 2.5B)",
        "### Large models",
        "## Installation",
        "# API examples",
    ):
        assert heading in readme
    for path in ("/v1/noul", "/v1/choice", "/v1/shared", "/v1/score", "/v1/multi", "/v1/act", "/v1/soam", "/v1/systemone"):
        assert path in readme
    assert '`facts: "auto"`' in readme
    assert "not `/v1/facts`" in readme
    assert "convenience alias over the native SystemOne contract" in readme
    assert "same native SystemOne execution path" in readme
    assert "`/v1/soam` vs `/v1/systemone`" in readme
    assert "Extra inference layer" in readme
    assert "**Decision in, probabilities out.**" in readme
    assert "20261006T082750Z" in readme
    assert "20261006T083557Z" in readme
    assert "20261006T090441Z" in readme
    assert "20261006T091845Z" in readme
    assert "Basal 1.5 Mini 1.5B" in readme
    assert "Decider 0.8B" in readme
    assert "Laya Typed Decisions 421M" in readme
    assert "Basal 1.5 Main 4.5B" in readme
    assert "JevK5 4B" in readme
    assert "### Large models (4B–4.5B in this snapshot)" in readme
    assert readme.count("#### Runs") == 2
    assert "0 runtime errors" in readme
    assert "## Supported models and backends" in readme
    assert readme.index("## Supported models and backends") < readme.index("## Benchmark snapshot")
    assert "`decision-2.0-vega`" in readme
    assert "`basal-1.5-main`" in readme
    assert "`laya-english`" in readme
    benchmark_snapshot = readme[readme.index("## Benchmark snapshot"):readme.index("## Installation")]
    assert "Laya English 421M" not in benchmark_snapshot
    assert "earlier local snapshot from **2026-10-03**" not in readme

    eng_summary = Path("benchmarks/summary_eng.md").read_text(encoding="utf-8")
    pl_summary = Path("benchmarks/summary_pl.md").read_text(encoding="utf-8")
    assert "20261006T082750Z" in eng_summary
    assert "20261006T090441Z" in eng_summary
    assert "20261006T083557Z" in pl_summary
    assert "20261006T091845Z" in pl_summary
    assert "2026-10-03" not in eng_summary
    assert "2026-10-03" not in pl_summary


# --- Production audit regressions (Deqio 0.5 freeze) -------------------------


def _raw_decision(decision_id: str = "r1") -> dict:
    return {
        "id": decision_id,
        "option_ids": ["yes", "no"],
        "probabilities": [0.8, 0.2],
        "input_tokens": 3,
        "total_seconds": 0.002,
        "prompt_sha256": "sha",
        "probability_status": "test",
    }


def test_portable_endpoints_keep_watch_and_request_log_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watch storage takes a process lock and does disk I/O.

    Every portable decision handler must call it from a worker thread, exactly
    like the native SystemOne path does, otherwise a contended Watch lock or a
    slow disk freezes /health, the UI and every other in-flight request.
    """
    import threading

    import deqio.server as server

    loop_thread = threading.current_thread()
    offending: list[str] = []

    def _assert_off_loop(label: str) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        offending.append(label)

    class SpyWatch:
        def session_token(self):
            _assert_off_loop("session_token")
            return "session"

        def append(self, event, *, expected_session_id=None):
            _assert_off_loop(f"append:{event['endpoint']}")
            return "000001-00001-abcdef"

    def fake_log_write(**kwargs):
        _assert_off_loop(f"record_event:{kwargs['endpoint']}")

    identity = {"engine": "kev", "model_id": "demo", "backend": "mps", "runtime_instance_id": "rt"}
    monkeypatch.setattr(server, "_watch_store", lambda: SpyWatch())
    monkeypatch.setattr(server, "record_event", fake_log_write)
    monkeypatch.setattr(
        server,
        "run_decision",
        lambda payload, *, endpoint, contract: server.format_result(_raw_decision(), runtime_identity=identity),
    )
    monkeypatch.setattr(
        server,
        "_score_noul_inference",
        lambda row, mode, contract, request_id: (server.SETTINGS, identity, _raw_decision(request_id)),
    )
    monkeypatch.setattr(
        server,
        "_score_shared_inference",
        lambda rows, contract, request_id: (
            server.SETTINGS,
            identity,
            [_raw_decision(str(row["id"])) for row in rows],
            {"total_seconds": 0.004},
        ),
    )

    headers = {"content-type": "application/json"}
    status, _ = _asgi_post_json(
        "/v1/choice",
        json.dumps({"state": "s", "question": "q", "options": [{"id": "yes", "description": "Y"}, {"id": "no", "description": "N"}]}),
        headers,
    )
    assert status == 200
    status, _ = _asgi_post_json("/v1/noul", json.dumps({"state": "s", "question": "q"}), headers)
    assert status == 200
    status, _ = _asgi_post_json(
        "/v1/shared",
        json.dumps({"state": "s", "decisions": [{"question": "q", "options": [{"id": "yes", "description": "Y"}, {"id": "no", "description": "N"}]}]}),
        headers,
    )
    assert status == 200
    # A rejected request must record its Watch row off the loop as well.
    status, _ = _asgi_post_json(
        "/v1/choice",
        json.dumps({"state": "s", "question": "q", "options": [{"id": "dup", "description": "A"}, {"id": "dup", "description": "B"}]}),
        headers,
    )
    assert status == 400

    assert offending == []
    assert threading.current_thread() is loop_thread


def test_watch_event_reads_native_systemone_timing_and_usage() -> None:
    import deqio.server as server

    captured: list[dict] = []

    class SpyWatch:
        def append(self, event, *, expected_session_id=None):
            captured.append(event)
            return "000001-00001-abcdef"

    native_response = {
        "answers": {"ready": {"answer": "yes"}},
        "usage": {"input_tokens": 55},
        "deqio": {
            "provenance_schema_version": 1,
            "runtime": {"engine": "basal", "model_id": "basal-1.5-mini", "backend": "mlx", "runtime_instance_id": "rt-1"},
            "timing": {"total_ms": 12.3, "decisions": 1},
        },
    }
    original = server._watch_store
    server._watch_store = lambda: SpyWatch()
    try:
        event_id = server._record_watch_event(
            expected_session_id="session",
            endpoint="/v1/score",
            request_payload={"state": "s", "name": "ready", "question": {}},
            response_payload=native_response,
            status_code=200,
            request_id="systemone-1",
            mode="systemone",
            decisions=1,
            settings_snapshot=server.SETTINGS,
        )
    finally:
        server._watch_store = original

    assert event_id is not None
    event = captured[0]
    assert event["latency_ms"] == 12.3
    assert event["input_tokens"] == 55
    assert event["runtime_instance_id"] == "rt-1"
    assert event["model_id"] == "basal-1.5-mini"
    # The full native response is still stored for the Watch detail view.
    assert event["response"]["deqio"]["timing"]["total_ms"] == 12.3


def _activation_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, base_overrides: dict | None = None):
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
        **(base_overrides or {}),
    }
    config_path.write_text(json.dumps(base_data))
    catalog = {
        "models": [
            {"id": "decider-0.8b", "engine": "decider", "label": "Decider 0.8B", "backends": {"mps": {"model": "Mapika/decider-0.8b"}}},
            {"id": "decider-2b", "engine": "decider", "label": "Decider 2B", "backends": {"mps": {"model": "Mapika/decider-2b"}}},
        ]
    }
    rows = [
        {"model_id": "decider-0.8b", "backend": "mps", "engine": "decider", "installed": True, "host_compatible": True},
        {"model_id": "decider-2b", "backend": "mps", "engine": "decider", "installed": True, "host_compatible": True},
    ]

    class FakeRuntime:
        def __init__(self, settings):
            self.settings = settings
            self.closed = False

        def score(self, row, mode):
            return _raw_decision(row["id"])

        def close(self):
            self.closed = True

    class FakeBackendRuntime:
        loaded: list = []

        @staticmethod
        def load(settings):
            instance = FakeRuntime(settings)
            FakeBackendRuntime.loaded.append(instance)
            return instance

    old_settings = settings_from_data(config_path, base_data, apply_environment=True)
    monkeypatch.setattr(server, "SETTINGS", old_settings)
    monkeypatch.setattr(server, "runtime", FakeRuntime(old_settings))
    monkeypatch.setattr(server, "runtime_suspension", None)
    monkeypatch.setattr(server, "BackendRuntime", FakeBackendRuntime)
    monkeypatch.setattr(server, "_installation_rows", lambda: (catalog, rows))
    monkeypatch.setattr(server, "mark_installed", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_watch_session_token", lambda: None)
    monkeypatch.setattr(server, "_reset_session_metrics", lambda: None)
    for env_name in server.MODEL_SELECTION_ENV_VARS:
        monkeypatch.delenv(env_name, raising=False)
    return server, FakeBackendRuntime


def test_live_activation_keeps_non_selection_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_runtimes = tmp_path / "custom-runtimes"
    custom_log = tmp_path / "custom" / "requests.jsonl"
    monkeypatch.setenv("DEQIO_RUNTIME_DIR", str(custom_runtimes))
    monkeypatch.setenv("DEQIO_LOG", str(custom_log))
    monkeypatch.setenv("DEQIO_SIDECAR_STARTUP_SECONDS", "3600")
    monkeypatch.setenv("DEQIO_HF_OFFLINE_RUNTIME", "0")
    server, backend = _activation_fixture(tmp_path, monkeypatch)
    assert server.SETTINGS.runtime_dir == custom_runtimes.resolve()

    response = server.activate_model(server.ModelActivateRequest(model_id="decider-2b", backend="mps"))

    assert response["status"] == "ok"
    loaded = backend.loaded[-1].settings
    assert loaded.model_id == "decider-2b"
    assert loaded.runtime_dir == custom_runtimes.resolve()
    assert loaded.log_path == custom_log.resolve()
    assert loaded.sidecar_startup_seconds == 3600
    assert loaded.hf_offline_runtime is False
    assert server.SETTINGS is loaded
    # The persisted selection still does not bake the environment in.
    persisted = json.loads((tmp_path / "config.json").read_text())
    assert persisted["runtime_dir"] == ".model-runtimes"
    assert persisted["model_id"] == "decider-2b"


def test_live_activation_bookkeeping_failure_does_not_fail_or_stick_the_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The request-log directory cannot be created (its parent is a file) and
    # the registry cannot be written: both are bookkeeping, not the switch.
    (tmp_path / "blocker").write_text("not a directory")
    server, backend = _activation_fixture(
        tmp_path, monkeypatch, base_overrides={"log": str(tmp_path / "blocker" / "requests.jsonl")}
    )

    def broken_registry(*args, **kwargs):
        raise OSError("registry is read-only")

    monkeypatch.setattr(server, "mark_installed", broken_registry)
    monkeypatch.setattr(server, "switching_runtime", False)
    messages: list[str] = []
    monkeypatch.setattr(server, "info", messages.append)

    response = server.activate_model(server.ModelActivateRequest(model_id="decider-2b", backend="mps"))

    assert response["status"] == "ok"
    assert server.SETTINGS.model_id == "decider-2b"
    assert server.runtime is backend.loaded[-1]
    assert server.switching_runtime is False
    assert any("installed-model registry" in message for message in messages)


def test_live_activation_clears_switching_flag_when_switch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, backend = _activation_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "switching_runtime", False)

    def explode(settings, *, announce):
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "_load_warmed_runtime", explode)

    with pytest.raises(server.HTTPException) as error:
        server.activate_model(server.ModelActivateRequest(model_id="decider-2b", backend="mps"))

    assert error.value.status_code == 500
    assert server.switching_runtime is False
    assert server.runtime is None


def test_benchmark_restore_survives_registry_write_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    class FakeRuntime:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    restored = FakeRuntime()

    class FakeBackendRuntime:
        @staticmethod
        def load(settings):
            return restored

    def broken_registry(*args, **kwargs):
        raise OSError("registry lock timeout")

    monkeypatch.setattr(server, "runtime", None)
    monkeypatch.setattr(server, "runtime_suspension", {
        "owner": "benchmark",
        "reason": "benchmark",
        "lease_id": "lease-9",
        "owner_pid": 12345,
        "started_at": "2026-10-07T00:00:00+00:00",
        "previous_profile": server._active_model(),
        "restore_error": None,
    })
    monkeypatch.setattr(server, "BackendRuntime", FakeBackendRuntime)
    monkeypatch.setattr(server, "_warmup_runtime", lambda runtime, announce=False: None)
    monkeypatch.setattr(server, "mark_installed", broken_registry)
    monkeypatch.setattr(server, "_reset_session_metrics", lambda: None)
    messages: list[str] = []
    monkeypatch.setattr(server, "info", messages.append)

    response = server._resume_runtime_after_benchmark(lease_id="lease-9")

    assert response["status"] == "restored"
    assert server.runtime is restored
    assert restored.closed is False
    assert server.runtime_suspension is None
    assert any("installed-model registry" in message for message in messages)


def test_startup_survives_registry_write_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    class FakeRuntime:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    loaded = FakeRuntime()
    monkeypatch.delenv("DEQIO_SERVER_CONTROL", raising=False)
    monkeypatch.setattr(server, "_reset_watch_session", lambda **kwargs: {})
    monkeypatch.setattr(server, "startup_header", lambda **kwargs: None)
    monkeypatch.setattr(server, "server_ready", lambda **kwargs: None)
    monkeypatch.setattr(server, "_load_warmed_runtime", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(server, "mark_installed", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    monkeypatch.setattr(server, "runtime", None)

    async def invoke() -> None:
        async with server.lifespan(server.app):
            assert server.runtime is loaded
            assert loaded.closed is False

    asyncio.run(invoke())
    assert loaded.closed is True


def test_shutdown_closes_runtime_before_releasing_workspace_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    events: list[str] = []

    class FakeRuntime:
        def close(self):
            events.append("close")
            raise RuntimeError("sidecar refused to stop")

    monkeypatch.setenv("DEQIO_SERVER_CONTROL", "1")
    monkeypatch.setenv("DEQIO_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("DEQIO_SERVER_PORT", "8787")
    monkeypatch.setattr(server, "_reset_watch_session", lambda **kwargs: {})
    monkeypatch.setattr(server, "startup_header", lambda **kwargs: None)
    monkeypatch.setattr(server, "server_ready", lambda **kwargs: None)
    monkeypatch.setattr(server, "register_server", lambda *a, **k: events.append("register") or {"pid": 1, "token": "t"})
    monkeypatch.setattr(server, "unregister_server", lambda *a, **k: events.append("unregister"))
    monkeypatch.setattr(server, "_load_warmed_runtime", lambda *a, **k: FakeRuntime())
    monkeypatch.setattr(server, "mark_installed", lambda *a, **k: None)
    monkeypatch.setattr(server, "runtime", None)

    async def invoke() -> None:
        with pytest.raises(RuntimeError, match="refused to stop"):
            async with server.lifespan(server.app):
                pass

    asyncio.run(invoke())

    assert events == ["register", "close", "unregister"]
    assert server.runtime is None
    assert server.runtime_control_token is None


def test_delete_active_profile_replacement_restores_recorded_token_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from deqio import installations, model_manager
    from deqio.installations import registry_path

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "engine": "kev",
        "model_id": "big",
        "backend": "gguf",
        "model": "owner/big",
        "model_revision": "main",
        "model_catalog": "models.json",
        "runtime_dir": ".model-runtimes",
        "max_tokens": 32768,
        "mlx_cache_mib": 256,
        "log": "logs/requests.jsonl",
        "torch_dtype": "bfloat16",
        "sidecar_startup_seconds": 900,
    }))
    (tmp_path / "models.json").write_text(json.dumps({
        "models": [
            {"id": "big", "engine": "kev", "label": "Big", "backends": {"gguf": {"model": "owner/big", "runtime_key": "rt-big"}}},
            {"id": "small", "engine": "kev", "label": "Small", "backends": {"gguf": {"model": "owner/small", "runtime_key": "rt-small"}}},
        ]
    }))
    for key in ("rt-big", "rt-small"):
        python = model_manager._runtime_python(tmp_path / ".model-runtimes" / key)
        python.parent.mkdir(parents=True)
        python.write_text("")
    installations.mark_installed(config_path, "big", "gguf", verified=True, max_input_tokens=32768)
    installations.mark_installed(config_path, "small", "gguf", verified=True, max_input_tokens=4096)
    assert registry_path(config_path).is_file()

    from deqio.hardware import HostCapabilities

    host = HostCapabilities(
        system="Linux",
        machine="x86_64",
        backends=("gguf",),
        system_memory_gib=64.0,
        cuda_memory_gib=None,
        cuda_compute_capability=None,
    )
    monkeypatch.setattr(installations, "_cached_hf_state", lambda: {})
    monkeypatch.setattr(installations, "_artifact_ready", lambda *args, **kwargs: (True, False))
    monkeypatch.setattr(installations, "detect_host", lambda: host)
    monkeypatch.setattr(model_manager, "_cleanup_profile_artifacts", lambda **kwargs: None)

    args = argparse.Namespace(config=str(config_path), model_id="big", backend="gguf", yes=True, purge_cache=False)
    assert model_manager.cmd_delete(args) == 0

    persisted = json.loads(config_path.read_text())
    assert persisted["model_id"] == "small"
    assert persisted["max_tokens"] == 4096


# --- D8 / 0.5.4: one rejection path, one worker, honest token usage ----------


class _CountingWatch:
    """Watch spy: a fixed session plus every appended event."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def session_token(self):
        return "session-d8"

    def append(self, event, *, expected_session_id=None):
        self.events.append(event)
        return f"000001-{len(self.events):05d}-abcdef"


class _NoInferenceRuntime:
    """Any inference call means validation let a bad request through."""

    def _fail(self, *args, **kwargs):
        raise AssertionError("inference must not run for a rejected request")

    score = score_noul = score_shared = system_one = _fail


def _choice_body(**overrides) -> dict:
    body = {
        "state": "state",
        "question": "Choose",
        "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
    }
    body.update(overrides)
    return body


def _shared_body(decisions: list[dict]) -> dict:
    return {"state": "state", "decisions": decisions}


def _shared_decision(decision_id: str | None = None, question: str = "Choose", options=None) -> dict:
    decision = {
        "question": question,
        "options": options or [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
    }
    if decision_id is not None:
        decision["id"] = decision_id
    return decision


_D8_REJECTIONS = [
    # (case id, path, body, headers, expected status, inference blocked?)
    ("choice-no-options", "/v1/choice", _choice_body(options=[]), {}, 400),
    ("decision-alias-duplicate-options", "/v1/decision",
     _choice_body(options=[{"id": "x", "description": "A"}, {"id": "x", "description": "B"}]), {}, 400),
    ("choice-blank-question", "/v1/choice", _choice_body(question="   "), {}, 400),
    ("choice-blank-id", "/v1/choice", _choice_body(id="  "), {}, 400),
    ("noul-blank-question", "/v1/noul", {"state": "s", "question": " \t "}, {}, 400),
    ("noul-empty-id", "/v1/noul", {"state": "s", "question": "q", "id": ""}, {}, 400),
    ("noul-contract-without-policy", "/v1/noul", {"state": "s", "question": "q", "id": "n1"},
     {"deqio-contract": "input-completeness-v1"}, 422),
    ("shared-empty", "/v1/shared", _shared_body([]), {}, 400),
    ("shared-duplicate-decision-ids", "/v1/shared",
     _shared_body([_shared_decision("same"), _shared_decision("same")]), {}, 400),
    ("shared-duplicate-option-ids", "/v1/shared",
     _shared_body([_shared_decision(options=[{"id": "x", "description": "A"}, {"id": "x", "description": "B"}])]),
     {}, 400),
    ("shared-blank-decision-id", "/v1/shared", _shared_body([_shared_decision(" ")]), {}, 400),
    ("shared-blank-question", "/v1/shared", _shared_body([_shared_decision(question="")]), {}, 400),
    ("shared-contract-missing-decision-id", "/v1/shared",
     {**_shared_body([_shared_decision()]), "input_policy": {"require_complete": False}},
     {"deqio-contract": "input-completeness-v1"}, 422),
    ("score-empty-name", "/v1/score", {"state": "s", "name": " ", "question": {"instructions": "q"}}, {}, 422),
    ("act-conflicting-type", "/v1/act", {"state": "s", "question": {"type": "score"}}, {}, 422),
    ("multi-empty-question", "/v1/multi", {"state": "s", "question": {}}, {}, 422),
    ("systemone-no-questions", "/v1/systemone", {"state": "s", "questions": {}}, {}, 422),
    ("soam-no-questions", "/v1/soam", {"state": "s", "questions": {}}, {}, 422),
]


@pytest.mark.parametrize(
    ("path", "body", "headers", "expected_status"),
    [case[1:] for case in _D8_REJECTIONS],
    ids=[case[0] for case in _D8_REJECTIONS],
)
def test_every_rejection_reaching_a_decision_handler_is_accounted_exactly_once(
    path: str, body: dict, headers: dict, expected_status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4: one accounting path for every rejection before or during inference.

    Each failed call that reaches a decision handler adds exactly one to
    ``stats.errors``, writes exactly one console error line and exactly one
    Watch row, and never reaches the engine when the input is invalid.
    """
    import deqio.server as server

    watch = _CountingWatch()
    console: list[tuple] = []
    monkeypatch.setattr(server, "_watch_store", lambda: watch)
    monkeypatch.setattr(server, "runtime", _NoInferenceRuntime())
    monkeypatch.setattr(server, "log_request_error", lambda *a, **k: console.append((a, k)))
    before = server.stats["errors"]

    status, payload = _asgi_post_json(path, json.dumps(body), {"content-type": "application/json", **headers})

    assert status == expected_status, payload
    assert server.stats["errors"] - before == 1
    assert len(console) == 1
    assert len(watch.events) == 1
    assert watch.events[0]["endpoint"] == path
    assert watch.events[0]["status_code"] == expected_status


@pytest.mark.parametrize(
    "path",
    ["/v1/choice", "/v1/decision", "/v1/noul", "/v1/shared", "/v1/score", "/v1/multi", "/v1/act", "/v1/soam", "/v1/systemone"],
)
def test_runtime_failures_are_accounted_exactly_once(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    native_question = {"state": "s", "question": {"instructions": "q"}}
    native_questions = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "q"}}}
    bodies = {
        "/v1/choice": _choice_body(),
        "/v1/decision": _choice_body(),
        "/v1/noul": {"state": "s", "question": "q"},
        "/v1/shared": _shared_body([_shared_decision("d1")]),
        "/v1/score": native_question,
        "/v1/multi": native_question,
        "/v1/act": native_question,
        "/v1/soam": native_questions,
        "/v1/systemone": native_questions,
    }
    watch = _CountingWatch()
    console: list[tuple] = []
    monkeypatch.setattr(server, "_watch_store", lambda: watch)
    monkeypatch.setattr(server, "runtime", None)
    monkeypatch.setattr(server, "runtime_suspension", None)
    monkeypatch.setattr(server, "log_request_error", lambda *a, **k: console.append((a, k)))
    before = server.stats["errors"]

    status, _ = _asgi_post_json(path, json.dumps(bodies[path]), {"content-type": "application/json"})

    assert status == 503
    assert server.stats["errors"] - before == 1
    assert len(console) == 1
    assert len(watch.events) == 1


def test_portable_and_native_handlers_use_one_worker_thread_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B16: a portable request performs one executor hop, like the native path.

    Several hops per request (Watch session, inference, request log, Watch
    append) multiply the executor queueing a request can hit under fan-out.
    """
    import deqio.server as server

    hops: list[str] = []
    original_to_thread = asyncio.to_thread

    async def counting_to_thread(func, /, *args, **kwargs):
        hops.append(getattr(func, "__name__", repr(func)))
        return await original_to_thread(func, *args, **kwargs)

    identity = {"engine": "kev", "model_id": "demo", "backend": "mps", "runtime_instance_id": "rt"}
    monkeypatch.setattr(server.asyncio, "to_thread", counting_to_thread)
    monkeypatch.setattr(server, "_watch_store", lambda: _CountingWatch())
    monkeypatch.setattr(server, "record_event", lambda **kwargs: None)
    monkeypatch.setattr(
        server, "run_decision",
        lambda payload, *, endpoint, contract: server.format_result(_raw_decision(), runtime_identity=identity),
    )
    monkeypatch.setattr(
        server, "_score_noul_inference",
        lambda row, mode, contract, request_id: (server.SETTINGS, identity, _raw_decision(request_id)),
    )
    monkeypatch.setattr(
        server, "_score_shared_inference",
        lambda rows, contract, request_id: (
            server.SETTINGS, identity, [_raw_decision(str(row["id"])) for row in rows], {"total_seconds": 0.004},
        ),
    )
    headers = {"content-type": "application/json"}
    requests = [
        ("/v1/choice", _choice_body(), 200),
        ("/v1/noul", {"state": "s", "question": "q"}, 200),
        ("/v1/shared", _shared_body([_shared_decision("d1"), _shared_decision("d2")]), 200),
        ("/v1/choice", _choice_body(options=[]), 400),
        ("/v1/shared", _shared_body([]), 400),
    ]
    for path, body, expected in requests:
        hops.clear()
        status, _ = _asgi_post_json(path, json.dumps(body), headers)
        assert status == expected
        assert len(hops) == 1, (path, hops)


def test_input_limits_and_text_fields_are_validated_before_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """B15: bounded option/decision counts and no whitespace-only questions or IDs."""
    import deqio.server as server

    monkeypatch.setattr(server, "_watch_store", lambda: _CountingWatch())
    monkeypatch.setattr(server, "runtime", _NoInferenceRuntime())
    headers = {"content-type": "application/json"}

    too_many_options = [{"id": f"o{i}", "description": f"Option {i}"} for i in range(server.MAX_OPTIONS + 1)]
    status, payload = _asgi_post_json("/v1/choice", json.dumps(_choice_body(options=too_many_options)), headers)
    assert status == 400
    assert f"at most {server.MAX_OPTIONS}" in payload["detail"]

    too_many_decisions = [_shared_decision(f"d{i}") for i in range(server.MAX_DECISIONS + 1)]
    status, payload = _asgi_post_json("/v1/shared", json.dumps(_shared_body(too_many_decisions)), headers)
    assert status == 400
    assert f"at most {server.MAX_DECISIONS}" in payload["detail"]

    nested_options = _shared_body([_shared_decision("d1", options=too_many_options)])
    status, payload = _asgi_post_json("/v1/shared", json.dumps(nested_options), headers)
    assert status == 400
    assert "decisions[0].options" in payload["detail"]

    status, payload = _asgi_post_json("/v1/noul", json.dumps({"state": "s", "question": "   "}), headers)
    assert status == 400
    assert payload["detail"] == "question must not be empty or whitespace"

    status, payload = _asgi_post_json(
        "/v1/shared", json.dumps(_shared_body([_shared_decision("ok"), _shared_decision(" ")])), headers
    )
    assert status == 400
    assert payload["detail"] == "decisions[1].id must not be empty or whitespace"

    # Exactly at the limits is accepted (and therefore reaches inference).
    at_limit = [{"id": f"o{i}", "description": f"Option {i}"} for i in range(server.MAX_OPTIONS)]
    status, payload = _asgi_post_json("/v1/choice", json.dumps(_choice_body(options=at_limit)), headers)
    assert status == 500
    assert "inference must not run" in payload["detail"]


def _shared_engine_response(question_ids: list[str], *, input_tokens) -> dict:
    response = {
        "answers": {
            qid: {"type": "choice", "choice": "a", "probabilities": {"a": 0.75, "b": 0.25}}
            for qid in question_ids
        },
        "latency_ms": 9.0,
    }
    if input_tokens is not None:
        response["usage"] = {"input_tokens": input_tokens}
    return response


def _bare_systemone_runtime(response: dict):
    from deqio.systemone_runtime import SystemOneRuntime

    runtime_obj = SystemOneRuntime.__new__(SystemOneRuntime)
    runtime_obj.engine = "demo"
    runtime_obj._request = lambda state, questions, mode=None: ({"state": state, "questions": questions}, response, 9.0)
    return runtime_obj


def test_shared_runtime_reports_the_batch_token_count_once_not_per_decision() -> None:
    """B5: one engine call over a shared state has one measured token count."""
    rows = [
        {"id": f"d{i}", "state": "s", "question": "q", "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}]}
        for i in range(3)
    ]
    runtime_obj = _bare_systemone_runtime(_shared_engine_response(["q0", "q1", "q2"], input_tokens=120))

    raw_results, timing = runtime_obj.score_shared(rows)

    assert timing["batch_size"] == 3
    for raw in raw_results:
        assert raw["input_tokens"] is None
        assert raw["input_tokens_source"] == "engine_reported_batch"
        assert raw["batch_input_tokens"] == 120
        assert raw["batch_input_tokens_source"] == "engine_reported"

    # No usage from the engine: the batch is unknown and so is every decision.
    silent = _bare_systemone_runtime(_shared_engine_response(["q0", "q1"], input_tokens=None))
    silent_results, _ = silent.score_shared(rows[:2])
    assert [(r["input_tokens"], r["input_tokens_source"]) for r in silent_results] == [(None, "unknown")] * 2
    assert silent_results[0]["batch_input_tokens"] is None

    # With a single decision the batch measurement is the decision measurement.
    single = _bare_systemone_runtime(_shared_engine_response(["q0"], input_tokens=40))
    (only,), _ = single.score_shared(rows[:1])
    assert only["input_tokens"] == 40
    assert only["input_tokens_source"] == "engine_reported"


def test_shared_endpoint_records_batch_tokens_once_in_watch_log_and_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    rows_seen: list[list[dict]] = []

    class BatchRuntime:
        runtime_instance_id = "rt-batch"

        def score_shared(self, rows):
            rows_seen.append(rows)
            return [
                {
                    **_raw_decision(str(row["id"])),
                    "option_ids": ["a", "b"],
                    "input_tokens": None,
                    "input_tokens_source": "engine_reported_batch",
                    "batch_input_tokens": 120,
                    "batch_input_tokens_source": "engine_reported",
                }
                for row in rows
            ], {"total_seconds": 0.009, "batch_size": len(rows)}

    watch = _CountingWatch()
    logged: list[dict] = []
    monkeypatch.setattr(server, "_watch_store", lambda: watch)
    monkeypatch.setattr(server, "runtime", BatchRuntime())
    monkeypatch.setattr(server, "record_event", lambda **kwargs: logged.append(kwargs))

    body = {
        "state": "s",
        "decisions": [_shared_decision(f"d{i}") for i in range(3)],
        "input_policy": {"require_complete": False},
    }
    status, payload = _asgi_post_json(
        "/v1/shared", json.dumps(body),
        {"content-type": "application/json", "deqio-contract": "input-completeness-v1"},
    )

    assert status == 200, payload
    assert [result["input_tokens"] for result in payload["results"]] == [None, None, None]
    assert payload["input_receipt"]["usage"]["input_tokens"] == 120
    assert payload["input_receipt"]["usage"]["input_tokens_source"] == "engine_reported"
    for result in payload["results"]:
        assert result["input_receipt"]["usage"]["input_tokens"] is None
        assert result["input_receipt"]["usage"]["input_tokens_source"] == "engine_reported_batch"
    assert watch.events[-1]["input_tokens"] == 120
    assert logged[-1]["result"]["input_tokens"] == 120


def test_benchmark_shared_watch_row_counts_batch_tokens_once(tmp_path: Path) -> None:
    from deqio.benchmark import _run_case
    from deqio.watch_store import WatchStore

    class SharedRuntime:
        def identity_snapshot(self):
            return {"engine": "demo", "model_id": "demo-model", "backend": "mlx", "runtime_instance_id": "rt"}

        def score_shared(self, rows):
            return [
                {
                    **_raw_decision(str(row["id"])),
                    "option_ids": ["a", "b"],
                    "input_tokens": None,
                    "input_tokens_source": "unknown",
                    "batch_input_tokens": 90,
                    "batch_input_tokens_source": "engine_reported",
                }
                for row in rows
            ], {"total_seconds": 0.01, "batch_size": len(rows)}

    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    store = WatchStore(config_path)
    store.reset(reason="server-start")
    case = {
        "id": "shared-1",
        "type": "shared",
        "state": "s",
        "decisions": [
            {"id": f"d{i}", "question": "q", "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}], "expected": "a"}
            for i in range(3)
        ],
    }

    _run_case(SharedRuntime(), {"engine": "demo", "model_id": "demo-model", "backend": "mlx"}, case, watch=store)

    event = store.list_events(limit=1)["events"][0]
    assert event["input_tokens"] == 90


def test_zero_or_missing_engine_token_counts_are_unknown_not_measured() -> None:
    """B6: a prompt always has at least one token, so 0 means "not reported"."""
    from deqio.systemone_runtime import _input_token_usage

    assert _input_token_usage({"usage": {"input_tokens": 0}}) == (None, "unknown")
    assert _input_token_usage({"usage": {}}) == (None, "unknown")
    assert _input_token_usage({}) == (None, "unknown")
    assert _input_token_usage({"usage": {"input_tokens": 37}}) == (37, "engine_reported")


def test_sidecars_omit_input_tokens_they_did_not_measure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from deqio.jevk5_gguf_sidecar import _reported_input_tokens
    from deqio.semif_sidecar import _batch_input_tokens

    assert _batch_input_tokens([{}, {}]) is None
    assert _batch_input_tokens([{"input_tokens": 0}]) is None
    assert _batch_input_tokens([{"input_tokens": 30}]) == 30
    # Per-row counts of a prefix-sharing batch are not a batch measurement.
    assert _batch_input_tokens([{"input_tokens": 30}, {"input_tokens": 41}]) is None

    assert _reported_input_tokens([12, 30]) == 42
    assert _reported_input_tokens([12, None]) is None  # a partial sum is not a measurement
    assert _reported_input_tokens([0]) is None
    assert _reported_input_tokens([]) is None

    # Nimble does not expose token counts: its response must not invent one.
    package = tmp_path / "nimble" / "scoring"
    package.mkdir(parents=True)
    (tmp_path / "nimble" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "parallel_scorer.py").write_text(
        "class ParallelScorer:\n"
        "    def __init__(self, **config):\n"
        "        pass\n"
        "    def score(self, state, schema):\n"
        "        return {'output': {'q': True}, 'fields': {'q': {'scores': {'true': 0.9, 'false': 0.1}}}}\n",
        encoding="utf-8",
    )
    model_config = tmp_path / "model.json"
    model_config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in [name for name in sys.modules if name == "nimble" or name.startswith("nimble.")]:
        monkeypatch.delitem(sys.modules, name)

    import httpx

    from deqio.nimble_sidecar import create_app

    app_under_test = create_app(source_root=tmp_path, model_config=model_config, backend="mlx")

    async def call():
        transport = httpx.ASGITransport(app=app_under_test)
        async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
            return await client.post(
                "/v1/systemone",
                json={"state": "s", "questions": {"q": {"type": "noul", "instructions": "q"}}},
            )

    response = asyncio.run(call())
    for name in [name for name in sys.modules if name == "nimble" or name.startswith("nimble.")]:
        sys.modules.pop(name, None)

    assert response.status_code == 200
    assert response.json()["answers"]["q"]["noul"] == pytest.approx(0.9)
    assert "input_tokens" not in response.json().get("usage", {})


# --- D10 / 0.5.5: Watch token on non-loopback binds, native question limit --


_WATCH_TOKEN = "s3cret-watch-token"
_PROTECTED_WATCH_ROUTES = [
    ("GET", "/v1/watch"),
    ("GET", "/v1/watch/settings"),
    ("POST", "/v1/watch/settings"),
    ("GET", "/v1/watch/000001-00001-abcdef123456"),
    ("POST", "/v1/watch/clear"),
    ("GET", "/ui/watch"),
]


class _AsgiClient:
    """Minimal synchronous HTTP client over httpx.ASGITransport (no lifespan)."""

    def __init__(self, app) -> None:
        self.app = app

    def request(self, method: str, path: str, **kwargs):
        import httpx

        async def send():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path: str, **kwargs):
        return self.request("GET", path, **kwargs)


def _watch_client(monkeypatch: pytest.MonkeyPatch, token: str | None) -> _AsgiClient:
    import deqio.server as server

    monkeypatch.setattr(server, "watch_access_token", token, raising=False)
    return _AsgiClient(server.app)


def test_watch_routes_require_the_token_on_non_loopback_binds(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _watch_client(monkeypatch, _WATCH_TOKEN)
    for method, path in _PROTECTED_WATCH_ROUTES:
        body = {"auto_clear_minutes": 0} if method == "POST" else None
        assert client.request(method, path, json=body).status_code == 401, (method, path)
    assert client.get("/v1/watch", headers={"X-Deqio-Watch-Token": "wrong"}).status_code == 401
    assert client.get("/v1/watch", headers={"X-Deqio-Watch-Token": _WATCH_TOKEN}).status_code == 200
    assert client.get("/v1/watch/settings", headers={"Authorization": f"Bearer {_WATCH_TOKEN}"}).status_code == 200


def test_watch_token_is_exchanged_for_an_httponly_cookie_in_the_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _watch_client(monkeypatch, _WATCH_TOKEN)
    assert client.get("/ui/watch?token=wrong", follow_redirects=False).status_code == 401

    exchanged = client.get(f"/ui/watch?token={_WATCH_TOKEN}", follow_redirects=False)
    assert exchanged.status_code == 303
    assert exchanged.headers["location"] == "/ui/watch"
    cookie = exchanged.headers["set-cookie"].lower()
    assert "deqio_watch_token=" in cookie and "httponly" in cookie and "samesite=strict" in cookie

    browser_cookie = {"cookie": f"deqio_watch_token={_WATCH_TOKEN}"}
    assert client.get("/ui/watch", headers=browser_cookie).status_code == 200
    assert client.get("/v1/watch", headers=browser_cookie).status_code == 200
    assert client.get("/ui/watch").status_code == 401
    from_dashboard = client.get(f"/ui?token={_WATCH_TOKEN}", follow_redirects=False)
    assert from_dashboard.status_code == 303 and from_dashboard.headers["location"] == "/ui"


def test_loopback_binds_keep_watch_open_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _watch_client(monkeypatch, None)
    assert client.get("/v1/watch").status_code == 200
    assert client.get("/ui/watch").status_code == 200
    assert client.get("/ui/watch?token=anything", follow_redirects=False).status_code == 200


def test_decision_ui_and_health_stay_token_free_on_non_loopback_binds(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _watch_client(monkeypatch, _WATCH_TOKEN)
    assert client.get("/health").status_code == 200
    assert client.get("/ui").status_code == 200


@pytest.mark.parametrize(
    ("host", "configured", "expected"),
    [
        ("127.0.0.1", "configured-token", None),
        ("::1", None, None),
        ("localhost", None, None),
        ("0.0.0.0", "configured-token", "configured-token"),
        ("192.168.1.20", "configured-token", "configured-token"),
        ("::", None, "generated"),
    ],
)
def test_watch_token_is_required_exactly_for_non_loopback_binds(
    monkeypatch: pytest.MonkeyPatch, host: str, configured: str | None, expected: str | None
) -> None:
    import deqio.server as server

    class _Runtime:
        def close(self) -> None:
            return None

    monkeypatch.setenv("DEQIO_SERVER_HOST", host)
    monkeypatch.setenv("DEQIO_SERVER_PORT", "8787")
    monkeypatch.delenv("DEQIO_SERVER_CONTROL", raising=False)
    if configured is None:
        monkeypatch.delenv("DEQIO_WATCH_TOKEN", raising=False)
    else:
        monkeypatch.setenv("DEQIO_WATCH_TOKEN", configured)
    monkeypatch.setattr(server, "startup_header", lambda **kwargs: None)
    monkeypatch.setattr(server, "server_ready", lambda **kwargs: None)
    monkeypatch.setattr(server, "_reset_watch_session", lambda **kwargs: {})
    monkeypatch.setattr(server, "_load_warmed_runtime", lambda *args, **kwargs: _Runtime())
    monkeypatch.setattr(server, "_refresh_installed_registry", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "runtime", None)
    seen: dict[str, object] = {}

    async def invoke() -> None:
        async with server.lifespan(server.app):
            seen["token"] = getattr(server, "watch_access_token", None)

    asyncio.run(invoke())

    if expected == "generated":
        assert isinstance(seen["token"], str) and len(seen["token"]) >= 32
    else:
        assert seen["token"] == expected
    assert getattr(server, "watch_access_token", None) is None


def test_native_requests_reject_more_questions_than_the_limit_before_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deqio.server as server

    watch = _CountingWatch()
    monkeypatch.setattr(server, "_watch_store", lambda: watch)
    monkeypatch.setattr(server, "runtime", _NoInferenceRuntime())
    monkeypatch.setattr(server, "log_request_error", lambda *args, **kwargs: None)
    limit = getattr(server, "MAX_QUESTIONS", server.MAX_DECISIONS)
    questions = {f"q{index}": {"type": "noul", "question": "Is it?"} for index in range(limit + 1)}

    for path in ("/v1/systemone", "/v1/soam"):
        errors_before = server.stats["errors"]
        status, body = _asgi_post_json(
            path, json.dumps({"state": "s", "questions": questions}), {"content-type": "application/json"}
        )
        assert status == 422, (path, body)
        assert f"at most {limit}" in json.dumps(body)
        assert server.stats["errors"] == errors_before + 1
    assert len(watch.events) == 2
