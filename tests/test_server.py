import json
from pathlib import Path

import pytest

from semif_server.config import load_settings
from semif_server.server import app, format_result, format_shared_timing, percentile, state_hash
from semif_server.ui import DASHBOARD


def test_public_routes_are_registered() -> None:
    routes = {route.path for route in app.routes}

    assert "/health" in routes
    assert "/ui" in routes
    assert "/v1/stats" in routes
    assert "/v1/recent" in routes
    assert "/v1/noul" in routes
    assert "/v1/choice" in routes
    assert "/v1/decision" in routes
    assert "/v1/shared" in routes
    assert "/v1/cache/clear" in routes


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
        "SEMIF_BACKEND",
        "SEMIF_MODEL",
        "SEMIF_MODEL_REVISION",
        "SEMIF_MAX_TOKENS",
        "SEMIF_MLX_CACHE_MIB",
        "SEMIF_LOG",
        "SEMIF_TORCH_DTYPE",
        "SEMIF_GGUF",
        "SEMIF_LLAMA_THREADS",
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
                "llama_gguf": "models/test.gguf",
                "llama_threads": None,
            }
        )
    )

    settings = load_settings(config_path)

    assert settings.backend == "mlx"
    assert settings.model == str((tmp_path / "models/mlx").resolve())
    assert settings.log_path == (tmp_path / "logs/requests.jsonl").resolve()
    assert settings.llama_gguf == (tmp_path / "models/test.gguf").resolve()
    assert settings.max_tokens == 4096
    assert settings.mlx_cache_mib == 256
    assert settings.llama_threads is None


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
                "llama_gguf": "models/test.gguf",
                "llama_threads": None,
            }
        )
    )

    monkeypatch.setenv("SEMIF_BACKEND", "cuda")
    monkeypatch.setenv("SEMIF_MODEL", "Qwen/Qwen3.5-4B")
    monkeypatch.setenv("SEMIF_MAX_TOKENS", "8192")
    monkeypatch.setenv("SEMIF_LLAMA_THREADS", "8")

    settings = load_settings(config_path)

    assert settings.backend == "cuda"
    assert settings.model == "Qwen/Qwen3.5-4B"
    assert settings.max_tokens == 8192
    assert settings.llama_threads == 8


def test_ui_contains_all_playground_modes_and_cache_action() -> None:
    assert 'data-endpoint="noul"' in DASHBOARD
    assert 'data-endpoint="choice"' in DASHBOARD
    assert 'data-endpoint="shared"' in DASHBOARD
    assert "/v1/cache/clear" in DASHBOARD
    assert "Generated request JSON" in DASHBOARD
