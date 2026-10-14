"""0.6.1: remote decision APIs are catalog profiles on the ``remote`` backend.

``deqio models setup`` offers ``remote`` next to mlx/mps/gguf/cuda; the two profiles behind it
(``openai-decisions``, ``typesafe-jev``) are installed like a local model: model identifier, API
key (stored in the OS keychain through ``keyring``, never in the workspace), input-token budget,
one typed-decision probe against the real endpoint, then the registry record.  Serving, model
switching, benchmarking and deletion treat the profile exactly like a local one.  The D11
mechanisms this replaces (``provider`` in config.json, ``DEQIO_PROVIDER*``, provider flags,
``deqio benchmark --provider``) are gone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deqio import benchmark, config as config_module, hardware
from deqio.catalog import SUPPORTED_BACKENDS, SUPPORTED_ENGINES, get_profile, load_catalog
from deqio.installations import installation_record, installed_profiles, mark_installed
from deqio.workspace import ensure_workspace

from _remote_fakes import FakeKeyring, FakeOpenAIDecisionsAPI, FakeTypeSafeAPI

try:
    from deqio import credentials
except ImportError as error:  # a Deqio without the keychain layer: the credential tests fail on it
    credentials = None  # type: ignore[assignment]
    CREDENTIALS_IMPORT_ERROR: Exception | None = error
else:
    CREDENTIALS_IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parents[1]
REMOTE = {"openai-decisions": "gpt-6-luna", "typesafe-jev": "jev-latest"}


@pytest.fixture()
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> FakeKeyring:
    if CREDENTIALS_IMPORT_ERROR is not None:
        pytest.fail(f"deqio.credentials is missing: {CREDENTIALS_IMPORT_ERROR}")
    fake = FakeKeyring()
    monkeypatch.setattr(credentials, "_keyring", lambda: fake)
    return fake


def _workspace(tmp_path: Path, endpoints: dict[str, str] | None = None) -> Path:
    """A workspace whose catalog copy points the remote profiles at fake endpoints."""
    config = ensure_workspace(tmp_path / "ws")
    catalog_path = config.parent / "models.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    for entry in catalog["models"]:
        profile = (entry.get("backends") or {}).get("remote")
        if isinstance(profile, dict) and endpoints and entry["id"] in endpoints:
            profile["endpoint"] = endpoints[entry["id"]]
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return config


def _args(**values):
    return type("Args", (), values)()


# --- catalog and host -------------------------------------------------------------------


def test_remote_is_a_supported_backend_with_one_catalog_profile_per_provider() -> None:
    assert "remote" in SUPPORTED_BACKENDS
    assert {"openai-decisions", "typesafe-jev"} <= set(SUPPORTED_ENGINES)
    catalog = load_catalog(ROOT / "models.json")
    for name, default_model in REMOTE.items():
        profile = get_profile(catalog, name, "remote")
        entry = next(item for item in catalog["models"] if item["id"] == name)
        assert entry["engine"] == name and set(entry["backends"]) == {"remote"}
        assert profile["model"] == default_model and profile["wire_model"] == default_model
        assert profile["provider"] == name and profile["endpoint"].startswith("https://")
        assert profile["platform"] == "remote" and profile["source"] == "official"
        assert "runtime_key" not in profile and "packages" not in profile
        assert profile["capabilities"]["systemone"] is True and profile["capabilities"]["choice"] is True
        assert profile["capabilities"]["noul"] is True and profile["capabilities"]["score"] is True
        assert profile["capabilities"]["evidence"] is False and profile["capabilities"]["multi"] is False
        assert profile["usd_per_million_input_tokens"] > 0 and profile["default_max_input_tokens"] == 4096
    typesafe = get_profile(catalog, "typesafe-jev", "remote")
    assert typesafe["max_input_tokens"] == 32768 and typesafe["endpoint"] == "https://api.typesafe.ai/v1/systemone"
    assert get_profile(catalog, "openai-decisions", "remote")["endpoint"] == "https://api.openai.com/v1/decisions"
    packaged = load_catalog(ROOT / "src" / "deqio" / "data" / "models.json")
    assert packaged == catalog


def test_host_offers_remote_on_every_supported_system(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    assert hardware.host_backends() == ("mlx", "mps", "gguf", "remote")
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(hardware, "_nvidia_gpu_listed", lambda: False)
    assert hardware.host_backends() == ("gguf", "remote")
    monkeypatch.setattr(hardware, "_nvidia_gpu_listed", lambda: True)
    assert hardware.host_backends() == ("cuda", "gguf", "remote")


def test_remote_profile_is_compatible_without_a_memory_guardrail() -> None:
    host = hardware.HostCapabilities("Linux", "x86_64", ("gguf", "remote"), 2.0, None)
    profile = get_profile(load_catalog(ROOT / "models.json"), "openai-decisions", "remote")
    check = hardware.profile_compatibility("remote", profile, host=host)
    assert check["compatible"] is True and check["available_memory_gib"] is None and check["minimum_memory_gib"] is None


# --- keychain -----------------------------------------------------------------------------


def test_api_keys_live_in_the_os_keychain_under_the_deqio_service(fake_keyring: FakeKeyring) -> None:
    credentials.store_api_key("openai-decisions", "sk-test")
    assert fake_keyring.store == {("deqio", "openai-decisions"): "sk-test"}
    assert credentials.load_api_key("openai-decisions") == "sk-test"
    assert credentials.load_api_key("typesafe-jev") is None
    assert credentials.delete_api_key("openai-decisions") is True
    assert credentials.delete_api_key("openai-decisions") is False
    assert credentials.load_api_key("openai-decisions") is None


def test_missing_keychain_backend_is_an_explicit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    if CREDENTIALS_IMPORT_ERROR is not None:
        pytest.fail(f"deqio.credentials is missing: {CREDENTIALS_IMPORT_ERROR}")
    monkeypatch.setattr(credentials, "_keyring", lambda: FakeKeyring(available=False))
    with pytest.raises(credentials.KeychainUnavailableError, match="keychain"):
        credentials.store_api_key("openai-decisions", "sk-test")
    with pytest.raises(credentials.KeychainUnavailableError, match="keychain"):
        credentials.load_api_key("openai-decisions")


# --- models setup -------------------------------------------------------------------------


def _run_setup(config: Path, monkeypatch: pytest.MonkeyPatch, *, answers: list[str], api_key: str) -> int:
    import deqio.model_manager as manager

    monkeypatch.setattr(manager, "host_backends", lambda: ("gguf", "remote"))
    monkeypatch.setattr(manager, "detect_host", lambda: hardware.HostCapabilities("Linux", "x86_64", ("gguf", "remote"), 8.0, None))
    scripted = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(scripted))
    monkeypatch.setattr(manager.getpass, "getpass", lambda prompt="": api_key)
    return manager.main(["--config", str(config), "setup"])


def test_models_setup_installs_a_remote_profile_with_model_key_tokens_and_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_keyring: FakeKeyring, capsys: pytest.CaptureFixture[str]
) -> None:
    with FakeOpenAIDecisionsAPI() as api:
        config = _workspace(tmp_path, {"openai-decisions": api.endpoint})
        # backend "remote" is the 2nd entry of ("gguf", "remote"); the provider list is sorted by
        # catalog order (openai-decisions first); Enter keeps the default model id; token budget 2 = 8192.
        code = _run_setup(config, monkeypatch, answers=["2", "1", "", "2"], api_key="sk-live-test")
    out = capsys.readouterr().out
    assert code == 0, out
    assert "remote (2 compatible models)" in out
    assert "openai-decisions" in out and "typesafe-jev" in out
    assert "sk-live-test" not in out
    assert "Installed and verified profile: openai-decisions / remote" in out

    # exactly one typed-decision probe went to the endpoint, in the documented wire, with the key
    assert len(api.requests) == 1
    probe = api.requests[0]
    assert probe["headers"]["Authorization"] == "Bearer sk-live-test"
    assert probe["body"]["model"] == "gpt-6-luna" and probe["body"]["questions"][0]["type"] == "choice"

    record = installation_record(config, "openai-decisions", "remote")
    assert record is not None and record["verified_at"] and record["max_input_tokens"] == 8192
    assert record["remote_model"] == "gpt-6-luna" and record["api_key_store"] == "keychain"
    assert record["endpoint"] == api.endpoint
    assert fake_keyring.store == {("deqio", "openai-decisions"): "sk-live-test"}
    assert "sk-live-test" not in (config.parent / ".deqio" / "installed-models.json").read_text(encoding="utf-8")

    data = json.loads(config.read_text(encoding="utf-8"))
    assert (data["engine"], data["model_id"], data["backend"], data["model"], data["max_tokens"]) == (
        "openai-decisions", "openai-decisions", "remote", "gpt-6-luna", 8192
    )
    assert "sk-live-test" not in config.read_text(encoding="utf-8")
    assert "provider" not in data


def test_models_setup_stores_the_typed_model_identifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_keyring: FakeKeyring, capsys: pytest.CaptureFixture[str]
) -> None:
    with FakeTypeSafeAPI() as api:
        config = _workspace(tmp_path, {"typesafe-jev": api.endpoint})
        code = _run_setup(config, monkeypatch, answers=["2", "2", "jev-1.13.0", "1"], api_key="ts-key")
    assert code == 0, capsys.readouterr().out
    assert api.requests[0]["body"]["model"] == "jev-1.13.0" and api.requests[0]["path"] == "/v1/systemone"
    record = installation_record(config, "typesafe-jev", "remote")
    assert record["remote_model"] == "jev-1.13.0" and record["max_input_tokens"] == 4096
    assert json.loads(config.read_text(encoding="utf-8"))["model"] == "jev-1.13.0"
    assert fake_keyring.store[("deqio", "typesafe-jev")] == "ts-key"


def test_models_setup_refuses_a_remote_profile_whose_probe_fails_and_stores_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_keyring: FakeKeyring, capsys: pytest.CaptureFixture[str]
) -> None:
    with FakeOpenAIDecisionsAPI(status=401, body={"error": {"message": "Incorrect API key provided"}}) as api:
        config = _workspace(tmp_path, {"openai-decisions": api.endpoint})
        code = _run_setup(config, monkeypatch, answers=["2", "1", "", "1"], api_key="sk-bad")
    captured = capsys.readouterr()
    assert code == 2
    assert "HTTP 401" in captured.err
    assert installation_record(config, "openai-decisions", "remote") is None
    assert fake_keyring.store == {}
    assert json.loads(config.read_text(encoding="utf-8"))["backend"] != "remote"


def test_models_setup_offers_q_at_every_remote_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_keyring: FakeKeyring, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _workspace(tmp_path)
    assert _run_setup(config, monkeypatch, answers=["2", "1", "q"], api_key="unused") == 0
    assert "Cancelled." in capsys.readouterr().out
    assert fake_keyring.store == {} and installation_record(config, "openai-decisions", "remote") is None


# --- runtime, installed listing, switching, deletion ---------------------------------------


def _installed_remote(config: Path, name: str, *, remote_model: str, endpoint: str, key: str | None, fake_keyring: FakeKeyring) -> None:
    import deqio.model_manager as manager
    from deqio.catalog import get_model

    if key is not None:
        fake_keyring.store[("deqio", name)] = key
    mark_installed(
        config, name, "remote", verified=True, source="install", artifacts=[], max_input_tokens=8192,
        remote={"remote_model": remote_model, "endpoint": endpoint, "api_key_store": "keychain"},
    )
    catalog = load_catalog(config.parent / "models.json")
    data = manager._read_config(config)
    manager._write_config(
        config,
        manager._select_data(data, get_model(catalog, name), get_profile(catalog, name, "remote"), "remote",
                             max_input_tokens=8192, remote_model=remote_model),
    )


def test_backend_runtime_loads_a_remote_profile_from_config_and_keychain(
    tmp_path: Path, fake_keyring: FakeKeyring
) -> None:
    from deqio.backends import BackendRuntime
    from deqio.config import load_settings
    from deqio.remote_runtime import RemoteProviderRuntime

    with FakeOpenAIDecisionsAPI() as api:
        config = _workspace(tmp_path, {"openai-decisions": api.endpoint})
        _installed_remote(config, "openai-decisions", remote_model="gpt-6-luna", endpoint=api.endpoint, key="sk-1", fake_keyring=fake_keyring)
        settings = load_settings(config)
        assert settings.backend == "remote" and settings.engine == "openai-decisions" and settings.model == "gpt-6-luna"
        assert not hasattr(settings, "provider")
        runtime = BackendRuntime.load(settings)
        assert isinstance(runtime, RemoteProviderRuntime)
        identity = runtime.identity_snapshot()
        assert identity["model_id"] == "openai-decisions" and identity["model"] == "gpt-6-luna" and identity["backend"] == "remote"
        assert identity["max_input_tokens"] == 8192 and identity["installation_verified_at"]
        assert identity["provider"]["endpoint"] == api.endpoint and "sk-1" not in json.dumps(identity)
        raw = runtime.score({"id": "r", "state": "s", "question": "q?", "options": [{"id": "a"}, {"id": "b"}]}, "serial")
        assert raw["provider"]["name"] == "openai-decisions" and api.requests[0]["headers"]["Authorization"] == "Bearer sk-1"
        runtime.close()


def test_remote_runtime_refuses_a_config_mirror_that_differs_from_the_registry(
    tmp_path: Path, fake_keyring: FakeKeyring
) -> None:
    from deqio.backends import BackendRuntime
    from deqio.config import load_settings

    with FakeOpenAIDecisionsAPI() as api:
        config = _workspace(tmp_path, {"openai-decisions": api.endpoint})
        _installed_remote(config, "openai-decisions", remote_model="gpt-6-luna", endpoint=api.endpoint, key="sk-1", fake_keyring=fake_keyring)
        data = json.loads(config.read_text(encoding="utf-8"))
        data["model"] = "gpt-7-nova"
        config.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(RuntimeError, match="gpt-7-nova"):
            BackendRuntime.load(load_settings(config))
        assert api.requests == []


def test_remote_runtime_refuses_without_a_stored_key_instead_of_reading_the_environment(
    tmp_path: Path, fake_keyring: FakeKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deqio.backends import BackendRuntime
    from deqio.config import load_settings

    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    with FakeOpenAIDecisionsAPI() as api:
        config = _workspace(tmp_path, {"openai-decisions": api.endpoint})
        _installed_remote(config, "openai-decisions", remote_model="gpt-6-luna", endpoint=api.endpoint, key=None, fake_keyring=fake_keyring)
        with pytest.raises(RuntimeError, match="deqio models setup"):
            BackendRuntime.load(load_settings(config))
        assert api.requests == []


def test_installed_listing_shows_a_remote_profile_without_a_runtime_directory(tmp_path: Path, fake_keyring: FakeKeyring) -> None:
    config = _workspace(tmp_path)
    _installed_remote(config, "typesafe-jev", remote_model="jev-1.13.0", endpoint="https://api.typesafe.ai/v1/systemone", key="k", fake_keyring=fake_keyring)
    rows = installed_profiles(
        config_path=config, config_data=json.loads(config.read_text(encoding="utf-8")),
        catalog=load_catalog(config.parent / "models.json"), active_model_id="typesafe-jev", active_backend="remote",
    )
    row = next(item for item in rows if item["model_id"] == "typesafe-jev")
    assert row["installed"] and row["verified"] and row["status"] == "active" and row["backend"] == "remote"
    assert row["max_input_tokens"] == 8192 and row["host_compatible"] is True
    assert not (config.parent / ".model-runtimes").exists()


def test_models_use_restores_the_remote_model_and_token_budget(
    tmp_path: Path, fake_keyring: FakeKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.model_manager as manager

    config = _workspace(tmp_path)
    _installed_remote(config, "typesafe-jev", remote_model="jev-1.13.0", endpoint="https://api.typesafe.ai/v1/systemone", key="k", fake_keyring=fake_keyring)
    data = json.loads(config.read_text(encoding="utf-8"))
    data.update({"engine": "kev", "model_id": "kev-0.8b", "backend": "mps", "model": "jaredpalmer/kev-0.8b", "max_tokens": 4096})
    config.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(manager, "detect_host", lambda: hardware.HostCapabilities("Linux", "x86_64", ("gguf", "remote"), 8.0, None))
    assert manager.main(["--config", str(config), "use", "typesafe-jev", "--backend", "remote"]) == 0
    data = json.loads(config.read_text(encoding="utf-8"))
    assert (data["engine"], data["model_id"], data["backend"], data["model"], data["max_tokens"]) == (
        "typesafe-jev", "typesafe-jev", "remote", "jev-1.13.0", 8192
    )


def test_models_delete_removes_the_key_from_the_keychain_and_unregisters(
    tmp_path: Path, fake_keyring: FakeKeyring, capsys: pytest.CaptureFixture[str]
) -> None:
    import deqio.model_manager as manager

    config = _workspace(tmp_path)
    _installed_remote(config, "openai-decisions", remote_model="gpt-6-luna", endpoint="https://api.openai.com/v1/decisions", key="sk-1", fake_keyring=fake_keyring)
    assert manager.main(["--config", str(config), "delete", "openai-decisions", "--backend", "remote", "--yes"]) == 0
    assert installation_record(config, "openai-decisions", "remote") is None
    assert fake_keyring.store == {}
    assert "keychain" in capsys.readouterr().out.lower()


def test_activate_request_accepts_the_remote_backend() -> None:
    from deqio.server import ModelActivateRequest

    assert ModelActivateRequest(model_id="openai-decisions", backend="remote").backend == "remote"


# --- the D11 mechanisms replaced by the backend are gone ------------------------------------


def test_d11_provider_configuration_is_removed() -> None:
    assert not any(name.startswith("DEQIO_PROVIDER") for name in config_module.ENV_OVERRIDES)
    assert "provider" not in {field.name for field in config_module.Settings.__dataclass_fields__.values()}
    parser = benchmark.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--provider", "openai-decisions"])
    assert not hasattr(benchmark, "_provider_row")
    source = (ROOT / "src" / "deqio" / "providers.py").read_text(encoding="utf-8")
    assert "DEQIO_PROVIDER_" not in source and "flag_env" not in source
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "DEQIO_PROVIDER" not in agents and "remote" in agents
