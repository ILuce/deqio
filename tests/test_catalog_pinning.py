"""D12: the model catalog pins every runtime it installs.

Two installs of the same catalog must resolve the same engine code: every
runtime package is an exact PyPI version (`name==version`) or a GitHub commit
(`@<40-hex>`), never a branch, tag, range or bare name. Hugging Face artifacts
carry an immutable commit revision, except the explicit pending list below,
which may only shrink. Every profile says whether its runtime and weights are
`official` (published by the model's upstream) or `community` (a third-party
port), and launchers load the revision the catalog pins.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from deqio import model_manager as manager
from deqio.catalog import get_profile, load_catalog
from deqio.systemone_runtime import SystemOneRuntime
from deqio.workspace import ensure_workspace

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATHS = (ROOT / "models.json", ROOT / "src" / "deqio" / "data" / "models.json")

_NAME = r"[A-Za-z0-9][A-Za-z0-9_.\-]*(?:\[[A-Za-z0-9_,.\-]+\])?"
PINNED_SPEC = (
    re.compile(rf"^{_NAME}==[0-9][0-9A-Za-z.+\-]*$"),
    re.compile(rf"^{_NAME} @ git\+https://github\.com/[\w.\-]+/[\w.\-]+\.git@[0-9a-f]{{40}}$"),
    re.compile(rf"^{_NAME} @ https://github\.com/[\w.\-]+/[\w.\-]+/archive/[0-9a-f]{{40}}\.tar\.gz$"),
)
SHA40 = re.compile(r"^[0-9a-f]{40}$")

# Profiles whose runtime or weights are a third-party port rather than the
# model's upstream release. Reclassifying a profile is a deliberate edit here.
COMMUNITY_PROFILES = {
    "laya-english::mlx",
    "laya-multilingual::mlx",
    "laya-typed-decisions::mlx",
    "clef-flash::mlx",
    "clef::mlx",
}

# Hugging Face artifacts whose catalog revision is not yet an immutable commit
# SHA. Resolving them needs huggingface.co (owner-side, see HANDOFF.md) and, for
# engines that take no revision input, launcher support. The list may only
# shrink: a new unpinned artifact fails, and so does a pinned one left here.
HF_REVISION_PENDING = {
    # Kev: catalog uses the upstream HF release tag v1.0 (kev.serve --run repo@v1.0).
    **{
        f"kev-{size}::{backend}::jaredpalmer/kev-{size}": "v1.0"
        for size in ("0.8b", "4b", "9b")
        for backend in ("mlx", "cuda", "mps")
    },
    "kev-27b::cuda::jaredpalmer/kev-27b": "v1.0",
    # Pin-on-install profiles: `main` resolved to a commit at install/update time.
    "kev-0.8b::gguf::ggml-org/Kev-0.8B-GGUF": "main",
    "kev-4b::gguf::ggml-org/Kev-4B-GGUF": "main",
    "kev-9b::gguf::ggml-org/Kev-9B-GGUF": "main",
    "jevk5-4b::gguf::alibiserikbay/JevK5-GGUF": "main",
    "jevk5-9b::gguf::alibiserikbay/JevK5-GGUF": "main",
    "clef-flash::gguf::ggml-org/Clef-Flash-GGUF": "main",
    "clef::gguf::ggml-org/Clef-GGUF": "main",
    "laya-english::gguf::ggml-org/Laya-GGUF": "main",
    "decider-2b::gguf::Mapika/decider-2b-GGUF": "main",
    "decider-4b::gguf::Mapika/decider-4b-GGUF": "main",
    **{
        f"basal-1.5-{tier}::{backend}::Remek/basal-1.5-{repo}{suffix}": "main"
        for tier, repo in (("mini", "mini"), ("main", "4.5B"), ("max", "max"))
        for backend, suffix in (("mlx", "-MLX-8bit"), ("mps", ""), ("gguf", ""), ("cuda", ""))
    },
    **{
        f"basal-1.5-{tier}::gguf::Remek/basal-1.5-{repo}-GGUF": "main"
        for tier, repo in (("mini", "mini"), ("main", "4.5B"), ("max", "max"))
    },
    # Floating upstream weights; the engine CLI takes a repo id without a revision.
    "jevk5-4b::cuda::alibiserikbay/JevK5": "upstream-latest",
    "jevk5-9b::cuda::alibiserikbay/JevK5-9B": "upstream-latest",
    "open-jev-27b-v1.1::cuda::ZefanCai/Open-Jev-27B-v1.1": "upstream-latest",
    "clm-8b::cuda::Qwen/Qwen3-8B": "upstream-latest",
    "clm-8b::cuda::Contrastive-LM/CLM-v0.1-8B": "upstream-latest",
    **{
        f"decider-{size}::{backend}::Mapika/decider-{size}": None
        for size in ("0.8b", "2b", "4b")
        for backend in ("cuda", "mps")
    },
    "von::cuda::wfzyx/von": None,
    "von::mps::wfzyx/von": None,
}

LAYA_TORCH_REVISIONS = {
    # laya/revisions.py PINNED_REVISIONS at the pinned laya commit (v0.3.28).
    "laya-english": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    "laya-multilingual": "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
    "laya-typed-decisions": "1a793eb568e6718f15941d08f85432581df534e3",
}
LAYA_MLX_REVISIONS = {
    # laya-mlx v0.3.0 benchmarks/results/hub-publication.json (publisher record).
    "laya-english": "047678560251f28113ee8f5df4be82102c7bf336",
    "laya-multilingual": "ba40c87fcb357f1643d04d71323af9cdc3b9e591",
    "laya-typed-decisions": "28416e78cb26a239a4eabaa2e084904ec5e6cacb",
}


def _catalog() -> dict:
    return load_catalog(CATALOG_PATHS[1])


def _profiles():
    for model in _catalog()["models"]:
        for backend, profile in model["backends"].items():
            yield model, backend, profile


def _runtime_specs(profile: dict) -> list[str]:
    specs = [str(item) for item in profile.get("packages") or []]
    specs += [str(value) for key, value in sorted(profile.items()) if key.endswith("_package")]
    return specs


def _pinned(spec: str) -> bool:
    return any(pattern.match(spec) for pattern in PINNED_SPEC)


def _distribution(spec: str) -> str:
    return re.split(r"[\[ =<>!~@]", spec, maxsplit=1)[0].lower().replace("_", "-")


def _install_specs(command: list[str]) -> list[str]:
    """Package specs of one `uv pip install` command (flags and their values skipped)."""
    if command[:3] != ["uv", "pip", "install"]:
        return []
    specs, skip = [], False
    for token in command[3:]:
        if skip:
            skip = False
            continue
        if token in {"--python", "--index-url", "-r", "--extra-index-url"}:
            skip = True
            continue
        if token.startswith("-"):
            continue
        specs.append(token)
    return specs


def _hf_artifacts(profile: dict) -> list[tuple[str, object]]:
    downloads = manager._declared_downloads(profile)
    rows = [(str(item["repo_id"]), item.get("revision") or profile.get("model_revision")) for item in downloads]
    if profile.get("installer") == "nimble":
        rows.append((str(profile["repo_id"]), profile.get("model_revision")))
    elif not downloads and manager._looks_like_hf_repo(profile.get("model")):
        rows.append((str(profile["model"]), profile.get("model_revision")))
    return rows


def _lock_version(name: str) -> str:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return next(package["version"] for package in lock["package"] if package["name"] == name)


def test_packaged_catalog_and_workspace_catalog_are_identical() -> None:
    """Guard: `ensure_workspace` copies the packaged catalog; the repo copy must not drift."""
    assert CATALOG_PATHS[0].read_bytes() == CATALOG_PATHS[1].read_bytes()


def test_every_catalog_runtime_package_is_pinned_to_a_version_or_commit() -> None:
    unpinned = sorted(
        f"{model['id']}::{backend}: {spec}"
        for model, backend, profile in _profiles()
        for spec in _runtime_specs(profile)
        if not _pinned(spec)
    )
    assert unpinned == [], f"{len(unpinned)} unpinned runtime package specs:\n" + "\n".join(unpinned)


def test_installers_only_install_pinned_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every spec an installer passes to `uv pip install` (catalog or built-in) is pinned."""
    commands: list[list[str]] = []
    python = tmp_path / "python"
    monkeypatch.setattr(manager, "_ensure_venv", lambda *args, **kwargs: python)
    monkeypatch.setattr(manager, "_runtime_python", lambda env_dir: python)
    monkeypatch.setattr(manager, "_run", lambda command, **kwargs: commands.append(list(command)))
    monkeypatch.setattr(manager, "_checkout_nimble", lambda *args, **kwargs: tmp_path / "nimble-src")
    config = tmp_path / "config.json"
    data = {"runtime_dir": ".model-runtimes"}
    catalog = _catalog()

    for model_id, backend in (
        ("decision-2.0-kai", "cuda"),
        ("decider-2b", "gguf"),
        ("nimble-9b", "mlx"),
        ("nimble-9b", "cuda"),
        ("semif-qwen3.5-4b", "mlx"),
        ("laya-english", "mlx"),
        ("laya-english", "mps"),
        ("clef", "mlx"),
        ("von", "mps"),
        ("basal-1.5-main", "cuda"),
    ):
        profile = get_profile(catalog, model_id, backend)
        manager._install_runtime(config, {**data, "backend": backend}, profile, upgrade=True)

    # llama.cpp: the isolated build tools are installed first; the build itself
    # cannot run here, so stop at the missing binary.
    llama = get_profile(catalog, "kev-0.8b", "gguf")
    bin_dir = tmp_path / ".model-runtimes" / str(llama["runtime_key"]) / "bin"
    bin_dir.mkdir(parents=True)
    for tool in ("cmake", "ninja"):
        (bin_dir / tool).write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not produce llama-server"):
        manager._install_runtime(config, {**data, "backend": "gguf"}, llama, upgrade=True)

    monkeypatch.setattr(manager, "_install_llama_cpp_runtime", lambda *args, **kwargs: tmp_path / "jevk5-gguf")
    manager._install_runtime(config, {**data, "backend": "gguf"}, get_profile(catalog, "jevk5-4b", "gguf"), upgrade=True)

    installed = [spec for command in commands for spec in _install_specs(command)]
    assert installed, "no uv pip install command was captured"
    unpinned = sorted({spec for spec in installed if not _pinned(spec)})
    assert unpinned == [], "installers pass unpinned specs to uv pip install: " + ", ".join(unpinned)


def test_sidecar_transport_pins_match_the_core_lock() -> None:
    """Deqio's own sidecars run on FastAPI/Uvicorn in every engine runtime; CI tests the uv.lock versions."""
    expected = {name: _lock_version(name) for name in ("fastapi", "uvicorn")}
    specs = [spec for _, _, profile in _profiles() for spec in _runtime_specs(profile)]
    specs += [str(spec) for spec in getattr(manager, "SIDECAR_TRANSPORT_PACKAGES", ("fastapi>=0.110,<1",))]
    wrong = sorted(
        {spec for spec in specs if _distribution(spec) in expected and spec.split("==")[-1] != expected[_distribution(spec)]}
    )
    assert wrong == [], f"sidecar transport specs differ from uv.lock {expected}: {wrong}"


def test_every_hf_artifact_has_a_commit_revision_or_is_listed_pending() -> None:
    unpinned = {
        f"{model['id']}::{backend}::{repo}": revision
        for model, backend, profile in _profiles()
        for repo, revision in _hf_artifacts(profile)
        if not (isinstance(revision, str) and SHA40.match(revision))
    }
    assert unpinned == HF_REVISION_PENDING, (
        "unpinned HF artifacts differ from HF_REVISION_PENDING; "
        f"new: {sorted(set(unpinned) - set(HF_REVISION_PENDING))}, "
        f"resolved but still listed: {sorted(set(HF_REVISION_PENDING) - set(unpinned))}"
    )


def test_every_profile_declares_whether_it_is_official_or_community() -> None:
    sources = {f"{model['id']}::{backend}": profile.get("source") for model, backend, profile in _profiles()}
    missing = sorted(key for key, value in sources.items() if value not in {"official", "community"})
    assert missing == [], "profiles without source official|community: " + ", ".join(missing)
    assert {key for key, value in sources.items() if value == "community"} == COMMUNITY_PROFILES
    for model, backend, profile in _profiles():
        if profile.get("source") == "community":
            assert str(profile.get("source_url", "")).startswith("https://"), f"{model['id']}::{backend}"


def test_laya_mlx_is_a_pinned_community_profile() -> None:
    catalog = _catalog()
    for model_id, revision in LAYA_MLX_REVISIONS.items():
        profile = get_profile(catalog, model_id, "mlx")
        assert "laya-mlx==0.3.0" in profile["packages"]
        assert profile["model_revision"] == revision
        assert profile["source"] == "community"
        assert profile["source_url"] == "https://github.com/mizorewww/laya-mlx"


def test_profiles_sharing_a_runtime_agree_on_pinned_code() -> None:
    """Guard: one runtime directory holds one commit of each engine repository."""
    refs: dict[tuple[str, str], set[str]] = {}
    for _, _, profile in _profiles():
        runtime = str(profile.get("runtime_key"))
        for spec in _runtime_specs(profile):
            match = re.search(r"(git\+https://github\.com/[^@]+|https://github\.com/[\w.\-]+/[\w.\-]+/archive/)(.*)$", spec)
            if match:
                refs.setdefault((runtime, match.group(1)), set()).add(match.group(2))
        for key in ("llama_cpp_revision", "nimble_source_revision"):
            if profile.get(key):
                refs.setdefault((runtime, key), set()).add(str(profile[key]))
    conflicts = {key: value for key, value in refs.items() if len(value) > 1}
    assert conflicts == {}


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.mark.parametrize("existing_checkout", [False, True], ids=["fresh-install", "update-of-older-checkout"])
def test_nimble_source_checkout_is_the_catalog_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_checkout: bool
) -> None:
    """A real git upstream with a newer `main`: install and update must land on the pinned commit."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git("init", "-q", "-b", "main", cwd=upstream)
    _git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=upstream)
    for version in ("old", "pinned", "newer"):
        (upstream / "VERSION").write_text(version, encoding="utf-8")
        _git("add", "VERSION", cwd=upstream)
        _git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", version, cwd=upstream)
    pinned = _git("rev-parse", "HEAD~1", cwd=upstream)

    def run(command: list[str], **kwargs) -> None:
        if command and command[0] == "git":
            command = [part.replace("https://github.com/bespokelabsai/nimble.git", upstream.as_uri()) for part in command]
            subprocess.run(command, check=True, capture_output=True)

    monkeypatch.setattr(manager, "_run", run)
    monkeypatch.setattr(manager, "_ensure_venv", lambda *args, **kwargs: tmp_path / "python")
    config = ensure_workspace(tmp_path / "ws")
    source = config.parent / ".model-runtimes" / "nimble-src"
    if existing_checkout:
        # A checkout made before the pin existed: branch main, behind upstream.
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", upstream.as_uri(), str(source)], check=True)
        _git("reset", "-q", "--hard", "HEAD~2", cwd=source)
    profile = {**get_profile(_catalog(), "nimble-9b", "mlx"), "nimble_source_revision": pinned}

    manager._install_nimble(config, {"runtime_dir": ".model-runtimes", "backend": "mlx"}, profile, upgrade=existing_checkout)

    assert _git("rev-parse", "HEAD", cwd=source) == pinned
    assert (source / "VERSION").read_text(encoding="utf-8") == "pinned"


def test_nimble_profiles_pin_the_source_checkout() -> None:
    catalog = _catalog()
    for backend in ("mlx", "cuda"):
        profile = get_profile(catalog, "nimble-9b", backend)
        assert profile.get("nimble_source_revision") == "dcfdbd9a64f0d869f658d7a72f1beaee32737773"


def _launcher_settings(config_path: Path, **overrides) -> SimpleNamespace:
    values = {
        "engine": "laya", "backend": "mps", "model_id": "laya-english", "model": "convaiinnovations/laya",
        "model_revision": "", "max_tokens": 4096, "mlx_cache_mib": 256, "torch_dtype": "bfloat16",
        "hf_offline_runtime": True, "config_path": config_path, "runtime_dir": config_path.parent / ".model-runtimes",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("model_id", sorted(LAYA_TORCH_REVISIONS))
@pytest.mark.parametrize("backend", ["mps", "cuda"])
def test_laya_torch_launcher_loads_the_catalog_revision(tmp_path: Path, model_id: str, backend: str) -> None:
    config = ensure_workspace(tmp_path / "ws")
    profile = get_profile(_catalog(), model_id, backend)
    env: dict[str, str] = {}
    settings = _launcher_settings(config, backend=backend, model_id=model_id, model=profile["model"])

    SystemOneRuntime._command(settings, profile, tmp_path / "env", tmp_path / "env" / "bin" / "python", 9000, env)

    assert env.get("LAYA_REVISION") == LAYA_TORCH_REVISIONS[model_id]
    assert profile.get("model_revision") == LAYA_TORCH_REVISIONS[model_id]


@pytest.mark.parametrize("model_id", sorted(LAYA_MLX_REVISIONS))
def test_laya_mlx_launcher_passes_the_catalog_revision(tmp_path: Path, model_id: str) -> None:
    config = ensure_workspace(tmp_path / "ws")
    profile = get_profile(_catalog(), model_id, "mlx")
    settings = _launcher_settings(config, backend="mlx", model_id=model_id, model=profile["model"])

    command = SystemOneRuntime._command(settings, profile, tmp_path / "env", tmp_path / "env" / "bin" / "python", 9000, {})

    assert "--revision" in command, command
    assert command[command.index("--revision") + 1] == LAYA_MLX_REVISIONS[model_id]


def test_laya_mlx_sidecar_loads_the_requested_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []

    class Agent:
        def predict(self, state, questions):
            return {"answers": {key: {"answer": "yes"} for key in questions}}

    fake = types.ModuleType("laya_mlx")
    fake.load = lambda model_id, **kwargs: calls.append((model_id, kwargs)) or Agent()
    monkeypatch.setitem(sys.modules, "laya_mlx", fake)
    import uvicorn

    from deqio import laya_mlx_sidecar

    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.setdefault("app", app))
    revision = LAYA_MLX_REVISIONS["laya-english"]
    monkeypatch.setattr(sys, "argv", ["laya_mlx_sidecar", "--model", "aac6fef/laya-mlx", "--revision", revision, "--port", "9"])

    laya_mlx_sidecar.main()

    async def call() -> httpx.Response:
        transport = httpx.ASGITransport(app=captured["app"])
        async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
            return await client.post("/v1/systemone", json={"state": "s", "questions": {"q": {"type": "noul"}}})

    import asyncio

    response = asyncio.run(call())
    assert response.status_code == 200, response.text
    assert calls == [("aac6fef/laya-mlx", {"revision": revision})]
    assert json.loads(response.text)["model"] == "aac6fef/laya-mlx"
