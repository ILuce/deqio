from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir, snapshot_download

from .catalog import SUPPORTED_BACKENDS, apply_selection, get_model, get_profile, load_catalog
from .backends import BackendRuntime
from .config import read_config_data, settings_from_data, write_config_data
from .hardware import describe_host, detect_host, host_backends, profile_compatibility
from .installations import installed_profiles, installation_record, load_registry, mark_installed, profile_key, unmark_installed
from .process_lock import process_lock
from .runtime_control import discover_server
from .workspace import ensure_workspace


ROOT = Path.cwd()


TOKEN_BUDGET_PRESETS = (4096, 8192, 12288, 16384, 32768)


def _prompt_index(label: str, count: int) -> int | None:
    value = input(f"{label} [1-{count}, q]: ").strip().lower()
    if value in {"q", "quit", "exit"}:
        return None
    try:
        index = int(value) - 1
    except ValueError as error:
        raise RuntimeError("Invalid selection; enter a number or q to quit") from error
    if index not in range(count):
        raise RuntimeError("Invalid selection")
    return index


def _cancelled() -> int:
    print("Cancelled.")
    return 0


def _config_path(value: str | None) -> Path:
    configured = value or os.environ.get("DEQIO_CONFIG")
    if configured is not None:
        return Path(configured).expanduser().resolve()
    return ensure_workspace(Path.cwd())


def _read_config(path: Path) -> dict[str, Any]:
    _, data = read_config_data(path)
    return data


def _catalog_path(config_path: Path, data: dict[str, Any]) -> Path:
    value = str(data.get("model_catalog", "models.json"))
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _runtime_root(config_path: Path, data: dict[str, Any]) -> Path:
    value = str(data.get("runtime_dir", ".model-runtimes"))
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _runtime_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True, env=env)


def _write_config(path: Path, data: dict[str, Any]) -> None:
    write_config_data(path, data)


def _select_data(
    data: dict[str, Any],
    model_entry: dict[str, Any],
    profile: dict[str, Any],
    backend: str,
    *,
    max_input_tokens: int | None = None,
) -> dict[str, Any]:
    updated = apply_selection(data, model_entry, profile, backend)
    if max_input_tokens is not None:
        updated["max_tokens"] = int(max_input_tokens)
    return updated


def _profile_input_token_limit(profile: dict[str, Any]) -> int | None:
    """Return a catalogued hard token limit when this runtime exposes one.

    The generic ``max_input_tokens`` field is preferred. Legacy engine-specific
    limits remain honored so interactive setup cannot select a value that the
    sidecar command would immediately reject or silently reinterpret.
    """

    for key in ("max_input_tokens", "open_jev_max_length", "clm_max_tokens"):
        raw = profile.get(key)
        if raw is not None:
            value = int(raw)
            if value > 0:
                return value
    return None


def _default_max_input_tokens(
    config_path: Path,
    data: dict[str, Any],
    model_id: str,
    backend: str,
    profile: dict[str, Any],
) -> int:
    record = installation_record(config_path, model_id, backend) or {}
    if record.get("max_input_tokens") is not None:
        value = max(1, int(record["max_input_tokens"]))
    elif profile.get("default_max_input_tokens") is not None:
        value = max(1, int(profile["default_max_input_tokens"]))
    else:
        value = max(1, int(data.get("max_tokens", 4096)))
    hard_limit = _profile_input_token_limit(profile)
    return min(value, hard_limit) if hard_limit is not None else value


def _estimated_token_budget_memory_gib(
    profile: dict[str, Any],
    compatibility: dict[str, Any],
    value: int,
) -> float | None:
    """Conservative install-time memory guardrail for larger context budgets.

    This is deliberately only a provisioning estimate, not a statement about
    the model's effective attention/context limit. The exact completeness
    contract remains fail-closed unless a runtime instruments the model input.
    """

    minimum_raw = compatibility.get("minimum_memory_gib", profile.get("min_memory_gib"))
    if minimum_raw is None:
        return None
    minimum = float(minimum_raw)
    if value <= 4096:
        return minimum
    # KV/cache implementations differ by engine. Reserve at least 0.5 GiB for
    # every additional 4K tokens, scaling the reserve for larger models.
    per_extra_4k = max(0.5, minimum * 0.08)
    extra_4k = (float(value) - 4096.0) / 4096.0
    return minimum + (extra_4k * per_extra_4k)


def _token_budget_unavailable_reason(
    profile: dict[str, Any],
    compatibility: dict[str, Any],
    value: int,
) -> str | None:
    hard_limit = _profile_input_token_limit(profile)
    if hard_limit is not None and value > hard_limit:
        return f"profile limit is {hard_limit} tokens"

    available_raw = compatibility.get("available_memory_gib")
    estimated = _estimated_token_budget_memory_gib(profile, compatibility, value)
    if available_raw is not None and estimated is not None and estimated > float(available_raw):
        return (
            f"conservative memory guardrail estimates ~{estimated:.1f} GiB; "
            f"host has ~{float(available_raw):.1f} GiB usable"
        )
    return None


def _token_budget_options(
    profile: dict[str, Any],
    compatibility: dict[str, Any],
    *,
    default: int,
) -> tuple[list[int], list[tuple[int, str]]]:
    hard_limit = _profile_input_token_limit(profile)
    candidates = set(TOKEN_BUDGET_PRESETS)
    candidates.add(int(default))
    if hard_limit is not None and hard_limit < min(TOKEN_BUDGET_PRESETS):
        candidates.add(hard_limit)

    available: list[int] = []
    blocked: list[tuple[int, str]] = []
    for value in sorted(item for item in candidates if item > 0):
        reason = _token_budget_unavailable_reason(profile, compatibility, value)
        if reason is None:
            available.append(value)
        else:
            blocked.append((value, reason))
    return available, blocked


def _prompt_token_budget(
    *,
    model_id: str,
    backend: str,
    profile: dict[str, Any],
    compatibility: dict[str, Any],
    default: int,
) -> int | None:
    available, blocked = _token_budget_options(
        profile, compatibility, default=default
    )
    if not available:
        raise RuntimeError(
            f"No safe input-token budget is available for {model_id}/{backend} on this host"
        )

    print(f"\nMaximum input tokens for {model_id} / {backend}:")
    for index, value in enumerate(available, start=1):
        suffix = " — current/default" if value == default else ""
        estimated = _estimated_token_budget_memory_gib(profile, compatibility, value)
        memory = f"; ~{estimated:.1f} GiB guardrail" if estimated is not None else ""
        print(f"  {index}. {value} tokens{memory}{suffix}")
    if blocked:
        print("\nUnavailable token budgets for this host/profile:")
        for value, reason in blocked:
            print(f"  - {value} tokens: {reason}")
    print(
        "Token-budget memory checks are conservative installation guardrails; "
        "they do not prove the model's effective context limit or input completeness."
    )

    index = _prompt_index("Select max input tokens", len(available))
    return None if index is None else available[index]


def _validate_max_input_tokens(
    profile: dict[str, Any],
    value: int,
    *,
    compatibility: dict[str, Any] | None = None,
    force: bool = False,
) -> int:
    value = int(value)
    if value < 1:
        raise RuntimeError("max input tokens must be positive")
    compatibility = compatibility or {}
    reason = _token_budget_unavailable_reason(profile, compatibility, value)
    if reason is not None and not force:
        raise RuntimeError(
            f"Requested max input tokens ({value}) are not available for this profile/host: {reason}. "
            "Use --force only if you understand the context-memory risk."
        )
    if reason is not None and force:
        print(f"WARNING: forcing max input tokens {value}: {reason}")
    return value


def _ensure_venv(env_dir: Path, python_version: str) -> Path:
    python_path = _runtime_python(env_dir)
    if not python_path.is_file():
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["uv", "python", "install", python_version])
        _run(["uv", "venv", str(env_dir), "--python", python_version])
    return python_path


def _checkout_nimble(runtime_root: Path, *, source_key: str, upgrade: bool) -> Path:
    source_dir = runtime_root / source_key
    if not (source_dir / ".git").is_dir():
        if source_dir.exists():
            raise RuntimeError(f"Nimble source path exists but is not a git checkout: {source_dir}")
        _run(["git", "clone", "--depth", "1", "https://github.com/bespokelabsai/nimble.git", str(source_dir)])
    elif upgrade:
        _run(["git", "-C", str(source_dir), "pull", "--ff-only"])
    return source_dir


def _install_nimble(
    config_path: Path,
    data: dict[str, Any],
    profile: dict[str, Any],
    *,
    upgrade: bool,
) -> Path:
    runtime_root = _runtime_root(config_path, data)
    managed_models = (config_path.parent / "models").resolve()
    model_dir = Path(str(profile.get("model", "models/nimble-9b"))).expanduser()
    if not model_dir.is_absolute():
        model_dir = (config_path.parent / model_dir).resolve()
    else:
        model_dir = model_dir.resolve()
    # Nimble preparation may recursively replace its output on --force. Keep
    # that destructive output strictly inside Deqio's managed models/ tree.
    if model_dir == managed_models or not model_dir.is_relative_to(managed_models):
        raise RuntimeError(
            f"Nimble prepared model output must be a child of the workspace models directory: {model_dir}"
        )

    runtime_root.mkdir(parents=True, exist_ok=True)
    source_key = str(profile.get("source_key", "nimble-src"))
    source_dir = _checkout_nimble(runtime_root, source_key=source_key, upgrade=upgrade)
    backend = str(profile.get("nimble_backend") or data.get("backend") or "")
    if backend not in {"mlx", "cuda"}:
        raise RuntimeError("Nimble supports mlx and cuda profiles in Deqio")

    python_version = str(profile.get("python", "3.12"))
    env_dir = runtime_root / str(profile.get("runtime_key", f"nimble-{backend}"))
    python_path = _ensure_venv(env_dir, python_version)

    if backend == "mlx":
        _run([
            "uv", "pip", "install", "--python", str(python_path),
            "-r", str(source_dir / "requirements" / "mlx.txt"),
            "fastapi>=0.110", "uvicorn[standard]>=0.27",
        ])
    else:
        _run([
            "uv", "pip", "install", "--python", str(python_path),
            "torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu128",
        ])
        _run([
            "uv", "pip", "install", "--python", str(python_path),
            "-r", str(source_dir / "requirements" / "training.txt"),
            "fastapi>=0.110", "uvicorn[standard]>=0.27",
        ])

    prep_dir = runtime_root / "nimble-prep"
    prep_python = _ensure_venv(prep_dir, python_version)
    prep_command = ["uv", "pip", "install", "--python", str(prep_python)]
    if upgrade:
        prep_command.append("--upgrade")
    prep_command.extend([
        "torch==2.8.0",
        "-r", str(source_dir / "requirements" / "training.txt"),
        "huggingface-hub",
    ])
    _run(prep_command)

    model_config = runtime_root / str(profile.get("model_config", "nimble-model.json"))
    helper = Path(__file__).with_name("nimble_prepare.py").resolve()
    prepare = [
        str(prep_python), str(helper),
        "--source-root", str(source_dir),
        "--repo-id", str(profile.get("repo_id", "bespokelabs/Bespoke-Nimble-9B")),
        "--revision", str(profile.get("model_revision", "upstream-latest")),
        "--output-dir", str(model_dir),
        "--config", str(model_config),
    ]
    if upgrade:
        prepare.append("--force")
    _run(prepare)
    return env_dir


def _install_basal(
    config_path: Path,
    data: dict[str, Any],
    profile: dict[str, Any],
    *,
    upgrade: bool,
) -> Path:
    """Install an isolated official Basal 1.5 runtime for the selected profile."""

    backend = str(profile.get("basal_backend") or data.get("backend") or "")
    if backend not in {"mlx", "mps", "gguf", "cuda"}:
        raise RuntimeError("Basal supports mlx, mps, gguf and cuda profiles in Deqio")

    runtime_key = str(profile.get("runtime_key", f"basal-{backend}"))
    env_dir = _runtime_root(config_path, data) / runtime_key
    python_version = str(profile.get("python", "3.12"))
    python_path = _ensure_venv(env_dir, python_version)

    if str(profile.get("basal_runtime_version", "")) != "1.5.0":
        raise RuntimeError("Deqio 0.5 supports Basal 1.5 profiles only")

    if backend == "cuda":
        # Native Basal CUDA uses the upstream PyTorch engine. Install the CUDA
        # wheel first so the generic Basal dependency cannot resolve a CPU wheel.
        # Historical vLLM profiles still let vLLM own their torch resolution.
        if str(profile.get("basal_mode", "")) != "vllm":
            torch_command = [
                "uv", "pip", "install", "--python", str(python_path),
            ]
            if upgrade:
                torch_command.append("--upgrade")
            torch_command.extend([
                "torch==2.11.0",
                "--index-url", "https://download.pytorch.org/whl/cu128",
            ])
            _run(torch_command)
    package = str(profile.get("basal_package", "")).strip()
    if not package:
        raise RuntimeError("Basal profile is missing basal_package")
    command = ["uv", "pip", "install", "--python", str(python_path)]
    if upgrade:
        command.append("--upgrade")
    command.append(package)
    _run(command)
    return env_dir


def _install_decision2(
    config_path: Path,
    data: dict[str, Any],
    profile: dict[str, Any],
    *,
    upgrade: bool,
) -> Path:
    """Install the official Decision 2.0 Python runtime dependencies.

    The model package itself contains the official ``decision2`` runtime and is
    loaded from the pinned Hugging Face revision.  Deqio only supplies an
    isolated CUDA environment plus a localhost transport sidecar; it does not
    reimplement Decision 2.0 scoring or model conversion.
    """

    if str(profile.get("platform")) != "cuda":
        raise RuntimeError("Decision 2.0 in Deqio 0.5 requires the official CUDA runtime")

    runtime_key = str(profile.get("runtime_key", "decision2-cuda"))
    env_dir = _runtime_root(config_path, data) / runtime_key
    python_version = str(profile.get("python", "3.12.13"))
    python_path = _ensure_venv(env_dir, python_version)

    # The current official Decision 2.0 manifests are built/tested with the
    # versions below.  PyTorch is installed first because the hybrid Qwen
    # runtime dependencies import/build against it.  We intentionally do not
    # install any MLX/MPS/CPU-specific Decision implementation.
    torch_command = [
        "uv", "pip", "install", "--python", str(python_path),
    ]
    if upgrade:
        torch_command.append("--upgrade")
    torch_command.append("torch==2.12.0")
    _run(torch_command)

    packages = profile.get("packages")
    if not isinstance(packages, list) or not packages:
        raise RuntimeError("Decision 2.0 profile does not declare its pinned runtime packages")

    command = ["uv", "pip", "install", "--python", str(python_path)]
    if upgrade:
        command.append("--upgrade")
    command.extend(str(package) for package in packages)
    # FastAPI/Uvicorn are Deqio's localhost transport only; the model and all
    # scoring/runtime logic remain the verified package shipped by upstream.
    command.extend(["fastapi>=0.110,<1", "uvicorn[standard]>=0.27,<1"])
    _run(command)
    return env_dir



def _install_llama_cpp_runtime(
    config_path: Path,
    data: dict[str, Any],
    profile: dict[str, Any],
    *,
    upgrade: bool,
) -> Path:
    """Build the pinned official llama.cpp SystemOne server in an isolated runtime."""
    runtime_key = str(profile.get("runtime_key", "llama-cpp-systemone"))
    env_dir = _runtime_root(config_path, data) / runtime_key
    python_version = str(profile.get("python", "3.12"))
    python = _ensure_venv(env_dir, python_version)
    # Keep the GGUF provisioning path self-contained: cmake/ninja are installed
    # into the isolated runtime instead of assuming a global CMake install.
    tool_command = ["uv", "pip", "install", "--python", str(python)]
    if upgrade:
        tool_command.append("--upgrade")
    tool_command.extend(["cmake>=3.20,<5", "ninja>=1.11,<2"])
    _run(tool_command)
    bin_dir = env_dir / ("Scripts" if os.name == "nt" else "bin")
    cmake_exe = bin_dir / ("cmake.exe" if os.name == "nt" else "cmake")
    ninja_exe = bin_dir / ("ninja.exe" if os.name == "nt" else "ninja")
    if not cmake_exe.is_file() or not ninja_exe.is_file():
        raise RuntimeError(
            f"Isolated llama.cpp build tools are incomplete under {bin_dir}; "
            "expected both cmake and ninja"
        )
    build_env = os.environ.copy()
    build_env["PATH"] = str(bin_dir) + os.pathsep + build_env.get("PATH", "")

    revision = str(profile.get("llama_cpp_revision", "")).strip()
    if not revision:
        raise RuntimeError("llama.cpp GGUF profile is missing llama_cpp_revision")
    source = env_dir / "llama.cpp"
    build = env_dir / "llama-build"
    if not source.is_dir():
        _run(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git", str(source)])
    # Fetch the immutable commit used by the catalog.  This is deliberately not
    # a moving master checkout because decision-model support is runtime-sensitive.
    _run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", revision])
    _run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"])

    # A failed configure can leave CMakeCache.txt behind (for example when
    # Ninja was not visible on PATH).  Always configure the pinned runtime from
    # a clean build tree so a retry after Fix02 cannot inherit that broken cache.
    if build.exists():
        shutil.rmtree(build)

    cmake = [
        str(cmake_exe), "-G", "Ninja", "-S", str(source), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_MAKE_PROGRAM={ninja_exe}",
    ]
    if sys.platform == "darwin":
        xcrun = shutil.which("xcrun", path=build_env.get("PATH"))
        if not xcrun:
            raise RuntimeError(
                "Building the pinned llama.cpp runtime on macOS requires Xcode Command Line Tools; "
                "install them with: xcode-select --install"
            )
        try:
            clang = subprocess.run(
                [xcrun, "--find", "clang"], check=True, capture_output=True, text=True, env=build_env
            ).stdout.strip()
            clangxx = subprocess.run(
                [xcrun, "--find", "clang++"], check=True, capture_output=True, text=True, env=build_env
            ).stdout.strip()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Xcode Command Line Tools are present but clang/clang++ could not be resolved; "
                "run: xcode-select --install"
            ) from exc
        if not clang or not clangxx:
            raise RuntimeError(
                "Xcode Command Line Tools did not provide clang and clang++; run: xcode-select --install"
            )
        cmake.extend([
            "-DGGML_METAL=ON",
            f"-DCMAKE_C_COMPILER={clang}",
            f"-DCMAKE_CXX_COMPILER={clangxx}",
        ])
    elif shutil.which("nvidia-smi"):
        cmake.append("-DGGML_CUDA=ON")
    _run(cmake, env=build_env)
    build_cmd = [str(cmake_exe), "--build", str(build), "--config", "Release", "--target", "llama-server"]
    jobs = os.cpu_count() or 1
    build_cmd.extend(["-j", str(max(1, min(jobs, 8)))])
    _run(build_cmd, env=build_env)

    candidates = [build / "bin" / "llama-server", build / "bin" / "Release" / "llama-server.exe"]
    binary = next((item for item in candidates if item.is_file()), None)
    if binary is None:
        raise RuntimeError(f"Pinned llama.cpp build did not produce llama-server under {build / 'bin'}")
    target = env_dir / ("Scripts" if os.name == "nt" else "bin") / ("llama-server.exe" if os.name == "nt" else "llama-server")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, target)
    if os.name != "nt":
        target.chmod(target.stat().st_mode | 0o111)
    return env_dir


def _install_llama_cpp(
    config_path: Path, data: dict[str, Any], profile: dict[str, Any], *, upgrade: bool
) -> Path:
    return _install_llama_cpp_runtime(config_path, data, profile, upgrade=upgrade)


def _install_jevk5_gguf(
    config_path: Path, data: dict[str, Any], profile: dict[str, Any], *, upgrade: bool
) -> Path:
    env_dir = _install_llama_cpp_runtime(config_path, data, profile, upgrade=upgrade)
    python = _runtime_python(env_dir)
    package = str(profile.get("jevk5_package", "")).strip()
    if not package:
        raise RuntimeError("JevK5 GGUF profile is missing jevk5_package")
    command = ["uv", "pip", "install", "--python", str(python)]
    if upgrade:
        command.append("--upgrade")
    command.extend([package, "fastapi>=0.110,<1", "uvicorn[standard]>=0.27,<1"])
    _run(command)
    return env_dir


def _install_decider_gguf(
    config_path: Path, data: dict[str, Any], profile: dict[str, Any], *, upgrade: bool
) -> Path:
    runtime_key = str(profile.get("runtime_key", "decider-gguf"))
    env_dir = _runtime_root(config_path, data) / runtime_key
    python = _ensure_venv(env_dir, str(profile.get("python", "3.12")))
    package = str(profile.get("decider_package", "")).strip()
    if not package:
        raise RuntimeError("Decider GGUF profile is missing decider_package")
    command = ["uv", "pip", "install", "--python", str(python)]
    if upgrade:
        command.append("--upgrade")
    command.extend([package, "fastapi>=0.110,<1", "uvicorn[standard]>=0.27,<1"])
    env = os.environ.copy()
    if sys.platform == "darwin":
        env["CMAKE_ARGS"] = "-DGGML_METAL=on"
    elif shutil.which("nvidia-smi"):
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
    _run(command, env=env)
    return env_dir

def _install_runtime(config_path: Path, data: dict[str, Any], profile: dict[str, Any], *, upgrade: bool) -> Path:
    if profile.get("installer") == "nimble":
        return _install_nimble(config_path, data, profile, upgrade=upgrade)
    if profile.get("installer") == "basal":
        return _install_basal(config_path, data, profile, upgrade=upgrade)
    if profile.get("installer") == "decision2":
        return _install_decision2(config_path, data, profile, upgrade=upgrade)
    if profile.get("installer") == "llama_cpp":
        return _install_llama_cpp(config_path, data, profile, upgrade=upgrade)
    if profile.get("installer") == "jevk5_gguf":
        return _install_jevk5_gguf(config_path, data, profile, upgrade=upgrade)
    if profile.get("installer") == "decider_gguf":
        return _install_decider_gguf(config_path, data, profile, upgrade=upgrade)

    runtime_key = profile.get("runtime_key")
    packages = profile.get("packages")
    if not runtime_key or not isinstance(packages, list) or not packages:
        raise RuntimeError("This profile does not define an isolated runtime")

    env_dir = _runtime_root(config_path, data) / str(runtime_key)
    python_version = str(profile.get("python", "3.12"))
    python_path = _runtime_python(env_dir)

    if not python_path.is_file():
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["uv", "python", "install", python_version])
        _run(["uv", "venv", str(env_dir), "--python", python_version])

    command = ["uv", "pip", "install", "--python", str(python_path)]
    if upgrade:
        command.append("--upgrade")
    command.extend(str(package) for package in packages)
    _run(command)
    return env_dir


def _looks_like_hf_repo(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith(("./", "../", "~", "/", "models/")):
        return False
    return value.count("/") == 1 and " " not in value


def _declared_downloads(profile: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    download = profile.get("download")
    if isinstance(download, dict):
        result.append(download)
    downloads = profile.get("downloads")
    if isinstance(downloads, list):
        result.extend(item for item in downloads if isinstance(item, dict))
    return result


def _prefetch_one_download(
    config_path: Path, profile: dict[str, Any], download: dict[str, Any]
) -> str:
    repo_id = str(download.get("repo_id", ""))
    kind = download.get("type")
    local_dir_value = download.get("local_dir")
    kwargs: dict[str, Any] = {"repo_id": repo_id}
    revision = download.get("revision", profile.get("model_revision"))
    if revision and not str(revision).startswith("local-") and str(revision) != "upstream-latest":
        kwargs["revision"] = str(revision)
    if local_dir_value:
        local_dir = Path(str(local_dir_value)).expanduser()
        if not local_dir.is_absolute():
            local_dir = (config_path.parent / local_dir).resolve()
        else:
            local_dir = local_dir.resolve()
        models_root = (config_path.parent / "models").resolve()
        if local_dir == models_root or not local_dir.is_relative_to(models_root):
            raise RuntimeError(
                "Managed model download local_dir must be a child of the workspace models/ directory"
            )
        local_dir.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(local_dir)
    if kind == "snapshot":
        if isinstance(download.get("allow_patterns"), list):
            kwargs["allow_patterns"] = [str(item) for item in download["allow_patterns"]]
        if isinstance(download.get("ignore_patterns"), list):
            kwargs["ignore_patterns"] = [str(item) for item in download["ignore_patterns"]]
        path = snapshot_download(**kwargs)
        return f"Downloaded {repo_id} -> {path}"
    if kind == "file":
        filename = str(download["filename"])
        path = hf_hub_download(filename=filename, **kwargs)
        return f"Downloaded {repo_id}/{filename} -> {path}"
    raise RuntimeError(f"Unsupported download type: {kind!r}")


def _prefetch_declared_weights(config_path: Path, profile: dict[str, Any]) -> str:
    """Fetch every model artifact declared by a profile before verification."""
    downloads = _declared_downloads(profile)
    if downloads:
        return "\n".join(_prefetch_one_download(config_path, profile, item) for item in downloads)

    model = profile.get("model")
    if _looks_like_hf_repo(model):
        kwargs = {"repo_id": str(model)}
        revision = profile.get("model_revision")
        if revision and str(revision) != "upstream-latest":
            kwargs["revision"] = str(revision)
        path = snapshot_download(**kwargs)
        return f"Cached model weights: {model} -> {path}"

    return "Model artifact is already local/prepared; no Hub prefetch required."




def _resolved_hf_revision(repo_id: str, requested_revision: Any) -> str | None:
    """Resolve a Hub revision to an immutable commit SHA without making install fail if metadata lookup fails."""
    requested = None if requested_revision in (None, "", "upstream-latest") else str(requested_revision)
    if requested and requested.startswith("local-"):
        requested = None
    if requested and len(requested) == 40 and all(ch in "0123456789abcdefABCDEF" for ch in requested):
        return requested.lower()
    try:
        info = HfApi().model_info(repo_id, revision=requested)
    except Exception:
        return None
    sha = getattr(info, "sha", None)
    return str(sha) if sha else None


def _artifact_attestation(config_path: Path, data: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe the exact model artifacts prepared during the network-enabled install/update phase."""
    artifacts: list[dict[str, Any]] = []
    downloads = _declared_downloads(profile)
    for download in downloads:
        repo_id = str(download.get("repo_id", ""))
        if not repo_id:
            continue
        requested = download.get("revision")
        if requested in (None, ""):
            candidate = profile.get("model_revision")
            if candidate and not str(candidate).startswith("local-"):
                requested = candidate
        row: dict[str, Any] = {
            "source": "huggingface",
            "repo_id": repo_id,
            "requested_revision": requested if requested not in ("",) else None,
            "resolved_revision": _resolved_hf_revision(repo_id, requested),
        }
        if download.get("role"):
            row["role"] = str(download["role"])
        if download.get("filename"):
            row["filename"] = str(download["filename"])
        if download.get("local_dir"):
            row["local_dir"] = str(download["local_dir"])
        artifacts.append(row)

    if profile.get("installer") == "nimble":
        runtime_root = _runtime_root(config_path, data)
        config_path_value = runtime_root / str(profile.get("model_config", "nimble-model.json"))
        model_id = str(profile.get("repo_id", ""))
        resolved_revision = None
        if config_path_value.is_file():
            try:
                prepared = json.loads(config_path_value.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prepared = {}
            model_id = str(prepared.get("model_id") or model_id)
            resolved_revision = prepared.get("revision")
        if model_id:
            artifacts.append({
                "source": "huggingface",
                "role": "nimble-adapter",
                "repo_id": model_id,
                "requested_revision": profile.get("model_revision"),
                "resolved_revision": str(resolved_revision) if resolved_revision else _resolved_hf_revision(model_id, profile.get("model_revision")),
            })

        model_dir = Path(str(profile.get("model", "models/nimble-9b"))).expanduser()
        if not model_dir.is_absolute():
            model_dir = (config_path.parent / model_dir).resolve()
        contract_path = model_dir / "schema_config.json"
        if contract_path.is_file():
            try:
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                contract = {}
            base_model = contract.get("model")
            base_revision = contract.get("revision")
            if isinstance(base_model, str) and base_model:
                artifacts.append({
                    "source": "huggingface",
                    "role": "base-model",
                    "repo_id": base_model,
                    "requested_revision": base_revision,
                    "resolved_revision": str(base_revision) if base_revision else None,
                })
        return artifacts

    if not downloads:
        model = profile.get("model")
        if _looks_like_hf_repo(model):
            requested = profile.get("model_revision")
            artifacts.append({
                "source": "huggingface",
                "repo_id": str(model),
                "requested_revision": requested,
                "resolved_revision": _resolved_hf_revision(str(model), requested),
            })
        elif isinstance(model, str) and model:
            artifacts.append({
                "source": "local",
                "path": model,
                "requested_revision": profile.get("model_revision"),
                "resolved_revision": None,
            })
    return artifacts



def _pin_profile_hf_revisions(profile: dict[str, Any]) -> dict[str, Any]:
    """Resolve a pin-on-install profile to immutable Hub commits and exact files.

    Some official model repositories publish a moving ``main`` rather than a
    release tag. Deqio resolves those revisions once during the network-enabled
    install/update phase, resolves single-file patterns when required, and stores
    the resulting commit/file identity in the installation attestation. Runtime
    loading then reuses that immutable identity offline.
    """
    if not profile.get("pin_hf_revisions_at_install"):
        return profile

    pinned = deepcopy(profile)
    resolved: dict[str, str] = {}
    api = HfApi()

    def resolve(repo_id: str, requested: Any) -> str:
        if not repo_id:
            raise RuntimeError("Pinned profile contains an empty Hugging Face repo id")
        requested_text = None if requested in (None, "", "upstream-latest") else str(requested)
        if requested_text and len(requested_text) == 40 and all(ch in "0123456789abcdefABCDEF" for ch in requested_text):
            return requested_text.lower()
        key = f"{repo_id}@{requested_text or 'main'}"
        if key not in resolved:
            try:
                info = api.model_info(repo_id, revision=requested_text)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not resolve immutable revision for {repo_id}@{requested_text or 'main'}"
                ) from exc
            sha = getattr(info, "sha", None)
            if not sha:
                raise RuntimeError(f"Hugging Face did not return a commit SHA for {repo_id}")
            resolved[key] = str(sha)
        return resolved[key]

    downloads = _declared_downloads(pinned)
    for download in downloads:
        repo_id = str(download.get("repo_id", ""))
        if not repo_id:
            continue
        download["revision"] = resolve(repo_id, download.get("revision", "main"))
        if download.get("type") == "file" and not download.get("filename") and download.get("filename_pattern"):
            pattern = str(download["filename_pattern"])
            try:
                files = api.list_repo_files(repo_id, revision=str(download["revision"]))
            except Exception as exc:
                raise RuntimeError(
                    f"Could not enumerate pinned artifacts for {repo_id}@{download['revision']}"
                ) from exc
            matches = sorted(str(name) for name in files if fnmatch(str(name), pattern))
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one {pattern!r} artifact in {repo_id}@{download['revision']}; "
                    f"found {len(matches)}"
                )
            download["filename"] = matches[0]

    model = pinned.get("model")
    if _looks_like_hf_repo(model):
        model_repo = str(model)
        model_revision = resolve(model_repo, pinned.get("model_revision", "main"))
        pinned["model_revision"] = model_revision
        for download in downloads:
            if str(download.get("repo_id", "")) == model_repo and download.get("role") in {"basal-model", "basal-metadata"}:
                download["revision"] = model_revision
    return pinned

def _preflight_profile(
    model_id: str,
    backend: str,
    profile: dict[str, Any],
    *,
    force: bool,
) -> dict[str, Any]:
    compatibility = profile_compatibility(backend, profile)
    if not compatibility["compatible"] and not force:
        raise RuntimeError(
            f"{model_id}/{backend} is not compatible with this host: {compatibility['reason']}. "
            "Use --force only if you understand the memory/backend risk."
        )
    if not compatibility["compatible"]:
        print(f"WARNING: forcing {model_id}/{backend}: {compatibility['reason']}")
    elif compatibility.get("warning"):
        print(f"WARNING: {model_id}/{backend}: {compatibility['warning']}")
    return compatibility


def _verify_model_ready(
    config_path: Path,
    data: dict[str, Any],
    entry: dict[str, Any],
    profile: dict[str, Any],
    backend: str,
    *,
    max_input_tokens: int,
) -> None:
    selected = _select_data(
        data, entry, profile, backend, max_input_tokens=max_input_tokens
    )
    # Installation is the one lifecycle phase where network access is allowed.
    # The readiness probe forces the engine to resolve every transitive/base
    # model dependency now, before the profile is registered as installed.
    selected["hf_offline_runtime"] = False
    settings = settings_from_data(config_path, selected, apply_environment=False)
    print(
        f"Verifying model readiness online (timeout={settings.sidecar_startup_seconds}s): "
        f"{entry['id']} / {backend}"
    )
    # SystemOneRuntime.load already performs a real typed-decision readiness
    # probe. Reaching this point means the model can answer requests. Always
    # close the temporary runtime immediately after verification.
    runtime = BackendRuntime.load(settings)
    runtime.close()


def _install_profile(
    *,
    config_path: Path,
    data: dict[str, Any],
    catalog: dict[str, Any],
    model_id: str,
    backend: str,
    upgrade: bool,
    force: bool,
    max_input_tokens: int | None = None,
) -> int:
    entry = get_model(catalog, model_id)
    profile = _pin_profile_hf_revisions(get_profile(catalog, model_id, backend))
    compatibility = _preflight_profile(model_id, backend, profile, force=force)
    chosen_max_input_tokens = _validate_max_input_tokens(
        profile,
        max_input_tokens
        if max_input_tokens is not None
        else _default_max_input_tokens(config_path, data, model_id, backend, profile),
        compatibility=compatibility,
        force=force,
    )

    env_dir = _install_runtime(config_path, data, profile, upgrade=upgrade)
    print(f"Runtime ready: {env_dir}")
    if profile.get("installer") == "nimble":
        print("Nimble source, runtime dependencies and prepared model weights are ready.")
    else:
        print(_prefetch_declared_weights(config_path, profile))

    _verify_model_ready(
        config_path, data, entry, profile, backend, max_input_tokens=chosen_max_input_tokens
    )
    artifacts = _artifact_attestation(config_path, data, profile)
    mark_installed(
        config_path,
        model_id,
        backend,
        verified=True,
        source="update" if upgrade else "install",
        artifacts=artifacts,
        max_input_tokens=chosen_max_input_tokens,
    )
    print(
        f"Installed and verified profile: {model_id} / {backend} "
        f"(max input tokens: {chosen_max_input_tokens})"
    )
    return chosen_max_input_tokens


def _installation_rows(config_path: Path, data: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    return installed_profiles(
        config_path=config_path,
        config_data=data,
        catalog=catalog,
        active_model_id=str(data.get("model_id", "")),
        active_backend=str(data.get("backend", "")),
    )


def cmd_list(args: argparse.Namespace) -> int:
    config_path = _config_path(args.config)
    data = _read_config(config_path)
    catalog = load_catalog(_catalog_path(config_path, data))
    backend_filter = args.backend
    host = detect_host()
    rows: list[dict[str, Any]] = []

    for entry in catalog["models"]:
        backend_rows: list[dict[str, Any]] = []
        for backend, profile in (entry.get("backends") or {}).items():
            if backend_filter and backend != backend_filter:
                continue
            if not isinstance(profile, dict):
                continue
            check = profile_compatibility(str(backend), profile, host=host)
            backend_rows.append({
                "backend": str(backend),
                **check,
            })
        if args.compatible:
            backend_rows = [item for item in backend_rows if item["compatible"]]
        if not backend_rows:
            continue
        rows.append({
            "model_id": str(entry["id"]),
            "engine": str(entry["engine"]),
            "label": str(entry.get("label", entry["id"])),
            "description": str(entry.get("description", "")),
            "backends": backend_rows,
        })

    if args.json:
        print(json.dumps({"host": describe_host(host), "models": rows}, indent=2))
        return 0

    print(f"Host: {describe_host(host)}")
    print(f"{'MODEL ID':28} {'ENGINE':10} {'BACKENDS':28} DESCRIPTION")
    print("-" * 112)
    for row in rows:
        backend_text = ",".join(
            item["backend"] + ("!" if item.get("warning") else "")
            for item in row["backends"]
        )
        print(f"{row['model_id']:28} {row['engine']:10} {backend_text:28} {row['description']}")
    if any(item.get("warning") for row in rows for item in row["backends"]):
        print("\n! compatible, but below the catalog's recommended memory")
    return 0

def cmd_installed(args: argparse.Namespace) -> int:
    config_path = _config_path(args.config)
    data = _read_config(config_path)
    catalog = load_catalog(_catalog_path(config_path, data))
    rows = _installation_rows(config_path, data, catalog)
    rows = [row for row in rows if row["installed"]]
    if args.backend:
        rows = [row for row in rows if row["backend"] == args.backend]
    if not args.all_hosts:
        rows = [row for row in rows if row["host_compatible"]]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print("No installed model profiles were detected for this host.")
        return 0

    print(f"{'MODEL ID':28} {'BACKEND':8} {'ENGINE':10} {'STATUS':12} LABEL")
    print("-" * 92)
    for row in rows:
        marker = "*" if row["active"] else " "
        print(
            f"{marker}{str(row['model_id']):27} {str(row['backend']):8} "
            f"{str(row['engine']):10} {str(row['status']):12} {row['label']}"
        )
    print("\n* active selection")
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    config_path = _config_path(args.config)
    data = _read_config(config_path)
    catalog = load_catalog(_catalog_path(config_path, data))
    entry = get_model(catalog, args.model_id)
    profile = get_profile(catalog, args.model_id, args.backend)
    if args.install:
        _install_profile(
            config_path=config_path,
            data=data,
            catalog=catalog,
            model_id=args.model_id,
            backend=args.backend,
            upgrade=False,
            force=False,
            max_input_tokens=getattr(args, "max_input_tokens", None),
        )
    record = installation_record(config_path, args.model_id, args.backend) or {}
    updated = _select_data(
        data,
        entry,
        profile,
        args.backend,
        max_input_tokens=record.get("max_input_tokens"),
    )
    _write_config(config_path, updated)
    print(f"Selected {args.model_id} on {args.backend} in {config_path}")
    return 0


def cmd_use(args: argparse.Namespace) -> int:
    config_path = _config_path(args.config)
    data = _read_config(config_path)
    catalog = load_catalog(_catalog_path(config_path, data))
    rows = [
        row
        for row in _installation_rows(config_path, data, catalog)
        if row["installed"] and row["host_compatible"]
    ]
    if not rows:
        raise RuntimeError("No installed model profiles are available on this host. Run deqio models setup first.")

    if args.model_id:
        matches = [
            row
            for row in rows
            if row["model_id"] == args.model_id and (args.backend is None or row["backend"] == args.backend)
        ]
        if len(matches) != 1:
            available = ", ".join(f"{row['model_id']}:{row['backend']}" for row in rows)
            raise RuntimeError(
                f"Installed profile {args.model_id!r} with backend {args.backend!r} was not uniquely found. "
                f"Available: {available}"
            )
        selected = matches[0]
    else:
        print("Installed model profiles:")
        for index, row in enumerate(rows, start=1):
            active = " [active]" if row["active"] else ""
            print(f"  {index}. {row['label']} — {row['backend']} ({row['engine']}){active}")
        index = _prompt_index("Select profile", len(rows))
        if index is None:
            return _cancelled()
        selected = rows[index]

    entry = get_model(catalog, str(selected["model_id"]))
    profile = get_profile(catalog, str(selected["model_id"]), str(selected["backend"]))
    updated = _select_data(
        data,
        entry,
        profile,
        str(selected["backend"]),
        max_input_tokens=selected.get("max_input_tokens"),
    )
    _write_config(config_path, updated)
    print(f"Selected installed profile {selected['model_id']} / {selected['backend']}")
    print("Start/restart deqio serve to load it, or use POST /v1/models/activate while the server is running.")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    config_path = _config_path(args.config)
    data = _read_config(config_path)
    catalog = load_catalog(_catalog_path(config_path, data))
    model_id = args.model_id or str(data.get("model_id", "semif-qwen3.5-4b"))
    backend = args.backend or str(data.get("backend", "mlx"))

    _install_profile(
        config_path=config_path,
        data=data,
        catalog=catalog,
        model_id=model_id,
        backend=backend,
        upgrade=bool(args.upgrade),
        force=bool(getattr(args, "force", False)),
        max_input_tokens=getattr(args, "max_input_tokens", None),
    )
    return 0


def _registered_keys(config_path: Path) -> set[str]:
    registry = load_registry(config_path)
    profiles = registry.get("profiles", {}) if isinstance(registry, dict) else {}
    return {str(key) for key in profiles} if isinstance(profiles, dict) else set()


def _catalog_profile_for_key(catalog: dict[str, Any], key: str) -> tuple[dict[str, Any], dict[str, Any], str] | None:
    if "::" not in key:
        return None
    model_id, backend = key.split("::", 1)
    try:
        entry = get_model(catalog, model_id)
        profile = get_profile(catalog, model_id, backend)
    except RuntimeError:
        return None
    return entry, profile, backend


def _local_artifact_paths(config_path: Path, profile: dict[str, Any]) -> set[Path]:
    managed_root = (config_path.parent / "models").resolve()
    paths: set[Path] = set()

    def add_if_managed(path_value: str) -> None:
        path = Path(path_value).expanduser()
        resolved = (path if path.is_absolute() else config_path.parent / path).resolve()
        # Automatic deletion is deliberately restricted to Deqio's workspace
        # model directory. An external absolute/local model path is user-owned
        # and must never be recursively removed by `deqio models delete`.
        if resolved != managed_root and not resolved.is_relative_to(managed_root):
            return
        paths.add(resolved)

    for download in _declared_downloads(profile):
        if download.get("local_dir"):
            add_if_managed(str(download["local_dir"]))
    model = profile.get("model")
    if isinstance(model, str) and (model.startswith(("./", "../", "~/", "/", "models/"))):
        add_if_managed(model)
    return paths


def _hf_repo_ids(profile: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    model = profile.get("model")
    if _looks_like_hf_repo(model):
        result.add(str(model))
    for download in _declared_downloads(profile):
        if _looks_like_hf_repo(download.get("repo_id")):
            result.add(str(download["repo_id"]))
    return result


def _remaining_profile_values(
    catalog: dict[str, Any],
    keys: set[str],
    getter,
) -> set[Any]:
    values: set[Any] = set()
    for key in keys:
        resolved = _catalog_profile_for_key(catalog, key)
        if resolved is None:
            continue
        _, profile, _ = resolved
        values.update(getter(profile))
    return values


def _delete_hf_repo_cache(repo_id: str) -> None:
    try:
        cache = scan_cache_dir()
    except Exception as error:
        print(f"Could not inspect Hugging Face cache for {repo_id}: {error}")
        return
    revisions: list[str] = []
    for repo in cache.repos:
        if str(repo.repo_id) == repo_id:
            revisions.extend(str(revision.commit_hash) for revision in repo.revisions)
    if not revisions:
        print(f"HF cache not present: {repo_id}")
        return
    strategy = cache.delete_revisions(*revisions)
    print(f"Purging HF cache {repo_id}; expected free space: {strategy.expected_freed_size_str}")
    strategy.execute()


def _cleanup_profile_artifacts(
    *,
    config_path: Path,
    data: dict[str, Any],
    catalog: dict[str, Any],
    profile: dict[str, Any],
    remaining_keys: set[str],
    purge_cache: bool,
) -> None:
    remaining_runtime_keys = _remaining_profile_values(
        catalog, remaining_keys, lambda p: {str(p["runtime_key"])} if p.get("runtime_key") else set()
    )
    runtime_key = profile.get("runtime_key")
    if runtime_key and str(runtime_key) not in remaining_runtime_keys:
        runtime_dir = _runtime_root(config_path, data) / str(runtime_key)
        if runtime_dir.exists():
            shutil.rmtree(runtime_dir)
            print(f"Removed runtime: {runtime_dir}")

    remaining_local = _remaining_profile_values(
        catalog, remaining_keys, lambda p: _local_artifact_paths(config_path, p)
    )
    for path in _local_artifact_paths(config_path, profile):
        if path in remaining_local:
            print(f"Keeping shared model artifact: {path}")
            continue
        if path.is_dir():
            shutil.rmtree(path)
            print(f"Removed model artifact: {path}")
        elif path.exists():
            path.unlink()
            print(f"Removed model artifact: {path}")

    remaining_hf = _remaining_profile_values(catalog, remaining_keys, _hf_repo_ids)
    for repo_id in sorted(_hf_repo_ids(profile)):
        if repo_id in remaining_hf:
            print(f"Keeping shared HF cache: {repo_id}")
        elif purge_cache:
            _delete_hf_repo_cache(repo_id)
        else:
            print(f"Keeping global HF cache: {repo_id} (use --purge-cache to remove it)")

    # Nimble has shared source/preparation assets outside the backend runtime.
    if profile.get("installer") == "nimble":
        any_nimble = False
        for key in remaining_keys:
            resolved = _catalog_profile_for_key(catalog, key)
            if resolved is not None and resolved[1].get("installer") == "nimble":
                any_nimble = True
                break
        if not any_nimble:
            root = _runtime_root(config_path, data)
            for name in (
                str(profile.get("source_key", "nimble-src")),
                "nimble-prep",
                str(profile.get("model_config", "nimble-model.json")),
            ):
                path = root / name
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"Removed Nimble support directory: {path}")
                elif path.exists():
                    path.unlink()
                    print(f"Removed Nimble support file: {path}")


def cmd_delete(args: argparse.Namespace) -> int:
    config_path = _config_path(args.config)
    data = _read_config(config_path)
    catalog = load_catalog(_catalog_path(config_path, data))
    rows = [row for row in _installation_rows(config_path, data, catalog) if row["registered"]]
    if not rows:
        print("No registered model profiles to delete.")
        return 0

    if args.model_id:
        matches = [
            row for row in rows
            if row["model_id"] == args.model_id and (args.backend is None or row["backend"] == args.backend)
        ]
        if len(matches) != 1:
            available = ", ".join(f"{row['model_id']}:{row['backend']}" for row in rows)
            raise RuntimeError(
                f"Installed profile {args.model_id!r} with backend {args.backend!r} was not uniquely found. "
                f"Available: {available}"
            )
        selected = matches[0]
    else:
        print("Installed model profiles:")
        for index, row in enumerate(rows, start=1):
            active = " [active]" if row["active"] else ""
            print(f"  {index}. {row['label']} — {row['backend']} ({row['engine']}){active}")
        index = _prompt_index("Select profile to delete", len(rows))
        if index is None:
            return _cancelled()
        selected = rows[index]

    model_id = str(selected["model_id"])
    backend = str(selected["backend"])
    if not args.yes:
        answer = input(
            f"Delete {model_id} / {backend} from this Deqio workspace? [y/N/q]: "
        ).strip().lower()
        if answer in {"q", "quit", "exit"}:
            return _cancelled()
        if answer not in {"y", "yes"}:
            print("Delete cancelled.")
            return 0

    profile = get_profile(catalog, model_id, backend)
    removed = unmark_installed(config_path, model_id, backend)
    if removed is None:
        raise RuntimeError(f"Profile {model_id}/{backend} is not registered as installed")
    remaining_keys = _registered_keys(config_path)
    _cleanup_profile_artifacts(
        config_path=config_path,
        data=data,
        catalog=catalog,
        profile=profile,
        remaining_keys=remaining_keys,
        purge_cache=bool(args.purge_cache),
    )

    if selected["active"]:
        remaining_rows = [
            row for row in _installation_rows(config_path, data, catalog)
            if row["installed"] and row["host_compatible"]
        ]
        if remaining_rows:
            replacement = remaining_rows[0]
            entry = get_model(catalog, str(replacement["model_id"]))
            replacement_profile = get_profile(catalog, str(replacement["model_id"]), str(replacement["backend"]))
            _write_config(
                config_path,
                _select_data(
                    data,
                    entry,
                    replacement_profile,
                    str(replacement["backend"]),
                    max_input_tokens=replacement.get("max_input_tokens"),
                ),
            )
            print(f"Active profile removed; selected {replacement['model_id']} / {replacement['backend']} instead.")
        else:
            print("WARNING: the deleted profile was active and no installed replacement remains. Run deqio models setup.")

    print(f"Deleted profile: {model_id} / {backend}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config_path = _config_path(args.config)
    data = _read_config(config_path)
    catalog = load_catalog(_catalog_path(config_path, data))
    model_id = str(data.get("model_id", "semif-qwen3.5-4b"))
    backend = str(data.get("backend", "mlx"))
    entry = get_model(catalog, model_id)
    profile = get_profile(catalog, model_id, backend)
    rows = _installation_rows(config_path, data, catalog)
    state = next((row for row in rows if row["model_id"] == model_id and row["backend"] == backend), None)

    result: dict[str, Any] = {
        "config": str(config_path),
        "engine": entry["engine"],
        "model_id": model_id,
        "backend": backend,
        "model": profile.get("model"),
        "installation": state,
    }
    runtime_key = profile.get("runtime_key")
    if runtime_key:
        env_dir = _runtime_root(config_path, data) / str(runtime_key)
        result["runtime_dir"] = str(env_dir)
        result["runtime_installed"] = _runtime_python(env_dir).is_file()
    download = profile.get("download")
    if isinstance(download, dict) and download.get("local_dir"):
        local_dir = Path(str(download["local_dir"]))
        if not local_dir.is_absolute():
            local_dir = config_path.parent / local_dir
        result["local_model_dir"] = str(local_dir.resolve())
        result["local_model_present"] = local_dir.exists()
    print(json.dumps(result, indent=2))
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    config_path = _config_path(args.config)
    data = _read_config(config_path)
    catalog = load_catalog(_catalog_path(config_path, data))
    host = detect_host()

    available_backends = host_backends()
    if not available_backends:
        raise RuntimeError(
            "No supported Deqio backend was detected. Apple Silicon requires macOS; "
            "Linux/Windows can use Basal 1.5 GGUF, while CUDA additionally requires a working NVIDIA driver visible through nvidia-smi."
        )
    print(f"Host: {describe_host(host)}")
    print("Available backends for this host:")
    backend_rows: list[tuple[str, int]] = []
    for backend in available_backends:
        count = 0
        for item in catalog["models"]:
            profile = (item.get("backends") or {}).get(backend)
            if isinstance(profile, dict) and profile_compatibility(backend, profile, host=host)["compatible"]:
                count += 1
        backend_rows.append((backend, count))
    for index, (backend, count) in enumerate(backend_rows, start=1):
        print(f"  {index}. {backend} ({count} compatible models)")
    backend_index = _prompt_index("Select backend", len(available_backends))
    if backend_index is None:
        return _cancelled()
    backend = available_backends[backend_index]

    compatible: list[tuple[dict[str, Any], dict[str, Any]]] = []
    unavailable: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in catalog["models"]:
        profile = (item.get("backends") or {}).get(backend)
        if not isinstance(profile, dict):
            continue
        check = profile_compatibility(backend, profile, host=host)
        (compatible if check["compatible"] else unavailable).append((item, check))

    if not compatible:
        details = "; ".join(f"{item['id']}: {check['reason']}" for item, check in unavailable)
        raise RuntimeError(f"No {backend} models are compatible with this host. {details}")

    installed_keys = {
        profile_key(str(row["model_id"]), str(row["backend"]))
        for row in _installation_rows(config_path, data, catalog)
        if row["installed"]
    }

    print(f"\nModels compatible with {backend}:")
    for index, (item, check) in enumerate(compatible, start=1):
        warning = f" — WARNING: {check['warning']}" if check.get("warning") else ""
        minimum = check.get("minimum_memory_gib")
        memory = f"; ~{float(minimum):.1f} GiB minimum" if minimum is not None else ""
        marker = "[x]" if profile_key(str(item["id"]), backend) in installed_keys else "[ ]"
        print(f"  {index}. {marker} {item['id']} — {item.get('label', item['id'])}{memory}{warning}")
    if unavailable:
        print("\nUnavailable on this host:")
        for item, check in unavailable:
            print(f"  - {item['id']}: {check['reason']}")

    model_index = _prompt_index("Select model", len(compatible))
    if model_index is None:
        return _cancelled()
    entry, selected_check = compatible[model_index]
    profile = get_profile(catalog, str(entry["id"]), backend)

    default_max_input_tokens = _default_max_input_tokens(
        config_path, data, str(entry["id"]), backend, profile
    )
    max_input_tokens = _prompt_token_budget(
        model_id=str(entry["id"]),
        backend=backend,
        profile=profile,
        compatibility=selected_check,
        default=default_max_input_tokens,
    )
    if max_input_tokens is None:
        return _cancelled()

    print(f"\nInstalling {entry['id']} / {backend} with max input tokens {max_input_tokens}")
    chosen_max_input_tokens = _install_profile(
        config_path=config_path,
        data=data,
        catalog=catalog,
        model_id=str(entry["id"]),
        backend=backend,
        upgrade=False,
        force=False,
        max_input_tokens=max_input_tokens,
    )
    updated = _select_data(
        data, entry, profile, backend, max_input_tokens=chosen_max_input_tokens
    )
    _write_config(config_path, updated)
    print(f"Setup complete. Active profile: {entry['id']} / {backend}")
    print("The runtime and model weights are ready; normal serve/benchmark starts run with Hugging Face offline mode enabled.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deqio models",
        description="Install, inspect, select, and update decision engines/models for Deqio.",
    )
    parser.add_argument("--config", help="Path to config.json (default: ./config.json)")
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List catalogued models")
    list_cmd.add_argument("--backend", choices=SUPPORTED_BACKENDS)
    list_cmd.add_argument("--compatible", action="store_true", help="Show only profiles compatible with this host")
    list_cmd.add_argument("--json", action="store_true", help="Emit host compatibility as machine-readable JSON")
    list_cmd.set_defaults(func=cmd_list)

    installed_cmd = sub.add_parser("installed", help="List locally installed model/backend profiles")
    installed_cmd.add_argument("--backend", choices=SUPPORTED_BACKENDS)
    installed_cmd.add_argument("--all-hosts", action="store_true", help="Include installed profiles for other host backends")
    installed_cmd.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    installed_cmd.set_defaults(func=cmd_installed)

    setup_cmd = sub.add_parser("setup", help="Interactive backend/model installer")
    setup_cmd.set_defaults(func=cmd_setup)

    use_cmd = sub.add_parser("use", help="Select an already-installed model profile")
    use_cmd.add_argument("model_id", nargs="?")
    use_cmd.add_argument("--backend", choices=SUPPORTED_BACKENDS)
    use_cmd.set_defaults(func=cmd_use)

    select_cmd = sub.add_parser("select", help="Write any catalogued model/backend selection to config.json")
    select_cmd.add_argument("model_id")
    select_cmd.add_argument("--backend", choices=SUPPORTED_BACKENDS, required=True)
    select_cmd.add_argument("--install", action="store_true")
    select_cmd.set_defaults(func=cmd_select)

    install_cmd = sub.add_parser("install", help="Install the selected or named runtime/model")
    install_cmd.add_argument("model_id", nargs="?")
    install_cmd.add_argument("--backend", choices=SUPPORTED_BACKENDS)
    install_cmd.add_argument("--upgrade", action="store_true")
    install_cmd.add_argument("--force", action="store_true", help="Bypass host memory/backend compatibility checks")
    install_cmd.add_argument("--max-input-tokens", type=int, help="Maximum input-token budget stored for this profile")
    install_cmd.set_defaults(func=cmd_install)

    update_cmd = sub.add_parser("update", help="Upgrade an isolated engine runtime")
    update_cmd.add_argument("model_id", nargs="?")
    update_cmd.add_argument("--backend", choices=SUPPORTED_BACKENDS)
    update_cmd.add_argument("--force", action="store_true", help="Bypass host memory/backend compatibility checks")
    update_cmd.add_argument("--max-input-tokens", type=int, help="Maximum input-token budget stored for this profile")
    update_cmd.set_defaults(func=lambda a: cmd_install(argparse.Namespace(**vars(a), upgrade=True)))

    delete_cmd = sub.add_parser("delete", help="Delete an installed model profile from this workspace")
    delete_cmd.add_argument("model_id", nargs="?")
    delete_cmd.add_argument("--backend", choices=SUPPORTED_BACKENDS)
    delete_cmd.add_argument("--yes", action="store_true", help="Skip confirmation")
    delete_cmd.add_argument(
        "--purge-cache",
        action="store_true",
        help="Also remove this profile's primary Hugging Face repository from the global HF cache when unshared",
    )
    delete_cmd.set_defaults(func=cmd_delete)

    status_cmd = sub.add_parser("status", help="Show the active model/runtime selection")
    status_cmd.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        mutates_artifacts = args.command in {"setup", "install", "update", "delete"} or (
            args.command == "select" and bool(getattr(args, "install", False))
        )
        if mutates_artifacts:
            config_path = _config_path(args.config)
            lock_path = config_path.parent / ".deqio" / "model-management.lock"
            with process_lock(
                lock_path,
                timeout=0.0,
                live_owner_error="Another Deqio model-management operation is already running in this workspace (pid={pid})",
            ):
                running_server = discover_server(config_path)
                if running_server is not None:
                    raise RuntimeError(
                        "Stop the active Deqio server before installing, updating, or deleting model artifacts "
                        f"in this workspace (pid={running_server.get('pid')})."
                    )
                return int(args.func(args))
        return int(args.func(args))
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
