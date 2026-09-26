from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, scan_cache_dir, snapshot_download

from .catalog import SUPPORTED_BACKENDS, apply_selection, get_model, get_profile, load_catalog
from .backends import BackendRuntime
from .config import read_config_data, settings_from_data, write_config_data
from .hardware import describe_host, detect_host, host_backends, profile_compatibility
from .installations import installed_profiles, load_registry, mark_installed, profile_key, unmark_installed
from .workspace import ensure_workspace


ROOT = Path.cwd()


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


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def _write_config(path: Path, data: dict[str, Any]) -> None:
    write_config_data(path, data)


def _select_data(
    data: dict[str, Any], model_entry: dict[str, Any], profile: dict[str, Any], backend: str
) -> dict[str, Any]:
    return apply_selection(data, model_entry, profile, backend)


def _ensure_venv(env_dir: Path, python_version: str) -> Path:
    python_path = _runtime_python(env_dir)
    if not python_path.is_file():
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["uv", "python", "install", python_version])
        _run(["uv", "venv", str(env_dir), "--python", python_version])
    return python_path


def _checkout_nimble(runtime_root: Path, *, upgrade: bool) -> Path:
    source_dir = runtime_root / "nimble-src"
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
    runtime_root.mkdir(parents=True, exist_ok=True)
    source_dir = _checkout_nimble(runtime_root, upgrade=upgrade)
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

    model_dir = Path(str(profile.get("model", "models/nimble-9b"))).expanduser()
    if not model_dir.is_absolute():
        model_dir = (config_path.parent / model_dir).resolve()
    model_config = runtime_root / str(profile.get("model_config", "nimble-model.json"))
    helper = Path(__file__).with_name("nimble_prepare.py").resolve()
    prepare = [
        str(prep_python), str(helper),
        "--source-root", str(source_dir),
        "--repo-id", str(profile.get("repo_id", "bespokelabs/Bespoke-Nimble-9B-v2")),
        "--output-dir", str(model_dir),
        "--config", str(model_config),
    ]
    if upgrade:
        prepare.append("--force")
    _run(prepare)
    return env_dir


def _install_runtime(config_path: Path, data: dict[str, Any], profile: dict[str, Any], *, upgrade: bool) -> Path:
    if profile.get("installer") == "nimble":
        return _install_nimble(config_path, data, profile, upgrade=upgrade)

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
        local_dir.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(local_dir)
    if kind == "snapshot":
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
) -> None:
    selected = _select_data(data, entry, profile, backend)
    # Installation is the one lifecycle phase where network access is allowed.
    # The readiness probe forces the engine to resolve every transitive/base
    # model dependency now, before the profile is registered as installed.
    selected["hf_offline_runtime"] = False
    settings = settings_from_data(config_path, selected, apply_environment=False)
    print(
        f"Verifying model readiness online (timeout={settings.sidecar_startup_seconds}s): "
        f"{entry['id']} / {backend}"
    )
    runtime = BackendRuntime.load(settings)
    try:
        # SystemOneRuntime.load already performs a real typed-decision readiness
        # probe. Reaching this point means the model can answer requests.
        pass
    finally:
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
) -> None:
    entry = get_model(catalog, model_id)
    profile = get_profile(catalog, model_id, backend)
    _preflight_profile(model_id, backend, profile, force=force)

    env_dir = _install_runtime(config_path, data, profile, upgrade=upgrade)
    print(f"Runtime ready: {env_dir}")
    if profile.get("installer") == "nimble":
        print("Nimble source, runtime dependencies and prepared model weights are ready.")
    else:
        print(_prefetch_declared_weights(config_path, profile))

    _verify_model_ready(config_path, data, entry, profile, backend)
    mark_installed(
        config_path,
        model_id,
        backend,
        verified=True,
        source="update" if upgrade else "install",
    )
    print(f"Installed and verified profile: {model_id} / {backend}")


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
        )
    updated = _select_data(data, entry, profile, args.backend)
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
        value = input(f"Select profile [1-{len(rows)}]: ").strip()
        index = int(value) - 1
        if index not in range(len(rows)):
            raise RuntimeError("Invalid model selection")
        selected = rows[index]

    entry = get_model(catalog, str(selected["model_id"]))
    profile = get_profile(catalog, str(selected["model_id"]), str(selected["backend"]))
    updated = _select_data(data, entry, profile, str(selected["backend"]))
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
    paths: set[Path] = set()
    for download in _declared_downloads(profile):
        if download.get("local_dir"):
            path = Path(str(download["local_dir"])).expanduser()
            paths.add((path if path.is_absolute() else config_path.parent / path).resolve())
    model = profile.get("model")
    if isinstance(model, str) and (model.startswith(("./", "../", "~/", "/", "models/"))):
        path = Path(model).expanduser()
        paths.add((path if path.is_absolute() else config_path.parent / path).resolve())
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
            for name in ("nimble-src", "nimble-prep", str(profile.get("model_config", "nimble-model.json"))):
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
        value = input(f"Select profile to delete [1-{len(rows)}]: ").strip()
        index = int(value) - 1
        if index not in range(len(rows)):
            raise RuntimeError("Invalid model selection")
        selected = rows[index]

    model_id = str(selected["model_id"])
    backend = str(selected["backend"])
    if not args.yes:
        answer = input(f"Delete {model_id} / {backend} from this Deqio workspace? [y/N]: ").strip().lower()
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
                _select_data(data, entry, replacement_profile, str(replacement["backend"])),
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
            "No supported accelerator backend was detected. Apple Silicon requires macOS; "
            "Linux/Windows CUDA requires a working NVIDIA driver visible through nvidia-smi."
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
    backend_index = int(input(f"Select backend [1-{len(available_backends)}]: ").strip()) - 1
    if backend_index not in range(len(available_backends)):
        raise RuntimeError("Invalid backend selection")
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

    model_index = int(input(f"Select model [1-{len(compatible)}]: ").strip()) - 1
    if model_index not in range(len(compatible)):
        raise RuntimeError("Invalid model selection")
    entry = compatible[model_index][0]
    profile = get_profile(catalog, str(entry["id"]), backend)

    print(f"\nInstalling {entry['id']} / {backend}")
    _install_profile(
        config_path=config_path,
        data=data,
        catalog=catalog,
        model_id=str(entry["id"]),
        backend=backend,
        upgrade=False,
        force=False,
    )
    updated = _select_data(data, entry, profile, backend)
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
    install_cmd.set_defaults(func=cmd_install)

    update_cmd = sub.add_parser("update", help="Upgrade an isolated engine runtime")
    update_cmd.add_argument("model_id", nargs="?")
    update_cmd.add_argument("--backend", choices=SUPPORTED_BACKENDS)
    update_cmd.add_argument("--force", action="store_true", help="Bypass host memory/backend compatibility checks")
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
        return int(args.func(args))
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
