from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download

from .catalog import SUPPORTED_BACKENDS, apply_selection, get_model, get_profile, load_catalog
from .config import read_config_data, write_config_data
from .installations import installed_profiles, mark_installed
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
        "--repo-id", str(profile.get("repo_id", "bespokelabs/Bespoke-Nimble-9B")),
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


def _install_native(config_path: Path, profile: dict[str, Any]) -> str:
    download = profile.get("download")
    if not isinstance(download, dict):
        return "Runtime is ready; remote weights will be resolved by the selected backend on first load."

    base = config_path.parent
    local_dir = Path(str(download["local_dir"])).expanduser()
    if not local_dir.is_absolute():
        local_dir = (base / local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    kind = download.get("type")
    if kind == "snapshot":
        snapshot_download(repo_id=str(download["repo_id"]), local_dir=str(local_dir))
        return f"Downloaded {download['repo_id']} -> {local_dir}"
    if kind == "file":
        path = hf_hub_download(
            repo_id=str(download["repo_id"]),
            filename=str(download["filename"]),
            local_dir=str(local_dir),
        )
        return f"Downloaded {download['repo_id']}/{download['filename']} -> {path}"
    raise RuntimeError(f"Unsupported download type: {kind!r}")


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

    print(f"{'MODEL ID':28} {'ENGINE':10} {'BACKENDS':22} DESCRIPTION")
    print("-" * 105)
    for entry in catalog["models"]:
        backends = sorted((entry.get("backends") or {}).keys())
        if backend_filter and backend_filter not in backends:
            continue
        print(
            f"{str(entry['id']):28} {str(entry['engine']):10} "
            f"{','.join(backends):22} {entry.get('description', '')}"
        )
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
    updated = _select_data(data, entry, profile, args.backend)
    _write_config(config_path, updated)
    print(f"Selected {args.model_id} on {args.backend} in {config_path}")
    if args.install:
        return cmd_install(
            argparse.Namespace(
                config=str(config_path),
                model_id=args.model_id,
                backend=args.backend,
                upgrade=False,
            )
        )
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
    entry = get_model(catalog, model_id)
    profile = get_profile(catalog, model_id, backend)

    env_dir = _install_runtime(config_path, data, profile, upgrade=bool(args.upgrade))
    print(f"Runtime ready: {env_dir}")
    if entry["engine"] == "semif":
        print(_install_native(config_path, profile))
    elif profile.get("installer") == "nimble":
        print("Nimble source, runtime dependencies and prepared model weights are ready.")
    else:
        print("Model weights are downloaded by the engine on first start unless already cached.")

    mark_installed(
        config_path,
        model_id,
        backend,
        verified=False,
        source="update" if bool(args.upgrade) else "install",
    )
    print(f"Registered installed profile: {model_id} / {backend}")
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


def _host_backends() -> tuple[str, ...]:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine == "arm64":
        return ("mlx", "mps")
    if system in {"Linux", "Windows"}:
        return ("cuda",)
    return SUPPORTED_BACKENDS


def cmd_setup(args: argparse.Namespace) -> int:
    config_path = _config_path(args.config)
    data = _read_config(config_path)
    catalog = load_catalog(_catalog_path(config_path, data))

    available_backends = _host_backends()
    print("Available backends for this host:")
    for index, backend in enumerate(available_backends, start=1):
        count = sum(backend in (item.get("backends") or {}) for item in catalog["models"])
        print(f"  {index}. {backend} ({count} compatible models)")
    backend_index = int(input(f"Select backend [1-{len(available_backends)}]: ").strip()) - 1
    if backend_index not in range(len(available_backends)):
        raise RuntimeError("Invalid backend selection")
    backend = available_backends[backend_index]

    compatible = [item for item in catalog["models"] if backend in (item.get("backends") or {})]
    print(f"\nModels supporting {backend}:")
    for index, item in enumerate(compatible, start=1):
        print(f"  {index}. {item['id']} — {item.get('label', item['id'])}")
    model_index = int(input(f"Select model [1-{len(compatible)}]: ").strip()) - 1
    if model_index not in range(len(compatible)):
        raise RuntimeError("Invalid model selection")
    entry = compatible[model_index]
    profile = get_profile(catalog, str(entry["id"]), backend)
    updated = _select_data(data, entry, profile, backend)
    _write_config(config_path, updated)
    print(f"\nSelected {entry['id']} / {backend}")

    answer = input("Install/download the selected runtime now? [Y/n]: ").strip().lower()
    if answer not in {"n", "no"}:
        return cmd_install(
            argparse.Namespace(
                config=str(config_path),
                model_id=str(entry["id"]),
                backend=backend,
                upgrade=False,
            )
        )
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
    install_cmd.set_defaults(func=cmd_install)

    update_cmd = sub.add_parser("update", help="Upgrade an isolated engine runtime")
    update_cmd.add_argument("model_id", nargs="?")
    update_cmd.add_argument("--backend", choices=SUPPORTED_BACKENDS)
    update_cmd.set_defaults(func=lambda a: cmd_install(argparse.Namespace(**vars(a), upgrade=True)))

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
