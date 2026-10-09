from __future__ import annotations

import json
import os
import secrets
import tempfile
import urllib.error
import urllib.request
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .process_lock import pid_alive, process_lock

CONTROL_SCHEMA_VERSION = 1
CONTROL_FILENAME = "server-control.json"
CONTROL_LOCK_FILENAME = ".server-control.lock"
OWNER_LOCK_FILENAME = ".server-owner.lock"
MODEL_MANAGEMENT_LOCK_FILENAME = "model-management.lock"
CONTROL_HEADER = "X-Deqio-Control-Token"

SERVER_ALREADY_RUNNING = (
    "Another Deqio server is already running for this workspace (pid={pid}). "
    "Stop that server before starting a second one."
)
MODEL_MANAGEMENT_BUSY = (
    "A Deqio model-management operation or benchmark is running in this workspace "
    "(pid={pid}); wait for it to finish before starting deqio serve."
)

# Ownership leases held by this process, keyed by control-file path. A lease is
# an exclusive kernel lock held for as long as the server is registered; the
# kernel drops it on any exit (crash, SIGKILL, OOM), so a recycled PID can never
# make a dead server look alive (B7).
_OWNER_LEASES: dict[str, ExitStack] = {}


def control_path(config_path: Path) -> Path:
    return Path(config_path).expanduser().resolve().parent / ".deqio" / CONTROL_FILENAME


def _control_host(bind_host: str) -> str:
    host = str(bind_host or "127.0.0.1").strip()
    if host in {"0.0.0.0", "::", "[::]", ""}:
        return "127.0.0.1"
    return host


def _base_url(host: str, port: int) -> str:
    host = _control_host(host)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{int(port)}"


def model_management_lock_path(config_path: Path) -> Path:
    """Lock held by `deqio models` mutations and benchmarks for their whole run."""
    return control_path(config_path).parent / MODEL_MANAGEMENT_LOCK_FILENAME


def ensure_model_management_idle(config_path: Path) -> None:
    """Fail fast when a model-management operation or benchmark owns the workspace."""
    with process_lock(
        model_management_lock_path(config_path), timeout=0.0, live_owner_error=MODEL_MANAGEMENT_BUSY
    ):
        pass


def _read_control(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _control_pid(payload: dict[str, Any] | None) -> int:
    try:
        return int((payload or {}).get("pid", 0))
    except (TypeError, ValueError):
        return 0


def _owner_lock_path(path: Path) -> Path:
    return path.parent / OWNER_LOCK_FILENAME


def _owner_lock_is_free(path: Path) -> bool:
    """Whether no live process holds the server ownership lock (call under the control lock)."""
    try:
        with process_lock(_owner_lock_path(path), timeout=0.0):
            return True
    except RuntimeError:
        return False


def _release_owner_lease(path: Path) -> None:
    lease = _OWNER_LEASES.pop(str(path), None)
    if lease is not None:
        lease.close()


def _write_control(path: Path, payload: dict[str, Any]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=".server-control-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        try:
            os.chmod(temp_name, 0o600)
        except OSError:
            pass
        Path(temp_name).replace(path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    finally:
        try:
            Path(temp_name).unlink()
        except FileNotFoundError:
            pass


def register_server(config_path: Path, *, host: str, port: int, pid: int | None = None) -> dict[str, Any]:
    path = control_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current_pid = int(pid or os.getpid())
    lock_dir = path.parent / CONTROL_LOCK_FILENAME
    # `deqio models` mutations and benchmarks hold the model-management lock for
    # their whole run and refuse to start while a server is registered. Holding
    # it around the registration makes "nothing is installing or benchmarking"
    # and "this server owns the workspace" one atomic step (B8).
    with process_lock(
        model_management_lock_path(config_path), timeout=0.0, live_owner_error=MODEL_MANAGEMENT_BUSY
    ):
        with process_lock(lock_dir, timeout=5.0):
            lease = _OWNER_LEASES.get(str(path))
            new_lease = lease is None
            if lease is None:
                lease = ExitStack()
                lease.enter_context(
                    process_lock(_owner_lock_path(path), timeout=0.0, live_owner_error=SERVER_ALREADY_RUNNING)
                )
            try:
                existing = _read_control(path)
                existing_pid = _control_pid(existing)
                if existing_pid > 0 and existing_pid != current_pid and pid_alive(existing_pid):
                    # A freshly taken lease proves that a file written under the
                    # ownership lock belongs to a dead server whose PID was
                    # recycled. Files of older servers keep PID-based liveness.
                    if not (new_lease and existing is not None and existing.get("owner_lock") is True):
                        raise RuntimeError(SERVER_ALREADY_RUNNING.format(pid=existing_pid))
                payload = {
                    "schema_version": CONTROL_SCHEMA_VERSION,
                    "workspace": str(Path(config_path).expanduser().resolve().parent),
                    "pid": current_pid,
                    "host": str(host),
                    "port": int(port),
                    "base_url": _base_url(host, port),
                    "token": secrets.token_urlsafe(32),
                    "owner_lock": True,
                }
                _write_control(path, payload)
            except BaseException:
                if new_lease:
                    lease.close()
                raise
            _OWNER_LEASES[str(path)] = lease
    return payload


def unregister_server(config_path: Path, *, pid: int | None = None) -> None:
    path = control_path(config_path)
    lock_dir = path.parent / CONTROL_LOCK_FILENAME
    try:
        with process_lock(lock_dir, timeout=5.0):
            current = _read_control(path)
            if current is not None and pid is not None and _control_pid(current) != int(pid):
                return
            if current is not None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            _release_owner_lease(path)
    except RuntimeError:
        # Shutdown must not fail solely because another process is completing a
        # short control-file critical section. Releasing the ownership lease is
        # enough: a leftover file is then provably stale for discover/register.
        _release_owner_lease(path)


def discover_server(config_path: Path) -> dict[str, Any] | None:
    path = control_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / CONTROL_LOCK_FILENAME
    # Read and stale-file cleanup use the same kernel lock as registration and
    # unregistration. Otherwise a discoverer could read an old dead PID, a new
    # server could replace the file, and the discoverer could then unlink the
    # new server's control record based on the stale read.
    with process_lock(lock_path, timeout=5.0):
        payload = _read_control(path)
        if payload is None or int(payload.get("schema_version", 0)) != CONTROL_SCHEMA_VERSION:
            return None
        try:
            pid = int(payload["pid"])
            port = int(payload["port"])
            token = str(payload["token"])
            base_url = str(payload["base_url"])
        except (KeyError, TypeError, ValueError):
            return None
        if not token or not base_url or port < 1 or port > 65535:
            return None
        # B7: a dead PID, or -- for servers that hold the ownership lock -- a free
        # lock, which also catches a crashed server whose PID was recycled. The
        # HTTP port is no liveness signal: uvicorn binds it only after the
        # lifespan, i.e. after the model has loaded.
        if not pid_alive(pid) or (payload.get("owner_lock") is True and _owner_lock_is_free(path)):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return None
        expected_workspace = str(Path(config_path).expanduser().resolve().parent)
        if str(payload.get("workspace")) != expected_workspace:
            return None
        return payload


def _decode_response(response: Any) -> dict[str, Any]:
    raw = response.read()
    if not raw:
        return {}
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Deqio server returned a non-JSON runtime-control response") from error
    if not isinstance(value, dict):
        raise RuntimeError("Deqio server returned an invalid runtime-control response")
    return value


def control_request(
    server: dict[str, Any],
    action: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    if action not in {"suspend", "resume"}:
        raise ValueError(f"Unsupported runtime-control action: {action}")
    url = f"{server['base_url']}/v1/internal/runtime/{action}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            CONTROL_HEADER: str(server["token"]),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            return _decode_response(response)
    except urllib.error.HTTPError as error:
        try:
            detail = _decode_response(error)
        except RuntimeError:
            detail = {"detail": str(error)}
        message = detail.get("detail", detail)
        raise RuntimeError(f"Deqio server runtime-control {action} failed (HTTP {error.code}): {message}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach the active Deqio server for runtime-control {action}: {error}") from error
