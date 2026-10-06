from __future__ import annotations

import json
import os
import secrets
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .process_lock import pid_alive, process_lock

CONTROL_SCHEMA_VERSION = 1
CONTROL_FILENAME = "server-control.json"
CONTROL_LOCK_FILENAME = ".server-control.lock"
CONTROL_HEADER = "X-Deqio-Control-Token"


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


def register_server(config_path: Path, *, host: str, port: int, pid: int | None = None) -> dict[str, Any]:
    path = control_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current_pid = int(pid or os.getpid())
    lock_dir = path.parent / CONTROL_LOCK_FILENAME
    with process_lock(lock_dir, timeout=5.0):
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                existing_pid = int(existing.get("pid", 0)) if isinstance(existing, dict) else 0
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                existing_pid = 0
            if existing_pid > 0 and existing_pid != current_pid and pid_alive(existing_pid):
                raise RuntimeError(
                    f"Another Deqio server is already running for this workspace (pid={existing_pid}). "
                    "Stop that server before starting a second one."
                )

        token = secrets.token_urlsafe(32)
        payload = {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "workspace": str(Path(config_path).expanduser().resolve().parent),
            "pid": current_pid,
            "host": str(host),
            "port": int(port),
            "base_url": _base_url(host, port),
            "token": token,
        }
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
    return payload


def unregister_server(config_path: Path, *, pid: int | None = None) -> None:
    path = control_path(config_path)
    lock_dir = path.parent / CONTROL_LOCK_FILENAME
    try:
        with process_lock(lock_dir, timeout=5.0):
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                return
            if pid is not None and int(current.get("pid", -1)) != int(pid):
                return
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    except RuntimeError:
        # Shutdown must not fail solely because another process is completing a
        # short control-file registration critical section. A stale control file
        # is self-healed by discover/register on the next operation.
        return


def discover_server(config_path: Path) -> dict[str, Any] | None:
    path = control_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / CONTROL_LOCK_FILENAME
    # Read and stale-file cleanup use the same kernel lock as registration and
    # unregistration. Otherwise a discoverer could read an old dead PID, a new
    # server could replace the file, and the discoverer could then unlink the
    # new server's control record based on the stale read.
    with process_lock(lock_path, timeout=5.0):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != CONTROL_SCHEMA_VERSION:
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
        if not pid_alive(pid):
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
