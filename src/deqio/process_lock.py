from __future__ import annotations

import json
import os
import shutil
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO
from uuid import uuid4


LEGACY_OWNERLESS_STALE_SECONDS = 60.0


def pid_alive(pid: int) -> bool:
    """Return whether *pid* currently exists (or exists but is not signalable)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_owner(handle: TextIO) -> dict[str, Any] | None:
    try:
        handle.seek(0)
        value = json.loads(handle.read() or "{}")
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_owner(handle: TextIO, owner: dict[str, Any]) -> None:
    # Keep the locked inode/range intact while refreshing diagnostic metadata.
    # This matters on Windows where msvcrt.locking() locks a byte range.
    payload = json.dumps(owner, ensure_ascii=False, indent=2) + "\n"
    handle.seek(0)
    handle.write(payload)
    handle.truncate()
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError:
        pass


def _try_lock(handle: TextIO) -> bool:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(handle: TextIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _legacy_owner(path: Path) -> tuple[dict[str, Any] | None, int]:
    owner: dict[str, Any] | None = None
    try:
        value = json.loads((path / "owner.json").read_text(encoding="utf-8"))
        owner = value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        pass
    try:
        owner_pid = int((owner or {}).get("pid", 0))
    except (TypeError, ValueError):
        owner_pid = 0
    return owner, owner_pid


def _migrate_legacy_lock_directory(
    path: Path,
    *,
    live_owner_error: str | None,
    initialization_grace: float,
) -> None:
    """Safely reap a directory-style lock left by pre-release Deqio builds.

    Old Watch locks had no owner metadata and old benchmark locks created the
    directory shortly before writing ``owner.json``.  A recent ownerless
    directory is therefore ambiguous and must never be deleted optimistically.
    """
    if not path.is_dir():
        return

    _, owner_pid = _legacy_owner(path)
    if owner_pid > 0 and pid_alive(owner_pid):
        if live_owner_error is not None:
            raise RuntimeError(live_owner_error.format(pid=owner_pid))
        raise RuntimeError(f"Process lock is held by a legacy Deqio process (pid={owner_pid}): {path}")
    if owner_pid > 0:
        shutil.rmtree(path)
        return

    grace = max(0.0, float(initialization_grace))
    if grace:
        try:
            age = max(0.0, time.time() - path.stat().st_mtime)
        except OSError as error:
            raise RuntimeError(f"Cannot inspect legacy process lock: {path}: {error}") from error
        if age < grace:
            time.sleep(grace - age)
            _, owner_pid = _legacy_owner(path)
            if owner_pid > 0 and pid_alive(owner_pid):
                if live_owner_error is not None:
                    raise RuntimeError(live_owner_error.format(pid=owner_pid))
                raise RuntimeError(
                    f"Process lock is held by a legacy Deqio process (pid={owner_pid}): {path}"
                )
            if owner_pid > 0:
                shutil.rmtree(path)
                return

    try:
        age = max(0.0, time.time() - path.stat().st_mtime)
    except OSError as error:
        raise RuntimeError(f"Cannot inspect legacy process lock: {path}: {error}") from error
    if age < LEGACY_OWNERLESS_STALE_SECONDS:
        raise RuntimeError(
            f"Legacy process lock has no valid owner metadata and is too recent to reap safely: {path}"
        )
    shutil.rmtree(path)


def _open_lock_file(path: Path) -> TextIO:
    """Open one regular lock inode without following a final-component symlink."""
    if path.is_symlink():
        raise RuntimeError(f"Refusing symlink process lock path: {path}")

    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        fd = os.open(path, flags, 0o600)
    except OSError as error:
        raise RuntimeError(f"Cannot safely open process lock: {path}: {error}") from error

    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"Process lock path is not a regular file: {path}")
        try:
            os.fchmod(fd, 0o600)
        except (AttributeError, OSError):
            pass
        if info.st_size == 0:
            os.write(fd, b"\n")
            os.fsync(fd)
        return os.fdopen(fd, "r+", encoding="utf-8")
    except Exception:
        os.close(fd)
        raise


@contextmanager
def process_lock(
    lock_path: Path,
    *,
    timeout: float = 10.0,
    initialization_grace: float = 0.0,
    live_owner_error: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Acquire a crash-safe cross-process lock.

    The kernel lock is the ownership primitive, so it is released automatically
    when a process exits, including SIGKILL/crash paths. The file itself stays on
    disk intentionally: unlinking a live lock file can create split-brain locks
    on different inodes. JSON content is diagnostic metadata only.
    """
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_lock_directory(
        path,
        live_owner_error=live_owner_error,
        initialization_grace=initialization_grace,
    )

    deadline = time.monotonic() + max(0.0, float(timeout))
    handle = _open_lock_file(path)
    acquired = False
    try:
        while not acquired:
            acquired = _try_lock(handle)
            if acquired:
                break

            owner = _read_owner(handle) or {}
            try:
                owner_pid = int(owner.get("pid", 0))
            except (TypeError, ValueError):
                owner_pid = 0
            if live_owner_error is not None:
                pid_display: int | str = owner_pid if owner_pid > 0 else "unknown"
                raise RuntimeError(live_owner_error.format(pid=pid_display))
            if time.monotonic() >= deadline:
                suffix = f" (pid={owner_pid})" if owner_pid > 0 else ""
                raise RuntimeError(f"Timed out waiting for process lock: {path}{suffix}")
            time.sleep(0.01)

        owner = {
            "pid": os.getpid(),
            "lock_id": uuid4().hex,
            "created_at": time.time(),
        }
        _write_owner(handle, owner)
        yield owner
    finally:
        if acquired:
            _unlock(handle)
        handle.close()
