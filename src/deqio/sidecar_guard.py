from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Sequence


def _pid_alive(pid: int) -> bool:
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


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _wait_for_process_group_exit(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_group_alive(pgid):
            return True
        time.sleep(0.05)
    return not _process_group_alive(pgid)


def _terminate_tree(process: subprocess.Popen[object]) -> None:
    if os.name == "nt":
        # Windows lacks POSIX process groups here.  The guard can reliably own
        # and terminate the direct child; target-Windows process-tree behavior
        # still requires platform validation before being claimed as equivalent.
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=4)
            return
        except subprocess.TimeoutExpired:
            process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        return

    # The engine was started with start_new_session=True, therefore its PID is
    # also the process-group ID.  Do not return merely because the group leader
    # exited: a wrapper can die while a model-server grandchild remains alive in
    # the same group and continues holding RAM/Metal/CUDA resources.
    pgid = process.pid
    deadline = time.monotonic() + 4.0
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        if process.poll() is None:
            process.terminate()

    # Reap the direct child promptly. A zombie group leader otherwise makes
    # killpg(..., 0) report the group as alive for the whole grace period.
    if process.poll() is None:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass

    remaining = max(0.0, deadline - time.monotonic())
    if _wait_for_process_group_exit(pgid, remaining):
        return

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        if process.poll() is None:
            process.kill()
    if process.poll() is None:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    _wait_for_process_group_exit(pgid, 2.0)


def run(parent_pid: int, command: Sequence[str], *, control_stdin: bool = False) -> int:
    if parent_pid <= 0:
        raise ValueError("parent_pid must be positive")
    if not command:
        raise ValueError("sidecar command must not be empty")

    stopping = False
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    # The parent owns stdin as a private control pipe.  A literal ``stop``
    # requests a graceful shutdown; EOF means the parent disappeared without
    # closing the runtime cleanly.  This works on Windows as well as POSIX and
    # avoids relying on signal delivery semantics for normal shutdown.
    def watch_parent_pipe() -> None:
        try:
            for line in sys.stdin:
                if line.strip().lower() == "stop":
                    stop_event.set()
                    return
        finally:
            stop_event.set()

    if control_stdin:
        threading.Thread(
            target=watch_parent_pipe,
            name="deqio-sidecar-parent-pipe",
            daemon=True,
        ).start()

    popen_kwargs: dict[str, object] = {}
    if os.name != "nt":
        # The actual engine and everything it spawns get their own process
        # group so the guard can reliably terminate the complete runtime tree.
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(list(command), **popen_kwargs)
    try:
        while True:
            code = process.poll()
            if code is not None:
                return int(code)
            if stopping or stop_event.is_set():
                return 0
            # Checking both PPID and process existence makes parent death
            # detection immediate on POSIX and still useful on Windows.
            if os.getppid() != parent_pid or not _pid_alive(parent_pid):
                return 0
            time.sleep(0.2)
    finally:
        _terminate_tree(process)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Supervise one Deqio inference sidecar.")
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--control-stdin", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    return run(args.parent_pid, command, control_stdin=bool(args.control_stdin))


if __name__ == "__main__":
    raise SystemExit(main())
