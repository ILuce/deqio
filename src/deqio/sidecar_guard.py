from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

# The guard is launched with the Deqio interpreter, so the Deqio package is the
# one dependency it may rely on. One liveness probe is shared with the server
# and the runtime-control code; it must never signal the probed process.
from deqio.process_lock import pgid_alive, pid_alive


def _wait_for_process_group_exit(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pgid_alive(pgid):
            return True
        time.sleep(0.05)
    return not pgid_alive(pgid)


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


def run(
    parent_pid: int,
    command: Sequence[str],
    *,
    control_stdin: bool = False,
    engine_pid_file: Path | None = None,
) -> int:
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
        if engine_pid_file is not None:
            # The engine tree lives in its own session, which the Deqio parent
            # cannot address through this guard's process group. Publishing the
            # engine PID lets the parent reap that tree if the guard itself is
            # killed before it can run its own cleanup.
            engine_pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
        while True:
            code = process.poll()
            if code is not None:
                return int(code)
            if stopping or stop_event.is_set():
                return 0
            # Checking both PPID and process existence makes parent death
            # detection immediate on POSIX and still useful on Windows.
            if os.getppid() != parent_pid or not pid_alive(parent_pid):
                return 0
            time.sleep(0.2)
    finally:
        _terminate_tree(process)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Supervise one Deqio inference sidecar.")
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--control-stdin", action="store_true")
    parser.add_argument("--engine-pid-file", type=Path, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    return run(
        args.parent_pid,
        command,
        control_stdin=bool(args.control_stdin),
        engine_pid_file=args.engine_pid_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
