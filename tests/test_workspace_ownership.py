"""Workspace ownership between `deqio serve`, model management and benchmarks.

D10: audit items B7 (stale control file after a crash and PID reuse) and B8
(`deqio serve` must respect the model-management lock). The tests use real
processes: the kernel lock semantics are the behavior under test.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from deqio.process_lock import pid_alive
from deqio.runtime_control import control_path, discover_server, register_server, unregister_server

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX process semantics; Windows is unsupported in 0.5")

REGISTER_AND_HOLD = r'''
import sys
import time
from pathlib import Path

from deqio.runtime_control import register_server

register_server(Path(sys.argv[1]), host="127.0.0.1", port=int(sys.argv[2]))
Path(sys.argv[3]).write_text("registered")
time.sleep(120)
'''

HOLD_MODEL_MANAGEMENT_LOCK = r'''
import sys
import time
from pathlib import Path

from deqio.process_lock import process_lock

with process_lock(Path(sys.argv[1]), timeout=5.0):
    Path(sys.argv[2]).write_text("held")
    time.sleep(120)
'''

BUSY = "model-management operation or benchmark"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _config(tmp_path: Path) -> Path:
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    return config


def _model_lock(config: Path) -> Path:
    return config.resolve().parent / ".deqio" / "model-management.lock"


def _spawn(code: str, *args: object) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", code, *map(str, args)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for(path: Path, process: subprocess.Popen, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            raise AssertionError(f"helper process exited early with code {process.returncode}")
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


def _kill(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        process.kill()
    if process is not None:
        process.wait(timeout=10)


# --- B7: liveness of the control file -----------------------------------------


def test_registered_server_that_is_still_loading_its_model_stays_discoverable(tmp_path: Path) -> None:
    """uvicorn binds its port only after lifespan startup, i.e. after the model load.

    A loading server therefore refuses TCP connections for minutes. Treating a
    refused connection as a stale control file would let `models update`, a
    second `serve` or a benchmark run next to it.
    """
    config = _config(tmp_path)
    ready = tmp_path / "registered"
    server = _spawn(REGISTER_AND_HOLD, config, _free_port(), ready)
    try:
        _wait_for(ready, server)
        found = discover_server(config)
        assert found is not None and found["pid"] == server.pid
        assert control_path(config).is_file()
    finally:
        _kill(server)


def test_discovery_detects_a_crashed_server_whose_pid_was_recycled(tmp_path: Path) -> None:
    config = _config(tmp_path)
    ready = tmp_path / "registered"
    server = _spawn(REGISTER_AND_HOLD, config, _free_port(), ready)
    unrelated: subprocess.Popen | None = None
    try:
        _wait_for(ready, server)
        os.kill(server.pid, signal.SIGKILL)
        server.wait(timeout=10)
        # The kernel hands the crashed server's PID to an unrelated process.
        unrelated = _spawn("import time; time.sleep(120)")
        path = control_path(config)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["pid"] = unrelated.pid
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert pid_alive(unrelated.pid)

        assert discover_server(config) is None
        assert not path.exists()
        # A new server may take over the workspace instead of being refused.
        registration = register_server(config, host="127.0.0.1", port=8787, pid=os.getpid())
        try:
            assert discover_server(config)["token"] == registration["token"]
        finally:
            unregister_server(config, pid=os.getpid())
        assert discover_server(config) is None
    finally:
        _kill(server)
        _kill(unrelated)


def test_control_file_of_an_older_server_keeps_pid_based_liveness(tmp_path: Path) -> None:
    """0.5.4 servers hold no ownership lock: never call their live PID stale."""
    config = _config(tmp_path)
    holder = _spawn("import time; time.sleep(120)")
    try:
        path = control_path(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema_version": 1, "workspace": str(config.resolve().parent), "pid": holder.pid,
            "host": "127.0.0.1", "port": 8787, "base_url": "http://127.0.0.1:8787", "token": "legacy",
        }), encoding="utf-8")
        found = discover_server(config)
        assert found is not None and found["pid"] == holder.pid
    finally:
        _kill(holder)


# --- B8: `deqio serve` and the model-management lock --------------------------


def test_server_registration_is_refused_while_model_management_runs(tmp_path: Path) -> None:
    """Registration and the model-management check are one atomic step (no TOCTOU)."""
    config = _config(tmp_path)
    held = tmp_path / "held"
    holder = _spawn(HOLD_MODEL_MANAGEMENT_LOCK, _model_lock(config), held)
    try:
        _wait_for(held, holder)
        with pytest.raises(RuntimeError, match=BUSY):
            register_server(config, host="127.0.0.1", port=8787)
        assert not control_path(config).exists()
    finally:
        _kill(holder)
    registration = register_server(config, host="127.0.0.1", port=8787)
    unregister_server(config, pid=registration["pid"])


def test_serve_refuses_to_start_during_model_management(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from deqio import cli
    from deqio.workspace import ensure_workspace

    config = ensure_workspace(tmp_path / "ws")
    monkeypatch.setenv("DEQIO_CONFIG", str(config))
    started: list[object] = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: started.append(args))
    held = tmp_path / "held"
    holder = _spawn(HOLD_MODEL_MANAGEMENT_LOCK, _model_lock(config), held)
    try:
        _wait_for(held, holder)
        assert cli.main(["serve", "--port", str(_free_port())]) == 2
    finally:
        _kill(holder)
    assert started == []
    assert BUSY in capsys.readouterr().err


def test_real_deqio_serve_exits_cleanly_while_a_model_update_holds_the_workspace(tmp_path: Path) -> None:
    from deqio.workspace import ensure_workspace

    config = ensure_workspace(tmp_path / "ws")
    held = tmp_path / "held"
    holder = _spawn(HOLD_MODEL_MANAGEMENT_LOCK, _model_lock(config), held)
    try:
        _wait_for(held, holder)
        served = subprocess.run(
            [sys.executable, "-m", "deqio", "serve", "--port", str(_free_port())],
            cwd=config.parent,
            env={**os.environ, "DEQIO_CONFIG": str(config)},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert served.returncode == 2, served.stdout + served.stderr
        assert BUSY in served.stderr
        assert not control_path(config).exists()
    finally:
        _kill(holder)
