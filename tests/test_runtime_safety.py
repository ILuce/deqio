from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


def test_process_lock_blocks_competing_process_and_releases(tmp_path: Path) -> None:
    from deqio.process_lock import process_lock

    lock_path = tmp_path / "runtime.lock"
    script = r'''
import sys
from pathlib import Path
from deqio.process_lock import process_lock

try:
    with process_lock(Path(sys.argv[1]), timeout=0.2, live_owner_error="busy pid={pid}"):
        print("acquired")
except RuntimeError as error:
    print(str(error))
    raise SystemExit(3)
'''
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}

    with process_lock(lock_path, timeout=1.0):
        blocked = subprocess.run(
            [sys.executable, "-c", script, str(lock_path)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert blocked.returncode == 3
        assert "busy pid=" in blocked.stdout

    acquired = subprocess.run(
        [sys.executable, "-c", script, str(lock_path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert acquired.returncode == 0
    assert "acquired" in acquired.stdout


@pytest.mark.skipif(os.name == "nt", reason="SIGKILL parent-death integration test is POSIX-specific")
def test_sidecar_guard_terminates_engine_when_deqio_parent_is_killed(tmp_path: Path) -> None:
    from deqio.process_lock import pid_alive

    guard = Path("src/deqio/sidecar_guard.py").resolve()
    guard_pid_file = tmp_path / "guard.pid"
    child_pid_file = tmp_path / "child.pid"
    helper_code = r'''
import os
import subprocess
import sys
import time
from pathlib import Path

guard = Path(sys.argv[1])
guard_pid_file = Path(sys.argv[2])
child_pid_file = Path(sys.argv[3])
child_code = "import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
guard_process = subprocess.Popen([
    sys.executable,
    str(guard),
    "--parent-pid",
    str(os.getpid()),
    "--",
    sys.executable,
    "-c",
    child_code,
    str(child_pid_file),
])
guard_pid_file.write_text(str(guard_process.pid))
time.sleep(60)
'''
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            helper_code,
            str(guard),
            str(guard_pid_file),
            str(child_pid_file),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_until(lambda: guard_pid_file.is_file() and child_pid_file.is_file())
        child_pid = int(child_pid_file.read_text())
        assert pid_alive(child_pid)

        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5)

        assert _wait_until(lambda: not pid_alive(child_pid), timeout=6.0)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child_pid_file.is_file():
            child_pid = int(child_pid_file.read_text())
            if pid_alive(child_pid):
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_sidecar_guard_control_pipe_stops_engine_tree(tmp_path: Path) -> None:
    from deqio.process_lock import pid_alive

    guard = Path("src/deqio/sidecar_guard.py").resolve()
    child_pid_file = tmp_path / "child.pid"
    child_code = (
        "import os,sys,time; from pathlib import Path; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    )
    guarded = subprocess.Popen(
        [
            sys.executable,
            str(guard),
            "--parent-pid",
            str(os.getpid()),
            "--control-stdin",
            "--",
            sys.executable,
            "-c",
            child_code,
            str(child_pid_file),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert _wait_until(child_pid_file.is_file)
        child_pid = int(child_pid_file.read_text())
        assert pid_alive(child_pid)
        assert guarded.stdin is not None
        guarded.stdin.write("stop\n")
        guarded.stdin.flush()
        assert guarded.wait(timeout=7) == 0
        guarded.stdin.close()
        assert _wait_until(lambda: not pid_alive(child_pid), timeout=3.0)
    finally:
        if guarded.poll() is None:
            guarded.kill()
            guarded.wait(timeout=5)
        if child_pid_file.is_file():
            child_pid = int(child_pid_file.read_text())
            if pid_alive(child_pid):
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_runtime_error_status_distinguishes_client_and_engine_failures() -> None:
    import deqio.server as server
    from deqio.systemone_runtime import (
        SystemOneCapabilityError,
        SystemOneProtocolError,
        SystemOneSidecarHTTPError,
        SystemOneUnavailableError,
    )

    assert server._runtime_error_status(SystemOneCapabilityError("multi", "unsupported")) == 422
    assert server._runtime_error_status(SystemOneUnavailableError("timeout")) == 503
    assert server._runtime_error_status(SystemOneProtocolError("bad response")) == 502
    assert server._runtime_error_status(SystemOneSidecarHTTPError(503, "loading")) == 503
    assert server._runtime_error_status(SystemOneSidecarHTTPError(500, "failed")) == 502
    assert server._runtime_error_status(SystemOneSidecarHTTPError(422, "invalid")) == 422
    assert server._runtime_error_status(ValueError("bad request")) == 400
    assert server._runtime_error_status(RuntimeError("internal failure")) == 500


def test_benchmark_cli_deduplicates_identical_model_flags() -> None:
    import argparse
    from deqio import benchmark

    rows = [
        {"model_id": "demo", "backend": "mlx", "label": "Demo", "engine": "demo"},
        {"model_id": "other", "backend": "mps", "label": "Other", "engine": "other"},
    ]
    args = argparse.Namespace(all=False, model=["demo:mlx", "demo:mlx", "other:mps"])
    selected = benchmark._select_profiles(rows, args)
    assert selected is not None
    assert [(row["model_id"], row["backend"]) for row in selected] == [
        ("demo", "mlx"),
        ("other", "mps"),
    ]


def test_release_dependencies_have_compatible_bounds() -> None:
    import tomllib

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert "fastapi>=0.110,<1" in project["dependencies"]
    assert "uvicorn[standard]>=0.27,<1" in project["dependencies"]
    assert "huggingface-hub>=1,<2" in project["dependencies"]
    assert "pydantic>=2,<3" in project["dependencies"]
    assert "pytest>=8,<10" in project["optional-dependencies"]["dev"]
    assert "httpx>=0.27,<1" in project["optional-dependencies"]["dev"]

@pytest.mark.skipif(os.name == "nt", reason="SIGKILL lock-release integration test is POSIX-specific")
def test_process_lock_is_released_by_kernel_after_sigkill(tmp_path: Path) -> None:
    from deqio.process_lock import process_lock

    lock_path = tmp_path / "crash.lock"
    ready = tmp_path / "ready"
    script = r'''
import sys
import time
from pathlib import Path
from deqio.process_lock import process_lock

lock_path = Path(sys.argv[1])
ready = Path(sys.argv[2])
with process_lock(lock_path, timeout=1.0):
    ready.write_text("ready")
    time.sleep(60)
'''
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    owner = subprocess.Popen(
        [sys.executable, "-c", script, str(lock_path), str(ready)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_until(ready.is_file)
        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout=5)
        with process_lock(lock_path, timeout=1.0):
            pass
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)


def test_systemone_endpoint_offloads_blocking_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    calls: list[str] = []

    def fake_execute(payload, *, endpoint):
        calls.append(f"execute:{endpoint}")
        return {"ok": True}

    async def fake_to_thread(func, *args, **kwargs):
        calls.append("to_thread")
        return func(*args, **kwargs)

    monkeypatch.setattr(server, "_execute_system_one", fake_execute)
    monkeypatch.setattr(server.asyncio, "to_thread", fake_to_thread)
    payload = server.SystemOneRequest(
        state="state",
        questions={"ready": {"type": "noul", "instructions": "Ready?"}},
    )

    result = asyncio.run(server.system_one(payload))

    assert result == {"ok": True}
    assert calls == ["to_thread", "execute:/v1/systemone"]


def test_contract_rejects_blank_ids_before_inference() -> None:
    import deqio.server as server

    with pytest.raises(server.InputContractHTTPError) as error:
        server._validate_unique_ids([""], label="decision_ids")
    assert error.value.payload["code"] == "empty_id"
    assert error.value.payload["field"] == "decision_ids"
    assert error.value.payload["inference_performed"] is False


def test_watch_write_failure_is_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    class BrokenWatch:
        def append(self, *_args, **_kwargs):
            raise OSError("disk full")

    messages: list[str] = []
    monkeypatch.setattr(server, "_watch_store", lambda: BrokenWatch())
    monkeypatch.setattr(server, "info", messages.append)

    result = server._record_watch_event(
        expected_session_id="session",
        endpoint="/v1/noul",
        request_payload={"state": "x", "question": "ready?"},
        response_payload={"decision": "yes", "top_probability": 0.9},
        status_code=200,
        request_id="req",
        mode="serial",
        decisions=1,
        settings_snapshot=server.SETTINGS,
    )

    assert result is None
    assert any("Watch event write failed" in message for message in messages)


def test_watch_session_lookup_failure_is_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    import deqio.server as server

    class BrokenWatch:
        def session_token(self):
            raise OSError("watch unavailable")

    messages: list[str] = []
    monkeypatch.setattr(server, "_watch_store", lambda: BrokenWatch())
    monkeypatch.setattr(server, "info", messages.append)

    assert server._watch_session_token() is None
    assert any("Watch session lookup failed" in message for message in messages)


def test_console_output_failure_is_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    from deqio import console

    def broken_print(*_args, **_kwargs):
        raise BrokenPipeError("stdout closed")

    monkeypatch.setattr(builtins, "print", broken_print)
    console.info("still not fatal")


def test_managed_download_refuses_local_dir_outside_workspace_models(
    tmp_path: Path,
) -> None:
    from deqio.model_manager import _prefetch_one_download

    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside"

    with pytest.raises(RuntimeError, match="workspace models"):
        _prefetch_one_download(
            config_path,
            {"model_revision": "upstream-latest"},
            {
                "type": "snapshot",
                "repo_id": "owner/model",
                "local_dir": str(outside),
            },
        )

    assert not outside.exists()


def test_request_metadata_log_failure_does_not_fail_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses
    import deqio.server as server

    # Opening a directory as the JSONL file raises IsADirectoryError.
    settings = dataclasses.replace(server.SETTINGS, log_path=tmp_path)
    messages: list[str] = []
    monkeypatch.setattr(server, "info", messages.append)
    monkeypatch.setattr(server, "log_request_success", lambda *args, **kwargs: None)

    before_requests = server.stats["requests"]
    server.record_event(
        request_id="req-log-failure",
        mode="serial",
        state="state",
        question="question",
        result={"decision": "yes", "probabilities": {"yes": 1.0}, "timing": {"total_ms": 1.0}},
        endpoint="/v1/noul",
        settings_snapshot=settings,
    )

    assert server.stats["requests"] == before_requests + 1
    assert any("Request metadata log write failed" in message for message in messages)


def test_process_lock_rejects_symlink_without_touching_target(tmp_path: Path) -> None:
    from deqio.process_lock import process_lock

    victim = tmp_path / "victim.txt"
    victim.write_text("DO NOT TOUCH", encoding="utf-8")
    lock_path = tmp_path / "runtime.lock"
    try:
        lock_path.symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(RuntimeError, match="symlink process lock"):
        with process_lock(lock_path):
            pass

    assert victim.read_text(encoding="utf-8") == "DO NOT TOUCH"


def test_process_lock_does_not_reap_recent_ownerless_legacy_lock(tmp_path: Path) -> None:
    from deqio.process_lock import process_lock

    lock_path = tmp_path / "legacy.lock"
    lock_path.mkdir()

    with pytest.raises(RuntimeError, match="too recent to reap safely"):
        with process_lock(lock_path, initialization_grace=0.01):
            pass

    assert lock_path.is_dir()


def test_process_lock_reaps_stale_ownerless_legacy_lock(tmp_path: Path) -> None:
    from deqio.process_lock import LEGACY_OWNERLESS_STALE_SECONDS, process_lock

    lock_path = tmp_path / "legacy.lock"
    lock_path.mkdir()
    stale = time.time() - LEGACY_OWNERLESS_STALE_SECONDS - 5
    os.utime(lock_path, (stale, stale))

    with process_lock(lock_path):
        assert lock_path.is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group cleanup test")
def test_sidecar_guard_cleans_grandchild_after_engine_leader_exits(tmp_path: Path) -> None:
    from deqio.process_lock import pid_alive

    guard = Path("src/deqio/sidecar_guard.py").resolve()
    grandchild_pid_file = tmp_path / "grandchild.pid"
    wrapper_code = r'''
import subprocess
import sys
from pathlib import Path

pid_file = Path(sys.argv[1])
child = subprocess.Popen([
    sys.executable,
    "-c",
    "import os,time,sys; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)",
    str(pid_file),
])
while not pid_file.is_file():
    pass
# Exit deliberately without waiting for the grandchild.
'''
    guarded = subprocess.Popen(
        [
            sys.executable,
            str(guard),
            "--parent-pid",
            str(os.getpid()),
            "--",
            sys.executable,
            "-c",
            wrapper_code,
            str(grandchild_pid_file),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    grandchild_pid: int | None = None
    try:
        assert _wait_until(grandchild_pid_file.is_file)
        grandchild_pid = int(grandchild_pid_file.read_text())
        assert pid_alive(grandchild_pid)
        assert guarded.wait(timeout=8) == 0
        assert _wait_until(lambda: not pid_alive(grandchild_pid), timeout=3.0)
    finally:
        if guarded.poll() is None:
            guarded.kill()
            guarded.wait(timeout=5)
        if grandchild_pid is not None and pid_alive(grandchild_pid):
            try:
                os.kill(grandchild_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_config_atomic_write_does_not_follow_predictable_temp_symlink(tmp_path: Path) -> None:
    from deqio.config import write_config_data

    config = tmp_path / "config.json"
    config.write_text('{"old": true}\n', encoding="utf-8")
    victim = tmp_path / "victim.txt"
    victim.write_text("DO NOT TOUCH", encoding="utf-8")
    predictable = tmp_path / "config.json.tmp"
    try:
        predictable.symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")

    write_config_data(config, {"new": True})

    assert json.loads(config.read_text(encoding="utf-8")) == {"new": True}
    assert victim.read_text(encoding="utf-8") == "DO NOT TOUCH"


def test_registry_atomic_write_does_not_follow_predictable_temp_symlink(tmp_path: Path) -> None:
    from deqio.installations import mark_installed, registry_path

    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    path = registry_path(config)
    path.parent.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("DO NOT TOUCH", encoding="utf-8")
    predictable = path.with_suffix(path.suffix + ".tmp")
    try:
        predictable.symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")

    mark_installed(config, "demo", "mlx", verified=True)

    assert victim.read_text(encoding="utf-8") == "DO NOT TOUCH"
    registry = json.loads(path.read_text(encoding="utf-8"))
    assert "demo::mlx" in registry["profiles"]


def test_watch_store_prunes_old_rotated_files_to_bound_session_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.watch_store as watch_module

    monkeypatch.setattr(watch_module, "WATCH_FILE_MAX_LINES", 1)
    monkeypatch.setattr(watch_module, "WATCH_MAX_STORAGE_BYTES", 1)
    store = watch_module.WatchStore(tmp_path / "config.json")
    store.reset(reason="test")
    session_id = store.session_token()

    for index in range(3):
        event_id = store.append(
            {
                "timestamp": f"2026-10-06T12:00:0{index}+00:00",
                "source": "api",
                "endpoint": "/v1/noul",
                "status_code": 200,
                "request_id": f"req-{index}",
                "decisions": 1,
                "request": {"state": f"state-{index}"},
                "response": {"decision": True},
            },
            expected_session_id=session_id,
        )
        assert event_id is not None

    snapshot = store.list_events(limit=10)
    assert snapshot["session"]["requests"] == 3
    assert snapshot["session"]["storage"]["retained_events"] == 1
    assert snapshot["session"]["storage"]["dropped_events"] == 2
    assert snapshot["pagination"]["total"] == 1
    assert [row["request_id"] for row in snapshot["events"]] == ["req-2"]
    assert len(list(store.root.glob("events-*.jsonl"))) == 1


def test_model_management_mutations_use_one_workspace_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import deqio.model_manager as model_manager
    from deqio.process_lock import process_lock

    config_path = tmp_path / "config.json"
    lock_path = tmp_path / ".deqio" / "model-management.lock"
    called = False

    def fake_install(_args):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(model_manager, "cmd_install", fake_install)

    with process_lock(lock_path, timeout=1.0):
        result = model_manager.main(
            ["--config", str(config_path), "install", "demo", "--backend", "mlx"]
        )

    assert result == 2
    assert called is False
    assert "model-management operation is already running" in capsys.readouterr().err


def test_model_management_mutation_refuses_active_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import deqio.model_manager as model_manager

    config_path = tmp_path / "config.json"
    called = False

    def fake_install(_args):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(model_manager, "cmd_install", fake_install)
    monkeypatch.setattr(
        model_manager,
        "discover_server",
        lambda _path: {"pid": 4321, "base_url": "http://127.0.0.1:8787"},
    )

    result = model_manager.main(
        ["--config", str(config_path), "install", "demo", "--backend", "mlx"]
    )

    assert result == 2
    assert called is False
    assert "Stop the active Deqio server" in capsys.readouterr().err


def test_benchmark_workspace_lock_conflicts_with_model_management_lock(tmp_path: Path) -> None:
    from deqio.benchmark import _benchmark_workspace_lock
    from deqio.process_lock import process_lock

    config_path = tmp_path / "config.json"
    model_lock = tmp_path / ".deqio" / "model-management.lock"

    with process_lock(model_lock, timeout=1.0):
        with pytest.raises(RuntimeError, match="model-management operation"):
            with _benchmark_workspace_lock(config_path):
                pass
