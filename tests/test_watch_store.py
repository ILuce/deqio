import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from deqio.watch_store import WatchStore


def _event(index: int, *, latency_ms: float = 1.0) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "api",
        "endpoint": "/v1/choice",
        "method": "POST",
        "status_code": 200,
        "request_id": f"req-{index}",
        "engine": "demo",
        "model_id": "demo-model",
        "backend": "mlx",
        "runtime_instance_id": "runtime-1",
        "mode": "serial",
        "decisions": 1,
        "decision": "a",
        "top_probability": 0.9,
        "latency_ms": latency_ms,
        "input_tokens": 10,
        "request": {"id": f"req-{index}", "state": "state"},
        "response": {"id": f"req-{index}", "decision": "a"},
    }


def test_watch_store_rotates_jsonl_files_at_line_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.watch_store as watch_module

    monkeypatch.setattr(watch_module, "WATCH_FILE_MAX_LINES", 2)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    store = WatchStore(config_path)
    session_id = store.reset(reason="test")["id"]

    for index in range(3):
        assert store.append(_event(index), expected_session_id=session_id)

    event_files = sorted(store.root.glob("events-*.jsonl"))
    assert [len(path.read_text(encoding="utf-8").splitlines()) for path in event_files] == [2, 1]
    snapshot = store.list_events(limit=10)
    assert snapshot["pagination"]["total"] == 3
    assert [row["request_id"] for row in snapshot["events"]] == ["req-2", "req-1", "req-0"]
    assert snapshot["session"]["storage"]["max_lines_per_file"] == 2

    middle = store.list_events(limit=2, offset=1)
    assert [row["request_id"] for row in middle["events"]] == ["req-1", "req-0"]
    assert middle["pagination"]["has_more"] is False


def test_watch_store_auto_clear_deletes_history_but_keeps_preference(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    store = WatchStore(config_path)
    session_id = store.reset(reason="test")["id"]
    store.set_auto_clear_minutes(30)
    store.append(_event(1), expected_session_id=session_id)
    assert list(store.root.glob("events-*.jsonl"))

    session = json.loads(store.session_path.read_text(encoding="utf-8"))
    session["started_at"] = (datetime.now(timezone.utc) - timedelta(minutes=31)).isoformat()
    store.session_path.write_text(json.dumps(session), encoding="utf-8")

    assert store.auto_clear_if_due() is True
    assert list(store.root.glob("events-*.jsonl")) == []
    after = store.session()
    assert after["requests"] == 0
    assert after["reset_reason"] == "auto-clear"
    assert after["auto_clear_minutes"] == 30
    assert store.preferences()["auto_clear_minutes"] == 30


def test_watch_store_manual_reset_deletes_all_rotated_history_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deqio.watch_store as watch_module

    monkeypatch.setattr(watch_module, "WATCH_FILE_MAX_LINES", 1)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    store = WatchStore(config_path)
    session_id = store.reset(reason="test")["id"]
    for index in range(4):
        store.append(_event(index), expected_session_id=session_id)
    assert len(list(store.root.glob("events-*.jsonl"))) == 4

    replacement = store.reset(reason="manual-clear")
    assert replacement["requests"] == 0
    assert list(store.root.glob("events-*.jsonl")) == []


def test_append_after_a_torn_write_keeps_the_next_event_readable(tmp_path: Path) -> None:
    """B14: a crash or ENOSPC mid-write leaves an unterminated, unacknowledged line.

    The next append must not be glued onto it (which made that event
    unreadable and kept `total` and `returned` apart forever).
    """
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    store = WatchStore(config_path)
    first = [store.append(_event(index)) for index in range(3)]
    session = json.loads(store.session_path.read_text(encoding="utf-8"))
    events_file = store.root / session["files"][-1]["name"]
    with events_file.open("ab") as stream:
        stream.write(b'{"event_id":"000001-00004-torn","endpoint":"/v1/cho')

    new_id = store.append(_event(99))

    assert new_id is not None
    event = store.get_event(new_id)
    assert event is not None and event["request_id"] == "req-99"
    listing = store.list_events(limit=50)
    assert [row["event_id"] for row in listing["events"]] == [new_id, *reversed(first)]
    assert listing["pagination"]["total"] == listing["pagination"]["returned"] == 4
    for line in events_file.read_text(encoding="utf-8").splitlines():
        json.loads(line)
