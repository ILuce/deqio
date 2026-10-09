from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .process_lock import process_lock

WATCH_SCHEMA_VERSION = 1
WATCH_FILE_MAX_LINES = 10_000
WATCH_FILE_MAX_BYTES = 16 * 1024 * 1024
WATCH_MAX_STORAGE_BYTES = 256 * 1024 * 1024
WATCH_RECENT_LATENCY_SAMPLES = 2_048
WATCH_DEFAULT_LIST_LIMIT = 500
WATCH_MAX_LIST_LIMIT = 5_000
WATCH_AUTO_CLEAR_OPTIONS = (0, 15, 30, 60, 120, 240)


class WatchStore:
    """Session-temporary, disk-backed request/response history.

    The store intentionally lives under the workspace's ``.deqio`` directory so
    the server and independent CLI processes (notably ``deqio benchmark``) can
    append to the same session without keeping an unbounded in-memory history.
    Event files are JSONL and rotate after ``WATCH_FILE_MAX_LINES`` records.
    """

    def __init__(self, config_path: Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.root = self.config_path.parent / ".deqio" / "watch"
        self.session_path = self.root / "session.json"
        self.preferences_path = self.root / "preferences.json"
        self.lock_dir = self.root / ".lock"

    def _locked(self, *, timeout: float = 10.0):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.root.chmod(0o700)
        except OSError:
            pass
        return process_lock(self.lock_dir, timeout=timeout)

    @staticmethod
    def _iso_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            temp.chmod(0o600)
        except OSError:
            pass
        temp.replace(path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _preferences_locked(self) -> dict[str, Any]:
        value = self._read_json(self.preferences_path) or {}
        raw = value.get("auto_clear_minutes", 0)
        try:
            minutes = int(raw)
        except (TypeError, ValueError):
            minutes = 0
        if minutes not in WATCH_AUTO_CLEAR_OPTIONS:
            minutes = 0
        return {
            "schema_version": WATCH_SCHEMA_VERSION,
            "auto_clear_minutes": minutes,
        }

    def preferences(self) -> dict[str, Any]:
        with self._locked():
            value = self._preferences_locked()
            if not self.preferences_path.is_file():
                self._write_json_atomic(self.preferences_path, value)
            return value

    def set_auto_clear_minutes(self, minutes: int) -> dict[str, Any]:
        minutes = int(minutes)
        if minutes not in WATCH_AUTO_CLEAR_OPTIONS:
            allowed = ", ".join(str(value) for value in WATCH_AUTO_CLEAR_OPTIONS)
            raise ValueError(f"auto_clear_minutes must be one of: {allowed}")
        with self._locked():
            payload = {
                "schema_version": WATCH_SCHEMA_VERSION,
                "auto_clear_minutes": minutes,
            }
            self._write_json_atomic(self.preferences_path, payload)
            session = self._ensure_session_locked()
            session["auto_clear_minutes"] = minutes
            session["next_auto_clear_at"] = self._next_auto_clear_at(session, minutes)
            self._write_json_atomic(self.session_path, session)
            return payload

    def _new_session(self, *, reason: str) -> dict[str, Any]:
        now = self._iso_now()
        preferences = self._preferences_locked()
        session = {
            "schema_version": WATCH_SCHEMA_VERSION,
            "id": uuid4().hex,
            "started_at": now,
            "last_event_at": None,
            "reset_reason": reason,
            "requests": 0,
            "decisions": 0,
            "errors": 0,
            "events_dropped": 0,
            "latency_samples_ms": [],
            "files": [],
            "auto_clear_minutes": int(preferences["auto_clear_minutes"]),
        }
        session["next_auto_clear_at"] = self._next_auto_clear_at(
            session, int(preferences["auto_clear_minutes"])
        )
        return session

    @staticmethod
    def _next_auto_clear_at(session: dict[str, Any], minutes: int) -> str | None:
        if minutes <= 0:
            return None
        try:
            started = datetime.fromisoformat(str(session["started_at"]))
        except (KeyError, TypeError, ValueError):
            return None
        return datetime.fromtimestamp(
            started.timestamp() + minutes * 60,
            timezone.utc,
        ).isoformat()

    def _event_paths(self) -> list[Path]:
        return sorted(self.root.glob("events-*.jsonl"))

    def _delete_event_files_locked(self) -> None:
        for path in self._event_paths():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        for path in self.root.glob(".*.tmp"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _ensure_session_locked(self) -> dict[str, Any]:
        session = self._read_json(self.session_path)
        if not session or int(session.get("schema_version", 0)) != WATCH_SCHEMA_VERSION:
            self._delete_event_files_locked()
            session = self._new_session(reason="implicit-start")
            self._write_json_atomic(self.session_path, session)
        return session

    def _auto_clear_due_locked(self, session: dict[str, Any]) -> bool:
        preferences = self._preferences_locked()
        minutes = int(preferences["auto_clear_minutes"])
        if minutes <= 0:
            return False
        try:
            started = datetime.fromisoformat(str(session["started_at"])).timestamp()
        except (KeyError, TypeError, ValueError):
            return True
        return time.time() >= started + minutes * 60

    def _maybe_auto_clear_locked(self, session: dict[str, Any]) -> dict[str, Any]:
        if not self._auto_clear_due_locked(session):
            preferences = self._preferences_locked()
            minutes = int(preferences["auto_clear_minutes"])
            session["auto_clear_minutes"] = minutes
            session["next_auto_clear_at"] = self._next_auto_clear_at(session, minutes)
            return session
        self._delete_event_files_locked()
        session = self._new_session(reason="auto-clear")
        self._write_json_atomic(self.session_path, session)
        return session

    def reset(self, *, reason: str) -> dict[str, Any]:
        """Delete every event file and create a fresh session.

        Preferences intentionally survive a reset so a user-selected automatic
        cleanup interval remains active after server restart or manual clear.
        """

        with self._locked():
            self._delete_event_files_locked()
            session = self._new_session(reason=reason)
            self._write_json_atomic(self.session_path, session)
            return dict(session)

    def session_token(self) -> str:
        with self._locked():
            session = self._maybe_auto_clear_locked(self._ensure_session_locked())
            self._write_json_atomic(self.session_path, session)
            return str(session["id"])

    def auto_clear_if_due(self) -> bool:
        """Run scheduled cleanup without loading event payloads into memory.

        Returns ``True`` when the current Watch session was replaced because
        the configured interval elapsed. The server calls this periodically so
        auto-clear remains wall-clock driven even while no requests arrive.
        """

        with self._locked():
            session = self._ensure_session_locked()
            if not self._auto_clear_due_locked(session):
                return False
            self._delete_event_files_locked()
            session = self._new_session(reason="auto-clear")
            self._write_json_atomic(self.session_path, session)
            return True

    @staticmethod
    def _percentile(values: list[float], p: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = round((len(ordered) - 1) * p)
        return ordered[index]

    def _public_session(self, session: dict[str, Any]) -> dict[str, Any]:
        latencies = [
            float(value)
            for value in session.get("latency_samples_ms", [])
            if isinstance(value, (int, float))
        ]
        files = [row for row in session.get("files", []) if isinstance(row, dict)]
        return {
            "id": session.get("id"),
            "started_at": session.get("started_at"),
            "last_event_at": session.get("last_event_at"),
            "reset_reason": session.get("reset_reason"),
            "requests": int(session.get("requests", 0)),
            "decisions": int(session.get("decisions", 0)),
            "errors": int(session.get("errors", 0)),
            "latency_ms": {
                "p50": self._percentile(latencies, 0.50),
                "p95": self._percentile(latencies, 0.95),
                "samples": len(latencies),
            },
            "storage": {
                "kind": "temporary-jsonl",
                "directory": str(self.root),
                "files": len(files),
                "retained_events": sum(max(0, int(row.get("lines", 0))) for row in files),
                "dropped_events": max(0, int(session.get("events_dropped", 0))),
                "max_lines_per_file": WATCH_FILE_MAX_LINES,
                "max_bytes_per_file": WATCH_FILE_MAX_BYTES,
                "max_storage_bytes": WATCH_MAX_STORAGE_BYTES,
            },
            "auto_clear_minutes": int(session.get("auto_clear_minutes", 0)),
            "next_auto_clear_at": session.get("next_auto_clear_at"),
        }

    def session(self) -> dict[str, Any]:
        with self._locked():
            session = self._maybe_auto_clear_locked(self._ensure_session_locked())
            self._write_json_atomic(self.session_path, session)
            return self._public_session(session)

    def _file_size_locked(self, file_row: dict[str, Any]) -> int:
        try:
            stored = int(file_row.get("bytes", -1))
        except (TypeError, ValueError):
            stored = -1
        if stored >= 0:
            return stored
        try:
            stored = (self.root / str(file_row.get("name", ""))).stat().st_size
        except (FileNotFoundError, OSError):
            stored = 0
        file_row["bytes"] = int(stored)
        return int(stored)

    def _prune_storage_locked(self, session: dict[str, Any], files: list[dict[str, Any]]) -> None:
        total_bytes = sum(self._file_size_locked(row) for row in files)
        dropped = max(0, int(session.get("events_dropped", 0)))
        # Keep the active/newest file even if one unusually large event alone
        # exceeds the retention budget; pruning must never delete the event that
        # append() is about to return to its caller. The next rotations remain
        # bounded by removing older files first.
        while len(files) > 1 and total_bytes > WATCH_MAX_STORAGE_BYTES:
            oldest = files.pop(0)
            size = self._file_size_locked(oldest)
            try:
                (self.root / str(oldest.get("name", ""))).unlink()
            except FileNotFoundError:
                pass
            total_bytes = max(0, total_bytes - size)
            dropped += max(0, int(oldest.get("lines", 0)))
        session["events_dropped"] = dropped
        session["files"] = files

    def append(self, event: dict[str, Any], *, expected_session_id: str | None = None) -> str | None:
        serialized_event = json.loads(
            json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        )
        with self._locked():
            session = self._maybe_auto_clear_locked(self._ensure_session_locked())
            if expected_session_id is not None and expected_session_id != str(session.get("id")):
                return None

            files = [row for row in session.get("files", []) if isinstance(row, dict)]
            current = files[-1] if files else None
            current_lines = int(current.get("lines", 0)) if current else 0
            current_bytes = self._file_size_locked(current) if current else 0

            # event_id has fixed-width file/line components, so serializing once
            # with the current candidate gives an exact rotation decision.
            candidate_index = int(current.get("index", 0)) if current else 1
            candidate_line = current_lines + 1 if current else 1
            serialized_event["event_id"] = (
                f"{candidate_index:06d}-{candidate_line:05d}-{uuid4().hex[:12]}"
            )
            encoded = (
                json.dumps(
                    serialized_event,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")

            can_append_current = bool(
                current
                and current_lines < WATCH_FILE_MAX_LINES
                and current_bytes + len(encoded) <= WATCH_FILE_MAX_BYTES
            )
            if can_append_current:
                file_row = current
                line_number = current_lines + 1
            else:
                index = int(current.get("index", 0)) + 1 if current else 1
                file_row = {
                    "index": index,
                    "name": f"events-{index:06d}.jsonl",
                    "lines": 0,
                    "bytes": 0,
                }
                files.append(file_row)
                line_number = 1
                serialized_event["event_id"] = f"{index:06d}-{line_number:05d}-{uuid4().hex[:12]}"
                encoded = (
                    json.dumps(
                        serialized_event,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")

            target = self.root / str(file_row["name"])
            self._drop_unterminated_tail(target)
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(descriptor, "ab") as stream:
                stream.write(encoded)
                stream.flush()

            file_row["lines"] = line_number
            file_row["bytes"] = self._file_size_locked(file_row) + len(encoded)
            session["requests"] = int(session.get("requests", 0)) + 1
            session["decisions"] = int(session.get("decisions", 0)) + int(
                serialized_event.get("decisions", 0)
            )
            if int(serialized_event.get("status_code", 500)) >= 400:
                session["errors"] = int(session.get("errors", 0)) + 1
            latency = serialized_event.get("latency_ms")
            samples = [
                float(value)
                for value in session.get("latency_samples_ms", [])
                if isinstance(value, (int, float))
            ]
            if isinstance(latency, (int, float)):
                samples.append(float(latency))
            session["latency_samples_ms"] = samples[-WATCH_RECENT_LATENCY_SAMPLES:]
            session["last_event_at"] = serialized_event.get("timestamp") or self._iso_now()
            minutes = int(self._preferences_locked()["auto_clear_minutes"])
            session["auto_clear_minutes"] = minutes
            session["next_auto_clear_at"] = self._next_auto_clear_at(session, minutes)
            self._prune_storage_locked(session, files)
            self._write_json_atomic(self.session_path, session)
            return str(serialized_event["event_id"])

    @staticmethod
    def _drop_unterminated_tail(path: Path) -> None:
        """Remove a torn final record before appending to *path* (lock held).

        A crash or ENOSPC in the middle of a write leaves the last line without
        its newline. That event was never acknowledged (the session metadata is
        written only after a complete append), so it is not part of the session.
        Appending after it would glue the next event onto it and make that event
        unreadable (B14). Truncating back to the last newline keeps the physical
        lines equal to the session's line count, which listing relies on.
        """
        try:
            stream = path.open("r+b")
        except FileNotFoundError:
            return
        with stream:
            size = stream.seek(0, os.SEEK_END)
            if size == 0:
                return
            stream.seek(size - 1)
            if stream.read(1) == b"\n":
                return
            keep = 0
            position = size
            while position > 0:
                start = max(0, position - 65536)
                stream.seek(start)
                chunk = stream.read(position - start)
                newline = chunk.rfind(b"\n")
                if newline >= 0:
                    keep = start + newline + 1
                    break
                position = start
            stream.truncate(keep)

    @staticmethod
    def _row(event: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "event_id",
            "timestamp",
            "source",
            "endpoint",
            "method",
            "status_code",
            "request_id",
            "engine",
            "model_id",
            "backend",
            "runtime_instance_id",
            "mode",
            "decisions",
            "decision",
            "top_probability",
            "latency_ms",
            "input_tokens",
        )
        return {key: event.get(key) for key in keys}

    @staticmethod
    def _read_range(path: Path, *, start: int, stop: int) -> list[dict[str, Any]]:
        """Read only a zero-based line range from a rotated event file.

        A Watch event may contain a large request/response payload. Even though
        every JSONL file is capped at 10,000 records, reading a whole file just
        to render one UI page could still allocate a large amount of RAM. Keep
        listing bounded by streaming the file and decoding only the requested
        range.
        """

        rows: list[dict[str, Any]] = []
        if stop <= start:
            return rows
        try:
            stream = path.open("r", encoding="utf-8")
        except FileNotFoundError:
            return rows
        with stream:
            for line_index, line in enumerate(stream):
                if line_index < start:
                    continue
                if line_index >= stop:
                    break
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
        return rows

    def list_events(self, *, limit: int = WATCH_DEFAULT_LIST_LIMIT, offset: int = 0) -> dict[str, Any]:
        limit = max(1, min(int(limit), WATCH_MAX_LIST_LIMIT))
        offset = max(0, int(offset))
        # Keep the process lock while event files are open/read.  On POSIX an
        # unlink racing a reader is harmless, but on Windows it can fail and
        # leave a partially reset Watch session.  A list call decodes at most
        # WATCH_MAX_LIST_LIMIT records, so this lock hold remains bounded.
        with self._locked():
            session = self._maybe_auto_clear_locked(self._ensure_session_locked())
            self._write_json_atomic(self.session_path, session)
            files = [row.copy() for row in session.get("files", []) if isinstance(row, dict)]
            public_session = self._public_session(session)

            total = sum(max(0, int(row.get("lines", 0))) for row in files)
            remaining_offset = offset
            events: list[dict[str, Any]] = []
            for file_row in reversed(files):
                count = int(file_row.get("lines", 0))
                if remaining_offset >= count:
                    remaining_offset -= count
                    continue
                needed = limit - len(events)
                newest_exclusive = max(0, count - remaining_offset)
                oldest_inclusive = max(0, newest_exclusive - needed)
                rows = self._read_range(
                    self.root / str(file_row.get("name", "")),
                    start=oldest_inclusive,
                    stop=newest_exclusive,
                )
                rows.reverse()
                events.extend(rows)
                remaining_offset = 0
                if len(events) >= limit:
                    break

            return {
                "session": public_session,
                "events": [self._row(event) for event in events],
                "pagination": {
                    "offset": offset,
                    "limit": limit,
                    "total": total,
                    "returned": len(events),
                    "has_more": offset + len(events) < total,
                },
            }

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        try:
            index_text = event_id.split("-", 1)[0]
            index = int(index_text)
        except (ValueError, IndexError):
            return None
        path = self.root / f"events-{index:06d}.jsonl"
        with self._locked():
            try:
                stream = path.open("r", encoding="utf-8")
            except FileNotFoundError:
                return None
            with stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict) and str(event.get("event_id")) == event_id:
                        return event
        return None
