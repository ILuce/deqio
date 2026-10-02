from pathlib import Path

from deqio.benchmark import _run_case
from deqio.watch_store import WatchStore


class _FakeChoiceRuntime:
    def identity_snapshot(self):
        return {
            "engine": "demo",
            "model_id": "demo-model",
            "backend": "mlx",
            "runtime_instance_id": "runtime-benchmark-1",
        }

    def score(self, row, mode):
        assert mode == "serial"
        return {
            "id": row["id"],
            "option_ids": ["a", "b"],
            "probabilities": [0.8, 0.2],
            "input_tokens": 17,
            "input_tokens_source": "engine_reported",
            "total_seconds": 0.012,
            "prompt_sha256": "a" * 64,
            "probability_status": "engine probabilities",
            "score_provenance": {
                "kind": "engine_probability",
                "source": "engine.probabilities",
                "synthetic": False,
            },
        }


def test_benchmark_direct_runtime_request_is_written_to_watch(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    store = WatchStore(config_path)
    store.reset(reason="server-start")

    result = _run_case(
        _FakeChoiceRuntime(),
        {"engine": "demo", "model_id": "demo-model", "backend": "mlx"},
        {
            "id": "choice-watch-1",
            "type": "choice",
            "state": "state",
            "question": "Pick one",
            "options": [
                {"id": "a", "description": "A"},
                {"id": "b", "description": "B"},
            ],
            "expected": "a",
        },
        watch=store,
    )

    assert result.passed is True
    snapshot = store.list_events(limit=10)
    assert snapshot["session"]["requests"] == 1
    assert snapshot["events"][0]["source"] == "benchmark"
    assert snapshot["events"][0]["endpoint"] == "/v1/choice"
    assert snapshot["events"][0]["model_id"] == "demo-model"

    detail = store.get_event(snapshot["events"][0]["event_id"])
    assert detail is not None
    assert detail["request"]["id"] == "choice-watch-1"
    assert detail["response"]["decision"] == "a"
    assert detail["response"]["input_tokens"] == 17
    assert detail["response"]["input_tokens_source"] == "engine_reported"
    assert detail["response"]["score_provenance"]["kind"] == "engine_probability"
