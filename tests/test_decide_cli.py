"""D13 (0.6.2): ``deqio decide`` — one decision from the command line through the SDK.

Every route builds exactly the documented request (parsed here with the server's pydantic models), the answer
is printed as a summary or as the full JSON (``--json``), and the exit code tells usage errors (2) from the
server's rejections or an unreachable server (1) apart.  The requests travel over a real localhost HTTP server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fake_deqio_http import FakeDeqioServer  # noqa: E402

CHOICE_RESPONSE = {
    "id": "choice-1", "decision": "bug", "probabilities": {"bug": 0.9, "feature": 0.1}, "top_probability": 0.9,
    "option_logits": {}, "input_tokens": 21, "timing": {"total_ms": 3.0}, "prompt_sha256": "ab" * 32,
    "probability_status": "engine probabilities",
    "provenance": {"runtime": {"engine": "stub", "model_id": "stub-model", "backend": "gguf"},
                   "provider": {"name": "local", "kind": "local"}, "remote_request_id": None, "cost_estimate": None},
}
NATIVE_RESPONSE = {
    "model": "stub/model", "answers": {"decision": {"type": "score", "score": "high"}}, "usage": {"input_tokens": 33},
    "deqio": {"runtime": {"engine": "stub", "model_id": "stub-model", "backend": "gguf"},
              "provider": {"name": "openai-decisions", "kind": "remote"}, "remote_request_id": "req_9",
              "cost_estimate": {"amount": 1e-05, "currency": "USD"}},
}


def _run(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    from deqio.cli import main

    code = main(["decide", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_decide_help_is_listed_and_routes_are_documented(capsys: pytest.CaptureFixture[str]) -> None:
    from deqio.cli import main
    from deqio.decide import ROUTES

    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "deqio decide" in out and "deqio openapi" in out
    assert ROUTES == ("noul", "choice", "decision", "shared", "score", "multi", "act", "soam", "systemone")
    with pytest.raises(SystemExit) as exit_info:
        main(["decide", "--help"])
    assert exit_info.value.code == 0
    assert "--option" in capsys.readouterr().out


def test_decide_builds_the_documented_request_for_every_route(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from deqio.server import DecisionRequest, NoulRequest, SharedRequest, SystemOneRequest, TypedSystemOneRequest

    question_file = tmp_path / "question.json"
    question_file.write_text(json.dumps({"instructions": "Choose the next action.", "criteria": {"t": "T"}, "costs": {"go": {"t": 0}}}), encoding="utf-8")
    state_file = tmp_path / "state.txt"
    state_file.write_text("state from a file\n", encoding="utf-8")
    with FakeDeqioServer() as server:
        base = ["--url", server.base_url]
        cases = [
            (["noul", "--state", "s", "--question", "Is it?"], "/v1/noul", NoulRequest, {"state": "s", "question": "Is it?"}),
            (["choice", "--state", "s", "--question", "Which?", "--option", "bug=Broken", "--option", "feature=New", "--id", "c-1", "--mode", "direct"],
             "/v1/choice", DecisionRequest,
             {"state": "s", "question": "Which?", "options": [{"id": "bug", "description": "Broken"}, {"id": "feature", "description": "New"}],
              "id": "c-1", "mode": "direct"}),
            (["decision", "--state-file", str(state_file), "--question", "Which?", "--option", "a", "--option", "b"], "/v1/decision", DecisionRequest,
             {"state": "state from a file\n", "question": "Which?", "options": [{"id": "a", "description": "a"}, {"id": "b", "description": "b"}]}),
            (["shared", "--state-json", '{"text": "s"}', "--decisions-json", '[{"id": "k", "question": "Kind?", "options": {"a": "A", "b": "B"}}]'],
             "/v1/shared", SharedRequest,
             {"state": {"text": "s"}, "decisions": [{"question": "Kind?", "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}], "id": "k"}]}),
            (["score", "--state", "s", "--instructions", "Rate it.", "--criterion", "low", "--criterion", "high"], "/v1/score", TypedSystemOneRequest,
             {"state": "s", "question": {"criteria": ["low", "high"], "instructions": "Rate it."}, "name": "decision"}),
            (["multi", "--state", "s", "--criterion", "damage=Physical damage", "--criterion", "refund=Refund asked", "--name", "labels", "--facts", "auto"],
             "/v1/multi", TypedSystemOneRequest,
             {"state": "s", "question": {"criteria": {"damage": "Physical damage", "refund": "Refund asked"}}, "name": "labels", "facts": "auto"}),
            (["act", "--state", "s", "--question-json", f"@{question_file}"], "/v1/act", TypedSystemOneRequest,
             {"state": "s", "question": json.loads(question_file.read_text()), "name": "decision"}),
            (["soam", "--state", "s", "--questions-json", '{"ok": {"type": "noul"}}'], "/v1/soam", SystemOneRequest,
             {"state": "s", "questions": {"ok": {"type": "noul"}}}),
            (["systemone", "--state", "s", "--questions-json", '{"ok": {"type": "noul"}}', "--facts", "off"], "/v1/systemone", SystemOneRequest,
             {"state": "s", "questions": {"ok": {"type": "noul"}}, "facts": "off"}),
        ]
        for args, path, model, expected in cases:
            server.answer(path, CHOICE_RESPONSE if path in ("/v1/noul", "/v1/choice", "/v1/decision") else
                          {"results": [CHOICE_RESPONSE], "shared_timing": {}, "provenance": CHOICE_RESPONSE["provenance"]} if path == "/v1/shared"
                          else NATIVE_RESPONSE)
            code, out, err = _run([*base, *args], capsys)
            assert code == 0, (args, err)
            sent = server.last
            assert sent["path"] == path and sent["body"] == expected, (args, sent["body"])
            model.model_validate(sent["body"])
            assert "provenance:" in out and "engine=stub" in out, out
            if path in ("/v1/noul", "/v1/choice", "/v1/decision"):
                assert "decision: bug" in out and '"bug": 0.9' in out
            elif path == "/v1/shared":
                assert "choice-1: bug" in out
            else:
                assert '"score": "high"' in out and "provider=openai-decisions" in out
                assert "remote_request_id=req_9" in out and "cost_estimate=1e-05 USD" in out

        code, out, _err = _run([*base, "noul", "--state", "s", "--question", "Is it?", "--contract", "--require-complete", "--json"], capsys)
        assert code == 0 and json.loads(out) == CHOICE_RESPONSE
        assert server.last["headers"]["deqio-contract"] == "input-completeness-v1"
        assert server.last["body"] == {"state": "s", "question": "Is it?", "input_policy": {"require_complete": True}}


def test_decide_exit_codes_distinguish_usage_server_rejection_and_unreachable_server(capsys: pytest.CaptureFixture[str]) -> None:
    with FakeDeqioServer() as server:
        base = ["--url", server.base_url]
        usage_errors = [
            ["noul", "--question", "Is it?"],  # no state
            ["choice", "--state", "s", "--question", "Which?", "--option", "only-one"],
            ["choice", "--state", "s", "--question", "Which?", "--option", "a=A", "--option", "b"],  # mixed forms
            ["score", "--state", "s"],  # no criteria
            ["score", "--state", "s", "--criterion", "low", "--question-json", "{}"],
            ["shared", "--state", "s"],
            ["soam", "--state", "s", "--questions-json", "not json"],
            ["systemone", "--state", "s", "--questions-json", "@/nonexistent/questions.json"],
            ["noul", "--state-json", '"text"', "--question", "Is it?"],
        ]
        for args in usage_errors:
            code, _out, err = _run([*base, *args], capsys)
            assert code == 2 and "error" in err, (args, err)
        assert server.requests == []  # nothing was sent for a usage error

        server.answer("/v1/choice", {"detail": "options contains 1 options; at least 2 are required"}, status=400)
        code, out, err = _run([*base, "choice", "--state", "s", "--question", "Which?", "--option", "a", "--option", "b"], capsys)
        assert code == 1 and out == "" and "HTTP 400" in err and "at least 2" in err
        with pytest.raises(SystemExit) as bad_route:
            _run([*base, "nope", "--state", "s"], capsys)
        assert bad_route.value.code == 2
        closed = server.base_url
    code, _out, err = _run(["--url", closed, "--timeout", "2", "noul", "--state", "s", "--question", "Is it?"], capsys)
    assert code == 1 and "cannot reach" in err and "deqio serve" in err
