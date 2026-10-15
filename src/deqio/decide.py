"""``deqio decide`` — one decision from the command line through a running ``deqio serve`` (0.6.2, D13).

    deqio decide noul   --state TEXT --question TEXT
    deqio decide choice --state TEXT --question TEXT --option ID[=DESCRIPTION] ...
    deqio decide shared --state TEXT --decisions-json JSON|@FILE
    deqio decide score|multi|act --state TEXT --instructions TEXT --criterion KEY[=DESCRIPTION] ... | --question-json JSON|@FILE
    deqio decide soam|systemone --state TEXT --questions-json JSON|@FILE

The command is a thin wrapper around the ``deqio_client`` SDK: it builds exactly the request the route documents
(``deqio openapi``), posts it to ``--url`` (default ``http://127.0.0.1:8787``) and prints the server's answer —
a short summary with the provenance line, or the complete JSON response with ``--json``.  Exit codes: 0 on a
decision, 1 when the server rejected the request or could not be reached (the server's ``detail`` is printed to
stderr), 2 on a usage error.  Nothing is retried or defaulted silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from deqio_client import DeqioClient, DeqioConnectionError, DeqioError

ROUTES = ("noul", "choice", "decision", "shared", "score", "multi", "act", "soam", "systemone")
DEFAULT_URL = "http://127.0.0.1:8787"


class UsageError(ValueError):
    """A command line that cannot be turned into one documented request."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deqio decide",
        description="Ask a running Deqio server for one decision and print its answer.",
        epilog=(
            "JSON arguments accept inline JSON or @FILE. --state-json gives a structured state (object or array) instead "
            "of --state text. --criterion KEY gives an ordered list of levels (score); --criterion KEY=DESCRIPTION gives "
            "a {key: description} mapping (multi, act); the two forms cannot be mixed."
        ),
    )
    parser.add_argument("route", choices=ROUTES, help="decision route (decision = the /v1/choice alias)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"server base URL (default: {DEFAULT_URL})")
    parser.add_argument("--timeout", type=float, default=60.0, help="request timeout in seconds (default: 60)")
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--state", help="state text")
    state.add_argument("--state-file", help="read the state text from a file")
    state.add_argument("--state-json", help="structured state as inline JSON or @FILE")
    parser.add_argument("--question", help="question text (noul, choice, decision)")
    parser.add_argument("--option", action="append", default=[], metavar="ID[=DESCRIPTION]", help="choice option (repeat)")
    parser.add_argument("--id", help="explicit decision id (noul, choice, decision)")
    parser.add_argument("--mode", choices=("direct", "serial"), help="scoring mode (noul, choice, decision)")
    parser.add_argument("--contract", action="store_true", help="negotiate Deqio-Contract: input-completeness-v1 (portable routes)")
    parser.add_argument("--require-complete", action="store_true", help="input_policy.require_complete=true (portable routes)")
    parser.add_argument("--decisions-json", help="shared decisions: inline JSON array or @FILE")
    parser.add_argument("--instructions", help="native question instructions (score, multi, act)")
    parser.add_argument("--criterion", action="append", default=[], metavar="KEY[=DESCRIPTION]", help="native criterion (repeat)")
    parser.add_argument("--question-json", help="complete native question object as inline JSON or @FILE (score, multi, act)")
    parser.add_argument("--name", default="decision", help="native question name (score, multi, act; default: decision)")
    parser.add_argument("--questions-json", help="native questions object as inline JSON or @FILE (soam, systemone)")
    parser.add_argument("--facts", choices=("off", "auto"), help="facts modifier (native routes)")
    parser.add_argument("--json", action="store_true", help="print the complete JSON response instead of the summary")
    return parser


def _load_json_argument(value: str, *, label: str) -> Any:
    text = value
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise UsageError(f"{label}: cannot read {path}: {error}") from error
    try:
        return json.loads(text)
    except ValueError as error:
        raise UsageError(f"{label}: invalid JSON ({error})") from error


def _state(args: argparse.Namespace) -> Any:
    if args.state_json is not None:
        value = _load_json_argument(args.state_json, label="--state-json")
        if not isinstance(value, (dict, list)):
            raise UsageError("--state-json must be a JSON object or array; use --state for text")
        return value
    if args.state_file is not None:
        path = Path(args.state_file).expanduser()
        try:
            return path.read_text(encoding="utf-8")
        except OSError as error:
            raise UsageError(f"--state-file: cannot read {path}: {error}") from error
    if args.state is None:
        raise UsageError("one of --state, --state-file or --state-json is required")
    return args.state


def _pairs(values: list[str], *, flag: str) -> tuple[list[str], dict[str, str] | None]:
    """Parse ``KEY`` / ``KEY=DESCRIPTION`` items: all keys, and the mapping when every item has a description."""
    keys: list[str] = []
    mapping: dict[str, str] = {}
    described = 0
    for item in values:
        key, separator, description = item.partition("=")
        key = key.strip()
        if not key:
            raise UsageError(f"{flag}: empty key in {item!r}")
        if key in keys:
            raise UsageError(f"{flag}: duplicate key {key!r}")
        keys.append(key)
        if separator:
            described += 1
            mapping[key] = description
    if described not in (0, len(values)):
        raise UsageError(f"{flag}: use either KEY for every item or KEY=DESCRIPTION for every item, not both")
    return keys, (mapping if described else None)


def _input_policy(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.require_complete or args.contract:
        return {"require_complete": bool(args.require_complete)}
    return None


def _native_question(args: argparse.Namespace) -> dict[str, Any]:
    if args.question_json is not None:
        if args.instructions is not None or args.criterion:
            raise UsageError("--question-json cannot be combined with --instructions/--criterion")
        question = _load_json_argument(args.question_json, label="--question-json")
        if not isinstance(question, dict) or not question:
            raise UsageError("--question-json must be a non-empty JSON object")
        return question
    if not args.criterion:
        raise UsageError(f"{args.route}: give --criterion KEY[=DESCRIPTION] (repeat) or --question-json")
    keys, mapping = _pairs(args.criterion, flag="--criterion")
    question: dict[str, Any] = {"criteria": mapping if mapping is not None else keys}
    if args.instructions is not None:
        question["instructions"] = args.instructions
    return question


def build_request(args: argparse.Namespace) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Return ``(client method name, keyword arguments, request body as the client will send it)``."""
    state = _state(args)
    route = args.route
    if route in ("noul", "choice", "decision"):
        if not args.question or not args.question.strip():
            raise UsageError(f"{route}: --question is required")
        kwargs: dict[str, Any] = {"id": args.id, "mode": args.mode, "input_policy": _input_policy(args), "contract": args.contract}
        if route == "noul":
            if args.option:
                raise UsageError("noul takes no --option")
            return "noul", {"state": state, "question": args.question, **kwargs}, {}
        if len(args.option) < 2:
            raise UsageError(f"{route}: give at least two --option ID[=DESCRIPTION]")
        keys, mapping = _pairs(args.option, flag="--option")
        options = [{"id": key, "description": (mapping or {}).get(key, key)} for key in keys]
        return route, {"state": state, "question": args.question, "options": options, **kwargs}, {}
    if route == "shared":
        if args.decisions_json is None:
            raise UsageError("shared: --decisions-json JSON|@FILE is required")
        decisions = _load_json_argument(args.decisions_json, label="--decisions-json")
        if not isinstance(decisions, list) or not decisions:
            raise UsageError("--decisions-json must be a non-empty JSON array of {question, options, id?}")
        return "shared", {"state": state, "decisions": decisions, "input_policy": _input_policy(args), "contract": args.contract}, {}
    if route in ("score", "multi", "act"):
        return route, {"state": state, "question": _native_question(args), "name": args.name, "facts": args.facts}, {}
    if args.questions_json is None:
        raise UsageError(f"{route}: --questions-json JSON|@FILE is required")
    questions = _load_json_argument(args.questions_json, label="--questions-json")
    if not isinstance(questions, dict) or not questions:
        raise UsageError("--questions-json must be a non-empty JSON object {name: question}")
    return route, {"state": state, "questions": questions, "facts": args.facts}, {}


def summarize(route: str, response: dict[str, Any]) -> str:
    lines: list[str] = []
    if route in ("noul", "choice", "decision"):
        lines.append(f"decision: {response.get('decision')}  (top probability {response.get('top_probability')})")
        lines.append("probabilities: " + json.dumps(response.get("probabilities"), ensure_ascii=False))
        lines.append(_provenance_line(response.get("provenance") or {}))
    elif route == "shared":
        for result in response.get("results", []):
            lines.append(f"{result.get('id')}: {result.get('decision')}  (top probability {result.get('top_probability')})")
        lines.append(_provenance_line(response.get("provenance") or {}))
    else:
        for name, answer in (response.get("answers") or {}).items():
            lines.append(f"{name}: " + json.dumps(answer, ensure_ascii=False))
        lines.append(_provenance_line(response.get("deqio") or {}))
    return "\n".join(lines)


def _provenance_line(provenance: dict[str, Any]) -> str:
    runtime = provenance.get("runtime") or {}
    provider = provenance.get("provider") or {}
    cost = provenance.get("cost_estimate") or {}
    parts = [
        f"engine={runtime.get('engine')}",
        f"model={runtime.get('model_id')}",
        f"backend={runtime.get('backend')}",
        f"provider={provider.get('name')}",
    ]
    if provenance.get("remote_request_id"):
        parts.append(f"remote_request_id={provenance['remote_request_id']}")
    if cost:
        parts.append(f"cost_estimate={cost.get('amount')} {cost.get('currency')}")
    return "provenance: " + " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        method, kwargs, _body = build_request(args)
        client = DeqioClient(args.url, timeout=args.timeout)
        response = getattr(client, method)(**kwargs)
    except UsageError as error:
        parser.print_usage(sys.stderr)
        print(f"deqio decide: error: {error}", file=sys.stderr)
        return 2
    except ValueError as error:  # the SDK refused the request shape before sending it
        print(f"deqio decide: error: {error}", file=sys.stderr)
        return 2
    except DeqioError as error:
        print(f"error: HTTP {error.status} from {error.path}: {json.dumps(error.detail, ensure_ascii=False)}", file=sys.stderr)
        return 1
    except DeqioConnectionError as error:
        print(f"error: {error}. Is deqio serve running at {args.url}?", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(response, indent=2, ensure_ascii=False))
    else:
        print(summarize(args.route, response))
    return 0
