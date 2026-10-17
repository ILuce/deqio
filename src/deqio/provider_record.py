"""Record one real exchange with a remote decision provider for the contract tests (D11 / D11b).

    python -m deqio.provider_record --provider openai-decisions --out tests/fixtures/providers/openai-decisions/choice.json
    python -m deqio.provider_record --provider typesafe-jev --out tests/fixtures/providers/typesafe-jev/choice.json

The API key comes from the OS keychain entry written by ``deqio models setup`` (``--key keychain``,
the default) or, explicitly, from the vendor's conventional environment variable (``--key env``:
``OPENAI_API_KEY`` / ``TYPESAFE_API_KEY``).  The key is never written.  The cassette replaces the
fixture of the same name (real recordings since 0.6.2), and ``tests/test_providers.py`` replays it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .credentials import load_api_key
from .providers import RemoteDecisionProvider, cassette_from_exchange, provider_spec

SAMPLE_STATE = "A customer reports that the API returns HTTP 500 after the latest deployment."
SAMPLE_QUESTIONS = {
    "decision": {
        "type": "choice",
        "instructions": "Classify the report.",
        "criteria": {
            "bug": "An existing function no longer works",
            "feature": "A request for new functionality",
            "support": "A question about how to use the product",
        },
    }
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m deqio.provider_record", description=__doc__)
    parser.add_argument("--provider", required=True, help="openai-decisions | typesafe-jev")
    parser.add_argument("--model", default="", help="Remote model (default: the provider's default)")
    parser.add_argument("--endpoint", default="", help="Endpoint URL (default: the provider's documented endpoint)")
    parser.add_argument("--key", choices=("keychain", "env"), default="keychain", help="Where the API key is read from")
    parser.add_argument("--out", required=True, help="Cassette path (JSON)")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def _api_key(spec, source: str) -> str:
    if source == "env":
        key = os.environ.get(spec.api_key_env, "")
        if not key:
            raise RuntimeError(f"--key env: {spec.api_key_env} is not set")
        return key
    key = load_api_key(spec.name)
    if not key:
        raise RuntimeError(f"no API key for {spec.name!r} in the OS keychain; run deqio models setup or use --key env")
    return key


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = provider_spec(args.provider)
    if spec.kind != "remote":
        print(f"error: {spec.name!r} is not a remote provider", file=sys.stderr)
        return 1
    recorded: dict = {}

    def recording_transport(url, headers, body, timeout):
        from .providers import _urllib_transport

        status, response_headers, payload = _urllib_transport(url, headers, body, timeout)
        recorded.update({"status": status, "headers": response_headers, "body": payload})
        return status, response_headers, payload

    try:
        provider = RemoteDecisionProvider(
            spec, api_key=_api_key(spec, args.key), model=args.model, endpoint=args.endpoint,
            timeout=args.timeout, transport=recording_transport,
        )
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    started = time.perf_counter()
    try:
        payload, meta = provider.request(SAMPLE_STATE, SAMPLE_QUESTIONS)
    except Exception as error:  # the cassette is still written so the failure is reviewable
        if not recorded:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"warning: provider answered but the response was not usable: {error}", file=sys.stderr)
        payload = provider.build_request(SAMPLE_STATE, SAMPLE_QUESTIONS)[2]
        meta = None
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    cassette = cassette_from_exchange(
        provider, SAMPLE_STATE, SAMPLE_QUESTIONS, payload, recorded["status"], recorded["headers"], recorded["body"]
    )
    cassette["recorded_latency_ms"] = round(elapsed_ms, 3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cassette, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} (HTTP {recorded['status']}, {elapsed_ms:.1f} ms)")
    if meta is not None:
        print(f"remote_request_id={meta.remote_request_id} input_tokens={meta.input_tokens} model={meta.model}")
    return 0 if meta is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
