"""Record one real exchange with a remote decision provider for the contract tests (D11).

    python -m deqio.provider_record --provider openai-decisions --out tests/fixtures/providers/openai-decisions/choice.json

The provider's flag and API key must be in the environment; the key is never written.  The
cassette replaces the synthetic fixture of the same name, and ``tests/test_providers.py`` replays it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
    parser.add_argument("--base-url", default="", help="Endpoint base URL (default: the provider's)")
    parser.add_argument("--out", required=True, help="Cassette path (JSON)")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = provider_spec(args.provider)
    recorded: dict = {}

    def recording_transport(url, headers, body, timeout):
        from .providers import _urllib_transport

        status, response_headers, payload = _urllib_transport(url, headers, body, timeout)
        recorded.update({"status": status, "headers": response_headers, "body": payload})
        return status, response_headers, payload

    provider = RemoteDecisionProvider(
        spec, model=args.model, base_url=args.base_url, timeout=args.timeout, transport=recording_transport
    )
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
    cassette = cassette_from_exchange(provider, payload, recorded["status"], recorded["headers"], recorded["body"])
    cassette["recorded_latency_ms"] = round(elapsed_ms, 3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cassette, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} (HTTP {recorded['status']}, {elapsed_ms:.1f} ms)")
    if meta is not None:
        print(f"remote_request_id={meta.remote_request_id} input_tokens={meta.input_tokens}")
    return 0 if meta is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
