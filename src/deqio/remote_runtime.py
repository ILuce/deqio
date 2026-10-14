"""Runtime adapter for remote decision providers (D11).

Exposes the surface the server and the benchmark use on ``SystemOneRuntime`` (``score``,
``score_noul``, ``score_shared``, ``system_one``, identity, ``close``) without a sidecar: every
request goes to the configured remote provider, and every raw result carries ``provider``,
``remote_request_id`` and ``cost_estimate`` for provenance.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any
from uuid import uuid4

from . import __version__
from .config import Settings
from .providers import LOCAL_PROVIDER, ProviderError, RemoteDecisionProvider, cost_estimate, provider_spec
from .systemone_runtime import SystemOneProtocolError, _choice_raw, _noul_raw

REMOTE_BACKEND = "remote"


class RemoteProviderRuntime:
    def __init__(self, settings: Settings, provider: RemoteDecisionProvider) -> None:
        self.settings = settings
        self.provider = provider
        self.engine = provider.spec.name
        self.runtime_instance_id = f"remote-{uuid4().hex}"
        self.profile: dict[str, Any] = {
            "model": provider.model,
            "model_revision": None,
            "source": "remote",
            "capabilities": {
                "choice": True,
                "noul": True,
                "shared": True,
                "multi_question": True,
                "score": False,
                "multi": False,
                "act": False,
            },
        }
        self._identity = self._build_identity()

    @classmethod
    def load(cls, settings: Settings) -> "RemoteProviderRuntime":
        spec = provider_spec(settings.provider)
        if spec.kind != "remote":
            raise ProviderError(f"Provider {spec.name!r} is not remote; use SystemOneRuntime for {LOCAL_PROVIDER}")
        provider = RemoteDecisionProvider(
            spec,
            model=getattr(settings, "provider_model", "") or "",
            base_url=getattr(settings, "provider_base_url", "") or "",
            environ=os.environ,
        )
        return cls(settings, provider)

    # --- identity -------------------------------------------------------------------------
    def _build_identity(self) -> dict[str, Any]:
        descriptor = self.provider.describe()
        return {
            "deqio_version": __version__,
            "runtime_instance_id": self.runtime_instance_id,
            "engine": self.engine,
            "model_id": self.provider.model,
            "backend": REMOTE_BACKEND,
            "model": self.provider.model,
            "requested_revision": None,
            "artifacts": [],
            "artifact_revisions_resolved": False,
            "installation_installed_at": None,
            "installation_verified_at": None,
            "artifact_verification_status": "remote",
            "max_input_tokens": int(getattr(self.settings, "max_tokens", 0) or 0),
            "source": "remote",
            "capabilities": deepcopy(self.profile["capabilities"]),
            "provider": descriptor,
        }

    def identity_snapshot(self) -> dict[str, Any]:
        return deepcopy(self._identity)

    def refresh_identity(self) -> None:
        self._identity = self._build_identity()

    def provider_descriptor(self) -> dict[str, Any]:
        return self.provider.describe()

    def input_completeness_capability(self) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "reason": "remote_provider_input_not_instrumented",
            "detail": "A remote decision API does not expose the exact model input it consumed.",
        }

    def clear_cache(self) -> dict[str, Any]:
        return {"cleared": False, "reason": "remote_provider_has_no_local_cache"}

    def close(self) -> None:
        return None

    # --- decisions ------------------------------------------------------------------------
    def _annotate(self, raw: dict[str, Any], response_meta, *, scope: str) -> dict[str, Any]:
        raw["provider"] = self.provider.describe()
        raw["remote_request_id"] = response_meta.remote_request_id
        raw["cost_estimate"] = cost_estimate(self.provider.spec, response_meta.input_tokens, scope=scope)
        return raw

    def _call(self, state: Any, questions: dict[str, Any], request_options: dict[str, Any] | None = None):
        payload, meta = self.provider.request(state, questions, request_options)
        response = {"answers": meta.answers, "usage": {"input_tokens": meta.input_tokens}}
        if meta.remote_request_id:
            response["id"] = meta.remote_request_id
        return payload, response, meta

    def score(self, row: dict[str, Any], mode: str) -> dict[str, Any]:
        question_id = "decision"
        questions = {
            question_id: {
                "type": "choice",
                "instructions": str(row["question"]),
                "criteria": {str(option["id"]): str(option.get("description", "")) or None for option in row["options"]},
            }
        }
        payload, response, meta = self._call(row["state"], questions)
        answer = response["answers"].get(question_id)
        if not isinstance(answer, dict):
            raise SystemOneProtocolError(f"{self.engine} response is missing answers.decision")
        raw = _choice_raw(row=row, answer=answer, response=response, payload=payload, latency_ms=meta.latency_ms, preserve_native=False)
        return self._annotate(raw, meta, scope="decision")

    def score_noul(self, row: dict[str, Any], mode: str) -> dict[str, Any]:
        question_id = "decision"
        questions = {question_id: {"type": "noul", "instructions": str(row["question"])}}
        payload, response, meta = self._call(row["state"], questions)
        answer = response["answers"].get(question_id)
        if not isinstance(answer, dict):
            raise SystemOneProtocolError(f"{self.engine} response is missing answers.decision")
        raw = _noul_raw(row=row, answer=answer, response=response, payload=payload, latency_ms=meta.latency_ms, preserve_native=False)
        return self._annotate(raw, meta, scope="decision")

    def score_shared(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not rows:
            return [], {"total_seconds": 0.0, "batch_size": 0}
        questions: dict[str, Any] = {}
        qids: list[str] = []
        for index, row in enumerate(rows):
            qid = f"q{index}"
            qids.append(qid)
            questions[qid] = {
                "type": "choice",
                "instructions": str(row["question"]),
                "criteria": {str(option["id"]): str(option.get("description", "")) or None for option in row["options"]},
            }
        payload, response, meta = self._call(rows[0]["state"], questions)
        raw_results = []
        for row, qid in zip(rows, qids):
            answer = response["answers"].get(qid)
            if not isinstance(answer, dict):
                raise SystemOneProtocolError(f"{self.engine} response is missing answers.{qid}")
            raw = _choice_raw(row=row, answer=answer, response=response, payload=payload, latency_ms=meta.latency_ms, preserve_native=False)
            raw["batch_input_tokens"] = meta.input_tokens
            raw["batch_input_tokens_source"] = "engine_reported" if meta.input_tokens else "unknown"
            if len(rows) > 1:
                raw["input_tokens"] = None
                raw["input_tokens_source"] = "engine_reported_batch" if meta.input_tokens else "unknown"
            raw_results.append(self._annotate(raw, meta, scope="batch"))
        return raw_results, {
            "total_seconds": meta.latency_ms / 1000.0,
            "batch_size": len(rows),
            "engine": self.engine,
            "remote_request_id": meta.remote_request_id,
            "cost_estimate": cost_estimate(self.provider.spec, meta.input_tokens, scope="batch"),
        }

    def system_one(
        self, state: Any, questions: dict[str, Any], request_options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(questions, dict) or not questions:
            raise RuntimeError("questions must be a non-empty object")
        payload, response, meta = self._call(state, questions, dict(request_options or {}))
        native = dict(meta.raw)
        native.setdefault("answers", meta.answers)
        return native, {
            "total_seconds": meta.latency_ms / 1000.0,
            "decisions": len(questions),
            "engine": self.engine,
            "remote_request_id": meta.remote_request_id,
            "cost_estimate": cost_estimate(self.provider.spec, meta.input_tokens, scope="batch"),
        }
