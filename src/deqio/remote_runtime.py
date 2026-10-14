"""Runtime adapter for ``remote`` catalog profiles (D11, backend form since 0.6.1).

A remote profile (``openai-decisions``, ``typesafe-jev``) is installed like a local model and
served by this adapter instead of a sidecar: it exposes the surface the server and the benchmark
use on ``SystemOneRuntime`` (``score``, ``score_noul``, ``score_shared``, ``system_one``, identity,
``close``), sends every request to the provider's documented endpoint and carries ``provider``,
``remote_request_id`` and ``cost_estimate`` into every raw result for provenance.

The model identifier comes from the installation registry (typed at ``deqio models setup``) and
``config.json`` is only its mirror (B10): a mirror that differs refuses to load.  The API key is read
from the OS keychain (``deqio.credentials``) and never from the environment or the workspace.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from . import __version__
from .catalog import get_model, get_profile, load_catalog
from .config import Settings
from .credentials import load_api_key
from .installations import installation_record
from .providers import ProviderError, RemoteDecisionProvider, cost_estimate, provider_spec
from .systemone_runtime import SystemOneCapabilityError, SystemOneProtocolError, _choice_raw, _noul_raw

REMOTE_BACKEND = "remote"
PROBE_QUESTION = {
    "ready": {
        "type": "choice",
        "instructions": "Is the decision model reachable and able to answer a typed decision?",
        "criteria": {
            "ready": "The model can answer this request.",
            "not_ready": "The model cannot answer this request.",
        },
    }
}


class RemoteProviderRuntime:
    def __init__(
        self,
        settings: Settings,
        provider: RemoteDecisionProvider,
        *,
        entry: dict[str, Any] | None = None,
        profile: dict[str, Any] | None = None,
        record: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.engine = provider.spec.name
        self.runtime_instance_id = f"remote-{uuid4().hex}"
        self.entry = dict(entry or {"id": settings.model_id, "engine": provider.spec.name})
        self.profile: dict[str, Any] = deepcopy(profile) if profile else {
            "model": provider.model,
            "model_revision": "remote",
            "source": "official",
            "capabilities": {
                "systemone": True, "choice": True, "noul": True, "score": True, "multi_question": True,
                "multi": False, "act": False, "facts": False, "evidence": False,
            },
        }
        self.record = dict(record or {})
        self._identity = self._build_identity()

    @classmethod
    def load(
        cls, settings: Settings, *, api_key: str | None = None, remote_model: str | None = None
    ) -> "RemoteProviderRuntime":
        if settings.backend != REMOTE_BACKEND:
            raise ProviderError(f"RemoteProviderRuntime serves backend {REMOTE_BACKEND!r}, not {settings.backend!r}")
        catalog = load_catalog(settings.model_catalog)
        entry = get_model(catalog, settings.model_id)
        profile = get_profile(catalog, settings.model_id, REMOTE_BACKEND)
        if str(entry.get("engine")) != settings.engine:
            raise RuntimeError(
                f"config engine={settings.engine!r} does not match catalog engine={entry.get('engine')!r} "
                f"for model_id={settings.model_id!r}"
            )
        name = str(profile.get("provider") or entry.get("engine"))
        spec = provider_spec(name)
        if spec.kind != "remote":
            raise ProviderError(f"Catalog profile {settings.model_id}/remote names a non-remote provider {name!r}")
        record = installation_record(settings.config_path, settings.model_id, REMOTE_BACKEND) or {}
        if remote_model is None:
            model = str(record.get("remote_model") or "")
            if not model:
                raise RuntimeError(
                    f"Remote profile {settings.model_id}/remote is not installed in this workspace. Run: deqio models setup"
                )
            # B10: config.json mirrors the installed model identifier; a different value never loads.
            if str(settings.model) != model:
                raise RuntimeError(
                    f"config.json model={settings.model!r} differs from the installed remote model {model!r} of "
                    f"{settings.model_id}/remote; run deqio models use {settings.model_id} --backend remote"
                )
        else:
            model = str(remote_model)
        endpoint = str(profile.get("endpoint") or spec.default_endpoint)
        key = api_key if api_key is not None else load_api_key(name)
        if not key:
            raise RuntimeError(
                f"No API key for {name!r} in the OS keychain (service 'deqio', account {name!r}). Run: deqio models setup"
            )
        provider = RemoteDecisionProvider(spec, api_key=key, model=model, endpoint=endpoint)
        return cls(settings, provider, entry=entry, profile=profile, record=record)

    # --- identity -------------------------------------------------------------------------
    def _build_identity(self) -> dict[str, Any]:
        descriptor = self.provider.describe()
        capabilities = self.profile.get("capabilities") if isinstance(self.profile.get("capabilities"), dict) else {}
        return {
            "deqio_version": __version__,
            "runtime_instance_id": self.runtime_instance_id,
            "engine": self.engine,
            "model_id": str(self.settings.model_id),
            "backend": REMOTE_BACKEND,
            "model": self.provider.model,
            "requested_revision": None,
            "artifacts": [],
            "artifact_revisions_resolved": False,
            "installation_installed_at": self.record.get("installed_at"),
            "installation_verified_at": self.record.get("verified_at"),
            "artifact_verification_status": "remote",
            "max_input_tokens": int(getattr(self.settings, "max_tokens", 0) or 0),
            "source": str(self.profile.get("source") or "official"),
            "platform": REMOTE_BACKEND,
            "capabilities": deepcopy(capabilities),
            "provider": descriptor,
        }

    def identity_snapshot(self) -> dict[str, Any]:
        return deepcopy(self._identity)

    def refresh_identity(self) -> None:
        self.record = installation_record(self.settings.config_path, self.settings.model_id, REMOTE_BACKEND) or self.record
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
    def probe(self) -> dict[str, Any]:
        """One typed decision against the real endpoint: the install-time readiness check."""
        _payload, response, _meta = self._call("Deqio installation readiness probe.", deepcopy(PROBE_QUESTION))
        answer = response["answers"].get("ready")
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise SystemOneProtocolError(f"{self.engine} did not answer the readiness probe with a choice")
        return answer

    def _annotate(self, raw: dict[str, Any], response_meta, *, scope: str) -> dict[str, Any]:
        raw["provider"] = self.provider.describe()
        raw["remote_request_id"] = response_meta.remote_request_id
        raw["cost_estimate"] = cost_estimate(self.provider.spec, response_meta.input_tokens, scope=scope)
        return raw

    def _call(self, state: Any, questions: dict[str, Any], request_options: dict[str, Any] | None = None):
        payload, meta = self.provider.request(state, questions, request_options)
        response: dict[str, Any] = {
            "model": meta.model or self.provider.model,
            "answers": meta.answers,
            "usage": {"input_tokens": meta.input_tokens},
        }
        if meta.remote_request_id:
            response["id"] = meta.remote_request_id
        return payload, response, meta

    def _check_capabilities(self, questions: dict[str, Any], request_options: dict[str, Any]) -> None:
        capabilities = self.profile.get("capabilities") if isinstance(self.profile.get("capabilities"), dict) else {}
        if request_options:
            unsupported = ", ".join(sorted(request_options))
            raise SystemOneCapabilityError(
                unsupported, f"Remote provider {self.engine} does not support System One request option(s): {unsupported}"
            )
        if len(questions) > 1 and not capabilities.get("multi_question", False):
            raise SystemOneCapabilityError("multi_question", "Active profile does not support multiple System One questions")
        for question_id, question in questions.items():
            if not isinstance(question, dict):
                raise RuntimeError(f"Question {question_id!r} must be an object")
            question_type = str(question.get("type", "choice"))
            if not capabilities.get(question_type, False):
                raise SystemOneCapabilityError(
                    question_type, f"Active profile does not support System One capability {question_type!r}"
                )
            for option in ("option_keys", "evidence"):
                if question.get(option) and not capabilities.get(option, False):
                    raise SystemOneCapabilityError(option, f"Remote provider {self.engine} does not support {option}")

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
        options = dict(request_options or {})
        self._check_capabilities(questions, options)
        _payload, response, meta = self._call(state, questions)
        native = dict(response)
        native["remote_request_id"] = meta.remote_request_id
        return native, {
            "total_seconds": meta.latency_ms / 1000.0,
            "batch_size": len(questions),
            "decisions": len(questions),
            "engine": self.engine,
            "remote_request_id": meta.remote_request_id,
            "cost_estimate": cost_estimate(self.provider.spec, meta.input_tokens, scope="batch"),
        }
