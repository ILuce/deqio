from __future__ import annotations

import hashlib
import json
import os
import socket
import platform
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import __version__
from .catalog import get_model, get_profile, load_catalog
from .config import Settings
from .installations import installation_record
from .console import (
    sidecar_model_wait,
    sidecar_process_ready,
    sidecar_ready,
    sidecar_start,
    start_sidecar_log_pump,
)


def _runtime_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _runtime_executable(env_dir: Path, name: str) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / f"{name}.exe"
    return env_dir / "bin" / name


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, process: subprocess.Popen[Any], timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Engine sidecar exited during startup with code {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting {timeout:.1f}s for engine sidecar process on port {port}")


def _json_body(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 180.0,
    body: bytes | None = None,
) -> dict[str, Any]:
    body = body if body is not None else _json_body(payload)
    request = Request(url, data=body, headers={"content-type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"System One sidecar returned HTTP {error.code}: {detail}") from error
    except (TimeoutError, socket.timeout) as error:
        raise RuntimeError(
            f"Timed out waiting for model response from {url} after {timeout:.1f}s"
        ) from error
    except URLError as error:
        reason = getattr(error, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise RuntimeError(
                f"Timed out waiting for model response from {url} after {timeout:.1f}s"
            ) from error
        raise RuntimeError(f"System One sidecar is unavailable: {error}") from error
    if not isinstance(result, dict):
        raise RuntimeError("System One sidecar returned a non-object response")
    return result


def _sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalise_probabilities(
    answer: dict[str, Any], option_ids: list[str]
) -> tuple[list[float], dict[str, Any]]:
    values = answer.get("probabilities")
    if isinstance(values, dict):
        probs = [float(values.get(option_id, 0.0)) for option_id in option_ids]
        total = sum(probs)
        if total > 0:
            normalized = abs(total - 1.0) > 1e-6
            output = [value / total for value in probs]
            return output, {
                "kind": "engine_probability",
                "source": "engine.probabilities",
                "synthetic": False,
                "normalized": normalized,
                "transforms": ["renormalized"] if normalized else [],
                "raw_logits_available": False,
                "calibration": "unspecified",
            }
    choice = answer.get("choice")
    if choice in option_ids:
        return [1.0 if option_id == choice else 0.0 for option_id in option_ids], {
            "kind": "synthetic_one_hot",
            "source": "engine.choice",
            "synthetic": True,
            "normalized": False,
            "transforms": ["one_hot_fallback"],
            "raw_logits_available": False,
            "calibration": "not-applicable",
        }
    raise RuntimeError("External engine did not return choice probabilities")


def _input_token_usage(response: dict[str, Any]) -> tuple[int | None, str]:
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    value = usage.get("input_tokens")
    if value is None:
        return None, "unknown"
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, "unknown"
    if parsed < 0:
        return None, "unknown"
    return parsed, "engine_reported"

def _choice_raw(
    *, row: dict[str, Any], answer: dict[str, Any], response: dict[str, Any], payload: dict[str, Any], latency_ms: float
) -> dict[str, Any]:
    option_ids = [str(option["id"]) for option in row["options"]]
    probabilities, score_provenance = _normalise_probabilities(answer, option_ids)
    input_tokens, input_tokens_source = _input_token_usage(response)
    probability_status = (
        "synthetic one-hot probabilities derived from the engine choice; not a model confidence score"
        if score_provenance["synthetic"]
        else (
            "probabilities reported by the selected System One engine and renormalized by Deqio; raw option logits unavailable"
            if score_provenance["normalized"]
            else "probabilities reported by the selected System One engine; raw option logits unavailable"
        )
    )
    return {
        "id": row["id"],
        "option_ids": option_ids,
        "probabilities": probabilities,
        "input_tokens": input_tokens,
        "input_tokens_source": input_tokens_source,
        "engine_payload_sha256": hashlib.sha256(_json_body(payload)).hexdigest(),
        "total_seconds": latency_ms / 1000.0,
        "prompt_sha256": _sha(payload),
        "probability_status": probability_status,
        "score_provenance": score_provenance,
    }


def _noul_raw(
    *, row: dict[str, Any], answer: dict[str, Any], response: dict[str, Any], payload: dict[str, Any], latency_ms: float
) -> dict[str, Any]:
    raw_p_yes = float(answer.get("noul"))
    p_yes = max(0.0, min(1.0, raw_p_yes))
    clamped = p_yes != raw_p_yes
    input_tokens, input_tokens_source = _input_token_usage(response)
    return {
        "id": row["id"],
        "option_ids": ["yes", "no"],
        "probabilities": [p_yes, 1.0 - p_yes],
        "input_tokens": input_tokens,
        "input_tokens_source": input_tokens_source,
        "engine_payload_sha256": hashlib.sha256(_json_body(payload)).hexdigest(),
        "total_seconds": latency_ms / 1000.0,
        "prompt_sha256": _sha(payload),
        "probability_status": (
            "native Noul probability reported by the selected System One engine and clamped to [0, 1]"
            if clamped
            else "native Noul probability reported by the selected System One engine"
        ),
        "score_provenance": {
            "kind": "engine_probability",
            "source": "engine.noul",
            "synthetic": False,
            "normalized": False,
            "transforms": ["clamped_to_unit_interval"] if clamped else [],
            "raw_logits_available": False,
            "calibration": "unspecified",
        },
    }


def _runtime_identity(settings: Settings, profile: dict[str, Any], instance_id: str) -> dict[str, Any]:
    record = installation_record(settings.config_path, settings.model_id, settings.backend) or {}
    artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), list) else []
    resolved = bool(artifacts) and all(
        (
            bool(item.get("resolved_revision"))
            if item.get("source") == "huggingface"
            else bool(item.get("sha256") or item.get("resolved_revision"))
        )
        for item in artifacts
        if isinstance(item, dict)
    )
    return {
        "deqio_version": __version__,
        "runtime_instance_id": instance_id,
        "engine": settings.engine,
        "model_id": settings.model_id,
        "backend": settings.backend,
        "model": profile.get("model", settings.model),
        "requested_revision": profile.get("model_revision"),
        "artifacts": deepcopy(artifacts),
        "artifact_revisions_resolved": bool(resolved),
        "installation_verified_at": record.get("verified_at"),
        "max_input_tokens": int(record.get("max_input_tokens", settings.max_tokens)),
    }


class SystemOneRuntime:
    """Adapter around a Jev-compatible engine running in an isolated uv environment."""

    def __init__(
        self,
        settings: Settings,
        entry: dict[str, Any],
        profile: dict[str, Any],
        process: subprocess.Popen[Any],
        port: int,
        log_thread: Any = None,
    ) -> None:
        self.settings = settings
        self.entry = entry
        self.profile = profile
        self.process = process
        self.port = port
        self.name = settings.backend
        self.engine = settings.engine
        self.base_url = f"http://127.0.0.1:{port}"
        self._log_thread = log_thread
        self.runtime_instance_id = uuid4().hex
        self._identity = _runtime_identity(settings, profile, self.runtime_instance_id)

    def identity_snapshot(self) -> dict[str, Any]:
        """Return immutable request-facing identity for this loaded runtime instance."""
        return deepcopy(self._identity)

    def refresh_identity(self) -> None:
        """Refresh installation metadata without changing this runtime instance identity."""
        self._identity = _runtime_identity(self.settings, self.profile, self.runtime_instance_id)

    @classmethod
    def load(cls, settings: Settings) -> "SystemOneRuntime":
        catalog = load_catalog(settings.model_catalog)
        entry = get_model(catalog, settings.model_id)
        profile = get_profile(catalog, settings.model_id, settings.backend)
        if str(entry.get("engine")) != settings.engine:
            raise RuntimeError(
                f"config engine={settings.engine!r} does not match catalog engine={entry.get('engine')!r} "
                f"for model_id={settings.model_id!r}"
            )
        runtime_key = str(profile.get("runtime_key", ""))
        if not runtime_key:
            raise RuntimeError(f"Model {settings.model_id} does not define an isolated runtime")
        env_dir = settings.runtime_dir / runtime_key
        python = _runtime_python(env_dir)
        if not python.is_file():
            raise RuntimeError(
                f"Runtime for {settings.model_id} is not installed. Run: "
                f"deqio models setup"
            )

        cls._validate_accelerator(settings, python)

        port = _free_port()
        env = os.environ.copy()
        command = cls._command(settings, profile, env_dir, python, port, env)
        if settings.hf_offline_runtime:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        address = f"http://127.0.0.1:{port}"
        sidecar_start(engine=settings.engine, address=address)
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        log_thread = start_sidecar_log_pump(process, settings.engine)
        runtime = cls(settings, entry, profile, process, port, log_thread)
        try:
            _wait_for_port(port, process, timeout=float(settings.sidecar_process_ready_seconds))
            sidecar_process_ready(engine=settings.engine, address=address)
            sidecar_model_wait(engine=settings.engine, timeout_seconds=float(settings.sidecar_startup_seconds))
            runtime._probe_model_ready(timeout=float(settings.sidecar_startup_seconds))
        except Exception:
            runtime.close()
            raise
        sidecar_ready(engine=settings.engine, address=address)
        return runtime

    @staticmethod
    def _validate_accelerator(settings: Settings, python: Path) -> None:
        if settings.backend == "mlx":
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise RuntimeError("MLX backend requires macOS on Apple Silicon (Darwin arm64)")
            return
        if settings.backend == "mps":
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise RuntimeError("MPS backend requires macOS on Apple Silicon (Darwin arm64)")
            probe = subprocess.run(
                [
                    str(python),
                    "-c",
                    "import torch,sys; sys.exit(0 if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 1)",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if probe.returncode != 0:
                raise RuntimeError("MPS backend was selected, but PyTorch MPS is not available in the model runtime")

    @staticmethod
    def _command(
        settings: Settings,
        profile: dict[str, Any],
        env_dir: Path,
        python: Path,
        port: int,
        env: dict[str, str],
    ) -> list[str]:
        engine = settings.engine
        model = str(profile["model"])

        if engine == "semif":
            sidecar = Path(__file__).with_name("semif_sidecar.py").resolve()
            return [
                str(python), str(sidecar),
                "--backend", settings.backend,
                "--model", settings.model,
                "--revision", settings.model_revision,
                "--max-tokens", str(settings.max_tokens),
                "--mlx-cache-mib", str(settings.mlx_cache_mib),
                "--torch-dtype", settings.torch_dtype,
                "--port", str(port),
            ]

        if engine == "kev":
            if settings.backend in {"cuda", "mps"}:
                env["KEV_BACKEND"] = "torch"
            else:
                env["KEV_BACKEND"] = "auto"
            return [str(python), "-m", "kev.serve", "--run", model, "--port", str(port)]

        if engine == "jevk5":
            if settings.backend != "cuda":
                raise RuntimeError("JevK5's native Deqio profile requires CUDA")
            return [
                str(python), "-m", "jevk5.server",
                "--model", model,
                "--host", "127.0.0.1",
                "--port", str(port),
            ]

        if engine == "open-jev":
            if settings.backend != "cuda":
                raise RuntimeError("Open-Jev's current Deqio profiles require CUDA")
            return [
                str(python), "-m", "jev.server",
                "--checkpoint", settings.model,
                "--device", "cuda:0",
                "--max-length", str(getattr(settings, "max_tokens", profile.get("open_jev_max_length", 4096))),
                "--batch-size", "1",
                "--no-prefix-cache",
                "--host", "127.0.0.1",
                "--port", str(port),
            ]

        if engine == "clm":
            if settings.backend != "cuda":
                raise RuntimeError("CLM's current Deqio profile requires CUDA")
            sidecar = Path(__file__).with_name("clm_sidecar.py").resolve()
            checkpoint = Path(str(profile.get("clm_checkpoint", ""))).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = (settings.config_path.parent / checkpoint).resolve()
            if not checkpoint.is_file():
                raise RuntimeError(
                    f"CLM checkpoint is missing: {checkpoint}. "
                    f"Run: deqio models setup"
                )
            return [
                str(python), str(sidecar),
                "--encoder-model", model,
                "--embedding-model", str(profile.get("clm_embedding_model", "qwen3-8b")),
                "--checkpoint", str(checkpoint),
                "--max-tokens", str(getattr(settings, "max_tokens", profile.get("clm_max_tokens", 2048))),
                "--gpu-memory-utilization", str(profile.get("clm_gpu_memory_utilization", 0.35)),
                "--port", str(port),
            ]

        if engine == "basal":
            if settings.backend not in {"mlx", "cuda"}:
                raise RuntimeError("Basal's Deqio profiles support mlx and cuda")
            executable = _runtime_executable(env_dir, "basal-serve")
            if not executable.is_file():
                raise RuntimeError(f"Basal executable was not installed: {executable}")
            command = [str(executable), "--model", model]
            mode = profile.get("basal_mode")
            if mode:
                command.extend(["--mode", str(mode)])
            command.extend(["--port", str(port)])
            return command

        if engine == "decider":
            env["DECIDER_MODEL"] = model
            env["DECIDER_DEVICE"] = "cuda" if settings.backend == "cuda" else "mps"
            return [
                str(python), "-m", "uvicorn", "decider.serve:app",
                "--host", "127.0.0.1", "--port", str(port), "--no-access-log",
            ]

        if engine == "laya":
            launcher = str(profile.get("launcher", "laya"))
            if launcher == "laya_mlx":
                sidecar = Path(__file__).with_name("laya_mlx_sidecar.py").resolve()
                return [str(python), str(sidecar), "--model", model, "--port", str(port)]
            env["LAYA_HOST"] = "127.0.0.1"
            env["LAYA_PORT"] = str(port)
            env["LAYA_DEVICE"] = settings.backend
            # Bind the HTTP process first; the Deqio readiness probe loads the selected model.
            env["LAYA_PRELOAD"] = "0"
            env["LAYA_MODELS"] = str(profile.get("laya_model", ""))
            return [str(python), "-m", "laya.serve"]

        if engine == "nimble":
            sidecar = Path(__file__).with_name("nimble_sidecar.py").resolve()
            source_root = settings.runtime_dir / str(profile.get("source_key", "nimble-src"))
            model_config = settings.runtime_dir / str(profile.get("model_config", "nimble-model.json"))
            if not source_root.is_dir():
                raise RuntimeError(f"Nimble source checkout is missing: {source_root}")
            if not model_config.is_file():
                raise RuntimeError(
                    f"Nimble prepared model config is missing: {model_config}. "
                    f"Run: deqio models setup"
                )
            return [
                str(python), str(sidecar),
                "--source-root", str(source_root),
                "--model-config", str(model_config),
                "--backend", settings.backend,
                "--port", str(port),
            ]

        if engine == "von":
            executable = _runtime_executable(env_dir, "von")
            if not executable.is_file():
                raise RuntimeError(f"Von executable was not installed: {executable}")
            return [str(executable), "serve", "--host", "127.0.0.1", "--port", str(port)]

        raise RuntimeError(f"Unsupported external engine: {engine}")

    def _probe_model_ready(self, *, timeout: float) -> None:
        payload = {
            "model": str(self.profile.get("wire_model", self.settings.model)),
            "state": "Deqio startup readiness probe.",
            "questions": {
                "ready": {
                    "type": "choice",
                    "instructions": "Is the inference model loaded and able to answer a typed decision?",
                    "criteria": {
                        "ready": "The model can answer this request.",
                        "not_ready": "The model cannot answer this request.",
                    },
                }
            },
        }
        response = _post_json(f"{self.base_url}/v1/systemone", payload, timeout=timeout)
        answers = response.get("answers")
        if not isinstance(answers, dict) or not isinstance(answers.get("ready"), dict):
            raise RuntimeError("Engine sidecar opened its port but did not pass the model-readiness probe")

    def _request(
        self, state: Any, questions: dict[str, Any], execution_mode: str | None = None
    ) -> tuple[dict[str, Any], dict[str, Any], float]:
        payload = {
            "model": str(self.profile.get("wire_model", self.settings.model)),
            "state": state,
            "questions": questions,
        }
        if execution_mode:
            payload["execution_mode"] = execution_mode
        body = _json_body(payload)
        started = time.perf_counter()
        response = _post_json(
            f"{self.base_url}/v1/systemone", payload, timeout=180.0, body=body
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency_ms = float(response.get("latency_ms", elapsed_ms) or elapsed_ms)
        return payload, response, latency_ms

    def score(self, row: dict[str, Any], mode: str) -> dict[str, Any]:
        question_id = "decision"
        criteria = {
            str(option["id"]): str(option.get("description", "")) or None
            for option in row["options"]
        }
        questions = {
            question_id: {
                "type": "choice",
                "instructions": str(row["question"]),
                "criteria": criteria,
            }
        }
        payload, response, latency_ms = self._request(row["state"], questions, mode)
        answers = response.get("answers")
        if not isinstance(answers, dict) or not isinstance(answers.get(question_id), dict):
            raise RuntimeError("External engine response is missing answers.decision")
        return _choice_raw(
            row=row,
            answer=answers[question_id],
            response=response,
            payload=payload,
            latency_ms=latency_ms,
        )

    def score_noul(self, row: dict[str, Any], mode: str) -> dict[str, Any]:
        question_id = "decision"
        questions = {
            question_id: {
                "type": "noul",
                "instructions": str(row["question"]),
            }
        }
        payload, response, latency_ms = self._request(row["state"], questions, mode)
        answers = response.get("answers")
        if not isinstance(answers, dict) or not isinstance(answers.get(question_id), dict):
            raise RuntimeError("External engine response is missing answers.decision")
        return _noul_raw(
            row=row,
            answer=answers[question_id],
            response=response,
            payload=payload,
            latency_ms=latency_ms,
        )

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
                "criteria": {
                    str(option["id"]): str(option.get("description", "")) or None
                    for option in row["options"]
                },
            }

        payload, response, latency_ms = self._request(rows[0]["state"], questions, "shared")
        answers = response.get("answers")
        if not isinstance(answers, dict):
            raise RuntimeError("External engine response is missing answers")
        raw_results = []
        for row, qid in zip(rows, qids):
            answer = answers.get(qid)
            if not isinstance(answer, dict):
                raise RuntimeError(f"External engine response is missing answers.{qid}")
            raw_results.append(
                _choice_raw(
                    row=row,
                    answer=answer,
                    response=response,
                    payload=payload,
                    latency_ms=latency_ms,
                )
            )
        return raw_results, {
            "total_seconds": latency_ms / 1000.0,
            "batch_size": len(rows),
            "engine": self.engine,
        }

    def input_completeness_capability(self) -> dict[str, Any]:
        """Describe whether this engine can prove the exact model input it consumed.

        System-One-compatible HTTP servers expose decisions and often token counts,
        but the current wire does not expose the exact rendered/tokenized sequence,
        attention masking, or cache consumption needed for a complete receipt.
        """
        return {
            "status": "unavailable",
            "reason": "backend_model_input_not_instrumented",
            "engine": self.engine,
        }

    def clear_cache(self) -> dict[str, Any]:
        if self.engine == "semif":
            result = _post_json(f"{self.base_url}/v1/cache/clear", {})
            result.setdefault("model_loaded", True)
            return result
        return {
            "engine": self.engine,
            "backend": self.name,
            "cache_clear_supported": False,
            "prefix_cache": "not exposed by this engine sidecar",
            "model_loaded": True,
        }

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self._log_thread is not None:
            self._log_thread.join(timeout=1)
