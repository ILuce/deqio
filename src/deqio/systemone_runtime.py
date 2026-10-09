from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import platform
import signal
import subprocess
import sys
import tempfile
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
from .process_lock import pgid_alive
from .console import (
    info,
    sidecar_model_wait,
    sidecar_process_ready,
    sidecar_ready,
    sidecar_start,
    sidecar_stop,
    sidecar_stopped,
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


# CUDA probe run with a profile's own runtime interpreter. Exit status 3: PyTorch
# cannot be imported; 4: PyTorch sees no CUDA device.
_TORCH_CUDA_PROBE_NO_TORCH = 3
_TORCH_CUDA_PROBE = (
    "import sys\n"
    "try:\n"
    "    import torch\n"
    "except Exception:\n"
    "    sys.exit(3)\n"
    "sys.exit(0 if torch.cuda.is_available() else 4)\n"
)


class SystemOneUnavailableError(RuntimeError):
    """The isolated inference runtime could not be reached in time."""


class SystemOneProtocolError(RuntimeError):
    """The isolated runtime returned an invalid or incomplete response."""


class SystemOneSidecarHTTPError(RuntimeError):
    """HTTP failure returned by an otherwise reachable System One sidecar."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = int(status_code)
        self.detail = detail
        super().__init__(f"System One sidecar returned HTTP {self.status_code}: {detail}")


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
        raise SystemOneSidecarHTTPError(error.code, detail) from error
    except (TimeoutError, socket.timeout) as error:
        raise SystemOneUnavailableError(
            f"Timed out waiting for model response from {url} after {timeout:.1f}s"
        ) from error
    except URLError as error:
        reason = getattr(error, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise SystemOneUnavailableError(
                f"Timed out waiting for model response from {url} after {timeout:.1f}s"
            ) from error
        raise SystemOneUnavailableError(f"System One sidecar is unavailable: {error}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SystemOneProtocolError("System One sidecar returned invalid JSON") from error
    if not isinstance(result, dict):
        raise SystemOneProtocolError("System One sidecar returned a non-object response")
    return result


def _sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _protocol_float(value: Any, *, field: str, minimum: float | None = None, maximum: float | None = None) -> float:
    """Parse a numeric sidecar field and classify malformed output as protocol failure."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise SystemOneProtocolError(f"External engine returned a non-numeric {field}") from error
    if not math.isfinite(parsed):
        raise SystemOneProtocolError(f"External engine returned a non-finite {field}")
    if minimum is not None and parsed < minimum:
        raise SystemOneProtocolError(f"External engine returned {field} below {minimum}")
    if maximum is not None and parsed > maximum:
        raise SystemOneProtocolError(f"External engine returned {field} above {maximum}")
    return parsed


def _normalise_probabilities(
    answer: dict[str, Any], option_ids: list[str]
) -> tuple[list[float], dict[str, Any]]:
    values = answer.get("probabilities")
    if isinstance(values, dict):
        missing = [option_id for option_id in option_ids if option_id not in values]
        if missing:
            raise SystemOneProtocolError(
                "External engine probabilities are missing option(s): " + ", ".join(missing)
            )
        try:
            probs = [
                _protocol_float(
                    values[option_id],
                    field=f"choice probability for {option_id!r}",
                    minimum=0.0,
                )
                for option_id in option_ids
            ]
        except SystemOneProtocolError as error:
            raise SystemOneProtocolError("External engine returned invalid choice probabilities") from error
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
    raise SystemOneProtocolError("External engine did not return choice probabilities")


def _native_choice_probabilities(
    answer: dict[str, Any], option_ids: list[str]
) -> tuple[list[float], dict[str, Any]]:
    """Return Decision-owned probabilities unchanged and fail if they are absent.

    Decision 2.0 explicitly forbids Deqio-side normalization and synthetic
    one-hot fallbacks.  The stable Deqio Choice endpoint may reshape the map
    into its historical ordered list, but it must not change the values.
    """
    values = answer.get("probabilities")
    if not isinstance(values, dict):
        raise SystemOneProtocolError("Decision 2.0 did not return native choice probabilities")
    missing = [option_id for option_id in option_ids if option_id not in values]
    if missing:
        raise SystemOneProtocolError(
            "Decision 2.0 probabilities are missing option(s): " + ", ".join(missing)
        )
    try:
        probs = [
            _protocol_float(
                values[option_id],
                field=f"Decision 2.0 choice probability for {option_id!r}",
                minimum=0.0,
                maximum=1.0,
            )
            for option_id in option_ids
        ]
    except SystemOneProtocolError as error:
        raise SystemOneProtocolError(
            "Decision 2.0 returned invalid native choice probabilities"
        ) from error
    return probs, {
        "kind": "engine_probability",
        "source": "decision2.probabilities",
        "synthetic": False,
        "normalized": False,
        "transforms": [],
        "raw_logits_available": False,
        "calibration": "decision2-native",
    }


def _input_token_usage(response: dict[str, Any]) -> tuple[int | None, str]:
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    value = usage.get("input_tokens")
    if value is None:
        return None, "unknown"
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, "unknown"
    if parsed < 1:
        # Every prompt has at least one token: 0 is an engine's "not measured"
        # placeholder, never a measurement, and must not be attested as one.
        return None, "unknown"
    return parsed, "engine_reported"

def _choice_raw(
    *,
    row: dict[str, Any],
    answer: dict[str, Any],
    response: dict[str, Any],
    payload: dict[str, Any],
    latency_ms: float,
    preserve_native: bool = False,
) -> dict[str, Any]:
    option_ids = [str(option["id"]) for option in row["options"]]
    probabilities, score_provenance = (
        _native_choice_probabilities(answer, option_ids)
        if preserve_native
        else _normalise_probabilities(answer, option_ids)
    )
    input_tokens, input_tokens_source = _input_token_usage(response)
    probability_status = (
        "native Decision 2.0 probabilities preserved unchanged by Deqio"
        if preserve_native
        else (
            "synthetic one-hot probabilities derived from the engine choice; not a model confidence score"
            if score_provenance["synthetic"]
            else (
                "probabilities reported by the selected System One engine and renormalized by Deqio; raw option logits unavailable"
                if score_provenance["normalized"]
                else "probabilities reported by the selected System One engine; raw option logits unavailable"
            )
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
    *,
    row: dict[str, Any],
    answer: dict[str, Any],
    response: dict[str, Any],
    payload: dict[str, Any],
    latency_ms: float,
    preserve_native: bool = False,
) -> dict[str, Any]:
    try:
        raw_p_yes = _protocol_float(answer.get("noul"), field="Noul probability")
    except SystemOneProtocolError as error:
        raise SystemOneProtocolError("External engine returned an invalid Noul probability") from error
    if preserve_native and not 0.0 <= raw_p_yes <= 1.0:
        raise SystemOneProtocolError("Decision 2.0 returned an invalid native Noul probability")
    p_yes = raw_p_yes if preserve_native else max(0.0, min(1.0, raw_p_yes))
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
            "native Decision 2.0 P(true) preserved unchanged by Deqio"
            if preserve_native
            else (
                "native Noul probability reported by the selected System One engine and clamped to [0, 1]"
                if clamped
                else "native Noul probability reported by the selected System One engine"
            )
        ),
        "score_provenance": {
            "kind": "engine_probability",
            "source": "engine.noul",
            "synthetic": False,
            "normalized": False,
            "transforms": ["clamped_to_unit_interval"] if clamped else [],
            "raw_logits_available": False,
            "calibration": "decision2-native" if preserve_native else "unspecified",
        },
    }


def _runtime_identity(
    settings: Settings,
    profile: dict[str, Any],
    instance_id: str,
    runtime_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    identity = {
        "deqio_version": __version__,
        "runtime_instance_id": instance_id,
        "engine": settings.engine,
        "model_id": settings.model_id,
        "backend": settings.backend,
        "model": profile.get("model", settings.model),
        "requested_revision": profile.get("model_revision"),
        "artifacts": deepcopy(artifacts),
        "artifact_revisions_resolved": bool(resolved),
        "installation_installed_at": record.get("installed_at"),
        "installation_verified_at": record.get("verified_at"),
        "artifact_verification_status": "verified" if record.get("verified_at") and resolved else "unresolved",
        "max_input_tokens": int(record.get("max_input_tokens", settings.max_tokens)),
    }
    for key in ("family", "runtime", "platform", "source", "capabilities", "precision"):
        if profile.get(key) is not None:
            identity[key] = deepcopy(profile[key])
    if runtime_metadata:
        identity["runtime_metadata"] = deepcopy(runtime_metadata)
        for key in ("runtime_version", "cuda_device", "cuda_runtime", "torch_version"):
            if runtime_metadata.get(key) is not None:
                identity[key] = runtime_metadata[key]
    quantization = profile.get("quantization")
    if isinstance(quantization, dict):
        identity["quantization"] = deepcopy(quantization)
    return identity


class SystemOneCapabilityError(RuntimeError):
    """A requested native SystemOne feature is not supported by the active profile."""

    def __init__(self, capability: str, message: str) -> None:
        super().__init__(message)
        self.capability = capability


def _profile_with_installed_pin(settings: Settings, profile: dict[str, Any]) -> dict[str, Any]:
    """Use the immutable Hub revision recorded when a pin-on-install profile was verified."""
    if not profile.get("pin_hf_revisions_at_install"):
        return profile
    selected_revision = str(settings.model_revision or "")
    if len(selected_revision) == 40 and all(ch in "0123456789abcdefABCDEF" for ch in selected_revision):
        pinned = deepcopy(profile)
        pinned["model_revision"] = selected_revision
        return pinned

    record = installation_record(settings.config_path, settings.model_id, settings.backend) or {}
    artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), list) else []
    model_repo = str(profile.get("model", ""))
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("source") != "huggingface" or str(artifact.get("repo_id", "")) != model_repo:
            continue
        # Pin-on-install is a generic catalog feature.  Do not couple runtime
        # recovery to Basal-specific artifact roles: official GGUF profiles for
        # Decider, Kev, JevK5, Laya and Clef use their own role names while the
        # repository revision is still the immutable identity we need here.
        resolved = artifact.get("resolved_revision")
        if resolved:
            pinned = deepcopy(profile)
            pinned["model_revision"] = str(resolved)
            return pinned

    raise RuntimeError(
        f"Installed profile {settings.model_id}/{settings.backend} is missing its immutable model revision; "
        "run deqio models update for this profile"
    )


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
        runtime_metadata: dict[str, Any] | None = None,
        engine_pid_file: Path | None = None,
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
        self._engine_pid_file = engine_pid_file
        self.runtime_instance_id = uuid4().hex
        self._runtime_metadata = deepcopy(runtime_metadata or {})
        self._identity = _runtime_identity(
            settings, profile, self.runtime_instance_id, self._runtime_metadata
        )

    def identity_snapshot(self) -> dict[str, Any]:
        """Return immutable request-facing identity for this loaded runtime instance."""
        return deepcopy(self._identity)

    def refresh_identity(self) -> None:
        """Refresh installation metadata without changing this runtime instance identity."""
        self._identity = _runtime_identity(
            self.settings, self.profile, self.runtime_instance_id, self._runtime_metadata
        )

    @classmethod
    def load(cls, settings: Settings) -> "SystemOneRuntime":
        catalog = load_catalog(settings.model_catalog)
        entry = get_model(catalog, settings.model_id)
        profile = deepcopy(get_profile(catalog, settings.model_id, settings.backend))
        profile = _profile_with_installed_pin(settings, profile)
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
        runtime_metadata = (
            cls._decision2_runtime_metadata(profile, python)
            if settings.engine == "decision2"
            else (
                cls._basal_runtime_metadata(profile, python)
                if settings.engine == "basal" and profile.get("family") == "basal1.5"
                else {}
            )
        )

        port = _free_port()
        env = os.environ.copy()
        engine_command = cls._command(settings, profile, env_dir, python, port, env)
        guard = Path(__file__).with_name("sidecar_guard.py").resolve()
        descriptor, pid_file_name = tempfile.mkstemp(prefix="deqio-sidecar-", suffix=".pid")
        os.close(descriptor)
        engine_pid_file = Path(pid_file_name)
        command = [
            sys.executable,
            str(guard),
            "--parent-pid",
            str(os.getpid()),
            "--control-stdin",
            "--engine-pid-file",
            str(engine_pid_file),
            "--",
            *engine_command,
        ]
        if settings.hf_offline_runtime:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        address = f"http://127.0.0.1:{port}"
        sidecar_start(engine=settings.engine, address=address)
        env["PYTHONUNBUFFERED"] = "1"
        popen_kwargs: dict[str, Any] = {}
        if os.name != "nt":
            # The direct child is the Deqio supervisor. It gets its own process
            # group so close() can stop it reliably; the supervisor starts the
            # actual engine tree in a separate process group and tears that tree
            # down both on graceful shutdown and if the Deqio parent disappears.
            popen_kwargs["start_new_session"] = True
        try:
            process = subprocess.Popen(
                command,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **popen_kwargs,
            )
        except Exception:
            engine_pid_file.unlink(missing_ok=True)
            raise

        # From here on the supervisor is alive: every failure below must end in
        # the one supervisor cleanup path, including failures that happen before
        # the runtime object exists (log pump start, identity/registry parsing).
        log_thread = None
        try:
            log_thread = start_sidecar_log_pump(process, settings.engine)
            runtime = cls(
                settings,
                entry,
                profile,
                process,
                port,
                log_thread,
                runtime_metadata,
                engine_pid_file=engine_pid_file,
            )
            _wait_for_port(port, process, timeout=float(settings.sidecar_process_ready_seconds))
            sidecar_process_ready(engine=settings.engine, address=address)
            sidecar_model_wait(engine=settings.engine, timeout_seconds=float(settings.sidecar_startup_seconds))
            runtime._probe_model_ready(timeout=float(settings.sidecar_startup_seconds))
        except Exception:
            _stop_supervisor(
                process,
                engine=settings.engine,
                log_thread=log_thread,
                engine_pid_file=engine_pid_file,
            )
            raise
        sidecar_ready(engine=settings.engine, address=address)
        return runtime

    @staticmethod
    def _validate_accelerator(settings: Settings, python: Path) -> None:
        if settings.engine == "decision2":
            if settings.backend != "cuda":
                raise RuntimeError(
                    "Decision 2.0 requires the official CUDA runtime. "
                    "No CPU, MPS or MLX fallback will be used."
                )
            if platform.system() != "Linux":
                suffix = (
                    " This model is not available on macOS/Apple Silicon in Deqio 0.5."
                    if platform.system() == "Darwin"
                    else " The current official Deqio profile is supported on Linux CUDA hosts."
                )
                raise RuntimeError(
                    "Decision 2.0 requires the official CUDA runtime." + suffix +
                    " No CPU, MPS or MLX fallback will be used."
                )
            probe = subprocess.run(
                [
                    str(python),
                    "-c",
                    "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if probe.returncode != 0:
                raise RuntimeError(
                    "Decision 2.0 requires the official CUDA runtime, but CUDA is unavailable. "
                    "No CPU, MPS or MLX fallback will be used."
                )
            return
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
            return
        if settings.backend == "cuda":
            label = f"{settings.model_id}/cuda"
            if platform.system() != "Linux":
                raise RuntimeError(f"{label} requires a Linux host with an NVIDIA GPU. No CPU fallback will be used.")
            # B9: every CUDA engine in the catalog runs on PyTorch. Probe CUDA in
            # the profile's own runtime, so an engine that would silently run on
            # CPU is never served or verified as a CUDA profile.
            probe = subprocess.run(
                [str(python), "-c", _TORCH_CUDA_PROBE],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if probe.returncode == _TORCH_CUDA_PROBE_NO_TORCH:
                raise RuntimeError(
                    f"{label} requires CUDA, but the model runtime has no importable PyTorch; reinstall it with: "
                    f"deqio models update {settings.model_id} --backend cuda. No CPU fallback will be used."
                )
            if probe.returncode != 0:
                raise RuntimeError(
                    f"{label} requires CUDA, but CUDA is not available to PyTorch in the model runtime "
                    "(no usable NVIDIA GPU or driver). No CPU fallback will be used."
                )

    @staticmethod
    def _decision2_runtime_metadata(
        profile: dict[str, Any], python: Path
    ) -> dict[str, Any]:
        """Capture actual CUDA/runtime and pinned package-manifest provenance."""
        probe = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import json,torch; "
                    "print(json.dumps({"
                    "'torch_version': torch.__version__, "
                    "'cuda_runtime': torch.version.cuda, "
                    "'cuda_device': torch.cuda.get_device_name(0), "
                    "'cuda_compute_capability': '.'.join(map(str, torch.cuda.get_device_capability(0)))"
                    "}))"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            metadata = json.loads(probe.stdout.strip())
        except json.JSONDecodeError as error:
            raise RuntimeError("Could not read Decision 2.0 CUDA runtime metadata") from error

        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=str(profile["model"]),
                revision=str(profile["model_revision"]),
                local_files_only=True,
            )
        ).resolve()
        config_path = snapshot / "config.json"
        manifest_path = snapshot / "MODEL_MANIFEST.json"
        if not config_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(
                "Pinned Decision 2.0 package is missing config.json or MODEL_MANIFEST.json"
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
        if config.get("decision_format") != "vllm-sr-decision" or runtime.get("entry") != "decision2.Decision2.from_pretrained":
            raise RuntimeError("Pinned package is not an official Decision 2.0 System One package")
        runtime_source = manifest.get("runtime_source") if isinstance(manifest.get("runtime_source"), dict) else {}
        metadata.update({
            "runtime_version": runtime_source.get("commit"),
            "runtime_entry": runtime.get("entry"),
            "package_schema": config.get("package_schema"),
            "decision_format": config.get("decision_format"),
            "format_version": config.get("format_version"),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "manifest_model_name": manifest.get("model_name"),
            "manifest_max_input_tokens": manifest.get("max_input_tokens"),
            "official_package_verified_on_load": True,
        })
        return metadata

    @staticmethod
    def _basal_runtime_metadata(profile: dict[str, Any], python: Path) -> dict[str, Any]:
        """Capture the installed official Basal runtime version and serving profile."""
        probe = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import json; "
                    "from importlib.metadata import version; "
                    "print(json.dumps({'runtime_version': version('basal')}))"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            metadata = json.loads(probe.stdout.strip())
        except json.JSONDecodeError as error:
            raise RuntimeError("Could not read Basal runtime metadata") from error
        expected = str(profile.get("basal_runtime_version", "")).strip()
        installed = str(metadata.get("runtime_version", "")).strip()
        if expected and installed != expected:
            raise RuntimeError(
                f"Basal runtime version mismatch: expected {expected}, installed {installed or 'unknown'}"
            )
        metadata.update({
            "basal_mode": profile.get("basal_mode"),
            "basal_backend": profile.get("basal_backend"),
            "official_runtime": True,
        })
        return metadata

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
        launcher = str(profile.get("launcher", ""))

        def local_gguf(directory_key: str, pattern_key: str, default_pattern: str = "*Q8_0.gguf") -> Path:
            directory_value = profile.get(directory_key)
            if not directory_value:
                raise RuntimeError(f"GGUF profile is missing {directory_key}")
            directory = Path(str(directory_value)).expanduser()
            if not directory.is_absolute():
                directory = (settings.config_path.parent / directory).resolve()
            pattern = str(profile.get(pattern_key, default_pattern))
            matches = sorted(path for path in directory.glob(pattern) if path.is_file())
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one GGUF artifact matching {pattern!r} in {directory}; "
                    f"found {len(matches)}"
                )
            return matches[0].resolve()

        # Generic official GGUF launchers are resolved before engine-specific
        # backend restrictions.  The launcher still uses each upstream's native
        # SystemOne/decision readout; Deqio only owns transport and lifecycle.
        if launcher == "llama_cpp":
            executable = _runtime_executable(env_dir, "llama-server")
            if not executable.is_file():
                raise RuntimeError(f"Pinned llama.cpp server was not installed: {executable}")
            gguf = local_gguf("llama_cpp_gguf_dir", "llama_cpp_gguf_pattern")
            return [
                str(executable),
                "-m", str(gguf),
                "-c", str(settings.max_tokens),
                "-ngl", "99",
                "--host", "127.0.0.1",
                "--port", str(port),
                "--log-disable",
            ]

        if launcher == "decider_gguf":
            sidecar = Path(__file__).with_name("decider_gguf_sidecar.py").resolve()
            gguf = local_gguf("decider_gguf_dir", "decider_gguf_pattern")
            return [
                str(python), str(sidecar),
                "--model-dir", str(gguf.parent),
                "--gguf", str(gguf),
                "--max-tokens", str(settings.max_tokens),
                "--port", str(port),
            ]

        if launcher == "jevk5_gguf":
            executable = _runtime_executable(env_dir, "llama-server")
            if not executable.is_file():
                raise RuntimeError(f"Pinned llama.cpp server was not installed: {executable}")
            sidecar = Path(__file__).with_name("jevk5_gguf_sidecar.py").resolve()
            gguf = local_gguf("jevk5_gguf_dir", "jevk5_gguf_filename")
            return [
                str(python), str(sidecar),
                "--llama-server", str(executable),
                "--gguf", str(gguf),
                "--temperature", str(profile.get("jevk5_temperature", 1.0)),
                "--knockout-temperature", str(profile.get("jevk5_knockout_temperature", 1.0)),
                "--max-tokens", str(settings.max_tokens),
                "--port", str(port),
            ]

        if engine == "semif":
            sidecar = Path(__file__).with_name("semif_sidecar.py").resolve()
            command = [
                str(python), str(sidecar),
                "--backend", settings.backend,
                "--model", settings.model,
                "--revision", settings.model_revision,
                "--max-tokens", str(settings.max_tokens),
                "--mlx-cache-mib", str(settings.mlx_cache_mib),
            ]
            mlx_bits = profile.get("semif_mlx_bits")
            if settings.backend == "mlx" and mlx_bits is not None:
                command.extend(["--mlx-bits", str(int(mlx_bits))])
            command.extend([
                "--torch-dtype", settings.torch_dtype,
                "--port", str(port),
            ])
            return command

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

        if engine == "decision2":
            if settings.backend != "cuda":
                raise RuntimeError(
                    "Decision 2.0 requires the official CUDA runtime; no fallback is allowed"
                )
            sidecar = Path(__file__).with_name("decision2_sidecar.py").resolve()
            return [
                str(python),
                str(sidecar),
                "--model",
                model,
                "--revision",
                str(profile.get("model_revision", settings.model_revision)),
                "--port",
                str(port),
            ]

        if engine == "basal":
            if settings.backend not in {"mlx", "mps", "gguf", "cuda"}:
                raise RuntimeError("Basal profiles support mlx, mps, gguf and cuda")
            executable = _runtime_executable(env_dir, "basal-serve")
            if not executable.is_file():
                raise RuntimeError(f"Basal executable was not installed: {executable}")
            command = [str(executable), "--model", model]
            revision = profile.get("model_revision")
            if revision and str(revision) not in {"upstream-latest", "main"}:
                command.extend(["--revision", str(revision)])
            mode = profile.get("basal_mode")
            if mode:
                command.extend(["--mode", str(mode)])
            if settings.backend == "gguf":
                gguf_value = profile.get("basal_gguf")
                if gguf_value:
                    gguf_path = Path(str(gguf_value)).expanduser()
                    if not gguf_path.is_absolute():
                        gguf_path = (settings.config_path.parent / gguf_path).resolve()
                else:
                    gguf_dir_value = profile.get("basal_gguf_dir")
                    gguf_pattern = str(profile.get("basal_gguf_pattern", "*Q8_0.gguf"))
                    if not gguf_dir_value:
                        raise RuntimeError("Basal GGUF profile is missing its artifact directory")
                    gguf_dir = Path(str(gguf_dir_value)).expanduser()
                    if not gguf_dir.is_absolute():
                        gguf_dir = (settings.config_path.parent / gguf_dir).resolve()
                    matches = sorted(path for path in gguf_dir.glob(gguf_pattern) if path.is_file())
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"Expected exactly one Basal GGUF artifact matching {gguf_pattern!r} in {gguf_dir}; "
                            f"found {len(matches)}"
                        )
                    gguf_path = matches[0].resolve()
                if not gguf_path.is_file():
                    raise RuntimeError(f"Basal GGUF weights are missing: {gguf_path}")
                command.extend(["--gguf", str(gguf_path)])
            if profile.get("basal_soam") is True:
                command.extend(["--soam", "on"])
            command.extend(["--port", str(port)])
            return command


        if engine == "clef":
            if settings.backend != "mlx":
                raise RuntimeError("Clef's current Deqio profiles require MLX")
            sidecar = Path(__file__).with_name("clef_sidecar.py").resolve()
            max_length = int(
                getattr(settings, "max_tokens", profile.get("max_input_tokens", 16384))
            )
            return [
                str(python), str(sidecar),
                "--model", model,
                "--revision", str(profile.get("model_revision", "upstream-latest")),
                "--name", str(profile.get("wire_model", model)),
                "--max-length", str(max_length),
                "--port", str(port),
            ]

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
        # llama-server intentionally binds its HTTP port before the model is
        # loaded and reports HTTP 503 {"message": "Loading model"} during
        # that window. Treat 503 as a transient readiness state only here;
        # normal inference requests still surface every HTTP error immediately.
        deadline = time.monotonic() + timeout
        last_unavailable: SystemOneSidecarHTTPError | None = None
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Engine sidecar exited during model startup with code {self.process.returncode}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = f" Last response: {last_unavailable}" if last_unavailable else ""
                raise RuntimeError(
                    f"Timed out waiting {timeout:.1f}s for engine model readiness.{detail}"
                )
            try:
                response = _post_json(
                    f"{self.base_url}/v1/systemone",
                    payload,
                    timeout=remaining,
                )
            except SystemOneSidecarHTTPError as error:
                if error.status_code != 503:
                    raise
                last_unavailable = error
                time.sleep(min(0.2, max(0.0, remaining)))
                continue
            answers = response.get("answers")
            if not isinstance(answers, dict) or not isinstance(answers.get("ready"), dict):
                raise RuntimeError(
                    "Engine sidecar opened its port but did not pass the model-readiness probe"
                )
            return

    def _request(
        self,
        state: Any,
        questions: dict[str, Any],
        execution_mode: str | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], float]:
        payload = {
            "model": str(self.profile.get("wire_model", self.settings.model)),
            "state": state,
            "questions": questions,
        }
        if execution_mode:
            payload["execution_mode"] = execution_mode
        if request_options:
            reserved = {"model", "state", "questions", "execution_mode"}
            overlap = reserved.intersection(request_options)
            if overlap:
                raise RuntimeError("System One request options may not override: " + ", ".join(sorted(overlap)))
            payload.update(request_options)
        body = _json_body(payload)
        started = time.perf_counter()
        response = _post_json(
            f"{self.base_url}/v1/systemone", payload, timeout=180.0, body=body
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency_ms = _protocol_float(
            response.get("latency_ms", elapsed_ms) or elapsed_ms,
            field="latency_ms",
            minimum=0.0,
        )
        return payload, response, latency_ms

    def system_one(
        self,
        state: Any,
        questions: dict[str, Any],
        request_options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one native System One request without Deqio-side score rewriting."""
        if not isinstance(questions, dict) or not questions:
            raise RuntimeError("questions must be a non-empty object")
        request_options = dict(request_options or {})
        capabilities = self.profile.get("capabilities")
        capabilities = capabilities if isinstance(capabilities, dict) else {}

        if self.engine == "decision2":
            if request_options:
                unsupported = ", ".join(sorted(request_options))
                raise SystemOneCapabilityError(
                    unsupported,
                    f"Decision 2.0 does not support System One request option(s): {unsupported}",
                )
            if len(questions) > 1 and not capabilities.get("multi_question", False):
                raise SystemOneCapabilityError(
                    "multi_question", "Active profile does not support multiple System One questions"
                )
            for question_id, question in questions.items():
                if not isinstance(question, dict):
                    raise RuntimeError(f"Question {question_id!r} must be an object")
                question_type = str(question.get("type", ""))
                if question_type not in {"choice", "noul", "score"}:
                    raise SystemOneCapabilityError(
                        question_type,
                        f"Decision 2.0 does not support question type {question_type!r}; "
                        "supported types are choice, noul and score",
                    )
                if not capabilities.get(question_type, False):
                    raise SystemOneCapabilityError(
                        question_type, f"Active profile does not support System One capability {question_type!r}"
                    )
            _payload, response, latency_ms = self._request(state, questions)

        elif self.engine == "basal":
            if self.profile.get("family") != "basal1.5":
                raise RuntimeError("Native /v1/systemone exposure requires a Basal 1.5 profile")
            if len(questions) > 1 and not capabilities.get("soam", False):
                raise SystemOneCapabilityError(
                    "soam", "Active Basal profile does not support State Once, Ask Many"
                )

            facts = request_options.get("facts")
            if facts is not None:
                if facts not in {"off", "auto"}:
                    raise RuntimeError('facts must be "off" or "auto"')
                if facts == "auto" and not capabilities.get("facts", False):
                    raise SystemOneCapabilityError(
                        "facts", "Active Basal backend does not support facts: auto"
                    )
            allowed_types = {"choice", "noul", "score", "multi", "act"}
            for question_id, question in questions.items():
                if not isinstance(question, dict):
                    raise RuntimeError(f"Question {question_id!r} must be an object")
                question_type = str(question.get("type", "choice"))
                if question_type not in allowed_types:
                    raise SystemOneCapabilityError(
                        question_type, f"Basal 1.5 does not support question type {question_type!r}"
                    )
                if not capabilities.get(question_type, False):
                    raise SystemOneCapabilityError(
                        question_type,
                        f"Active Basal backend does not support System One capability {question_type!r}",
                    )
                if "option_keys" in question and not capabilities.get("option_keys", False):
                    raise SystemOneCapabilityError(
                        "option_keys", "Active Basal backend does not support option_keys"
                    )
                if question.get("evidence"):
                    if question_type == "multi":
                        raise SystemOneCapabilityError(
                            "evidence", "Basal evidence is not supported for multi questions"
                        )
                    if not capabilities.get("evidence", False):
                        raise SystemOneCapabilityError(
                            "evidence",
                            f"Evidence is not available on the active Basal {self.settings.backend} backend",
                        )

            _payload, response, latency_ms = self._request(
                state, questions, request_options=request_options
            )
        elif capabilities.get("systemone", False):
            if request_options:
                unsupported = ", ".join(sorted(request_options))
                raise SystemOneCapabilityError(
                    unsupported,
                    f"Active profile does not support System One request option(s): {unsupported}",
                )
            if len(questions) > 1 and not capabilities.get("multi_question", False):
                raise SystemOneCapabilityError(
                    "multi_question", "Active profile does not support multiple System One questions"
                )
            for question_id, question in questions.items():
                if not isinstance(question, dict):
                    raise RuntimeError(f"Question {question_id!r} must be an object")
                question_type = str(question.get("type", "choice"))
                if not capabilities.get(question_type, False):
                    raise SystemOneCapabilityError(
                        question_type,
                        f"Active profile does not support System One capability {question_type!r}",
                    )
                if "option_keys" in question and not capabilities.get("option_keys", False):
                    raise SystemOneCapabilityError(
                        "option_keys", "Active profile does not support option_keys"
                    )
                if question.get("evidence") and not capabilities.get("evidence", False):
                    raise SystemOneCapabilityError(
                        "evidence", "Evidence is not available on the active profile"
                    )
            _payload, response, latency_ms = self._request(state, questions)

        else:
            raise SystemOneCapabilityError(
                "systemone", "Active profile does not expose native System One"
            )

        return response, {
            "total_seconds": latency_ms / 1000.0,
            "batch_size": len(questions),
            "engine": self.engine,
        }

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
            raise SystemOneProtocolError("External engine response is missing answers.decision")
        return _choice_raw(
            row=row,
            answer=answers[question_id],
            response=response,
            payload=payload,
            latency_ms=latency_ms,
            preserve_native=self.engine == "decision2",
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
            raise SystemOneProtocolError("External engine response is missing answers.decision")
        return _noul_raw(
            row=row,
            answer=answers[question_id],
            response=response,
            payload=payload,
            latency_ms=latency_ms,
            preserve_native=self.engine == "decision2",
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
            raise SystemOneProtocolError("External engine response is missing answers")
        raw_results = []
        for row, qid in zip(rows, qids):
            answer = answers.get(qid)
            if not isinstance(answer, dict):
                raise SystemOneProtocolError(f"External engine response is missing answers.{qid}")
            raw_results.append(
                _choice_raw(
                    row=row,
                    answer=answer,
                    response=response,
                    payload=payload,
                    latency_ms=latency_ms,
                    preserve_native=self.engine == "decision2",
                )
            )
        # One engine call measured the whole batch. Report that count once as
        # batch usage; a per-decision count is unknown unless the batch is a
        # single decision (AGENTS.md: never copy an aggregate per decision).
        # ``engine_reported_batch`` says why: the engine measured the batch.
        batch_tokens, batch_source = _input_token_usage(response)
        for raw in raw_results:
            raw["batch_input_tokens"] = batch_tokens
            raw["batch_input_tokens_source"] = batch_source
            if len(raw_results) > 1:
                raw["input_tokens"] = None
                raw["input_tokens_source"] = (
                    "engine_reported_batch" if batch_source == "engine_reported" else "unknown"
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
        _stop_supervisor(
            self.process,
            engine=self.engine,
            log_thread=self._log_thread,
            engine_pid_file=self._engine_pid_file,
        )


def _stop_supervisor(
    process: subprocess.Popen[Any],
    *,
    engine: str,
    log_thread: Any = None,
    engine_pid_file: Path | None = None,
) -> None:
    """Stop one sidecar supervisor and everything it owns.

    This is the single cleanup path for a spawned supervisor: ``close()`` and
    the failure path of ``load()`` both end here, so a runtime that was never
    fully constructed is torn down exactly like a healthy one.
    """
    pid_value = getattr(process, "pid", None)
    pid = int(pid_value) if isinstance(pid_value, int) else None
    sidecar_stop(engine=engine, pid=pid)

    control_pipe = getattr(process, "stdin", None)
    try:
        if process.poll() is None:
            # Ask the supervisor to stop the engine tree itself first. This
            # is the portable graceful path (notably on Windows, where
            # Popen.terminate() is a hard TerminateProcess call and would
            # skip the supervisor's child cleanup).
            if control_pipe is not None:
                try:
                    control_pipe.write("stop\n")
                    control_pipe.flush()
                except (BrokenPipeError, OSError, ValueError):
                    pass

            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

        if process.poll() is None:
            terminated_group = False
            if os.name != "nt" and pid is not None:
                try:
                    os.killpg(pid, signal.SIGTERM)
                    terminated_group = True
                except ProcessLookupError:
                    pass
                except OSError:
                    # Fall back to the direct child when the process was not
                    # started as a session leader for any reason.
                    terminated_group = False
            if not terminated_group and process.poll() is None:
                process.terminate()

            # The supervisor's own escalation (SIGTERM, 4 s, SIGKILL, 2 s, a
            # final 2 s group wait) takes up to ~8.2 s on a stubborn engine
            # tree. Give it that time before taking the tree away from it.
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                killed_group = False
                if os.name != "nt" and pid is not None:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                        killed_group = True
                    except ProcessLookupError:
                        pass
                    except OSError:
                        killed_group = False
                if not killed_group and process.poll() is None:
                    process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError(
                        f"Inference sidecar supervisor pid={pid} did not stop after forced termination"
                    ) from error
    finally:
        if control_pipe is not None:
            try:
                control_pipe.close()
            except (OSError, ValueError):
                pass
        _reap_orphaned_engine_tree(process, engine=engine, engine_pid_file=engine_pid_file)

    if log_thread is not None:
        log_thread.join(timeout=1)
    sidecar_stopped(engine=engine, pid=pid)


def _reap_orphaned_engine_tree(
    process: subprocess.Popen[Any],
    *,
    engine: str,
    engine_pid_file: Path | None,
) -> None:
    """Last-resort cleanup of the engine session when the supervisor was killed.

    The supervisor runs the engine in its own session so it can terminate the
    whole engine tree without killing itself. That also means the Deqio parent
    cannot reach the engine through the supervisor's process group. When the
    supervisor died from a signal (OOM killer, an external kill, or the forced
    escalation above) its own cleanup never ran, so the engine tree is reaped
    here using the PID the supervisor published at start.
    """
    if engine_pid_file is None:
        return
    returncode = getattr(process, "returncode", None)
    try:
        if os.name != "nt" and isinstance(returncode, int) and returncode < 0:
            try:
                engine_pid = int(engine_pid_file.read_text(encoding="utf-8").strip() or "0")
            except (OSError, ValueError):
                engine_pid = 0
            if engine_pid > 0 and _is_engine_session_leader(engine_pid) and pgid_alive(engine_pid):
                info(
                    f"Sidecar supervisor for engine={engine} was killed before its cleanup ran; "
                    f"terminating the orphaned engine process group pgid={engine_pid}"
                )
                _kill_process_group(engine_pid)
    finally:
        # Keep the file while the supervisor is still running: a later retry
        # of close() must still be able to find the engine tree.
        if returncode is not None:
            try:
                engine_pid_file.unlink()
            except OSError:
                pass


def _is_engine_session_leader(pid: int) -> bool:
    """Guard against PID reuse before signalling a published engine PID.

    The supervisor starts the engine with ``start_new_session=True``, so the
    engine PID is also its session ID. An unrelated process that inherited
    the number after a PID wrap is almost never a session leader.
    """
    try:
        return os.getsid(pid) == pid
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _kill_process_group(pgid: int) -> None:
    deadline = time.monotonic() + 4.0
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        return
    while time.monotonic() < deadline:
        if not pgid_alive(pgid):
            return
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        return
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and pgid_alive(pgid):
        time.sleep(0.05)
