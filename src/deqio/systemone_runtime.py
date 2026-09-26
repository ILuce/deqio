from __future__ import annotations

import hashlib
import json
import os
import socket
import platform
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .catalog import get_model, get_profile, load_catalog
from .config import Settings
from .console import sidecar_ready, sidecar_start, start_sidecar_log_pump


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
    raise RuntimeError(f"Timed out waiting for engine sidecar on port {port}")


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers={"content-type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"System One sidecar returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"System One sidecar is unavailable: {error}") from error
    if not isinstance(result, dict):
        raise RuntimeError("System One sidecar returned a non-object response")
    return result


def _sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalise_probabilities(answer: dict[str, Any], option_ids: list[str]) -> list[float]:
    values = answer.get("probabilities")
    if isinstance(values, dict):
        probs = [float(values.get(option_id, 0.0)) for option_id in option_ids]
        total = sum(probs)
        if total > 0:
            return [value / total for value in probs]
    choice = answer.get("choice")
    if choice in option_ids:
        return [1.0 if option_id == choice else 0.0 for option_id in option_ids]
    raise RuntimeError("External engine did not return choice probabilities")


def _choice_raw(
    *, row: dict[str, Any], answer: dict[str, Any], response: dict[str, Any], payload: dict[str, Any], latency_ms: float
) -> dict[str, Any]:
    option_ids = [str(option["id"]) for option in row["options"]]
    probabilities = _normalise_probabilities(answer, option_ids)
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return {
        "id": row["id"],
        "option_ids": option_ids,
        "probabilities": probabilities,
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "total_seconds": latency_ms / 1000.0,
        "prompt_sha256": _sha(payload),
        "probability_status": "probabilities reported by the selected System One engine; raw option logits unavailable",
    }


def _noul_raw(
    *, row: dict[str, Any], answer: dict[str, Any], response: dict[str, Any], payload: dict[str, Any], latency_ms: float
) -> dict[str, Any]:
    p_yes = float(answer.get("noul"))
    p_yes = max(0.0, min(1.0, p_yes))
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return {
        "id": row["id"],
        "option_ids": ["yes", "no"],
        "probabilities": [p_yes, 1.0 - p_yes],
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "total_seconds": latency_ms / 1000.0,
        "prompt_sha256": _sha(payload),
        "probability_status": "native Noul probability reported by the selected System One engine",
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
                f"uv run deqio models install {settings.model_id} --backend {settings.backend}"
            )

        cls._validate_accelerator(settings, python)

        port = _free_port()
        env = os.environ.copy()
        command = cls._command(settings, profile, env_dir, python, port, env)
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
        try:
            _wait_for_port(port, process, timeout=float(settings.sidecar_startup_seconds))
        except Exception:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if log_thread is not None:
                log_thread.join(timeout=1)
            raise
        sidecar_ready(engine=settings.engine, address=address)
        return cls(settings, entry, profile, process, port, log_thread)

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
            env["LAYA_PRELOAD"] = "1"
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
                    f"Run: uv run deqio models install {settings.model_id} --backend {settings.backend}"
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
        started = time.perf_counter()
        response = _post_json(f"{self.base_url}/v1/systemone", payload)
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
