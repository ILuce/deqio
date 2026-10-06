from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

_GIB = 1024 ** 3


@dataclass(frozen=True)
class HostCapabilities:
    system: str
    machine: str
    backends: tuple[str, ...]
    system_memory_gib: float | None
    cuda_memory_gib: float | None
    cuda_compute_capability: float | None = None

    def memory_for_backend(self, backend: str) -> float | None:
        if backend in {"mlx", "mps", "gguf"}:
            if self.system_memory_gib is None:
                return None
            # Apple Silicon uses unified memory. Keep a conservative OS/runtime
            # reserve so a 16 GiB Mac is not treated as if all 16 GiB were
            # available to model weights and serving buffers.
            return max(0.0, self.system_memory_gib - 4.0)
        if backend == "cuda":
            if self.cuda_memory_gib is None:
                return None
            return max(0.0, self.cuda_memory_gib - 1.0)
        return None


def host_backends() -> tuple[str, ...]:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine == "arm64":
        return ("mlx", "mps", "gguf")
    if system in {"Linux", "Windows"}:
        return ("cuda", "gguf") if shutil.which("nvidia-smi") else ("gguf",)
    return ()


def _system_memory_gib() -> float | None:
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
            )
            return int(result.stdout.strip()) / _GIB
        except (OSError, ValueError, subprocess.CalledProcessError):
            return None

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / _GIB
    except (AttributeError, OSError, ValueError):
        return None


def _cuda_memory_gib() -> float | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    values: list[float] = []
    for line in result.stdout.splitlines():
        try:
            values.append(float(line.strip()) / 1024.0)
        except ValueError:
            continue
    return max(values) if values else None


def _cuda_compute_capability() -> float | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    values: list[float] = []
    for line in result.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return max(values) if values else None


def detect_host() -> HostCapabilities:
    return HostCapabilities(
        system=platform.system(),
        machine=platform.machine().lower(),
        backends=host_backends(),
        system_memory_gib=_system_memory_gib(),
        cuda_memory_gib=_cuda_memory_gib(),
        cuda_compute_capability=_cuda_compute_capability(),
    )


def profile_compatibility(
    backend: str,
    profile: dict[str, Any],
    *,
    host: HostCapabilities | None = None,
) -> dict[str, Any]:
    host = host or detect_host()
    systems = profile.get("systems")
    if isinstance(systems, list) and systems and host.system not in {str(item) for item in systems}:
        supported = ", ".join(str(item) for item in systems)
        return {
            "compatible": False,
            "reason": f"profile supports operating systems: {supported}; host is {host.system}",
            "available_memory_gib": host.memory_for_backend(backend),
            "minimum_memory_gib": profile.get("min_memory_gib"),
            "recommended_memory_gib": profile.get("recommended_memory_gib"),
        }
    if backend not in host.backends:
        return {
            "compatible": False,
            "reason": f"backend {backend} is not available on this host",
            "available_memory_gib": host.memory_for_backend(backend),
            "minimum_memory_gib": profile.get("min_memory_gib"),
            "recommended_memory_gib": profile.get("recommended_memory_gib"),
        }

    if backend == "cuda" and profile.get("min_cuda_compute_capability") is not None:
        required = float(profile["min_cuda_compute_capability"])
        detected = host.cuda_compute_capability
        if detected is None:
            return {
                "compatible": False,
                "reason": (
                    f"requires CUDA compute capability >= {required:.1f}; "
                    "host capability could not be detected"
                ),
                "available_memory_gib": host.memory_for_backend(backend),
                "minimum_memory_gib": profile.get("min_memory_gib"),
                "recommended_memory_gib": profile.get("recommended_memory_gib"),
            }
        if detected < required:
            return {
                "compatible": False,
                "reason": (
                    f"requires CUDA compute capability >= {required:.1f}; "
                    f"host reports {detected:.1f}"
                ),
                "available_memory_gib": host.memory_for_backend(backend),
                "minimum_memory_gib": profile.get("min_memory_gib"),
                "recommended_memory_gib": profile.get("recommended_memory_gib"),
            }

    available = host.memory_for_backend(backend)
    minimum_raw = profile.get("min_memory_gib")
    recommended_raw = profile.get("recommended_memory_gib")
    minimum = float(minimum_raw) if minimum_raw is not None else None
    recommended = float(recommended_raw) if recommended_raw is not None else None

    if minimum is not None and available is not None and available < minimum:
        return {
            "compatible": False,
            "reason": (
                f"needs about {minimum:.1f} GiB model memory; "
                f"host has about {available:.1f} GiB usable for {backend}"
            ),
            "available_memory_gib": available,
            "minimum_memory_gib": minimum,
            "recommended_memory_gib": recommended,
        }

    warning = None
    if recommended is not None and available is not None and available < recommended:
        warning = (
            f"compatible but below the recommended {recommended:.1f} GiB "
            f"(about {available:.1f} GiB usable)"
        )

    return {
        "compatible": True,
        "reason": warning or "compatible",
        "warning": warning,
        "available_memory_gib": available,
        "minimum_memory_gib": minimum,
        "recommended_memory_gib": recommended,
    }


def describe_host(host: HostCapabilities | None = None) -> str:
    host = host or detect_host()
    if host.system == "Darwin" and host.machine == "arm64":
        memory = "unknown" if host.system_memory_gib is None else f"{host.system_memory_gib:.1f} GiB unified memory"
        return f"Apple Silicon macOS, {memory}"
    if "cuda" in host.backends:
        memory = "unknown" if host.cuda_memory_gib is None else f"{host.cuda_memory_gib:.1f} GiB GPU memory"
        return f"{host.system} {host.machine}, {memory}"
    return f"{host.system} {host.machine}"
