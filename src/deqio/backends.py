from __future__ import annotations

from .config import Settings


class BackendRuntime:
    """Server-facing runtime factory.

    All supported engines run in isolated SystemOne sidecars.  Keeping this
    tiny facade preserves the existing import surface for the server, benchmark
    runner, and model manager without retaining the obsolete in-process model
    implementation.
    """

    @classmethod
    def load(cls, settings: Settings) -> "BackendRuntime":
        from .systemone_runtime import SystemOneRuntime

        return SystemOneRuntime.load(settings)  # type: ignore[return-value]
