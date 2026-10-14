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
        # 0.6.1: a `remote` profile is served over HTTPS by the provider named in the catalog;
        # it is an installed profile like any other and never a fallback for a sidecar.
        if settings.backend == "remote":
            from .remote_runtime import RemoteProviderRuntime

            return RemoteProviderRuntime.load(settings)  # type: ignore[return-value]
        from .systemone_runtime import SystemOneRuntime

        return SystemOneRuntime.load(settings)  # type: ignore[return-value]
