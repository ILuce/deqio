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
        from .providers import LOCAL_PROVIDER, provider_spec

        # D11: a remote decision provider answers without a sidecar; it is selected
        # explicitly (config.json `provider` / DEQIO_PROVIDER) and never used as a fallback.
        if provider_spec(getattr(settings, "provider", LOCAL_PROVIDER)).kind == "remote":
            from .remote_runtime import RemoteProviderRuntime

            return RemoteProviderRuntime.load(settings)  # type: ignore[return-value]
        from .systemone_runtime import SystemOneRuntime

        return SystemOneRuntime.load(settings)  # type: ignore[return-value]
