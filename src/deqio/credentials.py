"""API keys of remote decision profiles live in the operating-system keychain (0.6.1).

``deqio models setup`` stores the key a user types for a ``remote`` profile with ``keyring``
(macOS Keychain, Windows Credential Manager, Linux Secret Service / KWallet) under the service
``deqio`` and the provider name as the account.  Nothing under the workspace (``config.json``,
``.deqio/``) ever holds the key, and the key is only read when a Deqio process loads that
profile.  A host without a keychain backend is an explicit error: there is no file or
environment fallback.
"""

from __future__ import annotations

from typing import Any

SERVICE = "deqio"


class KeychainUnavailableError(RuntimeError):
    """No OS keychain backend can store or read Deqio's API keys on this host."""


def _keyring() -> Any:
    """The ``keyring`` module; patched by tests with an in-memory stand-in."""
    import keyring

    return keyring


def _unavailable(action: str, error: Exception) -> KeychainUnavailableError:
    return KeychainUnavailableError(
        f"Could not {action} the API key in the OS keychain: {error}. Deqio stores remote API keys only in the "
        "system keychain (macOS Keychain, Windows Credential Manager, Linux Secret Service/KWallet); install or "
        "unlock one and run deqio models setup again."
    )


def keychain_backend_name() -> str:
    try:
        backend = _keyring().get_keyring()
    except Exception as error:  # keyring.errors.NoKeyringError and friends
        raise _unavailable("find", error) from error
    return str(getattr(backend, "name", backend.__class__.__name__))


def store_api_key(provider: str, api_key: str) -> None:
    key = str(api_key or "")
    if not key.strip():
        raise ValueError("an API key must not be empty")
    try:
        _keyring().set_password(SERVICE, str(provider), key)
    except Exception as error:
        raise _unavailable("store", error) from error


def load_api_key(provider: str) -> str | None:
    try:
        value = _keyring().get_password(SERVICE, str(provider))
    except Exception as error:
        raise _unavailable("read", error) from error
    return str(value) if value else None


def delete_api_key(provider: str) -> bool:
    """Remove the stored key; ``False`` when no key was stored."""
    keyring = _keyring()
    try:
        if keyring.get_password(SERVICE, str(provider)) is None:
            return False
        keyring.delete_password(SERVICE, str(provider))
    except Exception as error:
        raise _unavailable("delete", error) from error
    return True
