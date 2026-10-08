"""Test-session isolation.

``deqio.server`` resolves its configuration at import time from the current
working directory. Without this hook, running ``pytest`` inside a developer
workspace would reset that workspace's live Watch session and append test
traffic to its ``logs/requests.jsonl``. Point the suite at a disposable
workspace materialized from the packaged templates instead.

Tests that need a specific configuration keep creating their own ``tmp_path``
workspaces and monkeypatching ``server.SETTINGS`` as before.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

if "DEQIO_CONFIG" not in os.environ:
    from deqio.workspace import ensure_workspace

    _workspace = Path(tempfile.mkdtemp(prefix="deqio-tests-")).resolve()
    os.environ["DEQIO_CONFIG"] = str(ensure_workspace(_workspace))
    atexit.register(shutil.rmtree, _workspace, True)
