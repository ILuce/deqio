from __future__ import annotations

import shutil
from importlib.resources import files
from pathlib import Path


DEFAULT_CONFIG_NAME = "config.json"
DEFAULT_CATALOG_NAME = "models.json"
DEFAULT_BENCHMARK = Path("benchmarks/basic.json")


def _resource(relative: str):
    node = files("deqio.data")
    for part in Path(relative).parts:
        node = node.joinpath(part)
    return node


def _copy_resource(relative: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _resource(relative).open("rb") as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def ensure_workspace(root: Path | None = None) -> Path:
    """Create the small editable Deqio workspace files when they are absent.

    PyPI/tool installations do not have a repository checkout beside them. The
    package therefore ships pristine templates and materializes them in the
    current working directory on first command that needs configuration.

    Existing files are never overwritten. This keeps repository checkouts and
    user-edited benchmark/catalog files fully compatible with the historical
    Deqio layout.
    """
    base = (root or Path.cwd()).expanduser().resolve()
    targets = {
        "config.json": base / DEFAULT_CONFIG_NAME,
        "models.json": base / DEFAULT_CATALOG_NAME,
        "benchmarks/basic.json": base / DEFAULT_BENCHMARK,
    }
    for resource_name, destination in targets.items():
        if not destination.exists():
            _copy_resource(resource_name, destination)

    (base / "models").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    return targets["config.json"]
