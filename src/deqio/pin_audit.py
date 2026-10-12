"""Compare an installed Deqio workspace with the pins of its model catalog (D12b).

Run it in the workspace (the directory that holds ``config.json``)::

    uv run python -m deqio.pin_audit > pin_audit.txt

The audit is read-only. It prints tab-separated sections: engine runtime packages
installed under the runtime directory against the catalog pins, the Nimble source
checkout against ``nimble_source_revision``, every Hugging Face artifact whose
catalog revision is not a commit SHA (with the revision the local install recorded
and, online, the revision the Hub serves now), and the Nimble Hub commits.
``--offline`` skips the Hub and ``--all-artifacts`` also lists pinned artifacts.
The exit status is 1 when a pin differs, a pinned package is missing from an
installed runtime, or a Hub lookup fails.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from . import __version__
from .catalog import load_catalog
from .installations import installation_record
from .model_manager import (
    _catalog_path,
    _config_path,
    _declared_downloads,
    _git_head,
    _looks_like_hf_repo,
    _read_config,
    _runtime_root,
)

SHA40 = re.compile(r"^[0-9a-f]{40}$")
_NAME = r"(?P<name>[A-Za-z0-9][A-Za-z0-9_.\-]*)(?:\[[^\]]*\])?"
_PINS = (
    re.compile(rf"^{_NAME} @ git\+\S+@(?P<ref>[0-9a-f]{{40}})$"),
    re.compile(rf"^{_NAME} @ \S+/archive/(?P<ref>[0-9a-f]{{40}})\.tar\.gz$"),
    re.compile(rf"^{_NAME}==(?P<ref>\S+)$"),
)
_ARCHIVE_URL = re.compile(r"/archive/([0-9a-f]{40})\.tar\.gz$")


def _dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def hf_artifacts(profile: dict[str, Any]) -> list[tuple[str, Any]]:
    """(repo_id, catalog revision) of every Hub artifact a profile installs."""
    downloads = _declared_downloads(profile)
    rows = [(str(item["repo_id"]), item.get("revision") or profile.get("model_revision")) for item in downloads]
    if profile.get("installer") == "nimble":
        rows.append((str(profile["repo_id"]), profile.get("model_revision")))
    elif not downloads and _looks_like_hf_repo(profile.get("model")):
        rows.append((str(profile["model"]), profile.get("model_revision")))
    return rows


def _catalog_pins(catalog: dict[str, Any]) -> dict[str, dict[str, str]]:
    """runtime_key -> {distribution: pinned version or commit} over every profile of that runtime."""
    pins: dict[str, dict[str, str]] = {}
    for model in catalog.get("models", []):
        for profile in (model.get("backends") or {}).values():
            runtime = str(profile.get("runtime_key") or "")
            specs = [str(item) for item in profile.get("packages") or []]
            specs += [str(value) for key, value in profile.items() if key.endswith("_package")]
            for spec in specs:
                match = next((found for found in (pattern.match(spec) for pattern in _PINS) if found), None)
                if match:
                    pins.setdefault(runtime, {})[_dist(match["name"])] = match["ref"]
                else:  # an unpinned spec can never match what is installed
                    pins.setdefault(runtime, {})[_dist(re.split(r"[\[ =<>!~@;]", spec, maxsplit=1)[0])] = spec
    return pins


def _installed(runtime_dir: Path) -> dict[str, str]:
    """Distribution -> installed commit (git/archive installs) or version, from the runtime's dist-info."""
    found: dict[str, str] = {}
    infos = sorted(runtime_dir.glob("lib/python*/site-packages/*.dist-info"))
    for info in infos:
        name, _, version = info.name[: -len(".dist-info")].rpartition("-")
        ref = version
        direct = info / "direct_url.json"
        if direct.is_file():
            try:
                data = json.loads(direct.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            commit = (data.get("vcs_info") or {}).get("commit_id")
            archive = _ARCHIVE_URL.search(str(data.get("url", "")))
            ref = str(commit) if commit else archive.group(1) if archive else str(data.get("url") or version)
        found[_dist(name)] = ref
    return found


def _error(error: Exception) -> str:
    text = str(error).splitlines()[0][:120] if str(error) else ""
    return f"ERROR {type(error).__name__}: {text}".rstrip(": ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deqio.pin_audit", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="workspace config.json (default: DEQIO_CONFIG or ./config.json)")
    parser.add_argument("--offline", action="store_true", help="skip the Hugging Face lookups")
    parser.add_argument("--all-artifacts", action="store_true", help="also list artifacts pinned to a commit")
    args = parser.parse_args(argv)

    config_path = _config_path(args.config)
    if not config_path.is_file():
        print(f"pin audit: no config at {config_path}; run it in the Deqio workspace or pass --config", file=sys.stderr)
        return 2
    data = _read_config(config_path)
    catalog_path = _catalog_path(config_path, data)
    catalog = load_catalog(catalog_path)
    runtime_root = _runtime_root(config_path, data)
    counts = {"DIFF": 0, "MISSING": 0, "ERROR": 0}
    hub = "offline" if args.offline else "online"
    print(f"# deqio pin audit {__version__}\tcatalog={catalog_path}\truntime_root={runtime_root}\thub={hub}")

    print("# runtime packages: runtime\tpackage\tinstalled\tcatalog\tstatus")
    absent = []
    for runtime, pins in sorted(_catalog_pins(catalog).items()):
        if not (runtime_root / runtime).is_dir():
            absent.append(runtime)
            continue
        installed = _installed(runtime_root / runtime)
        for dist, ref in sorted(pins.items()):
            have = installed.get(dist)
            status = "MISSING" if have is None else "OK" if have == ref else "DIFF"
            counts[status] = counts.get(status, 0) + (status != "OK")
            print(f"{runtime}\t{dist}\t{have or '-'}\t{ref}\t{status}")
    print(f"# runtimes not installed: {', '.join(absent) or 'none'}")

    print("# nimble source: checkout\tinstalled\tcatalog\tstatus")
    profiles = [
        (str(model.get("id")), str(backend), profile)
        for model in catalog.get("models", [])
        for backend, profile in (model.get("backends") or {}).items()
    ]
    for key in sorted({str(p.get("source_key", "nimble-src")) for _, _, p in profiles if p.get("installer") == "nimble"}):
        expected = next(str(p.get("nimble_source_revision", "")) for _, _, p in profiles
                        if p.get("installer") == "nimble" and str(p.get("source_key", "nimble-src")) == key)
        source = runtime_root / key
        head = _git_head(source) if (source / ".git").exists() else None
        if head is None:
            print(f"{key}\t-\t{expected}\tnot installed")
            continue
        status = "OK" if head == expected else "DIFF"
        counts["DIFF"] += status == "DIFF"
        print(f"{key}\t{head}\t{expected}\t{status}")

    print("# hub artifacts: profile\trepo\tcatalog_revision\tinstalled_revision\thub_revision")
    api = None if args.offline else HfApi()
    answers: dict[tuple[str, str | None], str] = {}
    for model_id, backend, profile in profiles:
        record: dict[str, Any] | None = None
        for repo, revision in hf_artifacts(profile):
            if isinstance(revision, str) and SHA40.match(revision) and not args.all_artifacts:
                continue
            if record is None:
                record = installation_record(config_path, model_id, backend) or {}
            artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), list) else []
            installed = next((str(item["resolved_revision"]) for item in artifacts if isinstance(item, dict)
                              and item.get("repo_id") == repo and item.get("resolved_revision")), "-")
            answer = "offline"
            if api is not None:
                ask = None if revision in (None, "", "upstream-latest") else str(revision)
                if (repo, ask) not in answers:
                    try:
                        answers[(repo, ask)] = str(api.model_info(repo, revision=ask).sha)
                    except Exception as error:  # report every failure; the audit itself must finish
                        answers[(repo, ask)] = _error(error)
                answer = answers[(repo, ask)]
                counts["ERROR"] += answer.startswith("ERROR")
            shown = "-" if revision in (None, "") else str(revision)
            print(f"{model_id}::{backend}\t{repo}\t{shown}\t{installed}\t{answer}")

    print("# nimble hub commits: repo\tcommit\tcreated_at\ttitle")
    if api is None:
        print("# skipped (offline)")
    else:
        for repo in sorted({str(p["repo_id"]) for _, _, p in profiles if p.get("installer") == "nimble"}):
            try:
                commits = list(api.list_repo_commits(repo))[:8]
            except Exception as error:  # same contract as the artifact lookups
                counts["ERROR"] += 1
                print(f"{repo}\t{_error(error)}")
                continue
            for commit in commits:
                created = getattr(commit, "created_at", "")
                created_text = created.isoformat() if hasattr(created, "isoformat") else str(created)
                print(f"{repo}\t{commit.commit_id}\t{created_text}\t{getattr(commit, 'title', '')}")

    print(f"# result: {counts['DIFF']} DIFF, {counts['MISSING']} MISSING, {counts['ERROR']} ERROR")
    return 1 if any(counts.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
