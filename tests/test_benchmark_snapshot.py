"""0.6.2 freeze: the README "Benchmark snapshot" is the generator's output over the committed run summaries.

The owner's six 0.6.2 runs (models up to 1B, 1-3B and above 3B; PL and ENG suites) live under ``runs/<group>-<lang>/
summary.json`` with the suite path reduced to its file name and without ``results.jsonl`` (git-ignored). Every table
in README and in ``benchmarks/summary_{pl,eng}.md`` must be exactly what ``deqio benchmark tables`` renders from those
files, so a hand-edited number, a stale table or a run without its committed summary fails here. The remote-API
table (run ``20261015T120300Z``, Deqio 0.6.1, no committed summary) stays pinned row by row in ``test_server.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import deqio
from deqio.benchmark_tables import build_tables, load_run, render_markdown
from deqio.catalog import load_catalog

ROOT = Path(__file__).resolve().parents[1]

# committed run directory -> (README/summary heading, suite file name, language report)
SNAPSHOT_RUNS = {
    "runs/small-pl": ("Models up to 1B — PL Bench (0.6.2 snapshot)", "pl.json", "benchmarks/summary_pl.md"),
    "runs/small-eng": ("Models up to 1B — ENG Bench (0.6.2 snapshot)", "basic.json", "benchmarks/summary_eng.md"),
    "runs/medium-pl": ("Models from 1B to 3B — PL Bench (0.6.2 snapshot)", "pl.json", "benchmarks/summary_pl.md"),
    "runs/medium-eng": ("Models from 1B to 3B — ENG Bench (0.6.2 snapshot)", "basic.json", "benchmarks/summary_eng.md"),
    "runs/max-pl": ("Models above 3B — PL Bench (0.6.2 snapshot)", "pl.json", "benchmarks/summary_pl.md"),
    "runs/max-eng": ("Models above 3B — ENG Bench (0.6.2 snapshot)", "basic.json", "benchmarks/summary_eng.md"),
}
REMOTE_HEADING = "### Remote decision APIs — PL Bench (0.6.2 snapshot)"


def _catalog() -> dict:
    return load_catalog(Path(deqio.__file__).with_name("data") / "models.json")


def _rendered(run_dir: str, title: str) -> str:
    return render_markdown(build_tables(load_run(ROOT / run_dir), catalog=_catalog()), title=title)


def test_committed_run_summaries_are_portable_and_complete() -> None:
    """Six summaries, schema 2, Deqio 0.6.2, suite named by file only (no local path), no runtime errors."""
    assert "*.jsonl" in (ROOT / ".gitignore").read_text(encoding="utf-8").split()  # results.jsonl never enters git
    for run_dir, (_, suite_file, _) in SNAPSHOT_RUNS.items():
        path = ROOT / run_dir / "summary.json"
        assert path.is_file(), f"{run_dir}/summary.json is not committed"
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert summary["schema_version"] == 2
        assert summary["suite"] == suite_file, f"{run_dir}: suite must be the file name, not a local path ({summary['suite']!r})"
        assert "/Users/" not in path.read_text(encoding="utf-8") and "/home/" not in path.read_text(encoding="utf-8")
        assert summary["run_id"].startswith("20261017T") and summary["created_at"].startswith("2026-10-17")
        for model in summary["models"]:
            assert model.get("load_error") is None, f"{run_dir}: {model['model_id']}:{model['backend']} did not load"
            assert model["runtime_identity"]["deqio_version"] == "0.6.2"
            assert model["summary"]["overall"]["runtime_errors"] == 0
            assert (model["engine"], model["backend"]) != ("decision3", "mlx")  # removed in 0.6.2; a stale row must not be committed
    groups = {run_dir: {f"{m['model_id']}:{m['backend']}" for m in json.loads((ROOT / run_dir / "summary.json").read_text(encoding="utf-8"))["models"]}
              for run_dir in SNAPSHOT_RUNS}
    assert groups["runs/small-pl"] == groups["runs/small-eng"] == {"d3-lite:gguf", "decider-0.8b:mps", "kev-0.8b:mlx", "laya-english:mlx", "von:mps"}
    assert groups["runs/medium-pl"] == groups["runs/medium-eng"] == {"d3-nano:gguf", "basal-2.0-mini:mlx", "decider-2b:mps"}
    assert groups["runs/max-pl"] == groups["runs/max-eng"] == {"d3-mini:gguf", "basal-2.0-main:mlx", "decider-4b:gguf", "jevk5-4b:gguf", "kev-4b:mlx"}


def test_readme_and_summary_reports_carry_the_generator_output_verbatim() -> None:
    """Every 0.6.2 table block equals `deqio benchmark tables --summary runs/<dir> --title "<heading>"`."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    snapshot = readme[readme.index("## Benchmark snapshot"):readme.index("## Installation")]
    for run_dir, (title, _, report) in SNAPSHOT_RUNS.items():
        block = _rendered(run_dir, title)
        assert block.startswith(f"### {title}\n\n#### Run\n\n- `20261017T")
        assert block in snapshot, f"README lacks the rendered tables of {run_dir} ({title})"
        assert block in (ROOT / report).read_text(encoding="utf-8"), f"{report} lacks the rendered tables of {run_dir}"
        assert "excluded:" not in block and "did not load:" not in block
    headings = [f"### {title}" for title, _, _ in SNAPSHOT_RUNS.values()] + [REMOTE_HEADING]
    positions = [snapshot.index(heading) for heading in headings]
    assert positions == sorted(positions), "groups go up to 1B, 1-3B, above 3B (PL before ENG), then the remote APIs"
    assert snapshot.count("#### Run\n") == len(headings) == 7
    assert "Deqio 0.6.2 · 2026-10-17" in snapshot and "Deqio 0.6.1 · 2026-10-15" in snapshot  # remote tables: the 2026-10-15 run


def test_readme_drops_the_historical_05_tables() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for stale in ("### Historical 0.5 snapshot", "20261006T082750Z", "20261006T083557Z", "20261006T090441Z", "20261006T091845Z",
                  "Basal 1.5 Mini 1.5B", "Basal 1.5 Main 4.5B", "### Small models (up to 2.5B)", "### Large models (4B",
                  "20261015T120300Z` · suite `deqio-pl-60` — 60 cases / 93 scored decisions · Deqio 0.6.1 · 2026-10-15\n- excluded:"):
        assert stale not in readme, f"README still carries {stale!r}"
    assert "#### Runs" not in readme
    # the 0.5 tables remain readable in the bundled reports, below the 0.6.2 tables
    for report, run_ids in (("benchmarks/summary_eng.md", ("20261006T082750Z", "20261006T090441Z")),
                            ("benchmarks/summary_pl.md", ("20261006T083557Z", "20261006T091845Z"))):
        text = (ROOT / report).read_text(encoding="utf-8")
        assert "## Historical 0.5 snapshot (2026-10-06)" in text
        assert text.index("(0.6.2 snapshot)") < text.index("## Historical 0.5 snapshot (2026-10-06)")
        historical = text[text.index("## Historical 0.5 snapshot (2026-10-06)"):]
        assert all(run_id in historical for run_id in run_ids) and "(0.6.2 snapshot)" not in historical


def test_readme_model_table_lists_every_catalog_profile_and_the_remote_backend() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme[readme.index("## Supported models and backends"):readme.index("## Benchmark snapshot")]
    assert "| Model | Model ID | MLX | MPS | CUDA | GGUF | Remote |" in table
    assert "| TypeSafe Jev (remote) | `typesafe-jev` | — | — | — | — | ✓ |" in table
    assert "| OpenAI Decisions API (remote) | `openai-decisions` | — | — | — | — | ✓ |" in table
    rows = [line for line in table.splitlines() if line.startswith("| ") and line.count("|") == 8 and "Model ID" not in line and "---" not in line]
    assert len(rows) == 31  # 29 local profiles + 2 remote profiles
    assert sum(row.endswith("| ✓ |") for row in rows) == 2
    catalog_ids = [entry["id"] for entry in _catalog()["models"]]
    assert len(catalog_ids) == 31
    for model_id in catalog_ids:
        assert f"| `{model_id}` |" in table, f"model table lacks the catalog profile {model_id}"


def test_changelog_062_heading_is_dated_for_the_tag() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "\n## [0.6.2] — 2026-10-17\n" in changelog
    section = changelog[changelog.index("## [0.6.2]"):changelog.index("## [0.6.1]")]
    assert "not tagged yet" not in section  # older 0.5.x headings keep their historical wording
    assert "runs/small-pl" in section and "runs/max-eng" in section and "Remote" in section
    assert "listed once" in section  # the Revision column fix (same commit pinned for several artifacts)
