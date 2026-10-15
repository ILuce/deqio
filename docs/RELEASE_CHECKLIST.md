# Release checklist (Deqio + deqio-client)

One Deqio release = one tag `vX.Y.Z` on a commit whose `pyproject.toml`, `src/deqio/__init__.py`,
`clients/python/pyproject.toml`, `clients/python/src/deqio_client/__init__.py`, `uv.lock` and
`tests/test_server.py::test_release_version_is_consistent` all name the same version, published as two
distributions: `deqio-client` (the dependency-free SDK, `clients/python`) first, then `deqio` (which depends on
`deqio-client>=X.Y.Z,<X.(Y+1)`). Nothing below is optional; a gate that was not run is reported as not run, never
as passed.

## 1. Before the version bump

- [ ] Every change is in `CHANGELOG.md` under `## [X.Y.Z] — YYYY-MM-DD (not tagged yet)` (Added / Changed / Fixed /
      Removed), with the public-contract statement in the section intro: nothing removed or renamed; additive
      response fields named.
- [ ] `AGENTS.md`, `README.md`, `SECURITY.md` describe the behaviour being released (routes, CLI, catalog rules).
- [ ] Every runtime pin and Hub revision that changed has its before/after benchmark on the same suite
      (`deqio benchmark compare`); a profile whose benchmark did not run stays unverified and says so in the CHANGELOG.
- [ ] Remote providers: the real cassettes recorded with `python -m deqio.provider_record` replace the synthetic
      ones under `tests/fixtures/providers/` and `uv run pytest -q tests/test_providers.py` replays them.

## 2. Version bump (one commit)

- [ ] `pyproject.toml` → `version = "X.Y.Z"` and `deqio-client>=X.Y.Z,<X.(Y+1)` in `dependencies`.
- [ ] `src/deqio/__init__.py` → `__version__ = "X.Y.Z"`.
- [ ] `clients/python/pyproject.toml` → `version = "X.Y.Z"`; `clients/python/src/deqio_client/__init__.py` →
      `__version__ = "X.Y.Z"`.
- [ ] `uv lock` (with the uv version `ci.yml` pins) → `uv.lock` names both packages at `X.Y.Z`; `uv lock --check`.
- [ ] `tests/test_server.py::test_release_version_is_consistent` → `"X.Y.Z"`.
- [ ] `uv run deqio openapi --out docs/openapi.json` (the document carries the version; the test keeps it equal).
- [ ] CHANGELOG heading keeps `(not tagged yet)` until step 5.

## 3. Local gates (source checkout)

```bash
uv sync --frozen --extra dev
uv run --frozen ruff check src tests clients
uv run --frozen python -m compileall -q src tests clients
uv run --frozen pytest -q -W error::DeprecationWarning        # includes tests/test_live_server_smoke.py
git diff --check
uv build --no-sources --all-packages                          # dist/deqio_client-X.Y.Z*, dist/deqio-X.Y.Z*
uv run --isolated --no-project --with dist/deqio_client-*.whl --with dist/deqio-*.whl tests/package_smoke.py
uv run --isolated --no-project --with dist/deqio_client-*.tar.gz --with dist/deqio-*.tar.gz tests/package_smoke.py
```

- [ ] All green on the development interpreter; CI green on 3.10 and 3.12 for the release commit (push first).
- [ ] Apple Silicon / CUDA acceptance for runtime or lifecycle changes (audit protocol): install, probe, benchmark,
      Ctrl-C shutdown, no orphan sidecar.

## 4. Freeze

- [ ] CHANGELOG heading `## [X.Y.Z] — YYYY-MM-DD` (drop `(not tagged yet)`), commit `Release X.Y.Z`.
- [ ] `git tag vX.Y.Z && git push --tags`; `publish.yml` refuses a tag that does not equal `pyproject.toml`'s version.
- [ ] `git archive --format=zip -o deqio<XYZ>_state.zip vX.Y.Z` — the next session's source of truth (never from
      the working tree).

## 5. Publish (GitHub release → `publish.yml`)

- [ ] PyPI projects `deqio` and `deqio-client` both have the repository's `publish.yml` registered as a Trusted
      Publisher (environment `pypi`); `release-test.yml` publishes both to TestPyPI first (`workflow_dispatch`).
- [ ] Create the GitHub release for `vX.Y.Z`; the workflow builds both distributions, smokes the wheels and sdists in
      isolation and publishes `dist/*` — `deqio_client` is uploaded before `deqio` so the dependency resolves.
- [ ] After publishing: `uv tool install deqio==X.Y.Z` on a clean machine, `deqio --version`, `deqio models list`,
      `python -c "import deqio_client; print(deqio_client.__version__)"`.

## 6. After the release

- [ ] CHANGELOG gets a fresh `## [Unreleased]`; `CONTEXT_HANDOFF.md` records the tag, the state ZIP and the gates that
      were run on hardware.
