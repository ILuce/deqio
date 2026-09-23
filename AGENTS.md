# AGENTS.md

## Purpose

This repository contains a small local HTTP server around SemIf.

Its primary purpose is to expose fast semantic decisions to other applications, through a stable local API.

Keep the project small, deterministic, observable, and easy to reinstall with `uv`.

---

## Core principles

1. Keep this repository independent from downstream applications and private integrations.
2. Keep SemIf as an upstream dependency.
3. Do not vendor SemIf source code into this repository.
4. Do not commit model weights.
5. Do not commit `.venv`.
6. Prefer simple code over framework abstractions.
7. Do not introduce infrastructure such as Docker, Redis, Prometheus or Grafana unless explicitly requested.
8. Preserve backward compatibility of existing HTTP endpoints unless a task explicitly allows breaking changes.
9. Keep one model instance per server process.
10. Do not enable multiple Uvicorn workers for inference.

---

## Package management

This repository uses `uv` exclusively.

Do not use:

```text
pip install
python -m venv
requirements.txt
poetry
pipenv
```

Use:

```bash
uv sync
uv add
uv remove
uv lock
uv run
```

Python dependencies must be declared in `pyproject.toml`.

`uv.lock` must be committed.

---

## Runtime backend

The current release uses the MLX backend on macOS Apple Silicon.

Do not claim Linux or Windows support until a backend abstraction is implemented and validated in this repository.

When additional backends are introduced, backend-specific code must remain behind a small internal abstraction and API handlers must not contain backend-specific implementation details.

---

## Stable HTTP API

The public API currently consists of:

```text
GET  /health
GET  /ui
GET  /v1/stats
GET  /v1/recent

POST /v1/noul
POST /v1/choice
POST /v1/decision
POST /v1/shared
```

`/v1/decision` is retained as a compatibility alias for `/v1/choice`.

Do not silently change request or response schemas.

Any intentional schema change must also update:

```text
README.md
UI request builder
tests
```

---

## Change workflow

All repository modifications must be prepared and reviewed as a Git patch.

Do not perform undocumented direct edits.

For every change:

1. Inspect the current repository state.
2. Read the relevant source files before modifying them.
3. Check:

```bash
git status --short
```

4. Implement only the requested change.
5. Produce the patch:

```bash
git diff --binary > change.patch
```

6. Review the patch:

```bash
git diff --check
git diff
```

7. Run the required validation.
8. Report:

   * files changed
   * behavior changed
   * tests executed
   * validation result
   * known limitations

When applying a supplied patch, always validate it first:

```bash
git apply --check change.patch
```

Only after validation:

```bash
git apply change.patch
```

Never use:

```bash
git apply --reject
```

unless explicitly requested.

A patch must either apply cleanly or be corrected.

---

## Validation

Minimum validation after Python changes:

```bash
uv sync --frozen --extra dev
uv run python -m compileall -q src tests
uv run pytest
```

For server changes, also verify:

```bash
curl http://127.0.0.1:8787/health
```

Changes affecting decision endpoints must test at least:

```text
/v1/noul
/v1/choice
/v1/shared
```

Changes affecting the web UI must verify that:

1. `/ui` loads successfully.
2. Noul requests can be submitted.
3. Choice options can be added and removed.
4. Shared decisions can be submitted.
5. Generated JSON matches the API request.
6. Server responses are displayed without page reload.

---

## UI rules

The UI is a developer/debugging interface, not a consumer application.

Keep it:

* lightweight
* dependency-free
* responsive
* readable
* functional without a build step

Prefer plain:

```text
HTML
CSS
JavaScript
```

Do not introduce React, Vue, Svelte, npm or another frontend toolchain unless explicitly requested.

The UI should expose enough information to understand model behavior:

* endpoint
* state
* question
* options
* generated request JSON
* decision
* probabilities
* logits
* latency
* cache hit
* token count

Avoid visual effects or unnecessary animation.

---

## Logging

Do not log the complete state by default.

Persist:

* timestamp
* request ID
* endpoint/mode
* state hash
* question
* option IDs
* decision
* probabilities
* latency
* token count
* cache status

---

## Performance

Performance is part of the API contract.

Avoid changes that:

* reload the model between requests
* create additional model instances
* serialize inference unnecessarily beyond backend requirements
* disable prefix reuse
* convert classification into text generation

Record latency regressions when modifying inference code.

---

## SemIf updates

SemIf is pinned to a Git revision through `pyproject.toml`.

Do not automatically track the upstream `master` branch.

When upgrading SemIf:

1. update the pinned revision
2. run:

```bash
uv lock
uv sync
```

3. start the server
4. test all decision endpoints
5. compare representative probability outputs
6. compare latency
7. commit the updated `uv.lock`

Probability changes after an upstream or model update are expected and must be treated as an explicit behavioral change.

---

## Git discipline

Before beginning work:

```bash
git status --short
```

Do not overwrite unrelated user modifications.

Do not use destructive commands such as:

```text
git reset --hard
git clean -fd
git checkout -- .
```

unless explicitly requested.

Keep commits focused on one logical change.

Suggested commit prefixes:

```text
feat:
fix:
perf:
refactor:
docs:
test:
chore:
```

---

## Project scope

This repository should remain a small SemIf serving and inspection layer.

Do not add documentation, source comments, configuration, examples, or tests that name private downstream projects or private integration targets unless explicitly required for a public integration.

Do not push, publish, tag, or create a remote repository unless the user explicitly requests that action.
