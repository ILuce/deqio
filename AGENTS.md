# AGENTS.md

## Purpose

This repository contains a small local HTTP server around SemIf.

Its purpose is to expose fast semantic decisions to other applications through a stable local API while keeping the model resident, observable, and easy to reinstall with `uv`.

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

## Configuration

`config.json` is the checked-in source of runtime defaults.

The server currently reads:

```text
backend
model
model_revision
max_tokens
mlx_cache_mib
log
torch_dtype
llama_gguf
llama_threads
```

Keep the default values in `config.json`; do not move them back into Python source.

Environment-variable support is override-only and must remain backward compatible for the documented `SEMIF_*` variables.

Relative filesystem paths from `config.json` must resolve relative to the configuration file, not the caller's working directory.

---

## Runtime backends

Supported backend names are:

```text
mlx       macOS Apple Silicon
cuda      Linux or Windows with one visible NVIDIA CUDA GPU
llamacpp  CPU backend on macOS, Linux, or Windows
```

Backend-specific loading, scoring, cache cleanup, and shutdown belong in `src/semif_server/backends.py`.

API handlers must not import MLX, Torch, or llama.cpp backend implementations directly.

All backends must expose the same server semantics:

```text
direct scoring
serial prefix reuse
shared-state scoring
runtime cache clearing
```

Do not claim support for a platform/backend combination unless it is supported by the pinned SemIf dependency.

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
POST /v1/cache/clear
```

`/v1/decision` is retained as a compatibility alias for `/v1/choice`.

`/v1/cache/clear` clears reusable runtime cache state but must not unload the active model or delete model files/disk caches.

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

Changes affecting runtime caching must test:

```text
/v1/cache/clear
```

Changes affecting the web UI must verify that:

1. `/ui` loads successfully.
2. Noul requests can be submitted.
3. Choice options can be added and removed.
4. Shared decisions and their options can be added and removed.
5. Text and JSON state modes both generate valid requests.
6. Generated JSON matches the API request.
7. Server responses are displayed without page reload.
8. Runtime cache clearing is available and reports success/failure clearly.

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
* state and state format
* question
* options
* generated request JSON
* decision
* probabilities
* logits
* latency
* cache hit
* token count
* active backend

Avoid visual effects or unnecessary animation.

---

## Logging

Do not log the complete state by default.

Persist:

* timestamp
* request ID
* backend
* endpoint/mode
* state hash
* question
* decision
* probabilities
* latency
* token count
* cache status

---

## Cache semantics

The cache-clear operation is runtime hygiene, not model unloading.

It should clear:

* the current serial prefix cache for every backend;
* MLX inactive allocator cache when using MLX;
* the CUDA allocator cache when using CUDA;
* llama.cpp scoring-context memory when using llama.cpp.

The model weights must remain loaded after cache clearing.

Do not delete Hugging Face cache files or local model files from the HTTP cache endpoint.

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

1. update the pinned revision;
2. run:

```bash
uv lock
uv sync
```

3. start the available backends;
4. test all decision endpoints and runtime-cache clearing;
5. compare representative probability outputs;
6. compare latency;
7. commit the updated `uv.lock`.

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
