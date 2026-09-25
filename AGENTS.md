# AGENTS.md

## Purpose

Deqio is a small multi-engine runtime for typed AI decisions.

It exposes one stable HTTP API and a lightweight browser UI over independent decision engines while keeping installation reproducible with `uv`.

Keep the project small, deterministic, observable, fast, and easy to reinstall.

## Core principles

1. Preserve the public HTTP API unless a task explicitly permits a breaking change.
2. Keep model/engine-specific behavior behind runtime adapters.
3. Do not vendor upstream engine source code or model weights.
4. Do not commit `.venv`, `.model-runtimes`, model weights, local Deqio state, or request logs.
5. Use `uv` exclusively for Python and dependency management.
6. Keep one selected decision model resident per server process.
7. Do not convert classification into autoregressive answer generation.
8. Do not claim backend support that the upstream runtime does not actually provide.
9. Keep the browser UI dependency-free: plain HTML, CSS, and JavaScript only.
10. Treat probability calibration as engine-specific.

## Naming

The project, Python package, CLI, logs, and local state use the Deqio name:

```text
project:      deqio
package:      deqio
CLI:          deqio
console:      [deqio]
local state:  .deqio/
```

SemIf remains the name of one supported upstream engine. Do not rename upstream engine names, model IDs, package names, or protocol concepts to Deqio.

## Runtime architecture

The main environment contains Deqio and the native SemIf integration.

Other engines live in isolated environments under:

```text
.model-runtimes/<runtime-key>/
```

This isolation is intentional. Independent engines can require conflicting PyTorch, Transformers, MLX, or accelerator versions.

The model catalog is `models.json`.

The active selection is stored in `config.json` as:

```text
engine
model_id
backend
model
```

Supported wrapper backend labels are:

```text
mlx
mps
cuda
```

A catalog profile must exist for every allowed `(model_id, backend)` pair.

## Package management

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
uv venv
uv pip install --python ...
```

The public CLI is:

```bash
uv run deqio serve
uv run deqio models list
uv run deqio models installed
uv run deqio models setup
uv run deqio models use
uv run deqio models status
uv run deqio status
```

## Configuration

Default configuration lives in `config.json`.

Environment overrides use the `DEQIO_` prefix, including:

```text
DEQIO_CONFIG
DEQIO_ENGINE
DEQIO_MODEL_ID
DEQIO_BACKEND
DEQIO_MODEL
DEQIO_MODEL_REVISION
DEQIO_MAX_TOKENS
DEQIO_MLX_CACHE_MIB
DEQIO_LOG
DEQIO_TORCH_DTYPE
DEQIO_RUNTIME_DIR
DEQIO_MODEL_CATALOG
DEQIO_SIDECAR_STARTUP_SECONDS
```

Keep project configuration under the `DEQIO_` namespace. `SemIf` remains the name of one upstream engine.

## Public HTTP API

Keep these endpoints backward compatible:

```text
GET  /
GET  /health
GET  /ui
GET  /v1/models
GET  /v1/models/installed
GET  /v1/stats
GET  /v1/recent

POST /v1/noul
POST /v1/choice
POST /v1/decision
POST /v1/shared
POST /v1/cache/clear
POST /v1/models/activate
```

`/v1/decision` is a compatibility alias for `/v1/choice`.

External engines should be normalized to the same response shape. If an engine does not expose raw logits, return an empty `option_logits` object. Never fabricate logits from probabilities and label them as raw logits.

## Model catalog rules

`models.json` is the source of truth for selectable models.

Every entry needs:

- stable local `id`
- `engine`
- human-readable label and description
- upstream source URL
- explicit compatible backends
- model identifier/path for each backend
- isolated runtime install packages when applicable

Do not silently fall back from an unsupported backend to CPU or another backend.

Installed-profile state is local machine state under `.deqio/` and must not be committed. Installation and activation are separate concerns: the live activation endpoint must never install packages or silently download a new runtime. It may only switch to a profile already detected as installed.

Live model switching must keep one resident model at a time. Unload the old runtime before loading the new one, block inference during the transition, persist `config.json` only after the new runtime passes warmup, and attempt to restore the previous runtime on failure.

Do not alias MPS to MLX or MLX to MPS. They are distinct Apple Silicon runtime stacks.

## Change workflow

Before work:

```bash
git status --short
```

After changes:

```bash
uv run python -m compileall -q src tests
uv run pytest
git diff --check
git diff
```

Produce a patch with:

```bash
git diff --binary > change.patch
```

Before applying a supplied patch:

```bash
git apply --check change.patch
git apply change.patch
```

Do not use `git apply --reject` unless explicitly requested.

Do not use destructive Git commands unless explicitly requested.

## Validation

Minimum validation after Python changes:

```bash
uv run python -m compileall -q src tests
uv run pytest
```

Changes to model management must also verify:

```bash
uv run deqio models list
uv run deqio models installed
uv run deqio status
```

Changes to serving must verify:

```text
/health
/v1/noul
/v1/choice
/v1/shared
```

Changes to UI must verify:

1. `/ui` loads.
2. Noul requests can be submitted.
3. Choice options can be added and removed.
4. Shared decisions can be submitted.
5. generated JSON matches the API request.
6. responses render without page reload.
7. active engine/model/backend are visible.
8. installed model profiles are listed.
9. switching an installed profile keeps the same public API URL.

## Logging

Console logging is part of the developer-facing interface.

Rules:

- identify the public server as `[deqio]`;
- prefix external runtime output as `[sidecar:<engine>]`;
- suppress internal sidecar HTTP access-log noise;
- keep the main Uvicorn access log disabled;
- log wrapper-owned `/v1/noul`, `/v1/choice`, `/v1/decision`, and `/v1/shared` requests clearly;
- print the UI, API docs, and health URLs at startup;
- describe random sidecar ports as internal inference-only;
- keep useful model download, loading, accelerator, and runtime diagnostics visible.

Do not persist the full request state by default.

## Performance

Avoid changes that:

- reload model weights per request;
- create duplicate resident models;
- add text generation to typed decisions;
- disable native batching/shared-state behavior without reason;
- merge incompatible engine dependencies into the main environment;
- put UI rendering or model-selection work in the normal decision request path.

External engines should remain resident in a localhost-only child process for the lifetime of the main server.
