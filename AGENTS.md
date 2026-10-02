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

The main environment contains the Deqio core only. Every decision engine, including SemIf, runs in an isolated environment under:

```text
.model-runtimes/<runtime-key>/
```

This isolation is intentional. Independent engines can require conflicting PyTorch, Transformers, MLX, or accelerator versions.

Keep runtime environments separate from local model checkpoints. Downloaded or prepared weights that Deqio manages explicitly live under:

```text
models/
```

For example, the SemIf MLX snapshot and prepared Nimble checkpoint use this directory. Do not move these weights into `.model-runtimes/`: runtime environments should be rebuildable without forcing large model downloads or checkpoint merges again.

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

The installed public CLI is:

```bash
deqio serve
deqio models list
deqio models installed
deqio models setup
deqio models use
deqio models delete
deqio models status
deqio status
deqio --version
deqio version
```

From a source checkout, prefix these commands with `uv run`. User-facing runtime errors and help text for the installed tool should use `deqio ...`, not `uv run deqio ...`. Missing model/runtime guidance should point users to `deqio models setup`.

Every interactive terminal selection must offer `q` as a non-error exit path. `quit` and `exit` may be accepted as aliases.

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
DEQIO_SIDECAR_PROCESS_READY_SECONDS
DEQIO_HF_OFFLINE_RUNTIME
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

Decision responses expose Decision Provenance / Attestation v1 under `provenance`. The model/runtime identity must be captured under the same inference lock as the score so a live model switch cannot pair a result with the wrong model identity. Provenance must include the loaded runtime instance ID, engine, model ID, backend, model source, recorded artifact revisions, and structured score provenance.

Score provenance must distinguish engine-reported probabilities from Deqio transforms or synthetic fallbacks. In particular, one-hot probabilities synthesized from a choice-only engine response must be marked synthetic and must never be presented as native model confidence. Renormalization/clamping must be explicit transforms. Calibration remains engine-specific and must not be inferred by Deqio.

Attestation v1 is a local unsigned response binding. Its digest binds the decision result, prompt hash, runtime identity, and score provenance, but it is not a cryptographic signature or remote trust proof. `complete` may only be true when runtime identity, resolved model artifact revisions, and score provenance are all available. `/v1/shared` must preserve per-result provenance and expose a batch-level binding to the same runtime instance.

An optional versioned input-completeness contract is negotiated with `Deqio-Contract: input-completeness-v1`. Clients without the header retain the existing schema. Unknown versions must fail before inference; never silently downgrade. For negotiated requests, hash the exact HTTP body bytes before FastAPI parsing, reject duplicate JSON keys, and require explicit stable decision/option IDs. `input_policy.require_complete=true` is fail-closed: if the active runtime cannot prove exact model-boundary completeness, reject before inference with `input_completeness_unavailable`. Never infer completeness from `truncation=false`, configured `max_tokens`, token counts alone, or absence of an error. Missing usage is `null`/`unknown`, never a fabricated zero.

A negotiated response may use attestation schema v2 to bind its `input_receipt` to the same request/result/runtime/score identity while preserving attestation v1 semantics for non-negotiated clients. `status=unknown` must remain distinct from `complete`. Shared receipts require unique decision IDs and must describe each decision rather than copying one aggregate token count as if it were per-decision measurement. Do not claim model-boundary completeness until the engine adapter actually instruments rendered input, logical token IDs, masks/cache behavior, and the effective scoring limit.

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

Installed-profile state is local machine state under `.deqio/` and must not be committed. The registry key `model_id::backend` is authoritative: shared runtime directories or Hugging Face cache entries must never make a different backend appear installed.

Successful install/update should also record model artifact provenance in the local registry. For Hugging Face artifacts, store the requested revision (if any) and best-effort resolved immutable commit SHA. Existing pre-provenance registry entries remain valid, but their response attestation is incomplete until an install/update refreshes the artifact metadata.

Each installed profile may also record `max_input_tokens`. `models setup` must ask for this value (with a sensible profile/config default), direct install/update may accept `--max-input-tokens`, and selecting/activating an installed profile must restore its recorded token budget. Treat this as configuration, not proof of a backend's effective context capacity.

`models setup` / `models install` are the network-enabled provisioning phase. They must install the runtime, resolve/download all weights (including transitive base-model dependencies), start the engine, and pass a real typed-decision model-readiness probe before marking a profile verified. Normal serving/benchmarking should run Hugging Face in offline mode by default and fail clearly if the prepared cache is incomplete.

Model installation must run the host compatibility preflight. Catalogued memory requirements are conservative guardrails; direct install may expose an explicit force override, but interactive setup should not offer profiles below their minimum requirement.

Installation and activation are separate concerns: the live activation endpoint must never install packages or silently download a new runtime. It may only switch to a profile already detected as installed.

Sidecar process readiness and model readiness are distinct states. Opening the localhost port only proves the process is listening; a runtime is ready only after a typed-decision probe succeeds. Use the short process-ready timeout for the port and the longer configurable startup timeout for first model initialization.

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
uv run deqio --version
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
