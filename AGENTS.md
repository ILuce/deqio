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

Prepared local checkpoints such as Nimble use this directory. Do not move managed weights into `.model-runtimes/`: runtime environments should be rebuildable without forcing large model downloads or checkpoint merges again. SemIf MLX now consumes the canonical Hugging Face checkpoint directly and applies the upstream 8-bit MLX quantization path at load time.

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
gguf
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
DEQIO_WATCH_TOKEN
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
GET  /v1/watch
GET  /v1/watch/settings
GET  /v1/watch/{event_id}

POST /v1/watch/settings
POST /v1/watch/clear
POST /v1/noul
POST /v1/choice
POST /v1/decision
POST /v1/shared
POST /v1/score
POST /v1/multi
POST /v1/act
POST /v1/soam
POST /v1/systemone
POST /v1/cache/clear
POST /v1/models/activate
```

`/v1/decision` is a compatibility alias for `/v1/choice`.

Decision handlers keep blocking work off the event loop with exactly one worker thread per request: the async handler only negotiates the input contract, then one `asyncio.to_thread` call does validation, inference, the request log and Watch. Every rejection that reaches a decision handler (validation, input contract, runtime/protocol error) goes through `server._reject`, which adds exactly one to `stats.errors`, writes one console line and one Watch row. Portable inputs are validated before inference: no blank questions or explicitly supplied IDs, unique IDs, at most `MAX_OPTIONS` options per decision and `MAX_DECISIONS` decisions per Shared request. Native SystemOne requests (`/v1/systemone`, `/v1/soam`) accept at most `MAX_QUESTIONS` questions and are otherwise rejected with `422` through `server._reject`.

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

When a runnable profile uses quantized weights, record the quantization contract in the profile (`quantization` metadata and any engine-specific loader option) so provenance and tests can distinguish it from full precision. Prefer verified 8-bit MLX variants, then verified 6-bit variants, but never replace an engine-specific adapter/head with a backbone-only quant. GGUF availability alone does not justify a Deqio profile. A GGUF backend may be catalogued only when the exact current model identity has an official/authoritative Q8_0 (8-bit or better) artifact and a native decision/SystemOne readout. Community conversions and similarly named but different model versions must not be substituted silently.

Do not silently fall back from an unsupported backend to CPU or another backend.

Installed-profile state is local machine state under `.deqio/` and must not be committed. The registry key `model_id::backend` is authoritative: shared runtime directories or Hugging Face cache entries must never make a different backend appear installed.

Successful install/update should also record model artifact provenance in the local registry. For Hugging Face artifacts, store the requested revision (if any) and best-effort resolved immutable commit SHA. Existing pre-provenance registry entries remain valid, but their response attestation is incomplete until an install/update refreshes the artifact metadata. Updating a shared runtime (`runtime_key`) is a registry state transition: every registered profile of that runtime loses `verified_at` (kept in an `update_pending` block) until the update passes the readiness probe, so a failed update never leaves profiles claiming a verification the rebuilt runtime did not pass. `deqio models delete` unregisters a profile only after its artifacts are removed.

Each installed profile may also record `max_input_tokens`. `models setup` must present a bounded interactive token-budget list after model selection. Prefer the standard presets `4096`, `8192`, `12288`, `16384`, and `32768`, while honoring runtime-specific hard limits and conservative host-memory guardrails; values that fail those guards should be shown as unavailable rather than selectable. Direct install/update may accept `--max-input-tokens`, and selecting/activating an installed profile must restore its recorded token budget. The memory calculation is only a provisioning guardrail and may not be presented as proof of a backend's effective context capacity or completeness.

`models setup` / `models install` are the network-enabled provisioning phase. They must install the runtime, resolve/download all weights (including transitive base-model dependencies), start the engine, and pass a real typed-decision model-readiness probe before marking a profile verified. Normal serving/benchmarking should run Hugging Face in offline mode by default and fail clearly if the prepared cache is incomplete. Installed-state detection for Hub-backed artifacts must validate the exact requested revision/tag/commit when the catalog pins one; repository presence alone is insufficient because offline `snapshot_download(..., revision=...)` may still fail.

Model installation must run the host compatibility preflight. Catalogued memory requirements are conservative guardrails; direct install may expose an explicit force override, but interactive setup should not offer profiles below their minimum requirement.

Installation and activation are separate concerns: the live activation endpoint must never install packages or silently download a new runtime. It may only switch to a profile already detected as installed.

Sidecar process readiness and model readiness are distinct states. Opening the localhost port only proves the process is listening; a runtime is ready only after a typed-decision probe succeeds. Use the short process-ready timeout for the port and the longer configurable startup timeout for first model initialization.

Live model switching must keep one resident model at a time. Unload the old runtime before loading the new one, block inference during the transition, persist `config.json` only after the new runtime passes warmup, and attempt to restore the previous runtime on failure.

Do not alias MPS to MLX or MLX to MPS. They are distinct Apple Silicon runtime stacks.

The catalog profile is the only source of the model an engine loads: launchers take model, revision and checkpoint paths from the profile, and `config.json` / `DEQIO_MODEL` / `DEQIO_MODEL_REVISION` are a mirror written by `deqio models use/select`. A mirror value that differs from the catalog profile must fail loading with an explicit error before any sidecar is spawned (pin-on-install profiles may run the immutable revision recorded by the installer). Never override the catalog silently.

Basal 1.5 fully replaces the removed Basal 1.0 catalog profiles. The only Basal family is `basal-1.5-mini`, `basal-1.5-main`, and `basal-1.5-max`. Use the official Basal v1.5.0 `basal-serve` runtime and official model artifacts. The accepted backends are MLX 8-bit, native MPS, GGUF Q8_0, and native CUDA. Advanced semantics remain native SystemOne semantics internally: Choice, Noul, Score, Multi, Act, `facts: "auto"`, SOAM, option keys, and Evidence where the selected backend actually supports it. Patch 4 may expose thin convenience HTTP wrappers for native `score`, `multi`, `act`, and `soam`, but they must inject/forward to the same `SystemOneRuntime.system_one` path and must not implement alternative scoring. Do not create standalone Facts/Evidence/OptionKey endpoints: those remain request/question modifiers. Do not softmax Multi probabilities, do not re-decide Act results, and never silently ignore unsupported Evidence. Evidence is native-PyTorch-only for the current Basal 1.5 profiles (MPS/CUDA), not MLX/GGUF.

Clef Flash and Clef use pinned `mlx-community` 8-bit snapshots on MLX, retaining the BF16 joint schema head and the snapshot-bundled `clef_mlx.py` System One server with truncation disabled. They also expose audited official `ggml-org` Q8_0 GGUF profiles through llama.cpp's native SystemOne endpoint. Do not replace the decision head with a normal text-generation path, and do not alias MPS to MLX.

Profiles with a native SystemOne API must declare `capabilities.systemone` plus their actual per-feature support. `/ui`, `POST /v1/systemone`, `POST /v1/soam`, and the native typed convenience wrappers use the same capability metadata; do not expose a feature in the GUI that the active runtime cannot execute, and do not silently hide a supported native feature from the GUI. Preserve native probabilities and response fields rather than synthesizing replacements.

## Change workflow

Before work:

```bash
git status --short
```

After changes:

```bash
uv run ruff check src tests
uv run python -m compileall -q src tests
uv run pytest
git diff --check
git diff
```

Record user-visible changes in `CHANGELOG.md` (Keep a Changelog). `.github/workflows/ci.yml` runs the same lint, byte-compile and test steps on every push and pull request with Python 3.10 and 3.12; keep tests runnable on 3.10 (no stdlib-only-in-3.11 imports without an explicit version check).

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
uv run ruff check src tests
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
/v1/score
/v1/multi
/v1/act
/v1/soam
/v1/systemone
```

Changes to UI must verify:

1. `/ui` loads.
2. Noul requests can be submitted.
3. Choice options can be added and removed.
4. Shared decisions can be submitted.
5. Native Score/Multi/Act/SOAM examples follow the active profile capabilities and reload when the active profile changes.
6. generated JSON matches the API request.
7. responses render without page reload.
8. active engine/model/backend are visible.
9. installed model profiles are listed.
10. switching an installed profile keeps the same public API URL.
11. `/ui/watch` loads and links back to `/ui`.
12. watch rows refresh for API decision requests and direct benchmark cases, and row details expose the full temporary request/response payload.
13. model switches keep history in the same server session while every row retains its own engine/model/backend/runtime identity.
14. Watch auto-clear can be configured from both `/ui` and `/ui/watch`.
15. rotated Watch files never exceed 10,000 JSONL event lines.

## Watch UI and session inspection

`/ui/watch` is a separate operational view for decision traffic. Track parsed request/response pairs for `/v1/noul`, `/v1/choice`, `/v1/decision`, `/v1/shared`, `/v1/score`, `/v1/multi`, `/v1/act`, `/v1/soam`, and `/v1/systemone`, including failed calls that reach those endpoint handlers. Direct `deqio benchmark` case execution must write equivalent events to the same Watch store for the same workspace. Do not record watch/stats/health polling endpoints or internal readiness/warmup probes as user decision traffic.

Full Watch payloads are temporary disk-backed session data under `.deqio/watch/`, not unbounded RAM state and not part of the normal persistent JSONL request log. Store one JSON event per line and rotate to a new `events-*.jsonl` file after 10,000 event lines. Listing must be paginated/lightweight and must not read an entire potentially large event file into memory just to render one page; detailed request/response bodies should be fetched by event ID.

Delete Watch event files when the server starts, on explicit Watch clear, and when the configured automatic cleanup interval expires. Retention preferences may survive those resets. Model switches do not clear a running server's Watch history: each event must carry source, engine, model ID, backend, and runtime instance identity so UI filtering can separate models safely. If a reset/auto-clear replaces the session while a request is still in flight, reject that stale append instead of inserting it into the new session. When `deqio serve` binds a non-loopback address, Watch (`/v1/watch*`, `/ui/watch`) requires the per-server Watch token (`DEQIO_WATCH_TOKEN` or generated at startup; a configured token is never echoed). Loopback binds keep Watch open, and the token never gates decision routes.

The server and benchmark CLI can be separate processes writing the same workspace store. Keep appends/rotation/session metadata cross-process safe and keep file permissions private where the host supports it.

The main `/ui` must link to `/ui/watch`, and the watch page must link back to `/ui`. Keep both pages dependency-free. Runtime-instance IDs in Watch details must not expand the grid into neighboring latency fields: show a compact/ellipsized value, expose the full value on hover, and keep a direct copy action. The automatic refresh must keep every page loaded with "Load older" and must not pile up requests behind a slow one; every Watch handler reports failures in the health badge instead of leaving an unhandled rejection. The Watch script is executed under Node in `tests/test_watch_ui_behavior.py`; keep it free of browser-only globals beyond `document`, `fetch`, `setInterval`, `alert`, `confirm`, `prompt` and `navigator.clipboard`.

## Benchmark suites

Benchmark suites are editable schema-version-1 JSON files under the workspace `benchmarks/` directory. `deqio benchmark` without `--suite` must discover all valid `*.json` files there and present a suite selector before model selection. Packaged defaults currently include `basic.json` (`ENG Bench`) and `pl.json` (`PL Bench`), but the runtime must not hard-code those two filenames for discovery. `--suite PATH` remains the explicit/non-interactive override. Invalid JSON/suites should be skipped with a clear warning rather than making every other valid suite unusable.

### Benchmark runtime ownership

A benchmark must have exclusive inference ownership for its workspace without killing the Deqio HTTP server. If `deqio serve` is running, the benchmark asks the server through the workspace-local authenticated control channel to close its active runtime; `/ui`, `/ui/watch`, health, history and administrative APIs remain alive. Inference requests must return a clear temporary `503` and model activation must return `409` while the benchmark owns inference. Load one benchmark profile at a time and always close it before loading the next. At benchmark completion restore the server's previous profile. If the benchmark owner process disappears, the server watchdog should restore automatically. Never implement this by broad `pkill`/process-name matching. A workspace benchmark lock prevents two benchmark CLIs from running model workloads concurrently. Workspace ownership is decided by kernel locks, never by probing the HTTP port (uvicorn binds it only after the model has loaded): a registered server holds `.deqio/.server-owner.lock` for its lifetime and registers only while holding `model-management.lock`, so `deqio serve`, `deqio models` mutations and benchmarks exclude each other atomically.

### Benchmark comparison

`deqio benchmark compare` compares concrete completed run directories, not suite names. New benchmark summaries must persist canonical profile identity including at least model ID, backend, precision/quantization where relevant, significant runtime identity/version, and pinned artifact revisions. Only exact canonical-profile intersections are common. Never merge the same model across different backends or quantizations. Show only-A/only-B explicitly. Comparison output must include accuracy, decision accuracy, median/P95 latency, throughput, rank/rank shift, and per-decision-type breakdown only where both runs have the metric. Different suites may be compared factually but must not be interpreted automatically as a pure language/quality delta. JSON output is part of the agent/automation contract. Legacy summaries without canonical identity must not be silently treated as common profiles. A profile whose runtime failed to load (`load_error` in the summary) is reported as such — CLI table, UI and `load_failed_left`/`load_failed_right` in the comparison — never as `0 cases` or as a legacy profile.

## Logging

Console logging is part of the developer-facing interface.

Rules:

- identify the public server as `[deqio]`;
- prefix external runtime output as `[sidecar:<engine>]`;
- suppress internal sidecar HTTP access-log noise;
- keep the main Uvicorn access log disabled;
- log wrapper-owned `/v1/noul`, `/v1/choice`, `/v1/decision`, `/v1/shared`, `/v1/score`, `/v1/multi`, `/v1/act`, `/v1/soam`, and `/v1/systemone` requests clearly;
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
