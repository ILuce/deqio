# Changelog

All notable changes to Deqio are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Releases before 0.5.3 were not
tracked in this file; their history is in the git log and the 0.5.3
production audit.

## [Unreleased]

## [0.5.6] — 2026-10-09 (not tagged yet)

One patch on top of 0.5.5 closing the last P2 items of the 0.5.3 production
audit (D10b: B10, B12, B13). No public route was added or removed; the only
response change is additive (two new lists in `deqio benchmark compare` JSON and
`GET /v1/benchmarks/compare`). Tag `v0.5.6` after `ci.yml` is green.

### Added

- `deqio benchmark compare` and `GET /v1/benchmarks/compare` report profiles
  whose runtime failed to load in their own lists, `load_failed_left` and
  `load_failed_right` (each record carries `load_error`), with a dedicated
  warning (B12). Previously such profiles were filed as "legacy profiles
  without canonical identity".
- Regression tests: `tests/test_model_source_rule.py`,
  `tests/test_benchmark_load_error.py` (a real `deqio benchmark` run whose
  engine fails to load) and `tests/test_watch_ui_behavior.py`, which executes
  the Watch page's JavaScript under Node with a DOM and `fetch` stub (skipped
  when `node` is not installed; GitHub's runners have it).

### Changed

- One rule for the model an engine loads (B10). Launchers use the catalog
  profile: the SemIf sidecar receives the profile's model and revision and the
  Open-Jev sidecar the profile's checkpoint path, not `config.json` values.
  `config.json` / `DEQIO_MODEL` / `DEQIO_MODEL_REVISION` are a mirror of the
  selected profile: a value that differs from the catalog profile makes loading
  fail with an explicit error before any sidecar is spawned, so response
  provenance can never name different weights than the ones served. Pin-on-
  install profiles (Basal 1.5) keep running the immutable revision recorded by
  the installer.
- Watch page (B13): the automatic 2 s refresh re-reads at least as many rows as
  are currently shown, so pages loaded with "Load older" are no longer replaced
  by the first 500; a refresh still in flight is not duplicated by the timer;
  "Load older", event details, "Clear session" and the auto-clear setting
  report failures in the health badge instead of leaving unhandled promise
  rejections.

### Fixed

- The benchmark summary table prints `load error: <reason>` for a profile that
  failed to load instead of a row of zeros (`0 cases, 0.0 %, 0 errors`), and
  the `/ui` benchmark summary shows the same (B12).

## [0.5.5] — 2026-10-09 (not tagged yet)

One patch on top of 0.5.4: the model-management state machine (D9: audit
items B1, B2, B3) and runtime guards (D10: B7, B8, B9, B11, B14, a Watch
token for non-loopback binds and a native question limit). No public route or
response field was added or removed, and loopback clients see no change. Tag
`v0.5.5` only after `ci.yml` is green and the Apple Silicon acceptance
protocol from the audit passes.

### Added

- Watch token for non-loopback binds. When `deqio serve --host` is not a
  loopback address, Watch (`/v1/watch`, `/v1/watch/settings`,
  `/v1/watch/{event_id}`, `/v1/watch/clear` and `/ui/watch`) answers `401`
  without the per-server token: `DEQIO_WATCH_TOKEN`, or a token generated at
  startup and printed once in the banner. Scripts send `X-Deqio-Watch-Token`
  (or `Authorization: Bearer`); a browser opens `/ui/watch?token=...` (or
  `/ui?token=...`) once and keeps an HttpOnly, SameSite=Strict cookie. A
  configured token is never echoed. Loopback binds, decision routes, `/ui`
  and `/health` are unchanged.
- `MAX_QUESTIONS` = 64 for native `/v1/systemone` and `/v1/soam` requests:
  more questions are rejected with `422` before inference, through the single
  rejection path (`stats.errors` + 1, one console line, one Watch row).
- Regression tests for every item above, most of them with real processes:
  `tests/test_model_management.py`, `tests/test_workspace_ownership.py`, and
  additions to the runtime-safety, server, Watch-store and live-server tests
  (a real `deqio serve --host 0.0.0.0`).

### Changed

- Updating a shared runtime is a registry state transition (B1). Before
  `deqio models update` rebuilds a runtime, every registered profile with the
  same `runtime_key` (for example the ten `kev` or the six `decision2-cuda`
  profiles) loses `verified_at`, which is kept in an `update_pending` block
  together with its previous `source`. The other profiles are restored only
  after the updated profile passes the readiness probe. A failed or
  interrupted update leaves them `installed` but not `verified` (and their
  response provenance `unresolved`) instead of claiming a verification the
  rebuilt runtime never passed; a successful update, or a successful load of
  that profile by `deqio serve`, verifies them again.
- Server liveness for `deqio benchmark`, `deqio models` and a second
  `deqio serve` (B7) is a kernel lock, `.deqio/.server-owner.lock`, that the
  registered server holds for its lifetime (`"owner_lock": true` in
  `server-control.json`). After a crash, a recycled PID no longer blocks the
  workspace until the control file is removed by hand: the free lock proves
  the file stale. The audit's suggested `/health` probe was not used: uvicorn
  binds the port only after the lifespan, so a server that is still loading
  its model (up to `sidecar_startup_seconds`) refuses connections and would
  have been declared dead. Control files written by older servers keep
  PID-based liveness.
- `deqio serve` respects `model-management.lock` (B8): it exits with code 2
  and a clear message while `deqio models setup/install/update/delete` or a
  benchmark runs in the workspace. The server holds that lock while it
  registers, so the check and the registration are one atomic step.
- CUDA detection (B9): Linux offers the `cuda` backend only when
  `nvidia-smi -L` exits 0 and lists a GPU (previously whenever the binary
  existed, as on WSL2 or GPU-less cloud images). Every CUDA profile, not only
  Decision 2.0, probes `torch.cuda.is_available()` in its own runtime before
  it is loaded or verified, and fails closed with an explicit error instead of
  silently running on the CPU.

### Fixed

- A runtime venv whose interpreter is gone (interrupted creation, uninstalled
  uv-managed Python) is rebuilt with `uv venv --clear` instead of failing with
  "already exists" until removed by hand (B2). A non-empty directory without
  `pyvenv.cfg` is reported and never deleted on a guess (uv 0.12 refuses
  `--clear` there, older uv deletes it). The generic installer now uses the
  same path. A llama.cpp or Nimble source directory without `.git` is removed
  and cloned again; previously the git commands failed or ran against an
  enclosing repository.
- `deqio models delete` unregisters a profile only after its artifacts are
  gone (B3). A cleanup error (busy file, symlinked runtime directory) keeps the
  profile registered and is reported as `error: ...` with exit code 2;
  `deqio models` reports any other `OSError` (for example `uv` missing from
  `PATH`) the same way instead of a traceback.
- The JevK5 GGUF sidecar exits when its inner `llama-server` dies (B11), so
  Deqio reports the runtime as unavailable (`503`) instead of turning the
  sidecar's endless `500` into `502`.
- Watch truncates a torn final line (crash or ENOSPC during a write) before
  the next append instead of gluing the next event onto it, which made that
  event unreadable and `total` disagree with the listing (B14).
- `test_sidecar_guard_cleans_grandchild_after_engine_leader_exits` was flaky:
  the grandchild's PID file could be read while still empty (11 of 400 trials
  in a stress run). The PID is now published atomically, the guard path no
  longer depends on the working directory and the deadlines tolerate loaded
  CI runners.

## [0.5.4] — 2026-10-09 (not tagged yet)

One patch on top of the frozen 0.5.3 tree: the freeze follow-up from the
production audit (D7: CI, lint, docs, Python 3.10, Windows decision) and the
audit items B4, B5, B6, B15 and B16 (D8). No public route, request schema or
response field was added or removed. Tag `v0.5.4` only after `ci.yml` is green
and the Apple Silicon acceptance protocol from the audit passes (no orphaned
model processes after `kill -9`, memory within ±10 % after five live switches).

### Added

- `.github/workflows/ci.yml`: `uv lock --check`, `ruff check`, byte-compile
  and the full test suite on every push and pull request, Python 3.10 and
  3.12. Actions are pinned to the same commit SHAs as the release workflows.
- `ruff` (`>=0.16,<0.17`) in the `dev` extra with a correctness-only rule set
  (`E4`, `E7`, `E9`, `F`); formatting is not enforced in 0.5.x.
- `tomli` in the `dev` extra for Python < 3.11, so the two tests that read
  `pyproject.toml` run on the declared minimum interpreter.
- `CHANGELOG.md` and `SECURITY.md`.
- README: supported platforms, all `deqio benchmark` flags (`--suite`,
  `--model`, `--all`, `--output`, `--config`), a warning that `--host 0.0.0.0`
  exposes the full Watch payloads, Shared token semantics and input limits.
- `tests/test_live_server_smoke.py`: boots the real `deqio serve` path
  (uvicorn child process, workspace control file, contract middleware) with a
  stub engine and exercises `/health`, all nine decision routes, negotiated
  requests, rejection accounting, `/v1/stats`, `/v1/watch` and a clean Ctrl-C
  shutdown over real HTTP.

### Changed

- Windows is explicitly unsupported in 0.5: `host_backends()` returns no
  backends on Windows and the install preflight reports
  "Windows is not supported in Deqio 0.5" instead of offering CUDA/GGUF
  profiles that were never validated there. Package classifiers declare macOS
  and Linux.
- `uv.lock` was re-resolved. The resolution is unchanged for every package the
  project depends on; the only addition is `ruff`, and 44 unreachable entries
  left over from a removed `semif-phase1` git dependency (torch, transformers,
  mlx, nvidia-*, llama-cpp-python and their dependencies) were dropped. They
  were never installed by `uv sync`.
- One accounting path for failed decision requests (B4). Every rejection that
  reaches a decision handler — validation before inference, input-contract
  failures, runtime and protocol errors, on `/v1/choice`, `/v1/decision`,
  `/v1/noul`, `/v1/shared`, `/v1/score`, `/v1/multi`, `/v1/act`, `/v1/soam` and
  `/v1/systemone` — adds exactly one to `stats.errors`, writes one console
  line and one Watch row. Previously some were counted, some only appeared in
  Watch and some (duplicate IDs in Shared, native 422s) left no trace.
  Requests refused before a handler runs (FastAPI body validation, the
  negotiated-body middleware) are not decision traffic and are not counted.
- Portable handlers use one worker thread per request (B16): the async handler
  only negotiates the input contract; validation, inference, the request log
  and Watch run in a single `asyncio.to_thread` call, as the native path
  already did (previously three to four executor hops per request).
- `/v1/shared` token usage (B5): one engine call scores the whole batch, so
  for batches of two or more decisions the per-decision `input_tokens` is now
  `null` (`input_tokens_source: "engine_reported_batch"` in negotiated
  receipts when the engine measured the batch, `"unknown"` otherwise) instead
  of a copy of the batch total. The batch count is reported once: in the
  negotiated batch `input_receipt`, in the Watch row (previously summed N
  times) and in the request log. Benchmark Watch rows for Shared cases follow
  the same rule. Single-decision batches keep their measured count.

### Fixed

- Tests no longer fail on Python 3.10 (`import tomllib` is 3.11+).
- The test suite passes on a clean checkout: one test read the git-ignored,
  developer-local `config.json` and only passed because release ZIPs had been
  built from a working tree; it now reads the packaged template.
- Two lint findings in tests (a semicolon statement and an unused variable that
  now carries an assertion).
- A reported token count of `0` is no longer attested as
  `engine_reported` (B6); it is `null`/`unknown`. The Nimble sidecar no longer
  sends a hard-coded `0`; the SemIf sidecar reports usage only for a single
  question (per-row counts of a prefix-sharing batch are neither a sum nor a
  maximum of the real engine input); the JevK5 sidecar reports a total only
  when every question reported one (a partial sum is not a measurement).
- Portable input validation before inference (B15), all `400`:
  whitespace-only `question` (Choice, Noul, each Shared decision);
  blank explicitly supplied IDs (`id` on Choice/Noul, `decisions[].id` —
  previously Noul silently replaced `""` with a generated ID);
  more than `MAX_OPTIONS` = 128 options per decision or `MAX_DECISIONS` = 64
  decisions per Shared request. An empty option `description` stays allowed:
  engines receive the option ID alone.
- The startup banner printed `http://127.0.0.1:8787/...` regardless of the
  `--host`/`--port` the server was actually bound to; it now reports the real
  bind address.
- `/ui` shows `-` instead of an empty value when a result has no measured
  `input_tokens` (Shared batches, engines that do not report usage).

## [0.5.3] — 2026-10-08 (frozen state after the production audit)

0.5.3 is 0.5.1 plus audit Patch 6 plus the version bump; it is the baseline
for 0.5.4 and is tagged as-is once the Apple Silicon soak passes.

### Fixed (audit Patch 6)

- Watch and the request log no longer block the server event loop in
  `/v1/choice`, `/v1/noul` and `/v1/shared` (A1).
- `pid_alive()` is a pure probe on Windows and is shared by the sidecar
  guard (A2).
- A killed sidecar supervisor no longer leaves the engine process tree holding
  accelerator memory (A3); failures after spawning a sidecar always stop the
  supervisor (A4).
- Live model activation keeps non-selection `DEQIO_*` overrides, never sticks
  in `switching`, and treats registry bookkeeping as best-effort (A5–A7).
- Deleting the active profile restores the replacement's token budget (A8).
- Watch shows latency and tokens for native SystemOne responses (A9).
- The test suite no longer touches the developer workspace (A10); an empty
  test regained its assertions (A11).
