# Changelog

All notable changes to Deqio are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Releases before 0.5.3 were not
tracked in this file; their history is in the git log and the 0.5.3
production audit.

## [Unreleased]

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
