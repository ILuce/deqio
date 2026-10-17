# Changelog

All notable changes to Deqio are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Releases before 0.5.3 were not
tracked in this file; their history is in the git log and the 0.5.3
production audit.

## [Unreleased]

## [0.6.2] — 2026-10-17 (not tagged yet)

D13 on top of the tagged 0.6.1, plus the removal of the Decision 3.0 MLX backend. Public HTTP routes and response
fields are unchanged; nothing was removed or renamed and no response field was added. New: the OpenAPI 3.1 document
of the decision routes, the Python SDK `deqio-client`, `deqio decide`, `deqio openapi`, `deqio benchmark tables`, the
release checklist. The core package gains one dependency, `deqio-client` (the workspace member `clients/python`,
itself dependency-free). Catalog: the five `d3-*:mlx` profiles are gone (see "Removed"); every other profile and pin
is unchanged.

### Added
- `GET /openapi.json` (and `/docs`) now describes the responses of the eight decision routes and the `/v1/decision`
  alias (`deqio.openapi`): the portable result with `provenance` (`provider`, `remote_request_id`, `cost_estimate`,
  attestation, runtime identity, score provenance), the `/v1/shared` batch, the native System One reply with its
  `deqio` block, the `Deqio-Contract` header on the portable routes and the 400/422/502/503 error shapes. OpenAPI
  version `3.1.0`; tag `decisions`; `operationId`s `decide_<route>`. The models document and validate — they are
  not FastAPI `response_model`s, so the handlers serialise exactly what they did before. `deqio openapi [--out FILE]`
  writes the document; `docs/openapi.json` is the committed copy and a test keeps it equal to the generated one.
- Python SDK `deqio-client` (`clients/python`, package `deqio_client`, no dependencies, never imports `deqio`;
  version in lockstep with Deqio): `DeqioClient(base_url)` with `noul`, `choice`, `decision`, `shared`, `score`,
  `multi`, `act`, `soam`, `systemone` and `health`; `contract=True` negotiates `input-completeness-v1`; option
  mappings `{id: description}` are accepted next to the explicit list; every method returns the server's JSON
  unchanged and raises `DeqioError` (status, `detail`) on a rejection or `DeqioConnectionError` when the server
  cannot be reached — no retry, no fallback.
- `deqio decide ROUTE [--url URL] --state TEXT|--state-file F|--state-json J ...`: one decision from the command line
  through the SDK (`--question`/`--option ID[=DESCRIPTION]` for noul/choice/decision, `--decisions-json` for shared,
  `--instructions`/`--criterion KEY[=DESCRIPTION]` or `--question-json` for score/multi/act, `--questions-json` for
  soam/systemone, `--contract`/`--require-complete`, `--facts`); prints a summary with a provenance line or the
  full response with `--json`; exit 0 decision, 1 server rejection or unreachable server, 2 usage error.
- `docs/RELEASE_CHECKLIST.md`: the release procedure for both distributions (version bump places, gates, tag,
  state ZIP, Trusted Publishing order). `tests/test_live_server_smoke.py` validates every decision route's response
  against the OpenAPI models and drives the SDK and `deqio decide` over the real server; `tests/package_smoke.py`
  checks both distributions, `deqio decide --help`, the exit code against a closed port and `deqio openapi`.

- `deqio benchmark tables --run RUN_ID | --summary PATH [--models MODEL_ID[:BACKEND] ...] [--title] [--json] [--out]`
  (`deqio.benchmark_tables`): the README / `benchmarks/summary_*.md` tables of a completed run from its `summary.json`
  and `results.jsonl` — quality (case and decision accuracy, per type), performance (median, P95, throughput,
  estimated cost per 1 000 scored decisions for remote profiles) and identity (engine, runtime, precision, source,
  model/endpoint, pinned revision); rows sorted by decision accuracy, profiles that did not load listed, rows of the
  removed Decision 3.0 MLX profiles excluded by name. README "Benchmark snapshot" now opens with the 0.6.2
  tables (models up to 1B and the remote APIs from the owner's PL run `20261015T120300Z`; the 1–3B and 3B+ groups
  follow) above the historical 0.5 tables; `benchmarks/summary_pl.md` carries the same 0.6.2 section.

### Changed
- `tests/fixtures/providers/*/choice.json` are the owner's real recordings (2026-10-15, `python -m
  deqio.provider_record`): OpenAI Decisions `gpt-6-luna` (HTTP 200, 147 input tokens, request id from the
  `x-request-id` header) and TypeSafe `jev-1.13.0` (HTTP 200, 351 input tokens, no request id). Both equal the
  documented wires byte for byte on the request and parse unchanged, so no mapping changed (D11c not needed); the
  replay test now takes every expectation from the recording instead of the former synthetic values (0.91, 412).
- `pyproject.toml`: uv workspace with the member `clients/python`; core dependency `deqio-client>=0.6.2,<0.7`
  (`uv.lock` regenerated with uv 0.12.18, no other resolution change). `ci.yml` lints and byte-compiles `clients`
  too; `publish.yml`/`release-test.yml` build both distributions (`uv build --no-sources --all-packages`), smoke
  each wheel and sdist pair in isolation and publish `dist/*` (`deqio-client` before `deqio`); the release tag check
  also requires the two versions to match. The README gains the "Python SDK, `deqio decide` and the OpenAPI
  document" section. Version 0.6.2.

### Removed (2026-10-17, after the owner's MLX-vs-GGUF benchmark)
- Decision 3.0 on MLX: the catalog profiles `d3:mlx`, `d3-flash:mlx`, `d3-mini:mlx`, `d3-nano:mlx`, `d3-lite:mlx`
  (community BF16 conversions by `ahmedandaloes`, runtime `d3-mlx`, pins `mlx-vlm==0.7.6` / `mlx==0.32.3`), the
  installer `d3_mlx`, the launcher branch, Deqio's own forward `deqio.d3_mlx_sidecar` (`DecisionForward`,
  `deqio-d3-mlx-forward/2`) with `runtime_metadata.forward`, and the opt-in runtime tests
  (`tests/test_decision3_mlx_runtime.py`, `tests/_d3_mlx_synthetic.py`). On the owner's Mac the MLX path scored
  41.9 % / 56.8 % decision accuracy (PL / ENG, runs `small-pl` / `small-eng`, 2026-10-17) against 90.3 % / 90.8 % for
  the same `d3-lite` on GGUF: even with the official `decision_config.json` semantics the community conversion does
  not reproduce the official readout, and no verified MLX conversion of Decision 3.0 exists. `decision3` on `mlx` is
  refused by the launcher by name ("Decision 3.0 runs on CUDA (official server) or GGUF (llama.cpp); backend 'mlx'
  has no launcher and no fallback is allowed"); a stale `config.json` or registry row naming it fails loading the same
  way; `deqio benchmark tables` excludes such rows by name. On Apple Silicon the Decision 3.0 GGUF profiles are the
  supported path. Also dropped as dead code: `providers.remote_provider_names`, `providers.local_provider_descriptor`,
  `credentials.keychain_backend_name`, `openapi.RESPONSE_SCHEMAS` (never referenced).

### Fixed (0.6.2 fix 3, 2026-10-15, after the owner's first Decision 3.0 MLX benchmark; superseded by the removal above)
- Decision 3.0 MLX: with fix 2 `d3-lite` answered the second option / "yes" on 60 of 60 `deqio-pl-60` cases
  (decision accuracy 54.8 %, Basal 2.0 main 94.6 % on the same run) — the readout did not depend on the state. Two
  causes, both verified against the official `vllm-sr/d3-*` package (`d3_runtime.py`, `decision_config.json`): the
  community `d3_mlx.py` selects the option probabilities as `probs[code_ids[i]]`, the codes' *tokenizer ids* (`A` = 32,
  `B` = 33 …), from the 255-way softmax of the `[255, hidden]` readout — rows of codes `AG`/`AH`/… instead of rows
  0..n-1 — and fix 2 copied that line; and `decision_config.json` declares `attention_mode: noncausal_full_attention`
  (the official runtime runs the full-attention layers without a causal mask while the Gated DeltaNet layers stay
  causal), whereas fix 2 ran mlx-vlm's causal path. `deqio.d3_mlx_sidecar.DecisionForward` now follows the
  checkpoint's `decision_config.json` (`d3-code-readout-v1`): readout row `i` is option `i`, softmax over the first `n`
  rows divided by the config's `temperature`, the full-attention layers get no mask in `noncausal_full_attention` mode
  and the causal mask in `causal` mode, the linear-attention layers never get one; a snapshot without the file, with
  another `format_id`/`pooling`/`attention_mode`, or whose `codes`/`token_ids` differ from the loader's is refused
  with the cause in the sidecar reply. The forward marker is `deqio-d3-mlx-forward/2` (`/health`, answers,
  `runtime_metadata.forward`; `/1` results are not comparable). The opt-in runtime tests prove on a synthetic
  checkpoint that an earlier position now sees later tokens through the full-attention layers only, that `causal`
  mode equals mlx-vlm's own path, and that the probabilities equal the positional readout. No catalog, pin or
  dependency change; the MLX profiles stay `source: community` and unverified until the owner's benchmark.

## [0.6.1] — 2026-10-14

Public HTTP routes and response fields are unchanged; nothing was removed or renamed. The catalog families Decision 2.0
and Basal 1.5 and the 0.6.0.dev0 provider configuration (`provider*` in `config.json`, `DEQIO_PROVIDER*`,
`deqio benchmark --provider*`) are removed; remote decision APIs are catalog profiles on the new `remote` backend.
0.6.0.dev0 was never tagged, so this is the first 0.6 release; the 0.6.0 SDK/OpenAPI scope (D13) moves to 0.6.2.

### Added
- `remote` backend: `deqio models setup` offers it beside mlx/mps/gguf/cuda on every supported host; the catalog
  profiles `openai-decisions` (OpenAI Decisions API, `gpt-6-luna`) and `typesafe-jev` (TypeSafe System One API,
  `jev-latest`) are installed like local models — model identifier, API key, token budget, one probe decision — and
  served, switched, benchmarked and deleted like them (`deqio.remote_runtime`).
- API keys live in the OS keychain through the new core dependency `keyring` (`deqio.credentials`; service `deqio`);
  a host without a keychain backend is an explicit error. `deqio models delete MODEL --backend remote` removes the key.
- Decision 3.0 (`decision3`): catalog profiles `d3`, `d3-flash`, `d3-mini`, `d3-nano`, `d3-lite` on CUDA (the
  official package's `d3_server.py` run from the pinned snapshot; `MODEL_MANIFEST.json` identity verified against the
  catalog), GGUF Q8_0 (`ggml-org`, through llama.cpp) and MLX (community BF16 conversions by `ahmedandaloes` through
  the new `deqio.d3_mlx_sidecar`; choice and noul only). Native probabilities are preserved unchanged.
- Basal 2.0: catalog profiles `basal-2.0-mini`, `basal-2.0-main`, `basal-2.0-max` on MLX 8-bit and GGUF Q8_0 with the
  official `rkinas/basal` v2.0.0 runtime (`af1ce43a`) and the author's artifacts pinned by commit.
- `deqio models install|update|select --remote-model` for remote profiles; `python -m deqio.provider_record --key
  keychain|env`.

### Changed
- D11b: the remote wire mappings follow the vendors' documentation instead of the guessed provider wire v1 —
  OpenAI Decisions (`input`, `questions` array with `name`/`choices`/`levels`, `predicate` for noul, `answers` array,
  `usage.input_tokens`, `refusal` is an explicit error) and TypeSafe (`POST https://api.typesafe.ai/v1/systemone`,
  native System One wire, `TYPESAFE_API_KEY` convention, versioned `model` in the reply). The synthetic cassettes
  under `tests/fixtures/providers/` are rewritten in the documented shapes.
- llama.cpp pinned for every GGUF profile at `cbe28bb3cb0758f0c566a9a3561766821aeb0748` (2026-10-13, "fix vllm d3
  multi seq case", ggml-org/llama.cpp#30408) instead of `99b95488`; the existing GGUF profiles (Kev, Clef, Laya)
  need their before/after benchmark before the tag.
- `deqio benchmark` resolves a profile's installed token budget and remote model identifier from the registry.
- Version 0.6.1.

### Removed
- Decision 2.0 engine, sidecar and the six `decision-2.0-*` profiles; Basal 1.5 profiles (`basal-1.5-*`, all four
  backends); `config.json` `provider`/`provider_model`/`provider_base_url`, `DEQIO_PROVIDER*`, the
  `DEQIO_PROVIDER_OPENAI_DECISIONS`/`DEQIO_PROVIDER_TYPESAFE_JEV` flags and `deqio benchmark --provider*`.

### Fixed (0.6.1 fix 1, 2026-10-14, after the owner's first install on Apple Silicon)
- Decision 3.0 MLX: the runtime pin `mlx-vlm==0.3.9` / `mlx==0.29.3` had no `mlx_vlm.models.qwen3_5` (the module
  `d3_mlx.py` imports), so the install-time probe failed with ImportError; the profiles now pin `mlx-vlm==0.7.6`
  (2026-10-05, the last release before the catalog cutoff) and `mlx==0.32.3` (macOS 14+ wheels). The sidecar reports
  a runtime import/load failure in its HTTP reply, so Deqio's console shows the cause instead of a bare `HTTP 500`.
- Basal 2.0: memory guardrails restored to the Basal 1.5 values for the same weights and quantizations (mini 1.8/4,
  main 5.2/8, max 12.0/20 GiB on MLX; 1.7/5.1/12.0 on GGUF), so `basal-2.0-max` is offered on a 16 GiB Mac again.
  `basal-2.0-mini` keeps the model's own context (`max_position_embeddings` 8192 in both the 1.5 and the 2.0 mini
  checkpoints; main/max 32768) as its token-budget limit, and every Basal 2.0 profile now runs `basal-serve` with
  `--context-overflow refuse`: a prompt longer than the model context is an explicit 422, never a state whose middle
  basal-serve cut silently.

### Fixed (0.6.1 fix 2, 2026-10-14, after the owner's second Apple Silicon install)
- Decision 3.0 MLX: with the corrected pin the community `d3_mlx.py` loaded, and its own inference loop failed at
  the first full-attention layer (`AttributeError: 'NoneType' object has no attribute 'offset'`): it calls every
  decoder layer as `layer(h)` without a cache or a mask, and the `qwen3_5` attention of every released mlx-vlm
  (0.5.0 … 0.7.8) reads `cache.offset` unconditionally; it also hard-codes `fa_idx = 2`, a linear-attention layer in
  d3-lite. `deqio.d3_mlx_sidecar` now owns the forward pass (`DecisionForward`, `deqio-d3-mlx-forward/1`): the
  community file supplies the loader (weights, readout head, tokenizer, answer codes) and the d3 prompt contract,
  Deqio runs the checkpoint through mlx-vlm's standard causal path with a fresh per-question cache and the first
  full-attention / linear-attention layer indices taken from the checkpoint's `layer_types`, refusing a model whose
  layers do not match them. The path is named in the sidecar's `/health` and answers (`forward`) and in the MLX
  profiles' `runtime_metadata.forward` (additive, provenance only). Opt-in runtime tests
  (`tests/test_decision3_mlx_runtime.py`, synthetic checkpoint from `tests/_d3_mlx_synthetic.py`) reproduce the
  community failure and prove Deqio's forward is causal and deterministic on any host with `mlx` (Linux `mlx-cpu`
  included); they skip without `mlx_vlm` or `DEQIO_D3_MLX_SNAPSHOT`.

## [0.6.0.dev0] — 2026-10-13 (0.6.0-dev; not tagged)

One patch on top of 0.5.9: the decision-provider abstraction (D11). No public route, response field or CLI command was
removed or renamed. Additive fields: `provenance.provider`, `provenance.remote_request_id`, `provenance.cost_estimate`
on `/v1/noul`, `/v1/choice` and `/v1/shared` (batch-level on `/v1/shared`), and `deqio.provider`,
`deqio.remote_request_id`, `deqio.cost_estimate` on `/v1/systemone`; `provider` inside `provenance.runtime`;
`provider_name`, `remote_request_id`, `cost_estimate` per row in benchmark `results.jsonl`.

### Added

- `deqio.providers`: `DecisionProvider` registry — `local` (the catalog sidecar, unchanged) and the remote providers
  `openai-decisions` (`POST https://api.openai.com/v1/decisions`, default model `gpt-6-luna`, `OPENAI_API_KEY`) and
  `typesafe-jev` (`TYPESAFE_JEV_BASE_URL` + `TYPESAFE_JEV_API_KEY`), each behind its own flag
  (`DEQIO_PROVIDER_OPENAI_DECISIONS=1`, `DEQIO_PROVIDER_TYPESAFE_JEV=1`). A remote provider is selected explicitly
  (`config.json` `provider`, `provider_model`, `provider_base_url` or `DEQIO_PROVIDER*`) and is never a fallback;
  a missing flag, key or endpoint, and any non-200 or non-decision answer, is an explicit error.
- `deqio.remote_runtime.RemoteProviderRuntime`: the server/benchmark runtime surface without a sidecar; results carry
  `provider`, `remote_request_id` and a USD `cost_estimate` from the API's reported `usage.input_tokens` and the
  provider's published input price (amount `null` when usage is not reported; batch-scoped on `/v1/shared`).
- `deqio benchmark --provider openai-decisions|typesafe-jev [--provider-model M] [--provider-base-url URL]`: runs a
  suite through a remote provider (no live-server suspension); `summary.json` identity is
  `engine=<provider>`, `model_id=<remote model>`, `backend=remote`.
- `python -m deqio.provider_record --provider NAME --out FILE`: records one real exchange as a cassette (no secrets);
  the shipped cassettes in `tests/fixtures/providers/` are synthetic examples of the provider wire v1 and are meant
  to be replaced by recordings.
- `tests/test_providers.py` (15 node IDs): registry, flags/keys, cassette contract, HTTP round trip through a local
  fake API, error surfacing, settings/env, provenance fields, benchmark provider run.

### Not verified in this patch

- The wire shape of OpenAI Decisions and TypeSafe Jev (`{"model","state","questions"}` → `{"id","answers","usage"}`)
  is the provider wire v1 assumption documented in `README.md`; it was not checked against the real APIs (no network
  from the session). The first `deqio benchmark --provider ...` run or `provider_record` will show any mismatch.


## [0.5.9] — 2026-10-12 (not tagged yet)

One patch on top of 0.5.8: Hub revisions pinned from the owner's `pin_audit` (D12c). No public route, response
field or CLI command was added, removed or renamed; `python -m deqio.clm_sidecar` gains an optional `--revision`.

### Changed

- 46 of the 48 `HF_REVISION_PENDING` artifacts now carry the commit `pin_audit` resolved on 2026-10-12 for the
  catalog's own reference (`v1.0` for Kev, `main` elsewhere): Kev ×10 (`kev.serve --run repo@<commit>`; the
  upstream resolver partitions on `@`), the pin-on-install GGUF profiles ×10 and Basal ×12 (+3 GGUF repos),
  Open-Jev 27B, CLM (`Qwen/Qwen3-8B` and the `CLM-v0.1-8B.pt` checkpoint), JevK5 CUDA ×2 and Decider ×6.
  Von stays pending with its reason in the test (`von serve` resolves `main` itself).
- Decider and JevK5 launchers: `DECIDER_MODEL` / `jevk5.server --model` receive the path of the pinned commit's
  cached snapshot (`snapshot_download(..., local_files_only=True)`); a missing snapshot refuses the start with the
  `deqio models setup` / `update` hint instead of passing the floating repo id.
- CLM launcher passes `--revision` to the sidecar, which adds `vllm serve --revision <commit>`.
- Pin-on-install profiles whose catalog now names the commit: the catalog wins; an installation verified at
  another commit is refused with `run deqio models update` (previously its recorded commit was loaded).
- Decision 2.0 Eos/Sol/Nox/Lux/Vega: `triton==3.7.1` → `triton==3.7.0`, the version `torch==2.12.0` (the
  installer's PyTorch-index pin) requires on Linux; 3.7.1 belongs to torch 2.12.1. Kai pins no triton.
- `tests/test_hf_revision_pins.py` (51 node IDs); `HF_REVISION_PENDING` shrinks to the two Von entries.

### Not changed on purpose

- llama.cpp stays at `99b95488`: no GGUF regression with `a657f7e9` was supplied.
- Nimble stays at Hub `bd792f44` (the newest Hub commit, "Release latest Bespoke Nimble 9B") and source
  `dcfdbd9a` (merge of #13, after #11 "Support latest Nimble 9B release and checkpoint defaults").
- Decision 3.0 (`vllm-sr/decision-30`) and its GGUF/MLX variants are not added here: a new engine is its own patch
  (scheduled for 0.7), and the Hub was not reachable from this session to pin or license-check it. Benchmarks of a
  new model run after it is integrated and gate its verification, not its integration.

### Upgrade notes

- `deqio models use` again for a selected Kev, Decider, JevK5, CLM, Open-Jev 27B or pin-on-install profile: the
  mirror in `config.json` must name the new commit (B10 refuses the old `v1.0` / `main` / empty value).
- Decider and JevK5 profiles installed before 0.5.9 need `deqio models update` once so the pinned snapshot is in
  the cache; the launcher refuses to start until it is.

## [0.5.8] — 2026-10-10 (not tagged yet)

One patch on top of 0.5.7: reproducible engine runtimes (D12b). No public route, response field or CLI
command was added, removed or renamed; `python -m deqio.pin_audit` is a new read-only maintenance module.

### Added

- `exclude_newer` on every catalog profile (`2026-10-10T00:00:00Z`). Installers pass it to every
  `uv pip install` that resolves from PyPI, so transitive dependencies resolve as of that instant on every
  install. Profiles sharing a runtime use one instant, and a profile without it is refused at install with
  an explicit error. The two commands that install one exact `torch==` pin from the PyTorch index run
  without it: once a cutoff is set, uv drops every file of an index that publishes no upload times.
- `python -m deqio.pin_audit` (read-only): installed runtime packages and the Nimble checkout against the
  catalog pins, the Hugging Face artifacts not yet pinned to a commit with the locally recorded and the
  current Hub revision, and the Nimble Hub commits; exit status 1 on a difference or a failed lookup. It
  replaces the shell snippets of the 0.5.7 handoff, whose heredoc broke when copied with its list indentation.
- `tests/test_reproducible_installs.py`.

### Fixed

- CLM profile: the catalog named the distribution `clm`, but upstream publishes it as `contrastive-lm`
  since 2026-09-24 (fe9b720) and uv refuses a direct URL whose metadata name differs, so the profile could
  not be installed. Same pinned commit, extras and `clm-serve` executable.

### Not changed on purpose

- `HF_REVISION_PENDING` keeps its 48 entries until the owner's `pin_audit` output is available; each
  replacement ships with launcher support where the engine takes no revision.
- llama.cpp stays at `99b95488`: no GGUF regression with `a657f7e9` has been reported.

### Upgrade notes

- `deqio models update MODEL_ID --backend BACKEND` re-resolves a runtime as of the cutoff; a package
  uploaded after 2026-10-10T00:00:00Z is replaced by its newest earlier release.
- Deqio never overwrites a workspace `models.json`. A workspace created by an older installed `deqio` keeps
  its first catalog: replace it with the packaged one to get the D12/D12b pins (installs refuse a profile
  without `exclude_newer`). Repository checkouts get the new catalog with the patch.

## [0.5.7] — 2026-10-10 (not tagged yet)

One patch on top of 0.5.6: catalog pinning (D12). Two installs of the same catalog now build the
same engine code. No public route or response field was added, removed or renamed. The existing
per-profile `source` field (already on 28 of 68 profiles, exported by `/v1/models` and provenance)
is now present on every profile.

### Changed

- Every engine runtime package in `models.json` is an exact PyPI version or a GitHub commit (D12):
  Kev `kev-1.0` → `6b719c3c`; Basal `refs/tags/v1.5.0` archive → archive of commit `cd63c083`;
  Decider CUDA/MPS `main` → v1.9.0 `f8c933af` (1.8.1 returned NaN probabilities on MPS for
  `temperature_by_type` models such as decider-2b v11; 1.9.0 changes nothing else on the
  `decider.serve` path); Decider GGUF `v1.8.1` → `50d0be0d`; Laya `main` → v0.3.28 `a4a8921a`;
  JevK5 `main` → v0.3.3 `f944fe37`; JevK5 GGUF `v0.3.0` → `6c6522fe`; Von `main` → v1.3.7 `587f1022`;
  Open-Jev `main` → `bd411888`; CLM `main` → `d5f9ef0f`; `laya-mlx` → `==0.3.0`; `fastapi`/`uvicorn`
  → the core lock versions `0.141.1`/`0.53.0`; `huggingface-hub>=1,<2` → `==1.31.0`.
- Installers pin what they add themselves: sidecar transport (`SIDECAR_TRANSPORT_PACKAGES`),
  llama.cpp build tools (`cmake==4.4.4`, `ninja==1.13.2`), the Nimble preparation Hub client.
- Nimble: install and update check out the catalog commit `nimble_source_revision` (`dcfdbd9a`,
  includes upstream #11 for the 2026-09-24 checkpoint) instead of cloning or pulling `main`; a
  checkout that cannot reach the pinned commit is an error.
- Every profile declares `source: official|community`; the five community ports (Laya MLX ×3,
  Clef MLX ×2) also name `source_url`.
- Laya loads the revision the catalog pins: CUDA/MPS through `LAYA_REVISION` (the upstream reviewed
  pins of laya v0.3.28), MLX through the sidecar's new `--revision` (the revisions the laya-mlx 0.3.0
  publisher recorded for its weights).

### Added

- `tests/test_catalog_pinning.py`: pinned specs in the catalog and in every installer command,
  sidecar transport equal to `uv.lock`, a Hugging Face revision ratchet (`HF_REVISION_PENDING`, may
  only shrink), source labels, shared-runtime consistency, the Nimble checkout against a real git
  upstream, the Laya launchers and the Laya MLX sidecar.

### Not changed on purpose

- llama.cpp stays at `99b95488`; the bump to `a657f7e9` waits for the GGUF regression on
  basic-150/pl-60 across the six shared-runtime GGUF profiles.
- Upgrades past the verified releases (Decider beyond 1.9.0, Laya 0.3.29/0.4.x with its new router
  default, laya-mlx 0.4.0) are separate benchmarked changes.
- Hugging Face revisions that need huggingface.co or launcher support stay in `HF_REVISION_PENDING`.

### Upgrade notes

- Pins apply when a runtime is (re)built: run `deqio models update MODEL_ID --backend BACKEND` for
  every installed profile.
- Laya profiles now carry `model_revision`; a `config.json` written before 0.5.7 is rejected by the
  B10 rule until `deqio models use` re-selects the profile.
- The 40 profiles that gained `source` get a new benchmark `canonical_id`: compare their pre/post
  0.5.7 runs by the per-profile numbers, not by the common set of `deqio benchmark compare`.

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
