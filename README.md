# Deqio

**Decisions in. Probabilities out.**

Deqio runs fast typed AI decision models behind one consistent local API and a lightweight browser UI. It supports multiple decision engines and lets you switch between installed models without changing your application code.

Public decision endpoints stay the same regardless of the active model:

```text
POST /v1/noul
POST /v1/choice
POST /v1/shared
```

Check the installed Deqio version with either:

```bash
deqio --version
# or
deqio version
```

## Benchmark snapshot

> **Apple Silicon macOS · 16.0 GiB unified memory**<br>
> `deqio-basic-150` · **150 requests** · **250 scored decisions** · 50 Noul / 50 Choice / 50 Shared<br>
> Results captured on **2026-09-26**. Higher accuracy is better; lower latency is better.

**Highlights from this run**

- **Highest overall accuracy:** Decider 4B / MPS — **95.3%** case accuracy, **97.2%** decision accuracy.
- **Lowest median latency:** Laya Multilingual / MLX — **13.2 ms**.
- **Perfect Noul + Choice accuracy:** Kev 4B / MLX — **100.0% / 100.0%**.

| Model | Backend | Accuracy | Decision accuracy | Median | P95 |
| --- | :---: | ---: | ---: | ---: | ---: |
| **Decider 4B** | MPS | **95.3%** | 97.2% | 435.7 ms | 1,358.6 ms |
| **Kev 4B** | MLX | **92.7%** | 95.2% | 317.6 ms | 507.9 ms |
| **Decider 2B** | MPS | **90.7%** | 94.0% | 185.3 ms | 524.5 ms |
| **SemIf / Qwen3.5 4B** | MLX | **90.7%** | 93.6% | 552.3 ms | 997.0 ms |
| Decider 0.8B | MPS | 84.0% | 88.8% | 292.2 ms | 376.8 ms |
| Kev 0.8B | MLX | 77.3% | 84.4% | 60.2 ms | 96.9 ms |
| Von | MPS | 64.0% | 72.8% | 61.0 ms | 104.5 ms |
| Laya Typed Decisions 421M | MLX | 62.0% | 72.4% | 34.2 ms | 74.6 ms |
| Laya English 421M | MLX | 54.7% | 66.8% | 34.5 ms | 74.9 ms |
| Laya Multilingual 322M | MLX | 44.7% | 56.4% | 13.2 ms | 27.1 ms |

*Sorted by overall case accuracy; ties are ordered by decision accuracy. A Shared case passes only when every expected decision in that request is correct, while decision accuracy scores each decision independently.*

<details>
<summary><strong>Per-type case accuracy</strong></summary>

| Model | Backend | Noul | Choice | Shared |
| --- | :---: | ---: | ---: | ---: |
| Decider 4B | MPS | 100.0% | 98.0% | 88.0% |
| Kev 4B | MLX | 100.0% | 100.0% | 78.0% |
| Decider 2B | MPS | 94.0% | 94.0% | 84.0% |
| SemIf / Qwen3.5 4B | MLX | 96.0% | 96.0% | 80.0% |
| Decider 0.8B | MPS | 88.0% | 94.0% | 70.0% |
| Kev 0.8B | MLX | 86.0% | 86.0% | 60.0% |
| Von | MPS | 86.0% | 58.0% | 48.0% |
| Laya Typed Decisions 421M | MLX | 74.0% | 78.0% | 34.0% |
| Laya English 421M | MLX | 68.0% | 66.0% | 30.0% |
| Laya Multilingual 322M | MLX | 62.0% | 48.0% | 24.0% |

</details>

> These are machine-specific results from one local run, not universal model rankings. Results can change with hardware, runtime versions, model revisions, and benchmark changes.

### Supported models

| Model | MLX | MPS | CUDA |
| --- | :---: | :---: | :---: |
| SemIf / Qwen3.5 4B | ✓ | ✓ | ✓ |
| Kev 0.8B | ✓ | ✓ | ✓ |
| Kev 4B | ✓ | ✓ | ✓ |
| Kev 9B | ✓ | ✓ | ✓ |
| Kev 27B | — | — | ✓ |
| JevK5 4B | — | — | ✓ |
| JevK5 9B | — | — | ✓ |
| Open-Jev 2B | — | — | ✓ |
| Open-Jev 9B | — | — | ✓ |
| Open-Jev 27B v1.1 | — | — | ✓ |
| CLM 8B | — | — | ✓ |
| Decider 0.8B | — | ✓ | ✓ |
| Decider 2B | — | ✓ | ✓ |
| Decider 4B | — | ✓ | ✓ |
| Laya English 421M | ✓ | ✓ | ✓ |
| Laya Multilingual 322M | ✓ | ✓ | ✓ |
| Laya Typed Decisions 421M | ✓ | ✓ | ✓ |
| Von | — | ✓ | ✓ |
| Bespoke Nimble 9B v2 | ✓ | — | ✓ |

## 1. Installation

The recommended installation is from **PyPI** with `uv tool`. Deqio keeps models, isolated runtimes, configuration, and benchmark results in the directory where you use it; the Python package itself stays small.

### macOS — Apple Silicon

Deqio supports both **MLX** and **MPS** on Apple Silicon.

1. Install `uv` if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install Deqio from PyPI:

```bash
uv tool install deqio
```

3. Create a workspace and enter it:

```bash
mkdir -p deqio-work
cd deqio-work
```

4. Choose and install a backend/model:

```bash
deqio models setup
```

On a Mac, the installer offers:

- `mlx` — recommended for models with native MLX support
- `mps` — PyTorch on Apple Silicon, required by models such as Decider

`models setup` detects host memory and hides profiles that do not meet the catalogued minimum for the selected backend. After selecting a model it shows a second interactive list of **maximum input-token budgets** (normally `4096`, `8192`, `12288`, `16384`, and `32768`). Values above a catalogued runtime limit or the conservative host-memory guardrail are listed as unavailable instead of being selectable. For example, a profile that already consumes almost all usable unified/GPU memory may only offer `4096`, while a smaller model can offer larger budgets.

The token-memory check is an installation guardrail, not a claim about the model's true attention/context capacity. Runtime-specific hard limits still win, and the input-completeness contract described below remains the only strict mechanism for refusing a request when complete model-boundary input cannot be proven. The chosen budget is stored with the installed profile and restored when that model/backend is selected again.

After the budget is selected, the installer installs the isolated runtime, downloads/prepares the model weights, starts the model once, and requires a real typed-decision readiness probe to pass before the profile is registered as installed.

For non-interactive provisioning, pass the same value explicitly:

```bash
deqio models install MODEL_ID --backend BACKEND --max-input-tokens 8192
```

This value is the Deqio-configured input budget for that installed profile. A backend/model may have a stricter effective limit; Deqio never treats the configured number alone as proof that an input reached the model intact.

On first setup Deqio creates editable `config.json`, `models.json`, and `benchmarks/basic.json` files in this workspace. Model runtimes and weights are kept outside the PyPI package.

5. Start Deqio:

```bash
deqio serve
```

After setup succeeds, normal `serve` and benchmark starts use the local Hugging Face cache in offline mode. They do not intentionally contact Hugging Face or download missing weights; if cached artifacts are missing, reinstall/update the profile instead.

---

### Linux — NVIDIA GPU

Deqio uses the **CUDA** backend on Linux.

1. Make sure the NVIDIA driver is working:

```bash
nvidia-smi
```

2. Install `uv` and Deqio:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install deqio
```

3. Create a workspace:

```bash
mkdir -p deqio-work
cd deqio-work
```

4. Choose and install a CUDA-compatible model:

```bash
deqio models setup
```

5. Start Deqio:

```bash
deqio serve
```

---

### Windows — NVIDIA GPU

Deqio uses the **CUDA** backend on Windows.

1. Make sure the NVIDIA driver is working in PowerShell:

```powershell
nvidia-smi
```

2. Install `uv` and Deqio:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv tool install deqio
```

3. Create a workspace:

```powershell
New-Item -ItemType Directory -Force deqio-work
Set-Location deqio-work
```

4. Choose and install a CUDA-compatible model:

```powershell
deqio models setup
```

5. Start Deqio:

```powershell
deqio serve
```

---

### Install from source

For development, clone the repository instead of installing the PyPI tool:

```bash
git clone https://github.com/ILuce/deqio.git
cd deqio
uv sync --extra dev
uv run deqio models setup
uv run deqio serve
```

When running from a source checkout, use `uv run deqio ...`; when installed from PyPI with `uv tool install deqio`, use `deqio ...` directly.

### Model lifecycle

Show models that are compatible with the current host:

```bash
deqio models list --compatible
```

For machine-readable host/memory information (useful for agents):

```bash
deqio models list --compatible --json
```

List locally installed profiles:

```bash
deqio models installed
```

Choose another installed profile:

```bash
deqio models use
```

Interactive CLI menus always accept `q` (also `quit` / `exit`) to leave without making a selection.

Update an installed model/runtime profile and refresh its recorded artifact revisions:

```bash
deqio models update MODEL_ID --backend BACKEND
```

Delete an installed profile interactively:

```bash
deqio models delete
```

Or explicitly:

```bash
deqio models delete kev-4b --backend mlx --yes
```

Deqio removes workspace-owned model/runtime artifacts when they are no longer shared by another installed profile. The global Hugging Face cache is kept by default because other applications may use it. Add `--purge-cache` if you explicitly want Deqio to remove the profile's primary unshared Hub repository from that global cache.

Show the current selection:

```bash
deqio status
```

Memory values in the catalog are conservative compatibility guardrails, not exact peak-memory guarantees. `models install ... --force` can bypass the guard for advanced users.

Model startup has two separate readiness deadlines: the sidecar process must open its local API first (default 300 seconds), then the model must answer a typed readiness probe (new-workspace default 1800 seconds). Override them when needed with `DEQIO_SIDECAR_PROCESS_READY_SECONDS` and `DEQIO_SIDECAR_STARTUP_SECONDS`. `DEQIO_HF_OFFLINE_RUNTIME=false` can temporarily disable normal offline runtime mode for diagnostics; installation/update already enables network access automatically.

Stop a running `deqio serve` process before deleting model/runtime files from the same workspace.

## 2. Using Deqio from the UI

Start the server:

```bash
deqio serve
```

Open:

```text
http://127.0.0.1:8787
```

The browser redirects to the Deqio UI.

### Select a model

At the top of the UI, choose one of the models already installed on the machine and click **Activate**. Deqio unloads the previous runtime, loads the selected model, runs a warmup, and keeps the public API on the same address.

### Noul — yes/no decision

Use **Noul** when the result should be a binary decision.

Fill in:

- `State` — the context
- `Question` — the yes/no question

Click **Send** to see the selected answer, probabilities, latency, token count, and cache information.

### Choice — choose between options

Use **Choice** when the model should select one option from a list.

Fill in:

- `State`
- `Question`
- one or more options with an `ID` and description

Add or remove options directly in the UI and click **Send**.

### Shared — several decisions over one state

Use **Shared** when several decisions should be evaluated against the same context.

Enter the shared `State`, add the decisions and their options, then send them together. This is preferable to several separate requests when an engine can reuse the common state efficiently.

### Watch — current model session

Click **Watch requests** in the main UI or open:

```text
http://127.0.0.1:8787/ui/watch
```

Watch is a server-focused view for the currently loaded model. It keeps an **in-memory session history** of model-serving calls to `/v1/noul`, `/v1/choice`, `/v1/decision`, and `/v1/shared`. The table shows endpoint, status, decision, probability, latency, and input-token information. Click any row to inspect the complete parsed request and Deqio response, including provenance/attestation fields when present.

The watch session is intentionally ephemeral: it resets when the server restarts, when the active model is switched, or when **Clear session** is used. Full request/response bodies are not added to the normal JSONL request log. This keeps the existing persistent logging behavior unchanged while still making live debugging possible.

Watch API endpoints are:

```text
GET  /v1/watch
GET  /v1/watch/{event_id}
POST /v1/watch/clear
```

### Useful links

```text
UI:        http://127.0.0.1:8787/ui
Watch:     http://127.0.0.1:8787/ui/watch
API docs:  http://127.0.0.1:8787/docs
Health:    http://127.0.0.1:8787/health
```

Stop the server with `Ctrl+C`.

## 3. Using Deqio as an API

Start Deqio once:

```bash
deqio serve
```

Base URL:

```text
http://127.0.0.1:8787
```

The UI is not involved when your application calls the API directly.

### Decision Provenance / Attestation v1

Starting with Deqio `0.2.1`, every decision response includes a `provenance` object captured from the same loaded runtime instance that produced the score. It identifies the engine/model/backend, the runtime instance, recorded model artifact revisions, and how the returned probabilities were obtained.

The score metadata distinguishes values reported by the engine from values transformed by Deqio. For example, a choice-only upstream response that Deqio expands to `1.0 / 0.0` is explicitly marked as `synthetic_one_hot`; it must not be interpreted as 100% model confidence. Renormalization and Noul clamping are also reported as transforms.

A response contains a local attestation digest binding the request result to its runtime and score provenance:

```json
{
  "provenance": {
    "schema_version": 1,
    "runtime": {
      "deqio_version": "0.2.1",
      "runtime_instance_id": "...",
      "engine": "decider",
      "model_id": "decider-4b",
      "backend": "mps",
      "model": "Mapika/decider-4b",
      "requested_revision": null,
      "artifacts": [
        {
          "source": "huggingface",
          "repo_id": "Mapika/decider-4b",
          "requested_revision": null,
          "resolved_revision": "<commit-sha>"
        }
      ],
      "artifact_revisions_resolved": true
    },
    "score": {
      "kind": "engine_probability",
      "source": "engine.probabilities",
      "synthetic": false,
      "normalized": false,
      "transforms": [],
      "raw_logits_available": false,
      "calibration": "unspecified"
    },
    "attestation": {
      "kind": "deqio-local-response",
      "signed": false,
      "complete": true,
      "sha256": "..."
    }
  }
}
```

The attestation is a **local, unsigned integrity/correlation binding**, not a cryptographic signature or remote trust proof. `complete: true` means the response has a runtime instance identity, resolved model artifact revisions, and known score provenance. Profiles installed before Deqio 0.2.1 may initially report `complete: false`; run `deqio models update MODEL_ID --backend BACKEND` to refresh installation metadata and resolved artifact revisions.

`/v1/shared` returns provenance for every result plus a batch-level provenance object that binds the result attestations to the same runtime instance. `/health` exposes `runtime_instance_id`, `provenance_schema_version`, and whether artifact revisions are resolved.

### Negotiated input-completeness contract

Clients that need an explicit statement about whether the complete declared input reached the model boundary can opt in with:

```text
Deqio-Contract: input-completeness-v1
```

and include an `input_policy` in the request:

```json
{
  "input_policy": {
    "require_complete": true,
    "overflow": "reject"
  }
}
```

The contract is intentionally **fail-closed**. Deqio hashes the exact HTTP request bytes before FastAPI body parsing, rejects duplicate JSON keys and unknown contract versions, validates decision/option IDs, and binds the resulting `input_receipt` into response attestation schema v2. A strict request is rejected before inference when the active backend cannot prove model-boundary completeness. Deqio does not silently truncate, silently downgrade the contract, or turn missing token usage into a synthetic zero.

Current external System-One sidecars do not expose enough model-boundary instrumentation to prove exact rendered token IDs, masks, and cache consumption. For those runtimes a strict `require_complete: true` request therefore returns `input_completeness_unavailable` with `inference_performed: false`. A negotiated non-strict request may still run, but its receipt is explicitly `status: "unknown"`; `unknown` is never promoted to `complete` merely because no truncation error was reported.

Example negotiated request:

```bash
curl -s \
  -X POST http://127.0.0.1:8787/v1/choice \
  -H 'Content-Type: application/json' \
  -H 'Deqio-Contract: input-completeness-v1' \
  -d '{
    "id": "route-17",
    "state": "Full state used for the decision",
    "question": "Which route should be selected?",
    "options": [
      {"id": "a", "description": "Route A"},
      {"id": "b", "description": "Route B"}
    ],
    "input_policy": {"require_complete": false, "overflow": "reject"}
  }'
```

Clients that do not send `Deqio-Contract` keep the existing response schema and behavior. The negotiated contract is additive and versioned independently from the normal decision API.

### Noul

```bash
curl -s \
  -X POST http://127.0.0.1:8787/v1/noul \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "The patch changed source code and no tests have been run yet.",
    "question": "Should tests be run before considering the task complete?"
  }'
```

Typical response (abridged):

```json
{
  "decision": "yes",
  "probabilities": {
    "yes": 0.97,
    "no": 0.03
  },
  "provenance": {
    "schema_version": 1,
    "runtime": {
      "engine": "decider",
      "model_id": "decider-4b",
      "backend": "mps",
      "runtime_instance_id": "..."
    },
    "score": {
      "kind": "engine_probability",
      "source": "engine.noul",
      "synthetic": false
    },
    "attestation": {
      "signed": false,
      "complete": true,
      "sha256": "..."
    }
  }
}
```

### Choice

```bash
curl -s \
  -X POST http://127.0.0.1:8787/v1/choice \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "A customer was charged twice for the same subscription renewal.",
    "question": "Which team should handle this request?",
    "options": [
      {"id": "billing", "description": "Payments, refunds, invoices, and duplicate charges."},
      {"id": "access", "description": "Login and account access problems."},
      {"id": "technical", "description": "Product defects and service failures."}
    ]
  }'
```

### Shared

```bash
curl -s \
  -X POST http://127.0.0.1:8787/v1/shared \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "A patch changed an authentication module. Unit tests passed, but integration tests have not been run.",
    "decisions": [
      {
        "id": "validation",
        "question": "What should happen next?",
        "options": [
          {"id": "run_integration_tests", "description": "Run integration tests before proceeding."},
          {"id": "finish", "description": "Finish without more validation."}
        ]
      },
      {
        "id": "release",
        "question": "Is the change ready to release?",
        "options": [
          {"id": "yes", "description": "The change is ready to release."},
          {"id": "no", "description": "More validation is required."}
        ]
      }
    ]
  }'
```

### Check and switch the active model through the API

List installed model profiles:

```bash
curl -s http://127.0.0.1:8787/v1/models/installed
```

Activate an already installed profile:

```bash
curl -s \
  -X POST http://127.0.0.1:8787/v1/models/activate \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "decider-0.8b",
    "backend": "mps"
  }'
```

The decision API URL does not change when the active model changes.


## Benchmarking installed models

Deqio includes an editable starter suite in `benchmarks/basic.json`: **50 Noul**, **50 Choice**, and **50 Shared** requests. Stop `deqio serve` before benchmarking so the benchmark can load each model with the machine's memory available.

Run it interactively and choose all installed models or selected profiles:

```bash
deqio benchmark
```

Run every installed model compatible with the current machine:

```bash
deqio benchmark --all
```

Or select profiles explicitly:

```bash
deqio benchmark \
  --model decider-0.8b:mps \
  --model laya-typed-decisions:mlx
```

The console shows live PASS/FAIL and latency for every request. Full `results.jsonl` and `summary.json` files are written under `.deqio/benchmarks/<timestamp>/`. Add or edit cases in `benchmarks/basic.json` as the benchmark grows.

> **Nimble note:** `Bespoke Nimble 9B v2` follows the upstream MLX/CUDA workflow. Its first installation downloads the adapter and pinned Qwen3.5-9B base, then prepares merged local weights, so it needs substantially more disk/RAM than the smaller models.

> **CUDA-only model note:** JevK5, Open-Jev and CLM use their upstream Linux/NVIDIA serving paths. CLM starts a local vLLM pooling encoder plus `clm-serve` inside one Deqio-managed sidecar. Kev 27B requires an 80 GB-class NVIDIA GPU.

For the complete request and response schemas, open:

```text
http://127.0.0.1:8787/docs
```

## License

MIT. See [LICENSE](LICENSE).
