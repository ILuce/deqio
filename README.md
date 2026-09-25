# Deqio

**Decisions in. Probabilities out.**

Deqio runs fast typed AI decision models behind one consistent local API and a lightweight browser UI. It supports multiple decision engines and lets you switch between installed models without changing your application code.

Public decision endpoints stay the same regardless of the active model:

```text
POST /v1/noul
POST /v1/choice
POST /v1/shared
```

### Supported models

| Model | MLX | MPS | CUDA |
| --- | :---: | :---: | :---: |
| SemIf / Qwen3.5 4B | ✓ | ✓ | ✓ |
| Kev 0.8B | ✓ | ✓ | ✓ |
| Kev 4B | ✓ | ✓ | ✓ |
| Kev 9B | ✓ | ✓ | ✓ |
| Decider 0.8B | — | ✓ | ✓ |
| Decider 2B | — | ✓ | ✓ |
| Decider 4B | — | ✓ | ✓ |
| Laya English 421M | ✓ | ✓ | ✓ |
| Laya Multilingual 322M | ✓ | ✓ | ✓ |
| Laya Typed Decisions 421M | ✓ | ✓ | ✓ |
| Von | — | ✓ | ✓ |
| Bespoke Nimble 9B | ✓ | — | ✓ |

## 1. Installation

### macOS — Apple Silicon

Deqio supports both **MLX** and **MPS** on Apple Silicon.

1. Install `uv` if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone Deqio and enter the project:

```bash
git clone https://github.com/ILuce/deqio.git
cd deqio
```

3. Install the project:

```bash
uv sync
```

4. Choose a backend and model:

```bash
uv run deqio models setup
```

On a Mac, the installer offers:

- `mlx` — recommended for models with native MLX support
- `mps` — PyTorch on Apple Silicon, required by models such as Decider

5. Start Deqio:

```bash
uv run deqio serve
```

The first start of a model may download its weights.

---

### Linux — NVIDIA GPU

Deqio uses the **CUDA** backend on Linux.

1. Make sure the NVIDIA driver is working:

```bash
nvidia-smi
```

2. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Clone and install Deqio:

```bash
git clone https://github.com/ILuce/deqio.git
cd deqio
uv sync
```

4. Choose a CUDA-compatible model:

```bash
uv run deqio models setup
```

5. Start Deqio:

```bash
uv run deqio serve
```

---

### Windows — NVIDIA GPU

Deqio uses the **CUDA** backend on Windows.

1. Make sure the NVIDIA driver is working in PowerShell:

```powershell
nvidia-smi
```

2. Install `uv`:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3. Clone and install Deqio:

```powershell
git clone https://github.com/ILuce/deqio.git
cd deqio
uv sync
```

4. Choose a CUDA-compatible model:

```powershell
uv run deqio models setup
```

5. Start Deqio:

```powershell
uv run deqio serve
```

---

### Models you already installed

List locally available models:

```bash
uv run deqio models installed
```

Choose another installed model before starting the server:

```bash
uv run deqio models use
```

Show the current selection:

```bash
uv run deqio status
```

## 2. Using Deqio from the UI

Start the server:

```bash
uv run deqio serve
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

### Useful links

```text
UI:        http://127.0.0.1:8787/ui
API docs:  http://127.0.0.1:8787/docs
Health:    http://127.0.0.1:8787/health
```

Stop the server with `Ctrl+C`.

## 3. Using Deqio as an API

Start Deqio once:

```bash
uv run deqio serve
```

Base URL:

```text
http://127.0.0.1:8787
```

The UI is not involved when your application calls the API directly.

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

Typical response:

```json
{
  "decision": "yes",
  "probabilities": {
    "yes": 0.97,
    "no": 0.03
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

Deqio includes an editable starter suite in `benchmarks/basic.json`: **10 Noul**, **10 Choice**, and **10 Shared** requests. Stop `deqio serve` before benchmarking so the benchmark can load each model with the machine's memory available.

Run it interactively and choose all installed models or selected profiles:

```bash
uv run deqio benchmark
```

Run every installed model compatible with the current machine:

```bash
uv run deqio benchmark --all
```

Or select profiles explicitly:

```bash
uv run deqio benchmark \
  --model decider-0.8b:mps \
  --model laya-typed-decisions:mlx
```

The console shows live PASS/FAIL and latency for every request. Full `results.jsonl` and `summary.json` files are written under `.deqio/benchmarks/<timestamp>/`. Add or edit cases in `benchmarks/basic.json` as the benchmark grows.

> **Nimble note:** `Bespoke Nimble 9B` follows the upstream MLX/CUDA workflow. Its first installation downloads the adapter and pinned Qwen3.5-9B base, then prepares merged local weights, so it needs substantially more disk/RAM than the smaller models.

For the complete request and response schemas, open:

```text
http://127.0.0.1:8787/docs
```

## License

MIT. See [LICENSE](LICENSE).
