# SemIf Local Server

Small local HTTP server for [SemIf](https://github.com/TheoLeeCJ/SemIf).

The server keeps one SemIf decision model loaded in memory and exposes generation-free semantic decisions over HTTP. It returns option probabilities and logits instead of generating answer text.

## Features

- `POST /v1/noul` — binary yes/no decisions
- `POST /v1/choice` — decisions between arbitrary options
- `POST /v1/shared` — multiple decisions over one shared state
- `POST /v1/cache/clear` — clear reusable runtime caches without unloading the model
- serial prefix-cache reuse
- request latency and probability statistics
- lightweight browser playground and dashboard
- JSONL request logs
- FastAPI/OpenAPI interface
- MLX, CUDA, and llama.cpp backends

`POST /v1/decision` is kept as a compatibility alias for `POST /v1/choice`.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- one of the supported runtimes below

| Backend | Platforms | Hardware |
| --- | --- | --- |
| `mlx` | macOS Apple Silicon | Apple GPU / unified memory |
| `cuda` | Linux, Windows | one visible NVIDIA CUDA GPU |
| `llamacpp` | macOS, Linux, Windows | CPU |

The default `config.json` is configured for the MLX backend on Apple Silicon.

## Clone

```bash
git clone https://github.com/ILuce/semif-local
cd semif-local-server
uv python install 3.12
```

## Configuration

Runtime defaults live in `config.json`:

```json
{
  "backend": "mlx",
  "model": "models/semif-qwen3.5-4b-mlx-4bit",
  "model_revision": "local-vinci00-semif-qwen35-4b-mlx4",
  "max_tokens": 4096,
  "mlx_cache_mib": 256,
  "log": "logs/requests.jsonl",
  "torch_dtype": "bfloat16",
  "llama_gguf": "models/llamacpp/Qwen_Qwen3.5-4B-Q4_K_M.gguf",
  "llama_threads": null
}
```

Relative file paths are resolved relative to `config.json`.

The original environment variables remain available as optional overrides:

- `SEMIF_MODEL`
- `SEMIF_MODEL_REVISION`
- `SEMIF_MAX_TOKENS`
- `SEMIF_MLX_CACHE_MIB`
- `SEMIF_LOG`

Additional overrides:

- `SEMIF_CONFIG` — alternate JSON configuration file
- `SEMIF_BACKEND` — `mlx`, `cuda`, or `llamacpp`
- `SEMIF_TORCH_DTYPE` — `bfloat16`, `float16`, or `float32`
- `SEMIF_GGUF` — llama.cpp GGUF path
- `SEMIF_LLAMA_THREADS` — positive CPU thread count

## macOS Apple Silicon — MLX

Install the locked project:

```bash
uv sync --frozen
```

Download the 4-bit MLX model:

```bash
uv run hf download \
  vinci00/semif-qwen3.5-4b-mlx-4bit \
  --local-dir models/semif-qwen3.5-4b-mlx-4bit
```

Use the default `config.json`:

```json
{
  "backend": "mlx",
  "model": "models/semif-qwen3.5-4b-mlx-4bit",
  "model_revision": "local-vinci00-semif-qwen35-4b-mlx4",
  "max_tokens": 4096,
  "mlx_cache_mib": 256,
  "log": "logs/requests.jsonl",
  "torch_dtype": "bfloat16",
  "llama_gguf": "models/llamacpp/Qwen_Qwen3.5-4B-Q4_K_M.gguf",
  "llama_threads": null
}
```

Run:

```bash
uv run semif-server
```

## Linux — CUDA

Install:

```bash
uv sync --frozen
```

A compatible NVIDIA driver is required. SemIf expects exactly one visible CUDA GPU for a scorer process.

Set `config.json` to:

```json
{
  "backend": "cuda",
  "model": "Qwen/Qwen3.5-4B",
  "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
  "max_tokens": 4096,
  "mlx_cache_mib": 256,
  "log": "logs/requests.jsonl",
  "torch_dtype": "bfloat16",
  "llama_gguf": "models/llamacpp/Qwen_Qwen3.5-4B-Q4_K_M.gguf",
  "llama_threads": null
}
```

Then run with one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run semif-server
```

The model is downloaded by Transformers/Hugging Face on first use unless `model` points to a local Transformers checkpoint.

## Windows — CUDA

Install from PowerShell:

```powershell
uv sync --frozen
```

Use the same CUDA `config.json` shown in the Linux section.

Expose one GPU and start the server:

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
uv run semif-server
```

A compatible NVIDIA driver is required. The model is downloaded on first use unless a local Transformers checkpoint is configured.

## macOS / Linux / Windows — llama.cpp CPU

Install the llama.cpp extra:

```bash
uv sync --frozen --extra llamacpp
```

Download a compatible GGUF:

```bash
mkdir -p models/llamacpp
uv run hf download \
  bartowski/Qwen_Qwen3.5-4B-GGUF \
  Qwen_Qwen3.5-4B-Q4_K_M.gguf \
  --local-dir models/llamacpp
```

On PowerShell, create the directory with:

```powershell
New-Item -ItemType Directory -Force models/llamacpp
```

Then run the same `uv run hf download ...` command on one line or using PowerShell backticks for line continuation.

Set `config.json` to:

```json
{
  "backend": "llamacpp",
  "model": "Qwen/Qwen3.5-4B",
  "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
  "max_tokens": 4096,
  "mlx_cache_mib": 256,
  "log": "logs/requests.jsonl",
  "torch_dtype": "bfloat16",
  "llama_gguf": "models/llamacpp/Qwen_Qwen3.5-4B-Q4_K_M.gguf",
  "llama_threads": null
}
```

`llama_threads: null` lets SemIf use the visible CPU count. Set a positive integer to cap it.

Start:

```bash
uv run semif-server
```

`llama-cpp-python` is an upstream native dependency. On platforms where a compatible wheel is unavailable, a local C/C++ build toolchain may be required during installation.

## Server

Default addresses:

- server: `http://127.0.0.1:8787`
- OpenAPI: `http://127.0.0.1:8787/docs`
- playground/dashboard: `http://127.0.0.1:8787/ui`
- health: `http://127.0.0.1:8787/health`

The model is loaded once when the server starts and remains resident.

## Browser playground

The `/ui` page can build and send all public decision request types without manually writing JSON:

- `Noul` — context, question, direct/serial mode
- `Choice` — context, question, dynamic option list, direct/serial mode
- `Shared` — one context and multiple dynamic decisions/options
- Text or structured JSON state
- live generated request JSON
- probabilities, logits, token count, latency and cache status
- runtime statistics and recent requests
- runtime-cache clearing without unloading model weights

The cache button clears reusable prefix state and backend allocator/context caches where supported. It does not delete model files or the Hugging Face disk cache.

## API examples

Binary decision:

```bash
curl -s \
  -X POST http://127.0.0.1:8787/v1/noul \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "The cookie contains sugar, chocolate and vanilla.",
    "question": "Is the cookie sweet?"
  }'
```

Choice decision:

```bash
curl -s \
  -X POST http://127.0.0.1:8787/v1/choice \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "The customer cannot access their account.",
    "question": "Which team should handle this request?",
    "options": [
      {"id": "access", "description": "Account access support."},
      {"id": "billing", "description": "Billing support."}
    ]
  }'
```

Clear runtime cache:

```bash
curl -s -X POST http://127.0.0.1:8787/v1/cache/clear
```

## Development

Install development dependencies:

```bash
uv sync --frozen --extra dev
```

For llama.cpp development:

```bash
uv sync --frozen --extra dev --extra llamacpp
```

Validate:

```bash
uv run python -m compileall -q src tests
uv run pytest
```

Repository changes should follow `AGENTS.md`.

## Updating SemIf

SemIf is pinned to an immutable Git revision in `pyproject.toml`.

When intentionally upgrading it:

1. change the `rev` value under `[tool.uv.sources]`;
2. run `uv lock`;
3. run the full test suite;
4. start each backend available on the test machine;
5. test `noul`, `choice`, `shared`, and cache clearing;
6. compare representative probabilities and latency;
7. commit both `pyproject.toml` and `uv.lock`.

## License

This project is licensed under the MIT License. See `LICENSE`.

SemIf and model weights are separate upstream works and remain subject to their own licenses and terms.
