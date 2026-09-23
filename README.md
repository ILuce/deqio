# SemIf Local Server

Small local HTTP server for [SemIf](https://github.com/TheoLeeCJ/SemIf).

The server keeps a SemIf decision model loaded in memory and exposes typed semantic decisions through a simple local HTTP API. It returns option probabilities and logits without generating an answer text.

## Current status

This release uses the native **MLX backend** and is intended for **macOS on Apple Silicon**.

SemIf also contains Torch/CUDA and llama.cpp paths upstream, but this server does not expose those backends yet. Linux and Windows support should be added only after a backend abstraction is implemented and tested here.

## Features

- `noul` — binary yes/no decisions
- `choice` — decisions between arbitrary options
- `shared` — multiple decisions over one exact state
- serial prefix-cache reuse
- request latency and probability statistics
- lightweight browser dashboard
- JSONL request logs
- FastAPI/OpenAPI interface

## Requirements

- macOS on Apple Silicon
- Git
- [uv](https://docs.astral.sh/uv/)

The project pins Python 3.12 through `.python-version`.

## Install from scratch

Clone the repository and enter it:

```bash
git clone https://github.com/ILuce/semif-local.git
cd semif-local-server
```

Install the locked runtime:

```bash
uv sync --frozen
```

For development and tests:

```bash
uv sync --frozen --extra dev
```

No manual virtual-environment activation is required. Use `uv run ...` for project commands.

## Download the model

The default configuration expects the MLX 4-bit model in `models/semif-qwen3.5-4b-mlx-4bit`.

```bash
mkdir -p models

uv run hf download \
  vinci00/semif-qwen3.5-4b-mlx-4bit \
  --local-dir models/semif-qwen3.5-4b-mlx-4bit
```

Model weights are intentionally excluded from Git.

## Run

```bash
uv run semif-server
```

Default addresses:

- server: `http://127.0.0.1:8787`
- OpenAPI: `http://127.0.0.1:8787/docs`
- dashboard: `http://127.0.0.1:8787/ui`
- health: `http://127.0.0.1:8787/health`

The model is loaded once when the server starts and stays resident for subsequent requests.

## Quick test

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
      {
        "id": "access",
        "description": "Account access support."
      },
      {
        "id": "billing",
        "description": "Billing support."
      }
    ]
  }'
```

`POST /v1/decision` is kept as a compatibility alias for `POST /v1/choice`.

## Configuration

The following environment variables are supported:

- `SEMIF_MODEL` — local model directory
- `SEMIF_MODEL_REVISION` — provenance label for a local model
- `SEMIF_MAX_TOKENS` — maximum SemIf input size, default `4096`
- `SEMIF_MLX_CACHE_MIB` — MLX inactive-allocation cache limit, default `256`
- `SEMIF_LOG` — JSONL request log path, default `logs/requests.jsonl`

Example:

```bash
SEMIF_MLX_CACHE_MIB=512 uv run semif-server
```

## Development

Install development dependencies:

```bash
uv sync --frozen --extra dev
```

Validate the source:

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
4. start the server and test the decision endpoints;
5. commit both `pyproject.toml` and `uv.lock`.

Do not track the upstream branch implicitly.

## License

This project is licensed under the MIT License. See `LICENSE`.

SemIf and the model weights are separate upstream works and remain subject to their own licenses and terms.
