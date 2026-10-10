# Deqio

**Decision in, probabilities out.**

Deqio runs decision models that return structured answers and probabilities instead of free-form text. You provide a **state** (text or JSON), ask a typed question, and receive a constrained decision such as yes/no, one option, an ordered score, multiple labels, or a cost-aware action.

Deqio keeps model/runtime differences behind one local server, adds provenance and observability, and provides a browser UI, live Watch view, benchmarks, and benchmark comparison.

## Why use Deqio?

A normal generative LLM might answer:

> "I think the refund should probably be approved."

A decision model can return:

```json
{
  "choice": "approve",
  "probabilities": {
    "approve": 0.91,
    "review": 0.07,
    "reject": 0.02
  }
}
```

That makes Deqio useful for routing, policy checks, guardrails, agent next-step selection, triage, scoring, and other workflows where the allowed answers are known in advance.

---

## API at a glance

Deqio exposes three layers of decision API.

### Portable Deqio endpoints

These are the stable cross-model Deqio contracts:

```text
POST /v1/noul      yes / no
POST /v1/choice    exactly one option
POST /v1/shared    several Deqio decisions over one shared state
```

`POST /v1/decision` remains a compatibility alias for `/v1/choice`.

### Native typed endpoints

These are thin convenience wrappers over the active model's native SystemOne implementation:

```text
POST /v1/score     ordered score / expected level
POST /v1/multi     multiple independent labels
POST /v1/act       cost-aware action selection
```

They are available only when the active profile advertises the matching native capability. Unsupported features fail explicitly.

### SOAM / raw SystemOne

```text
POST /v1/soam       friendly State Once, Ask Many entrypoint
POST /v1/systemone  raw canonical native SystemOne contract
```

Both routes use the **same native SystemOne execution path**. `/v1/soam` is not a second inference protocol, does not add another model call, and does not split the request into separate decisions. It exists to make the common **one state + many typed questions** pattern explicit and easier to discover.

The practical distinction is intent:

- use **`/v1/soam`** in agent orchestration, workflow validation, and other applications where one shared state should be evaluated by several questions in one native request;
- use **`/v1/systemone`** when you want the raw/canonical SystemOne contract directly, for low-level integrations, adapters, tests, or exact upstream-shaped requests.

Today both accept the same `state`, `questions`, and request modifiers, and both preserve the same native response fields. Changing between them changes the public route/intent recorded by Deqio, not the underlying decision algorithm. A multi-question request sent to `/v1/systemone` is still SOAM-shaped execution.

`facts`, `evidence`, and `option_keys` are **modifiers**, not standalone decision endpoints:

- `facts: "auto"` — request-level deterministic-facts support where the runtime provides it.
- `evidence: true` — per-question evidence spans where supported.
- `option_keys: "show" | "hide"` — per-question option-key presentation where supported.

Basal 1.5 currently exposes the broadest native feature set in Deqio: Choice, Noul, Score, Multi, Act, Facts, SOAM, option keys, and Evidence on native PyTorch profiles that support it.

---

## Supported models and backends

The table below is generated from the model profiles shipped in the current Deqio 0.5 catalog. A check mark means Deqio has a profile for that model/backend combination; actual availability still depends on the operating system, accelerator, memory, and upstream runtime requirements.

| Model | Model ID | MLX | MPS | CUDA | GGUF |
| --- | --- | :---: | :---: | :---: | :---: |
| SemIf / Qwen3.5 4B | `semif-qwen3.5-4b` | ✓ 8-bit | ✓ | ✓ | — |
| Decision 2.0 Kai 0.6B | `decision-2.0-kai` | — | — | ✓ | — |
| Decision 2.0 Eos 0.8B | `decision-2.0-eos` | — | — | ✓ | — |
| Decision 2.0 Sol 2B | `decision-2.0-sol` | — | — | ✓ | — |
| Decision 2.0 Nox 4B | `decision-2.0-nox` | — | — | ✓ | — |
| Decision 2.0 Lux 9B | `decision-2.0-lux` | — | — | ✓ | — |
| Decision 2.0 Vega 27B | `decision-2.0-vega` | — | — | ✓ | — |
| Kev 0.8B | `kev-0.8b` | ✓ | ✓ | ✓ | ✓ Q8_0 |
| Kev 4B | `kev-4b` | ✓ | ✓ | ✓ | ✓ Q8_0 |
| Kev 9B | `kev-9b` | ✓ | ✓ | ✓ | ✓ Q8_0 |
| Kev 27B | `kev-27b` | — | — | ✓ | — |
| JevK5 4B | `jevk5-4b` | — | — | ✓ | ✓ Q8_0 |
| JevK5 9B | `jevk5-9b` | — | — | ✓ | ✓ Q8_0 |
| Open-Jev 2B | `open-jev-2b` | — | — | ✓ | — |
| Open-Jev 9B | `open-jev-9b` | — | — | ✓ | — |
| Open-Jev 27B v1.1 | `open-jev-27b-v1.1` | — | — | ✓ | — |
| CLM 8B | `clm-8b` | — | — | ✓ | — |
| Basal 1.5 Mini — 1.5B | `basal-1.5-mini` | ✓ 8-bit | ✓ | ✓ | ✓ Q8_0 |
| Basal 1.5 Main — 4.5B | `basal-1.5-main` | ✓ 8-bit | ✓ | ✓ | ✓ Q8_0 |
| Basal 1.5 Max — 11B | `basal-1.5-max` | ✓ 8-bit | ✓ | ✓ | ✓ Q8_0 |
| Clef Flash 9B | `clef-flash` | ✓ 8-bit | — | — | ✓ Q8_0 |
| Clef 27B | `clef` | ✓ 8-bit | — | — | ✓ Q8_0 |
| Decider 0.8B | `decider-0.8b` | — | ✓ | ✓ | — |
| Decider 2B | `decider-2b` | — | ✓ | ✓ | ✓ Q8_0 |
| Decider 4B | `decider-4b` | — | ✓ | ✓ | ✓ Q8_0 |
| Laya English 421M | `laya-english` | ✓ | ✓ | ✓ | ✓ Q8_0 |
| Laya Multilingual 322M | `laya-multilingual` | ✓ | ✓ | ✓ | — |
| Laya Typed Decisions 421M | `laya-typed-decisions` | ✓ | ✓ | ✓ | — |
| Von | `von` | — | ✓ | ✓ | — |
| Bespoke Nimble 9B | `nimble-9b` | ✓ | — | ✓ | — |

- `✓ 8-bit` marks the cataloged MLX 8-bit path; `✓ Q8_0` marks the accepted GGUF Q8_0 profile.
- Decision 2.0 profiles are intentionally **CUDA-only** in Deqio 0.5 and require the official CUDA runtime.
- Basal 1.5 spans all four backend classes (MLX 8-bit, MPS, GGUF Q8_0, CUDA) and currently exposes Deqio's broadest native feature set.
- Backend presence does not imply identical capabilities. Deqio checks each profile at runtime and rejects unsupported native features explicitly.
- Run `deqio models list` for the full catalog and `deqio models installed` for profiles available locally.

## Benchmark snapshot

The snapshots below come from completed Deqio 0.5 runs on **2026-10-06** using the same local Apple Silicon macOS / 16 GiB setup. Small and large models were benchmarked in separate runs so each group could use the same machine without competing for memory.

The ENG and PL suites differ in size, difficulty, domains, and question composition, so ENG-vs-PL deltas are descriptive differences between the listed runs — **not** a pure measurement of language quality.

### Small models (up to 2.5B)

#### Runs

- **ENG:** run `20261006T082750Z`, suite `deqio-basic-150` — 150 cases / 250 scored decisions.
- **PL:** run `20261006T083557Z`, suite `deqio-pl-60` — 60 cases / 93 scored decisions.

All six profiles below completed both runs with **0 runtime errors**.

#### Quality

`Accuracy` is whole-case accuracy. `Decision accuracy` scores individual decisions/assertions, which matters especially for shared-state cases containing several decisions.

| Model | Backend | ENG accuracy | ENG decision accuracy | PL accuracy | PL decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: |
| **Decider 2B** | MPS | **90.7%** | **94.0%** | **91.7%** | **93.5%** |
| **Basal 1.5 Mini 1.5B** | MLX 8-bit | 84.7% | 90.0% | 83.3% | 89.2% |
| **Decider 0.8B** | MPS | 84.0% | 88.8% | 75.0% | 83.9% |
| **Kev 0.8B** | MLX | 76.7% | 84.0% | 66.7% | 78.5% |
| **Von** | MPS | 64.0% | 73.2% | 51.7% | 62.4% |
| **Laya Typed Decisions 421M** | MLX | 62.0% | 72.4% | 31.7% | 44.1% |

The quality ranking is stable across both runs: **Decider 2B → Basal 1.5 Mini → Decider 0.8B → Kev 0.8B → Von → Laya Typed Decisions**.

#### Performance

Throughput is completed scored decisions per second for the whole run. Lower latency is better; higher throughput is better.

| Model | ENG median | PL median | ENG P95 | PL P95 | ENG throughput | PL throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decider 2B | 301.6 ms | 334.3 ms | 578.6 ms | 512.2 ms | 5.14/s | 4.96/s |
| Basal 1.5 Mini 1.5B | 403.8 ms | 318.9 ms | 880.9 ms | 572.9 ms | 3.28/s | 4.29/s |
| Decider 0.8B | 310.6 ms | 316.6 ms | 369.1 ms | 403.2 ms | 5.72/s | 4.87/s |
| Kev 0.8B | 58.3 ms | 59.5 ms | 94.3 ms | 92.5 ms | 25.38/s | 23.85/s |
| Von | 61.7 ms | 75.7 ms | 116.2 ms | 138.9 ms | 21.29/s | 18.83/s |
| Laya Typed Decisions 421M | **32.9 ms** | **36.5 ms** | **70.6 ms** | 99.2 ms | **42.87/s** | **30.99/s** |

#### What this small-model snapshot shows

- **Decider 2B is the quality-first small-model leader in this snapshot.** It is #1 in whole-case accuracy on both suites (90.7% ENG / 91.7% PL) and stays near 94% decision accuracy in both languages/suites.
- **Basal 1.5 Mini is the strongest broader-SystemOne profile in this group.** It ranks #2 on both suites while exposing native capabilities beyond the basic benchmark shape, including Multi, Act, Facts and SOAM on the MLX profile. These ENG/PL suites exercise Noul, Choice and shared-state decisions; they do **not** by themselves benchmark Basal's Multi/Act/Facts capabilities.
- **Decider 0.8B remains strong for its size**, especially on ENG (84.0% case / 88.8% decision accuracy), but drops to 75.0% case accuracy on the PL suite.
- **Kev 0.8B stands out as a speed/quality compromise:** about 58–60 ms median latency and 23.85–25.38 decisions/s while retaining 76.7% ENG and 66.7% PL case accuracy.
- **Laya Typed Decisions is the latency/throughput specialist:** it is the fastest profile here, but its quality is much lower on this PL suite (31.7% case / 44.1% decision accuracy). Use it when latency matters more than maximum decision quality and validate against your own domain.
- **Von sits between the ultra-fast and quality-first profiles:** 61.7–75.7 ms median latency with 64.0% ENG / 51.7% PL case accuracy.
- **Runtime stability was clean for this snapshot:** every listed profile recorded zero runtime errors in both runs.

Per-type results also show why a single aggregate number is not enough. Decider 2B stays unusually even across Choice, Noul and Shared; Basal Mini is very strong on ENG Choice and PL Noul while keeping Shared decision accuracy around 90%; Laya and Von vary much more strongly by suite and decision type. Use `deqio benchmark compare` for the full breakdown.

### Large models (4B–4.5B in this snapshot)

#### Runs

- **ENG:** run `20261006T090441Z`, suite `deqio-basic-150` — 150 cases / 250 scored decisions.
- **PL:** run `20261006T091845Z`, suite `deqio-pl-60` — 60 cases / 93 scored decisions.

All five profiles below completed both runs with **0 runtime errors**.

#### Quality

`Accuracy` is whole-case accuracy. `Decision accuracy` scores individual decisions/assertions, which matters especially for shared-state cases containing several decisions.

| Model | Backend | ENG accuracy | ENG decision accuracy | PL accuracy | PL decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: |
| **JevK5 4B** | GGUF Q8_0 | **96.7%** | **98.0%** | **98.3%** | **98.9%** |
| **Decider 4B** | MPS | 95.3% | 97.2% | 96.7% | 97.8% |
| **Basal 1.5 Main 4.5B** | MLX 8-bit | 93.3% | 96.0% | 95.0% | 96.8% |
| **SemIf / Qwen3.5 4B** | MLX 8-bit | 93.3% | 95.6% | 95.0% | 96.8% |
| **Kev 4B** | MLX | 92.7% | 95.2% | 96.7% | 97.8% |

**JevK5 4B** ranks first on whole-case and decision accuracy in both runs. The rest of the ordering is suite-dependent: Decider 4B is second on ENG, while Kev 4B rises to a tie with Decider on PL case and decision accuracy. Treat those changes as properties of these two benchmark suites, not as a pure language ranking.

#### Performance

Throughput is completed scored decisions per second for the whole run. Lower latency is better; higher throughput is better.

| Model | ENG median | PL median | ENG P95 | PL P95 | ENG throughput | PL throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Basal 1.5 Main 4.5B | 1,213.7 ms | 987.9 ms | 2,775.6 ms | 1,860.0 ms | 1.07/s | 1.37/s |
| Decider 4B | 436.7 ms | 438.9 ms | 1,361.1 ms | 1,330.6 ms | 2.54/s | 2.27/s |
| JevK5 4B Q8_0 | 594.9 ms | 570.6 ms | 1,864.5 ms | 1,817.6 ms | 1.76/s | 1.72/s |
| **Kev 4B** | **315.9 ms** | **358.8 ms** | **518.4 ms** | **541.2 ms** | **4.59/s** | **3.92/s** |
| SemIf / Qwen3.5 4B | 594.7 ms | 599.4 ms | 1,216.7 ms | 1,113.4 ms | 2.22/s | 2.19/s |

#### What this large-model snapshot shows

- **JevK5 4B Q8_0 is the quality leader in this snapshot.** It reaches 96.7% ENG / 98.3% PL case accuracy and 98.0% / 98.9% decision accuracy, with no runtime errors in either run.
- **Decider 4B is a strong quality/latency balance.** It stays at 95.3–96.7% case accuracy with roughly 437–439 ms median latency, and reaches 100% Noul accuracy in both suites.
- **Kev 4B is the fastest profile in this group.** Its median latency is about 316–359 ms with 3.92–4.59 decisions/s. It also scores 100% on both Choice and Noul in both runs; its lower aggregate score comes mainly from Shared cases.
- **Basal 1.5 Main is the broadest SystemOne profile in this group.** It scores 93.3% ENG / 95.0% PL case accuracy, but this basic benchmark measures Noul, Choice and Shared only. It does **not** measure Basal-specific Multi, Act, Facts, option keys or Evidence-capable native profiles, so the table should not be read as a complete comparison of Basal's feature value.
- **SemIf / Qwen3.5 4B is stable across both suites**, with 93.3% ENG / 95.0% PL case accuracy and about 595–599 ms median latency.
- **Runtime stability was clean:** all five profiles recorded zero runtime errors in both runs.

The decision-type breakdown adds useful context. JevK5 is consistently strong across Choice, Noul and Shared; Decider is perfect on Noul in both runs; Kev is perfect on Choice and Noul in both runs but weaker on Shared; Basal and SemIf remain strong but show more of their errors in Shared cases. Use `deqio benchmark compare` for the full per-type breakdown.

The bundled historical benchmark reports remain available for reference, but they may contain older model selections or measurements and should not be treated as the source for the fresh 2026-10-06 tables above:

- [ENG benchmark summary](benchmarks/summary_eng.md)
- [PL benchmark summary](benchmarks/summary_pl.md)

Run your own current comparison with:

```bash
deqio benchmark
deqio benchmark compare
```

---

## Installation

### Recommended: PyPI + `uv tool`

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install Deqio:

```bash
uv tool install deqio
```

Create a workspace:

```bash
mkdir -p deqio-work
cd deqio-work
```

Install a model/backend interactively:

```bash
deqio models setup
```

Start the server:

```bash
deqio serve
```

Then open:

```text
http://127.0.0.1:8787/ui
http://127.0.0.1:8787/ui/watch
http://127.0.0.1:8787/docs
```

Deqio is local-first and binds to `127.0.0.1` by default. It does not provide a public-network authentication layer; if you deliberately expose it on another interface, put an appropriate authenticated network/reverse-proxy boundary in front of it. Watch (`/ui/watch`, `/v1/watch*`) keeps the full request and response payloads of the current session, so on any non-loopback bind (for example `deqio serve --host 0.0.0.0`) it requires a per-server Watch token: set `DEQIO_WATCH_TOKEN`, or use the token printed in the startup banner. Scripts send it as `X-Deqio-Watch-Token` (or `Authorization: Bearer`); in a browser, open `/ui/watch?token=<token>` once and the token is kept in an HttpOnly, SameSite=Strict cookie. Decision routes, `/ui`, `/v1/recent` (recent questions and decisions) and `/health` stay unauthenticated, and without TLS the token travels in clear text, so the reverse-proxy advice above still applies.

**Supported platforms:** macOS on Apple Silicon (MLX, MPS, GGUF) and Linux (CUDA when `nvidia-smi -L` lists an NVIDIA GPU, GGUF). Python 3.10 and 3.12 are tested in CI. Windows is not supported in Deqio 0.5: model setup reports it as unsupported instead of offering profiles that were never validated there.

### Development install

From a source checkout:

```bash
uv sync --frozen --extra dev
uv run ruff check src tests
uv run pytest
uv run deqio models setup
uv run deqio serve
```

CI (`.github/workflows/ci.yml`) runs the same lint, byte-compile and test steps on every push and pull request with Python 3.10 and 3.12. Release notes live in [CHANGELOG.md](CHANGELOG.md); security reporting in [SECURITY.md](SECURITY.md).

---

# API examples

The server defaults to `http://127.0.0.1:8787`.

## 1. Noul — yes/no

```bash
curl -s http://127.0.0.1:8787/v1/noul \
  -H 'content-type: application/json' \
  -d '{
    "state": "The patch changed authentication code. Unit tests passed but integration tests have not run.",
    "question": "Should integration tests run before release?"
  }'
```

Use Noul when the answer is fundamentally **true/false** or **yes/no**.

---

## 2. Choice — exactly one option

```bash
curl -s http://127.0.0.1:8787/v1/choice \
  -H 'content-type: application/json' \
  -d '{
    "state": "A customer was charged twice for one subscription renewal.",
    "question": "Which team should handle this request?",
    "options": [
      {"id": "billing", "description": "Payments, refunds and invoices."},
      {"id": "support", "description": "Product and account support."},
      {"id": "sales", "description": "New purchases and upgrades."}
    ]
  }'
```

Use Choice when **one and only one** option should win.

---

## 3. Shared — several portable Deqio decisions over one state

```bash
curl -s http://127.0.0.1:8787/v1/shared \
  -H 'content-type: application/json' \
  -d '{
    "state": "A deployment passed unit tests but failed one integration test.",
    "decisions": [
      {
        "question": "What should happen next?",
        "options": [
          {"id": "retry", "description": "Investigate and rerun validation."},
          {"id": "release", "description": "Release immediately."}
        ]
      },
      {
        "question": "How should the build be classified?",
        "options": [
          {"id": "blocked", "description": "Not ready for release."},
          {"id": "ready", "description": "Ready for release."}
        ]
      }
    ]
  }'
```

`/v1/shared` is the stable Deqio multi-decision API. It is distinct from native SOAM: Shared uses Deqio's portable decision contract, while SOAM forwards one native SystemOne request to a capable runtime. One engine call scores the whole batch, so a per-decision `input_tokens` is `null` when the batch has more than one decision (negotiated receipts say `input_tokens_source: "engine_reported_batch"`); the batch count appears once in the negotiated `input_receipt` and in Watch.

Portable requests are validated before inference with `400`: questions and explicitly supplied IDs must not be blank, option and decision IDs must be unique, a decision has at most 128 options and a Shared request at most 64 decisions. Native `/v1/systemone` and `/v1/soam` requests accept at most 64 questions and are rejected with `422` beyond that.

---

## 4. Score — ordered levels

`/v1/score` is a native SystemOne convenience endpoint. The endpoint injects `type: "score"`; the `question` object contains the remaining native fields.

```bash
curl -s http://127.0.0.1:8787/v1/score \
  -H 'content-type: application/json' \
  -d '{
    "state": "Checkout works after retry, but payment latency is elevated for some customers.",
    "question": {
      "instructions": "Rate the operational severity.",
      "criteria": ["low", "medium", "high", "critical"]
    }
  }'
```

Use Score for an **ordered scale** such as severity, risk, urgency, or satisfaction.

---

## 5. Multi — several independent labels

```bash
curl -s http://127.0.0.1:8787/v1/multi \
  -H 'content-type: application/json' \
  -d '{
    "state": "The laptop arrived with a cracked screen, the box was damaged, and the customer asks for a refund.",
    "question": {
      "instructions": "Which labels apply?",
      "criteria": {
        "damage": "Physical product damage.",
        "delivery": "Delivery or courier problem.",
        "refund": "Customer requests a refund.",
        "repair": "Customer requests repair."
      },
      "threshold": 0.5,
      "max": 3
    }
  }'
```

Multi is **not Choice with more winners**. Each label has an independent probability, so the probabilities do not have to sum to 1.

---

## 6. Act — choose an action using outcome costs

```bash
curl -s http://127.0.0.1:8787/v1/act \
  -H 'content-type: application/json' \
  -d '{
    "state": "The refund is probably valid, but approving an invalid refund is expensive.",
    "question": {
      "instructions": "Choose the next action.",
      "criteria": {
        "true": "The refund is justified.",
        "false": "The refund is not justified."
      },
      "costs": {
        "approve": {"true": 0, "false": 1000},
        "reject": {"true": 200, "false": 0},
        "human": {"true": 20, "false": 20}
      }
    }
  }'
```

Act is useful when the **cost of a mistake matters**. The native runtime remains the source of the final action; Deqio does not recompute it.

---

## 7. SOAM — State Once, Ask Many

Use `/v1/soam` when several typed questions should share one state in one native request. This is the recommended high-level route for agent/checkpoint workflows such as release validation, incident triage, policy checks, and orchestration: prepare one confirmed state, then ask for several independent decision views without re-sending or re-processing that state as unrelated requests.

`/v1/soam` is a **convenience alias over the native SystemOne contract**, not a separate inference engine. Deqio forwards the complete `state + questions` package through the same `SystemOneRuntime.system_one(...)` path used by `/v1/systemone`.

```bash
curl -s http://127.0.0.1:8787/v1/soam \
  -H 'content-type: application/json' \
  -d '{
    "state": "The laptop arrived with a cracked screen and the customer requests a refund.",
    "facts": "auto",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which workflow should handle this request?",
        "criteria": {
          "refund": "Refund workflow.",
          "repair": "Repair workflow.",
          "review": "Human review."
        },
        "option_keys": "show"
      },
      "damaged": {
        "type": "noul",
        "instructions": "Is the item damaged?"
      },
      "severity": {
        "type": "score",
        "instructions": "Rate severity.",
        "criteria": ["low", "medium", "high"]
      }
    }
  }'
```

The raw equivalent is `POST /v1/systemone` with the same request body. The model/runtime result is the same kind of native SystemOne response; the separate `/v1/soam` route mainly communicates that the caller intentionally uses the State Once, Ask Many pattern.

### Facts

`facts: "auto"` is a **request modifier**, not `/v1/facts`. It is forwarded only when the active native runtime advertises Facts support.

### Option keys

`option_keys: "show" | "hide"` is a **question modifier**, not a separate endpoint.

### Evidence

For a supporting backend, set:

```json
{
  "type": "noul",
  "instructions": "Is the item damaged?",
  "evidence": true
}
```

Evidence offsets are preserved from the native runtime. Unsupported backends return a capability error instead of silently ignoring the request.

---

## 8. Raw SystemOne

Advanced clients can call the native contract directly. Use this route when the integration should speak in canonical SystemOne terms rather than the friendlier SOAM naming:

```text
POST /v1/systemone
```

The request shape is:

```json
{
  "state": "text or JSON",
  "questions": {
    "name": {
      "type": "choice | noul | score | multi | act",
      "instructions": "..."
    }
  },
  "facts": "off | auto"
}
```

Deqio preserves native `model`, `answers`, `usage`, probabilities, selected values, score semantics, expected costs, evidence offsets, and other upstream fields. Deqio metadata is added under the separate `deqio` namespace.

### `/v1/soam` vs `/v1/systemone`

| Question | `/v1/soam` | `/v1/systemone` |
| --- | --- | --- |
| Main purpose | Clear application-level State Once, Ask Many entrypoint | Raw/canonical SystemOne entrypoint |
| Request shape | `state` + `questions` + supported modifiers | `state` + `questions` + supported modifiers |
| Multi-question support | Yes; this is the intended use | Yes; the same shape is valid |
| Runtime path | Native `SystemOneRuntime.system_one(...)` | Native `SystemOneRuntime.system_one(...)` |
| Extra inference layer | No | No |
| Response rewriting | No; native fields are preserved | No; native fields are preserved |
| Recommended for | agents, workflow orchestration, application workflows | low-level/native integrations, adapters, protocol tests |

If you are unsure which one to use, choose `/v1/soam` for a normal multi-question application workflow and `/v1/systemone` when you specifically need the canonical raw SystemOne route.

---

## Capability-aware API

Native endpoints depend on the active profile. Inspect installed profiles and capabilities with:

```bash
deqio models installed
```

The browser UI shows the same capability matrix. For example:

- a profile with Choice/Noul/Score but no Multi will reject `/v1/multi`;
- Basal MLX/GGUF supports many Basal 1.5 features but not Evidence;
- native Basal MPS/CUDA profiles expose Evidence where the upstream runtime supports it;

No unsupported feature is silently ignored.

---

## Browser UI and Watch

Start the server:

```bash
deqio serve
```

Main dashboard:

```text
http://127.0.0.1:8787/ui
```

The dashboard provides:

- installed model/backend switching;
- capability-aware Noul/Choice/Shared/Score/Multi/Act/SOAM playgrounds;
- generated request JSON;
- native response inspection;
- benchmark Summary / Results / Compare views.

Live request/benchmark view:

```text
http://127.0.0.1:8787/ui/watch
```

Watch remains available while a benchmark temporarily suspends the server's inference runtime.

Watch is intentionally observability-first: it stores the full request and response payloads for the current session under `.deqio/watch/`. Treat that directory as potentially sensitive application data. The in-memory latency history is bounded, event files rotate at 10,000 records or 16 MiB, total retained Watch event storage is capped at 256 MiB per session, and Watch can be cleared manually or configured for automatic session cleanup.

---

## Benchmarks

Run a benchmark interactively, or select everything up front:

```bash
deqio benchmark                                   # choose suite and profiles interactively
deqio benchmark --suite benchmarks/pl.json        # one suite file (default: pick from ./benchmarks/*.json)
deqio benchmark --model decider-2b:mps --model basal-1.5-mini:mlx   # repeatable MODEL_ID:BACKEND
deqio benchmark --all                             # every installed, host-compatible profile
deqio benchmark --all --output runs/today         # results directory (default: .deqio/benchmarks/<timestamp>)
```

`--all` and `--model` are mutually exclusive; `--config PATH` selects another workspace `config.json`. The benchmark gets exclusive inference ownership for the workspace. If `deqio serve` is running, the HTTP server and `/ui/watch` stay alive while its active model is temporarily unloaded. The previous server model is restored when the benchmark completes.

Compare two completed runs:

```bash
deqio benchmark compare                                       # interactive
deqio benchmark compare --left RUN_A --right RUN_B            # non-interactive
deqio benchmark compare --left RUN_A --right RUN_B --json     # JSON for agents/automation
deqio benchmark compare --left RUN_A --right RUN_B --output comparison.json
```

Deqio only compares profiles whose canonical runtime identity matches, including model, backend, precision/quantization, relevant runtime identity, and pinned artifact revisions. Profiles whose runtime failed to load in a run are listed separately (`load_failed_left` / `load_failed_right` in the JSON) and never compared.

---

## Correctness and provenance

Decision responses include Deqio runtime/model provenance. Installed profiles record artifact revisions where available so a result can be tied to the actual runtime/model identity used for inference.

Deqio also supports the optional input-completeness negotiation header:

```text
Deqio-Contract: input-completeness-v1
```

This mode is fail-closed: when a runtime cannot prove the required model-boundary completeness, Deqio rejects the request instead of fabricating certainty.

---

## Useful commands

```bash
deqio --version                                   # version
deqio models setup                                # interactive model installation
deqio models list                                 # catalog
deqio models installed                            # installed profiles
deqio models status                               # active profile
deqio models use MODEL_ID --backend BACKEND       # switch installed profile
deqio models install MODEL_ID --backend BACKEND   # install / update / delete
deqio models update MODEL_ID --backend BACKEND
deqio models delete MODEL_ID --backend BACKEND
deqio serve                                       # server (127.0.0.1:8787)
deqio benchmark                                   # benchmark, see "Benchmarks" for flags
deqio benchmark compare
```

---

## Design principles

Deqio 0.5 prefers correctness over pretending that every backend supports every feature:

- official/authoritative model and runtime paths;
- every engine runtime pinned to an exact version or commit, Hub artifacts to commit revisions (remaining exceptions are tracked in tests);
- every profile labeled `official` or `community`;
- no silent CPU/backend fallback;
- no silent capability degradation;
- native decision probabilities and semantics are preserved;
- one active inference owner per workspace during benchmark/server coordination;
- community conversions are not presented as official Deqio profiles.

## License

MIT. See [LICENSE](LICENSE).
