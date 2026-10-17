# PL Bench — Deqio benchmark summary (0.6.2 snapshot + historical 0.5 tables)

> **Suite:** `deqio-pl-60` — 60 requests · 93 scored decisions · 20 Noul / 20 Choice / 20 Shared per profile<br>
> **Host:** Apple Silicon macOS · 16.0 GiB unified memory · one profile resident at a time<br>
> **0.6.2 runs (2026-10-17, Deqio 0.6.2):** up to 1B `20261017T105442Z` · 1–3B `20261017T110819Z` · above 3B `20261017T121613Z`<br>
> **Remote APIs:** `20261015T120300Z` (2026-10-15, Deqio 0.6.1; the remote profiles are unchanged in 0.6.2)<br>
> **Historical 0.5 runs (2026-10-06):** small `20261006T083557Z` · large `20261006T091845Z`<br>
> **Execution errors:** 0 in every listed run

PL Bench is a dedicated Polish-native suite rather than a mechanical translation of ENG Bench. The ENG and PL suites differ in size and scenario composition, so cross-suite deltas should not be interpreted as a pure language-quality measurement.

## 0.6.2 snapshot

Rendered by `deqio benchmark tables` from the committed `runs/small-pl`, `runs/medium-pl` and `runs/max-pl`
summaries (`deqio benchmark tables --summary runs/small-pl --title "Models up to 1B — PL Bench (0.6.2 snapshot)"`);
`tests/test_benchmark_snapshot.py` keeps this file and the README equal to that output. The remote-API tables come
from the owner's 2026-10-15 run (its summary is not committed). Rows are sorted by decision accuracy; the Decision 3.0
MLX profiles removed in 0.6.2 do not appear — the runs were made after the removal.

### Models up to 1B — PL Bench (0.6.2 snapshot)

#### Run

- `20261017T105442Z` · suite `deqio-pl-60` — 60 cases / 93 scored decisions · Deqio 0.6.2 · 2026-10-17
- runtime errors across the listed profiles: 0

#### Quality

`Accuracy` is whole-case accuracy; `Decision accuracy` scores every decision/assertion (a Shared case holds several).

| Model | Backend | Accuracy | Decision accuracy | Noul | Choice | Shared | Shared decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision 3.0 d3-lite 0.85B | GGUF Q8_0 | 85.0% | 90.3% | 95.0% | 80.0% | 80.0% | 92.5% |
| Decider 0.8B | MPS | 75.0% | 83.9% | 75.0% | 75.0% | 75.0% | 90.6% |
| Kev 0.8B | MLX | 66.7% | 78.5% | 80.0% | 70.0% | 50.0% | 81.1% |
| Von | MPS | 51.7% | 62.4% | 50.0% | 70.0% | 35.0% | 64.2% |
| Laya English 421M | MLX | 38.3% | 53.8% | 45.0% | 55.0% | 15.0% | 56.6% |

#### Performance

Median and P95 are per-request latencies; throughput is scored decisions per second over the whole run; cost is the estimated USD per 1 000 scored decisions from the API's reported input tokens and its published input price (remote profiles only; an estimate, not a bill).

| Model | Backend | Median | P95 | Throughput | Cost / 1k decisions |
| --- | :---: | ---: | ---: | ---: | ---: |
| Decision 3.0 d3-lite 0.85B | GGUF Q8_0 | 83.2 ms | 244.3 ms | 12.35/s | — |
| Decider 0.8B | MPS | 299.8 ms | 390.0 ms | 5.13/s | — |
| Kev 0.8B | MLX | 60.2 ms | 94.1 ms | 23.07/s | — |
| Von | MPS | 75.3 ms | 140.0 ms | 19.44/s | — |
| Laya English 421M | MLX | 39.5 ms | 102.7 ms | 29.69/s | — |

#### Identity

| Model | Backend | Engine | Runtime | Precision | Source | Model / endpoint | Revision |
| --- | :---: | --- | --- | --- | --- | --- | --- |
| Decision 3.0 d3-lite 0.85B | GGUF Q8_0 | decision3 | llama.cpp-systemone | q8_0 | official | `ggml-org/vllm-d3-lite-GGUF` | a1f9fa12 |
| Decider 0.8B | MPS | decider | — | — | official | `Mapika/decider-0.8b` | a0a01d6f |
| Kev 0.8B | MLX | kev | — | — | official | `jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f` | bf75a6a8 |
| Von | MPS | von | — | — | official | `von` | 498ceba3 |
| Laya English 421M | MLX | laya | — | — | community | `aac6fef/laya-mlx` | 04767856 |

### Models from 1B to 3B — PL Bench (0.6.2 snapshot)

#### Run

- `20261017T110819Z` · suite `deqio-pl-60` — 60 cases / 93 scored decisions · Deqio 0.6.2 · 2026-10-17
- runtime errors across the listed profiles: 0

#### Quality

`Accuracy` is whole-case accuracy; `Decision accuracy` scores every decision/assertion (a Shared case holds several).

| Model | Backend | Accuracy | Decision accuracy | Noul | Choice | Shared | Shared decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision 3.0 d3-nano 2.2B | GGUF Q8_0 | 91.7% | 94.6% | 95.0% | 95.0% | 85.0% | 94.3% |
| Decider 2B | MPS | 91.7% | 93.5% | 95.0% | 95.0% | 85.0% | 92.5% |
| Basal 2.0 Mini — 1.5B | MLX 8-bit | 88.3% | 92.5% | 85.0% | 95.0% | 85.0% | 94.3% |

#### Performance

Median and P95 are per-request latencies; throughput is scored decisions per second over the whole run; cost is the estimated USD per 1 000 scored decisions from the API's reported input tokens and its published input price (remote profiles only; an estimate, not a bill).

| Model | Backend | Median | P95 | Throughput | Cost / 1k decisions |
| --- | :---: | ---: | ---: | ---: | ---: |
| Decision 3.0 d3-nano 2.2B | GGUF Q8_0 | 173.7 ms | 520.2 ms | 5.87/s | — |
| Decider 2B | MPS | 189.3 ms | 524.4 ms | 5.76/s | — |
| Basal 2.0 Mini — 1.5B | MLX 8-bit | 181.5 ms | 291.6 ms | 7.89/s | — |

#### Identity

| Model | Backend | Engine | Runtime | Precision | Source | Model / endpoint | Revision |
| --- | :---: | --- | --- | --- | --- | --- | --- |
| Decision 3.0 d3-nano 2.2B | GGUF Q8_0 | decision3 | llama.cpp-systemone | q8_0 | official | `ggml-org/vllm-d3-nano-GGUF` | 67baf946 |
| Decider 2B | MPS | decider | — | — | official | `Mapika/decider-2b` | ff99089b |
| Basal 2.0 Mini — 1.5B | MLX 8-bit | basal | basal-2.0-mlx 2.0.0 | mlx-8bit | official | `Remek/basal-2.0-mini-MLX-8bit` | f26d1e2d |

### Models above 3B — PL Bench (0.6.2 snapshot)

#### Run

- `20261017T121613Z` · suite `deqio-pl-60` — 60 cases / 93 scored decisions · Deqio 0.6.2 · 2026-10-17
- runtime errors across the listed profiles: 0

#### Quality

`Accuracy` is whole-case accuracy; `Decision accuracy` scores every decision/assertion (a Shared case holds several).

| Model | Backend | Accuracy | Decision accuracy | Noul | Choice | Shared | Shared decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision 3.0 d3-mini 4.5B | GGUF Q8_0 | 98.3% | 98.9% | 100.0% | 100.0% | 95.0% | 98.1% |
| JevK5 4B | GGUF Q8_0 | 98.3% | 98.9% | 100.0% | 100.0% | 95.0% | 98.1% |
| Decider 4B | GGUF Q8_0 | 96.7% | 97.8% | 100.0% | 95.0% | 95.0% | 98.1% |
| Kev 4B | MLX | 96.7% | 97.8% | 100.0% | 100.0% | 90.0% | 96.2% |
| Basal 2.0 — 4.5B | MLX 8-bit | 91.7% | 94.6% | 95.0% | 95.0% | 85.0% | 94.3% |

#### Performance

Median and P95 are per-request latencies; throughput is scored decisions per second over the whole run; cost is the estimated USD per 1 000 scored decisions from the API's reported input tokens and its published input price (remote profiles only; an estimate, not a bill).

| Model | Backend | Median | P95 | Throughput | Cost / 1k decisions |
| --- | :---: | ---: | ---: | ---: | ---: |
| Decision 3.0 d3-mini 4.5B | GGUF Q8_0 | 487.1 ms | 1,465.3 ms | 2.10/s | — |
| JevK5 4B | GGUF Q8_0 | 618.2 ms | 1,875.3 ms | 1.67/s | — |
| Decider 4B | GGUF Q8_0 | 283.7 ms | 854.4 ms | 3.55/s | — |
| Kev 4B | MLX | 354.1 ms | 554.4 ms | 3.95/s | — |
| Basal 2.0 — 4.5B | MLX 8-bit | 560.1 ms | 913.2 ms | 2.58/s | — |

#### Identity

| Model | Backend | Engine | Runtime | Precision | Source | Model / endpoint | Revision |
| --- | :---: | --- | --- | --- | --- | --- | --- |
| Decision 3.0 d3-mini 4.5B | GGUF Q8_0 | decision3 | llama.cpp-systemone | q8_0 | official | `ggml-org/vllm-d3-mini-GGUF` | c03ac9c3 |
| JevK5 4B | GGUF Q8_0 | jevk5 | jevk5-gguf-native | q8_0 | official | `alibiserikbay/JevK5-GGUF` | ec67b0bf |
| Decider 4B | GGUF Q8_0 | decider | decider-gguf-native | q8_0 | official | `Mapika/decider-4b-GGUF` | b79f09d9 |
| Kev 4B | MLX | kev | — | — | official | `jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52` | 6cfce5c2 |
| Basal 2.0 — 4.5B | MLX 8-bit | basal | basal-2.0-mlx 2.0.0 | mlx-8bit | official | `Remek/basal-2.0-4.5B-MLX-8bit` | 77d3b723 |

### Remote decision APIs — PL Bench (0.6.2 snapshot)

#### Run

- `20261015T120300Z` · suite `deqio-pl-60` — 60 cases / 93 scored decisions · Deqio 0.6.1 · 2026-10-15
- runtime errors across the listed profiles: 0

#### Quality

`Accuracy` is whole-case accuracy; `Decision accuracy` scores every decision/assertion (a Shared case holds several).

| Model | Backend | Accuracy | Decision accuracy | Noul | Choice | Shared | Shared decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TypeSafe Jev (remote) | remote | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| OpenAI Decisions API (remote) | remote | 98.3% | 98.9% | 100.0% | 100.0% | 95.0% | 98.1% |

#### Performance

Median and P95 are per-request latencies; throughput is scored decisions per second over the whole run; cost is the estimated USD per 1 000 scored decisions from the API's reported input tokens and its published input price (remote profiles only; an estimate, not a bill).

| Model | Backend | Median | P95 | Throughput | Cost / 1k decisions |
| --- | :---: | ---: | ---: | ---: | ---: |
| TypeSafe Jev (remote) | remote | 317.7 ms | 424.8 ms | 4.64/s | $0.0108 |
| OpenAI Decisions API (remote) | remote | 314.1 ms | 487.7 ms | 4.56/s | $0.0156 |

#### Identity

| Model | Backend | Engine | Runtime | Precision | Source | Model / endpoint | Revision |
| --- | :---: | --- | --- | --- | --- | --- | --- |
| TypeSafe Jev (remote) | remote | typesafe-jev | — | — | official | `jev-latest` · `https://api.typesafe.ai/v1/systemone` | — |
| OpenAI Decisions API (remote) | remote | openai-decisions | — | — | official | `gpt-6-luna` · `https://api.openai.com/v1/decisions` | — |

## Historical 0.5 snapshot (2026-10-06)

The tables below come from completed Deqio 0.5 runs on 2026-10-06 on the same host, kept for reference only: Deqio,
the runtimes and some pins changed since (Basal 1.5 was replaced by Basal 2.0 in 0.6.1), so they are not comparable
with the 0.6.2 tables above. Small and large models were benchmarked in separate runs so each group could use the same
machine without competing for memory.

### Small models

Run `20261006T083557Z` · `deqio-pl-60` · 60 cases. All 6 profiles completed with **0 runtime errors**.

#### Overall results

| Model | Backend | Accuracy | Decision accuracy | Median | P95 | Throughput |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Decider 2B | MPS | 91.7% | 93.5% | 334.3 ms | 512.2 ms | 4.96/s |
| Basal 1.5 Mini — 1.5B | MLX 8-bit | 83.3% | 89.2% | 318.9 ms | 572.9 ms | 4.29/s |
| Decider 0.8B | MPS | 75.0% | 83.9% | 316.6 ms | 403.2 ms | 4.87/s |
| Kev 0.8B | MLX | 66.7% | 78.5% | 59.5 ms | 92.5 ms | 23.85/s |
| Von | MPS | 51.7% | 62.4% | 75.7 ms | 138.9 ms | 18.83/s |
| Laya Typed Decisions 421M | MLX | 31.7% | 44.1% | 36.5 ms | 99.2 ms | 30.99/s |

#### Per-type case accuracy

| Model | Backend | Noul | Choice | Shared | Shared decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: |
| Decider 2B | MPS | 95.0% | 95.0% | 85.0% | 92.5% |
| Basal 1.5 Mini — 1.5B | MLX 8-bit | 90.0% | 85.0% | 75.0% | 90.6% |
| Decider 0.8B | MPS | 75.0% | 75.0% | 75.0% | 90.6% |
| Kev 0.8B | MLX | 80.0% | 70.0% | 50.0% | 81.1% |
| Von | MPS | 50.0% | 70.0% | 35.0% | 64.2% |
| Laya Typed Decisions 421M | MLX | 40.0% | 45.0% | 10.0% | 45.3% |

#### Highlights

- **Decider 2B** leads this small PL run at **91.7%** case accuracy and **93.5%** decision accuracy.
- **Basal 1.5 Mini — 1.5B** reaches **83.3% / 89.2%** while also exposing broader native SystemOne capabilities that this Noul/Choice/Shared suite does not measure directly.
- **Laya Typed Decisions 421M** is the lowest-latency profile in this group at **36.5 ms median** (30.99/s overall throughput).

### Large models

Run `20261006T091845Z` · `deqio-pl-60` · 60 cases. All 5 profiles completed with **0 runtime errors**.

#### Overall results

| Model | Backend | Accuracy | Decision accuracy | Median | P95 | Throughput |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| JevK5 4B | GGUF Q8_0 | 98.3% | 98.9% | 570.6 ms | 1,817.6 ms | 1.72/s |
| Decider 4B | MPS | 96.7% | 97.8% | 438.9 ms | 1,330.6 ms | 2.27/s |
| Kev 4B | MLX | 96.7% | 97.8% | 358.8 ms | 541.2 ms | 3.92/s |
| Basal 1.5 Main — 4.5B | MLX 8-bit | 95.0% | 96.8% | 987.9 ms | 1,860.0 ms | 1.37/s |
| SemIf / Qwen3.5 4B | MLX 8-bit | 95.0% | 96.8% | 599.4 ms | 1,113.4 ms | 2.19/s |

#### Per-type case accuracy

| Model | Backend | Noul | Choice | Shared | Shared decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: |
| JevK5 4B | GGUF Q8_0 | 100.0% | 100.0% | 95.0% | 98.1% |
| Decider 4B | MPS | 100.0% | 95.0% | 95.0% | 98.1% |
| Kev 4B | MLX | 100.0% | 100.0% | 90.0% | 96.2% |
| Basal 1.5 Main — 4.5B | MLX 8-bit | 100.0% | 95.0% | 90.0% | 96.2% |
| SemIf / Qwen3.5 4B | MLX 8-bit | 100.0% | 95.0% | 90.0% | 96.2% |

#### Highlights

- **JevK5 4B** leads this large PL run at **98.3%** case accuracy and **98.9%** decision accuracy.
- **Basal 1.5 Main — 4.5B** reaches **95.0% / 96.8%** while also exposing broader native SystemOne capabilities that this Noul/Choice/Shared suite does not measure directly.
- **Kev 4B** is the lowest-latency profile in this group at **358.8 ms median** (3.92/s overall throughput).
- **Kev 4B** is perfect on Noul and Choice in this run (**100.0% / 100.0%**); its lower aggregate comes from Shared cases.

### Notes

- `Accuracy` is whole-case accuracy. Shared requests pass only when every expected decision is correct.
- `Decision accuracy` scores the individual decisions/assertions and can therefore be higher than whole-case accuracy.
- Latency and throughput are machine/runtime specific; model load time is not included in request median/P95.
- These results are local reproducible snapshots for regression and comparison, not universal model rankings.
- Use `deqio benchmark compare` for canonical run-to-run comparison, deltas, ranks, per-type breakdowns, and profiles present on only one side.
