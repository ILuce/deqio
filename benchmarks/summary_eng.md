# ENG Bench — Deqio benchmark summary (0.6.2 snapshot + historical 0.5 tables)

> **Suite:** `deqio-basic-150` — 150 requests · 250 scored decisions · 50 Noul / 50 Choice / 50 Shared per profile<br>
> **Host:** Apple Silicon macOS · 16.0 GiB unified memory · one profile resident at a time<br>
> **0.6.2 runs (2026-10-17, Deqio 0.6.2):** up to 1B `20261017T105226Z` · 1–3B `20261017T110454Z` · above 3B `20261017T120524Z`<br>
> **Historical 0.5 runs (2026-10-06):** small `20261006T082750Z` · large `20261006T090441Z`<br>
> **Execution errors:** 0 in every listed run

This report records the ENG benchmark evidence used by the README: the 0.6.2 snapshot first, the historical 0.5 tables below. Small and large profiles were run separately on the same machine to avoid memory contention.

## 0.6.2 snapshot

Rendered by `deqio benchmark tables` from the committed `runs/small-eng`, `runs/medium-eng` and `runs/max-eng`
summaries (`deqio benchmark tables --summary runs/small-eng --title "Models up to 1B — ENG Bench (0.6.2 snapshot)"`);
`tests/test_benchmark_snapshot.py` keeps this file and the README equal to that output. The remote decision APIs have
no ENG run yet (their PL tables are in `summary_pl.md`). Rows are sorted by decision accuracy; the Decision 3.0 MLX
profiles removed in 0.6.2 do not appear — the runs were made after the removal.

### Models up to 1B — ENG Bench (0.6.2 snapshot)

#### Run

- `20261017T105226Z` · suite `deqio-basic-150` — 150 cases / 250 scored decisions · Deqio 0.6.2 · 2026-10-17
- runtime errors across the listed profiles: 0

#### Quality

`Accuracy` is whole-case accuracy; `Decision accuracy` scores every decision/assertion (a Shared case holds several).

| Model | Backend | Accuracy | Decision accuracy | Noul | Choice | Shared | Shared decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision 3.0 d3-lite 0.85B | GGUF Q8_0 | 86.0% | 90.8% | 94.0% | 94.0% | 70.0% | 88.7% |
| Decider 0.8B | MPS | 84.0% | 88.8% | 88.0% | 94.0% | 70.0% | 87.3% |
| Kev 0.8B | MLX | 76.7% | 84.0% | 84.0% | 86.0% | 60.0% | 83.3% |
| Von | MPS | 64.0% | 73.2% | 86.0% | 58.0% | 48.0% | 74.0% |
| Laya English 421M | MLX | 54.7% | 66.8% | 68.0% | 66.0% | 30.0% | 66.7% |

#### Performance

Median and P95 are per-request latencies; throughput is scored decisions per second over the whole run; cost is the estimated USD per 1 000 scored decisions from the API's reported input tokens and its published input price (remote profiles only; an estimate, not a bill).

| Model | Backend | Median | P95 | Throughput | Cost / 1k decisions |
| --- | :---: | ---: | ---: | ---: | ---: |
| Decision 3.0 d3-lite 0.85B | GGUF Q8_0 | 81.4 ms | 241.9 ms | 13.36/s | — |
| Decider 0.8B | MPS | 293.4 ms | 371.7 ms | 5.94/s | — |
| Kev 0.8B | MLX | 56.1 ms | 90.7 ms | 26.27/s | — |
| Von | MPS | 61.0 ms | 116.4 ms | 21.22/s | — |
| Laya English 421M | MLX | 34.8 ms | 73.8 ms | 41.52/s | — |

#### Identity

| Model | Backend | Engine | Runtime | Precision | Source | Model / endpoint | Revision |
| --- | :---: | --- | --- | --- | --- | --- | --- |
| Decision 3.0 d3-lite 0.85B | GGUF Q8_0 | decision3 | llama.cpp-systemone | q8_0 | official | `ggml-org/vllm-d3-lite-GGUF` | a1f9fa12 |
| Decider 0.8B | MPS | decider | — | — | official | `Mapika/decider-0.8b` | a0a01d6f |
| Kev 0.8B | MLX | kev | — | — | official | `jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f` | bf75a6a8 |
| Von | MPS | von | — | — | official | `von` | 498ceba3 |
| Laya English 421M | MLX | laya | — | — | community | `aac6fef/laya-mlx` | 04767856 |

### Models from 1B to 3B — ENG Bench (0.6.2 snapshot)

#### Run

- `20261017T110454Z` · suite `deqio-basic-150` — 150 cases / 250 scored decisions · Deqio 0.6.2 · 2026-10-17
- runtime errors across the listed profiles: 0

#### Quality

`Accuracy` is whole-case accuracy; `Decision accuracy` scores every decision/assertion (a Shared case holds several).

| Model | Backend | Accuracy | Decision accuracy | Noul | Choice | Shared | Shared decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Decider 2B | MPS | 90.7% | 94.0% | 94.0% | 94.0% | 84.0% | 94.0% |
| Decision 3.0 d3-nano 2.2B | GGUF Q8_0 | 88.7% | 92.4% | 96.0% | 90.0% | 80.0% | 92.0% |
| Basal 2.0 Mini — 1.5B | MLX 8-bit | 80.7% | 85.6% | 78.0% | 94.0% | 70.0% | 85.3% |

#### Performance

Median and P95 are per-request latencies; throughput is scored decisions per second over the whole run; cost is the estimated USD per 1 000 scored decisions from the API's reported input tokens and its published input price (remote profiles only; an estimate, not a bill).

| Model | Backend | Median | P95 | Throughput | Cost / 1k decisions |
| --- | :---: | ---: | ---: | ---: | ---: |
| Decider 2B | MPS | 278.8 ms | 527.0 ms | 5.25/s | — |
| Decision 3.0 d3-nano 2.2B | GGUF Q8_0 | 171.5 ms | 521.5 ms | 6.30/s | — |
| Basal 2.0 Mini — 1.5B | MLX 8-bit | 223.4 ms | 444.7 ms | 6.01/s | — |

#### Identity

| Model | Backend | Engine | Runtime | Precision | Source | Model / endpoint | Revision |
| --- | :---: | --- | --- | --- | --- | --- | --- |
| Decider 2B | MPS | decider | — | — | official | `Mapika/decider-2b` | ff99089b |
| Decision 3.0 d3-nano 2.2B | GGUF Q8_0 | decision3 | llama.cpp-systemone | q8_0 | official | `ggml-org/vllm-d3-nano-GGUF` | 67baf946 |
| Basal 2.0 Mini — 1.5B | MLX 8-bit | basal | basal-2.0-mlx 2.0.0 | mlx-8bit | official | `Remek/basal-2.0-mini-MLX-8bit` | f26d1e2d |

### Models above 3B — ENG Bench (0.6.2 snapshot)

#### Run

- `20261017T120524Z` · suite `deqio-basic-150` — 150 cases / 250 scored decisions · Deqio 0.6.2 · 2026-10-17
- runtime errors across the listed profiles: 0

#### Quality

`Accuracy` is whole-case accuracy; `Decision accuracy` scores every decision/assertion (a Shared case holds several).

| Model | Backend | Accuracy | Decision accuracy | Noul | Choice | Shared | Shared decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision 3.0 d3-mini 4.5B | GGUF Q8_0 | 97.3% | 98.4% | 96.0% | 100.0% | 96.0% | 98.7% |
| JevK5 4B | GGUF Q8_0 | 96.7% | 98.0% | 96.0% | 98.0% | 96.0% | 98.7% |
| Decider 4B | GGUF Q8_0 | 96.0% | 97.6% | 100.0% | 98.0% | 90.0% | 96.7% |
| Basal 2.0 — 4.5B | MLX 8-bit | 93.3% | 96.0% | 90.0% | 96.0% | 94.0% | 98.0% |
| Kev 4B | MLX | 92.7% | 95.2% | 100.0% | 100.0% | 78.0% | 92.0% |

#### Performance

Median and P95 are per-request latencies; throughput is scored decisions per second over the whole run; cost is the estimated USD per 1 000 scored decisions from the API's reported input tokens and its published input price (remote profiles only; an estimate, not a bill).

| Model | Backend | Median | P95 | Throughput | Cost / 1k decisions |
| --- | :---: | ---: | ---: | ---: | ---: |
| Decision 3.0 d3-mini 4.5B | GGUF Q8_0 | 476.1 ms | 1,468.1 ms | 2.27/s | — |
| JevK5 4B | GGUF Q8_0 | 575.1 ms | 1,826.7 ms | 1.82/s | — |
| Decider 4B | GGUF Q8_0 | 274.2 ms | 832.0 ms | 4.04/s | — |
| Basal 2.0 — 4.5B | MLX 8-bit | 673.4 ms | 1,386.6 ms | 1.97/s | — |
| Kev 4B | MLX | 311.6 ms | 504.8 ms | 4.70/s | — |

#### Identity

| Model | Backend | Engine | Runtime | Precision | Source | Model / endpoint | Revision |
| --- | :---: | --- | --- | --- | --- | --- | --- |
| Decision 3.0 d3-mini 4.5B | GGUF Q8_0 | decision3 | llama.cpp-systemone | q8_0 | official | `ggml-org/vllm-d3-mini-GGUF` | c03ac9c3 |
| JevK5 4B | GGUF Q8_0 | jevk5 | jevk5-gguf-native | q8_0 | official | `alibiserikbay/JevK5-GGUF` | ec67b0bf |
| Decider 4B | GGUF Q8_0 | decider | decider-gguf-native | q8_0 | official | `Mapika/decider-4b-GGUF` | b79f09d9 |
| Basal 2.0 — 4.5B | MLX 8-bit | basal | basal-2.0-mlx 2.0.0 | mlx-8bit | official | `Remek/basal-2.0-4.5B-MLX-8bit` | 77d3b723 |
| Kev 4B | MLX | kev | — | — | official | `jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52` | 6cfce5c2 |

## Historical 0.5 snapshot (2026-10-06)

The tables below come from completed Deqio 0.5 runs on 2026-10-06 on the same host, kept for reference only: Deqio,
the runtimes and some pins changed since (Basal 1.5 was replaced by Basal 2.0 in 0.6.1), so they are not comparable
with the 0.6.2 tables above.

### Small models

Run `20261006T082750Z` · `deqio-basic-150` · 150 cases. All 6 profiles completed with **0 runtime errors**.

#### Overall results

| Model | Backend | Accuracy | Decision accuracy | Median | P95 | Throughput |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Decider 2B | MPS | 90.7% | 94.0% | 301.6 ms | 578.6 ms | 5.14/s |
| Basal 1.5 Mini — 1.5B | MLX 8-bit | 84.7% | 90.0% | 403.8 ms | 880.9 ms | 3.28/s |
| Decider 0.8B | MPS | 84.0% | 88.8% | 310.6 ms | 369.1 ms | 5.72/s |
| Kev 0.8B | MLX | 76.7% | 84.0% | 58.3 ms | 94.3 ms | 25.38/s |
| Von | MPS | 64.0% | 73.2% | 61.7 ms | 116.2 ms | 21.29/s |
| Laya Typed Decisions 421M | MLX | 62.0% | 72.4% | 32.9 ms | 70.6 ms | 42.87/s |

#### Per-type case accuracy

| Model | Backend | Noul | Choice | Shared | Shared decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: |
| Decider 2B | MPS | 94.0% | 94.0% | 84.0% | 94.0% |
| Basal 1.5 Mini — 1.5B | MLX 8-bit | 82.0% | 96.0% | 76.0% | 90.7% |
| Decider 0.8B | MPS | 88.0% | 94.0% | 70.0% | 87.3% |
| Kev 0.8B | MLX | 84.0% | 86.0% | 60.0% | 83.3% |
| Von | MPS | 86.0% | 58.0% | 48.0% | 74.0% |
| Laya Typed Decisions 421M | MLX | 74.0% | 78.0% | 34.0% | 70.0% |

#### Highlights

- **Decider 2B** leads this small ENG run at **90.7%** case accuracy and **94.0%** decision accuracy.
- **Basal 1.5 Mini — 1.5B** reaches **84.7% / 90.0%** while also exposing broader native SystemOne capabilities that this Noul/Choice/Shared suite does not measure directly.
- **Laya Typed Decisions 421M** is the lowest-latency profile in this group at **32.9 ms median** (42.87/s overall throughput).

### Large models

Run `20261006T090441Z` · `deqio-basic-150` · 150 cases. All 5 profiles completed with **0 runtime errors**.

#### Overall results

| Model | Backend | Accuracy | Decision accuracy | Median | P95 | Throughput |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| JevK5 4B | GGUF Q8_0 | 96.7% | 98.0% | 594.9 ms | 1,864.5 ms | 1.76/s |
| Decider 4B | MPS | 95.3% | 97.2% | 436.7 ms | 1,361.1 ms | 2.54/s |
| Basal 1.5 Main — 4.5B | MLX 8-bit | 93.3% | 96.0% | 1,213.7 ms | 2,775.6 ms | 1.07/s |
| SemIf / Qwen3.5 4B | MLX 8-bit | 93.3% | 95.6% | 594.7 ms | 1,216.7 ms | 2.22/s |
| Kev 4B | MLX | 92.7% | 95.2% | 315.9 ms | 518.4 ms | 4.59/s |

#### Per-type case accuracy

| Model | Backend | Noul | Choice | Shared | Shared decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: |
| JevK5 4B | GGUF Q8_0 | 96.0% | 98.0% | 96.0% | 98.7% |
| Decider 4B | MPS | 100.0% | 98.0% | 88.0% | 96.0% |
| Basal 1.5 Main — 4.5B | MLX 8-bit | 94.0% | 98.0% | 88.0% | 96.0% |
| SemIf / Qwen3.5 4B | MLX 8-bit | 98.0% | 96.0% | 86.0% | 94.7% |
| Kev 4B | MLX | 100.0% | 100.0% | 78.0% | 92.0% |

#### Highlights

- **JevK5 4B** leads this large ENG run at **96.7%** case accuracy and **98.0%** decision accuracy.
- **Basal 1.5 Main — 4.5B** reaches **93.3% / 96.0%** while also exposing broader native SystemOne capabilities that this Noul/Choice/Shared suite does not measure directly.
- **Kev 4B** is the lowest-latency profile in this group at **315.9 ms median** (4.59/s overall throughput).
- **Kev 4B** is perfect on Noul and Choice in this run (**100.0% / 100.0%**); its lower aggregate comes from Shared cases.

### Notes

- `Accuracy` is whole-case accuracy. Shared requests pass only when every expected decision is correct.
- `Decision accuracy` scores the individual decisions/assertions and can therefore be higher than whole-case accuracy.
- Latency and throughput are machine/runtime specific; model load time is not included in request median/P95.
- These results are local reproducible snapshots for regression and comparison, not universal model rankings.
- Use `deqio benchmark compare` for canonical run-to-run comparison, deltas, ranks, per-type breakdowns, and profiles present on only one side.
