# ENG Bench — Deqio 0.5 current benchmark summary

> **Suite:** `deqio-basic-150`<br>
> **Date:** 2026-10-06<br>
> **Host:** Apple Silicon macOS · 16.0 GiB unified memory<br>
> **Workload per profile:** 150 requests · 250 scored decisions · 50 Noul / 50 Choice / 50 Shared<br>
> **Runs:** small `20261006T082750Z` · large `20261006T090441Z`<br>
> **Profiles:** 6 small + 5 large<br>
> **Execution errors:** 0 across 1650 model-case records

This report records the latest Deqio 0.5 ENG benchmark evidence used by the README. Small and large profiles were run separately on the same machine to avoid memory contention.

## Small models

Run `20261006T082750Z` · `deqio-basic-150` · 150 cases. All 6 profiles completed with **0 runtime errors**.

### Overall results

| Model | Backend | Accuracy | Decision accuracy | Median | P95 | Throughput |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Decider 2B | MPS | 90.7% | 94.0% | 301.6 ms | 578.6 ms | 5.14/s |
| Basal 1.5 Mini — 1.5B | MLX 8-bit | 84.7% | 90.0% | 403.8 ms | 880.9 ms | 3.28/s |
| Decider 0.8B | MPS | 84.0% | 88.8% | 310.6 ms | 369.1 ms | 5.72/s |
| Kev 0.8B | MLX | 76.7% | 84.0% | 58.3 ms | 94.3 ms | 25.38/s |
| Von | MPS | 64.0% | 73.2% | 61.7 ms | 116.2 ms | 21.29/s |
| Laya Typed Decisions 421M | MLX | 62.0% | 72.4% | 32.9 ms | 70.6 ms | 42.87/s |

### Per-type case accuracy

| Model | Backend | Noul | Choice | Shared | Shared decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: |
| Decider 2B | MPS | 94.0% | 94.0% | 84.0% | 94.0% |
| Basal 1.5 Mini — 1.5B | MLX 8-bit | 82.0% | 96.0% | 76.0% | 90.7% |
| Decider 0.8B | MPS | 88.0% | 94.0% | 70.0% | 87.3% |
| Kev 0.8B | MLX | 84.0% | 86.0% | 60.0% | 83.3% |
| Von | MPS | 86.0% | 58.0% | 48.0% | 74.0% |
| Laya Typed Decisions 421M | MLX | 74.0% | 78.0% | 34.0% | 70.0% |

### Highlights

- **Decider 2B** leads this small ENG run at **90.7%** case accuracy and **94.0%** decision accuracy.
- **Basal 1.5 Mini — 1.5B** reaches **84.7% / 90.0%** while also exposing broader native SystemOne capabilities that this Noul/Choice/Shared suite does not measure directly.
- **Laya Typed Decisions 421M** is the lowest-latency profile in this group at **32.9 ms median** (42.87/s overall throughput).

## Large models

Run `20261006T090441Z` · `deqio-basic-150` · 150 cases. All 5 profiles completed with **0 runtime errors**.

### Overall results

| Model | Backend | Accuracy | Decision accuracy | Median | P95 | Throughput |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| JevK5 4B | GGUF Q8_0 | 96.7% | 98.0% | 594.9 ms | 1,864.5 ms | 1.76/s |
| Decider 4B | MPS | 95.3% | 97.2% | 436.7 ms | 1,361.1 ms | 2.54/s |
| Basal 1.5 Main — 4.5B | MLX 8-bit | 93.3% | 96.0% | 1,213.7 ms | 2,775.6 ms | 1.07/s |
| SemIf / Qwen3.5 4B | MLX 8-bit | 93.3% | 95.6% | 594.7 ms | 1,216.7 ms | 2.22/s |
| Kev 4B | MLX | 92.7% | 95.2% | 315.9 ms | 518.4 ms | 4.59/s |

### Per-type case accuracy

| Model | Backend | Noul | Choice | Shared | Shared decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: |
| JevK5 4B | GGUF Q8_0 | 96.0% | 98.0% | 96.0% | 98.7% |
| Decider 4B | MPS | 100.0% | 98.0% | 88.0% | 96.0% |
| Basal 1.5 Main — 4.5B | MLX 8-bit | 94.0% | 98.0% | 88.0% | 96.0% |
| SemIf / Qwen3.5 4B | MLX 8-bit | 98.0% | 96.0% | 86.0% | 94.7% |
| Kev 4B | MLX | 100.0% | 100.0% | 78.0% | 92.0% |

### Highlights

- **JevK5 4B** leads this large ENG run at **96.7%** case accuracy and **98.0%** decision accuracy.
- **Basal 1.5 Main — 4.5B** reaches **93.3% / 96.0%** while also exposing broader native SystemOne capabilities that this Noul/Choice/Shared suite does not measure directly.
- **Kev 4B** is the lowest-latency profile in this group at **315.9 ms median** (4.59/s overall throughput).
- **Kev 4B** is perfect on Noul and Choice in this run (**100.0% / 100.0%**); its lower aggregate comes from Shared cases.

## Notes

- `Accuracy` is whole-case accuracy. Shared requests pass only when every expected decision is correct.
- `Decision accuracy` scores the individual decisions/assertions and can therefore be higher than whole-case accuracy.
- Latency and throughput are machine/runtime specific; model load time is not included in request median/P95.
- These results are local reproducible snapshots for regression and comparison, not universal model rankings.
- Use `deqio benchmark compare` for canonical run-to-run comparison, deltas, ranks, per-type breakdowns, and profiles present on only one side.
