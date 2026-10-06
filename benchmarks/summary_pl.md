# PL Bench — Deqio 0.5 current benchmark summary

> **Suite:** `deqio-pl-60`<br>
> **Date:** 2026-10-06<br>
> **Host:** Apple Silicon macOS · 16.0 GiB unified memory<br>
> **Workload per profile:** 60 requests · 93 scored decisions · 20 Noul / 20 Choice / 20 Shared<br>
> **Runs:** small `20261006T083557Z` · large `20261006T091845Z`<br>
> **Profiles:** 6 small + 5 large<br>
> **Execution errors:** 0 across 660 model-case records

PL Bench is a dedicated Polish-native suite rather than a mechanical translation of ENG Bench. The ENG and PL suites differ in size and scenario composition, so cross-suite deltas should not be interpreted as a pure language-quality measurement.

## Small models

Run `20261006T083557Z` · `deqio-pl-60` · 60 cases. All 6 profiles completed with **0 runtime errors**.

### Overall results

| Model | Backend | Accuracy | Decision accuracy | Median | P95 | Throughput |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Decider 2B | MPS | 91.7% | 93.5% | 334.3 ms | 512.2 ms | 4.96/s |
| Basal 1.5 Mini — 1.5B | MLX 8-bit | 83.3% | 89.2% | 318.9 ms | 572.9 ms | 4.29/s |
| Decider 0.8B | MPS | 75.0% | 83.9% | 316.6 ms | 403.2 ms | 4.87/s |
| Kev 0.8B | MLX | 66.7% | 78.5% | 59.5 ms | 92.5 ms | 23.85/s |
| Von | MPS | 51.7% | 62.4% | 75.7 ms | 138.9 ms | 18.83/s |
| Laya Typed Decisions 421M | MLX | 31.7% | 44.1% | 36.5 ms | 99.2 ms | 30.99/s |

### Per-type case accuracy

| Model | Backend | Noul | Choice | Shared | Shared decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: |
| Decider 2B | MPS | 95.0% | 95.0% | 85.0% | 92.5% |
| Basal 1.5 Mini — 1.5B | MLX 8-bit | 90.0% | 85.0% | 75.0% | 90.6% |
| Decider 0.8B | MPS | 75.0% | 75.0% | 75.0% | 90.6% |
| Kev 0.8B | MLX | 80.0% | 70.0% | 50.0% | 81.1% |
| Von | MPS | 50.0% | 70.0% | 35.0% | 64.2% |
| Laya Typed Decisions 421M | MLX | 40.0% | 45.0% | 10.0% | 45.3% |

### Highlights

- **Decider 2B** leads this small PL run at **91.7%** case accuracy and **93.5%** decision accuracy.
- **Basal 1.5 Mini — 1.5B** reaches **83.3% / 89.2%** while also exposing broader native SystemOne capabilities that this Noul/Choice/Shared suite does not measure directly.
- **Laya Typed Decisions 421M** is the lowest-latency profile in this group at **36.5 ms median** (30.99/s overall throughput).

## Large models

Run `20261006T091845Z` · `deqio-pl-60` · 60 cases. All 5 profiles completed with **0 runtime errors**.

### Overall results

| Model | Backend | Accuracy | Decision accuracy | Median | P95 | Throughput |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| JevK5 4B | GGUF Q8_0 | 98.3% | 98.9% | 570.6 ms | 1,817.6 ms | 1.72/s |
| Decider 4B | MPS | 96.7% | 97.8% | 438.9 ms | 1,330.6 ms | 2.27/s |
| Kev 4B | MLX | 96.7% | 97.8% | 358.8 ms | 541.2 ms | 3.92/s |
| Basal 1.5 Main — 4.5B | MLX 8-bit | 95.0% | 96.8% | 987.9 ms | 1,860.0 ms | 1.37/s |
| SemIf / Qwen3.5 4B | MLX 8-bit | 95.0% | 96.8% | 599.4 ms | 1,113.4 ms | 2.19/s |

### Per-type case accuracy

| Model | Backend | Noul | Choice | Shared | Shared decision accuracy |
| --- | :---: | ---: | ---: | ---: | ---: |
| JevK5 4B | GGUF Q8_0 | 100.0% | 100.0% | 95.0% | 98.1% |
| Decider 4B | MPS | 100.0% | 95.0% | 95.0% | 98.1% |
| Kev 4B | MLX | 100.0% | 100.0% | 90.0% | 96.2% |
| Basal 1.5 Main — 4.5B | MLX 8-bit | 100.0% | 95.0% | 90.0% | 96.2% |
| SemIf / Qwen3.5 4B | MLX 8-bit | 100.0% | 95.0% | 90.0% | 96.2% |

### Highlights

- **JevK5 4B** leads this large PL run at **98.3%** case accuracy and **98.9%** decision accuracy.
- **Basal 1.5 Main — 4.5B** reaches **95.0% / 96.8%** while also exposing broader native SystemOne capabilities that this Noul/Choice/Shared suite does not measure directly.
- **Kev 4B** is the lowest-latency profile in this group at **358.8 ms median** (3.92/s overall throughput).
- **Kev 4B** is perfect on Noul and Choice in this run (**100.0% / 100.0%**); its lower aggregate comes from Shared cases.

## Notes

- `Accuracy` is whole-case accuracy. Shared requests pass only when every expected decision is correct.
- `Decision accuracy` scores the individual decisions/assertions and can therefore be higher than whole-case accuracy.
- Latency and throughput are machine/runtime specific; model load time is not included in request median/P95.
- These results are local reproducible snapshots for regression and comparison, not universal model rankings.
- Use `deqio benchmark compare` for canonical run-to-run comparison, deltas, ranks, per-type breakdowns, and profiles present on only one side.
