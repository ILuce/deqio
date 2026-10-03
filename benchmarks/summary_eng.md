# ENG Bench — benchmark summary

> **Suite:** `deqio-basic-150`<br>
> **Run:** 2026-10-03<br>
> **Host:** Apple Silicon macOS · 16.0 GiB unified memory<br>
> **Workload:** 150 requests · 250 scored decisions · 50 Noul / 50 Choice / 50 Shared<br>
> **Profiles:** 10<br>
> **Execution errors:** 0 across 1500 model-case records

This report records one local Deqio benchmark run. It is a reproducible snapshot for this machine and model set, not a universal leaderboard.

## Overall results

| Model | Backend | Accuracy | Decision accuracy | Median | P95 |
| --- | :---: | ---: | ---: | ---: | ---: |
| Clef Flash 9B | MLX | 98.0% | 98.8% | 1,183.3 ms | 2,369.0 ms |
| Decider 4B | MPS | 95.3% | 97.2% | 537.6 ms | 1,352.3 ms |
| SemIf / Qwen3.5 4B | MLX | 93.3% | 95.6% | 644.6 ms | 1,234.0 ms |
| Kev 4B | MLX | 92.7% | 95.2% | 315.5 ms | 538.9 ms |
| Decider 2B | MPS | 90.7% | 94.0% | 310.7 ms | 582.7 ms |
| Basal 4.5B | MLX | 90.0% | 94.0% | 1,578.9 ms | 4,581.9 ms |
| Basal 1.5B | MLX | 85.3% | 90.0% | 407.8 ms | 1,216.4 ms |
| Kev 0.8B | MLX | 76.7% | 84.0% | 57.8 ms | 94.0 ms |
| Von | MPS | 64.0% | 73.2% | 63.2 ms | 130.7 ms |
| Laya English 421M | MLX | 54.7% | 66.8% | 33.5 ms | 77.4 ms |

A Shared request counts as a passed case only when every expected sub-decision is correct. Decision accuracy scores those sub-decisions independently, which is why it can be higher than case accuracy.

## Per-type case accuracy

| Model | Backend | Noul | Choice | Shared |
| --- | :---: | ---: | ---: | ---: |
| Clef Flash 9B | MLX | 100.0% | 98.0% | 96.0% |
| Decider 4B | MPS | 100.0% | 98.0% | 88.0% |
| SemIf / Qwen3.5 4B | MLX | 98.0% | 96.0% | 86.0% |
| Kev 4B | MLX | 100.0% | 100.0% | 78.0% |
| Decider 2B | MPS | 94.0% | 94.0% | 84.0% |
| Basal 4.5B | MLX | 92.0% | 96.0% | 82.0% |
| Basal 1.5B | MLX | 84.0% | 90.0% | 82.0% |
| Kev 0.8B | MLX | 84.0% | 86.0% | 60.0% |
| Von | MPS | 86.0% | 58.0% | 48.0% |
| Laya English 421M | MLX | 68.0% | 66.0% | 30.0% |

## Run highlights

- **Clef Flash 9B / MLX** produced the highest overall score: **98.0%** case accuracy and **98.8%** decision accuracy. It missed only 3 of 150 cases.
- **Kev 4B / MLX** reached **100.0% Noul** and **100.0% Choice** case accuracy; all of its 11 failed cases were Shared requests.
- **Decider 4B / MPS** reached **95.3%** case accuracy with a **537.6 ms** median.
- **SemIf / Qwen3.5 4B / MLX** reached **93.3%** case accuracy after the audited MLX 8-bit path was enabled.
- **Laya English 421M / MLX** was the lowest-latency profile at **33.5 ms** median, but its case accuracy was **54.7%**.

## Most difficult ENG cases in this run

The following cases were missed by the largest number of the 10 tested profiles:

| Case | Profiles that missed it | What it tests |
| --- | ---: | --- |
| `shared-22` | 9/10 | Stale cross-platform validation evidence after the code revision changed. |
| `shared-40` | 9/10 | Inclusive 90-day credential expiry, next-day validity, and rotation action. |
| `shared-26` | 7/10 | Non-idempotent payment timeout: duplicate-effect risk and reconciliation before retry. |
| `shared-23` | 7/10 | Applying an accuracy threshold before latency when selecting between two benchmarked models. |
| `choice-42` | 5/10 | Mapping partial service degradation to a severity rubric. |

The concentration of failures in Shared cases is visible in the per-type table: the benchmark requires several decisions to remain simultaneously correct under one shared state.

## Notes

- All 10 profiles completed the suite; the supplied results contain **no runtime/benchmark execution errors**.
- Latency includes model-specific runtime behavior and is machine-specific.
- Model load time is not included in request median/P95.
- The benchmark is intended for regression and local comparison. Hardware, runtime versions, revisions, token budgets, and benchmark edits can change the result.
