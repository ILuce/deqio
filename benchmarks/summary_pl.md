# PL Bench — benchmark summary

> **Suite:** `deqio-pl-60`<br>
> **Run:** 2026-10-03<br>
> **Host:** Apple Silicon macOS · 16.0 GiB unified memory<br>
> **Workload:** 60 requests · 93 scored decisions · 20 Noul / 20 Choice / 20 Shared<br>
> **Language:** Polish-native scenarios<br>
> **Profiles:** 10<br>
> **Execution errors:** 0 across 600 model-case records

PL Bench is a dedicated Polish suite rather than a mechanical translation of ENG Bench. It exercises Polish instructions and decision descriptions across release validation, permissions, finance, deadlines, routing, inventory, security, evidence freshness, arithmetic constraints, and multi-decision shared state.

## Overall results

| Model | Backend | Accuracy | Decision accuracy | Median | P95 |
| --- | :---: | ---: | ---: | ---: | ---: |
| Kev 4B | MLX | 96.7% | 97.8% | 312.5 ms | 494.9 ms |
| Decider 4B | MPS | 96.7% | 97.8% | 438.8 ms | 1,328.5 ms |
| Clef Flash 9B | MLX | 96.7% | 97.8% | 1,177.4 ms | 1,897.9 ms |
| SemIf / Qwen3.5 4B | MLX | 95.0% | 96.8% | 580.3 ms | 1,019.6 ms |
| Basal 4.5B | MLX | 93.3% | 95.7% | 891.4 ms | 2,318.5 ms |
| Decider 2B | MPS | 91.7% | 93.5% | 363.0 ms | 581.3 ms |
| Basal 1.5B | MLX | 85.0% | 89.2% | 290.6 ms | 754.3 ms |
| Kev 0.8B | MLX | 66.7% | 78.5% | 58.9 ms | 91.5 ms |
| Von | MPS | 51.7% | 62.4% | 76.3 ms | 142.3 ms |
| Laya English 421M | MLX | 38.3% | 53.8% | 37.8 ms | 103.5 ms |

## Per-type case accuracy

| Model | Backend | Noul | Choice | Shared |
| --- | :---: | ---: | ---: | ---: |
| Kev 4B | MLX | 100.0% | 100.0% | 90.0% |
| Decider 4B | MPS | 100.0% | 95.0% | 95.0% |
| Clef Flash 9B | MLX | 100.0% | 95.0% | 95.0% |
| SemIf / Qwen3.5 4B | MLX | 100.0% | 95.0% | 90.0% |
| Basal 4.5B | MLX | 95.0% | 95.0% | 90.0% |
| Decider 2B | MPS | 95.0% | 95.0% | 85.0% |
| Basal 1.5B | MLX | 80.0% | 85.0% | 90.0% |
| Kev 0.8B | MLX | 80.0% | 70.0% | 50.0% |
| Von | MPS | 50.0% | 70.0% | 35.0% |
| Laya English 421M | MLX | 45.0% | 55.0% | 15.0% |

## Polish-suite highlights

- **Kev 4B / MLX**, **Decider 4B / MPS**, and **Clef Flash 9B / MLX** tied at **96.7% case accuracy** and **97.8% decision accuracy**. Their median latencies were **312.5 ms**, **438.8 ms**, and **1,177.4 ms** respectively.
- **SemIf / Qwen3.5 4B / MLX** reached **95.0% case accuracy** and **96.8% decision accuracy**.
- **Basal 4.5B / MLX** reached **93.3% / 95.7%**, compared with **85.0% / 89.2%** for **Basal 1.5B / MLX** on the same PL suite.
- **Kev 4B / MLX** was perfect on the 20 Noul and 20 Choice cases and missed only two Shared requests.
- **Laya English 421M / MLX** remained the lowest-latency profile at **37.8 ms median**, but reached **38.3% case accuracy** on this Polish-native suite.

## Most difficult PL cases in this run

| Case | Profiles that missed it | What it tests |
| --- | ---: | --- |
| `pl-shared-05` | 9/10 | Budget arithmetic across two related purchase decisions: 18,000 PLN fits, 21,000 PLN does not. |
| `pl-choice-08` | 8/10 | Inventory arithmetic with a safety-stock constraint: 12 - 8 leaves 4, below the required reserve of 5. |
| `pl-noul-07` | 5/10 | Fresh-backup policy: a verified backup from 12 hours ago satisfies a 24-hour freshness requirement. |
| `pl-shared-13` | 4/10 | File validation where type and size pass but malware must still block final acceptance. |
| `pl-noul-03` | 4/10 | Current authoritative payment state versus stale earlier evidence. |

`pl-shared-05` is the clearest common failure: 9 of 10 tested profiles missed at least one of its two budget decisions. `pl-choice-08` was missed by 8 of 10 profiles, making simple arithmetic plus an explicit business constraint another useful regression target.

## ENG vs PL snapshot

The table below is useful for spotting changes worth investigating, but **it is not a pure language penalty/benefit measurement**. ENG Bench has 150 cases while PL Bench has 60, and the scenarios are not one-to-one translations.

| Model | ENG case acc. | PL case acc. | PL - ENG | ENG decision acc. | PL decision acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kev 4B | 92.7% | 96.7% | +4.0 pp | 95.2% | 97.8% |
| Decider 4B | 95.3% | 96.7% | +1.3 pp | 97.2% | 97.8% |
| Clef Flash 9B | 98.0% | 96.7% | -1.3 pp | 98.8% | 97.8% |
| SemIf / Qwen3.5 4B | 93.3% | 95.0% | +1.7 pp | 95.6% | 96.8% |
| Basal 4.5B | 90.0% | 93.3% | +3.3 pp | 94.0% | 95.7% |
| Decider 2B | 90.7% | 91.7% | +1.0 pp | 94.0% | 93.5% |
| Basal 1.5B | 85.3% | 85.0% | -0.3 pp | 90.0% | 89.2% |
| Kev 0.8B | 76.7% | 66.7% | -10.0 pp | 84.0% | 78.5% |
| Von | 64.0% | 51.7% | -12.3 pp | 73.2% | 62.4% |
| Laya English 421M | 54.7% | 38.3% | -16.3 pp | 66.8% | 53.8% |

The largest negative PL-vs-ENG case-accuracy deltas in this run occur for **Laya English**, **Von**, and **Kev 0.8B**. Larger models such as **Kev 4B**, **Basal 4.5B**, **SemIf 4B**, and **Decider 4B** were at least as accurate on the PL suite as on the larger ENG suite, but the suite-content caveat above is essential when interpreting that difference.

## Notes

- All 10 profiles completed the PL suite; the supplied results contain **no runtime/benchmark execution errors**.
- PL Bench is deliberately Polish-native and should evolve independently where Polish wording or domain conventions need dedicated coverage.
- Latency is machine-specific and model load time is not included in request median/P95.
- Keep the same suite revision when using this report for regression comparisons.
