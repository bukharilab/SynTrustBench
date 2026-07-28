# Provisional threshold policy

SynTrustBench v0.2 includes configurable thresholds so runs are deterministic and failure
injection can be tested. The defaults are **research hypotheses**, not community consensus,
clinical acceptability limits, regulatory rules, or validated safety boundaries.

## Why thresholds exist

A benchmark that only emits numbers cannot implement failure flags or non-compensable gates.
Conversely, an arbitrary cutoff can create false authority. SynTrustBench therefore:

1. publishes every threshold in the configuration contract;
2. records resolved thresholds in `run_manifest.json`;
3. labels them provisional in every report;
4. preserves raw metric values and uncertainty;
5. avoids a weighted overall score;
6. plans empirical calibration through known-failure injections and external case studies.

## Default decisions

| Dimension | Rule | Default |
|---|---|---:|
| Fidelity | mean normalized Wasserstein -> Conditional | > 0.20 |
| Fidelity | mean JSD -> Conditional | > 0.10 |
| Fidelity | dependency error -> Conditional | > 0.15 |
| Fidelity | minimum category coverage -> Fail | < 0.95 |
| Fidelity | clinical violation rate -> Fail | > 0.01 |
| Utility | AUROC retention -> Fail / Conditional | < 0.80 / < 0.90 |
| Privacy | exact training-copy rate -> Fail | > 0 |
| Privacy | distance attack AUC lower bound -> Fail | > 0.55 |
| Equity | worst-group TSTR AUROC -> Fail | < 0.60 |
| Equity | representation or utility gap -> Conditional | > 0.10 |
| Robustness | model/generator seed CV -> Conditional | > 0.05 |
| Robustness | missingness AUROC drop -> Conditional | > 0.10 |

## Calibration plan

Before the thresholds are described as empirically supported, the manuscript should:

- run the protocol on one public clinical dataset and at least two established generators;
- inject exact copies, subgroup removal, dependency disruption, and missingness;
- demonstrate monotonic or directionally correct metric responses;
- report threshold sensitivity analyses;
- solicit clinical/statistical review of clinical and subgroup gates;
- validate on a second dataset or explicitly retain the thresholds as provisional.
