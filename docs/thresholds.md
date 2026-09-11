# Provisional threshold policy

SynTrustBench v0.3 includes configurable thresholds so runs are deterministic and failure
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
| Fidelity | maximum evaluable per-rule clinical violation rate -> Fail | > 0.01 |
| Utility | TRTR AUROC lower 95% bound -> NotEvaluated | <= 0.50 |
| Utility | TSTR AUROC upper 95% bound -> Fail (if retention evaluable) | <= 0.50 |
| Utility | chance-corrected retention upper 95% bound -> Fail | < 0.80 |
| Utility | chance-corrected retention lower 95% bound -> Pass, absent failure | >= 0.90 |
| Privacy | exact training-copy rate -> Fail | > 0 |
| Privacy | distance attack AUROC lower 95% bound -> Fail | > 0.55 |
| Privacy | distance attack AUROC upper 95% bound -> Pass, subject to other checks | <= 0.55 |
| Privacy | distance attack AUROC CI crossing boundary -> Conditional | 0.55 |
| Privacy | share closer than real-real distance floor -> Conditional | > 0.05 |
| Equity | worst-group TSTR AUROC -> Fail | < 0.60 |
| Equity | representation or utility gap -> Conditional | > 0.10 |
| Robustness | model/generator seed CV -> Conditional | > 0.05 |
| Robustness | missingness AUROC drop -> Conditional | > 0.10 |

All remaining evaluable Utility cases are Conditional. Retention is chance-corrected in
Utility, subgroup descriptions, and robustness conclusion reversal. Core Predictive-Utility
Robustness fails a reversal when baseline retention >= `utility_retention_fail` and
maximum-missingness retention < it, only if Utility established TRTR as reliably above chance.
Required unavailable evidence produces a NotEvaluated dimension unless a Fail takes precedence;
any NotEvaluated dimension yields overall Conditional unless another dimension fails.
Missing optional robustness checks do not block Pass.

`privacy_attack_tpr_fail` retains its default 0.05 for configuration compatibility but is
ignored by v0.3 gating. TPR at 1% FPR is exploratory because reliable low-FPR uncertainty is
not implemented. `privacy_duplicate_rate_fail` also remains accepted, but any exact copy
fails regardless of its configured value. All other existing threshold values are unchanged.

## Calibration plan

Before the thresholds are described as empirically supported, the manuscript should:

- run the protocol on one public clinical dataset and at least two established generators;
- inject exact copies, subgroup removal, dependency disruption, and missingness;
- demonstrate monotonic or directionally correct metric responses;
- report threshold sensitivity analyses;
- solicit clinical/statistical review of clinical and subgroup gates;
- validate on a second dataset or explicitly retain the thresholds as provisional.
