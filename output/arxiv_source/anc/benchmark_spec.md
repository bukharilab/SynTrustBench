# SynTrustBench v0.1.0 benchmark specification

## Normative status

The words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, and **MAY** are normative. Version 0.1.0 defines the evidence standard and a pilot audit. It does not certify systems for deployment and does not rank models using numbers copied from heterogeneous papers.

## 1. Benchmark object

A benchmark submission is identified by the frozen tuple:

`{benchmark_version, modality_protocol, dataset_version, cohort_definition, task, split_hash, generator_version, evaluation_version}`.

Results with different tuples MUST NOT be placed on the same performance leaderboard. An update to cohort logic, split, privacy threat model, or metric implementation creates a new result namespace.

## 2. Tracks

### Track A — Evidence audit (implemented in v0.1.0)

Input: a primary report and its public supporting artifacts.  
Output: a five-dimensional Evidence Maturity Profile, an evaluability gate, missing-evidence flags, and a completed benchmark card.

Track A asks whether claims are supported. It does not compare reported metric values across studies.

### Track B — Controlled model execution (future)

Input: a runnable generator evaluated under a frozen protocol.  
Output: metric-level results, uncertainty, failure flags, and—only when eligibility conditions are met—a within-protocol SyntheticTrust summary.

Track B is out of scope for v0.1.0. No MIMIC-IV/eICU training result or empirical model leaderboard is claimed.

## 3. Modality protocols

Version 0.1.0 specifies reporting requirements for five task families:

1. structured/tabular EHR generation;
2. longitudinal clinical time-series generation;
3. clinical text generation;
4. medical-image generation or translation;
5. multimodal patient-record generation.

Each protocol MUST declare the generated unit (patient, encounter, sequence, note, image, volume, or linked multimodal record), conditioning information, intended use, and prohibited uses.

## 4. Required evaluation bundle

Every controlled submission MUST report all five dimensions. `Not applicable` requires a written protocol-level justification; `not measured` does not.

| Dimension | Minimum evidence bundle | Mandatory failure flag |
|---|---|---|
| Fidelity | Type-appropriate marginal distances; dependency/structure preservation; support coverage; rare-pattern analysis | Clinically impossible records or severe support collapse |
| Clinical utility/validity | Frozen TSTR-on-real evaluation; TRTR reference; discrimination and calibration when relevant; uncertainty across seeds | Material degradation beyond the prespecified non-inferiority margin, or unsafe clinical disagreement |
| Privacy | Declared threat model; membership inference at low FPR; additional attribute/reconstruction attack when applicable; complete DP accounting if claimed | Attack exceeds prespecified risk threshold, missing threat model, or incomplete DP accounting |
| Equity | Predefined subgroup denominators; subgroup fidelity, utility, and privacy; uncertainty | Any subgroup falls below a prespecified floor or is too small to support the claim |
| Robustness/generalization | At least five seeds; uncertainty; one shift/stress test; external/temporal validation when available | Conclusion reverses under a prespecified stressor or external validation |

## 5. Modality-specific alternatives

### Tabular EHR

- Fidelity: KS/Wasserstein for continuous variables, total variation/Jensen–Shannon for categorical variables, mixed-type association matrices, alpha-precision/beta-recall/authenticity.
- Utility: mortality/readmission/phenotyping TSTR with AUROC, AUPRC, calibration slope/intercept, and decision-curve net benefit where clinically meaningful.
- Safety: hard clinical-rule violations and implausible combinations.

### Clinical time series

- Fidelity: marginal and autocorrelation/cross-correlation structure, event timing and missingness patterns, sequence-length distribution, spectral or trajectory metrics.
- Utility: forecasting/classification TSTR on real patients with temporal splits.
- Robustness: irregular sampling, missingness, and horizon shift.

### Medical imaging

- Fidelity: domain-appropriate perceptual and structural metrics; anatomy/pathology consistency; diversity/coverage. Generic natural-image FID MUST NOT be the sole test.
- Utility: real held-out diagnostic/segmentation task and structured expert assessment.
- Privacy: patient/image membership inference and nearest-neighbor or reconstruction review; de-identification alone is not a test.

### Clinical text

- Fidelity/safety: clinical concept consistency, factuality, contradiction and hallucination rate, temporal/medication coherence, and structured expert review. ROUGE alone is insufficient.
- Utility: real held-out extraction/classification task.
- Privacy: canary exposure/leakage plus membership or extraction attack appropriate to access.

### Multimodal records

- All within-modality requirements apply.
- Cross-modal consistency and linkage privacy MUST be evaluated explicitly.

## 6. Seeds, uncertainty, and comparisons

- Stochastic generators MUST be trained/evaluated with at least five independently seeded runs unless a protocol pre-registers a justified alternative.
- Primary metrics MUST include 95% uncertainty intervals and the unit of resampling.
- Hyperparameter selection MUST be separated from the final test set.
- A single best seed MUST NOT stand in for the model.
- Missing/failed runs MUST be reported, not discarded silently.

## 7. Privacy protocol

A threat model MUST state:

1. attacker access (released data, black-box generator, white-box model, or gradients);
2. auxiliary/reference data;
3. target population and member/non-member construction;
4. attack training/calibration procedure;
5. operating point, including TPR at 1% FPR by default;
6. global and subgroup/rarity-stratified results;
7. whether any evaluation or model-selection step re-accesses private data.

A DP claim additionally MUST report adjacency, mechanism, clipping/noise parameters where relevant, sampling, steps/epochs, epsilon, delta, accountant and implementation version, and the privacy cost of tuning or selection.

## 8. Non-compensable eligibility

The Evidence Maturity Profile is always reported as a vector:

`EMP = [Fidelity, Utility, Privacy, Equity, Robustness]`, each in `[0,4]`, plus `Evaluability ∈ {Pass, Conditional, Fail}`.

There is no weighted sum in v0.1.0. For a future composite, every mandatory dimension MUST clear its floor and Evaluability MUST be Pass. This prevents excellent realism from numerically erasing absent privacy or equity evidence.

## 9. Benchmark card

Every submission MUST complete `benchmark_card.yaml`. Empty values are rendered as missing evidence. Cards MUST identify intended use, dataset governance constraints, model/data versions, evaluation code commit, seeds, threat model, subgroup definitions, metric uncertainty, evidence locators, failures, and limitations.

## 10. Governance and versioning

- Corrections are recorded in a dated change log with old value, new value, reason, source, and adjudicator.
- Metric additions or changed thresholds increment the benchmark minor version; changed task/cohort semantics increment the major version.
- The public corpus contains bibliographic metadata and annotations only, never restricted patient data.
- Benchmark maintainers MUST disclose unresolved conflicts and reviewer status.
