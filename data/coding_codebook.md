# SynTrustBench v0.1.0 coding codebook

## 1. Unit of analysis

One row represents one **included report**. A report may be a journal article, full conference paper, workshop paper, preprint, or thesis. `peer_reviewed` describes publication status and must not be inferred from technical quality. Benchmark or evaluation-framework papers remain in the corpus when they empirically evaluate synthetic clinical data, but `task_scope` distinguishes them from papers that introduce a generator.

Use `NR` when the primary source does not report a field. Use `NA` only when a field is structurally inapplicable (for example, `epsilon_values` when `dp_claimed=No`). Never treat de-identification, restricted access, HIPAA/GDPR alignment, or an author's use of the word “private” as a privacy test.

## 2. Source hierarchy

Code from the full primary report. Use an official supplement or repository for fields not present in the paper. Use a publisher or proceedings page for publication status and bibliographic metadata. Secondary reviews may locate sources but may not establish technical fields. Every row requires `primary_source_url`, `verification_status`, `verifier`, and `verified_date`.

## 3. Core classifications

| Field | Allowed values / rule |
|---|---|
| `publication_type` | Journal article; conference paper; workshop paper; preprint; thesis; other. |
| `peer_reviewed` | Yes; No; Unclear. Conference/workshop proceedings count as peer reviewed only when the venue documents review or the publication record establishes it. |
| `modality` | Tabular EHR; Clinical time series; Medical imaging; Clinical text; Multimodal/other. Assign the generated output, not an auxiliary input. |
| `task_scope` | Generator; generator plus benchmark; benchmark/evaluation framework; augmentation; translation/reconstruction; other. |
| `architecture_family` | GAN; diffusion; LLM; VAE; hybrid/other; kernel/marginal/other; evaluation framework. For frameworks, record the evaluated systems in `evaluated_systems`; do not label the framework as its baseline model. |
| `privacy_evaluated` | Yes only if the report performs a quantitative empirical attack/risk analysis or provides a formal privacy guarantee. Administrative safeguards and de-identification alone are No. |
| `dp_claimed` | Yes when the report explicitly describes the evaluated method as differentially private, even if the supporting evidence is incomplete. This records the author's claim; it does not validate it. |
| `formal_dp_verified` | Yes only when an algorithm-level differential-privacy guarantee is verifiable for the evaluated pipeline. The source must identify a DP mechanism and sufficient accounting or theorem details to interpret the guarantee. “DP-compatible,” comparison with a DP baseline, unaccounted gradient noise, an interpolation coefficient named epsilon, and privacy-themed model names are No. |
| `dp_evidence_status` | Complete; incomplete; not a DP claim. Records why `dp_claimed` and `formal_dp_verified` may differ. |
| `epsilon_reported` | Yes only if a numeric epsilon or explicit range is reported for the evaluated run(s). |
| `fairness_evaluated` | Yes only when representation, fidelity, privacy, or downstream utility is measured by a protected or clinically meaningful subgroup, or a formal fairness objective is evaluated. Merely conditioning on sex/age is No. |
| `robustness_evaluated` | Yes when the study explicitly tests sensitivity to seeds, perturbations, missingness, dataset shift, site/time transfer, or another predefined stressor. Multiple datasets alone are captured separately and do not automatically imply robustness. |

## 4. Evidence dimensions

The benchmark separates evidence into five non-interchangeable dimensions. A study cannot offset absent privacy evidence with excellent fidelity, or absent equity evidence with excellent utility.

### 4.1 Fidelity

Required question: does the synthetic output preserve the statistical and clinically relevant structure of the target data?

- Distributional evidence: marginal distances appropriate to variable type (for example, KS/Wasserstein for continuous variables and total variation/Jensen–Shannon for categorical variables).
- Dependency evidence: correlation or association preservation, temporal dependence, anatomical structure, or semantic/clinical consistency as appropriate to modality.
- Coverage evidence: precision/recall-style support coverage or explicit rare-pattern preservation.
- Human clinical review may complement but not replace quantitative evidence.

### 4.2 Clinical utility and validity

Required question: does analysis or training with synthetic data preserve decisions on real, held-out patients?

- Prefer TSTR evaluated on a frozen real test set, compared with TRTR and simple baselines.
- Report discrimination and task-appropriate precision-recall metrics for imbalanced outcomes.
- Report calibration when outputs represent risk.
- Report uncertainty over repeated seeds or bootstrap samples.
- For images/text, task-specific expert review or structured clinical agreement is an additional validity and safety test, not a substitute for held-out utility. A prespecified safety failure remains blocking even when utility is favorable.

### 4.3 Privacy

Required question: under a declared attacker, what can be inferred about training patients?

- State attacker access, auxiliary knowledge, target record selection, member/non-member construction, and decision threshold.
- Include membership inference; add attribute inference or reconstruction when the intended release creates those risks.
- Prefer TPR at low FPR and stratification by rarity or subgroup over a single average attack AUC.
- If claiming DP, report mechanism, adjacency relation, epsilon, delta, accountant, sampling assumptions, number of training steps, and all privacy-consuming tuning.

### 4.4 Equity

Required question: are fidelity, utility, and disclosure risk preserved across protected and clinically important subgroups?

- Predefine subgroup attributes and intersectional analyses where sample size permits.
- Report subgroup denominators and uncertainty.
- Evaluate prevalence/coverage, subgroup fidelity, subgroup utility, and subgroup privacy; classifier-parity metrics alone are insufficient for a synthetic-data claim.

### 4.5 Robustness and generalization

Required question: do conclusions survive plausible changes in seeds, sites, time, measurement, and population?

- Report at least five random seeds for stochastic model comparisons in the execution track.
- Include uncertainty intervals and predefined failure handling.
- Prefer external-site or temporal validation; otherwise use controlled missingness, perturbation, or distribution-shift tests.

## 5. Evidence Maturity Profile (EMP)

EMP rates **how a claim was tested and reported**. It does not rate whether a model's result was favorable.

| Level | Cross-dimensional meaning |
|---:|---|
| 0 | Dimension not quantitatively evaluated. |
| 1 | One descriptive, visual, surrogate, or weakly matched test. |
| 2 | Multiple appropriate tests, or one well-designed held-out/attack test with enough detail to interpret. |
| 3 | Stress-tested evidence: adversarial, subgroup, repeated, clinically reviewed, or externally validated as appropriate to the dimension, with uncertainty. |
| 4 | Independently reproducible evidence, or a complete formal guarantee where that is the relevant gold standard. Code, data-access pathway, configuration, and metric implementation are sufficient to reproduce the claim. |

Dimension-specific anchors:

| Dimension | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Fidelity | One marginal/visual or single global metric | Complementary marginal and dependency/coverage tests | Subgroup, clinical-review, repeated, or external fidelity with uncertainty | Level 3 plus independently reproducible implementation and accessible evaluation pathway |
| Utility | Synthetic-only/in-sample proxy or one weak downstream comparison | TSTR/real held-out evaluation or multiple task metrics | Repeated/external/subgroup utility plus calibration or clinical agreement where relevant | Level 3 plus independently reproducible task, split, and evaluation code |
| Privacy | Similarity/distance proxy only | Explicit empirical attack with stated member/non-member design, or partially specified formal DP | Multiple/calibrated attacks, low-FPR/subgroup/repeated analysis, or complete end-to-end DP accounting | Independently reproducible adversarial evaluation, or fully documented formal proof/accounting with a complete release pipeline |
| Equity | Descriptive subgroup counts only | Subgroup fidelity, utility, or privacy outcome | Multiple subgroup dimensions/metrics with uncertainty or evaluated mitigation | Level 3 plus independently reproducible subgroup definitions and evaluation |
| Robustness | One sensitivity check or informal multi-dataset claim | Repeated seeds, controlled perturbation, or explicit shift test | External-site/time validation or several prespecified stressors with uncertainty | Level 3 plus independently reproducible stress-test suite |

## 6. Evaluability gate

The gate prevents incomplete reporting from being hidden by a composite score.

- `Pass`: the central evaluation can be independently reconstructed from an identifiable model/configuration, cohort and preprocessing definition, split logic, metric implementation, released code, and a lawful data-access route or a fully specified open substitute. A missing dataset-version string does not by itself force `Conditional` when a frozen code or data artifact resolves the version unambiguously; a private clinical cohort without a reproducible substitute cannot pass.
- `Conditional`: the main evaluation is interpretable, but at least one reproducibility element is missing (for example, seeds, exact split, or released configuration).
- `Fail`: the evaluated system, dataset/cohort, split, or metric definition is too incomplete to reproduce the central claim.

A report may receive an EMP even when its gate is Conditional or Fail, because the pilot is an evidence audit. A future model submission is ineligible for a cross-system SynTrust comparison until the gate is Pass.

## 7. Composite scores

Version 0.1.0 does not compute a weighted SyntheticTrust Score and does not create a literature-derived model leaderboard. It reports the five-dimensional profile and the gate. A future controlled execution track may report a geometric-mean score only after every required dimension clears a prespecified floor and systems share the same benchmark tuple.

## 8. Reviewer workflow

1. Two reviewers code ambiguous task, privacy, DP, fairness, and eligibility fields independently.
2. A random 25–30% of remaining reports is double-coded.
3. Report raw agreement plus Cohen's kappa when prevalence is not extreme; report Gwet's AC1 as a sensitivity statistic for sparse fields.
4. Adjudicate disagreements against the primary source and retain both pre-adjudication values in the audit log.
5. Until a human second review is complete, `verification_status` must say `Primary source verified; human double-code pending` and no inter-rater statistic may be reported.
