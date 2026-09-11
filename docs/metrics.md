# Metric definitions: structured tabular protocol v0.3

Each dimension returns metric estimates, uncertainty where defensible, explicit availability,
warnings, and a provisional status. Dimension statuses are non-compensable. No weighted overall
trust score is calculated.

## Common uncertainty rules

- Predictive metrics use row-level bootstrap resampling of the frozen held-out real test set.
- TRTR/TSTR retention and the TSTR-minus-TRTR AUROC difference use paired bootstrap rows.
- Distribution distances use independent bootstrap resamples of real training and synthetic
  rows. A bootstrap standard-error interval is centered on the observed distance because naive
  percentile intervals for non-negative distances are biased upward near zero.
- Copy, proximity, and subgroup statistics resample their relevant rows.
- Ranges across model seeds, generator seeds, subgroups, or perturbations are labeled observed
  ranges, not confidence intervals.
- The default is 200 resamples at 95% confidence. Both values are recorded in the run manifest.
  Overall Utility AUROC and membership-AUROC decisions use 95% intervals; other metrics,
  including subgroup intervals, retain the configured confidence level.

## Fidelity

**Question:** Does the synthetic table preserve relevant distributions, dependencies, support,
missingness, and clinical rules?

**Inputs:** `real_train`, `synthetic`, column types, and clinical constraints.

### Continuous marginals

For continuous column \(j\):

\[
W_j^* = \frac{W_1(X_j^{real}, X_j^{syn})}{IQR(X_j^{real})}
\]

The denominator falls back to the real standard deviation and then one for constant columns.
The reported headline is the unweighted mean across continuous columns. KS statistics are also
reported descriptively.

### Categorical marginals

Jensen-Shannon divergence uses the union of real and synthetic levels and base-2 logarithms:

\[
JSD(P,Q) = \frac{1}{2}KL(P\|M) + \frac{1}{2}KL(Q\|M), \quad M=(P+Q)/2.
\]

Missing values are retained as an explicit missing category. Total-variation distance is
reported descriptively.

### Dependency preservation

A mixed association matrix is computed with:

- absolute Pearson correlation for continuous-continuous pairs;
- Cramer's V for categorical-categorical pairs;
- correlation ratio for categorical-continuous pairs.

Dependency error is the mean absolute difference between upper-triangle entries of the real and
synthetic matrices. It does not claim to capture higher-order or causal structure.

### Coverage, missingness, and clinical validity

- Minimum category coverage is the smallest share of real levels present in synthetic data.
- Range coverage measures overlap of observed synthetic and real numeric ranges relative to the
  real range.
- Out-of-range rate counts generated numeric values beyond the real observed range.
- Missingness gap is the largest absolute difference in column missingness rates.
- `clinical_constraint_violation_rate` is the maximum finite per-rule violation rate,
  with each rule's denominator restricted to rows with all required values present.
  Each constraint reports total eligible rows, evaluated rows, excluded-missing rows,
  violating rows, its violation rate, and availability. No evaluable rules returns `None`.
  Missing values are always excluded in this language version, including self-comparisons;
  there is no explicit missingness predicate. Missingness is evaluated separately.

**Failure logic:** Missing categories or clinical violations can trigger `Fail`; large marginal,
dependency, or missingness discrepancies trigger `Conditional`. Missing clinical constraints
produce `NotEvaluated`, not an invented zero-violation result. Any unavailable configured
constraint preserves unavailable evidence; a threshold-crossing evaluable rule still takes
precedence as `Fail`.

## Utility

**Question:** Can a model trained on synthetic data generalize to held-out real patients?

**Inputs:** all three tables, binary target, positive label, feature list, and random seed.

- TRTR: train on `real_train`, test on `real_test`.
- TSTR: train on `synthetic`, test on the identical `real_test` rows.
- The fixed v0.3 probe is a random forest with 200 trees and minimum leaf size five.
- Features are encoded once from real training data and reused for TRTR and TSTR.
- AUROC, AUPRC, and Brier score are reported for both regimes.

The primary retention quantity is:

\[
R_{chance} = \frac{AUROC_{TSTR}-0.5}{AUROC_{TRTR}-0.5}.
\]

The machine-readable key remains `auroc_utility_retention`. The paired difference
`tstr_minus_trtr_auroc` is TSTR AUROC minus TRTR AUROC,
with a paired held-out-row bootstrap interval.

**Decision logic:** If TRTR AUROC's lower 95% confidence bound is at or below 0.50,
retention is `NotEvaluated`. Otherwise, TSTR AUROC's upper 95% bound at or below 0.50,
or retention's upper 95% bound below 0.80, produces `Fail`. Retention's lower 95% bound
at least 0.90 produces `Pass` when no failure applies; remaining evaluable cases are
`Conditional`. Unavailable required intervals produce `NotEvaluated`. Thresholds are provisional.

## Privacy

Exact-copy analysis uses every synthetic and real-training row. Nearest-neighbor exposure
and the distance membership attack use reproducible bounded samples when a table exceeds
the configured `privacy.distance_max_rows` or `privacy.attack_max_rows`. Reports record the
actual reference, query, member, nonmember, and released sample sizes. This prevents
quadratic distance calculations from making large clinical tables impractical while keeping
the sampling decision visible.

**Question:** Does the released table expose information about real training records under the
tested threat model?

**Inputs:** all three tables, declared types, and canonicalization precision.

### Exact training copies

Rows are type-normalized, numeric values are rounded to the configured precision, and complete
row hashes are compared. The output is the share of synthetic rows matching a real training row.

### Nearest-neighbor exposure

Continuous values are standardized with real-training statistics and categorical values are
one-hot encoded from real-training levels. SynTrustBench reports:

- median synthetic-to-training closest-record distance;
- share closer than the fifth percentile of non-self real-to-real distances;
- fifth percentile of the closest/second-closest distance ratio.

These are empirical indicators, not formal privacy guarantees.

### Distance membership attack

The attacker observes the released synthetic table and a candidate record. Its score is the
negative distance from the candidate to the closest synthetic row. Members are `real_train`;
nonmembers are held-out `real_test`. Attack AUROC and true-positive rate at 1% false-positive
rate are reported.

**Failure logic:** Any exact training copy fails. Membership-AUROC lower 95% CI > 0.55
produces `Fail`; upper 95% CI <= 0.55 produces `Pass`, subject to other privacy checks;
an interval crossing 0.55 produces `Conditional`. Inadequate member/nonmember sample sizes
make the required attack `NotEvaluated`. Elevated nearest-neighbor exposure is `Conditional`.
Fail takes precedence over unavailable evidence and Conditional.

TPR at 1% FPR remains reported as exploratory, not decision-driving, because sufficiently
reliable low-FPR uncertainty is not currently implemented. Its legacy threshold key is
accepted but ignored by v0.3 gating. All thresholds remain provisional and attacker-specific.

## Equity

**Question:** Are representation, utility, and empirical privacy exposure acceptable across
clinically relevant or protected groups?

**Inputs:** protected attributes, all three tables, and the frozen TRTR/TSTR predictions.

For each subgroup, SynTrustBench reports:

- real-training, held-out real-test, and synthetic sample sizes;
- representation prevalence and absolute gap;
- subgroup TRTR and TSTR AUROC with bootstrap intervals;
- chance-corrected subgroup utility retention, `(TSTR AUROC_g - 0.5) / (TRTR AUROC_g - 0.5)`,
  with a paired bootstrap interval;
- exact-copy and nearest-neighbor exposure among synthetic subgroup rows.

Subgroups below the configured total or per-class counts are labeled `Insufficient evidence`.
Subgroup retention is calculated only when the existing subgroup TRTR CI lower bound exceeds
0.50. Otherwise its estimate and bounds are `None` and `utility_retention_status` is
`NotEvaluated`; evaluable retention has status `Available`. Bootstrap samples with TRTR
AUROC <= 0.50 are excluded as nonfinite. Unavailable descriptive retention does not remove
the subgroup from the existing TSTR-based Equity gate.

The maximum utility gap is calculated **within each protected attribute**, then summarized by
the largest within-attribute gap. Values from unrelated attributes are never subtracted.

**Failure logic:** Worst-group TSTR AUROC below 0.60 fails. Representation or within-attribute
utility gaps above 0.10 are conditional. Missing or underpowered subgroup evidence is reported
without fabricating a stable score.

## Core Predictive-Utility Robustness

The machine-readable dimension key remains `robustness`.

**Question:** Do benchmark conclusions remain stable under resampling, model refits, missingness,
reduced training data, generator seeds, and available shifts?

**Inputs:** the utility task, perturbation configuration, optional independently generated
synthetic tables, and optional time/site column.

Required core tests:

1. Held-out bootstrap stability: relative width of the TSTR AUROC interval.
2. Downstream-model refit stability: coefficient of variation across configured classifier
   seeds. This is explicitly not generator stability.
3. MCAR missingness: TSTR AUROC after configured missingness rates, repeated under three
   perturbation seeds.
4. Reduced synthetic training size: TSTR AUROC across configured fractions.

Optional when supplied:

5. Generator-seed stability across independently generated synthetic tables.
6. Temporal evaluation across sufficiently sized held-out groups.
7. Site evaluation across sufficiently sized held-out groups.

The existing combined shift metric evaluates the configured time column first, otherwise
the site column; the other check is recorded as optional and unavailable.

**Failure logic:** Both baseline and maximum-missingness retention use the chance-corrected
formula. Missingness retention is `(AUROC_missingness - 0.5) / (AUROC_TRTR - 0.5)` and is
calculated only when Utility established TRTR as reliably above chance. Baseline retention
>= `utility_retention_fail` and perturbed retention < that same threshold (default 0.80)
produces a conclusion reversal and `Fail`. Unavailable retention cannot create a reversal.
The separate missingness AUROC-drop calculation and threshold are unchanged.
Other large spreads, drops, or coefficients of variation are conditional.
Unavailable required checks produce `NotEvaluated` unless a failure takes precedence.
Unavailable optional checks remain `NotEvaluated` in metric details, labeled
`Not evaluated — optional`, and do not prevent a core Pass.

## Aggregation

Required-metric aggregation within a dimension is: any `Fail` -> `Fail`; otherwise any
`NotEvaluated` -> `NotEvaluated`; otherwise any `Conditional` -> `Conditional`; otherwise
`Pass`. Optional/descriptive unavailability does not become a required failure.
Reports explicitly list unavailable required evidence. Across dimensions:

- any dimension `Fail` -> benchmark gate `Fail`;
- otherwise any dimension `NotEvaluated` or `Conditional` -> benchmark gate `Conditional`;
- all dimensions `Pass` -> benchmark gate `Pass`.

This gate is a provisional benchmark decision. It is not a certification, formal safety
determination, or substitute for use-case-specific review.
