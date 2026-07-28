# Metric definitions: structured tabular protocol v0.2

Each dimension returns metric estimates, uncertainty where defensible, explicit availability,
warnings, and a provisional status. Dimension statuses are non-compensable. No weighted overall
trust score is calculated.

## Common uncertainty rules

- Predictive metrics use row-level bootstrap resampling of the frozen held-out real test set.
- TRTR/TSTR retention uses the same bootstrap rows for numerator and denominator.
- Distribution distances use independent bootstrap resamples of real training and synthetic
  rows. A bootstrap standard-error interval is centered on the observed distance because naive
  percentile intervals for non-negative distances are biased upward near zero.
- Copy, proximity, and subgroup statistics resample their relevant rows.
- Ranges across model seeds, generator seeds, subgroups, or perturbations are labeled observed
  ranges, not confidence intervals.
- The default is 200 resamples at 95% confidence. Both values are recorded in the run manifest.

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
- Clinical violation rate is the share of synthetic rows violating at least one valid,
  user-supplied constraint.

**Failure logic:** Missing categories or clinical violations can trigger `Fail`; large marginal,
dependency, or missingness discrepancies trigger `Conditional`. Missing clinical constraints
produce `Conditional`, not an invented zero-violation result.

## Utility

**Question:** Can a model trained on synthetic data generalize to held-out real patients?

**Inputs:** all three tables, binary target, positive label, feature list, and random seed.

- TRTR: train on `real_train`, test on `real_test`.
- TSTR: train on `synthetic`, test on the identical `real_test` rows.
- The fixed v0.2 probe is a random forest with 200 trees and minimum leaf size five.
- Features are encoded once from real training data and reused for TRTR and TSTR.
- AUROC, AUPRC, and Brier score are reported for both regimes.

The primary retention quantity is:

\[
R_{AUROC} = \frac{AUROC_{TSTR}}{AUROC_{TRTR}}.
\]

**Failure logic:** Retention below 0.80 fails; below 0.90 is conditional. A confidence interval
crossing the conditional boundary prevents an unqualified pass. An upper TSTR AUROC confidence
bound at or below 0.50 fails.

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

**Failure logic:** Any exact training copy fails. An attack whose AUROC lower confidence bound
exceeds 0.55 or whose TPR at 1% FPR exceeds 0.05 fails. Elevated nearest-neighbor exposure is
conditional. These thresholds are provisional, attacker-specific research defaults.

## Equity

**Question:** Are representation, utility, and empirical privacy exposure acceptable across
clinically relevant or protected groups?

**Inputs:** protected attributes, all three tables, and the frozen TRTR/TSTR predictions.

For each subgroup, SynTrustBench reports:

- real-training, held-out real-test, and synthetic sample sizes;
- representation prevalence and absolute gap;
- subgroup TRTR and TSTR AUROC with bootstrap intervals;
- subgroup utility retention;
- exact-copy and nearest-neighbor exposure among synthetic subgroup rows.

Subgroups below the configured total or per-class counts are labeled `Insufficient evidence`.

The maximum utility gap is calculated **within each protected attribute**, then summarized by
the largest within-attribute gap. Values from unrelated attributes are never subtracted.

**Failure logic:** Worst-group TSTR AUROC below 0.60 fails. Representation or within-attribute
utility gaps above 0.10 are conditional. Missing or underpowered subgroup evidence is reported
without fabricating a stable score.

## Robustness

**Question:** Do benchmark conclusions remain stable under resampling, model refits, missingness,
reduced training data, generator seeds, and available shifts?

**Inputs:** the utility task, perturbation configuration, optional independently generated
synthetic tables, and optional time/site column.

Required v0.2 tests:

1. Held-out bootstrap stability: relative width of the TSTR AUROC interval.
2. Downstream-model refit stability: coefficient of variation across configured classifier
   seeds. This is explicitly not generator stability.
3. MCAR missingness: TSTR AUROC after configured missingness rates, repeated under three
   perturbation seeds.
4. Reduced synthetic training size: TSTR AUROC across configured fractions.

Optional when supplied:

5. Generator-seed stability across independently generated synthetic tables.
6. Temporal/site shift gap across sufficiently sized held-out groups.

**Failure logic:** A utility conclusion that crosses the failure boundary under missingness
fails robustness. Other large spreads, drops, or coefficients of variation are conditional.
Unavailable optional tests remain visible as `Not evaluated`.

## Aggregation

Within a dimension, minimum requirements use explicit logical gates. Across dimensions:

- any dimension `Fail` -> benchmark gate `Fail`;
- otherwise any dimension `Conditional` -> benchmark gate `Conditional`;
- all dimensions `Pass` -> benchmark gate `Pass`.

This gate is a provisional benchmark decision. It is not a certification, formal safety
determination, or substitute for use-case-specific review.
