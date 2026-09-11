# Executable protocol input and output

## Scope

Protocol v0.3 evaluates one structured analytic table per split. It does not claim to evaluate
clinical text, images, waveforms, longitudinal event streams, or multimodal records. Those
modalities require separate feature, attack, and task protocols.

## Required inputs

### `real_train.csv`

Real records used directly or indirectly to fit the synthetic-data generator. Privacy copy and
proximity checks use this split as the potentially exposed membership set. TRTR trains its
reference model on this split.

### `real_test.csv`

Real records held out from generator training. Both TRTR and TSTR are tested on these same rows.
The distance-based membership attack uses this split as nonmembers. Exact train/test row
overlap stops the run with an error reporting the number of overlapping held-out rows.

### `synthetic.csv`

One generated table following the declared analytic schema. A collapsed target or missing
subgroup is treated as measured generator behavior rather than silently repaired.

### `config.yaml`

The configuration declares:

- dataset name and `modality: tabular`;
- mutually exclusive continuous, categorical, and binary column lists;
- binary prediction target and positive label;
- optional explicit utility features or exclusions;
- optional `data.entity_id` for real-split entity independence checks;
- protected or clinically relevant subgroup attributes;
- row-level clinical constraints;
- generator identity and seed;
- bootstrap and random-seed settings;
- subgroup sample-size requirements;
- robustness perturbations;
- optional temporal or site shift columns;
- optional overrides to provisional thresholds.

See `configs/example.yaml` and `schemas/config.schema.json`.

Large-table nearest-neighbor calculations are bounded by
`privacy.distance_max_rows` and `privacy.attack_max_rows`. Exact-copy analysis still uses
the complete tables. The report records all distance and attack sample sizes.

### Optional entity identifier

```yaml
data:
  entity_id: patient_id
```

The column must exist in both real splits, and every real row must have a nonmissing ID.
Missing IDs stop validation with the column name, affected split, and missing count.
IDs use a common representation: surrounding text whitespace is stripped and finite numeric
values/text are compared as exact decimals (e.g. `1`, `1.0`, and `"1"` match, as do numeric
strings with leading zeros). Textual IDs remain case-sensitive. Any cross-split overlap stops
the run and reports the number of overlapping unique identifiers.
The ID is excluded from automatically selected utility features. It can be used only when
explicitly listed in `utility_task.features` and declared in the existing column-type configuration.
No column name is automatically inferred. Without `data.entity_id`, reports state:

> Only exact row overlap was checked; patient-level independence could not be verified.

## Optional generator-seed inputs

Repeated `--synthetic-replicate` arguments accept independently generated tables. They must use
the same schema and should differ only by generator seed. Without them, SynTrustBench reports
generator-seed stability as `NotEvaluated`, labeled `Not evaluated — optional`; it does not relabel downstream model refits as
generator stability.

## Validation behavior

The run stops with an error when:

- a required file or column is missing;
- a declared column appears in more than one type list;
- the real train or held-out test target is missing or not binary;
- the configured positive label is absent from either real split;
- a protected attribute or utility feature is undeclared;
- the task or modality is unsupported;
- a table is empty;
- real train/test rows overlap;
- a configured entity ID is absent, missing in any real row, or shared across real splits.

Nonfatal conditions, including rare outcomes, are recorded as
warnings in `summary.json`, `report.md`, and `execution_log.txt`.

## Outputs

### `summary.json`

Top-level run result containing dimension statuses, minimum-requirement benchmark gate, metric
coverage, failure flags, warnings, and all public metric details.

### `metrics.csv`

One row per metric with:

- estimate;
- interval or observed range;
- interval type;
- direction;
- availability;
- failure flag;
- provisional threshold;
- interpretation note.

An observed range across seeds or perturbations is never labeled as a confidence interval.

### `subgroup_results.csv`

One row per attribute and subgroup with real and synthetic sample sizes, representation gap,
TRTR/TSTR AUROC and confidence intervals, chance-corrected utility retention and its
`utility_retention_status`, training-copy rate, and proximity
exposure. Small cells are marked `Insufficient evidence` instead of receiving unstable utility
scores.

### `benchmark_card.yaml`

Compact run card describing the protocol, dataset, generator, split contract, dimension profile,
privacy threat model, missing tests, and interpretation boundaries.

### `report.md`

Readable report for researchers and reviewers. It intentionally avoids a single weighted score.

### `run_manifest.json`

Resolved configuration, SHA-256 hashes of all supplied inputs, random seeds, package versions,
and runtime platform.

### `execution_log.txt`

Concise audit log suitable for CI and repository artifacts.

### `details.json`

Per-column fidelity results, per-constraint counts/rates/availability, and robustness curves
that would make the main report too dense. Clinical-constraint aggregation is the maximum
evaluable per-rule violation rate; missing required values are excluded for each rule.

## Benchmark gate versus evidence evaluability

The executable `Pass`, `Conditional`, or `Fail` gate summarizes provisional minimum dataset
requirements. It does **not** decide whether a paper is evaluable. Evidence evaluability belongs
to Component 1, Evidence Assessment, and is based on reporting completeness and reproducibility.

Metrics and dimensions preserve `NotEvaluated` when required evidence is unavailable.
Required-metric precedence is Fail, NotEvaluated, Conditional, then Pass. Overall precedence
is Fail, then Conditional for any NotEvaluated or Conditional dimension, then Pass.
Reports list unavailable required evidence. Core Predictive-Utility Robustness requires
model-seed variation, bootstrap uncertainty, missingness perturbation, and training-size
variation; generator-seed, temporal, and site checks are optional when unavailable.
