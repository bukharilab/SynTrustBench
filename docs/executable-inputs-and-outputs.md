# Executable protocol input and output

## Scope

Protocol v0.2 evaluates one structured analytic table per split. It does not claim to evaluate
clinical text, images, waveforms, longitudinal event streams, or multimodal records. Those
modalities require separate feature, attack, and task protocols.

## Required inputs

### `real_train.csv`

Real records used directly or indirectly to fit the synthetic-data generator. Privacy copy and
proximity checks use this split as the potentially exposed membership set. TRTR trains its
reference model on this split.

### `real_test.csv`

Real records held out from generator training. Both TRTR and TSTR are tested on these same rows.
The distance-based membership attack uses this split as nonmembers. Train/test row overlap is
reported as a warning because it can bias utility and privacy results.

### `synthetic.csv`

One generated table following the declared analytic schema. A collapsed target or missing
subgroup is treated as measured generator behavior rather than silently repaired.

### `config.yaml`

The configuration declares:

- dataset name and `modality: tabular`;
- mutually exclusive continuous, categorical, and binary column lists;
- binary prediction target and positive label;
- optional explicit utility features or exclusions;
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

## Optional generator-seed inputs

Repeated `--synthetic-replicate` arguments accept independently generated tables. They must use
the same schema and should differ only by generator seed. Without them, SynTrustBench reports
generator-seed stability as `Not evaluated`; it does not relabel downstream model refits as
generator stability.

## Validation behavior

The run stops with an error when:

- a required file or column is missing;
- a declared column appears in more than one type list;
- the real train or held-out test target is missing or not binary;
- the configured positive label is absent from either real split;
- a protected attribute or utility feature is undeclared;
- the task or modality is unsupported;
- a table is empty.

Nonfatal conditions, including real train/test overlap and rare outcomes, are recorded as
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
TRTR/TSTR AUROC and confidence intervals, utility retention, training-copy rate, and proximity
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

Per-column fidelity results and robustness curves that would make the main report too dense.

## Benchmark gate versus evidence evaluability

The executable `Pass`, `Conditional`, or `Fail` gate summarizes provisional minimum dataset
requirements. It does **not** decide whether a paper is evaluable. Evidence evaluability belongs
to Component 1, Evidence Assessment, and is based on reporting completeness and reproducibility.
