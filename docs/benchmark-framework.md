# SynTrustBench benchmark framework

## Status and scope

SynTrustBench is a proposed, versioned evaluation framework. It is not a formal standard,
regulatory certification, or claim that a synthetic dataset is safe for every use.

The framework has two connected components:

- **Evidence Assessment** examines whether a published trustworthiness claim is complete,
  reproducible, and interpretable.
- **Executable Evaluation** calculates trustworthiness metrics from real training data, a
  held-out real test set, synthetic data, and a machine-readable configuration.

The executable release currently supports structured tabular clinical data. The broader
framework identifies questions relevant to text, images, waveforms, and multimodal records,
but those modalities do not yet have executable protocols.

## Benchmark identity

An executable result is identified by:

```text
benchmark version
dataset and cohort version
task definition
split hash
generator and seed
evaluation configuration
software version
```

Results should only be compared when these elements are compatible. A changed cohort,
split, threat model, or metric implementation creates a different evaluation context.

## Five dimensions

| Dimension | Question |
|---|---|
| Fidelity | Does the synthetic table preserve relevant distributions, dependencies, support, missingness, and clinical constraints? |
| Utility | Can a model trained on synthetic data generalize to held-out real patients relative to a real-trained baseline? |
| Privacy | Does releasing the synthetic table expose measurable information about real training records under a declared threat model? |
| Equity | Are representation, utility, and privacy outcomes consistent across protected or clinically important subgroups? |
| Robustness | Do conclusions remain stable across resampling, perturbations, training sizes, shifts, and generator seeds when available? |

The dimensions are non-compensable. SynTrustBench reports a profile rather than a weighted
overall score.

## Component 1: Evidence Assessment

### Input

A manuscript and any supporting artifacts available to the assessor, including
supplementary material, code, configuration, data statements, and reported results.

### Output

- a five-dimensional Evidence Maturity Profile;
- an evaluability status of `Pass`, `Conditional`, or `Fail`;
- dimension-specific missing-evidence notes;
- reproducibility warnings;
- a completed evidence card.

Evidence maturity describes the quality of support for a claim, not the quality of the
synthetic dataset. A transparent study can document poor model performance well; a strong
reported result can still be unevaluable.

## Component 2: Executable Evaluation

### Required input

```text
real_train.csv
real_test.csv
synthetic.csv
config.yaml
```

The generator may use `real_train.csv`, but it must not use `real_test.csv`. The synthetic
table must follow the same analytic schema. The configuration declares column types,
prediction target, positive label, protected attributes, clinical constraints, uncertainty
settings, robustness tests, and generator metadata.

### Output

Each run writes metric-level estimates, confidence intervals, subgroup results, warnings,
failure flags, a benchmark card, a readable report, input hashes, and an execution manifest.

### Minimum tabular protocol

| Dimension | Current calculations |
|---|---|
| Fidelity | normalized Wasserstein distance, categorical Jensen–Shannon divergence, mixed dependency error, support coverage, missingness gap, and constraint violations |
| Utility | TRTR and TSTR AUROC, AUPRC, Brier score, and utility retention on the same held-out real rows |
| Privacy | exact training copies, nearest-neighbor exposure, and a distance-based membership attack with an explicit threat model |
| Equity | subgroup representation, subgroup TRTR/TSTR performance, worst-group performance, within-attribute gaps, and subgroup privacy indicators |
| Robustness | held-out bootstrap stability, missingness stress, training-size stress, downstream-model seeds, optional generator seeds, and optional temporal or site shift |

Exact calculations and uncertainty procedures are defined in [metrics.md](metrics.md).

## Reporting decisions

### Uncertainty

Primary estimates include the confidence level and resampling unit. Generator-level
stability is only reported when independently generated synthetic tables are supplied.
Dataset bootstrap stability is not relabeled as generator-seed stability.

### Privacy

Every privacy interpretation states:

1. what the attacker knows;
2. what the attacker can access;
3. what the attacker is trying to infer;
4. how members and nonmembers are constructed;
5. which attack and operating point are used.

Distance from training data is an empirical indicator, not a formal privacy guarantee.
Differential-privacy claims require complete mechanism and accounting information.

### Equity

Subgroup comparisons include sample sizes and uncertainty. A subgroup that is too small for
stable estimation is reported as `Insufficient evidence` rather than assigned a misleading
numeric score.

### Gate

`Pass`, `Conditional`, and `Fail` are provisional benchmark decisions based on
dimension-level requirements. They are not regulatory judgments. A severe privacy or
subgroup failure cannot be offset by stronger fidelity.

## Versioning

- Metric additions and threshold changes increment the minor protocol version.
- Changes to cohort, task, or dimension semantics require a major version change.
- Corrections to the evidence corpus are recorded in `data/change_log.csv`.
- Public repository artifacts must never contain restricted clinical data.
- Known limitations and unavailable tests remain visible in the benchmark card and report.
