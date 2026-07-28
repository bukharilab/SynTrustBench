# Proposed first clinical case study

## Objective

Demonstrate that SynTrustBench produces differentiated, reproducible findings and detects known
synthetic-data failures. The objective is not to declare a universally best generator.

## Dataset

**UCI Diabetes 130-US Hospitals for Years 1999-2008**

- Public tabular clinical dataset with hospital encounters for patients with diabetes.
- Contains demographic variables including race, sex, and age bands.
- Supports a clinically interpretable readmission task without PhysioNet credentialing.
- Official source: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

### Cohort decisions to freeze before generation

- Retain one analytically justified encounter per patient, or use a patient-level group split.
- Define outcome as readmission within 30 days versus all other outcomes.
- Split by patient identifier before generator training.
- Fit preprocessing on `real_train` only.
- Exclude identifiers and leakage-prone post-outcome fields.
- Pre-register protected attributes and minimum subgroup sizes.

## Generators

Minimum:

1. GaussianCopulaSynthesizer as a transparent statistical baseline.
2. CTGANSynthesizer as an established adversarial tabular generator.

Stronger:

3. TVAESynthesizer as a variational neural baseline.

Use the same training table, analytic schema, generated row count, and documented seeds. CTGAN and
TVAE are available through the Synthetic Data Vault ecosystem:
https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers

## Downstream task

Binary prediction of 30-day readmission.

- Freeze one model family and hyperparameter set before comparing generators.
- Report TRTR and TSTR AUROC, AUPRC, and Brier score.
- Use the identical held-out real test set for every generator and seed.
- Report subgroup results for race, sex, and an age grouping chosen before analysis.

## Repeated runs

- Minimum paper: one documented seed per generator plus bootstrap uncertainty.
- Stronger paper: at least three generator seeds per model, supplied to the CLI as synthetic
  replicates.
- Downstream-model refit seeds remain a separate robustness result.

## Controlled failure injection

Apply the same degradation operators to one baseline generated table:

1. Insert selected real training rows while holding synthetic row count constant.
2. Reduce a preselected minority subgroup and resample remaining rows to preserve total size.
3. Shuffle a clinically important predictive feature while preserving its marginal distribution.
4. Add increasing MCAR missingness to selected predictor columns.

Pre-register the expected response before running:

| Failure | Primary expected response |
|---|---|
| Training-row insertion | exact-copy and proximity privacy indicators worsen |
| Subgroup reduction | representation gap and equity warnings worsen |
| Feature shuffling | dependency fidelity and TSTR utility worsen |
| Missingness | missingness fidelity worsens; robustness quantifies conclusion stability |

The Results section should report raw before/after metrics and uncertainty, not only status colors.
