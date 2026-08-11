# SynTrustBench

SynTrustBench evaluates trustworthiness claims about synthetic clinical data in two
complementary ways:

1. **Evidence assessment** asks whether a published claim is documented well enough to
   reproduce and interpret.
2. **Executable evaluation** measures a structured synthetic dataset against its real
   training data and a held-out real test set.

Both components report fidelity, utility, privacy, equity, and robustness separately.
SynTrustBench does not average them into a single score: strong fidelity cannot cancel a
privacy failure or a large subgroup gap.

The executable protocol currently supports **structured tabular clinical data**. Text,
imaging, waveform, and multimodal protocols require modality-specific evaluation and are
not implemented in this release.

## Install

SynTrustBench requires Python 3.10 or newer.

```bash
git clone https://github.com/bukharilab/SynTrustBench.git
cd SynTrustBench
python -m pip install -e .
```

Check the installation:

```bash
syntrustbench --version
```

## Run the example

The example contains a small simulated cohort for checking the software. It is not a
clinical validation dataset.

```bash
syntrustbench validate \
  --real-train examples/demo/real_train.csv \
  --real-test examples/demo/real_test.csv \
  --synthetic examples/demo/synthetic.csv \
  --config configs/example.yaml

syntrustbench evaluate \
  --real-train examples/demo/real_train.csv \
  --real-test examples/demo/real_test.csv \
  --synthetic examples/demo/synthetic.csv \
  --config configs/example.yaml \
  --output runs/demo
```

A completed run writes a readable report, metric-level CSV files, a benchmark card,
machine-readable JSON, input hashes, and an execution log. Generated runs belong under
`runs/`, which is intentionally ignored by Git.

The same workflow is available from Python:

```python
from syntrustbench import evaluate

result = evaluate(
    real_train="examples/demo/real_train.csv",
    real_test="examples/demo/real_test.csv",
    synthetic="examples/demo/synthetic.csv",
    config="configs/example.yaml",
    output_dir="runs/demo",
)

print(result.benchmark_gate)
print(result.summary["dimension_status"])
```

## Required inputs

| File | Purpose |
|---|---|
| `real_train.csv` | Real records used to train the data generator |
| `real_test.csv` | Held-out real records never shown to the generator |
| `synthetic.csv` | Synthetic records with the same analytic schema |
| `config.yaml` | Column types, target, protected attributes, constraints, seeds, and uncertainty settings |

See [Executable inputs and outputs](docs/executable-inputs-and-outputs.md) for the complete
contract and [Metric definitions](docs/metrics.md) for the calculations.

## Repository layout

```text
src/syntrustbench/         Python package and command-line interface
tests/                     Package tests
configs/                   Example configuration and evidence card
examples/demo/             Small runnable input files
schemas/                   Machine-readable contracts
docs/                      Benchmark and metric documentation
data/                      Literature-audit annotations and codebook
analysis/evidence-audit/   Generated evidence-audit tables
scripts/evidence/          Evidence corpus and scoring scripts
scripts/experiments/       Controlled failure-injection runner
notebooks/                 Thin tutorials built on the package
paper/                     Manuscript source, figures, and frozen release files
```

There is one repository because the evidence assessment and executable evaluation are two
parts of the same benchmark. A separate package repository would make sense only if the
software later adopts its own release cycle and maintainer community.

## Evidence assessment

The current corpus contains 30 source-audited reports. The audit records what each report
documents; it does not treat missing reporting as proof that a model performs badly.

To reproduce the audit:

```bash
python scripts/evidence/build_annotations.py
python scripts/evidence/validate_release.py
python scripts/evidence/score_evidence.py \
  --input data/study_annotations.csv \
  --output-dir analysis/evidence-audit \
  --figure-dir paper/figures
```

The corpus remains a bounded pilot inherited from the earlier review. Independent
double-coding and broader corpus expansion remain open work.

## Scientific status

SynTrustBench is an open, versioned benchmark framework—not a regulatory certification or
formal standard. Current thresholds are provisional research decisions. The bundled demo
proves that the software runs; it does not establish clinical generalizability.

The next empirical release should freeze one public clinical cohort and task, compare at
least two generators, use repeated generator seeds, and test controlled failures. Saved
outputs alone are not evidence that every failure is detected selectively.

## Documentation

- [Benchmark framework](docs/benchmark-framework.md)
- [Evidence assessment](docs/evidence-assessment.md)
- [Executable inputs and outputs](docs/executable-inputs-and-outputs.md)
- [Metric definitions](docs/metrics.md)
- [Thresholds](docs/thresholds.md)
- [Case-study protocol](docs/case-study.md)
- [Security and clinical-data handling](SECURITY.md)

## Citation and license

SynTrustBench: An Evidence-Gated and Executable Benchmark for Trustworthiness Claims in Synthetic Clinical Data
Neeam Shahriar Hayder, Syed Ahmad Chan Bukhari
doi: https://doi.org/10.64898/2026.08.05.26359803
