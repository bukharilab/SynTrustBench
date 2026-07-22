# SynTrustBench

SynTrustBench is an evidence-gated benchmark specification for evaluating how convincingly synthetic clinical data studies establish trustworthiness. Version 0.1.0 is a **benchmark specification and pilot corpus audit**, not a model-performance leaderboard.

The artifact has two purposes:

1. turn claims such as "realistic," "private," or "clinically useful" into testable evidence requirements; and
2. audit the 30 reports synthesized in the companion manuscript against the same requirements.

## What v0.1.0 contains

- `benchmark_spec.md` — normative benchmark definition;
- `data/study_annotations.csv` — one row per included report;
- `data/study_annotations.jsonl` — typed rows validated by the JSON Schema;
- `data/SynTrustBench_corpus_audit.xlsx` — point-in-time export of the canonical audit workbook, including the preserved original tab, codebook, evidence summary, change log, and related-benchmark table;
- `data/change_log.csv` — dated, source-linked material corrections;
- `data/coding_codebook.md` — extraction and maturity-scoring rules;
- `data/annotation_schema.json` — machine-readable field schema;
- `benchmark_card.yaml` — submission/reporting card;
- `benchmark_card.schema.json` — machine-validating card schema;
- `examples/ppgan_benchmark_card.yaml` — completed evidence-card example;
- `scripts/build_annotations.py` — single source for the 62-field corpus and JSON Schema;
- `scripts/score_evidence.py` — validation and code-generated audit summaries;
- `requirements.txt` — Python dependencies for validation and figures;
- `results/` — generated evidence-audit outputs;
- `arxiv/main.tex` and `arxiv/references.bib` — arXiv manuscript source.

## Scope and claim boundary

SynTrustBench v0.1.0 scores the **maturity of reported evidence**, not the quality of a generative model. A high evidence score means that a claim was tested rigorously and reported sufficiently; it does not mean that the model performed well. Missing evidence is coded `NR` (not reported), never silently converted into model failure.

The later model-execution track will compare systems only within a frozen tuple of dataset, cohort, task, split, modality protocol, and benchmark version. No cross-paper leaderboard is produced from heterogeneous published numbers.

## Reproduce the pilot audit

```bash
python -m pip install -r requirements.txt
python scripts/build_annotations.py
python scripts/validate_release.py
python scripts/score_evidence.py \
  --input data/study_annotations.csv \
  --output-dir results \
  --figure-dir figures
```

The script validates the 30-record corpus, recomputes all counts and percentages, checks maturity values and differential-privacy fields, and writes the tables consumed by the manuscript.

## Status

Version: `0.1.0 release candidate`  
Evidence window: January 2018–December 2025  
Last source verification: 2026-07-21  
Independent human double-coding: pending coauthor adjudication; no inter-rater statistic is claimed in v0.1.0 until completed.

## Citation

See `CITATION.cff`. The permanent repository URL and archival DOI must be inserted only after the repository is made public and archived.
