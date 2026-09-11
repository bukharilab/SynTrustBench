# Changelog

## 0.3.0

- Changed overall and subgroup AUROC retention to chance-corrected retention, unavailable
  when the corresponding TRTR lower confidence bound does not exceed chance. Added the
  paired TSTR-minus-TRTR AUROC difference and CI-based Utility gating.
- Applied the same chance-corrected retention to missingness conclusion reversal.
- Added metric/dimension `NotEvaluated` semantics; unavailable required evidence prevents
  an overall Pass but does not override a measured Fail.
- Labeled Core Predictive-Utility Robustness and separated required checks from optional
  generator-seed, temporal, and site checks.
- Corrected privacy CI gating; TPR at 1% FPR is exploratory and not decision-driving.
  Legacy threshold keys remain accepted without changing their values.
- Made exact train/test row overlap fatal. Added optional `data.entity_id` validation,
  requiring nonmissing identifiers in every real row and rejecting normalized cross-split overlap.
- Changed clinical-constraint aggregation to the maximum evaluable per-rule violation rate;
  missing required values are excluded independently for each rule.

## Unreleased

- Integrated the evidence assessment and executable tabular protocol in one repository.
- Added the installable `syntrustbench` package and command-line interface.
- Added fidelity, utility, privacy, equity, and robustness reports with uncertainty and
  failure flags.
- Added bounded, reproducible nearest-neighbor privacy evaluation for large tables.
- Added runnable demo data, schemas, tests, continuous integration, and a tutorial notebook.
- Replaced standards and compliance terminology with benchmark-framework terminology.
- Reorganized manuscript, audit, documentation, and software files into reader-facing folders.

## 0.1.0 — 2026-07-21

- Released the 30-report evidence corpus, maturity profile, evaluability gate, evidence card,
  and reproducible audit analysis.
