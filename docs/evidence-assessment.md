# Relationship to Evidence Assessment

SynTrustBench is one framework with two complementary components.

## Component 1: Evidence Assessment

Evaluates whether a paper, supplement, repository, and configuration provide enough evidence to
interpret trustworthiness claims. Outputs include the Evidence Maturity Profile, evidence
evaluability gate, and dimension-specific missing-evidence report.

The question is:

> Is the published claim supported by sufficiently complete and reproducible evidence?

## Component 2: Executable Evaluation

Evaluates actual real and synthetic datasets under the versioned tabular protocol implemented in
this package. Outputs include metric estimates, uncertainty, subgroup results, stress tests,
warnings, and a non-compensable benchmark gate.

The question is:

> How does this synthetic dataset behave when measured under a consistent protocol?

## The distinction

Evidence maturity and dataset quality are not interchangeable:

- A transparent study can document that a synthetic dataset performs poorly.
- A dataset can report apparently strong numbers while its evidence remains impossible to verify.

The manuscript should apply both components but report their outputs separately. The executable
benchmark gate must never be called an evaluability gate.
