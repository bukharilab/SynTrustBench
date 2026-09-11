"""Top-level executable evaluation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .equity import evaluate_equity
from .fidelity import evaluate_fidelity
from .models import BenchmarkResult
from .privacy import evaluate_privacy
from .reporting import write_outputs
from .robustness import evaluate_robustness
from .utility import evaluate_utility
from .validation import load_submission


def evaluate(
    real_train: str | Path | pd.DataFrame,
    real_test: str | Path | pd.DataFrame,
    synthetic: str | Path | pd.DataFrame,
    config: str | Path | dict[str, Any],
    output_dir: str | Path,
    *,
    synthetic_replicates: list[str | Path | pd.DataFrame] | None = None,
    progress: Callable[[str], None] | None = None,
) -> BenchmarkResult:
    """Run the complete v0.3 structured tabular protocol."""

    report_progress = progress or (lambda _message: None)
    report_progress("Loading and validating submission")
    submission = load_submission(
        real_train,
        real_test,
        synthetic,
        config,
        synthetic_replicates=synthetic_replicates,
    )
    seed_sequence = np.random.SeedSequence(submission.config.random_seed)
    rng_fidelity, rng_utility, rng_privacy, rng_equity, rng_robustness = [
        np.random.default_rng(seed)
        for seed in seed_sequence.spawn(5)
    ]

    report_progress("Evaluating fidelity")
    fidelity = evaluate_fidelity(submission, rng_fidelity)
    report_progress("Evaluating utility")
    utility = evaluate_utility(submission, rng_utility)
    report_progress("Evaluating privacy")
    privacy = evaluate_privacy(submission, rng_privacy)
    report_progress("Evaluating equity")
    equity = evaluate_equity(submission, utility, privacy, rng_equity)
    report_progress("Evaluating robustness")
    robustness = evaluate_robustness(submission, utility, rng_robustness)
    dimensions = {
        "fidelity": fidelity,
        "utility": utility,
        "privacy": privacy,
        "equity": equity,
        "robustness": robustness,
    }
    report_progress("Writing benchmark report")
    summary, gate = write_outputs(submission, dimensions, output_dir)
    report_progress("SynTrustBench evaluation complete")
    return BenchmarkResult(
        dimensions=dimensions,
        benchmark_gate=gate,
        summary=summary,
        output_dir=Path(output_dir),
    )
