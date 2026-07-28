"""Executable stability and stress tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import Submission
from .stats import metric, safe_auroc, train_classifier


def _coefficient_of_variation(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size < 2 or abs(finite.mean()) <= 1e-12:
        return float("nan")
    return float(finite.std(ddof=1) / abs(finite.mean()))


def _inject_missingness(
    frame: pd.DataFrame,
    columns: list[str],
    rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    perturbed = frame.copy()
    for column in columns:
        mask = rng.random(len(perturbed)) < rate
        perturbed.loc[mask, column] = np.nan
    return perturbed


def evaluate_robustness(
    submission: Submission,
    utility_result: dict,
    rng: np.random.Generator,
) -> dict:
    config = submission.config
    thresholds = config.thresholds
    internal = utility_result["_internal"]
    encoder = internal["encoder"]
    baseline_retention = internal["retention"]
    trtr_auroc = utility_result["metrics"]["trtr_auroc"]["estimate"]
    baseline_tstr = utility_result["metrics"]["tstr_auroc"]["estimate"]
    metrics: dict[str, dict] = {}
    curves: dict[str, dict] = {}
    warnings: list[str] = []
    weaknesses: list[str] = []

    model_seed_aurocs: dict[str, float] = {}
    for seed in config.model_seeds:
        labels, probabilities = train_classifier(
            submission.synthetic,
            submission.real_test,
            config.target,
            encoder,
            seed,
        )
        model_seed_aurocs[str(seed)] = safe_auroc(labels, probabilities)
    model_seed_cv = _coefficient_of_variation(list(model_seed_aurocs.values()))
    model_seed_flag = (
        np.isfinite(model_seed_cv)
        and model_seed_cv > thresholds["robustness_model_seed_cv_conditional"]
    )
    metrics["downstream_model_seed_auroc_cv"] = metric(
        model_seed_cv,
        ci_low=min(model_seed_aurocs.values()) if model_seed_aurocs else None,
        ci_high=max(model_seed_aurocs.values()) if model_seed_aurocs else None,
        interval_type="observed range across downstream model seeds",
        higher_is_worse=True,
        flag=model_seed_flag,
        threshold=(
            f"> {thresholds['robustness_model_seed_cv_conditional']:.2f} -> Conditional"
        ),
        note=(
            "Coefficient of variation of TSTR AUROC across fixed downstream "
            "classifier refits. This is not generator-seed stability."
        ),
    )
    curves["downstream_model_seed_auroc"] = model_seed_aurocs

    tstr_metric = utility_result["metrics"]["tstr_auroc"]
    if (
        tstr_metric["ci_width"] is not None
        and tstr_metric["estimate"] is not None
        and tstr_metric["estimate"] > 0
    ):
        relative_width = tstr_metric["ci_width"] / tstr_metric["estimate"]
    else:
        relative_width = float("nan")
    bootstrap_flag = (
        np.isfinite(relative_width)
        and relative_width > thresholds["robustness_bootstrap_width_conditional"]
    )
    metrics["tstr_bootstrap_relative_ci_width"] = metric(
        relative_width,
        ci_low=tstr_metric["ci_low"],
        ci_high=tstr_metric["ci_high"],
        interval_type="held-out real-test bootstrap interval being summarized",
        higher_is_worse=True,
        flag=bootstrap_flag,
        threshold=(
            f"> {thresholds['robustness_bootstrap_width_conditional']:.2f} -> Conditional"
        ),
        note="Width of the TSTR AUROC confidence interval divided by its estimate.",
    )

    missingness_results: dict[str, dict[str, float]] = {}
    maximum_rate = max(config.missingness_rates)
    maximum_rate_aurocs: list[float] = []
    for rate_index, rate in enumerate(config.missingness_rates):
        rate_values: list[float] = []
        for repetition in range(3):
            perturbation_rng = np.random.default_rng(
                config.random_seed + 10_000 + rate_index * 100 + repetition
            )
            perturbed = _inject_missingness(
                submission.synthetic,
                config.feature_columns,
                rate,
                perturbation_rng,
            )
            labels, probabilities = train_classifier(
                perturbed,
                submission.real_test,
                config.target,
                encoder,
                config.random_seed,
            )
            rate_values.append(safe_auroc(labels, probabilities))
        missingness_results[str(rate)] = {
            "mean_auroc": float(np.mean(rate_values)),
            "sd_auroc": float(np.std(rate_values, ddof=1)),
        }
        if rate == maximum_rate:
            maximum_rate_aurocs = rate_values
    curves["missingness_tstr_auroc"] = missingness_results
    maximum_rate_mean = float(np.mean(maximum_rate_aurocs))
    missingness_drop = baseline_tstr - maximum_rate_mean
    perturbed_retention = (
        maximum_rate_mean / trtr_auroc
        if trtr_auroc is not None and trtr_auroc > 0
        else float("nan")
    )
    conclusion_reversal = (
        np.isfinite(baseline_retention)
        and baseline_retention >= thresholds["utility_retention_fail"]
        and np.isfinite(perturbed_retention)
        and perturbed_retention < thresholds["utility_retention_fail"]
    )
    missingness_flag = (
        conclusion_reversal
        or missingness_drop > thresholds["robustness_missingness_drop_conditional"]
    )
    metrics["tstr_auroc_drop_at_maximum_missingness"] = metric(
        missingness_drop,
        ci_low=min(maximum_rate_aurocs),
        ci_high=max(maximum_rate_aurocs),
        interval_type="observed range across three perturbation seeds",
        higher_is_worse=True,
        flag=missingness_flag,
        threshold=(
            f"> {thresholds['robustness_missingness_drop_conditional']:.2f} or "
            "utility conclusion reversal -> Conditional/Fail"
        ),
        note=(
            f"Baseline TSTR AUROC minus mean TSTR AUROC after {maximum_rate:.0%} "
            "MCAR feature missingness."
        ),
    )

    training_size_results: dict[str, float] = {}
    for fraction in config.training_size_fractions:
        if fraction == 1.0:
            reduced = submission.synthetic
        else:
            reduced = submission.synthetic.sample(
                frac=fraction, random_state=config.random_seed
            )
        labels, probabilities = train_classifier(
            reduced,
            submission.real_test,
            config.target,
            encoder,
            config.random_seed,
        )
        training_size_results[str(fraction)] = safe_auroc(labels, probabilities)
    curves["training_size_tstr_auroc"] = training_size_results
    training_size_spread = float(
        max(training_size_results.values()) - min(training_size_results.values())
    )
    training_size_flag = (
        training_size_spread
        > thresholds["robustness_training_size_spread_conditional"]
    )
    metrics["training_size_auroc_spread"] = metric(
        training_size_spread,
        ci_low=min(training_size_results.values()),
        ci_high=max(training_size_results.values()),
        interval_type="observed range across configured training fractions",
        higher_is_worse=True,
        flag=training_size_flag,
        threshold=(
            f"> {thresholds['robustness_training_size_spread_conditional']:.2f} "
            "-> Conditional"
        ),
        note="Best-minus-worst TSTR AUROC across synthetic training-set fractions.",
    )

    generator_seed_frames = [submission.synthetic] + submission.synthetic_replicates
    if len(generator_seed_frames) >= 2:
        generator_seed_aurocs: dict[str, float] = {}
        for index, frame in enumerate(generator_seed_frames):
            labels, probabilities = train_classifier(
                frame,
                submission.real_test,
                config.target,
                encoder,
                config.random_seed,
            )
            generator_seed_aurocs[str(index)] = safe_auroc(labels, probabilities)
        generator_seed_cv = _coefficient_of_variation(
            list(generator_seed_aurocs.values())
        )
        generator_seed_flag = (
            np.isfinite(generator_seed_cv)
            and generator_seed_cv
            > thresholds["robustness_generator_seed_cv_conditional"]
        )
        metrics["generator_seed_tstr_auroc_cv"] = metric(
            generator_seed_cv,
            ci_low=min(generator_seed_aurocs.values()),
            ci_high=max(generator_seed_aurocs.values()),
            interval_type="observed range across supplied generator-seed datasets",
            higher_is_worse=True,
            flag=generator_seed_flag,
            threshold=(
                f"> {thresholds['robustness_generator_seed_cv_conditional']:.2f} "
                "-> Conditional"
            ),
            note="TSTR AUROC variation across independently supplied synthetic datasets.",
        )
        curves["generator_seed_tstr_auroc"] = generator_seed_aurocs
    else:
        generator_seed_flag = False
        metrics["generator_seed_tstr_auroc_cv"] = metric(
            None,
            higher_is_worse=True,
            status="Not evaluated",
            note=(
                "Supply one or more --synthetic-replicate files generated under "
                "different generator seeds to enable this test."
            ),
        )
        warnings.append(
            "Generator-seed stability was not evaluated; only one synthetic dataset was supplied."
        )

    shift_column = config.time_column or config.site_column
    if shift_column:
        shift_frame = submission.real_test
        if config.time_column:
            values = pd.to_numeric(shift_frame[shift_column], errors="coerce")
            split = float(values.median())
            groups = {
                "earlier": values <= split,
                "later": values > split,
            }
        else:
            groups = {
                str(level): shift_frame[shift_column].astype(str) == str(level)
                for level in sorted(shift_frame[shift_column].astype(str).unique())
            }
        shift_scores: dict[str, float] = {}
        for label, mask in groups.items():
            subset = shift_frame[mask]
            if (
                len(subset) < config.min_subgroup_n
                or subset[config.target].nunique() < 2
            ):
                continue
            labels, probabilities = train_classifier(
                submission.synthetic,
                subset,
                config.target,
                encoder,
                config.random_seed,
            )
            shift_scores[label] = safe_auroc(labels, probabilities)
        if len(shift_scores) >= 2:
            shift_gap = float(max(shift_scores.values()) - min(shift_scores.values()))
            shift_flag = (
                shift_gap > thresholds["robustness_shift_gap_conditional"]
            )
            metrics["temporal_or_site_shift_auroc_gap"] = metric(
                shift_gap,
                ci_low=min(shift_scores.values()),
                ci_high=max(shift_scores.values()),
                interval_type="observed range across temporal/site groups",
                higher_is_worse=True,
                flag=shift_flag,
                threshold=(
                    f"> {thresholds['robustness_shift_gap_conditional']:.2f} "
                    "-> Conditional"
                ),
                note=f"TSTR AUROC spread across groups defined by {shift_column}.",
            )
            curves["shift_tstr_auroc"] = shift_scores
        else:
            shift_flag = False
            metrics["temporal_or_site_shift_auroc_gap"] = metric(
                None,
                higher_is_worse=True,
                status="Not evaluated",
                note="Configured shift groups lacked sufficient evaluable held-out rows.",
            )
            warnings.append("Temporal/site shift was configured but could not be estimated.")
    else:
        shift_flag = False
        metrics["temporal_or_site_shift_auroc_gap"] = metric(
            None,
            higher_is_worse=True,
            status="Not evaluated",
            note="No shift.time_column or shift.site_column was configured.",
        )

    if conclusion_reversal:
        weaknesses.append(
            "The utility conclusion reversed under the maximum missingness perturbation."
        )
    for name, value in metrics.items():
        if value["flag"] and name != "tstr_auroc_drop_at_maximum_missingness":
            weaknesses.append(f"{name} crossed its provisional stability threshold.")

    required_flags = [
        model_seed_flag,
        bootstrap_flag,
        missingness_flag,
        training_size_flag,
        generator_seed_flag,
        shift_flag,
    ]
    status = "Fail" if conclusion_reversal else (
        "Conditional" if any(required_flags) else "Pass"
    )
    return {
        "dimension": "robustness",
        "status": status,
        "metrics": metrics,
        "failure_flags": [
            f"robustness:{name}" for name, value in metrics.items() if value["flag"]
        ],
        "warnings": warnings,
        "weaknesses": sorted(set(weaknesses)),
        "curves": curves,
        "details": {
            "generator_seed_test_available": len(generator_seed_frames) >= 2,
            "temporal_or_site_shift_available": (
                metrics["temporal_or_site_shift_auroc_gap"]["status"] == "Available"
            ),
            "conclusion_reversal_under_missingness": conclusion_reversal,
        },
    }
