"""Fidelity evaluation for structured tabular data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from .constraints import constraint_violations
from .models import Submission
from .stats import (
    aggregate_status,
    association_matrix,
    categorical_values,
    metric,
    percentile_interval,
    two_sample_standard_error_interval,
)


def _jensen_shannon(real: pd.Series, synthetic: pd.Series) -> tuple[float, float]:
    real_values = categorical_values(real)
    synthetic_values = categorical_values(synthetic)
    levels = sorted(set(real_values) | set(synthetic_values))
    real_probability = (
        real_values.value_counts().reindex(levels, fill_value=0).to_numpy(dtype=float)
    )
    synthetic_probability = (
        synthetic_values.value_counts().reindex(levels, fill_value=0).to_numpy(dtype=float)
    )
    real_probability /= real_probability.sum()
    synthetic_probability /= synthetic_probability.sum()
    middle = 0.5 * (real_probability + synthetic_probability)

    def divergence(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    jsd = 0.5 * divergence(real_probability, middle) + 0.5 * divergence(
        synthetic_probability, middle
    )
    total_variation = 0.5 * float(
        np.abs(real_probability - synthetic_probability).sum()
    )
    return float(jsd), total_variation


def _normalized_wasserstein(real: pd.Series, synthetic: pd.Series) -> float:
    real_numeric = pd.to_numeric(real, errors="coerce").dropna().to_numpy()
    synthetic_numeric = pd.to_numeric(synthetic, errors="coerce").dropna().to_numpy()
    if real_numeric.size == 0 or synthetic_numeric.size == 0:
        return float("nan")
    scale = float(np.subtract(*np.percentile(real_numeric, [75, 25])))
    if scale <= 1e-12:
        scale = float(np.std(real_numeric))
    if scale <= 1e-12:
        scale = 1.0
    return float(wasserstein_distance(real_numeric, synthetic_numeric) / scale)


def _mean_wasserstein(
    real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]
) -> float:
    values = [_normalized_wasserstein(real[column], synthetic[column]) for column in columns]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def _mean_jsd(
    real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]
) -> float:
    values = [_jensen_shannon(real[column], synthetic[column])[0] for column in columns]
    return float(np.mean(values)) if values else float("nan")


def _association_error(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    continuous: list[str],
    categorical: list[str],
) -> float:
    real_matrix = association_matrix(real, continuous, categorical).to_numpy()
    synthetic_matrix = association_matrix(synthetic, continuous, categorical).to_numpy()
    upper = np.triu_indices_from(real_matrix, k=1)
    if len(upper[0]) == 0:
        return 0.0
    return float(np.mean(np.abs(real_matrix[upper] - synthetic_matrix[upper])))


def evaluate_fidelity(submission: Submission, rng: np.random.Generator) -> dict:
    config = submission.config
    real = submission.real_train
    synthetic = submission.synthetic
    continuous = config.continuous
    categorical = config.categorical + config.binary
    thresholds = config.thresholds
    iterations = config.bootstrap_iterations
    confidence = config.confidence_level
    metrics: dict[str, dict] = {}
    weaknesses: list[str] = []
    warnings: list[str] = []

    per_column_wasserstein = {
        column: _normalized_wasserstein(real[column], synthetic[column])
        for column in continuous
    }
    per_column_ks = {}
    for column in continuous:
        real_values = pd.to_numeric(real[column], errors="coerce").dropna()
        synthetic_values = pd.to_numeric(synthetic[column], errors="coerce").dropna()
        per_column_ks[column] = (
            float(ks_2samp(real_values, synthetic_values).statistic)
            if len(real_values) and len(synthetic_values)
            else float("nan")
        )
    mean_wasserstein = _mean_wasserstein(real, synthetic, continuous)
    if continuous:
        low, high = two_sample_standard_error_interval(
            mean_wasserstein,
            lambda real_rows, synthetic_rows: _mean_wasserstein(
                real.iloc[real_rows], synthetic.iloc[synthetic_rows], continuous
            ),
            len(real),
            len(synthetic),
            rng,
            iterations,
            confidence,
            lower_bound=0.0,
        )
    else:
        low, high = None, None
    wasserstein_flag = (
        np.isfinite(mean_wasserstein)
        and mean_wasserstein > thresholds["fidelity_wasserstein_conditional"]
    )
    metrics["mean_normalized_wasserstein"] = metric(
        mean_wasserstein,
        ci_low=low,
        ci_high=high,
        interval_type="two-sample bootstrap standard-error interval",
        higher_is_worse=True,
        flag=wasserstein_flag,
        threshold=f"> {thresholds['fidelity_wasserstein_conditional']:.2f} -> Conditional",
        note="Mean column-wise Wasserstein distance divided by the real-data IQR.",
    )
    mean_ks = float(np.nanmean(list(per_column_ks.values()))) if continuous else None
    metrics["mean_ks_statistic"] = metric(
        mean_ks,
        higher_is_worse=True,
        note="Descriptive mean of continuous-column two-sample KS statistics.",
    )

    per_column_jsd: dict[str, float] = {}
    per_column_tv: dict[str, float] = {}
    for column in categorical:
        jsd, total_variation = _jensen_shannon(real[column], synthetic[column])
        per_column_jsd[column] = jsd
        per_column_tv[column] = total_variation
    mean_jsd = _mean_jsd(real, synthetic, categorical)
    if categorical:
        low, high = two_sample_standard_error_interval(
            mean_jsd,
            lambda real_rows, synthetic_rows: _mean_jsd(
                real.iloc[real_rows], synthetic.iloc[synthetic_rows], categorical
            ),
            len(real),
            len(synthetic),
            rng,
            iterations,
            confidence,
            lower_bound=0.0,
            upper_bound=1.0,
        )
    else:
        low, high = None, None
    jsd_flag = (
        np.isfinite(mean_jsd)
        and mean_jsd > thresholds["fidelity_jsd_conditional"]
    )
    metrics["mean_jensen_shannon_divergence"] = metric(
        mean_jsd,
        ci_low=low,
        ci_high=high,
        interval_type="two-sample bootstrap standard-error interval",
        higher_is_worse=True,
        flag=jsd_flag,
        threshold=f"> {thresholds['fidelity_jsd_conditional']:.2f} -> Conditional",
        note="Mean categorical-column Jensen-Shannon divergence, base 2.",
    )
    metrics["mean_total_variation_distance"] = metric(
        float(np.mean(list(per_column_tv.values()))) if categorical else None,
        higher_is_worse=True,
        note="Descriptive mean categorical total-variation distance.",
    )

    association_error = _association_error(real, synthetic, continuous, categorical)
    low, high = two_sample_standard_error_interval(
        association_error,
        lambda real_rows, synthetic_rows: _association_error(
            real.iloc[real_rows],
            synthetic.iloc[synthetic_rows],
            continuous,
            categorical,
        ),
        len(real),
        len(synthetic),
        rng,
        iterations,
        confidence,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    association_flag = association_error > thresholds["fidelity_association_conditional"]
    metrics["association_matrix_error"] = metric(
        association_error,
        ci_low=low,
        ci_high=high,
        interval_type="two-sample bootstrap standard-error interval",
        higher_is_worse=True,
        flag=association_flag,
        threshold=f"> {thresholds['fidelity_association_conditional']:.2f} -> Conditional",
        note=(
            "Mean absolute error between mixed association matrices "
            "(Pearson, correlation ratio, and Cramer's V)."
        ),
    )

    category_coverage: dict[str, float] = {}
    for column in categorical:
        real_levels = set(categorical_values(real[column]))
        synthetic_levels = set(categorical_values(synthetic[column]))
        coverage = len(real_levels & synthetic_levels) / max(1, len(real_levels))
        category_coverage[column] = float(coverage)
        missing = sorted(real_levels - synthetic_levels)
        if missing:
            weaknesses.append(f"{column} omits real-data categories: {missing}")
    minimum_category_coverage = min(category_coverage.values(), default=1.0)
    metrics["minimum_category_coverage"] = metric(
        minimum_category_coverage,
        higher_is_worse=False,
        flag=minimum_category_coverage
        < thresholds["fidelity_category_coverage_fail"],
        threshold=f"< {thresholds['fidelity_category_coverage_fail']:.2f} -> Fail",
        note="Minimum share of real-data categories represented, across categorical columns.",
    )

    range_coverage: dict[str, float] = {}
    out_of_range_rates: dict[str, float] = {}
    for column in continuous:
        real_values = pd.to_numeric(real[column], errors="coerce").dropna()
        synthetic_values = pd.to_numeric(synthetic[column], errors="coerce").dropna()
        if not len(real_values) or not len(synthetic_values):
            continue
        real_min, real_max = float(real_values.min()), float(real_values.max())
        synthetic_min, synthetic_max = (
            float(synthetic_values.min()),
            float(synthetic_values.max()),
        )
        real_span = real_max - real_min
        overlap = max(0.0, min(real_max, synthetic_max) - max(real_min, synthetic_min))
        range_coverage[column] = 1.0 if real_span <= 1e-12 else min(1.0, overlap / real_span)
        out_of_range_rates[column] = float(
            ((synthetic_values < real_min) | (synthetic_values > real_max)).mean()
        )
    metrics["mean_range_coverage"] = metric(
        float(np.mean(list(range_coverage.values()))) if range_coverage else None,
        higher_is_worse=False,
        note="Mean overlap of synthetic and real observed ranges relative to the real range.",
    )
    metrics["mean_out_of_range_rate"] = metric(
        float(np.mean(list(out_of_range_rates.values())))
        if out_of_range_rates
        else None,
        higher_is_worse=True,
        note="Mean share of synthetic numeric values outside the real observed range.",
    )

    real_missingness = real.isna().mean()
    synthetic_missingness = synthetic.isna().mean()
    per_column_missingness_gap = {
        column: float(abs(synthetic_missingness[column] - real_missingness[column]))
        for column in config.all_columns
    }
    maximum_missingness_gap = max(per_column_missingness_gap.values(), default=0.0)
    missingness_flag = (
        maximum_missingness_gap
        > thresholds["fidelity_missingness_gap_conditional"]
    )
    metrics["maximum_missingness_rate_gap"] = metric(
        maximum_missingness_gap,
        higher_is_worse=True,
        flag=missingness_flag,
        threshold=(
            f"> {thresholds['fidelity_missingness_gap_conditional']:.2f} -> Conditional"
        ),
        note="Largest absolute real-versus-synthetic missingness-rate difference.",
    )

    constraint_details: dict = {}
    violation_rate, per_rule, constraint_warnings = constraint_violations(
        synthetic, config.clinical_constraints, details=constraint_details
    )
    warnings.extend(constraint_warnings)
    if violation_rate is None:
        metrics["clinical_constraint_violation_rate"] = metric(
            None,
            higher_is_worse=True,
            status="NotEvaluated",
            flag=False,
            note="No clinical constraint rows could be evaluated.",
        )
    else:
        constraint_flag = (
            violation_rate > thresholds["fidelity_constraint_rate_fail"]
        )
        constraint_low, constraint_high = percentile_interval(
            violation_rate,
            lambda rows: constraint_violations(
                synthetic.iloc[rows], config.clinical_constraints
            )[0],
            len(synthetic),
            rng,
            iterations,
            confidence,
        )
        metrics["clinical_constraint_violation_rate"] = metric(
            violation_rate,
            ci_low=constraint_low,
            ci_high=constraint_high,
            interval_type="synthetic-row bootstrap",
            higher_is_worse=True,
            flag=constraint_flag,
            threshold=f"> {thresholds['fidelity_constraint_rate_fail']:.2f} -> Fail",
            note=("Maximum evaluable per-constraint violation rate; each rule uses its own "
                  f"evaluated rows, excluding missing required values: {per_rule}"),
        )

    hard_fail = (
        metrics["minimum_category_coverage"]["flag"]
        or (
            metrics["clinical_constraint_violation_rate"]["status"] == "Available"
            and metrics["clinical_constraint_violation_rate"]["flag"]
        )
    )
    conditional = any(
        metrics[name]["flag"]
        for name in (
            "mean_normalized_wasserstein",
            "mean_jensen_shannon_divergence",
            "association_matrix_error",
            "maximum_missingness_rate_gap",
        )
    ) or metrics["clinical_constraint_violation_rate"]["status"] != "Available"
    constraint_unavailable = violation_rate is None or any(
        detail["status"] == "NotEvaluated" for detail in constraint_details.values()
    )
    metrics["clinical_constraint_violation_rate"]["required"] = True
    if constraint_unavailable and not metrics["clinical_constraint_violation_rate"]["flag"]:
        metrics["clinical_constraint_violation_rate"]["status"] = "NotEvaluated"
    status = aggregate_status([
        "Fail" if hard_fail else "Pass",
        "NotEvaluated" if constraint_unavailable else "Pass",
        "Conditional" if conditional else "Pass",
    ])

    for name, value in metrics.items():
        if value["flag"] and value["status"] == "Available":
            weaknesses.append(f"{name} crossed its provisional threshold")
    return {
        "dimension": "fidelity",
        "status": status,
        "metrics": metrics,
        "failure_flags": [
            f"fidelity:{name}"
            for name, value in metrics.items()
            if value["flag"] and value["status"] == "Available"
        ],
        "warnings": warnings,
        "weaknesses": sorted(set(weaknesses)),
        "details": {
            "per_column_normalized_wasserstein": per_column_wasserstein,
            "per_column_ks_statistic": per_column_ks,
            "per_column_jensen_shannon_divergence": per_column_jsd,
            "per_column_total_variation_distance": per_column_tv,
            "per_column_category_coverage": category_coverage,
            "per_column_range_coverage": range_coverage,
            "per_column_out_of_range_rate": out_of_range_rates,
            "per_column_missingness_rate_gap": per_column_missingness_gap,
            "per_constraint_violation_rate": per_rule,
            "clinical_constraints": constraint_details,
        },
    }
