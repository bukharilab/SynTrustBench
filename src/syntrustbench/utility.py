"""Downstream utility evaluation using TRTR and TSTR."""

from __future__ import annotations

import numpy as np

from .models import Submission
from .stats import (
    FeatureEncoder,
    metric,
    percentile_interval,
    safe_auprc,
    safe_auroc,
    safe_brier,
    train_classifier,
)


def _paired_interval(
    point: float,
    y_true: np.ndarray,
    first: np.ndarray,
    second: np.ndarray | None,
    statistic,
    rng: np.random.Generator,
    iterations: int,
    confidence_level: float,
) -> tuple[float | None, float | None]:
    if second is None:
        return percentile_interval(
            point,
            lambda rows: statistic(y_true[rows], first[rows]),
            len(y_true),
            rng,
            iterations,
            confidence_level,
        )
    return percentile_interval(
        point,
        lambda rows: statistic(y_true[rows], first[rows], second[rows]),
        len(y_true),
        rng,
        iterations,
        confidence_level,
    )


def evaluate_utility(submission: Submission, rng: np.random.Generator) -> dict:
    config = submission.config
    encoder = FeatureEncoder(
        config.continuous_features, config.categorical_features
    ).fit(submission.real_train)
    y_true, trtr_probability = train_classifier(
        submission.real_train,
        submission.real_test,
        config.target,
        encoder,
        config.random_seed,
    )
    tstr_y, tstr_probability = train_classifier(
        submission.synthetic,
        submission.real_test,
        config.target,
        encoder,
        config.random_seed,
    )
    if not np.array_equal(y_true, tstr_y):
        raise RuntimeError("TRTR and TSTR test labels are not aligned.")

    iterations = config.bootstrap_iterations
    confidence = config.confidence_level
    metrics: dict[str, dict] = {}

    functions = {
        "auroc": safe_auroc,
        "auprc": safe_auprc,
        "brier": safe_brier,
    }
    trtr_values: dict[str, float] = {}
    tstr_values: dict[str, float] = {}
    for name, function in functions.items():
        trtr_point = function(y_true, trtr_probability)
        tstr_point = function(y_true, tstr_probability)
        trtr_values[name] = trtr_point
        tstr_values[name] = tstr_point
        trtr_low, trtr_high = _paired_interval(
            trtr_point,
            y_true,
            trtr_probability,
            None,
            function,
            rng,
            iterations,
            confidence,
        )
        tstr_low, tstr_high = _paired_interval(
            tstr_point,
            y_true,
            tstr_probability,
            None,
            function,
            rng,
            iterations,
            confidence,
        )
        higher_is_worse = name == "brier"
        metrics[f"trtr_{name}"] = metric(
            trtr_point,
            ci_low=trtr_low,
            ci_high=trtr_high,
            interval_type="held-out real-test bootstrap",
            higher_is_worse=higher_is_worse,
            note=f"Train real, test held-out real: {name.upper()}.",
        )
        metrics[f"tstr_{name}"] = metric(
            tstr_point,
            ci_low=tstr_low,
            ci_high=tstr_high,
            interval_type="held-out real-test bootstrap",
            higher_is_worse=higher_is_worse,
            note=f"Train synthetic, test held-out real: {name.upper()}.",
        )

    trtr_auroc = trtr_values["auroc"]
    tstr_auroc = tstr_values["auroc"]
    retention = (
        tstr_auroc / trtr_auroc
        if np.isfinite(trtr_auroc) and trtr_auroc > 0
        else float("nan")
    )

    def retention_statistic(
        labels: np.ndarray, trtr: np.ndarray, tstr: np.ndarray
    ) -> float:
        real_score = safe_auroc(labels, trtr)
        synthetic_score = safe_auroc(labels, tstr)
        return (
            synthetic_score / real_score
            if np.isfinite(real_score) and real_score > 0
            else float("nan")
        )

    retention_low, retention_high = _paired_interval(
        retention,
        y_true,
        trtr_probability,
        tstr_probability,
        retention_statistic,
        rng,
        iterations,
        confidence,
    )
    thresholds = config.thresholds
    retention_flag = (
        not np.isfinite(retention)
        or retention < thresholds["utility_retention_conditional"]
    )
    metrics["auroc_utility_retention"] = metric(
        retention,
        ci_low=retention_low,
        ci_high=retention_high,
        interval_type="paired held-out real-test bootstrap",
        higher_is_worse=False,
        flag=retention_flag,
        threshold=(
            f"< {thresholds['utility_retention_fail']:.2f} -> Fail; "
            f"< {thresholds['utility_retention_conditional']:.2f} -> Conditional"
        ),
        note="TSTR AUROC divided by TRTR AUROC on the same held-out real rows.",
    )

    trtr_auprc = trtr_values["auprc"]
    tstr_auprc = tstr_values["auprc"]
    auprc_retention = (
        tstr_auprc / trtr_auprc
        if np.isfinite(trtr_auprc) and trtr_auprc > 0
        else float("nan")
    )
    metrics["auprc_utility_retention"] = metric(
        auprc_retention,
        higher_is_worse=False,
        note="Descriptive TSTR AUPRC divided by TRTR AUPRC.",
    )

    tstr_below_chance = (
        metrics["tstr_auroc"]["ci_high"] is not None
        and metrics["tstr_auroc"]["ci_high"] <= 0.5
    )
    metrics["tstr_auroc"]["flag"] = tstr_below_chance
    metrics["tstr_auroc"]["threshold"] = "upper confidence bound <= 0.50 -> Fail"

    if not np.isfinite(retention) or tstr_below_chance:
        status = "Fail"
    elif (
        retention < thresholds["utility_retention_fail"]
        or (
            retention_high is not None
            and retention_high < thresholds["utility_retention_fail"]
        )
    ):
        status = "Fail"
    elif (
        retention < thresholds["utility_retention_conditional"]
        or (
            retention_low is not None
            and retention_low < thresholds["utility_retention_conditional"]
        )
    ):
        status = "Conditional"
    else:
        status = "Pass"

    weaknesses: list[str] = []
    if status != "Pass":
        weaknesses.append(
            "Synthetic-trained discrimination did not clear the provisional "
            "utility-retention requirement."
        )
    return {
        "dimension": "utility",
        "status": status,
        "metrics": metrics,
        "failure_flags": [
            f"utility:{name}" for name, value in metrics.items() if value["flag"]
        ],
        "warnings": [],
        "weaknesses": weaknesses,
        "_internal": {
            "encoder": encoder,
            "y_true": y_true,
            "trtr_probability": trtr_probability,
            "tstr_probability": tstr_probability,
            "retention": retention,
        },
    }
