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
            0.95 if name == "auroc" else confidence,
        )
        tstr_low, tstr_high = _paired_interval(
            tstr_point,
            y_true,
            tstr_probability,
            None,
            function,
            rng,
            iterations,
            0.95 if name == "auroc" else confidence,
        )
        higher_is_worse = name == "brier"
        metrics[f"trtr_{name}"] = metric(
            trtr_point,
            ci_low=trtr_low,
            ci_high=trtr_high,
            interval_type="held-out real-test bootstrap" + (" (95%)" if name == "auroc" else ""),
            higher_is_worse=higher_is_worse,
            note=f"Train real, test held-out real: {name.upper()}.",
        )
        metrics[f"tstr_{name}"] = metric(
            tstr_point,
            ci_low=tstr_low,
            ci_high=tstr_high,
            interval_type="held-out real-test bootstrap" + (" (95%)" if name == "auroc" else ""),
            higher_is_worse=higher_is_worse,
            note=f"Train synthetic, test held-out real: {name.upper()}.",
        )

    trtr_auroc = trtr_values["auroc"]
    tstr_auroc = tstr_values["auroc"]
    trtr_low = metrics["trtr_auroc"]["ci_low"]
    retention_evaluable = (
        trtr_low is not None and np.isfinite(trtr_low) and trtr_low > 0.5
        and np.isfinite(trtr_auroc) and trtr_auroc > 0.5
    )
    retention = (
        (tstr_auroc - 0.5) / (trtr_auroc - 0.5)
        if retention_evaluable else float("nan")
    )

    def retention_statistic(
        labels: np.ndarray, trtr: np.ndarray, tstr: np.ndarray
    ) -> float:
        real_score = safe_auroc(labels, trtr)
        synthetic_score = safe_auroc(labels, tstr)
        return (
            (synthetic_score - 0.5) / (real_score - 0.5)
            if np.isfinite(real_score) and real_score > 0.5
            else float("nan")
        )

    retention_low, retention_high = (None, None)
    if retention_evaluable:
        retention_low, retention_high = _paired_interval(
            retention, y_true, trtr_probability, tstr_probability,
            retention_statistic, rng, iterations, 0.95,
        )
    thresholds = config.thresholds
    metrics["auroc_utility_retention"] = metric(
        retention if retention_evaluable else None,
        ci_low=retention_low,
        ci_high=retention_high,
        interval_type="paired held-out real-test bootstrap (95%)",
        higher_is_worse=False,
        threshold=(
            f"Provisional: upper 95% bound < {thresholds['utility_retention_fail']:.2f} -> Fail; "
            f"lower 95% bound >= {thresholds['utility_retention_conditional']:.2f} -> Pass"
        ),
        note=(
            "Chance-corrected retention: (TSTR AUROC - 0.5) / (TRTR AUROC - 0.5). "
            "NotEvaluated unless the TRTR AUROC lower 95% bound exceeds 0.50."
        ),
    )
    difference = tstr_auroc - trtr_auroc
    difference_low, difference_high = _paired_interval(
        difference, y_true, trtr_probability, tstr_probability,
        lambda labels, trtr, tstr: safe_auroc(labels, tstr) - safe_auroc(labels, trtr),
        rng, iterations, 0.95,
    )
    metrics["tstr_minus_trtr_auroc"] = metric(
        difference, ci_low=difference_low, ci_high=difference_high,
        interval_type="paired held-out real-test bootstrap (95%)",
        higher_is_worse=False,
        note="Paired AUROC difference: TSTR minus TRTR on the same held-out real rows.",
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

    if not retention_evaluable:
        status = "NotEvaluated"
    elif tstr_below_chance or (
        retention_high is not None
        and retention_high < thresholds["utility_retention_fail"]
    ):
        status = "Fail"
    elif (
        retention_low is None or retention_high is None
        or not np.isfinite(retention_low) or not np.isfinite(retention_high)
        or metrics["tstr_auroc"]["ci_high"] is None
    ):
        status = "NotEvaluated"
    elif retention_low >= thresholds["utility_retention_conditional"]:
        status = "Pass"
    else:
        status = "Conditional"
    metrics["auroc_utility_retention"]["status"] = status
    metrics["auroc_utility_retention"]["required"] = True
    metrics["auroc_utility_retention"]["flag"] = status in {"Fail", "Conditional"}

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
            "chance_corrected_retention": retention,
            "retention_evaluable": retention_evaluable,
        },
    }
