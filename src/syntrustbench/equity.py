"""Subgroup representation, utility, and empirical privacy exposure."""

from __future__ import annotations

import numpy as np

from .models import Submission
from .stats import categorical_values, metric, percentile_interval, safe_auroc


def _subgroup_interval(
    labels: np.ndarray,
    probability: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
    confidence_level: float,
) -> tuple[float, float | None, float | None]:
    point = safe_auroc(labels, probability)
    low, high = percentile_interval(
        point,
        lambda rows: safe_auroc(labels[rows], probability[rows]),
        len(labels),
        rng,
        iterations,
        confidence_level,
    )
    return point, low, high


def evaluate_equity(
    submission: Submission,
    utility_result: dict,
    privacy_result: dict,
    rng: np.random.Generator,
) -> dict:
    config = submission.config
    metrics: dict[str, dict] = {}
    subgroup_rows: list[dict] = []
    warnings: list[str] = []
    weaknesses: list[str] = []
    thresholds = config.thresholds

    if not config.protected_attributes:
        metrics["maximum_representation_gap"] = metric(
            None,
            higher_is_worse=True,
            status="Not evaluated",
            note="No protected or clinically relevant subgroup attributes were configured.",
        )
        metrics["worst_group_tstr_auroc"] = metric(
            None,
            higher_is_worse=False,
            status="Not evaluated",
            note="No subgroup attributes were configured.",
        )
        metrics["maximum_within_attribute_utility_gap"] = metric(
            None,
            higher_is_worse=True,
            status="Not evaluated",
            note="No subgroup attributes were configured.",
        )
        warnings.append("Equity was not evaluated because protected_attributes is empty.")
        return {
            "dimension": "equity",
            "status": "Conditional",
            "metrics": metrics,
            "failure_flags": [],
            "warnings": warnings,
            "weaknesses": [],
            "subgroup_rows": [],
        }

    internal_utility = utility_result["_internal"]
    y_true = internal_utility["y_true"]
    trtr_probability = internal_utility["trtr_probability"]
    tstr_probability = internal_utility["tstr_probability"]
    internal_privacy = privacy_result["_internal"]
    duplicate_mask = internal_privacy["duplicate_mask"]
    synthetic_dcr = internal_privacy["synthetic_dcr"]
    real_real_floor = internal_privacy["real_real_floor"]

    test = submission.real_test.reset_index(drop=True)
    real = submission.real_train.reset_index(drop=True)
    synthetic = submission.synthetic.reset_index(drop=True)
    evaluable: dict[str, dict[str, dict]] = {}
    insufficient_count = 0

    for attribute in config.protected_attributes:
        real_levels = set(categorical_values(real[attribute]))
        synthetic_levels = set(categorical_values(synthetic[attribute]))
        test_levels = set(categorical_values(test[attribute]))
        levels = sorted(real_levels | synthetic_levels | test_levels)
        evaluable[attribute] = {}
        for level in levels:
            real_mask = categorical_values(real[attribute]).to_numpy() == level
            synthetic_mask = (
                categorical_values(synthetic[attribute]).to_numpy() == level
            )
            test_mask = categorical_values(test[attribute]).to_numpy() == level
            n_test = int(test_mask.sum())
            n_synthetic = int(synthetic_mask.sum())
            positive = int(y_true[test_mask].sum()) if n_test else 0
            negative = n_test - positive
            real_share = float(real_mask.mean())
            synthetic_share = float(synthetic_mask.mean())
            representation_gap = abs(synthetic_share - real_share)
            subgroup_dcr = synthetic_dcr[synthetic_mask]
            subgroup_dcr = subgroup_dcr[np.isfinite(subgroup_dcr)]
            row = {
                "attribute": attribute,
                "subgroup": level,
                "n_real_train": int(real_mask.sum()),
                "n_real_test": n_test,
                "n_synthetic": n_synthetic,
                "n_test_positive": positive,
                "real_train_prevalence": real_share,
                "synthetic_prevalence": synthetic_share,
                "representation_gap": representation_gap,
                "subgroup_exact_duplicate_rate": (
                    float(duplicate_mask[synthetic_mask].mean())
                    if n_synthetic
                    else None
                ),
                "subgroup_share_below_dcr_floor": (
                    float(np.mean(subgroup_dcr < real_real_floor))
                    if len(subgroup_dcr)
                    else None
                ),
            }
            if (
                n_test < config.min_subgroup_n
                or positive < config.min_subgroup_outcomes
                or negative < config.min_subgroup_outcomes
            ):
                insufficient_count += 1
                row.update(
                    {
                        "trtr_auroc": None,
                        "trtr_auroc_ci_low": None,
                        "trtr_auroc_ci_high": None,
                        "tstr_auroc": None,
                        "tstr_auroc_ci_low": None,
                        "tstr_auroc_ci_high": None,
                        "utility_retention": None,
                        "utility_retention_ci_low": None,
                        "utility_retention_ci_high": None,
                        "status": "Insufficient evidence",
                    }
                )
            else:
                labels = y_true[test_mask]
                trtr = trtr_probability[test_mask]
                tstr = tstr_probability[test_mask]
                trtr_point, trtr_low, trtr_high = _subgroup_interval(
                    labels,
                    trtr,
                    rng,
                    config.bootstrap_iterations,
                    config.confidence_level,
                )
                tstr_point, tstr_low, tstr_high = _subgroup_interval(
                    labels,
                    tstr,
                    rng,
                    config.bootstrap_iterations,
                    config.confidence_level,
                )
                retention = (
                    tstr_point / trtr_point
                    if np.isfinite(trtr_point) and trtr_point > 0
                    else float("nan")
                )

                def retention_statistic(rows: np.ndarray) -> float:
                    real_score = safe_auroc(labels[rows], trtr[rows])
                    synthetic_score = safe_auroc(labels[rows], tstr[rows])
                    return (
                        synthetic_score / real_score
                        if np.isfinite(real_score) and real_score > 0
                        else float("nan")
                    )

                retention_low, retention_high = percentile_interval(
                    retention,
                    retention_statistic,
                    len(labels),
                    rng,
                    config.bootstrap_iterations,
                    config.confidence_level,
                )
                row.update(
                    {
                        "trtr_auroc": trtr_point,
                        "trtr_auroc_ci_low": trtr_low,
                        "trtr_auroc_ci_high": trtr_high,
                        "tstr_auroc": tstr_point,
                        "tstr_auroc_ci_low": tstr_low,
                        "tstr_auroc_ci_high": tstr_high,
                        "utility_retention": retention,
                        "utility_retention_ci_low": retention_low,
                        "utility_retention_ci_high": retention_high,
                        "status": "Evaluable",
                    }
                )
                evaluable[attribute][level] = row
            subgroup_rows.append(row)

    representation_gaps = [row["representation_gap"] for row in subgroup_rows]
    maximum_representation_gap = max(representation_gaps, default=float("nan"))
    representation_flag = (
        np.isfinite(maximum_representation_gap)
        and maximum_representation_gap
        > thresholds["equity_representation_gap_conditional"]
    )
    metrics["maximum_representation_gap"] = metric(
        maximum_representation_gap,
        higher_is_worse=True,
        flag=representation_flag,
        threshold=(
            f"> {thresholds['equity_representation_gap_conditional']:.2f} -> Conditional"
        ),
        note="Largest absolute real-training versus synthetic subgroup prevalence gap.",
    )

    all_evaluable = [
        row for by_attribute in evaluable.values() for row in by_attribute.values()
    ]
    if all_evaluable:
        worst = min(all_evaluable, key=lambda row: row["tstr_auroc"])
        worst_flag = (
            worst["tstr_auroc"] < thresholds["equity_worst_group_fail"]
        )
        worst_uncertain = (
            not worst_flag
            and worst["tstr_auroc_ci_low"] is not None
            and worst["tstr_auroc_ci_low"] < thresholds["equity_worst_group_fail"]
        )
        metrics["worst_group_tstr_auroc"] = metric(
            worst["tstr_auroc"],
            ci_low=worst["tstr_auroc_ci_low"],
            ci_high=worst["tstr_auroc_ci_high"],
            interval_type="within-subgroup held-out real-test bootstrap",
            higher_is_worse=False,
            flag=worst_flag,
            threshold=f"< {thresholds['equity_worst_group_fail']:.2f} -> Fail",
            note=(
                "Lowest evaluable subgroup TSTR AUROC: "
                f"{worst['attribute']}={worst['subgroup']}."
            ),
        )
    else:
        worst_flag = False
        worst_uncertain = True
        metrics["worst_group_tstr_auroc"] = metric(
            None,
            higher_is_worse=False,
            status="Not evaluated",
            note="No subgroup met the minimum sample and outcome-count requirements.",
        )

    within_attribute_gaps: dict[str, float] = {}
    for attribute, rows in evaluable.items():
        values = [row["tstr_auroc"] for row in rows.values()]
        if len(values) >= 2:
            within_attribute_gaps[attribute] = float(max(values) - min(values))
    maximum_utility_gap = max(within_attribute_gaps.values(), default=float("nan"))
    utility_gap_flag = (
        np.isfinite(maximum_utility_gap)
        and maximum_utility_gap > thresholds["equity_max_gap_conditional"]
    )
    metrics["maximum_within_attribute_utility_gap"] = metric(
        maximum_utility_gap if np.isfinite(maximum_utility_gap) else None,
        higher_is_worse=True,
        flag=utility_gap_flag,
        threshold=f"> {thresholds['equity_max_gap_conditional']:.2f} -> Conditional",
        note=(
            "Largest best-minus-worst TSTR AUROC gap calculated within the same "
            f"protected attribute; per-attribute gaps={within_attribute_gaps}."
        ),
    )

    if representation_flag:
        weaknesses.append("At least one subgroup is materially under- or over-represented.")
    if worst_flag:
        weaknesses.append("Worst-group synthetic-trained utility is below the provisional floor.")
    elif worst_uncertain:
        warnings.append(
            "The worst-group utility confidence interval crosses the provisional floor."
        )
    if utility_gap_flag:
        weaknesses.append("A protected attribute has a large within-attribute utility gap.")
    if insufficient_count:
        warnings.append(
            f"{insufficient_count} subgroup rows were marked Insufficient evidence rather "
            "than assigned unstable performance scores."
        )

    if worst_flag:
        status = "Fail"
    elif representation_flag or utility_gap_flag or worst_uncertain or not all_evaluable:
        status = "Conditional"
    else:
        status = "Pass"
    return {
        "dimension": "equity",
        "status": status,
        "metrics": metrics,
        "failure_flags": [
            f"equity:{name}" for name, value in metrics.items() if value["flag"]
        ],
        "warnings": warnings,
        "weaknesses": weaknesses,
        "subgroup_rows": subgroup_rows,
        "details": {"within_attribute_utility_gaps": within_attribute_gaps},
    }
