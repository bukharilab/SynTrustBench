"""Empirical privacy indicators under an explicit distance-based threat model."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.neighbors import NearestNeighbors

from .models import Submission
from .stats import FeatureEncoder, metric, percentile_interval, safe_auroc
from .validation import canonical_row_keys


def _bounded_rows(
    frame,
    maximum_rows: int,
    rng: np.random.Generator,
):
    """Return a reproducible row subset and its positional indices."""

    if len(frame) <= maximum_rows:
        positions = np.arange(len(frame))
    else:
        positions = np.sort(rng.choice(len(frame), size=maximum_rows, replace=False))
    return frame.iloc[positions], positions


def _tpr_at_fpr(
    labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01
) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    eligible = false_positive_rate <= target_fpr
    return float(true_positive_rate[eligible].max()) if eligible.any() else 0.0


def evaluate_privacy(submission: Submission, rng: np.random.Generator) -> dict:
    config = submission.config
    real = submission.real_train
    test = submission.real_test
    synthetic = submission.synthetic
    thresholds = config.thresholds
    iterations = config.bootstrap_iterations
    confidence = config.confidence_level
    metrics: dict[str, dict] = {}
    weaknesses: list[str] = []
    warnings: list[str] = []

    real_keys = set(canonical_row_keys(real, config).tolist())
    synthetic_keys = canonical_row_keys(synthetic, config)
    duplicate_mask = np.array([key in real_keys for key in synthetic_keys], dtype=bool)
    duplicate_rate = float(duplicate_mask.mean())
    duplicate_low, duplicate_high = percentile_interval(
        duplicate_rate,
        lambda rows: float(duplicate_mask[rows].mean()),
        len(synthetic),
        rng,
        iterations,
        confidence,
    )
    duplicate_flag = duplicate_rate > thresholds["privacy_duplicate_rate_fail"]
    metrics["exact_training_duplicate_rate"] = metric(
        duplicate_rate,
        ci_low=duplicate_low,
        ci_high=duplicate_high,
        interval_type="synthetic-row bootstrap",
        higher_is_worse=True,
        flag=duplicate_flag,
        threshold=f"> {thresholds['privacy_duplicate_rate_fail']:.3f} -> Fail",
        note=(
            "Share of synthetic rows matching a complete real-training row after "
            f"{config.exact_duplicate_decimals}-decimal canonicalization."
        ),
    )
    if duplicate_flag:
        weaknesses.append("The released synthetic table contains copied training rows.")

    privacy_continuous = config.continuous
    privacy_categorical = config.categorical + config.binary
    encoder = FeatureEncoder(privacy_continuous, privacy_categorical).fit(real)

    distance_real, _ = _bounded_rows(
        real, config.privacy_distance_max_rows, rng
    )
    distance_synthetic, distance_synthetic_positions = _bounded_rows(
        synthetic, config.privacy_distance_max_rows, rng
    )
    real_matrix = encoder.transform(distance_real)
    synthetic_matrix = encoder.transform(distance_synthetic)
    if len(real_matrix) < 2:
        raise ValueError("At least two real training rows are required for privacy distances.")
    if len(distance_real) < len(real) or len(distance_synthetic) < len(synthetic):
        warnings.append(
            "Nearest-neighbor exposure was estimated from reproducible bounded samples "
            f"(real reference n={len(distance_real):,}; synthetic query "
            f"n={len(distance_synthetic):,}). Exact-copy analysis still used all rows."
        )

    neighbors = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(real_matrix)
    synthetic_distances, _ = neighbors.kneighbors(synthetic_matrix)
    real_distances, _ = neighbors.kneighbors(real_matrix)
    distance_to_closest = synthetic_distances[:, 0]
    full_synthetic_dcr = np.full(len(synthetic), np.nan)
    full_synthetic_dcr[distance_synthetic_positions] = distance_to_closest
    real_real_floor = float(np.percentile(real_distances[:, 1], 5))
    share_below_floor = float(np.mean(distance_to_closest < real_real_floor))

    median_distance = float(np.median(distance_to_closest))
    median_low, median_high = percentile_interval(
        median_distance,
        lambda rows: float(np.median(distance_to_closest[rows])),
        len(distance_to_closest),
        rng,
        iterations,
        confidence,
    )
    metrics["median_distance_to_closest_training_record"] = metric(
        median_distance,
        ci_low=median_low,
        ci_high=median_high,
        interval_type="synthetic-row bootstrap",
        higher_is_worse=False,
        note=(
            "Median standardized mixed-feature distance to the closest real training row; "
            f"estimated with {len(distance_synthetic):,} synthetic queries against "
            f"{len(distance_real):,} real references; descriptive only, not a privacy "
            "guarantee."
        ),
    )
    floor_low, floor_high = percentile_interval(
        share_below_floor,
        lambda rows: float(np.mean(distance_to_closest[rows] < real_real_floor)),
        len(distance_to_closest),
        rng,
        iterations,
        confidence,
    )
    floor_flag = share_below_floor > thresholds["privacy_dcr_share_conditional"]
    metrics["share_closer_than_real_real_floor"] = metric(
        share_below_floor,
        ci_low=floor_low,
        ci_high=floor_high,
        interval_type="synthetic-row bootstrap",
        higher_is_worse=True,
        flag=floor_flag,
        threshold=f"> {thresholds['privacy_dcr_share_conditional']:.2f} -> Conditional",
        note=(
            "Share of synthetic rows closer to training data than the fifth percentile "
            "of non-self real-to-real distances in the bounded distance sample."
        ),
    )
    distance_ratio = distance_to_closest / np.maximum(
        synthetic_distances[:, 1], 1e-12
    )
    ratio_p05 = float(np.percentile(distance_ratio, 5))
    metrics["nearest_neighbor_distance_ratio_p05"] = metric(
        ratio_p05,
        higher_is_worse=False,
        note=(
            "Fifth percentile of closest-to-second-closest distance ratio; "
            "low values identify unusually isolated matches."
        ),
    )

    attack_real, _ = _bounded_rows(real, config.privacy_attack_max_rows, rng)
    attack_test, _ = _bounded_rows(test, config.privacy_attack_max_rows, rng)
    attack_synthetic, _ = _bounded_rows(
        synthetic, config.privacy_attack_max_rows, rng
    )
    attack_available = (
        len(attack_real) >= 20
        and len(attack_test) >= 20
        and len(attack_synthetic) >= 20
    )
    attack_scores: np.ndarray | None = None
    attack_labels: np.ndarray | None = None
    if attack_available:
        released_matrix = encoder.transform(attack_synthetic)
        member_matrix = encoder.transform(attack_real)
        nonmember_matrix = encoder.transform(attack_test)
        released_neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(
            released_matrix
        )
        member_distance, _ = released_neighbors.kneighbors(member_matrix)
        nonmember_distance, _ = released_neighbors.kneighbors(nonmember_matrix)
        attack_scores = np.concatenate(
            [-member_distance[:, 0], -nonmember_distance[:, 0]]
        )
        attack_labels = np.concatenate(
            [np.ones(len(member_distance)), np.zeros(len(nonmember_distance))]
        )
        attack_auc = safe_auroc(attack_labels, attack_scores)
        attack_low, attack_high = percentile_interval(
            attack_auc,
            lambda rows: safe_auroc(attack_labels[rows], attack_scores[rows]),
            len(attack_labels),
            rng,
            iterations,
            confidence,
        )
        attack_tpr = _tpr_at_fpr(attack_labels, attack_scores, 0.01)
        attack_flag = (
            attack_low is not None
            and attack_low > thresholds["privacy_attack_auc_fail"]
        )
        attack_tpr_flag = attack_tpr > thresholds["privacy_attack_tpr_fail"]
        metrics["distance_membership_attack_auc"] = metric(
            attack_auc,
            ci_low=attack_low,
            ci_high=attack_high,
            interval_type="candidate-record bootstrap",
            higher_is_worse=True,
            flag=attack_flag,
            threshold=(
                f"lower confidence bound > "
                f"{thresholds['privacy_attack_auc_fail']:.2f} -> Fail"
            ),
            note=(
                "Black-box distance attack: candidate closeness to released synthetic "
                "data predicts membership in real training versus held-out real data; "
                f"members n={len(attack_real):,}, nonmembers n={len(attack_test):,}, "
                f"released rows n={len(attack_synthetic):,}."
            ),
        )
        metrics["distance_membership_attack_tpr_at_1pct_fpr"] = metric(
            attack_tpr,
            higher_is_worse=True,
            flag=attack_tpr_flag,
            threshold=f"> {thresholds['privacy_attack_tpr_fail']:.2f} -> Fail",
            note="True-positive rate at a one-percent false-positive rate.",
        )
    else:
        warnings.append(
            "Distance membership attack was not run because at least one input had fewer "
            "than 20 rows."
        )
        metrics["distance_membership_attack_auc"] = metric(
            None,
            higher_is_worse=True,
            status="Not evaluated",
            flag=False,
            note="Insufficient sample size for the configured attack.",
        )
        metrics["distance_membership_attack_tpr_at_1pct_fpr"] = metric(
            None,
            higher_is_worse=True,
            status="Not evaluated",
            flag=False,
            note="Insufficient sample size for the configured attack.",
        )

    attack_failure = any(
        metrics[name]["flag"]
        for name in (
            "distance_membership_attack_auc",
            "distance_membership_attack_tpr_at_1pct_fpr",
        )
    )
    if duplicate_flag or attack_failure:
        status = "Fail"
    elif floor_flag or not attack_available:
        status = "Conditional"
    else:
        status = "Pass"

    if floor_flag:
        weaknesses.append(
            "An elevated share of synthetic rows is unusually close to training records."
        )
    if attack_failure:
        weaknesses.append(
            "The tested membership attacker extracted signal above the provisional limit."
        )
    return {
        "dimension": "privacy",
        "status": status,
        "metrics": metrics,
        "failure_flags": [
            f"privacy:{name}" for name, value in metrics.items() if value["flag"]
        ],
        "warnings": warnings,
        "weaknesses": weaknesses,
        "threat_model": {
            "attacker_knowledge": "released synthetic table and a candidate record",
            "attacker_access": "black-box access to released rows; no generator access",
            "attacker_goal": "infer whether the candidate was in real_train",
            "members": "real_train",
            "nonmembers": "held-out real_test",
            "attack": "distance to closest released synthetic row",
            "interpretation": (
                "Empirical risk under this attacker only; not a formal privacy guarantee."
            ),
        },
        "details": {
            "real_real_distance_floor_p05": real_real_floor,
            "attack_available": attack_available,
            "distance_reference_rows": len(distance_real),
            "distance_query_rows": len(distance_synthetic),
            "attack_member_rows": len(attack_real),
            "attack_nonmember_rows": len(attack_test),
            "attack_released_rows": len(attack_synthetic),
        },
        "_internal": {
            "duplicate_mask": duplicate_mask,
            "synthetic_dcr": full_synthetic_dcr,
            "real_real_floor": real_real_floor,
            "synthetic_keys": synthetic_keys,
            "attack_scores": attack_scores,
            "attack_labels": attack_labels,
        },
    }
