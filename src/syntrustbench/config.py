"""Configuration loading and validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .models import BenchmarkConfig


DEFAULT_THRESHOLDS: dict[str, float] = {
    "fidelity_wasserstein_conditional": 0.20,
    "fidelity_jsd_conditional": 0.10,
    "fidelity_association_conditional": 0.15,
    "fidelity_category_coverage_fail": 0.95,
    "fidelity_constraint_rate_fail": 0.01,
    "fidelity_missingness_gap_conditional": 0.10,
    "utility_retention_fail": 0.80,
    "utility_retention_conditional": 0.90,
    "privacy_duplicate_rate_fail": 0.0,
    "privacy_dcr_share_conditional": 0.05,
    "privacy_attack_auc_fail": 0.55,
    "privacy_attack_tpr_fail": 0.05,
    "equity_representation_gap_conditional": 0.10,
    "equity_max_gap_conditional": 0.10,
    "equity_worst_group_fail": 0.60,
    "robustness_model_seed_cv_conditional": 0.05,
    "robustness_generator_seed_cv_conditional": 0.05,
    "robustness_bootstrap_width_conditional": 0.20,
    "robustness_missingness_drop_conditional": 0.10,
    "robustness_training_size_spread_conditional": 0.10,
    "robustness_shift_gap_conditional": 0.10,
}


def _load_raw(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return copy.deepcopy(config)
    path = Path(config)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return raw


def _as_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of column names.")
    return list(value)


def load_config(config: str | Path | dict[str, Any]) -> BenchmarkConfig:
    """Load the versioned v0.2 tabular configuration."""

    raw = _load_raw(config)
    dataset = raw.get("dataset") or {}
    columns = raw.get("columns") or {}
    utility = raw.get("utility_task") or {}
    generator = raw.get("generator") or {}
    shift = raw.get("shift") or {}
    uncertainty = raw.get("uncertainty") or {}
    robustness = raw.get("robustness") or {}
    subgroup = raw.get("subgroup_analysis") or {}
    privacy = raw.get("privacy") or {}

    continuous = _as_string_list(columns.get("continuous"), "columns.continuous")
    categorical = _as_string_list(columns.get("categorical"), "columns.categorical")
    binary = _as_string_list(columns.get("binary"), "columns.binary")
    target = str(utility.get("target") or dataset.get("outcome") or "")

    if not target:
        raise ValueError("utility_task.target is required.")
    if dataset.get("modality", "tabular") != "tabular":
        raise ValueError("SynTrustBench v0.2 implements structured tabular data only.")
    if utility.get("type", "binary_classification") != "binary_classification":
        raise ValueError("The initial executable protocol supports binary classification only.")

    declared = continuous + categorical + binary
    duplicated = sorted({column for column in declared if declared.count(column) > 1})
    if duplicated:
        raise ValueError(f"Columns must appear in exactly one type list: {duplicated}")
    if target not in declared:
        raise ValueError("utility_task.target must be declared in columns.binary.")
    if target not in binary:
        raise ValueError("The v0.2 binary task target must be declared in columns.binary.")

    protected = _as_string_list(raw.get("protected_attributes"), "protected_attributes")
    undeclared_protected = sorted(set(protected) - set(declared))
    if undeclared_protected:
        raise ValueError(f"Protected attributes are not declared columns: {undeclared_protected}")

    excluded = _as_string_list(utility.get("exclude_features"), "utility_task.exclude_features")
    requested_features = utility.get("features")
    if requested_features is None:
        features = [column for column in declared if column != target and column not in excluded]
    else:
        features = _as_string_list(requested_features, "utility_task.features")
    invalid_features = sorted(set(features) - set(declared))
    if invalid_features:
        raise ValueError(f"Utility features are not declared columns: {invalid_features}")
    if target in features:
        raise ValueError("The target cannot be included as a utility feature.")
    if not features:
        raise ValueError("At least one utility feature is required.")

    thresholds = dict(DEFAULT_THRESHOLDS)
    custom_thresholds = raw.get("thresholds") or {}
    unknown_thresholds = sorted(set(custom_thresholds) - set(thresholds))
    if unknown_thresholds:
        raise ValueError(f"Unknown threshold keys: {unknown_thresholds}")
    thresholds.update({key: float(value) for key, value in custom_thresholds.items()})

    bootstrap_iterations = int(uncertainty.get("bootstrap_iterations", 200))
    confidence_level = float(uncertainty.get("confidence_level", 0.95))
    random_seed = int(uncertainty.get("random_seed", generator.get("seed", 42) or 42))
    if bootstrap_iterations < 20:
        raise ValueError("uncertainty.bootstrap_iterations must be at least 20.")
    if not 0.50 < confidence_level < 1.0:
        raise ValueError("uncertainty.confidence_level must be between 0.50 and 1.0.")

    model_seeds = [int(seed) for seed in robustness.get(
        "downstream_model_seeds", [random_seed + offset for offset in range(5)]
    )]
    missingness_rates = [float(rate) for rate in robustness.get(
        "missingness_rates", [0.05, 0.10, 0.20]
    )]
    training_fractions = [float(value) for value in robustness.get(
        "training_size_fractions", [0.25, 0.50, 0.75, 1.0]
    )]
    if any(not 0 < rate < 1 for rate in missingness_rates):
        raise ValueError("robustness.missingness_rates values must be between 0 and 1.")
    if any(not 0 < fraction <= 1 for fraction in training_fractions):
        raise ValueError("robustness.training_size_fractions values must be in (0, 1].")

    privacy_distance_max_rows = int(privacy.get("distance_max_rows", 10_000))
    privacy_attack_max_rows = int(privacy.get("attack_max_rows", 10_000))
    if privacy_distance_max_rows < 500:
        raise ValueError("privacy.distance_max_rows must be at least 500.")
    if privacy_attack_max_rows < 500:
        raise ValueError("privacy.attack_max_rows must be at least 500.")

    return BenchmarkConfig(
        name=str(dataset.get("name", "unnamed_dataset")),
        modality="tabular",
        continuous=continuous,
        categorical=categorical,
        binary=binary,
        target=target,
        task_type="binary_classification",
        positive_label=str(utility.get("positive_label", "1")),
        protected_attributes=protected,
        clinical_constraints=_as_string_list(
            raw.get("clinical_constraints"), "clinical_constraints"
        ),
        generator_name=str(generator.get("name", "unspecified")),
        generator_seed=(
            int(generator["seed"]) if generator.get("seed") is not None else None
        ),
        feature_columns=features,
        excluded_features=excluded,
        time_column=shift.get("time_column"),
        site_column=shift.get("site_column"),
        bootstrap_iterations=bootstrap_iterations,
        confidence_level=confidence_level,
        random_seed=random_seed,
        model_seeds=model_seeds,
        missingness_rates=missingness_rates,
        training_size_fractions=training_fractions,
        min_subgroup_n=int(subgroup.get("minimum_n", 30)),
        min_subgroup_outcomes=int(subgroup.get("minimum_outcomes_per_class", 10)),
        thresholds=thresholds,
        exact_duplicate_decimals=int(privacy.get("exact_duplicate_decimals", 10)),
        privacy_distance_max_rows=privacy_distance_max_rows,
        privacy_attack_max_rows=privacy_attack_max_rows,
    )
