"""Shared data structures for the executable protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class BenchmarkConfig:
    name: str
    modality: str
    continuous: list[str]
    categorical: list[str]
    binary: list[str]
    target: str
    task_type: str
    positive_label: str
    protected_attributes: list[str]
    clinical_constraints: list[str]
    generator_name: str
    generator_seed: int | None
    feature_columns: list[str]
    excluded_features: list[str]
    time_column: str | None
    site_column: str | None
    bootstrap_iterations: int
    confidence_level: float
    random_seed: int
    model_seeds: list[int]
    missingness_rates: list[float]
    training_size_fractions: list[float]
    min_subgroup_n: int
    min_subgroup_outcomes: int
    thresholds: dict[str, float]
    exact_duplicate_decimals: int
    privacy_distance_max_rows: int
    privacy_attack_max_rows: int

    @property
    def all_columns(self) -> list[str]:
        columns: list[str] = []
        for column in self.continuous + self.categorical + self.binary:
            if column not in columns:
                columns.append(column)
        for column in (self.time_column, self.site_column):
            if column and column not in columns:
                columns.append(column)
        return columns

    @property
    def categorical_features(self) -> list[str]:
        categorical = set(self.categorical + self.binary)
        return [column for column in self.feature_columns if column in categorical]

    @property
    def continuous_features(self) -> list[str]:
        return [column for column in self.feature_columns if column in self.continuous]


@dataclass
class Submission:
    real_train: pd.DataFrame
    real_test: pd.DataFrame
    synthetic: pd.DataFrame
    config: BenchmarkConfig
    input_hashes: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    synthetic_replicates: list[pd.DataFrame] = field(default_factory=list)
    replicate_hashes: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    dimensions: dict[str, dict[str, Any]]
    benchmark_gate: str
    summary: dict[str, Any]
    output_dir: Path

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
