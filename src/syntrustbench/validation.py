"""Input loading, hashing, coercion, and leakage checks."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .models import BenchmarkConfig, Submission


def _dataframe_hash(frame: pd.DataFrame) -> str:
    canonical = frame.to_csv(index=False, lineterminator="\n", na_rep="__NA__")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_table(source: str | Path | pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
        return frame, _dataframe_hash(frame)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Input table does not exist: {path}")
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path), content_hash


def _coerce_frame(
    frame: pd.DataFrame, config: BenchmarkConfig, label: str
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    missing_columns = sorted(set(config.all_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{label} is missing declared columns: {missing_columns}")
    if frame.empty:
        raise ValueError(f"{label} contains no rows.")

    frame = frame[config.all_columns].copy()
    for column in config.continuous:
        original_nonmissing = int(frame[column].notna().sum())
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        converted_nonmissing = int(frame[column].notna().sum())
        if converted_nonmissing < original_nonmissing:
            warnings.append(
                f"{label}.{column}: "
                f"{original_nonmissing - converted_nonmissing} values became missing "
                "during numeric coercion."
            )
    for column in config.categorical + config.binary:
        frame[column] = frame[column].astype("object")
        present = frame[column].notna()
        frame.loc[present, column] = frame.loc[present, column].astype(str)
    return frame, warnings


def _map_target(
    frame: pd.DataFrame,
    config: BenchmarkConfig,
    label: str,
    *,
    require_two_classes: bool,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    target = config.target
    if frame[target].isna().any():
        raise ValueError(f"{label}.{target} contains missing outcomes.")
    values = frame[target].astype(str)
    unique = set(values.unique())
    positive = config.positive_label
    if positive not in unique and require_two_classes:
        raise ValueError(
            f"{label}.{target} does not contain positive_label={positive!r}; "
            f"observed values are {sorted(unique)}."
        )
    if positive not in unique and not require_two_classes:
        warnings.append(
            f"{label}.{target} does not contain positive_label={positive!r}; "
            "all generated outcomes will map to the negative class."
        )
    if require_two_classes and len(unique) != 2:
        raise ValueError(
            f"{label}.{target} must contain exactly two classes; observed {sorted(unique)}."
        )
    if not require_two_classes and len(unique) != 2:
        warnings.append(
            f"{label}.{target} contains {len(unique)} observed class(es); "
            "this generator failure will be reflected in utility results."
        )
    frame[target] = (values == positive).astype(int)
    return frame, warnings


def _canonical_keys(
    frame: pd.DataFrame, config: BenchmarkConfig, include_target: bool = True
) -> pd.Series:
    columns = list(config.all_columns)
    if not include_target:
        columns = [column for column in columns if column != config.target]
    canonical = frame[columns].copy()
    for column in config.continuous:
        if column in canonical:
            canonical[column] = pd.to_numeric(
                canonical[column], errors="coerce"
            ).round(config.exact_duplicate_decimals)
    return pd.util.hash_pandas_object(canonical, index=False)


def _validate_splits(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    config: BenchmarkConfig,
) -> list[str]:
    warnings: list[str] = []
    train_keys = set(_canonical_keys(real_train, config).tolist())
    test_keys = _canonical_keys(real_test, config)
    overlap = int(test_keys.isin(train_keys).sum())
    if overlap > 0:
        raise ValueError(
            f"Real train/test overlap detected: {overlap} overlapping held-out rows "
            "exactly match training rows. Evaluation stopped."
        )
    if config.entity_id is None:
        warnings.append(
            "Only exact row overlap was checked; patient-level independence could not be verified."
        )
    for label, frame in (("real_train", real_train), ("real_test", real_test)):
        prevalence = float(frame[config.target].mean())
        if prevalence < 0.01 or prevalence > 0.99:
            warnings.append(
                f"{label} outcome prevalence is {prevalence:.2%}; discrimination and "
                "subgroup estimates may be unstable."
            )
    return warnings


def _entity_keys(values: pd.Series) -> set:
    """Compare stripped text and exact numeric equivalents without float rounding."""
    keys = set()
    for value in values:
        text = str(value).strip()
        try:
            numeric = Decimal(text)
        except InvalidOperation:
            numeric = None
        keys.add(("numeric", numeric) if numeric is not None and numeric.is_finite()
                 else ("text", text))
    return keys


def load_submission(
    real_train: str | Path | pd.DataFrame,
    real_test: str | Path | pd.DataFrame,
    synthetic: str | Path | pd.DataFrame,
    config: str | Path | dict[str, Any],
    synthetic_replicates: list[str | Path | pd.DataFrame] | None = None,
) -> Submission:
    """Load and validate the four-file submission contract."""

    benchmark_config = load_config(config)
    if isinstance(config, dict):
        config_bytes = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    else:
        config_bytes = Path(config).read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    raw_train, train_hash = _read_table(real_train)
    raw_test, test_hash = _read_table(real_test)
    raw_synthetic, synthetic_hash = _read_table(synthetic)

    if benchmark_config.entity_id is not None:
        entity_id = benchmark_config.entity_id
        for label, frame in (("real_train", raw_train), ("real_test", raw_test)):
            if entity_id not in frame.columns:
                raise ValueError(f"{label} is missing entity identifier column {entity_id!r}.")
            missing = int(frame[entity_id].isna().sum())
            if missing:
                raise ValueError(
                    f"{label}.{entity_id}: {missing} missing entity identifiers. Evaluation stopped."
                )
        overlap = _entity_keys(raw_train[entity_id]) & _entity_keys(raw_test[entity_id])
        if overlap:
            raise ValueError(
                f"Real train/test entity overlap detected: {len(overlap)} overlapping "
                f"identifiers in {entity_id!r}. Evaluation stopped."
            )

    train, warnings_train = _coerce_frame(raw_train, benchmark_config, "real_train")
    test, warnings_test = _coerce_frame(raw_test, benchmark_config, "real_test")
    generated, warnings_synthetic = _coerce_frame(
        raw_synthetic, benchmark_config, "synthetic"
    )
    train, target_warnings_train = _map_target(
        train, benchmark_config, "real_train", require_two_classes=True
    )
    test, target_warnings_test = _map_target(
        test, benchmark_config, "real_test", require_two_classes=True
    )
    generated, target_warnings_synthetic = _map_target(
        generated, benchmark_config, "synthetic", require_two_classes=False
    )

    warnings = (
        warnings_train
        + warnings_test
        + warnings_synthetic
        + target_warnings_train
        + target_warnings_test
        + target_warnings_synthetic
    )
    warnings.extend(_validate_splits(train, test, benchmark_config))

    replicate_frames: list[pd.DataFrame] = []
    replicate_hashes: list[str] = []
    for index, source in enumerate(synthetic_replicates or [], start=1):
        raw, source_hash = _read_table(source)
        frame, replicate_warnings = _coerce_frame(
            raw, benchmark_config, f"synthetic_replicate_{index}"
        )
        frame, target_warnings = _map_target(
            frame,
            benchmark_config,
            f"synthetic_replicate_{index}",
            require_two_classes=False,
        )
        replicate_frames.append(frame)
        replicate_hashes.append(source_hash)
        warnings.extend(replicate_warnings)
        warnings.extend(target_warnings)

    return Submission(
        real_train=train,
        real_test=test,
        synthetic=generated,
        config=benchmark_config,
        input_hashes={
            "real_train_sha256": train_hash,
            "real_test_sha256": test_hash,
            "synthetic_sha256": synthetic_hash,
            "config_sha256": config_hash,
        },
        warnings=warnings,
        synthetic_replicates=replicate_frames,
        replicate_hashes=replicate_hashes,
    )


def canonical_row_keys(frame: pd.DataFrame, config: BenchmarkConfig) -> np.ndarray:
    """Stable row hashes used for exact-copy checks."""

    return _canonical_keys(frame, config).to_numpy()
