"""Controlled degradation operators used by tests and the tutorial notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_training_duplicates(
    synthetic: pd.DataFrame,
    real_train: pd.DataFrame,
    fraction: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Replace a fraction of synthetic rows with copied real training rows."""

    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one.")
    count = max(1, int(round(len(synthetic) * fraction)))
    if set(synthetic.columns) != set(real_train.columns):
        raise ValueError("real_train and synthetic columns must match.")
    rng = np.random.default_rng(seed)
    output = synthetic.copy().reset_index(drop=True)
    target_rows = rng.choice(len(output), size=count, replace=False)
    source_rows = rng.choice(len(real_train), size=count, replace=count > len(real_train))
    output.loc[target_rows, synthetic.columns] = (
        real_train.iloc[source_rows][synthetic.columns].to_numpy()
    )
    return output


def reduce_subgroup(
    synthetic: pd.DataFrame,
    attribute: str,
    subgroup: str,
    removal_fraction: float = 0.90,
    seed: int = 42,
    preserve_row_count: bool = True,
) -> pd.DataFrame:
    """Remove a subgroup and optionally resample remaining rows to preserve table size."""

    if not 0 < removal_fraction <= 1:
        raise ValueError("removal_fraction must be in (0, 1].")
    mask = synthetic[attribute].astype(str) == str(subgroup)
    subgroup_rows = synthetic[mask]
    if subgroup_rows.empty:
        raise ValueError(f"No rows matched {attribute}={subgroup!r}.")
    removed = subgroup_rows.sample(frac=removal_fraction, random_state=seed).index
    reduced = synthetic.drop(removed).reset_index(drop=True)
    if preserve_row_count and len(reduced) < len(synthetic):
        additional = reduced.sample(
            n=len(synthetic) - len(reduced),
            replace=True,
            random_state=seed + 1,
        )
        reduced = pd.concat([reduced, additional], ignore_index=True)
        reduced = reduced.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return reduced


def shuffle_feature(
    synthetic: pd.DataFrame, column: str, seed: int = 42
) -> pd.DataFrame:
    """Destroy dependencies involving one feature while preserving its marginal values."""

    output = synthetic.copy()
    output[column] = np.random.default_rng(seed).permutation(
        output[column].to_numpy()
    )
    return output


def add_missingness(
    synthetic: pd.DataFrame,
    columns: list[str],
    rate: float = 0.30,
    seed: int = 42,
) -> pd.DataFrame:
    """Inject missing completely at random values into selected features."""

    if not 0 < rate < 1:
        raise ValueError("rate must be between zero and one.")
    output = synthetic.copy()
    rng = np.random.default_rng(seed)
    for column in columns:
        output.loc[rng.random(len(output)) < rate, column] = np.nan
    return output
