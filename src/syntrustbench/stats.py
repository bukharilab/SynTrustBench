"""Statistical helpers shared across benchmark dimensions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


MISSING_CATEGORY = "__MISSING__"


def categorical_values(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), MISSING_CATEGORY).astype(str)


def metric(
    estimate: float | int | str | None,
    *,
    ci_low: float | None = None,
    ci_high: float | None = None,
    interval_type: str | None = None,
    higher_is_worse: bool,
    flag: bool = False,
    status: str = "Available",
    threshold: str | None = None,
    note: str = "",
) -> dict:
    width = None
    if ci_low is not None and ci_high is not None:
        width = float(ci_high - ci_low)
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": width,
        "interval_type": interval_type,
        "higher_is_worse": higher_is_worse,
        "flag": bool(flag),
        "status": status,
        "threshold": threshold,
        "note": note,
    }


def percentile_interval(
    point: float,
    statistic: Callable[[np.ndarray], float],
    sample_size: int,
    rng: np.random.Generator,
    iterations: int,
    confidence_level: float,
) -> tuple[float | None, float | None]:
    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        rows = rng.integers(0, sample_size, sample_size)
        values[index] = statistic(rows)
    values = values[np.isfinite(values)]
    if values.size < max(10, iterations // 4):
        return None, None
    alpha = 1.0 - confidence_level
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def two_sample_basic_interval(
    point: float,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    real_size: int,
    synthetic_size: int,
    rng: np.random.Generator,
    iterations: int,
    confidence_level: float,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float | None, float | None]:
    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        real_rows = rng.integers(0, real_size, real_size)
        synthetic_rows = rng.integers(0, synthetic_size, synthetic_size)
        values[index] = statistic(real_rows, synthetic_rows)
    values = values[np.isfinite(values)]
    if values.size < max(10, iterations // 4):
        return None, None
    alpha = 1.0 - confidence_level
    quantile_low, quantile_high = np.quantile(
        values, [alpha / 2.0, 1.0 - alpha / 2.0]
    )
    low = 2.0 * point - float(quantile_high)
    high = 2.0 * point - float(quantile_low)
    if lower_bound is not None:
        low = max(lower_bound, low)
    if upper_bound is not None:
        high = min(upper_bound, high)
    return min(low, point), max(high, point)


def two_sample_standard_error_interval(
    point: float,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    real_size: int,
    synthetic_size: int,
    rng: np.random.Generator,
    iterations: int,
    confidence_level: float,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float | None, float | None]:
    """Bootstrap standard-error interval centered on the observed two-sample statistic.

    Distance statistics are non-negative and their naive percentile bootstrap can be
    noticeably biased upward near zero. Centering the interval on the observed estimate
    keeps the reported uncertainty interpretable while retaining two-sample resampling.
    """

    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        real_rows = rng.integers(0, real_size, real_size)
        synthetic_rows = rng.integers(0, synthetic_size, synthetic_size)
        values[index] = statistic(real_rows, synthetic_rows)
    values = values[np.isfinite(values)]
    if values.size < max(10, iterations // 4):
        return None, None
    standard_error = float(values.std(ddof=1))
    critical_value = float(norm.ppf(0.5 + confidence_level / 2.0))
    low = point - critical_value * standard_error
    high = point + critical_value * standard_error
    if lower_bound is not None:
        low = max(lower_bound, low)
    if upper_bound is not None:
        high = min(upper_bound, high)
    return min(low, point), max(high, point)


def safe_auroc(y_true: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probability))


def safe_auprc(y_true: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, probability))


def safe_brier(y_true: np.ndarray, probability: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(brier_score_loss(y_true, probability))


class FeatureEncoder:
    """Deterministic encoding fitted on the real training data."""

    def __init__(self, continuous: list[str], categorical: list[str]):
        self.continuous = continuous
        self.categorical = categorical

    def fit(self, frame: pd.DataFrame) -> "FeatureEncoder":
        self.medians = {
            column: float(pd.to_numeric(frame[column], errors="coerce").median())
            for column in self.continuous
        }
        self.means = {
            column: float(
                pd.to_numeric(frame[column], errors="coerce").fillna(self.medians[column]).mean()
            )
            for column in self.continuous
        }
        self.stds = {
            column: float(
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(self.medians[column])
                .std()
            )
            for column in self.continuous
        }
        self.categories = {
            column: sorted(categorical_values(frame[column]).unique().tolist())
            for column in self.categorical
        }
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        for column in self.continuous:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(
                self.medians[column]
            )
            denominator = self.stds[column] if self.stds[column] > 1e-12 else 1.0
            parts.append(((values - self.means[column]) / denominator).to_numpy()[:, None])
        for column in self.categorical:
            values = categorical_values(frame[column]).to_numpy()
            for level in self.categories[column]:
                parts.append((values == level).astype(float)[:, None])
        return np.hstack(parts) if parts else np.zeros((len(frame), 0))


def train_classifier(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    encoder: FeatureEncoder,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_train = train[target].astype(int).to_numpy()
    y_test = test[target].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        return y_test, np.full(len(y_test), float(y_train.mean()))
    classifier = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )
    classifier.fit(encoder.transform(train), y_train)
    return y_test, classifier.predict_proba(encoder.transform(test))[:, 1]


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(categorical_values(x), categorical_values(y))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi_squared = chi2_contingency(table, correction=False)[0]
    sample_size = table.to_numpy().sum()
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt((chi_squared / sample_size) / denominator)) if denominator else 0.0


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    category = categorical_values(categories).to_numpy()
    numeric = pd.to_numeric(values, errors="coerce").to_numpy()
    valid = np.isfinite(numeric)
    category, numeric = category[valid], numeric[valid]
    if numeric.size < 3:
        return 0.0
    overall_mean = numeric.mean()
    numerator = sum(
        len(group) * (group.mean() - overall_mean) ** 2
        for level in np.unique(category)
        for group in [numeric[category == level]]
    )
    denominator = ((numeric - overall_mean) ** 2).sum()
    return float(np.sqrt(numerator / denominator)) if denominator else 0.0


def association_matrix(
    frame: pd.DataFrame, continuous: list[str], categorical: list[str]
) -> pd.DataFrame:
    columns = continuous + categorical
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    for left_index, left in enumerate(columns):
        for right_index in range(left_index + 1, len(columns)):
            right = columns[right_index]
            if left in continuous and right in continuous:
                pair = frame[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
                value = abs(pair[left].corr(pair[right])) if len(pair) >= 3 else 0.0
            elif left in categorical and right in categorical:
                value = cramers_v(frame[left], frame[right])
            else:
                category, numeric = (
                    (left, right) if left in categorical else (right, left)
                )
                value = correlation_ratio(frame[category], frame[numeric])
            value = float(value) if np.isfinite(value) else 0.0
            matrix.iloc[left_index, right_index] = value
            matrix.iloc[right_index, left_index] = value
    return matrix
