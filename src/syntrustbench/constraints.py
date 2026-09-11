"""Safe evaluation of row-level clinical constraints."""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd


_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def _validate_expression(expression: str, columns: set[str]) -> ast.Expression:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid constraint syntax: {expression!r}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"Unsupported operation {type(node).__name__} in constraint {expression!r}."
            )
        if isinstance(node, ast.Name) and node.id not in columns:
            raise ValueError(f"Unknown column {node.id!r} in constraint {expression!r}.")
    return tree


def evaluate_constraint(frame: pd.DataFrame, expression: str) -> np.ndarray:
    """Evaluate a restricted arithmetic/comparison expression on a DataFrame."""

    tree = _validate_expression(expression, set(frame.columns))
    environment = {column: frame[column] for column in frame.columns}
    compiled = compile(tree, "<clinical-constraint>", "eval")
    result = eval(compiled, {"__builtins__": {}}, environment)  # noqa: S307
    if isinstance(result, (bool, np.bool_)):
        return np.full(len(frame), bool(result), dtype=bool)
    series = pd.Series(result, index=frame.index)
    return series.fillna(False).astype(bool).to_numpy()


def constraint_violations(
    frame: pd.DataFrame, constraints: list[str], *, details: dict | None = None
) -> tuple[float | None, dict[str, float | None], list[str]]:
    """Return the maximum evaluable per-rule violation rate, rates, and warnings."""

    if not constraints:
        return None, {}, []
    rates: dict[str, float | None] = {}
    warnings: list[str] = []
    for expression in constraints:
        try:
            tree = _validate_expression(expression, set(frame.columns))
            required = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            evaluated = frame[list(required)].notna().all(axis=1).to_numpy()
            satisfied = evaluate_constraint(frame.iloc[np.flatnonzero(evaluated)], expression)
        except ValueError as exc:
            warnings.append(str(exc))
            rates[expression] = None
            if details is not None:
                details[expression] = {
                    "total_eligible_rows": len(frame), "evaluated_rows": 0,
                    "excluded_missing_rows": None, "violating_rows": 0,
                    "violation_rate": None, "status": "NotEvaluated", "note": str(exc),
                }
            continue
        violated = np.zeros(len(frame), dtype=bool)
        violated[evaluated] = ~satisfied
        count = int(evaluated.sum())
        rates[expression] = float(violated.sum() / count) if count else None
        if details is not None:
            details[expression] = {
                "total_eligible_rows": len(frame), "evaluated_rows": count,
                "excluded_missing_rows": len(frame) - count,
                "violating_rows": int(violated.sum()), "violation_rate": rates[expression],
                "status": "Available" if count else "NotEvaluated",
            }
    finite_rates = [rate for rate in rates.values() if rate is not None and np.isfinite(rate)]
    return max(finite_rates, default=None), rates, warnings
