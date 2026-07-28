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
    frame: pd.DataFrame, constraints: list[str]
) -> tuple[float | None, dict[str, float], list[str]]:
    """Return overall violation rate, per-rule rates, and invalid-rule warnings."""

    if not constraints:
        return None, {}, []
    violated_any = np.zeros(len(frame), dtype=bool)
    rates: dict[str, float] = {}
    warnings: list[str] = []
    valid_count = 0
    for expression in constraints:
        try:
            satisfied = evaluate_constraint(frame, expression)
        except ValueError as exc:
            warnings.append(str(exc))
            continue
        valid_count += 1
        violated = ~satisfied
        rates[expression] = float(violated.mean())
        violated_any |= violated
    if valid_count == 0:
        return None, rates, warnings
    return float(violated_any.mean()), rates, warnings
