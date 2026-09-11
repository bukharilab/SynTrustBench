"""Machine-readable and human-readable benchmark outputs."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml

from ._version import __version__
from .models import Submission
from .stats import aggregate_status


DIMENSIONS = ["fidelity", "utility", "privacy", "equity", "robustness"]


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _clean(item)
            for key, item in value.items()
            if not key.startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def benchmark_gate(dimensions: dict[str, dict]) -> str:
    statuses = [dimensions[name]["status"] for name in DIMENSIONS]
    return aggregate_status(statuses, overall=True)


def _unavailable_required(dimensions: dict[str, dict]) -> list[str]:
    return [
        f"{dimension}.{name}"
        for dimension in DIMENSIONS
        for name, value in dimensions[dimension]["metrics"].items()
        if value.get("status") in {"NotEvaluated", "Not evaluated"}
        and value.get("required", True)
    ]


def _environment() -> dict[str, Any]:
    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pyyaml": yaml.__version__,
        "syntrustbench": __version__,
    }
    try:
        repository = Path(__file__).resolve().parents[2]
        environment["source_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        environment["source_worktree_dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        environment["source_commit"] = None
        environment["source_worktree_dirty"] = None
    return environment


def _metric_rows(dimensions: dict[str, dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        for name, value in dimensions[dimension]["metrics"].items():
            rows.append(
                {
                    "dimension": dimension,
                    "metric": name,
                    "estimate": value.get("estimate"),
                    "ci_low": value.get("ci_low"),
                    "ci_high": value.get("ci_high"),
                    "ci_width": value.get("ci_width"),
                    "interval_type": value.get("interval_type"),
                    "higher_is_worse": value.get("higher_is_worse"),
                    "availability": value.get("status"),
                    "failure_flag": value.get("flag"),
                    "provisional_threshold": value.get("threshold"),
                    "note": value.get("note"),
                }
            )
    return rows


def _coverage(dimensions: dict[str, dict]) -> dict[str, Any]:
    by_dimension: dict[str, dict[str, int]] = {}
    available_total = 0
    metric_total = 0
    for dimension in DIMENSIONS:
        values = list(dimensions[dimension]["metrics"].values())
        available = sum(
            value.get("status") in {"Available", "Pass", "Conditional", "Fail"}
            for value in values
        )
        by_dimension[dimension] = {"available": available, "total": len(values)}
        available_total += available
        metric_total += len(values)
    return {
        "available_metrics": available_total,
        "total_metrics": metric_total,
        "fraction_available": (
            available_total / metric_total if metric_total else None
        ),
        "by_dimension": by_dimension,
    }


def _benchmark_card(
    submission: Submission, dimensions: dict[str, dict], gate: str
) -> dict[str, Any]:
    config = submission.config
    unavailable = [
        f"{dimension}.{name}"
        for dimension in DIMENSIONS
        for name, value in dimensions[dimension]["metrics"].items()
        if value.get("status") in {"NotEvaluated", "Not evaluated"}
    ]
    return {
        "benchmark": {
            "name": "SynTrustBench",
            "component": "Executable Evaluation",
            "protocol": "structured-tabular-clinical-data",
            "version": __version__,
            "status": "open versioned benchmark framework; not a formal standard",
        },
        "submission": {
            "dataset": config.name,
            "generator": config.generator_name,
            "generator_seed": config.generator_seed,
            "rows": {
                "real_train": len(submission.real_train),
                "real_test": len(submission.real_test),
                "synthetic": len(submission.synthetic),
            },
            "target": config.target,
            "positive_label": config.positive_label,
            "protected_attributes": config.protected_attributes,
        },
        "results": {
            "benchmark_gate": gate,
            "dimension_status": {
                dimension: dimensions[dimension]["status"]
                for dimension in DIMENSIONS
            },
            "unavailable_tests": unavailable,
            "unavailable_required_evidence": _unavailable_required(dimensions),
        },
        "privacy_threat_model": dimensions["privacy"]["threat_model"],
        "reproducibility": {
            "input_hashes": submission.input_hashes,
            "synthetic_replicate_hashes": submission.replicate_hashes,
            "random_seed": config.random_seed,
            "bootstrap_iterations": config.bootstrap_iterations,
            "confidence_level": config.confidence_level,
            "downstream_model_seeds": config.model_seeds,
        },
        "interpretation": {
            "non_compensable": True,
            "single_weighted_score": False,
            "thresholds": "provisional and configurable; not clinically calibrated",
            "certification": False,
        },
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "not evaluated"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_report(
    submission: Submission,
    dimensions: dict[str, dict],
    gate: str,
    timestamp: str,
    warnings: list[str],
) -> str:
    config = submission.config
    lines = [
        f"# SynTrustBench report: {config.name}",
        "",
        f"- Protocol: structured tabular clinical data v{__version__}",
        f"- Generator: {config.generator_name}",
        f"- Generated: {timestamp}",
        f"- Minimum-requirement benchmark gate: **{gate}**",
        "",
        (
            "This gate summarizes non-compensable, provisional benchmark requirements. "
            "It is not an evidence-evaluability judgment, safety certification, or "
            "regulatory determination."
        ),
        "",
        "## Dimension profile",
        "",
        "| Dimension | Status | Headline result |",
        "|---|---|---|",
    ]
    headline = {
        "fidelity": ("association_matrix_error", "dependency error"),
        "utility": ("auroc_utility_retention", "chance-corrected AUROC utility retention"),
        "privacy": ("exact_training_duplicate_rate", "training-copy rate"),
        "equity": ("worst_group_tstr_auroc", "worst-group TSTR AUROC"),
        "robustness": (
            "tstr_auroc_drop_at_maximum_missingness",
            "missingness AUROC drop",
        ),
    }
    for dimension in DIMENSIONS:
        metric_name, label = headline[dimension]
        value = dimensions[dimension]["metrics"].get(metric_name, {})
        estimate = _format_value(value.get("estimate"))
        lines.append(
            f"| {dimensions[dimension].get('label', dimension.capitalize())} | **{dimensions[dimension]['status']}** "
            f"| {label}: {estimate} |"
        )

    for dimension in DIMENSIONS:
        result = dimensions[dimension]
        lines.extend(
            [
                "",
                f"## {result.get('label', dimension.capitalize())}: {result['status']}",
                "",
                "| Metric | Estimate | Interval or range | Availability | Flag |",
                "|---|---:|---|---|---|",
            ]
        )
        for name, value in result["metrics"].items():
            if value.get("ci_low") is not None:
                interval = (
                    f"[{_format_value(value['ci_low'])}, "
                    f"{_format_value(value['ci_high'])}]"
                )
                if value.get("interval_type"):
                    interval += f" ({value['interval_type']})"
            else:
                interval = "-"
            lines.append(
                f"| {name} | {_format_value(value.get('estimate'))} | {interval} | "
                f"{value.get('availability_label', value.get('status', 'Available'))} | "
                f"{'FLAG' if value.get('flag') else ''} |"
            )
        lines.append("")
        for name, value in result["metrics"].items():
            if dimension in {"utility", "privacy"} and value.get("note"):
                lines.append(f"- {name}: {value['note']}")
            if dimension in {"utility", "privacy"} and value.get("threshold"):
                lines.append(f"- {name} decision: {value['threshold']}")
        if dimension == "fidelity":
            for expression, detail in result.get("details", {}).get("clinical_constraints", {}).items():
                lines.append(
                    f"- Constraint `{expression}`: eligible={detail['total_eligible_rows']}; "
                    f"evaluated={detail['evaluated_rows']}; "
                    f"excluded missing={detail['excluded_missing_rows']}; "
                    f"violating={detail['violating_rows']}; "
                    f"violation rate={_format_value(detail['violation_rate'])}; {detail['status']}."
                )
        if dimension == "robustness":
            for name, status in result.get("details", {}).get("optional_checks", {}).items():
                lines.append(f"- {name}: {status}")
        if result.get("weaknesses"):
            lines.extend(
                ["", "**Observed weaknesses**", ""]
                + [f"- {item}" for item in result["weaknesses"]]
            )
        if result.get("warnings"):
            lines.extend(
                ["", "**Dimension warnings**", ""]
                + [f"- {item}" for item in result["warnings"]]
            )
        if dimension == "privacy":
            threat = result["threat_model"]
            lines.extend(
                [
                    "",
                    "**Tested privacy threat model**",
                    "",
                    f"- Attacker knowledge: {threat['attacker_knowledge']}",
                    f"- Attacker access: {threat['attacker_access']}",
                    f"- Goal: {threat['attacker_goal']}",
                    f"- Interpretation: {threat['interpretation']}",
                ]
            )

    lines.extend(["", "## Unavailable required evidence", ""])
    lines.extend([f"- {name}" for name in _unavailable_required(dimensions)] or ["- None."])
    lines.extend(["", "## Run warnings", ""])
    lines.extend([f"- {warning}" for warning in warnings] or ["- None."])
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- Evidence maturity is separate from measured dataset behavior.",
            "- Strong performance in one dimension does not offset failure in another.",
            "- A passing empirical privacy profile is not a formal privacy guarantee.",
            "- Thresholds are provisional research defaults and can be overridden in config.",
            "- The executable release evaluates structured tabular data only.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    submission: Submission,
    dimensions: dict[str, dict],
    output_dir: str | Path,
) -> tuple[dict[str, Any], str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    gate = benchmark_gate(dimensions)
    timestamp = datetime.now(timezone.utc).isoformat()
    all_warnings = sorted(
        set(
            submission.warnings
            + [
                warning
                for dimension in DIMENSIONS
                for warning in dimensions[dimension].get("warnings", [])
            ]
        )
    )
    failure_flags = sorted(
        {
            flag
            for dimension in DIMENSIONS
            for flag in dimensions[dimension].get("failure_flags", [])
        }
    )
    public_dimensions = {
        dimension: _clean(dimensions[dimension]) for dimension in DIMENSIONS
    }
    runtime_environment = _environment()
    summary = {
        "schema_version": "1.0",
        "benchmark": "SynTrustBench",
        "component": "Executable Evaluation",
        "protocol": "structured-tabular-clinical-data",
        "version": __version__,
        "timestamp_utc": timestamp,
        "dataset": submission.config.name,
        "generator": submission.config.generator_name,
        "benchmark_gate": gate,
        "dimension_status": {
            dimension: dimensions[dimension]["status"] for dimension in DIMENSIONS
        },
        "evaluation_coverage": _coverage(dimensions),
        "failure_flags": failure_flags,
        "warnings": all_warnings,
        "dimensions": public_dimensions,
        "interpretation": {
            "provisional_thresholds": True,
            "regulatory_certification": False,
            "single_weighted_score": False,
        },
    }
    manifest = {
        "schema_version": "1.0",
        "timestamp_utc": timestamp,
        "protocol_version": __version__,
        "input_hashes": submission.input_hashes,
        "synthetic_replicate_hashes": submission.replicate_hashes,
        "environment": runtime_environment,
        "protocol_constants": {
            "utility_probe": {
                "model": "sklearn.ensemble.RandomForestClassifier",
                "n_estimators": 200,
                "min_samples_leaf": 5,
                "n_jobs": -1,
            },
            "privacy_distance": (
                "Euclidean distance after real-trained continuous standardization "
                "and categorical one-hot encoding"
            ),
        },
        "configuration": _clean(vars(submission.config)),
    }
    card = _benchmark_card(submission, dimensions, gate)

    (output_path / "summary.json").write_text(
        json.dumps(_clean(summary), indent=2) + "\n", encoding="utf-8"
    )
    (output_path / "run_manifest.json").write_text(
        json.dumps(_clean(manifest), indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(_metric_rows(dimensions)).to_csv(
        output_path / "metrics.csv", index=False
    )
    pd.DataFrame(dimensions["equity"].get("subgroup_rows", [])).to_csv(
        output_path / "subgroup_results.csv", index=False
    )
    (output_path / "benchmark_card.yaml").write_text(
        yaml.safe_dump(_clean(card), sort_keys=False), encoding="utf-8"
    )
    (output_path / "details.json").write_text(
        json.dumps(
            _clean(
                {
                    dimension: {
                        "details": dimensions[dimension].get("details", {}),
                        "curves": dimensions[dimension].get("curves", {}),
                    }
                    for dimension in DIMENSIONS
                }
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_path / "report.md").write_text(
        _render_report(submission, dimensions, gate, timestamp, all_warnings),
        encoding="utf-8",
    )
    log_lines = [
        "SynTrustBench execution log",
        f"timestamp_utc: {timestamp}",
        f"protocol_version: {__version__}",
        f"benchmark_gate: {gate}",
        "input_hashes:",
        *[
            f"  {name}: {value}"
            for name, value in submission.input_hashes.items()
        ],
        "environment:",
        *[f"  {name}: {value}" for name, value in runtime_environment.items()],
        "warnings:",
        *([f"  - {warning}" for warning in all_warnings] or ["  - none"]),
    ]
    (output_path / "execution_log.txt").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )
    return summary, gate
