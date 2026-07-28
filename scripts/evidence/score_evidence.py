#!/usr/bin/env python3
"""Validate the SynTrustBench corpus and generate every reported audit number.

This script scores evidence maturity, not model performance. It deliberately
does not aggregate heterogeneous published metric values into a leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from build_annotations import COLUMNS as EXPECTED_FIELDS
from build_annotations import ENUM_COLUMNS


DIMENSIONS = {
    "fidelity": ("fidelity_evaluated", "fidelity_maturity"),
    "utility": ("utility_evaluated", "utility_maturity"),
    "privacy": ("privacy_evaluated", "privacy_maturity"),
    "equity": ("fairness_evaluated", "equity_maturity"),
    "robustness": ("robustness_evaluated", "robustness_maturity"),
}

BINARY_FIELDS = [
    "fidelity_evaluated",
    "clinical_validity_evaluated",
    "utility_evaluated",
    "privacy_evaluated",
    "membership_inference",
    "attribute_inference",
    "reidentification_or_reconstruction",
    "dp_claimed",
    "formal_dp_verified",
    "epsilon_reported",
    "delta_reported",
    "fairness_evaluated",
    "robustness_evaluated",
    "external_validation",
    "uncertainty_reported",
    "ambiguity_flag",
    "mimic_iii_used",
]

REQUIRED_FIELDS = [
    "report_id",
    "citation_key",
    "title",
    "year",
    "publication_type",
    "peer_reviewed",
    "primary_source_url",
    "modality",
    "task_scope",
    "architecture_family",
    "dataset_names",
    *BINARY_FIELDS,
    "fidelity_maturity",
    "utility_maturity",
    "privacy_maturity",
    "equity_maturity",
    "robustness_maturity",
    "evaluability_gate",
    "verification_status",
    "verifier",
    "verified_date",
    "eligibility_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    return [row for row in rows if row.get("report_id")], fields


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else math.nan


def percent_text(value: float) -> str:
    if math.isnan(value):
        return "NA"
    rounded = round(value, 1)
    return f"{int(rounded)}%" if rounded.is_integer() else f"{rounded:.1f}%"


def slug(value: str) -> str:
    return (
        value.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("+", "plus")
    )


def evaluability_omissions(row: dict[str, str]) -> list[str]:
    """Return objective minimum-field omissions used to audit a Pass label."""
    omissions: list[str] = []
    if row.get("code_available") != "Yes" or row.get("code_url") in {"", "NA", "NR"}:
        omissions.append("runnable public code")
    if row.get("dataset_versions") in {"", "NA", "NR"} or "version NR" in row.get(
        "dataset_versions", ""
    ):
        omissions.append("dataset/cohort version")
    if row.get("data_access") == "Private":
        omissions.append("lawful reproducibility pathway")
    if row.get("task_scope") != "Benchmark/evaluation framework" and row.get(
        "seeds_or_repeats"
    ) in {"", "NA", "NR"}:
        omissions.append("seeds or repeats")
    if row.get("utility_evaluated") == "Yes" and row.get("held_out_real_test") == "Unclear":
        omissions.append("held-out split logic")
    if row.get("evaluated_systems") in {"", "NA", "NR"}:
        omissions.append("evaluated system/version")
    return omissions


def validate(
    rows: list[dict[str, str]],
    fields: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    errors: list[str] = []
    issues: list[dict[str, str]] = []

    if len(rows) != 30:
        errors.append(f"Expected 30 included reports; found {len(rows)}")

    if fields != EXPECTED_FIELDS:
        errors.append("CSV header does not exactly match the 62-field v0.1.0 schema")

    ids = [row.get("report_id", "") for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("report_id values are not unique")

    keys = [row.get("citation_key", "") for row in rows]
    if len(keys) != len(set(keys)):
        errors.append("citation_key values are not unique")

    titles = [row.get("title", "") for row in rows]
    if len(titles) != len(set(titles)):
        errors.append("title values are not unique")

    for row in rows:
        rid = row.get("report_id", "<missing>")
        for field in REQUIRED_FIELDS:
            if not row.get(field):
                errors.append(f"{rid}: missing required field {field}")

        for field in BINARY_FIELDS:
            if row.get(field) not in {"Yes", "No"}:
                errors.append(
                    f"{rid}: {field} must be Yes or No, got {row.get(field)!r}"
                )

        for field, allowed in ENUM_COLUMNS.items():
            if row.get(field) not in set(allowed):
                errors.append(
                    f"{rid}: {field} must be one of {allowed}, got {row.get(field)!r}"
                )

        if row.get("eligibility_status") != "Included":
            errors.append(f"{rid}: every row in the pilot denominator must be Included")

        try:
            year = int(row.get("year", ""))
        except ValueError:
            errors.append(f"{rid}: year is not an integer")
        else:
            if not 2018 <= year <= 2025:
                errors.append(f"{rid}: year {year} is outside the frozen evidence window")

        if not row.get("primary_source_url", "").startswith(("https://", "http://")):
            errors.append(f"{rid}: primary_source_url is not an HTTP(S) URL")

        for dimension, (evaluated_field, maturity_field) in DIMENSIONS.items():
            try:
                level = int(row.get(maturity_field, ""))
            except ValueError:
                errors.append(f"{rid}: {maturity_field} is not an integer")
                continue
            if level not in range(5):
                errors.append(f"{rid}: {maturity_field} must be 0–4")
            evaluated = row.get(evaluated_field)
            if evaluated == "No" and level != 0:
                errors.append(
                    f"{rid}: {dimension} not evaluated but maturity is {level}"
                )
            if evaluated == "Yes" and level == 0:
                errors.append(
                    f"{rid}: {dimension} evaluated but maturity remains zero"
                )

        if row.get("formal_dp_verified") == "Yes" and row.get("dp_claimed") != "Yes":
            errors.append(f"{rid}: verified formal DP requires dp_claimed=Yes")

        expected_dp_status = (
            "Not a DP claim"
            if row.get("dp_claimed") == "No"
            else "Complete"
            if row.get("formal_dp_verified") == "Yes"
            else "Incomplete"
        )
        if row.get("dp_evidence_status") != expected_dp_status:
            errors.append(
                f"{rid}: dp_evidence_status should be {expected_dp_status!r}"
            )

        if row.get("epsilon_reported") == "Yes" and row.get("dp_claimed") != "Yes":
            errors.append(f"{rid}: a DP epsilon is reported but dp_claimed is not Yes")

        if row.get("epsilon_reported") == "Yes" and row.get("epsilon_values") in {
            "", "NA", "NR",
        }:
            errors.append(f"{rid}: epsilon_reported=Yes requires numeric epsilon_values")

        if row.get("delta_reported") == "Yes" and row.get("delta_values") in {
            "", "NA", "NR",
        }:
            errors.append(f"{rid}: delta_reported=Yes requires numeric delta_values")

        if row.get("formal_dp_verified") == "Yes":
            for field in ("dp_mechanism", "epsilon_reported"):
                if row.get(field) in {"", "NR", "NA", "No"}:
                    issues.append(
                        {
                            "report_id": rid,
                            "severity": "warning",
                            "field": field,
                            "message": "Formal DP claim lacks a complete reported parameter field",
                        }
                    )

        if row.get("privacy_evaluated") == "No" and row.get("privacy_test_type") not in {
            "",
            "NR",
            "NA",
            "None",
        }:
            errors.append(
                f"{rid}: privacy_evaluated is No but privacy_test_type contains evidence"
            )

        if row.get("task_scope") == "Benchmark/evaluation framework" and row.get(
            "architecture_family"
        ) != "Evaluation framework":
            errors.append(
                f"{rid}: evaluation framework is mislabeled as a generator architecture"
            )

        gate_omissions = evaluability_omissions(row)
        if row.get("evaluability_gate") == "Pass" and gate_omissions:
            errors.append(
                f"{rid}: Pass gate has objective omissions: {', '.join(gate_omissions)}"
            )

        if row.get("peer_reviewed") == "Yes" and row.get("publication_type") in {
            "Preprint",
            "Thesis",
        }:
            errors.append(
                f"{rid}: {row.get('publication_type')} cannot be coded peer-reviewed Yes without explanation"
            )

        if row.get("verification_status") != "Primary source verified; human double-code pending":
            issues.append(
                {
                    "report_id": rid,
                    "severity": "info",
                    "field": "verification_status",
                    "message": row.get("verification_status", ""),
                }
            )

        if row.get("ambiguity_flag") == "Yes":
            issues.append(
                {
                    "report_id": rid,
                    "severity": "human-review",
                    "field": "ambiguity_flag",
                    "message": row.get("evidence_notes", "Ambiguous classification"),
                }
            )

    return errors, issues


def counter_rows(
    section: str, field: str, rows: list[dict[str, str]], denominator: int
) -> list[dict[str, object]]:
    return [
        {
            "section": section,
            "measure": value,
            "count": count,
            "denominator": denominator,
            "percent": round(pct(count, denominator), 1),
            "source_filter": f"{field}={value}",
        }
        for value, count in sorted(Counter(row[field] for row in rows).items())
    ]


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    total = len(rows)
    summary: list[dict[str, object]] = [
        {
            "section": "Corpus",
            "measure": "Included reports",
            "count": total,
            "denominator": total,
            "percent": 100.0,
            "source_filter": "eligibility_status=Included",
        }
    ]
    for section, field in [
        ("Publication type", "publication_type"),
        ("Peer review", "peer_reviewed"),
        ("Modality", "modality"),
        ("Task scope", "task_scope"),
        ("Architecture", "architecture_family"),
        ("Evaluability", "evaluability_gate"),
    ]:
        summary.extend(counter_rows(section, field, rows, total))

    headline_fields = [
        ("Evidence", "Fidelity evaluated", "fidelity_evaluated"),
        ("Evidence", "Clinical validity evaluated", "clinical_validity_evaluated"),
        ("Evidence", "Utility evaluated", "utility_evaluated"),
        ("Evidence", "Privacy evaluated", "privacy_evaluated"),
        ("Privacy", "DP claimed by report", "dp_claimed"),
        ("Privacy", "Formal DP documentation threshold met", "formal_dp_verified"),
        ("Privacy", "Numeric epsilon reported", "epsilon_reported"),
        ("Privacy", "Numeric delta reported", "delta_reported"),
        ("Privacy", "Membership inference", "membership_inference"),
        ("Privacy", "Attribute inference", "attribute_inference"),
        ("Equity", "Fairness/subgroup evidence", "fairness_evaluated"),
        ("Robustness", "Explicit robustness evaluation", "robustness_evaluated"),
        ("Robustness", "External validation", "external_validation"),
        ("Reporting", "Uncertainty reported", "uncertainty_reported"),
        ("Dataset", "MIMIC-III used", "mimic_iii_used"),
    ]
    for section, measure, field in headline_fields:
        count = sum(row[field] == "Yes" for row in rows)
        summary.append(
            {
                "section": section,
                "measure": measure,
                "count": count,
                "denominator": total,
                "percent": round(pct(count, total), 1),
                "source_filter": f"{field}=Yes",
            }
        )
    return summary


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "$": r"\$",
    }
    return "".join(replacements.get(char, char) for char in value)


def write_tex_outputs(
    output_dir: Path,
    rows: list[dict[str, str]],
    summary: list[dict[str, object]],
) -> None:
    by_measure = {str(item["measure"]): item for item in summary}

    macro_map = {
        "CorpusN": "Included reports",
        "PrivacyN": "Privacy evaluated",
        "FormalDPN": "Formal DP documentation threshold met",
        "EpsilonN": "Numeric epsilon reported",
        "FairnessN": "Fairness/subgroup evidence",
        "RobustnessN": "Explicit robustness evaluation",
        "ExternalN": "External validation",
        "MIMICIIIN": "MIMIC-III used",
        "PassN": "Pass",
    }
    lines = ["% Generated by scripts/score_evidence.py; do not edit manually."]
    for macro, measure in macro_map.items():
        item = by_measure[measure]
        lines.append(f"\\newcommand{{\\{macro}}}{{{item['count']}}}")
        lines.append(
            f"\\newcommand{{\\{macro}Pct}}{{{percent_text(float(item['percent'])).replace('%', r'\%')}}}"
        )
    (output_dir / "generated_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    table_lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Evidence item & $n$ & \% \\",
        r"\midrule",
    ]
    for measure in [
        "Privacy evaluated",
        "DP claimed by report",
        "Formal DP documentation threshold met",
        "Numeric epsilon reported",
        "Fairness/subgroup evidence",
        "Explicit robustness evaluation",
        "External validation",
        "MIMIC-III used",
        "Pass",
    ]:
        item = by_measure[measure]
        table_lines.append(
            f"{tex_escape(measure)} & {item['count']} & {float(item['percent']):.1f} \\\\"
        )
    table_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "headline_table.tex").write_text(
        "\n".join(table_lines) + "\n", encoding="utf-8"
    )

    maturity_lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dimension & 0 & 1 & 2 & 3 & 4 & Median \\",
        r"\midrule",
    ]
    for dimension, (_, maturity_field) in DIMENSIONS.items():
        values = sorted(int(row[maturity_field]) for row in rows)
        counts = Counter(values)
        midpoint = len(values) // 2
        median = (
            values[midpoint]
            if len(values) % 2
            else (values[midpoint - 1] + values[midpoint]) / 2
        )
        maturity_lines.append(
            f"{dimension.title()} & "
            + " & ".join(str(counts[level]) for level in range(5))
            + f" & {median:g} \\\\"
        )
    maturity_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "maturity_table.tex").write_text(
        "\n".join(maturity_lines) + "\n", encoding="utf-8"
    )

    corpus_lines = [
        r"\begin{longtable}{P{1.1cm}P{3.0cm}P{2.6cm}P{3.8cm}P{1.8cm}P{2.2cm}}",
        r"\caption{Included-report audit overview. EMP is ordered as fidelity, utility, privacy, equity, and robustness. A superscript asterisk marks an unresolved ambiguity flag.}\label{tab:corpusoverview}\\",
        r"\toprule",
        r"ID & Citation & Modality & Contribution & EMP & Gate \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"ID & Citation & Modality & Contribution & EMP & Gate \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in sorted(rows, key=lambda item: item["report_id"]):
        profile = ",".join(
            row[DIMENSIONS[dimension][1]] for dimension in DIMENSIONS
        )
        corpus_lines.append(
            " & ".join(
                [
                    tex_escape(row["report_id"])
                    + (r"\textsuperscript{*}" if row["ambiguity_flag"] == "Yes" else ""),
                    rf"\citealp{{{row['citation_key']}}}",
                    tex_escape(row["modality"]),
                    tex_escape(row["task_scope"]),
                    rf"[{profile}]",
                    tex_escape(row["evaluability_gate"]),
                ]
            )
            + r" \\"
        )
    corpus_lines.extend([r"\bottomrule", r"\end{longtable}"])
    (output_dir / "corpus_table.tex").write_text(
        "\n".join(corpus_lines) + "\n", encoding="utf-8"
    )

    sensitivity_rows = [row for row in rows if row["ambiguity_flag"] == "No"]
    sensitivity_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Evidence item & Full $n/30$ & Full \% & Sensitivity $n/28$ & Sensitivity \% \\",
        r"\midrule",
    ]
    for label, field in [
        ("Privacy evaluated", "privacy_evaluated"),
        ("Formal DP documentation threshold met", "formal_dp_verified"),
        ("Numeric epsilon reported", "epsilon_reported"),
        ("Fairness/subgroup evidence", "fairness_evaluated"),
        ("Explicit robustness evaluation", "robustness_evaluated"),
        ("MIMIC-III used", "mimic_iii_used"),
    ]:
        full_n = sum(row[field] == "Yes" for row in rows)
        sensitivity_n = sum(row[field] == "Yes" for row in sensitivity_rows)
        sensitivity_lines.append(
            f"{tex_escape(label)} & {full_n}/30 & {pct(full_n, 30):.1f} & "
            f"{sensitivity_n}/28 & {pct(sensitivity_n, 28):.1f} \\\\"
        )
    sensitivity_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "sensitivity_table.tex").write_text(
        "\n".join(sensitivity_lines) + "\n", encoding="utf-8"
    )


def make_figures(figure_dir: Path, rows: list[dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise SystemExit(f"matplotlib/numpy required for figures: {exc}")

    figure_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (row["modality"], int(row["year"]), row["report_id"]))
    matrix = np.array(
        [
            [int(row[DIMENSIONS[d][1]]) for d in DIMENSIONS]
            for row in ordered
        ]
    )
    fig, ax = plt.subplots(figsize=(8.0, 9.2))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=4, aspect="auto")
    ax.set_xticks(range(len(DIMENSIONS)), [d.title() for d in DIMENSIONS], rotation=28, ha="right")
    ax.set_yticks(range(len(ordered)), [row["citation_key"] for row in ordered], fontsize=7)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if matrix[i, j] >= 3 else "black")
    ax.set_title("Evidence maturity by included report")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Evidence maturity (0–4)")
    fig.tight_layout()
    fig.savefig(figure_dir / "evidence_heatmap.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "evidence_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    modalities = sorted({row["modality"] for row in rows})
    x = np.arange(len(modalities))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for index, dimension in enumerate(DIMENSIONS):
        field = DIMENSIONS[dimension][1]
        shares = [
            100.0 * np.mean([int(row[field]) >= 2 for row in rows if row["modality"] == modality])
            for modality in modalities
        ]
        ax.bar(x + (index - 2) * width, shares, width, label=dimension.title())
    ax.set_xticks(x, modalities, rotation=18, ha="right")
    ax.set_ylabel("Reports with substantive evidence (EMP ≥2), %")
    ax.set_ylim(0, 100)
    ax.legend(ncols=3, frameon=False, fontsize=8)
    ax.set_title("Substantive evidence by modality")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "maturity_by_modality.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "maturity_by_modality.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows, fields = read_rows(args.input)
    errors, issues = validate(rows, fields)
    if errors:
        message = "Corpus validation failed:\n- " + "\n- ".join(errors)
        raise SystemExit(message)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(rows)
    write_csv(
        args.output_dir / "evidence_summary.csv",
        summary,
        ["section", "measure", "count", "denominator", "percent", "source_filter"],
    )
    write_csv(
        args.output_dir / "audit_issues.csv",
        issues,
        ["report_id", "severity", "field", "message"],
    )

    gate_rows = []
    for row in rows:
        omissions = evaluability_omissions(row)
        gate_rows.append(
            {
                "report_id": row["report_id"],
                "citation_key": row["citation_key"],
                "evaluability_gate": row["evaluability_gate"],
                "objective_omissions": "; ".join(omissions) if omissions else "None",
                "basis": (
                    "Objective minimum fields present; source-level judgment retained"
                    if not omissions
                    else "Objective omissions plus source-level evidence notes"
                ),
            }
        )
    write_csv(
        args.output_dir / "evaluability_audit.csv",
        gate_rows,
        [
            "report_id", "citation_key", "evaluability_gate",
            "objective_omissions", "basis",
        ],
    )

    maturity_rows = []
    for row in rows:
        maturity_rows.append(
            {
                "report_id": row["report_id"],
                "citation_key": row["citation_key"],
                "modality": row["modality"],
                **{d: int(row[field]) for d, (_, field) in DIMENSIONS.items()},
                "evaluability_gate": row["evaluability_gate"],
            }
        )
    write_csv(
        args.output_dir / "evidence_maturity_audit.csv",
        maturity_rows,
        ["report_id", "citation_key", "modality", *DIMENSIONS.keys(), "evaluability_gate"],
    )

    sensitivity_rows = [row for row in rows if row["ambiguity_flag"] == "No"]
    sensitivity_output = []
    for label, field in [
        ("Privacy evaluated", "privacy_evaluated"),
        ("Formal DP documentation threshold met", "formal_dp_verified"),
        ("Numeric epsilon reported", "epsilon_reported"),
        ("Fairness/subgroup evidence", "fairness_evaluated"),
        ("Explicit robustness evaluation", "robustness_evaluated"),
        ("MIMIC-III used", "mimic_iii_used"),
    ]:
        full_n = sum(row[field] == "Yes" for row in rows)
        sensitivity_n = sum(row[field] == "Yes" for row in sensitivity_rows)
        sensitivity_output.append(
            {
                "measure": label,
                "full_count": full_n,
                "full_denominator": len(rows),
                "full_percent": round(pct(full_n, len(rows)), 1),
                "sensitivity_count": sensitivity_n,
                "sensitivity_denominator": len(sensitivity_rows),
                "sensitivity_percent": round(pct(sensitivity_n, len(sensitivity_rows)), 1),
                "exclusion": "ambiguity_flag=Yes",
            }
        )
    write_csv(
        args.output_dir / "sensitivity_analysis.csv",
        sensitivity_output,
        [
            "measure", "full_count", "full_denominator", "full_percent",
            "sensitivity_count", "sensitivity_denominator",
            "sensitivity_percent", "exclusion",
        ],
    )

    write_tex_outputs(args.output_dir, rows, summary)
    make_figures(args.figure_dir, rows)

    output_names = [
        "evidence_summary.csv",
        "audit_issues.csv",
        "evaluability_audit.csv",
        "evidence_maturity_audit.csv",
        "generated_numbers.tex",
        "headline_table.tex",
        "maturity_table.tex",
        "corpus_table.tex",
        "sensitivity_analysis.csv",
        "sensitivity_table.tex",
    ]
    figure_names = [
        "evidence_heatmap.pdf",
        "evidence_heatmap.png",
        "maturity_by_modality.pdf",
        "maturity_by_modality.png",
    ]
    manifest = {
        "benchmark_version": "0.1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_n": len(rows),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "outputs": output_names,
        "figures": figure_names,
        "output_sha256": {
            name: sha256_file(args.output_dir / name) for name in output_names
        },
        "figure_sha256": {
            name: sha256_file(args.figure_dir / name) for name in figure_names
        },
        "claim_boundary": "Evidence maturity only; no model-performance leaderboard",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "reports": len(rows), "issues": len(issues)}))


if __name__ == "__main__":
    main()
