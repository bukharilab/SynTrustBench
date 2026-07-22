#!/usr/bin/env python3
"""Validate SynTrustBench interchange files and benchmark cards."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    annotation_schema = json.loads(
        (ROOT / "data" / "annotation_schema.json").read_text(encoding="utf-8")
    )
    annotation_validator = Draft202012Validator(
        annotation_schema, format_checker=FormatChecker()
    )
    jsonl_rows = [
        json.loads(line)
        for line in (ROOT / "data" / "study_annotations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    for index, row in enumerate(jsonl_rows, start=1):
        errors = sorted(annotation_validator.iter_errors(row), key=lambda item: list(item.path))
        if errors:
            raise SystemExit(
                f"JSONL row {index} failed schema validation: "
                + "; ".join(error.message for error in errors)
            )

    with (ROOT / "data" / "study_annotations.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(jsonl_rows):
        raise SystemExit("CSV and JSONL row counts differ")
    for csv_row, jsonl_row in zip(csv_rows, jsonl_rows, strict=True):
        if csv_row["report_id"] != jsonl_row["report_id"]:
            raise SystemExit("CSV and JSONL row order/identity differ")
        for field, value in jsonl_row.items():
            if csv_row[field] != str(value):
                raise SystemExit(
                    f"{csv_row['report_id']}: CSV/JSONL mismatch in {field}"
                )

    card_schema = json.loads(
        (ROOT / "benchmark_card.schema.json").read_text(encoding="utf-8")
    )
    card_validator = Draft202012Validator(card_schema)
    for relative_path in [
        Path("benchmark_card.yaml"),
        Path("examples/ppgan_benchmark_card.yaml"),
    ]:
        card = yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))
        errors = sorted(card_validator.iter_errors(card), key=lambda item: list(item.path))
        if errors:
            raise SystemExit(
                f"{relative_path} failed schema validation: "
                + "; ".join(error.message for error in errors)
            )

    print(
        json.dumps(
            {
                "status": "ok",
                "annotation_rows": len(jsonl_rows),
                "cards_validated": 2,
            }
        )
    )


if __name__ == "__main__":
    main()
