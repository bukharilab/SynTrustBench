"""Run the four controlled degradation checks on a tabular submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from syntrustbench import evaluate
from syntrustbench.demo import (
    add_missingness,
    inject_training_duplicates,
    reduce_subgroup,
    shuffle_feature,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Validate SynTrustBench against controlled synthetic-data failures."
    )
    command.add_argument("--real-train", required=True)
    command.add_argument("--real-test", required=True)
    command.add_argument("--synthetic", required=True)
    command.add_argument("--config", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--subgroup-attribute", required=True)
    command.add_argument("--subgroup-value", required=True)
    command.add_argument("--shuffle-feature", required=True)
    command.add_argument("--missingness-columns", nargs="+", required=True)
    command.add_argument("--quick", action="store_true")
    return command


def main() -> int:
    args = parser().parse_args()
    real_train = pd.read_csv(args.real_train)
    real_test = pd.read_csv(args.real_test)
    synthetic = pd.read_csv(args.synthetic)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.quick:
        config.setdefault("uncertainty", {})["bootstrap_iterations"] = 40
        seed = int(config["uncertainty"].get("random_seed", 42))
        config.setdefault("robustness", {})["downstream_model_seeds"] = [seed, seed + 1]

    variants = {
        "baseline": synthetic,
        "duplicate_injected": inject_training_duplicates(
            synthetic, real_train, fraction=0.12, seed=42
        ),
        "subgroup_reduced": reduce_subgroup(
            synthetic,
            args.subgroup_attribute,
            args.subgroup_value,
            removal_fraction=0.90,
            seed=42,
        ),
        "feature_shuffled": shuffle_feature(
            synthetic, args.shuffle_feature, seed=42
        ),
        "missingness_added": add_missingness(
            synthetic, args.missingness_columns, rate=0.30, seed=42
        ),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, frame in variants.items():
        variant_config = yaml.safe_load(yaml.safe_dump(config))
        generator = variant_config.setdefault("generator", {})
        generator["name"] = f"{generator.get('name', 'unspecified')}::{name}"
        results[name] = evaluate(
            real_train,
            real_test,
            frame,
            variant_config,
            output / name,
        )

    def value(result, dimension: str, metric_name: str):
        return result.dimensions[dimension]["metrics"][metric_name]["estimate"]

    rows = []
    for name, result in results.items():
        rows.append(
            {
                "variant": name,
                "benchmark_gate": result.benchmark_gate,
                **{
                    f"{dimension}_status": status
                    for dimension, status in result.summary["dimension_status"].items()
                },
                "training_copy_rate": value(
                    result, "privacy", "exact_training_duplicate_rate"
                ),
                "maximum_representation_gap": value(
                    result, "equity", "maximum_representation_gap"
                ),
                "association_matrix_error": value(
                    result, "fidelity", "association_matrix_error"
                ),
                "auroc_utility_retention": value(
                    result, "utility", "auroc_utility_retention"
                ),
                "maximum_missingness_rate_gap": value(
                    result, "fidelity", "maximum_missingness_rate_gap"
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        output / "failure_injection_comparison.csv", index=False
    )
    expectations = {
        "duplicate_injected": "privacy exposure should worsen",
        "subgroup_reduced": "equity representation should worsen",
        "feature_shuffled": "dependency fidelity and utility should worsen",
        "missingness_added": (
            "missingness fidelity should worsen; robustness should quantify whether "
            "the downstream conclusion remains stable"
        ),
        "interpretation": (
            "Directional validation on a controlled degradation; not proof of universal "
            "clinical validity."
        ),
    }
    (output / "pre_registered_expectations.json").write_text(
        json.dumps(expectations, indent=2) + "\n", encoding="utf-8"
    )
    print(output / "failure_injection_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
