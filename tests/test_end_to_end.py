from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from syntrustbench import evaluate
from syntrustbench.demo import inject_training_duplicates, reduce_subgroup
from tests.test_config import valid_config


def make_data(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    def cohort(size: int) -> pd.DataFrame:
        age = rng.normal(55, 12, size).clip(18, 90)
        sex = rng.choice(["F", "M"], size=size)
        risk = 0.04 * (age - 55) + (sex == "M") * 0.2 + rng.normal(0, 1, size)
        probability = 1 / (1 + np.exp(-risk))
        outcome = (rng.random(size) < probability).astype(int)
        return pd.DataFrame(
            {
                "age": age.round(2),
                "risk": risk.round(3),
                "sex": sex,
                "outcome": outcome,
            }
        )

    real_train = cohort(180)
    real_test = cohort(100)
    synthetic = real_train.sample(n=180, replace=True, random_state=seed).reset_index(
        drop=True
    )
    synthetic["age"] += rng.normal(0, 1.0, len(synthetic))
    synthetic["risk"] += rng.normal(0, 0.1, len(synthetic))
    return real_train, real_test, synthetic


class EndToEndTests(unittest.TestCase):
    def test_evaluate_writes_complete_output_contract(self):
        real_train, real_test, synthetic = make_data()
        progress_messages = []
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate(
                real_train,
                real_test,
                synthetic,
                valid_config(),
                directory,
                progress=progress_messages.append,
            )
            expected = {
                "summary.json",
                "metrics.csv",
                "subgroup_results.csv",
                "benchmark_card.yaml",
                "report.md",
                "execution_log.txt",
                "run_manifest.json",
                "details.json",
            }
            self.assertEqual(expected, {path.name for path in Path(directory).iterdir()})
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertIn(result.benchmark_gate, {"Pass", "Conditional", "Fail"})
            self.assertEqual(
                set(summary["dimension_status"]),
                {"fidelity", "utility", "privacy", "equity", "robustness"},
            )
            self.assertNotIn("evaluability_gate", summary)
            self.assertIn("input_hashes", json.loads(
                (Path(directory) / "run_manifest.json").read_text()
            ))
            self.assertEqual(progress_messages[-1], "SynTrustBench evaluation complete")

    def test_duplicate_injection_triggers_privacy_failure(self):
        real_train, real_test, synthetic = make_data()
        duplicated = inject_training_duplicates(synthetic, real_train, fraction=0.10)
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate(
                real_train,
                real_test,
                duplicated,
                valid_config(),
                directory,
            )
        privacy = result.dimensions["privacy"]
        self.assertEqual(privacy["status"], "Fail")
        self.assertGreater(
            privacy["metrics"]["exact_training_duplicate_rate"]["estimate"], 0
        )

    def test_subgroup_reduction_increases_representation_gap(self):
        real_train, real_test, synthetic = make_data()
        reduced = reduce_subgroup(
            synthetic, "sex", "F", removal_fraction=0.90, preserve_row_count=True
        )
        with tempfile.TemporaryDirectory() as baseline_dir, tempfile.TemporaryDirectory() as reduced_dir:
            baseline = evaluate(
                real_train, real_test, synthetic, valid_config(), baseline_dir
            )
            degraded = evaluate(
                real_train, real_test, reduced, valid_config(), reduced_dir
            )
        baseline_gap = baseline.dimensions["equity"]["metrics"][
            "maximum_representation_gap"
        ]["estimate"]
        degraded_gap = degraded.dimensions["equity"]["metrics"][
            "maximum_representation_gap"
        ]["estimate"]
        self.assertGreater(degraded_gap, baseline_gap)
        self.assertIn(degraded.dimensions["equity"]["status"], {"Conditional", "Fail"})

    def test_large_privacy_inputs_use_declared_bounds(self):
        real_train, real_test, synthetic = make_data()
        real_train = pd.concat([real_train] * 4, ignore_index=True)
        real_test = pd.concat([real_test] * 6, ignore_index=True)
        synthetic = pd.concat([synthetic] * 4, ignore_index=True)
        config = valid_config()
        config["privacy"] = {
            "distance_max_rows": 500,
            "attack_max_rows": 500,
        }

        with tempfile.TemporaryDirectory() as directory:
            result = evaluate(
                real_train,
                real_test,
                synthetic,
                config,
                directory,
            )

        details = result.dimensions["privacy"]["details"]
        self.assertEqual(details["distance_reference_rows"], 500)
        self.assertEqual(details["distance_query_rows"], 500)
        self.assertEqual(details["attack_member_rows"], 500)
        self.assertTrue(
            any("bounded samples" in warning for warning in result.dimensions["privacy"]["warnings"])
        )


if __name__ == "__main__":
    unittest.main()
