from __future__ import annotations

import unittest

from syntrustbench.config import load_config


def valid_config() -> dict:
    return {
        "dataset": {"name": "test", "modality": "tabular", "outcome": "outcome"},
        "columns": {
            "continuous": ["age", "risk"],
            "categorical": ["sex"],
            "binary": ["outcome"],
        },
        "protected_attributes": ["sex"],
        "clinical_constraints": ["age >= 18"],
        "utility_task": {
            "type": "binary_classification",
            "target": "outcome",
            "positive_label": "1",
        },
        "generator": {"name": "test-generator", "seed": 7},
        "uncertainty": {
            "bootstrap_iterations": 20,
            "confidence_level": 0.95,
            "random_seed": 7,
        },
        "subgroup_analysis": {
            "minimum_n": 10,
            "minimum_outcomes_per_class": 2,
        },
        "robustness": {
            "downstream_model_seeds": [7, 8],
            "missingness_rates": [0.10],
            "training_size_fractions": [0.5, 1.0],
        },
    }


class ConfigTests(unittest.TestCase):
    def test_valid_config_loads(self):
        config = load_config(valid_config())
        self.assertEqual(config.target, "outcome")
        self.assertEqual(config.feature_columns, ["age", "risk", "sex"])
        self.assertEqual(config.privacy_distance_max_rows, 10_000)

    def test_duplicate_type_declaration_fails(self):
        raw = valid_config()
        raw["columns"]["categorical"].append("age")
        with self.assertRaisesRegex(ValueError, "exactly one type list"):
            load_config(raw)

    def test_unsupported_modality_fails(self):
        raw = valid_config()
        raw["dataset"]["modality"] = "text"
        with self.assertRaisesRegex(ValueError, "tabular"):
            load_config(raw)


if __name__ == "__main__":
    unittest.main()
