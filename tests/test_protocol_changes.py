from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from syntrustbench import __version__, evaluate
from syntrustbench.config import load_config
from syntrustbench.constraints import constraint_violations
from syntrustbench.fidelity import evaluate_fidelity
from syntrustbench.equity import evaluate_equity
from syntrustbench.privacy import evaluate_privacy
from syntrustbench.reporting import DIMENSIONS, _benchmark_card, _coverage, _render_report, benchmark_gate
from syntrustbench.robustness import evaluate_robustness
from syntrustbench.stats import aggregate_status, metric, safe_auroc
from syntrustbench.utility import _paired_interval, evaluate_utility
from syntrustbench.validation import load_submission
from tests.test_config import valid_config
from tests.test_end_to_end import make_data


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.train, self.test, self.synthetic = make_data()
        self.raw = valid_config()
        self.submission = load_submission(self.train, self.test, self.synthetic, self.raw)

    def utility(self, trtr_ci=(0.7, 0.9), tstr_ci=(0.6, 0.9), retention_ci=(0.91, 1.1)):
        y = self.submission.real_test.outcome.to_numpy(dtype=int)
        trtr = np.where(y, 0.8, 0.2)
        tstr = trtr.copy()
        tstr[:30] = 1 - tstr[:30]
        intervals = [trtr_ci, tstr_ci, (0.5, 0.9), (0.5, 0.9), (0.1, 0.3), (0.1, 0.3)]
        if trtr_ci[0] is not None and trtr_ci[0] > 0.5:
            intervals.append(retention_ci)
        intervals.append((-0.4, -0.1))
        with patch('syntrustbench.utility.train_classifier', side_effect=[(y, trtr), (y, tstr)]), patch(
            'syntrustbench.utility._paired_interval', side_effect=intervals
        ) as bootstrap:
            result = evaluate_utility(self.submission, np.random.default_rng(7))
        return result, bootstrap.call_args_list, (y, trtr, tstr)

    def test_chance_retention_difference_and_paired_bootstrap(self):
        result, calls, (y, trtr, tstr) = self.utility()
        real_auc, synthetic_auc = safe_auroc(y, trtr), safe_auroc(y, tstr)
        self.assertAlmostEqual(result['metrics']['auroc_utility_retention']['estimate'],
                               (synthetic_auc - 0.5) / (real_auc - 0.5))
        self.assertAlmostEqual(result['metrics']['tstr_minus_trtr_auroc']['estimate'], synthetic_auc - real_auc)
        self.assertEqual(result['metrics']['tstr_minus_trtr_auroc']['ci_low'], -0.4)
        self.assertEqual(result['_internal']['chance_corrected_retention'],
                         result['metrics']['auroc_utility_retention']['estimate'])
        self.assertTrue(result['_internal']['retention_evaluable'])
        self.assertNotIn('retention', result['_internal'])
        for call in calls[-2:]:
            args = call.args
            self.assertIs(args[2], trtr)
            self.assertIs(args[3], tstr)
            self.assertEqual(args[-1], 0.95)
            self.assertAlmostEqual(args[4](y, trtr, tstr), args[0])
        retention_statistic = calls[-2].args[4]
        labels = np.array([0, 0, 1, 1])
        below_chance = np.array([0.9, 0.8, 0.2, 0.1])
        above_chance = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertTrue(np.isnan(retention_statistic(labels, below_chance, above_chance)))
        # Identical predictions must yield a zero-width paired difference interval.
        low, high = _paired_interval(0, y, trtr, trtr,
            lambda labels, a, b: safe_auroc(labels, b) - safe_auroc(labels, a),
            np.random.default_rng(8), 20, 0.95)
        self.assertEqual((low, high), (0, 0))

    def test_utility_ci_branches(self):
        cases = [
            ((0.5, 0.8), (0.2, 0.4), (0.1, 0.2), 'NotEvaluated'),
            ((0.49, 0.8), (0.6, 0.9), (0.91, 1.1), 'NotEvaluated'),
            ((None, None), (0.6, 0.9), (0.91, 1.1), 'NotEvaluated'),
            ((0.7, 0.9), (0.3, 0.5), (0.91, 1.1), 'Fail'),
            ((0.7, 0.9), (0.6, 0.9), (0.6, 0.79), 'Fail'),
            ((0.7, 0.9), (0.6, 0.9), (0.6, 0.80), 'Conditional'),
            ((0.7, 0.9), (0.6, 0.9), (0.89, 1.1), 'Conditional'),
            ((0.7, 0.9), (0.6, 0.9), (0.90, 1.1), 'Pass'),
            ((0.7, 0.9), (0.6, 0.9), (None, None), 'NotEvaluated'),
        ]
        for real_ci, synthetic_ci, retention_ci, expected in cases:
            with self.subTest(expected=expected, bounds=(real_ci, synthetic_ci, retention_ci)):
                result, calls, _ = self.utility(real_ci, synthetic_ci, retention_ci)
                self.assertEqual(result['status'], expected)
                self.assertEqual(result['metrics']['auroc_utility_retention']['status'], expected)
                if real_ci[0] is None or real_ci[0] <= 0.5:
                    self.assertIsNone(result['metrics']['auroc_utility_retention']['estimate'])
                    self.assertEqual(len(calls), 7)

    def test_utility_auroc_bounds_are_95_percent(self):
        self.submission.config.confidence_level = 0.8
        _, calls, _ = self.utility()
        self.assertEqual([call.args[-1] for call in calls], [0.95, 0.95, 0.8, 0.8, 0.8, 0.8, 0.95, 0.95])

    def test_required_and_overall_aggregation(self):
        for statuses, expected in [(['Pass'], 'Pass'), (['Conditional', 'Pass'], 'Conditional'),
            (['NotEvaluated', 'Conditional'], 'NotEvaluated'), (['Fail', 'NotEvaluated'], 'Fail')]:
            with self.subTest(statuses=statuses):
                self.assertEqual(aggregate_status(statuses), expected)
                dimensions = {name: {'status': 'Pass'} for name in DIMENSIONS}
                for name, status in zip(DIMENSIONS, statuses):
                    dimensions[name]['status'] = status
                self.assertEqual(benchmark_gate(dimensions), 'Conditional' if expected == 'NotEvaluated' else expected)

    def robustness(self):
        utility, _, _ = self.utility()
        y = self.submission.real_test.outcome.to_numpy(dtype=int)
        utility['metrics']['tstr_auroc'] = metric(1., ci_low=.99, ci_high=1., higher_is_worse=False)
        utility['metrics']['trtr_auroc']['estimate'] = 1.
        utility['_internal']['chance_corrected_retention'] = 1.
        with patch('syntrustbench.robustness.train_classifier', return_value=(y, y.astype(float))):
            return evaluate_robustness(self.submission, utility, np.random.default_rng(7))

    def test_optional_robustness_checks_do_not_block_pass(self):
        result = self.robustness()
        self.assertEqual(result['status'], 'Pass')
        self.assertEqual(result['dimension'], 'robustness')
        self.assertEqual(result['label'], 'Core Predictive-Utility Robustness')
        for name in ['generator_seed_tstr_auroc_cv', 'temporal_or_site_shift_auroc_gap']:
            self.assertEqual(result['metrics'][name]['status'], 'NotEvaluated')
            self.assertEqual(result['metrics'][name]['availability_label'], 'Not evaluated — optional')
            self.assertFalse(result['metrics'][name]['required'])

    def test_unavailable_required_robustness_checks(self):
        for field, value, name in [
            ('model_seeds', [], 'downstream_model_seed_auroc_cv'),
            ('model_seeds', [7], 'downstream_model_seed_auroc_cv'),
            ('missingness_rates', [], 'tstr_auroc_drop_at_maximum_missingness'),
            ('training_size_fractions', [], 'training_size_auroc_spread'),
            ('training_size_fractions', [1.], 'training_size_auroc_spread')]:
            with self.subTest(field=field, value=value):
                original = getattr(self.submission.config, field)
                setattr(self.submission.config, field, value)
                result = self.robustness()
                setattr(self.submission.config, field, original)
                self.assertEqual(result['status'], 'NotEvaluated')
                self.assertEqual(result['metrics'][name]['status'], 'NotEvaluated')
        utility, _, _ = self.utility()
        utility['metrics']['tstr_auroc']['ci_width'] = None
        y = self.submission.real_test.outcome.to_numpy(dtype=int)
        with patch('syntrustbench.robustness.train_classifier', return_value=(y, y.astype(float))):
            result = evaluate_robustness(self.submission, utility, np.random.default_rng(7))
        self.assertEqual(result['status'], 'NotEvaluated')
        self.assertEqual(result['metrics']['tstr_bootstrap_relative_ci_width']['status'], 'NotEvaluated')

    def privacy(self, bounds=(.4, .55), tpr=0., submission=None):
        submission = submission or self.submission
        with patch('syntrustbench.privacy.percentile_interval', side_effect=[(0., 0.), (0., 1.), (0., .1), bounds]), patch(
            'syntrustbench.privacy._tpr_at_fpr', return_value=tpr
        ):
            return evaluate_privacy(submission, np.random.default_rng(7))

    def test_membership_ci_branches_and_exploratory_tpr(self):
        self.submission.config.thresholds['privacy_dcr_share_conditional'] = 1.
        for bounds, expected in [((.4, .55), 'Pass'), ((.55, .6), 'Conditional'),
            ((.4, .6), 'Conditional'), ((.551, .7), 'Fail'), ((None, None), 'NotEvaluated')]:
            with self.subTest(bounds=bounds):
                low_tpr = self.privacy(bounds, 0.)
                high_tpr = self.privacy(bounds, 1.)
                self.assertEqual(low_tpr['status'], expected)
                self.assertEqual(high_tpr['status'], expected)
                self.assertEqual(high_tpr['metrics']['distance_membership_attack_auc']['status'], expected)
                tpr = high_tpr['metrics']['distance_membership_attack_tpr_at_1pct_fpr']
                self.assertEqual(tpr['estimate'], 1.)
                self.assertFalse(tpr['flag'])
                self.assertIn('not decision-driving', tpr['note'])
                self.assertIsNone(tpr['threshold'])

    def test_membership_small_samples_and_copy_priority(self):
        for part in ['real_train', 'real_test', 'synthetic']:
            small = copy.deepcopy(self.submission)
            setattr(small, part, getattr(small, part).iloc[:19])
            with self.subTest(part=part):
                result = self.privacy(submission=small)
                self.assertEqual(result['status'], 'NotEvaluated')
                self.assertEqual(result['metrics']['distance_membership_attack_auc']['status'], 'NotEvaluated')
        small.synthetic.iloc[0] = small.real_train.iloc[0]
        small.config.thresholds['privacy_duplicate_rate_fail'] = 1.
        self.assertEqual(self.privacy(submission=small)['status'], 'Fail')

    def test_nearest_neighbor_exposure_is_conditional(self):
        self.submission.config.thresholds['privacy_dcr_share_conditional'] = -1.
        self.assertEqual(self.privacy()['status'], 'Conditional')
        self.assertEqual(self.privacy((.6, .8))['status'], 'Fail')

    def test_exact_overlap_stops_before_any_evaluation(self):
        self.test.iloc[:2] = self.train.iloc[:2].to_numpy()
        with tempfile.TemporaryDirectory() as directory, patch('syntrustbench.engine.evaluate_fidelity') as evaluator:
            with self.assertRaisesRegex(ValueError, '2 overlapping held-out rows'):
                evaluate(self.train, self.test, self.synthetic, self.raw, directory)
            evaluator.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_entity_overlap_and_missing_column(self):
        self.raw['data'] = {'entity_id': 'patient_id'}
        self.train['patient_id'] = np.arange(len(self.train))
        self.test['patient_id'] = np.arange(len(self.test)) + 1000
        for missing_part in ['real_train', 'real_test']:
            with self.subTest(missing_part=missing_part):
                train = self.train.drop(columns='patient_id') if missing_part == 'real_train' else self.train
                test = self.test.drop(columns='patient_id') if missing_part == 'real_test' else self.test
                with self.assertRaisesRegex(ValueError, missing_part + ' is missing entity identifier'):
                    load_submission(train, test, self.synthetic, self.raw)
        self.test.loc[:2, 'patient_id'] = [0, 0, 1]
        with self.assertRaisesRegex(ValueError, '2 overlapping identifiers'):
            load_submission(self.train, self.test, self.synthetic, self.raw)

    def test_entity_missing_values_and_feature_exclusion(self):
        self.raw['data'] = {'entity_id': 'patient_id'}
        self.train['patient_id'] = np.arange(len(self.train))
        self.test['patient_id'] = np.arange(len(self.test)) + 1000
        submission = load_submission(self.train, self.test, self.synthetic, self.raw)
        self.assertNotIn('patient_id', submission.config.feature_columns)
        self.assertFalse(any('patient-level independence' in warning for warning in submission.warnings))
        self.raw['data']['entity_id'] = 'age'
        self.assertNotIn('age', load_config(self.raw).feature_columns)
        self.raw['utility_task']['features'] = ['age', 'risk']
        self.assertIn('age', load_config(self.raw).feature_columns)

    def test_constraint_maximum_rate_and_missingness_exclusion(self):
        frame = pd.DataFrame({'age': [20., 10., np.nan], 'risk': [np.nan, 1., 2.]})
        details = {}
        overall, rates, warnings = constraint_violations(frame, ['age >= 18', 'risk >= 0'], details=details)
        self.assertEqual(rates, {'age >= 18': .5, 'risk >= 0': 0.})
        self.assertEqual(overall, .5)
        self.assertEqual(warnings, [])
        self.assertEqual(details['age >= 18'], {
            'total_eligible_rows': 3, 'evaluated_rows': 2, 'excluded_missing_rows': 1,
            'violating_rows': 1, 'violation_rate': .5, 'status': 'Available'})
        self.assertEqual(constraint_violations(frame, ['age == age'])[0], 0.)
        self.assertEqual(constraint_violations(frame, ['age >= risk'])[1]['age >= risk'], 0.)

    def test_constraint_zero_evaluable_rows(self):
        details = {}
        overall, rates, _ = constraint_violations(pd.DataFrame({'age': [np.nan, np.nan]}), ['age >= 18'], details=details)
        self.assertIsNone(overall)
        self.assertIsNone(rates['age >= 18'])
        self.assertEqual(details['age >= 18']['status'], 'NotEvaluated')
        self.submission.synthetic['age'] = np.nan
        result = evaluate_fidelity(self.submission, np.random.default_rng(7))
        self.assertEqual(result['metrics']['clinical_constraint_violation_rate']['status'], 'NotEvaluated')
        self.assertEqual(result['status'], 'NotEvaluated')
        self.assertEqual(result['details']['clinical_constraints']['age >= 18']['excluded_missing_rows'], len(self.synthetic))

    def test_report_unavailable_evidence_and_independence_limitation(self):
        dimensions = {name: {'status': 'Pass', 'metrics': {}} for name in DIMENSIONS}
        dimensions['utility'] = self.utility(trtr_ci=(.5, .8))[0]
        dimensions['robustness'] = self.robustness()
        dimensions['privacy']['threat_model'] = {key: 'test' for key in [
            'attacker_knowledge', 'attacker_access', 'attacker_goal', 'interpretation']}
        report = _render_report(self.submission, dimensions, benchmark_gate(dimensions), 'test', self.submission.warnings)
        self.assertIn('Only exact row overlap was checked; patient-level independence could not be verified.', report)
        self.assertIn('Core Predictive-Utility Robustness', report)
        self.assertIn('Not evaluated — optional', report)
        self.assertIn('utility.auroc_utility_retention', report)
        card = _benchmark_card(self.submission, dimensions, benchmark_gate(dimensions))
        self.assertEqual(card['results']['unavailable_required_evidence'], ['utility.auroc_utility_retention'])
        self.assertEqual(card['results']['benchmark_gate'], 'Conditional')
        coverage = _coverage({'robustness': dimensions['robustness'], **{
            name: {'metrics': {}} for name in DIMENSIONS if name != 'robustness'}})
        self.assertEqual(coverage['available_metrics'], 4)

    def test_robustness_chance_corrected_reversal_and_raw_ratio_ignored(self):
        utility, _, _ = self.utility()
        utility['metrics']['trtr_auroc']['estimate'] = .8
        utility['metrics']['tstr_auroc'] = metric(.77, ci_low=.75, ci_high=.79, higher_is_worse=False)
        utility['_internal']['chance_corrected_retention'] = (.77 - .5) / (.8 - .5)
        y = self.submission.real_test.outcome.to_numpy(dtype=int)
        # Raw perturbed ratio is .70/.80=.875 (above .80), but chance-corrected is 2/3.
        for legacy_raw in [0., 1., float('nan')]:
            utility['_internal']['retention'] = legacy_raw
            with patch('syntrustbench.robustness.train_classifier', return_value=(y, y.astype(float))), patch(
                'syntrustbench.robustness.safe_auroc', return_value=.70
            ):
                result = evaluate_robustness(self.submission, utility, np.random.default_rng(7))
            self.assertTrue(result['details']['conclusion_reversal_under_missingness'])
            self.assertEqual(result['status'], 'Fail')
            self.assertAlmostEqual(result['metrics']['tstr_auroc_drop_at_maximum_missingness']['estimate'], .07)
        # The same configured Utility boundary controls reversal, including equality.
        for threshold, baseline, expected in [(.6, .9, False), (.8, .8, True), (.8, .79, False)]:
            self.submission.config.thresholds['utility_retention_fail'] = threshold
            utility['_internal']['chance_corrected_retention'] = baseline
            with patch('syntrustbench.robustness.train_classifier', return_value=(y, y.astype(float))), patch(
                'syntrustbench.robustness.safe_auroc', return_value=.70
            ):
                result = evaluate_robustness(self.submission, utility, np.random.default_rng(7))
            self.assertEqual(result['details']['conclusion_reversal_under_missingness'], expected)

    def test_unavailable_utility_retention_cannot_reverse_robustness(self):
        utility, _, _ = self.utility(trtr_ci=(.5, .8))
        self.assertFalse(utility['_internal']['retention_evaluable'])
        self.assertFalse(np.isfinite(utility['_internal']['chance_corrected_retention']))
        utility['_internal']['retention'] = 1.  # Legacy raw value must have no effect.
        y = self.submission.real_test.outcome.to_numpy(dtype=int)
        with patch('syntrustbench.robustness.train_classifier', return_value=(y, y.astype(float))), patch(
            'syntrustbench.robustness.safe_auroc', return_value=.5
        ):
            result = evaluate_robustness(self.submission, utility, np.random.default_rng(7))
        self.assertFalse(result['details']['conclusion_reversal_under_missingness'])
        self.assertNotEqual(result['status'], 'Fail')

    def test_equity_chance_retention_and_unavailable_description(self):
        utility, _, _ = self.utility()
        privacy = {'_internal': {'duplicate_mask': np.zeros(len(self.synthetic), dtype=bool),
                                'synthetic_dcr': np.ones(len(self.synthetic)), 'real_real_floor': 0.}}
        for trtr_low in [.6, .5, .49, None]:
            with self.subTest(trtr_low=trtr_low):
                def paired(point, statistic, size, *args):
                    self.assertAlmostEqual(point, .8)
                    with patch('syntrustbench.equity.safe_auroc', side_effect=[.8, .74]):
                        self.assertAlmostEqual(statistic(np.arange(size)), .8)
                    for chance in [.5, .4]:
                        with patch('syntrustbench.equity.safe_auroc', side_effect=[chance, .74]):
                            self.assertFalse(np.isfinite(statistic(np.arange(size))))
                    return .7, .9
                with patch('syntrustbench.equity._subgroup_interval', side_effect=[
                    (.8, trtr_low, .9), (.74, .7, .8)] * 2), patch(
                    'syntrustbench.equity.percentile_interval', side_effect=paired
                ) as interval:
                    result = evaluate_equity(self.submission, utility, privacy, np.random.default_rng(7))
                self.assertEqual(result['status'], 'Pass')
                for row in result['subgroup_rows']:
                    self.assertEqual(row['status'], 'Evaluable')
                    if trtr_low is not None and trtr_low > .5:
                        self.assertAlmostEqual(row['utility_retention'], .8)
                        self.assertEqual(row['utility_retention_ci_low'], .7)
                        self.assertEqual(row['utility_retention_ci_high'], .9)
                        self.assertEqual(row['utility_retention_status'], 'Available')
                    else:
                        for key in ['utility_retention', 'utility_retention_ci_low', 'utility_retention_ci_high']:
                            self.assertIsNone(row[key])
                        self.assertEqual(row['utility_retention_status'], 'NotEvaluated')
                        interval.assert_not_called()

    def test_fidelity_maximum_rule_failure_precedes_unavailable_rule(self):
        self.submission.config.clinical_constraints = ['age >= 18', 'risk >= 0']
        self.submission.synthetic['age'] = np.nan
        self.submission.synthetic['risk'] = -1.
        result = evaluate_fidelity(self.submission, np.random.default_rng(7))
        self.assertEqual(result['metrics']['clinical_constraint_violation_rate']['estimate'], 1.)
        self.assertEqual(result['details']['clinical_constraints']['age >= 18']['status'], 'NotEvaluated')
        self.assertEqual(result['status'], 'Fail')
        self.submission.synthetic['risk'] = 1.
        result = evaluate_fidelity(self.submission, np.random.default_rng(7))
        self.assertEqual(result['metrics']['clinical_constraint_violation_rate']['estimate'], 0.)
        self.assertEqual(result['metrics']['clinical_constraint_violation_rate']['status'], 'NotEvaluated')
        self.assertEqual(result['status'], 'NotEvaluated')

    def test_entity_identifiers_require_every_real_row(self):
        self.raw['data'] = {'entity_id': 'patient_id'}
        for split in ['real_train', 'real_test']:
            for all_missing in [False, True]:
                train, test = self.train.copy(), self.test.copy()
                train['patient_id'] = np.arange(len(train), dtype=float)
                test['patient_id'] = np.arange(len(test), dtype=float) + 1000
                frame = train if split == 'real_train' else test
                count = len(frame) if all_missing else 3
                frame.loc[frame.index[:count], 'patient_id'] = np.nan
                with self.subTest(split=split, count=count), self.assertRaisesRegex(
                    ValueError, rf'{split}\.patient_id: {count} missing entity identifiers'
                ):
                    load_submission(train, test, self.synthetic, self.raw)

    def test_entity_overlap_normalizes_types_and_counts_unique_ids(self):
        self.raw['data'] = {'entity_id': 'patient_id'}
        self.train['patient_id'] = np.arange(len(self.train))
        self.test['patient_id'] = [str(i + 1000) for i in range(len(self.test))]
        for representations in [['0', '0.0', ' 1 '], [0., '00', '1e0']]:
            self.test.loc[:2, 'patient_id'] = pd.Series(representations, dtype=object)
            with self.subTest(representations=representations), self.assertRaisesRegex(
                ValueError, '2 overlapping identifiers'
            ):
                load_submission(self.train, self.test, self.synthetic, self.raw)

    def test_protocol_version(self):
        from syntrustbench.cli import _parser
        self.assertEqual(__version__, '0.3.0')
        root = Path(__file__).resolve().parents[1]
        self.assertIn('version = "0.3.0"', (root / 'pyproject.toml').read_text())
        import contextlib
        import io
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
            _parser().parse_args(['--version'])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn('0.3.0', output.getvalue())

    def test_updated_schemas(self):
        # Keep schema verification dependency-free; JSON Schema is not a core dependency.
        root = Path(__file__).resolve().parents[1]
        summary_schema = json.loads((root / 'schemas/summary.schema.json').read_text())
        for name in DIMENSIONS:
            self.assertIn('NotEvaluated', summary_schema['properties']['dimension_status']['properties'][name]['enum'])
        self.assertNotIn('NotEvaluated', summary_schema['properties']['benchmark_gate']['enum'])
        config_schema = json.loads((root / 'schemas/config.schema.json').read_text())
        self.assertIn('entity_id', config_schema['properties']['data']['properties'])
        self.assertTrue(config_schema['$id'].endswith('config-v0.3.json'))


if __name__ == '__main__':
    unittest.main()
