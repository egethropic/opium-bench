"""Research-calibration leakage, negative controls and independent scoring."""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from lab.calibration import (binary_metrics, corpus_summary, evaluate_fitted, family_interval,
    fit_calibration, fit_readout, load_research_corpus, pool_hidden, readout_score,
    validate_config, validate_corpus)
from lab.calibration_validation import (RUBRIC, blinded_continuations, diagnostic_specs,
    grade_objective_continuation, perturbation_delta, score_blinded_sheet, select_operating_range)


def fixture():
    data = load_research_corpus()
    families = {s: sorted({r['family'] for r in data['rows'] if r['split'] == s})[:2]
                for s in ('train', 'probe', 'selection', 'heldout')}
    data['rows'] = [r for r in data['rows'] if r['family'] in families[r['split']]]
    data['version'] = 'synthetic-subset-fixture'
    return data


def activations(data):
    matrix = np.random.default_rng(73).normal(0, .02, (len(data['rows']), 6)) + 3
    for i, row in enumerate(data['rows']):
        matrix[i, 0 if row['concept'] == 'pain' else 1] += row['label'] * 2
    return {'1:final': matrix, '2:final': matrix.copy()}


def configuration(**kw):
    return {'layers': [1], 'downstream_layer': 2, 'poolings': ['final'], 'probe_methods': ['mean', 'ridge'],
            'ridge_alphas': [1., 10.], 'bootstrap_samples': 20, **kw}


class CorpusTests(unittest.TestCase):
    def test_frozen_authored_corpus_regenerates_and_has_planned_counts(self):
        corpus = load_research_corpus()
        path = Path(__file__).resolve().parents[1] / 'corpora' / 'build_research_v2.py'
        spec = importlib.util.spec_from_file_location('corpus_builder', path)
        builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
        self.assertEqual(corpus, builder.build())
        summary = corpus_summary(corpus)
        for split, pairs in (('train', 100), ('probe', 40), ('selection', 40), ('heldout', 60)):
            self.assertEqual(summary['splits'][split]['pairs'], {'pain': pairs, 'joy': pairs})
        self.assertEqual(len(corpus['rows']), 960)
        self.assertEqual(len({r['family'] for r in corpus['rows']}), 48)
        self.assertTrue(all(any(r['lexical'] == 'negation_and_topic' and r['label'] == 0 for r in corpus['rows'] if r['split'] == s) for s in summary['splits']))

    def test_scenario_and_template_leakage_and_duplicate_text_fail(self):
        for field in ('family', 'template_family', 'text', 'id'):
            corpus = fixture()
            source = corpus['rows'][0][field]
            next(r for r in corpus['rows'] if r['split'] == 'heldout')[field] = source
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_corpus(corpus)

    def test_unpaired_slice_mismatch_and_unsupported_rows_fail(self):
        for mutate in (lambda d: d['rows'].pop(), lambda d: d['rows'][0].update(style='unmatched'),
                       lambda d: d['rows'][0].update(label=True), lambda d: d['rows'][0].update(span=[0, 100000]),
                       lambda d: d['rows'][0].update(unknown=True)):
            corpus = fixture(); mutate(corpus)
            with self.assertRaises(ValueError): validate_corpus(corpus)

    def test_strict_config_and_future_versions(self):
        for config in ({'schema_version': 3}, {'poolings': ['last']}, {'ridge_alphas': [0]},
                       {'layers': [True]}, {'doses': [float('nan')]}, {'undeclared': 1}, {'downstream_layer': 0}):
            with self.subTest(config=config), self.assertRaises(ValueError): validate_config(config)
        cfg = validate_config({}, layer_count=36)
        self.assertEqual(cfg['downstream_layer'], 35)
        self.assertEqual(cfg['poolings'], ['final', 'mean'])


class PoolingTests(unittest.TestCase):
    def test_numpy_pooling_ignores_left_and_right_padding(self):
        h = np.array([[[99., 99.], [1., 3.], [3., 5.], [77., 77.]], [[2., 4.], [4., 6.], [77., 77.], [99., 99.]]])
        mask = np.array([[0, 1, 1, 0], [1, 1, 0, 0]])
        np.testing.assert_array_equal(pool_hidden(h, mask, 'final'), [[3., 5.], [4., 6.]])
        np.testing.assert_array_equal(pool_hidden(h, mask, 'mean'), [[2., 4.], [3., 5.]])
        np.testing.assert_array_equal(pool_hidden(h, mask, 'span', np.array([[0, 1, 0, 0], [0, 1, 0, 0]])), [[1., 3.], [4., 6.]])
        padded = h.copy(); padded[~mask.astype(bool)] = np.nan
        np.testing.assert_array_equal(pool_hidden(padded, mask, 'mean'), [[2., 4.], [3., 5.]])
        with self.assertRaisesRegex(ValueError, 'empty'): pool_hidden(h, np.zeros((2, 4)))
        with self.assertRaisesRegex(ValueError, 'empty span'): pool_hidden(h, mask, 'span', np.array([[1, 0, 0, 0], [0, 0, 1, 0]]))

    def test_torch_matches_numpy_without_gpu(self):
        import torch
        h = np.arange(24).reshape(2, 4, 3).astype(np.float32)
        mask = np.array([[0, 1, 1, 0], [1, 1, 0, 0]])
        span = np.array([[0, 1, 0, 0], [0, 1, 0, 0]])
        for policy in ('final', 'mean', 'span'):
            actual = pool_hidden(torch.tensor(h), torch.tensor(mask), policy, torch.tensor(span))
            np.testing.assert_array_equal(actual.numpy(), pool_hidden(h, mask, policy, span))


class FitTests(unittest.TestCase):
    def test_rank_deficient_ridge_and_null_readouts_are_finite(self):
        x = np.ones((4, 20)); x[:2, :3] = 2
        readout = fit_readout(x, [1, 1, 0, 0], 'ridge', .001)
        self.assertTrue(np.isfinite(readout['weight']).all())
        self.assertEqual(binary_metrics([1, 1, 0, 0], readout_score(readout, x))['auc'], 1.)
        for method in ('ridge', 'mean'):
            readout = fit_readout(np.ones((4, 20)), [1, 1, 0, 0], method)
            np.testing.assert_array_equal(readout_score(readout, np.ones((4, 20))), np.zeros(4))

    def test_heldout_permutation_cannot_change_fitted_vectors_or_site(self):
        corpus = fixture(); matrices = activations(corpus)
        baseline = fit_calibration(matrices, corpus, configuration())
        changed = copy.deepcopy(corpus); altered = {k: v.copy() for k, v in matrices.items()}
        for i, row in enumerate(changed['rows']):
            if row['split'] == 'heldout':
                row['label'] = 1 - row['label']
                for x in altered.values(): x[i] *= -500
        result = fit_calibration(altered, changed, configuration())
        self.assertEqual(result['metadata']['selected'], baseline['metadata']['selected'])
        self.assertEqual(result['metadata']['candidate_layers'], baseline['metadata']['candidate_layers'])
        for name in baseline['vectors']:
            np.testing.assert_array_equal(result['vectors'][name], baseline['vectors'][name], err_msg=name)
        self.assertNotEqual(result['metadata']['corpus']['sha256'], baseline['metadata']['corpus']['sha256'])

    def test_independent_fit_splits_and_deterministic_shuffles(self):
        corpus = fixture(); matrices = activations(corpus)
        result = fit_calibration(matrices, corpus, configuration())
        repeat = fit_calibration(matrices, corpus, configuration())
        for key in result['vectors']:
            np.testing.assert_array_equal(result['vectors'][key], repeat['vectors'][key])
        altered = {k: v.copy() for k, v in matrices.items()}
        for i, row in enumerate(corpus['rows']):
            if row['split'] == 'train':
                for x in altered.values(): x[i, 4] += row['label'] * 20
        train_change = fit_calibration(altered, corpus, configuration())
        np.testing.assert_array_equal(result['vectors']['probe_pain_weight'], train_change['vectors']['probe_pain_weight'])
        self.assertFalse(np.array_equal(result['vectors']['pain'], train_change['vectors']['pain']))
        self.assertNotEqual(result['metadata']['shuffle_seeds']['pain'], result['metadata']['shuffle_seeds']['joy'])

    def test_slice_metrics_expose_a_lexical_shortcut(self):
        corpus = fixture(); matrices = activations(corpus)
        for i, row in enumerate(corpus['rows']):
            if row['split'] == 'heldout' and row['lexical'] == 'negation_and_topic':
                for x in matrices.values(): x[i, 0 if row['concept'] == 'pain' else 1] = 3 + (1 - row['label']) * 2
        result = fit_calibration(matrices, corpus, configuration(probe_methods=['ridge'], ridge_alphas=[1.]))
        report = evaluate_fitted(result, matrices, corpus)
        pain = report['sites']['probe']['pain']
        self.assertGreater(pain['auc'], .5)
        self.assertEqual(pain['slices']['lexical']['negation_and_topic']['auc'], 0.)
        self.assertEqual(pain['uncertainty']['unit'], 'scenario_family')
        self.assertIn('cross_concept_discrimination', report['sites']['downstream'])
        self.assertIn('pain', report['shuffled_controls'])
        self.assertEqual(result['metadata']['status'], 'unvalidated')

    def test_family_bootstrap_clusters_correlated_paraphrases(self):
        rows = [{'family': f, 'label': y} for f in ('a', 'b', 'c') for y in (0, 1) for _ in range(5)]
        scores = np.array([r['label'] for r in rows]); report = family_interval(rows, scores, samples=20)
        self.assertEqual(report['families'], 3)
        self.assertEqual(report['auc_95'], [1., 1.])
        self.assertEqual(report, family_interval(rows, scores, samples=20))

    def test_incompatible_activation_shapes_fail(self):
        corpus = fixture(); matrices = activations(corpus)
        for invalid in ({'1:final': matrices['1:final']}, {**matrices, '2:final': np.zeros((1, 2))}):
            with self.assertRaises(ValueError): fit_calibration(invalid, corpus, configuration())


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.vectors = {'pain': np.array([1., 0., 0.]), 'joy': np.array([0., 1., 0.]), 'neutral': np.array([1., 1., 1.]), 'scale': 2.}
        self.hidden = np.array([[5., 4., 3.], [2., 8., 1.]])

    def test_signed_addition_attenuation_and_zero(self):
        np.testing.assert_array_equal(perturbation_delta(self.hidden, self.vectors, {'operator': 'sham'}), np.zeros_like(self.hidden))
        positive = perturbation_delta(self.hidden, self.vectors, {'operator': 'add', 'axis': 'pain', 'gain': .5})
        negative = perturbation_delta(self.hidden, self.vectors, {'operator': 'add', 'axis': 'pain', 'gain': -.5})
        np.testing.assert_array_equal(positive, -negative)
        attenuated = self.hidden + perturbation_delta(self.hidden, self.vectors, {'operator': 'attenuate', 'axes': ['pain'], 'fraction': 1.})
        np.testing.assert_array_equal(attenuated[:, 0], [0., 0.])
        np.testing.assert_array_equal(attenuated[:, 1:], self.hidden[:, 1:])

    def test_joint_projection_is_order_independent_and_checks_rank(self):
        a = perturbation_delta(self.hidden, self.vectors, {'operator': 'attenuate', 'axes': ['pain', 'joy'], 'fraction': .5})
        b = perturbation_delta(self.hidden, self.vectors, {'operator': 'attenuate', 'axes': ['joy', 'pain'], 'fraction': .5})
        np.testing.assert_allclose(a, b)
        vectors = {**self.vectors, 'joy': self.vectors['pain']}
        with self.assertRaisesRegex(ValueError, 'rank'):
            perturbation_delta(self.hidden, vectors, {'operator': 'attenuate', 'axes': ['pain', 'joy'], 'fraction': .5})

    def test_random_controls_match_actual_per_position_delta(self):
        combined = perturbation_delta(self.hidden, self.vectors, {'operator': 'combined', 'joy_gain': .5, 'pain_suppression': .5})
        spec = {'operator': 'random_matched', 'seed': 12}
        random = perturbation_delta(self.hidden, self.vectors, spec, combined)
        np.testing.assert_allclose(np.linalg.norm(combined, axis=1), np.linalg.norm(random, axis=1))
        self.assertFalse(np.allclose(combined, random))
        np.testing.assert_array_equal(random, perturbation_delta(self.hidden, self.vectors, spec, combined))
        with self.assertRaisesRegex(ValueError, 'actual'): perturbation_delta(self.hidden, self.vectors, spec)

    def test_sweeps_include_independent_controls_and_selection_guard(self):
        specs = diagnostic_specs([0, .5]); self.assertEqual(len(specs), 9)
        self.assertEqual(len({r['id'] for r in specs}), 9)
        rows = [{'split': 'selection', 'operator': 'combined', 'dose': dose, 'mean_next_token_kl': dose, 'mean_relative_delta': .1} for dose in (0., .5, 1.)]
        self.assertEqual(select_operating_range(rows)['selected_dose'], .5)
        for split in ('heldout', 'probe', None):
            rows[0]['split'] = split
            with self.assertRaisesRegex(ValueError, 'heldout'): select_operating_range(rows)


class ContinuationTests(unittest.TestCase):
    def records(self):
        return [{'record_id': f'r{i}', 'prompt': "Describe the visitor's response.", 'continuation': text,
                 'condition': condition, 'split': 'heldout', 'probe_score': 99}
                for i, (condition, text) in enumerate((('sham', 'The visitor checked the clock.'), ('active', 'The visitor looked pleased.')))]

    def test_export_hides_conditions_scores_and_preserves_rubric(self):
        sheet, key = blinded_continuations(self.records())
        for sample in sheet['samples']:
            self.assertEqual(set(sample), {'blind_id', 'prompt', 'continuation', 'ratings'})
            self.assertNotIn('probe_score', sample)
        self.assertEqual(sheet['rubric'], RUBRIC)
        self.assertEqual(sheet, blinded_continuations(self.records())[0])
        result = score_blinded_sheet(sheet, key)
        self.assertEqual(result['conditions']['active']['fields']['joy_experience_language']['missing'], 1)

    def test_scoring_rejects_tampered_text_rubric_or_missing_sample(self):
        for mutate in (lambda d: d['samples'][0].update(continuation='edited'), lambda d: d['samples'].pop(),
                       lambda d: d['rubric'].update(version='edited'), lambda d: d['samples'][0]['ratings'].update(coherence=True)):
            sheet, key = blinded_continuations(self.records()); mutate(sheet)
            with self.assertRaises(ValueError): score_blinded_sheet(sheet, key)

    def test_human_semantics_and_objective_grades_remain_separate(self):
        sheet, key = blinded_continuations(self.records())
        sheet['samples'][0]['ratings'].update(joy_experience_language=2, notes='A cheerful phrase', topic_only=False)
        result = score_blinded_sheet(sheet, key); self.assertNotIn('mood', result)
        self.assertTrue(grade_objective_continuation('5', expected=5)['objective_correct'])
        self.assertFalse(grade_objective_continuation('5 and I am happy', expected=5)['objective_correct'])
        self.assertFalse(grade_objective_continuation('{"x": NaN}', required_keys=['x'])['format_valid'])
        self.assertTrue(grade_objective_continuation('{"x": 2}', required_keys=['x'])['format_valid'])


if __name__ == '__main__': unittest.main()
