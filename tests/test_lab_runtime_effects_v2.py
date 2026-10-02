"""Numerical/scope acceptance for opt-in v2 effects on the CPU fixture model."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lab.controller_v2 import RecipeV2Controller
from lab.effects import joint_attenuation
from test_lab_runtime import package, runtime


def controller(preset=None, **options):
    config = {'recipe_version': 2, **options}
    if preset is not None:
        config['effect_presets'] = [{'id': 'opium', **preset}]
    return RecipeV2Controller(config)


def run(rt, c, pkg=None, *, thinking=False, event_callback=None, max_tokens=3):
    pkg = copy.deepcopy(pkg if pkg is not None else package())
    events = []
    def emit(event):
        events.append(event)
        if event_callback: event_callback(event)
        if event['type'] == 'token': c.advance()
    with patch.object(rt, '_package', return_value=pkg):
        result = rt.generate([{'role': 'user', 'content': 'hello'}], [],
                             {'temperature': 0, 'thinking': thinking, 'max_new_tokens': max_tokens},
                             'fixture', control=c.snapshot, emit=emit)
    return result, events


class RuntimeEffectsV2Tests(unittest.TestCase):
    def test_zero_sham_and_passive_measurement_preserve_tokens_and_hooks(self):
        rt = runtime()
        baseline = rt.generate([{'role': 'user', 'content': 'hello'}], [], {'temperature': 0, 'max_new_tokens': 3}, None)
        c = controller(condition='sham'); c.press()
        actual, events = run(rt, c)
        self.assertEqual(actual['token_ids'], baseline['token_ids'])
        self.assertTrue(all(e['measurements']['relative_delta'] == 0 for e in events))
        self.assertEqual(c.generated_tokens, 3)
        self.assertTrue(all(not block._forward_hooks for block in rt.blocks))
        self.assertFalse(any(e['type'] == 'prefill' for e in events))

    def test_signed_raw_and_orthogonal_joy_are_distinct(self):
        pkg = package(); pkg['vectors']['joy_raw'] = np.array([1., 1., 0., 0.]) / np.sqrt(2)
        results = {}
        for direction, sign in (('raw', 1), ('raw', -1), ('orthogonal', 1)):
            c = controller({'gains': {'joy': sign * .5}, 'joy_direction': direction, 'decay': {'shape': 'constant'}}); c.press()
            _, events = run(runtime(), c, pkg)
            m = events[0]['measurements']; results[direction, sign] = np.array([m['post'][k] - m['pre'][k] for k in ('pain', 'joy')])
        np.testing.assert_allclose(results['raw', 1], -results['raw', -1], atol=1e-6)
        self.assertGreater(results['raw', 1][0], 0)
        self.assertEqual(results['orthogonal', 1][0], 0)

    def test_signed_random_gain_changes_actual_unprobed_coordinate(self):
        differences = []
        for sign in (-1, 1):
            rt = runtime(); pkg = package()
            c = controller({'gains': {'random': sign * .5}, 'decay': {'shape': 'constant'}}); c.press()
            prepared = rt._prepare_control_v2(c.snapshot(), pkg)
            state = {'prepared': prepared, 'dose': rt._dose_v2(prepared, 'output'), 'first_forward': False}
            ids = torch.tensor([[30, 30]])
            baseline = rt._forward(input_ids=ids, use_cache=False).logits[0, 0, 13].item()
            with rt._hooks_v2(pkg, state):
                edited = rt._forward(input_ids=ids, use_cache=False).logits[0, 0, 13].item()
            differences.append(edited - baseline)
            self.assertAlmostEqual(state['measurements']['delivered_edit_norm'], 1., places=6)
        self.assertLess(differences[0], 0)
        self.assertGreater(differences[1], 0)
        self.assertAlmostEqual(differences[0], -differences[1], places=6)

    def test_baseline_is_attenuated_before_pulse_addition(self):
        c = controller({'gains': {'pain': .5}, 'attenuation': {'pain': 1}, 'decay': {'shape': 'constant'}}, baseline_pain=2)
        c.press(); _, events = run(runtime(), c)
        first = events[0]
        self.assertEqual(first['dose']['effective']['baseline_pain'], 2.)
        self.assertEqual(first['dose']['effective']['pain'], .5)
        # Reference scale=2 and probe scale=2: only the .5 pulse survives.
        self.assertAlmostEqual(first['measurements']['post']['pain'], .5, places=6)
        self.assertGreater(first['measurements']['delivered_edit_norm'], 0)

    def test_overlapping_joint_attenuation_matches_symmetric_projection(self):
        pkg = package(); pkg['vectors']['joy_raw'] = np.array([1., 1., 0., 0.]) / np.sqrt(2)
        c = controller({'attenuation': {'pain': .5, 'joy': 1}, 'joy_direction': 'raw', 'decay': {'shape': 'constant'}})
        c.press(); _, events = run(runtime(), c, pkg)
        before = np.array([1., 1., 2., 1.]) * 1.1**2
        expected = joint_attenuation(before, [pkg['vectors']['pain'], pkg['vectors']['joy_raw']], [.5, 1.])
        self.assertAlmostEqual(events[0]['measurements']['post']['pain'], expected[0]/2, places=6)
        self.assertAlmostEqual(events[0]['measurements']['post']['joy'], expected[1]/2, places=6)

    def test_rank_deficiency_unsupported_site_and_missing_raw_axis_fail_before_forward(self):
        overlap = package(); overlap['vectors']['joy_raw'] = overlap['vectors']['pain'].copy()
        cases = [(controller({'attenuation': {'pain': 1, 'joy': 1}, 'joy_direction': 'raw'}), overlap, 'rank deficient'),
                 (controller({'gains': {'pain': 1}, 'site': {'layer': 2}}), package(), 'layer'),
                 (controller({'gains': {'joy': 1}, 'joy_direction': 'raw'}), package(), 'joy_raw')]
        for c, pkg, message in cases:
            rt = runtime(); c.press()
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message): run(rt, c, pkg)
            self.assertEqual(rt.model.calls, [])
            self.assertTrue(all(not block._forward_hooks for block in rt.blocks))

    def test_reasoning_and_output_scope_follow_token_phase_and_include_eos(self):
        c = controller({'gains': {'pain': 1}, 'phases': ['output'], 'decay': {'shape': 'constant'}}); c.press()
        result, events = run(runtime(tokens=(2, 3, 5, 0)), c, thinking=True, max_tokens=4)
        self.assertEqual([e['phase'] for e in events], ['reasoning', 'reasoning', 'output', 'output'])
        self.assertEqual([e['dose']['effective']['pain'] for e in events], [0., 0., 1., 1.])
        self.assertEqual(c.generated_tokens, len(result['token_ids']))
        self.assertEqual(events[-1]['token_id'], 0)

    def test_prefill_positions_are_explicit_and_do_not_age_clocks(self):
        for positions, counts in (('last', [1]), ('all', [1, 1])):
            c = controller({'gains': {'pain': 1}, 'phases': ['prefill'], 'prefill_positions': positions,
                            'decay': {'shape': 'linear', 'cutoff': 2}}); c.press()
            seen_clocks = []
            _, events = run(runtime(), c, event_callback=lambda e: seen_clocks.append((e['type'], c.generated_tokens)))
            prefill = [e for e in events if e['type'] == 'prefill']; tokens = [e for e in events if e['type'] == 'token']
            self.assertEqual([e['positions'] for e in prefill], counts)
            self.assertTrue(all(e['generated_token_index'] == 0 for e in prefill))
            self.assertEqual(c.generated_tokens, 3)
            self.assertTrue(all(clock == 0 for kind, clock in seen_clocks if kind == 'prefill'))
            self.assertTrue(all(e['measurements']['relative_delta'] > 0 for e in prefill))
            self.assertTrue(all(e['measurements']['relative_delta'] == 0 for e in tokens))
            self.assertTrue(all(e['dose']['baseline']['pain'] == 0 for e in prefill))
            self.assertGreater(tokens[0]['measurements']['pre']['pain'], .605)

    def test_prefill_then_generation_composition_records_each_stage(self):
        c = controller({'gains': {'pain': .5}, 'phases': ['prefill', 'output'], 'prefill_positions': 'last',
                        'decay': {'shape': 'constant'}}); c.press()
        _, events = run(runtime(), c)
        prefill, token = events[:2]
        self.assertEqual(prefill['type'], 'prefill')
        self.assertEqual(token['type'], 'token')
        self.assertAlmostEqual(token['measurements']['pre']['pain'], prefill['measurements']['post']['pain'], places=6)
        self.assertAlmostEqual(token['measurements']['post']['pain'] - token['measurements']['pre']['pain'], .5, places=6)
        self.assertGreater(token['measurements']['delivered_edit_norm'], 0)
        self.assertIn('not an isolated', prefill['downstream_alignment'])

    def test_control_revision_and_held_baseline_remain_visible_when_aux_disabled(self):
        c = controller(baseline_pain=1)
        c.set_controls(enabled=False)
        _, events = run(runtime(), c)
        self.assertFalse(events[0]['dose']['enabled'])
        self.assertEqual(events[0]['dose']['effective']['baseline_pain'], 1)
        self.assertGreater(events[0]['measurements']['relative_delta'], 0)
        self.assertEqual(events[0]['dose']['control_revision'], c.control_revision)

    def test_recipe_mutation_during_generation_fails_before_next_forward(self):
        c = controller(); rt = runtime()
        def change(event):
            if event['type'] == 'token': c.recipe['effect_presets'][0]['gains']['joy'] = 2
        with self.assertRaisesRegex(ValueError, 'frozen recipe'): run(rt, c, event_callback=change)
        self.assertEqual(len(rt.model.calls), 1)
        self.assertTrue(all(not block._forward_hooks for block in rt.blocks))


if __name__ == '__main__': unittest.main()
