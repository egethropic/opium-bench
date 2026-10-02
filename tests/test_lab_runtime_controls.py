"""Actual rounded random-edit matching, counterfactual isolation and fail-closed controls."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lab.checkpoints import capture, restore, validate
from lab.controller_v2 import RecipeV2Controller
from lab.recipes_v2 import resolve_recipe
from lab.runtime_controls import (RuntimeControlUnavailable, match_rounded_random,
                                   prepare_random_match, validate_runtime_controls)
from test_lab_runtime import package, runtime
import test_lab_task_axes as task_fixtures

CONTROLS = {"random_norm_match": {"target_preset_id":"opium", "reference":"same_unedited_position",
                                 "relative_tolerance":.01, "absolute_tolerance":1e-6}}


def controller(**options):
    cfg = resolve_recipe({"recipe_version":2, "condition":"random", "conditions":["random"], **options})
    return RecipeV2Controller(cfg)


def generate(c, *, pkg=None, callback=None, rt=None, thinking=False, max_tokens=3, controls=CONTROLS):
    rt = runtime() if rt is None else rt
    events = []
    def emit(event):
        events.append(deepcopy(event))
        if event["type"] == "token":
            c.record_exposure(event["phase"], event["dose"]["coefficients"], event["dose"]["baseline"])
            c.advance()
        elif event["type"] == "prefill":
            c.record_exposure("prefill", event["dose"]["coefficients"], event["dose"]["baseline"], positions=event["positions"])
        if callback:
            callback(event, c)
    with patch.object(rt, "_package", return_value=deepcopy(package() if pkg is None else pkg)):
        result = rt.generate([{"role":"user", "content":"hello"}], [],
                             {"temperature":0, "thinking":thinking, "max_new_tokens":max_tokens},
                             "fixture", control=c.snapshot, emit=emit, runtime_controls=controls)
    return result, events


def scoped_config(**changes):
    cfg = resolve_recipe({"recipe_version":2, "condition":"random", "conditions":["random"]})
    for preset in cfg["effect_presets"]:
        if "stacking" in changes:
            preset["stacking"] = deepcopy(changes["stacking"])
        if preset["id"] in ("random", "opium"):
            preset.update(deepcopy(changes))
    return cfg


class RuntimeControlsTests(unittest.TestCase):
    def test_strict_envelope_and_compatibility_checks(self):
        self.assertEqual(validate_runtime_controls(), {})
        self.assertEqual(validate_runtime_controls(CONTROLS, recipe=controller().recipe), CONTROLS)
        bad = [[], {"x":1}, {"random_norm_match":None}, {"random_norm_match":{}},
               {"random_norm_match":{**CONTROLS["random_norm_match"], "relative_tolerance":.02}},
               {"random_norm_match":{**CONTROLS["random_norm_match"], "reference":"other_run"}}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_runtime_controls(value)
        cfg = deepcopy(controller().recipe)
        next(p for p in cfg["effect_presets"] if p["id"] == "random")["decay"]["cutoff"] += 1
        with self.assertRaisesRegex(ValueError, "identical"):
            validate_runtime_controls(CONTROLS, recipe=cfg)
        with self.assertRaisesRegex(ValueError, "baselines"):
            validate_runtime_controls(CONTROLS, recipe=controller(baseline_pain=1).recipe)

    def test_combined_target_norm_matched_without_delivering_target(self):
        c = controller(); c.press()
        _, events = generate(c, max_tokens=1)
        token = events[0]
        m = token["measurements"]
        match = token["dose"]["random_norm_match"]
        # Hidden value before layer1 hook = [1,1,2,1]*1.1**2. Opium
        # counterfactual removes coordinate0 and adds1.5 to coordinate1.
        target_norm = np.linalg.norm([-1.21, 1.5, 0, 0])
        self.assertAlmostEqual(match["target_edit_norms"], target_norm, places=6)
        self.assertAlmostEqual(m["delivered_edit_norm"], target_norm, places=6)
        self.assertEqual(m["pre"], m["post"])  # no pain/joy target delivered
        self.assertNotEqual(match["nominal_random_gain"], token["dose"]["coefficients"]["random_gain"])
        self.assertTrue(match["target_is_counterfactual"])
        self.assertEqual(c.scheduler.pulses[0]["preset_id"], "random")
        self.assertEqual(c.counts["total"], 1)
        self.assertAlmostEqual(c.scheduler.exposure["output"]["random_gain"], target_norm / 2, places=6)

    def test_negative_source_sign_and_actual_coefficient_exposure(self):
        cfg = controller().recipe
        next(p for p in cfg["effect_presets"] if p["id"] == "random")["gains"]["random"] = -.75
        c = RecipeV2Controller(cfg); c.press()
        _, events = generate(c, max_tokens=1)
        token = events[0]
        self.assertLess(token["dose"]["effective"]["random_gain"], 0)
        self.assertAlmostEqual(token["dose"]["random_norm_match"]["norm_ratios"], 1, places=6)
        self.assertEqual(token["dose"]["coefficients"]["random_gain"], c.scheduler.exposure["output"]["random_gain"])

    def test_pulse_decay_gating_and_phase_are_matched_without_phantom_edits(self):
        cfg = scoped_config(phases=["output"], decay={"shape":"linear", "clock":"tokens", "half_life":1, "cutoff":4})
        c = RecipeV2Controller(cfg); c.press()
        _, events = generate(c, rt=runtime(tokens=(2,3,5,0)), thinking=True, max_tokens=4)
        tokens = [e for e in events if e["type"] == "token"]
        self.assertEqual([e["phase"] for e in tokens], ["reasoning","reasoning","output","output"])
        self.assertEqual([e["dose"]["random_norm_match"]["status"] for e in tokens], ["inactive","inactive","matched","matched"])
        self.assertAlmostEqual(tokens[2]["dose"]["random_norm_match"]["target_coefficients"]["joy_orthogonal"], .375)
        self.assertGreater(tokens[2]["measurements"]["delivered_edit_norm"], tokens[3]["measurements"]["delivered_edit_norm"])
        c = controller(); c.press(); c.set_controls(enabled=False)
        _, events = generate(c)
        self.assertTrue(all(e["measurements"]["delivered_edit_norm"] == 0 for e in events))
        self.assertEqual(c.generated_tokens, 3)

    def test_prefill_per_position_and_generation_are_independent_stages(self):
        cfg = scoped_config(phases=["prefill","output"], prefill_positions="all", decay={"shape":"constant", "clock":"tokens", "half_life":1, "cutoff":10})
        c = RecipeV2Controller(cfg); c.press()
        _, events = generate(c, max_tokens=1)
        self.assertEqual([e["type"] for e in events], ["prefill", "prefill", "token"])
        for event in events:
            m = event["dose"]["random_norm_match"]
            np.testing.assert_allclose(m["norm_ratios"], 1, atol=1e-6)
            self.assertEqual(event["dose"]["generated_token_index"], 0)
        self.assertEqual(c.generated_tokens, 1)
        self.assertEqual(c.scheduler.exposure["prefill"]["positions"], 2)

    def test_same_channel_stacking_reconstructs_counterfactual_decay_not_nominal_gain_ratio(self):
        cfg = scoped_config(stacking={"policy":"capped_additive", "cap":4, "channel":"aux"},
                            decay={"shape":"linear", "clock":"tokens", "half_life":1,"cutoff":4})
        c = RecipeV2Controller(cfg); c.press(); c.advance(2); c.press()
        before = deepcopy(c.snapshot())
        match = prepare_random_match(c.snapshot(), CONTROLS)
        target = match["target_by_phase"]["output"]
        self.assertEqual(target["pain_attenuation"], 1)
        self.assertEqual(target["joy_orthogonal"], 1.125)
        self.assertEqual(c.snapshot(), before)
        _, events = generate(c, max_tokens=1)
        np.testing.assert_allclose(events[0]["dose"]["random_norm_match"]["norm_ratios"], 1, atol=1e-6)

    def test_mixed_live_controls_fail_before_next_forward_with_evidence(self):
        c = controller(); c.press()
        rt = runtime(); events = []
        def emit(event):
            events.append(event)
            if event["type"] == "token":
                c.advance()
                c.set_controls(pain=1)
        with patch.object(rt, "_package", return_value=package()), self.assertRaisesRegex(ValueError, "baselines"):
            rt.generate([{"role":"user","content":"hi"}], [], {"max_new_tokens":3}, "fixture",
                        control=c.snapshot, emit=emit, runtime_controls=CONTROLS)
        self.assertEqual(len(rt.model.calls), 1)
        self.assertEqual(events[-1]["type"], "runtime_control_unavailable")
        self.assertEqual(events[-1]["evidence"]["status"], "unavailable")
        self.assertTrue(all(not b._forward_hooks for b in rt.blocks))
        cfg = scoped_config(stacking={"policy":"capped_additive", "cap":4,"channel":"aux"})
        c = RecipeV2Controller(cfg); c.press(); c.scheduler.inject("joy")
        with self.assertRaisesRegex(ValueError, "mixed"):
            prepare_random_match(c.snapshot(), CONTROLS)

    def test_missing_axis_rank_failure_and_gain_limit_are_explicit(self):
        for kind in ("missing", "rank", "gain_limit"):
            pkg = package(); cfg = controller().recipe
            if kind == "missing":
                pkg["vectors"]["random"] = np.zeros(4)
            elif kind == "rank":
                pkg["vectors"]["joy_raw"] = pkg["vectors"]["pain"].copy()
                target = next(p for p in cfg["effect_presets"] if p["id"] == "opium")
                target.update(joy_direction="raw", attenuation={"pain":1,"joy":1})
            else:
                pkg["vectors"]["scale"] = np.array(.01)
            c = RecipeV2Controller(cfg); c.press()
            rt = runtime(); events=[]
            with patch.object(rt, "_package", return_value=pkg), self.assertRaises(ValueError):
                rt.generate([{"role":"user","content":"hi"}], [], {"max_new_tokens":1}, "fixture",
                            control=c.snapshot, emit=events.append, runtime_controls=CONTROLS)
            self.assertEqual(len(rt.model.calls), 1 if kind == "gain_limit" else 0)
            self.assertEqual(events[-1]["type"], "runtime_control_unavailable")
            if kind == "gain_limit":
                self.assertIn("absolute_errors", events[-1]["evidence"])

    def test_rounded_low_precision_matching_and_unattainable_target(self):
        before = torch.tensor([[1000., 1.], [2., 3.]], dtype=torch.float16).float()
        target = before + torch.tensor([[0., 1.], [0., .5]])
        actual, gains, details = match_rounded_random(before, target, torch.tensor([1.,0.]), 1., 1, torch.float16)
        np.testing.assert_allclose(details["norm_ratios"], [1,1], atol=.01)
        self.assertTrue(bool((gains <= 4).all()))
        self.assertTrue(torch.equal(actual, actual.to(torch.float16).float()))
        # bf16 at1000 has spacing4. A target of1 along a small coordinate has
        # no achievable norm along the large-coordinate random direction.
        with self.assertRaises(RuntimeControlUnavailable) as error:
            match_rounded_random(torch.tensor([1000.,1.]), torch.tensor([1000.,2.]), torch.tensor([1.,0.]), 1., 1, torch.bfloat16)
        self.assertEqual(error.exception.evidence["status"], "unavailable")
        self.assertGreater(error.exception.evidence["absolute_errors"], .01)

    def test_empty_controls_preserve_v2_events_exactly_and_legacy_is_rejected(self):
        a=controller(); a.press(); b=controller(); b.press()
        r1,e1 = generate(a, controls=None)
        r2,e2 = generate(b, controls={})
        self.assertEqual(r1,r2); self.assertEqual(e1,e2)
        self.assertFalse(any("random_norm_match" in e["dose"] for e in e1))
        rt = runtime()
        with self.assertRaisesRegex(ValueError, "recipe-v2"):
            rt.generate([{"role":"user","content":"hi"}], [], {"max_new_tokens":1}, None, runtime_controls=CONTROLS)
        self.assertEqual(rt.model.calls, [])

    def test_checkpoint_controls_preserved_hashed_and_not_recipe_fields(self):
        fixture = list(task_fixtures.TaskAxesCheckpointTests().fixture(difficulty="standard", framing="neutral"))
        fixture[0]["runtime_controls"] = deepcopy(CONTROLS)
        checkpoint = capture(*fixture)
        session, _, _, _ = restore(checkpoint)
        self.assertEqual(session["runtime_controls"], CONTROLS)
        self.assertNotIn("runtime_controls", session["config"])
        self.assertEqual(checkpoint["identity"]["runtime_controls_sha256"], task_fixtures.digest(CONTROLS))
        bad = deepcopy(checkpoint)
        bad["session"]["runtime_controls"]["random_norm_match"]["relative_tolerance"] = .1
        with self.assertRaises(ValueError):
            validate(task_fixtures.reseal(bad))

    def test_rounded_zero_match_preserves_operator_flag_for_exposure_parser(self):
        from lab.exposure import exposure_from_event
        cfg = controller().recipe
        target = next(p for p in cfg["effect_presets"] if p["id"] == "opium")
        target["gains"]["joy"] = 1e-7
        target["attenuation"]["pain"] = 0
        c = RecipeV2Controller(cfg); c.press()
        rt = runtime(); pkg = package()
        pkg["vectors"]["scale"] = np.array(1.)
        pkg["vectors"]["random"] = np.array([1.,0.,0.,0.])
        prepared = rt._prepare_control_v2(c.snapshot(), pkg, CONTROLS)
        state = {'prepared':prepared, 'dose':rt._dose_v2(prepared,'output'), 'first_forward':False}
        with rt._hooks_v2(pkg,state):
            rt.blocks[pkg['metadata']['layer']](torch.tensor([[[1000/1.1,0.,0.,0.]]]))
        self.assertGreater(state['dose']['effective']['random_gain'],0)
        self.assertTrue(state['dose']['applied'])
        self.assertFalse(state['dose']['numeric_nonzero'])
        self.assertEqual(state['measurements']['delivered_edit_norm'],0)
        sample=exposure_from_event(dict(type='token',generation_index=0,dose=state['dose'],measurements=state['measurements']),clock='tokens')
        self.assertEqual(sample['delivered_edit_norm_sum'],0)

    def test_checkpoint_yoke_is_bound_to_exact_restored_controller(self):
        from lab.yoke_runner import YokeDriver
        from test_lab_yoke_runner import make_schedule, reseal as seal_driver
        fixture = list(task_fixtures.TaskAxesCheckpointTests().fixture(difficulty="standard", framing="neutral"))
        schedule = make_schedule(events=[], presets=fixture[1].recipe["effect_presets"])
        driver = YokeDriver(fixture[1], schedule)
        fixture[0]["yoke_state"] = driver.snapshot()
        checkpoint = capture(*fixture)
        session, effect, _, _ = restore(checkpoint)
        restored = YokeDriver.restore(session["yoke_state"], effect)
        self.assertEqual(restored.snapshot(), driver.snapshot())
        self.assertEqual(checkpoint["semantics"]["yoke"], "source-exposure-driver-v1")
        bad = deepcopy(checkpoint)
        bad["session"]["yoke_state"]["controller_sha256"] = "f" * 64
        bad["session"]["yoke_state"] = seal_driver(bad["session"]["yoke_state"])
        with self.assertRaises(ValueError):
            validate(task_fixtures.reseal(bad))


if __name__ == "__main__":
    unittest.main()
