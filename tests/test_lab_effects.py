"""Deterministic effect traces; no models or GPU."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.effects import (EffectScheduler, content_hash, decay_level, import_preset_bundle,
                         joint_attenuation, preset_bundle, validate_preset, validate_presets)


def preset(identifier="effect", **kwargs):
    return validate_preset({"id": identifier, "gains": {"joy": 1}, **kwargs})


def coefficient(scheduler, name="joy_orthogonal", phase="output"):
    return scheduler.coefficients(phase)["coefficients"][name]


class PresetValidationTests(unittest.TestCase):
    def test_signed_axes_and_independent_attenuation_are_resolved_detached(self):
        source = {"gains": {"pain": -2, "joy": -3, "random": .5}, "attenuation": {"joy": .75}}
        result = validate_preset(source)
        self.assertEqual(result["gains"], {"pain": -2., "joy": -3., "random": .5})
        self.assertEqual(result["attenuation"], {"pain": 0., "joy": .75})
        result["gains"]["pain"] = 4
        self.assertEqual(source["gains"]["pain"], -2)

    def test_unknown_fields_versions_and_nonfinite_or_bool_numbers_fail(self):
        bad = [{"schema_version": 2}, {"schema_version": True}, {"script": "code"},
               {"gains": {"pain": True}}, {"gains": {"pain": 10**1000}}, {"gains": {"joy": float("inf")}},
               {"attenuation": {"joy": -1}}, {"decay": {"clock": "wall"}},
               {"site": {"location": "embedding"}}, {"site": {"layer": []}},
               {"phases": ["all"]}, {"phases": ["output", "output"]},
               {"stacking": {"cap": 0}}, {"joy_direction": []},
               {"operation_order": ["addition", "joint_attenuation", "baseline_challenge"]}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_preset(value)

    def test_adapter_rejects_unsupported_prefill_and_site(self):
        capabilities = {"phases": ["reasoning", "output"], "sites": ["residual_post"], "layers": [3]}
        validate_preset({"site": {"layer": 3}}, capabilities=capabilities)
        for value in ({"phases": ["prefill"]}, {"site": {"layer": 4}}):
            with self.assertRaises(ValueError):
                validate_preset(value, capabilities=capabilities)
        capabilities["phases"].append("prefill")
        capabilities["prefill_positions"] = ["last"]
        with self.assertRaises(ValueError):
            validate_preset({"phases": ["prefill"], "prefill_positions": "all"}, capabilities=capabilities)

    def test_library_roundtrip_hash_and_mutation(self):
        original = [preset()]
        bundle = preset_bundle(original)
        restored = import_preset_bundle(json.loads(json.dumps(bundle)))
        self.assertEqual(restored, original)
        restored[0]["label"] = "Edited later"
        self.assertNotEqual(restored, bundle["presets"])
        bundle["presets"][0]["gains"]["joy"] = .5
        with self.assertRaises(ValueError):
            import_preset_bundle(bundle)

    def test_multi_site_and_channel_policy_conflicts_fail(self):
        for second in (preset("two", site={"layer": 4}),
                       preset("two", stacking={"policy": "capped_additive"}),
                       preset("two", stacking={"cap": 2})):
            with self.assertRaises(ValueError):
                validate_presets([preset(), second])
        with self.assertRaises(ValueError):
            validate_presets([preset(), preset()])


class BoundaryTests(unittest.TestCase):
    def test_decay_exact_cutoff_and_first_token(self):
        expected = {"constant": [1, 1, 1, 1], "pulse": [1, 1, 0, 0],
                    "linear": [1, .5, 0, 0], "exponential": [1, .5, 0, 0]}
        for shape, values in expected.items():
            with self.subTest(shape=shape):
                s = EffectScheduler([preset(decay={"shape": shape, "half_life": 2, "cutoff": 4})])
                self.assertEqual(coefficient(s), 0)
                s.inject("effect")
                for increment, value in zip((0, 2, 2, 100), values):
                    s.advance_tokens(increment)
                    self.assertEqual(coefficient(s), value)

    def test_decision_boundary_is_idempotent_and_new_call_does_not_age(self):
        s = EffectScheduler([preset(decay={"clock": "decisions", "shape": "linear", "cutoff": 2})])
        s.complete_decision(1)
        s.inject("effect")
        self.assertEqual(coefficient(s), 1)
        s.advance_tokens(500)
        self.assertEqual(coefficient(s), 1)
        s.complete_decision(2)
        s.complete_decision(2)
        self.assertEqual(coefficient(s), .5)
        # A completed invalid decision ages equally; validity is not an input.
        s.complete_decision(3)
        self.assertEqual(coefficient(s), 0)
        self.assertEqual(s.pulses, [])
        with self.assertRaises(ValueError):
            s.complete_decision(1)

    def test_independent_pulse_ages_signed_combination_and_caps(self):
        s = EffectScheduler([preset(stacking={"policy": "capped_additive", "cap": 1.2}, decay={"half_life": 2})])
        s.inject("effect")
        s.advance_tokens(2)
        s.inject("effect")
        self.assertEqual(coefficient(s), 1.2)
        s.advance_tokens(2)
        self.assertEqual(coefficient(s), .75)
        self.assertEqual([p["age"] for p in s.coefficients()["pulses"]], [4, 2])
        minus = preset("minus", gains={"joy": -1}, stacking={"policy": "capped_additive", "cap": 2})
        plus = preset(stacking={"policy": "capped_additive", "cap": 2})
        s = EffectScheduler([minus, plus])
        s.inject("effect")
        s.inject("minus")
        self.assertEqual(coefficient(s), 0)
        self.assertEqual(len(s.pulses), 2)

    def test_raw_and_orthogonal_joy_remain_distinct_axes(self):
        s = EffectScheduler([preset("raw", joy_direction="raw", stacking={"channel": "raw"}),
                             preset("orthogonal", stacking={"channel": "orthogonal"})])
        s.inject("raw")
        s.inject("orthogonal")
        self.assertEqual(coefficient(s, "joy_raw"), 1)
        self.assertEqual(coefficient(s, "joy_orthogonal"), 1)

    def test_fraction_composition_does_not_exceed_full_attenuation(self):
        s = EffectScheduler([preset(gains={}, attenuation={"pain": .5, "joy": 1}, stacking={"policy": "capped_additive"})])
        s.inject("effect")
        s.inject("effect")
        self.assertEqual(coefficient(s, "pain_attenuation"), .75)
        self.assertEqual(coefficient(s, "joy_orthogonal_attenuation"), 1)

    def test_reset_only_named_channel_and_sham_does_not_reset(self):
        s = EffectScheduler([preset(), preset("other", gains={"pain": 1}, stacking={"channel": "other"})])
        s.inject("other")
        s.inject("effect")
        s.advance_tokens(128)
        s.inject("effect", active=False)
        self.assertEqual(coefficient(s), .5)
        s.inject("effect")
        self.assertEqual(coefficient(s), 1)
        self.assertEqual(coefficient(s, "pain"), .5)
        self.assertEqual(len(s.pulses), 2)

    def test_disable_expiration_cancellation_and_reenable_never_revive(self):
        s = EffectScheduler([preset(decay={"cutoff": 2})])
        s.inject("effect")
        s.schedule("effect", clock="tokens", at=50)
        s.set_enabled(False)
        s.set_enabled(True)
        s.advance_tokens(50)
        self.assertEqual(coefficient(s), 0)
        s.inject("effect")
        s.advance_tokens(2)
        s.set_enabled(True)
        self.assertEqual(s.pulses, [])
        self.assertEqual(coefficient(s), 0)

    def test_transition_cancellation_preserves_other_channel_and_clock(self):
        s = EffectScheduler([preset(), preset("other", stacking={"channel": "other"})])
        s.inject("effect")
        s.inject("other")
        s.advance_tokens(15)
        s.cancel(actor="schedule", channel="aux")
        self.assertEqual(s.generated_tokens, 15)
        self.assertEqual([p["preset_id"] for p in s.pulses], ["other"])

    def test_scheduled_injection_crossed_boundaries_keep_exact_age(self):
        s = EffectScheduler([preset(decay={"half_life": 2})])
        s.schedule("effect", clock="tokens", at=3)
        events = s.advance_tokens(5)
        self.assertEqual(events[0]["generated_tokens"], 3)
        self.assertEqual(coefficient(s), .5)
        self.assertEqual(s.pending, [])
        self.assertEqual(s.complete_decision(10), [])
        with self.assertRaises(ValueError):
            s.schedule("effect", clock="tokens", at=2)

    def test_prefill_separate_clock_and_phase_exposure(self):
        s = EffectScheduler([preset(phases=["prefill", "reasoning"], prefill_positions="last")])
        s.inject("effect")
        s.record_exposure("prefill", positions=1)
        s.record_exposure("reasoning", positions=2)
        self.assertEqual(s.generated_tokens, 0)
        self.assertEqual(s.completed_decisions, 0)
        self.assertEqual(coefficient(s), 0)
        self.assertEqual(s.exposure["prefill"]["positions"], 1)
        self.assertEqual(s.exposure["reasoning"]["joy_orthogonal"], 2)

    def test_mixed_prefill_scopes_have_position_specific_delivered_exposure(self):
        s = EffectScheduler([preset("last", phases=["prefill"], stacking={"channel": "last"}),
                             preset("all", phases=["prefill"], prefill_positions="all", stacking={"channel": "all"})])
        s.inject("last")
        s.inject("all")
        self.assertEqual(s.coefficients("prefill", prefill_position="last")["coefficients"]["joy_orthogonal"], 2)
        self.assertEqual(s.coefficients("prefill", prefill_position="other")["coefficients"]["joy_orthogonal"], 1)
        s.record_exposure("prefill", positions=9, prefill_position="other")
        s.record_exposure("prefill", positions=1, prefill_position="last")
        self.assertEqual(s.exposure["prefill"]["joy_orthogonal"], 11)
        self.assertEqual(s.generated_tokens, 0)

    def test_failed_scheduled_boundary_is_atomic(self):
        s = EffectScheduler([preset(stacking={"policy": "capped_additive"}, decay={"shape": "constant"})])
        s.inject("effect")
        s.schedule("effect", clock="tokens", at=1)
        before = s.snapshot()
        with patch("lab.effects.MAX_PULSES", 1), self.assertRaises(ValueError):
            s.advance_tokens(10)
        self.assertEqual(s.snapshot(), before)

    def test_aggregate_gain_cap_across_distinct_channels(self):
        s = EffectScheduler([preset("one", gains={"pain": -3}, stacking={"channel": "one"}),
                             preset("two", gains={"pain": -3}, stacking={"channel": "two"})])
        s.inject("one")
        s.inject("two")
        self.assertEqual(coefficient(s, "pain"), -4)

    def test_pending_and_live_pulses_roundtrip_exactly_and_pause_is_inert(self):
        s = EffectScheduler([preset(stacking={"policy": "capped_additive"})])
        s.inject("effect", metadata={"source": "run1"})
        s.record_exposure()
        s.advance_tokens(13)
        s.schedule("effect", clock="decisions", at=4)
        state = json.loads(json.dumps(s.snapshot()))
        t = EffectScheduler.restore(state)
        self.assertEqual(t.snapshot(), s.snapshot())
        t.snapshot()["presets"][0]["gains"]["joy"] = 3
        self.assertEqual(coefficient(t), coefficient(s))
        for scheduler in (s, t):
            scheduler.advance_tokens(7)
            scheduler.complete_decision(4)
            scheduler.advance_tokens(6)
        self.assertEqual(t.snapshot(), s.snapshot())

    def test_malformed_or_inconsistent_restore_fails(self):
        s = EffectScheduler([preset()])
        s.inject("effect")
        state = s.snapshot()
        bad = []
        for field, value in (("schema_version", 2), ("enabled", False), ("next_pulse_id", 1), ("sequence", 0), ("presets_hash", "bad")):
            candidate = copy.deepcopy(state)
            candidate[field] = value
            bad.append(candidate)
        candidate = copy.deepcopy(state)
        candidate["pulses"][0]["started_at"] = 100
        bad.append(candidate)
        candidate = copy.deepcopy(state)
        candidate["generated_tokens"] = 1000
        bad.append(candidate)
        candidate = copy.deepcopy(state)
        candidate["pulses"].append(copy.deepcopy(candidate["pulses"][0]))
        bad.append(candidate)
        candidate = copy.deepcopy(state)
        candidate["exposure"]["output"]["positions"] = -1
        bad.append(candidate)
        for candidate in bad:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                EffectScheduler.restore(candidate)

    def test_bad_injection_does_not_mutate_state(self):
        s = EffectScheduler([preset()])
        before = s.snapshot()
        for args in (("missing", {}), ("effect", {"actor": "unknown"}), ("effect", {"active": 1}), ("effect", {"metadata": {"nan": float("nan")}})):
            with self.assertRaises(ValueError):
                s.inject(args[0], **args[1])
            self.assertEqual(s.snapshot(), before)


class JointProjectionTests(unittest.TestCase):
    def test_full_joint_suppression_and_permutation_invariance(self):
        x = [2, 3, 7]
        directions = [[1, 0, 0], [1, 1, 0]]
        a = joint_attenuation(x, directions, [1, 1])
        for actual, expected in zip(a, [0, 0, 7]):
            self.assertAlmostEqual(actual, expected)
        a = joint_attenuation(x, directions, [.25, .75])
        b = joint_attenuation(x, directions[::-1], [.75, .25])
        for left, right in zip(a, b):
            self.assertAlmostEqual(left, right)

    def test_rank_deficiency_and_bad_fractions_fail(self):
        for directions, fractions in (([[1, 0], [2, 0]], [1, 1]), ([[0, 0]], [.5]), ([[1, 0]], [2])):
            with self.assertRaises(ValueError):
                joint_attenuation([1, 2], directions, fractions)
        self.assertEqual(joint_attenuation([1, 2], [], []), [1, 2])


if __name__ == "__main__":
    unittest.main()
