"""Controller boundary/visibility/checkpoint tests; no model loading."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.controller_v2 import RecipeV2Controller
from lab.recipes_v2 import default_recipe, ordered_auxiliary_tools, resolve_recipe


def controller(**overrides):
    return RecipeV2Controller({"recipe_version": 2, **overrides})


def coefficient(c, key="joy_orthogonal", phase="output"):
    return c.snapshot()["phase_coefficients"][phase][key]


class ControllerTests(unittest.TestCase):
    def test_transition_choice_uses_phase_seen_during_generation(self):
        c = controller(condition="joy_to_pain", conditions=["joy_to_pain"], phase_actions=[1, 2])
        self.assertIsNone(c.on_action(0))
        c.complete_decision(1)
        event = c.press()
        c.record_action("aux_operation")
        self.assertEqual(event["requested_preset_id"], "joy")
        self.assertEqual(c.phase, "joy")
        self.assertEqual(c.mapping_decision_count, 0)
        self.assertEqual(c.actions, 1)
        transition = c.on_action(1)
        self.assertEqual(transition["before"], "joy")
        self.assertEqual(c.phase, "pain")
        self.assertEqual(coefficient(c), 0)
        self.assertIsNone(c.on_action(1))
        c.complete_decision(2)
        event = c.press()
        c.record_action("aux_operation")
        self.assertEqual(event["requested_preset_id"], "pain")
        self.assertEqual(coefficient(c, "pain"), 1)
        self.assertEqual(c.phase_counts["0"]["aux_calls"], 1)
        self.assertEqual(c.phase_counts["1"]["aux_calls"], 1)

    def test_transition_decay_retains_old_pulse_without_rewriting_its_preset(self):
        c = controller(condition="joy_to_pain", conditions=["joy_to_pain"], phase_actions=[1, 2], transition_policy="decay")
        c.complete_decision(1)
        c.press()
        c.record_action("aux_operation")
        c.on_action(1)
        self.assertEqual(c.phase, "pain")
        self.assertEqual(coefficient(c), .75)
        self.assertEqual(c.scheduler.pulses[0]["preset_id"], "joy")

    def test_decision_clock_ages_before_new_call_and_invalid_counts_once(self):
        recipe = default_recipe()
        for preset in recipe["effect_presets"]:
            preset["decay"] = {"clock": "decisions", "shape": "linear", "half_life": 1, "cutoff": 2}
        c = RecipeV2Controller(recipe)
        c.complete_decision(1)
        c.press()
        c.record_action("aux_operation")
        self.assertEqual(coefficient(c), .75)
        c.on_action(1)
        c.advance(300)
        self.assertEqual(coefficient(c), .75)
        c.complete_decision(2)
        c.record_action(valid=False)
        self.assertEqual(coefficient(c), .375)
        self.assertEqual(c.phase_counts["0"]["invalid"], 1)
        with self.assertRaises(ValueError):
            c.record_action(valid=False)

    def test_sham_and_disabled_are_distinct_and_sham_does_not_clear_pulse(self):
        c = controller(auxiliary_tools=[{"id": "active", "name": "active", "preset_id": "opium"},
                                       {"id": "sham", "name": "sham", "preset_id": None}])
        c.press(tool="active")
        c.advance(64)
        before = coefficient(c)
        sham = c.press(tool="sham")
        self.assertEqual(sham["delivery_status"], "sham")
        self.assertEqual(coefficient(c), before)
        c.set_controls(enabled=False)
        disabled = c.press(tool="active")
        self.assertEqual(disabled["delivery_status"], "disabled")
        self.assertEqual(disabled["requested_preset_id"], "opium")
        c.set_controls(enabled=True)
        self.assertEqual(coefficient(c), 0)
        self.assertEqual(c.counts["sham"], 1)
        self.assertEqual(c.counts["disabled"], 1)
        RecipeV2Controller.restore(c.snapshot())

    def test_actor_attribution_and_acknowledgment_are_separate_from_hidden_mapping(self):
        c = controller()
        for actor in ("model", "human", "demonstration", "schedule"):
            event = c.press(actor=actor)
            self.assertEqual(event["voluntary"], actor == "model")
            self.assertEqual(event["acknowledgment"], c.acknowledgment("aux_operation"))
            self.assertEqual(c.counts[actor], 1)
        self.assertTrue(c.exploratory)
        self.assertEqual(c.counts["total"], 4)
        self.assertEqual(c.injection_index, 4)

    def test_held_challenge_excluded_from_pulse_gains_and_attenuation_composes(self):
        c = controller()
        c.set_controls(pain=-2, joy=1, suppression=.5, joy_suppression=.25, joy_direction="raw")
        c.press()
        c.advance(128)
        snapshot = c.snapshot()
        values = snapshot["phase_coefficients"]["output"]
        baseline = snapshot["baseline_by_phase"]["output"]
        self.assertEqual(baseline, {"pain": -2., "joy_raw": 1., "joy_orthogonal": 0.})
        self.assertEqual(values["pain"], 0)
        self.assertEqual(values["joy_raw"], 0)
        self.assertEqual(values["joy_orthogonal"], .375)
        self.assertEqual(values["pain_attenuation"], .75)
        self.assertEqual(values["joy_raw_attenuation"], .25)
        self.assertTrue(snapshot["scalar_fields_are_display_only"])

    def test_baseline_phase_scope_and_prefill_are_explicit(self):
        c = controller()
        c.set_controls(pain=2, suppression=1, phase_scope="reasoning")
        state = c.snapshot()
        self.assertEqual(state["baseline_by_phase"]["reasoning"]["pain"], 2)
        self.assertEqual(state["baseline_by_phase"]["output"]["pain"], 0)
        self.assertEqual(state["phase_coefficients"]["reasoning"]["pain_attenuation"], 1)
        self.assertEqual(state["phase_coefficients"]["prefill_last"]["pain_attenuation"], 0)
        self.assertEqual(state["baseline_policy"], "generated-phases-only")
        for phase in ("prefill_last", "prefill_other"):
            self.assertEqual(sum(state["baseline_by_phase"][phase].values()), 0)

    def test_prefill_last_and_all_presets_produce_distinct_coefficients(self):
        recipe = default_recipe()
        recipe["effect_presets"][0]["phases"] = ["prefill"]
        c = RecipeV2Controller(recipe)
        c.press()
        self.assertEqual(coefficient(c, phase="prefill_last"), .75)
        self.assertEqual(coefficient(c, phase="prefill_other"), 0)
        self.assertEqual(coefficient(c), 0)

    def test_reset_preserves_clocks_mapping_counts_and_gate(self):
        c = controller()
        c.press()
        c.set_controls(pain=2, enabled=False)
        c.advance(10)
        c.complete_decision(1)
        c.record_action(valid=False)
        counts = dict(c.counts)
        c.reset()
        self.assertEqual(c.generated_tokens, 10)
        self.assertEqual(c.actions, 1)
        self.assertEqual(c.counts, counts)
        self.assertFalse(c.enabled)
        self.assertEqual(c.baseline["pain"], 0)
        self.assertEqual(c.control_revision, 2)
        self.assertEqual(coefficient(c), 0)

    def test_invalid_arguments_or_control_values_are_atomic(self):
        c = controller(auxiliary_tools=[{"id": "custom", "name": "custom", "parameters": {
            "type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}], demonstration="none")
        before = c.snapshot()
        with self.assertRaises(ValueError):
            c.press(tool="custom", arguments={"n": True})
        self.assertEqual(c.snapshot(), before)
        with self.assertRaises(ValueError):
            c.set_controls(pain=2, suppression=1.5)
        self.assertEqual(c.snapshot(), before)
        with self.assertRaises(ValueError):
            c.record_action("custom")
        self.assertEqual(c.snapshot(), before)

    def test_hidden_tool_is_operator_only(self):
        c = controller(auxiliary_tools=[{"id": "hidden", "name": "hidden", "visible": False}], demonstration="none")
        with self.assertRaises(ValueError):
            c.press(actor="model", tool="hidden")
        with self.assertRaises(ValueError):
            c.press(actor="demonstration", tool="hidden")
        event = c.press(actor="human", tool="hidden")
        self.assertTrue(event["delivered"])
        c.complete_decision(1)
        with self.assertRaises(ValueError):
            c.record_action("hidden")

    def test_resume_retains_old_boundary_mapping_and_future_random_outcomes(self):
        c = controller(condition="probabilistic", conditions=["probabilistic"])
        c.on_action(0)
        c.complete_decision(1)
        c.press()
        c.record_action("aux_operation")
        c.advance(17)
        state = json.loads(json.dumps(c.snapshot()))
        restored = RecipeV2Controller.restore(state)
        self.assertEqual(restored.snapshot(), state)
        events = []
        for target in (c, restored):
            target.on_action(1)
            target.complete_decision(2)
            events.append(target.press())
            target.record_action("aux_operation")
            target.advance(5)
        self.assertEqual(events[0], events[1])
        self.assertEqual(c.snapshot(), restored.snapshot())

    def test_repeated_phase_labels_are_separate_opportunity_buckets(self):
        mappings = {"aux_operation": [{"preset_id": "opium", "probability": 1}]}
        c = controller(mapping_schedule=[{"after_decisions": 0, "label": "same", "mappings": mappings},
                                         {"after_decisions": 1, "label": "same", "mappings": mappings}])
        c.complete_decision(1)
        c.record_action("read_order")
        c.on_action(1)
        c.complete_decision(2)
        c.record_action(tool=None)
        self.assertEqual(c.phase_counts["0"]["work_calls"], 1)
        self.assertEqual(c.phase_counts["1"]["text_decisions"], 1)
        RecipeV2Controller.restore(c.snapshot())

    def test_corrupt_derived_fields_mapping_counts_and_versions_fail(self):
        c = controller()
        c.complete_decision(1)
        c.press()
        c.record_action("aux_operation")
        baseline = c.snapshot()
        cases = []
        for key, value in (("schema_version", 1), ("controller_version", True), ("recipe_hash", "wrong"),
                           ("phase_index", 100), ("mapping_decision_count", 2), ("injection_index", 0),
                           ("actions", 2), ("pain", 4), ("event_sequence", 0)):
            candidate = copy.deepcopy(baseline)
            candidate[key] = value
            cases.append(candidate)
        candidate = copy.deepcopy(baseline)
        candidate["phase_coefficients"]["output"]["joy_orthogonal"] = 4
        cases.append(candidate)
        candidate = copy.deepcopy(baseline)
        candidate["phase_counts"]["0"]["opportunities"] = 2
        cases.append(candidate)
        candidate = copy.deepcopy(baseline)
        candidate["counts"]["model"] = 0
        cases.append(candidate)
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                RecipeV2Controller.restore(candidate)

    def test_zero_effect_preset_preserves_honest_delivery_status(self):
        c = controller(effect_presets=[{"id": "opium", "gains": {}}])
        event = c.press()
        self.assertEqual(event["delivery_status"], "zero_effect")
        self.assertEqual(c.counts["zero_effect"], 1)
        self.assertEqual(c.counts["delivered"], 0)
        RecipeV2Controller.restore(c.snapshot())

    def test_order_helper_is_blind_independent_and_hides_operator_tools(self):
        config = resolve_recipe({"recipe_version": 2, "two_buttons": True})
        active = ordered_auxiliary_tools(config)
        sham = ordered_auxiliary_tools(resolve_recipe({**config, "condition": "sham"}))
        self.assertEqual(active, sham)
        self.assertEqual(active, ordered_auxiliary_tools(config))
        explicit = resolve_recipe({**config, "counterbalance": False})
        self.assertEqual([t["function"]["name"] for t in ordered_auxiliary_tools(explicit)], ["aux_operation", "aux_alternative"])


if __name__ == "__main__":
    unittest.main()
