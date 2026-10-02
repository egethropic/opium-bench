"""Versioned resolution, blind arms and independently seeded mapping draws."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.recipes_v2 import (default_recipe, mapping_at, model_cost_notice, recipe_hash,
                           resolve_recipe, resolve_tool_outcome)
from lab.tool_definitions import assert_blind_pair


def recipe(**kwargs):
    return resolve_recipe({"recipe_version": 2, **kwargs})


def certain(identifier):
    return [{"preset_id": identifier, "probability": 1}]


class RecipeTests(unittest.TestCase):
    def test_version_is_explicit_and_future_unknown_fields_fail(self):
        for value in ({}, {"recipe_version": 1}, {"recipe_version": True}, {"recipe_version": 3},
                      {"recipe_version": 2, "half_life_tokens": 32}, {"recipe_version": 2, "script": "anything"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_recipe(value)

    def test_defaults_canonical_detached_and_preserve_common_fields(self):
        a = default_recipe()
        b = resolve_recipe(json.loads(json.dumps(a)))
        self.assertEqual(a, b)
        self.assertEqual(recipe_hash(a), recipe_hash(b))
        self.assertEqual((a["task_family"], a["task_count"], a["action_budget"]), ("orders", 6, 32))
        b["effect_presets"][0]["gains"]["joy"] = 3
        self.assertNotEqual(a, b)
        self.assertEqual(recipe(id="My Study – α 2")["id"], "My Study – α 2")

    def test_all_legacy_named_conditions_resolve_v2_schedules(self):
        names = ("active", "sham", "joy", "pain", "random", "suppression", "joy_to_pain",
                 "joy_to_sham_to_pain", "probabilistic", "reversal")
        for name in names:
            with self.subTest(condition=name):
                value = recipe(condition=name, conditions=[name], two_buttons=name == "reversal")
                self.assertEqual(resolve_recipe(value), value)
                self.assertEqual(value["mapping_schedule"][0]["after_decisions"], 0)

    def test_changed_arm_regenerates_default_mapping_without_leaks(self):
        active = default_recipe()
        sham = resolve_recipe({**active, "condition": "sham"})
        self.assertTrue(assert_blind_pair(active["auxiliary_tools"], sham["auxiliary_tools"]))
        self.assertEqual(model_cost_notice(active), model_cost_notice(sham))
        self.assertEqual(resolve_tool_outcome(sham, "aux_operation", 0, 0)["preset_id"], None)
        self.assertEqual(resolve_tool_outcome(active, "aux_operation", 0, 0)["preset_id"], "opium")

    def test_exact_transition_boundaries(self):
        value = recipe(condition="joy_to_sham_to_pain", conditions=["joy_to_sham_to_pain"], phase_actions=[2, 5])
        for completed, expected in ((0, "joy"), (1, "joy"), (2, "sham"), (4, "sham"), (5, "pain"), (9, "pain")):
            with self.subTest(completed=completed):
                self.assertEqual(mapping_at(value, completed)["label"], expected)

    def test_reversal_changes_assignment_only_and_is_reproducible(self):
        value = recipe(condition="reversal", conditions=["reversal"], two_buttons=True, phase_actions=[2, 5])
        before, after = mapping_at(value, 1), mapping_at(value, 2)
        before_active = [name for name, outcomes in before["mappings"].items() if outcomes[0]["preset_id"]]
        after_active = [name for name, outcomes in after["mappings"].items() if outcomes[0]["preset_id"]]
        self.assertEqual(len(before_active), 1)
        self.assertEqual(len(after_active), 1)
        self.assertNotEqual(before_active, after_active)
        self.assertEqual(value, resolve_recipe(value))

    def test_custom_mapping_and_condition_label_are_authoritative(self):
        custom = recipe(id="Different study", condition="swap negative", conditions=["swap negative"],
                        auxiliary_tools=[{"name": "button_x", "id": "x", "preset_id": "joy"}],
                        mapping_schedule=[{"after_decisions": 0, "label": "start", "mappings": {"button_x": certain("joy")}},
                                          {"after_decisions": 3, "label": "swapped", "mappings": {"button_x": certain("pain")}}])
        self.assertEqual(custom["mapping_policy"], "explicit")
        self.assertEqual(resolve_tool_outcome(custom, "button_x", 3, 7)["preset_id"], "pain")
        self.assertEqual(custom, resolve_recipe(custom))

    def test_schedule_requires_complete_known_tools_and_probabilities(self):
        bad = [[], [{"after_decisions": 1, "label": "x", "mappings": {"aux_operation": certain("joy")}}],
               [{"after_decisions": 0, "label": "x", "mappings": {}}],
               [{"after_decisions": 0, "label": "x", "mappings": {"aux_operation": certain("missing")}}],
               [{"after_decisions": 0, "label": "x", "mappings": {"aux_operation": [{"preset_id": "joy", "probability": .4}]}}],
               [{"after_decisions": 0, "label": "x", "mappings": {"aux_operation": [{"preset_id": "joy", "probability": .5}, {"preset_id": "joy", "probability": .5}]}}]]
        for schedule in bad:
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                recipe(mapping_schedule=schedule)

    def test_seed_streams_and_counter_draws_are_independent(self):
        value = recipe(condition="probabilistic", conditions=["probabilistic"])
        self.assertEqual(len(set(value["rng_seeds"].values())), 5)
        draws = [resolve_tool_outcome(value, "aux_operation", 0, i) for i in range(20)]
        changed = resolve_recipe({**value, "rng_policy": "explicit", "rng_seeds": {**value["rng_seeds"], "generation": 18, "tasks": 19}})
        self.assertEqual(draws, [resolve_tool_outcome(changed, "aux_operation", 0, i) for i in range(20)])
        changed = resolve_recipe({**value, "rng_policy": "explicit", "rng_seeds": {**value["rng_seeds"], "outcomes": 20}})
        self.assertNotEqual([d["draw"] for d in draws], [resolve_tool_outcome(changed, "aux_operation", 0, i)["draw"] for i in range(20)])
        self.assertEqual({d["preset_id"] for d in draws}, {"joy", "pain"})

    def test_resolved_batch_seed_change_regenerates_only_derived_streams(self):
        original = default_recipe()
        changed = resolve_recipe({**original, "seed": original["seed"] + 1})
        self.assertNotEqual(original["rng_seeds"], changed["rng_seeds"])
        explicit = recipe(rng_seeds={"generation": 123})
        self.assertEqual(explicit["rng_policy"], "explicit")
        changed = resolve_recipe({**explicit, "seed": explicit["seed"] + 1})
        self.assertEqual(explicit["rng_seeds"], changed["rng_seeds"])

    def test_probability_extremes_are_deterministic(self):
        for probability, outcome in ((0, "joy"), (1, "pain")):
            value = recipe(condition="probabilistic", conditions=["probabilistic"], probability_pain=probability)
            for index in range(5):
                self.assertEqual(resolve_tool_outcome(value, "aux_operation", 0, index)["preset_id"], outcome)

    def test_custom_arguments_must_have_valid_demonstrations(self):
        tools = [{"id": "custom", "name": "custom", "parameters": {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 3}}, "required": ["n"]}}]
        with self.assertRaises(ValueError):
            recipe(auxiliary_tools=tools)
        value = recipe(auxiliary_tools=tools, demonstration_calls=[{"tool": "custom", "arguments": {"n": 2}}])
        self.assertEqual(value["demonstration_calls"][0]["arguments"], {"n": 2})
        recipe(auxiliary_tools=tools, demonstration="none")
        with self.assertRaises(ValueError):
            recipe(auxiliary_tools=tools, demonstration_calls=[{"tool": "custom", "arguments": {"n": 9}}])

    def test_costs_resolve_all_work_tools_and_invalid_fields_fail(self):
        value = recipe(base_decision_cost=2, task_tool_costs={"submit_answer": 3})
        self.assertEqual(value["task_tool_costs"], {"read_order": 0, "calculate_total": 0, "submit_answer": 3})
        self.assertIn("submit_answer: 3 extra", model_cost_notice(value))
        for kwargs in ({"task_tool_costs": {"other": 1}}, {"task_tool_costs": {"read_order": True}},
                       {"base_decision_cost": 33}, {"phase_actions": [5, 1]}, {"rng_seeds": {"unknown": 1}},
                       {"thinking": 1}, {"seed": -1}, {"temperature": float("nan")}, {"id": "../escape"},
                       {"demonstration": "at_decisions", "demonstration_decisions": []}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                recipe(**kwargs)

    def test_no_tools_conversation_recipe(self):
        value = recipe(task_family="conversation", auxiliary_tools=[], demonstration="none")
        self.assertEqual(value["mapping_schedule"][0]["mappings"], {})
        self.assertEqual(value["task_tool_costs"], {})
        with self.assertRaises(ValueError):
            resolve_tool_outcome(value, "aux_operation", 0, 0)


if __name__ == "__main__":
    unittest.main()
