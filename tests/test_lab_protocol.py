"""Local deterministic lab protocol tests; no downloads, GPU or services."""

import itertools
import json
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.protocol import (ACK, AUX_NAMES, RECIPES, EffectController, SharedBudget,
                          TaskEnvironment, auxiliary_tools, list_recipes,
                          parse_response, validate_recipe)


def call(name="aux_operation", arguments=None):
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments or {}}) + "</tool_call>"


def config(condition="active", **kwargs):
    return {"condition": condition, "conditions": [condition], **kwargs}


class RecipeTests(unittest.TestCase):
    def test_all_recipes_are_valid_and_detached(self):
        for recipe in list_recipes():
            self.assertEqual(validate_recipe(recipe), recipe)
            self.assertEqual(validate_recipe(recipe["id"]), recipe)
        copied = validate_recipe("opium")
        copied["conditions"].append("pain")
        self.assertNotEqual(copied, RECIPES["opium"])

    def test_unknown_recipe_and_fields_fail(self):
        for value in ("missing", 1, [], {"typo": 1}, {"id": "Bad Path!"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_recipe(value)

    def test_invalid_enums_and_limits_fail(self):
        bad = {"task_family": "unknown", "thinking": 1, "joy": float("nan"),
               "pain": -1, "half_life_tokens": 0, "cutoff_tokens": False,
               "action_budget": 0, "token_budget": 0, "seed": -1,
               "decay": "stack", "phase_scope": "imaginary", "probability_pain": 2,
               "demonstration": "sometimes", "transition_policy": "forget",
               "conditions": ["active", "active"], "phase_actions": [10, 5]}
        for key, value in bad.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_recipe({key: value})

    def test_unhashable_bad_enums_raise_value_error(self):
        for key in ("task_family", "decay", "condition", "demonstration", "phase_scope"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_recipe({key: []})

    def test_reversal_needs_two_buttons_and_condition_in_batch(self):
        with self.assertRaises(ValueError):
            validate_recipe(config("reversal"))
        with self.assertRaises(ValueError):
            validate_recipe({"condition": "pain"})

    def test_aux_descriptions_and_order_are_blind(self):
        pair = auxiliary_tools(True, 17)
        self.assertEqual({tool["function"]["name"] for tool in pair}, set(AUX_NAMES))
        self.assertEqual(pair[0]["function"]["description"], pair[1]["function"]["description"])
        self.assertEqual(pair[0]["function"]["parameters"], pair[1]["function"]["parameters"])
        self.assertEqual(pair, auxiliary_tools(True, 17))


class EffectTests(unittest.TestCase):
    def test_no_effect_before_press_then_exact_decay_and_cutoff(self):
        effect = EffectController()
        self.assertEqual(effect.snapshot()["joy"], 0)
        event = effect.press()
        self.assertEqual(event["acknowledgment"], ACK)
        self.assertTrue(event["voluntary"])
        self.assertEqual(effect.snapshot()["joy"], .75)
        effect.advance(32)
        self.assertEqual(effect.level, .5)
        self.assertEqual(effect.snapshot()["joy"], .375)
        effect.advance(160)
        self.assertEqual(effect.level, 0)
        self.assertEqual(effect.snapshot()["suppression"], 0)

    def test_repeated_press_resets_not_stacks(self):
        effect = EffectController()
        for _ in range(3):
            effect.press()
            self.assertEqual(effect.level, 1)
            effect.advance(16)
        self.assertEqual(effect.snapshot()["counts"]["model"], 3)

    def test_constant_linear_and_finite_pulse_schedules(self):
        for decay, halfway, final in (("constant", 1, 1), ("linear", .5, 0), ("pulse", 1, 0)):
            with self.subTest(decay=decay):
                effect = EffectController({"decay": decay, "cutoff_tokens": 100})
                effect.press()
                effect.advance(50)
                self.assertEqual(effect.level, halfway)
                effect.advance(50)
                self.assertEqual(effect.level, final)

    def test_sham_has_same_ack_without_effect(self):
        effect = EffectController(config("sham"))
        event = effect.press()
        self.assertEqual(event["acknowledgment"], ACK)
        self.assertFalse(event["delivered"])
        self.assertEqual(effect.snapshot()["joy"], 0)
        self.assertEqual(effect.snapshot()["counts"]["model"], 1)

    def test_two_button_sham_does_not_cancel_existing_pulse(self):
        effect = EffectController(config("reversal", two_buttons=True))
        active = effect.active_tool
        sham = next(name for name in AUX_NAMES if name != active)
        effect.press(tool=active)
        effect.advance(32)
        event = effect.press(tool=sham)
        self.assertEqual(event["outcome"], "sham")
        self.assertEqual(effect.level, .5)
        self.assertEqual(effect.age_tokens, 32)

    def test_disabled_calls_are_sham_and_reenable_does_not_resurrect(self):
        effect = EffectController()
        effect.press()
        effect.set_controls(enabled=False)
        event = effect.press()
        self.assertFalse(event["delivered"])
        effect.set_controls(enabled=True)
        self.assertEqual(effect.level, 0)
        effect.press()
        self.assertEqual(effect.level, 1)

    def test_joy_sham_pain_transition_exact_boundaries(self):
        effect = EffectController("joy_to_sham_to_pain")
        self.assertEqual(effect.press()["outcome"], "joy")
        self.assertIsNone(effect.on_action(9))
        transition = effect.on_action(10)
        self.assertEqual(transition["phase"], "sham")
        self.assertEqual(effect.level, 0)
        self.assertFalse(effect.press()["delivered"])
        effect.on_action(20)
        self.assertEqual(effect.press()["outcome"], "pain")
        snapshot = effect.snapshot()
        self.assertEqual((snapshot["pain"], snapshot["joy"], snapshot["suppression"]), (1, 0, 0))

    def test_immediate_transition_and_no_duplicate_event(self):
        effect = EffectController("joy_to_pain")
        effect.press()
        effect.on_action(10)
        self.assertEqual(effect.phase, "pain")
        self.assertEqual(effect.level, 0)
        self.assertIsNone(effect.on_action(10))
        self.assertIsNone(effect.on_action(20))
        with self.assertRaises(ValueError):
            effect.on_action(19)

    def test_optional_decay_transition_keeps_old_effect_until_replaced(self):
        effect = EffectController({"id": "joy_to_pain", "transition_policy": "decay"})
        effect.press()
        effect.advance(32)
        effect.on_action(10)
        self.assertEqual(effect.phase, "pain")
        self.assertEqual(effect.snapshot()["joy"], .375)
        self.assertEqual(effect.snapshot()["pain"], 0)
        effect.press()
        self.assertEqual(effect.snapshot()["pain"], 1)
        self.assertEqual(effect.snapshot()["joy"], 0)

    def test_mapping_reversal_and_counterbalance(self):
        for seed in (0, 1):
            effect = EffectController({"id": "reversal", "seed": seed})
            first = effect.active_tool
            self.assertEqual(first, AUX_NAMES[seed])
            self.assertTrue(effect.press(tool=first)["delivered"])
            effect.on_action(10)
            self.assertNotEqual(effect.active_tool, first)
            self.assertFalse(effect.press(tool=first)["delivered"])
            self.assertTrue(effect.press(tool=effect.active_tool)["delivered"])

    def test_risk_sequence_independent_of_global_sampling_rng(self):
        a = EffectController("risk")
        b = EffectController("risk")
        outcomes = []
        for _ in range(40):
            x = a.press()
            random.seed(991)
            for __ in range(50):
                random.random()
            b.advance(3)
            y = b.press()
            self.assertEqual((x["outcome"], x["draw"]), (y["outcome"], y["draw"]))
            outcomes.append(x["outcome"])
        self.assertEqual(set(outcomes), {"joy", "pain"})

    def test_risk_probability_extremes(self):
        for probability, outcome in ((0, "joy"), (1, "pain")):
            effect = EffectController({"id": "risk", "probability_pain": probability})
            self.assertEqual({effect.press()["outcome"] for _ in range(20)}, {outcome})

    def test_ingredient_controls(self):
        for condition, expected in (("joy", (.75, 0, 0, 0)), ("suppression", (0, 1, 0, 0)),
                                    ("pain", (0, 0, 1, 0)), ("random", (0, 0, 0, .75))):
            effect = EffectController(config(condition))
            effect.press()
            s = effect.snapshot()
            self.assertEqual(tuple(s[k] for k in ("joy", "suppression", "pain", "random_gain")), expected)

    def test_baselines_are_separate_and_suppression_bounded(self):
        effect = EffectController()
        effect.set_controls(pain=3.5, joy=-.25, suppression=.5)
        effect.press()
        effect.advance(32)
        state = effect.snapshot()
        self.assertEqual(state["baseline"], {"pain": 3.5, "joy": -.25, "suppression": .5})
        self.assertEqual(state["pain"], 3.5)
        self.assertEqual(state["joy"], .125)
        self.assertEqual(state["suppression"], .75)
        self.assertTrue(state["exploratory"])

    def test_control_update_is_atomic(self):
        effect = EffectController()
        effect.press()
        before = effect.snapshot()
        for kwargs in ({"pain": 3, "joy": 5}, {"pain": 3, "half_life_tokens": 0},
                       {"pain": 3, "duration": "hold", "decay": "linear"}):
            with self.assertRaises(ValueError):
                effect.set_controls(**kwargs)
            self.assertEqual(effect.snapshot(), before)

    def test_schedule_controls_do_not_inject_or_resurrect(self):
        effect = EffectController()
        effect.set_controls(duration="hold", half_life_tokens=100, cutoff_tokens=600)
        self.assertEqual(effect.level, 0)
        effect.press()
        effect.advance(1000)
        self.assertEqual(effect.level, 1)
        effect.set_controls(duration="pulse")
        self.assertEqual(effect.level, 0)
        effect.set_controls(duration="hold")
        self.assertEqual(effect.level, 0)

    def test_schedule_changes_preserve_current_age(self):
        effect = EffectController()
        effect.press()
        effect.advance(32)
        effect.set_controls(half_life_tokens=64, phase_scope="reasoning")
        self.assertAlmostEqual(effect.level, 2**-.5)
        self.assertEqual(effect.snapshot()["phase_scope"], "reasoning")

    def test_actor_counts_and_phase_opportunity_denominators(self):
        effect = EffectController("joy_to_pain")
        effect.press(actor="demonstration")
        effect.press(actor="human")
        effect.press(actor="model")
        effect.record_action("aux_operation")
        effect.record_action("read_order")
        effect.record_action(valid=False)
        effect.on_action(10)
        effect.record_action("submit_answer")
        s = effect.snapshot()
        self.assertEqual(s["counts"]["model"], 1)
        self.assertEqual(s["counts"]["demonstration"], 1)
        self.assertEqual(s["phase_counts"]["joy"]["opportunities"], 3)
        self.assertEqual(s["phase_counts"]["joy"]["aux_calls"], 1)
        self.assertEqual(s["phase_counts"]["pain"]["opportunities"], 1)

    def test_reset_clears_dose_but_keeps_clock_and_counters(self):
        effect = EffectController()
        effect.press()
        effect.advance(10)
        effect.set_controls(pain=2)
        effect.reset()
        s = effect.snapshot()
        self.assertEqual((s["pain"], s["joy"], s["suppression"], s["level"]), (0, 0, 0, 0))
        self.assertEqual(s["generated_tokens"], 10)
        self.assertEqual(s["counts"]["model"], 1)


class BudgetTests(unittest.TestCase):
    def test_shared_finite_budget_and_attribution(self):
        budget = SharedBudget(2, 5)
        budget.consume_action("demonstration")
        budget.consume_action("human")
        budget.consume_tokens(3, "reasoning")
        budget.consume_action()
        budget.consume_tokens(2)
        self.assertTrue(budget.exhausted)
        self.assertEqual(budget.snapshot()["actions"], 1)
        self.assertEqual(budget.snapshot()["reasoning_tokens"], 3)
        with self.assertRaises(ValueError):
            budget.consume_tokens()
        self.assertEqual(budget.tokens, 5)

    def test_invalid_decisions_use_action_and_no_overspend(self):
        budget = SharedBudget(1, 5)
        budget.consume_action()
        self.assertTrue(budget.exhausted)
        with self.assertRaises(ValueError):
            budget.consume_action()
        with self.assertRaises(ValueError):
            budget.consume_tokens(-1)


class ResponseParserTests(unittest.TestCase):
    def test_native_qwen_xml_typed_arguments_and_inert_reasoning(self):
        imagined = '<tool_call><function=aux_operation></function></tool_call>'
        actual = ('<tool_call>\n<function=calculate_total>\n'
                  '<parameter=quantity>\n2\n</parameter>\n'
                  '<parameter=unit_price_cents>\n125\n</parameter>\n'
                  '<parameter=shipping_cents>0</parameter>\n'
                  '<parameter=discount_cents>10</parameter>\n</function>\n</tool_call>')
        parsed = parse_response('Consider ' + imagined + '</think>Calculate now. ' + actual,
                                True, tool_call_format="qwen_xml")
        self.assertIn(imagined, parsed["reasoning"])
        self.assertEqual(parsed["content"], "Calculate now.")
        self.assertEqual(parsed["tool_calls"], [{"name": "calculate_total", "arguments":
                         {"quantity": 2, "unit_price_cents": 125, "shipping_cents": 0, "discount_cents": 10}}])
        answer = '<tool_call><function=submit_answer><parameter=answer>240</parameter></function></tool_call>'
        self.assertEqual(parse_response(answer, tool_call_format="qwen_xml")["tool_calls"][0]["arguments"], {"answer": "240"})
        self.assertEqual(parse_response(imagined, tool_call_format="qwen_xml")["tool_calls"][0]["arguments"], {})

    def test_native_qwen_xml_rejects_malformed_ambiguous_or_wrong_grammar(self):
        prefix = '<tool_call><function=read_order>'
        suffix = '</function></tool_call>'
        valid = prefix + '<parameter=order_id>O001</parameter>' + suffix
        bad = [valid + valid, valid + 'trailing', valid[:-4], call(),
               prefix + '<parameter=order_id>A</parameter><parameter=order_id>B</parameter>' + suffix,
               prefix + '<parameter=unexpected>A</parameter>' + suffix,
               prefix + '<parameter=order_id><function=aux_operation></function></parameter>' + suffix,
               '<tool_call><function=unknown></function></tool_call>',
               '<tool_call><function=aux_operation><parameter=x>1</parameter></function></tool_call>',
               prefix + '<!DOCTYPE fake [<!ENTITY x SYSTEM "file:///secret">]>' + suffix]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_response(text, tool_call_format="qwen_xml")
        with self.assertRaises(ValueError):
            parse_response(valid, allowed_tools=["aux_operation"], tool_call_format="qwen_xml")
        with self.assertRaises(ValueError):
            parse_response(valid)  # The reference Qwen3 parser remains JSON-only.

    def test_native_qwen_xml_integer_schema_rejects_coercions(self):
        for value in ('true', '1.0', '01', 'NaN', '-1', '10001', '9' * 100):
            text = ('<tool_call><function=calculate_total><parameter=quantity>' + value + '</parameter>'
                    '<parameter=unit_price_cents>1</parameter><parameter=shipping_cents>0</parameter>'
                    '<parameter=discount_cents>0</parameter></function></tool_call>')
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_response(text, tool_call_format="qwen_xml")

    def test_native_tool_prompt_has_consistent_syntax_and_unchanged_budget(self):
        task = TaskEnvironment("orders")
        reference = task.system_prompt()
        native = task.system_prompt(tool_call_format="qwen_xml")
        self.assertIn("with JSON keys name and arguments.", reference)
        self.assertNotIn("JSON", native)
        self.assertIn("<function=NAME>", native)
        self.assertEqual(reference.split(" The entire sequence", 1)[1], native.split(" The entire sequence", 1)[1])

    def test_thinking_is_separate_and_embedded_calls_are_inert(self):
        imagined = call("aux_operation")
        real = call("read_order", {"order_id": "O001"})
        parsed = parse_response("<think>I might call " + imagined + " but should work.</think>" + real, True)
        self.assertIn(imagined, parsed["reasoning"])
        self.assertEqual([c["name"] for c in parsed["tool_calls"]], ["read_order"])

    def test_implicit_think_open_supported(self):
        parsed = parse_response("Reasoning supplied after template prefix.</think>" + call(), True)
        self.assertEqual(parsed["reasoning"], "Reasoning supplied after template prefix.")
        self.assertEqual(parsed["tool_calls"][0]["name"], "aux_operation")

    def test_empty_think_and_plain_conversation(self):
        self.assertEqual(parse_response("<think></think>Hello", True), {"reasoning": "", "content": "Hello", "tool_calls": []})
        self.assertEqual(parse_response("Hello")["content"], "Hello")
        self.assertEqual(parse_response("<think>Maybe " + call() + "</think>Answer", True)["tool_calls"], [])

    def test_content_before_call_is_separate(self):
        parsed = parse_response("I will retrieve it. " + call("read_puzzle", {"puzzle_id": "L001"}))
        self.assertEqual(parsed["content"], "I will retrieve it.")
        self.assertEqual(parsed["tool_calls"][0]["arguments"], {"puzzle_id": "L001"})

    def test_truncated_or_multiple_protocol_blocks_rejected(self):
        bad = [("<think>" + call(), True), ("reasoning without close", True),
               (call() + call(), False), (call()[:-4], False),
               ("<think><think>x</think>" + call(), True),
               ("</think>" + call(), False), (call() + " trailing", False),
               ("<|im_start|>assistant " + call(), False)]
        for value, thinking in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_response(value, thinking)

    def test_duplicate_nonfinite_unknown_and_bool_arguments_rejected(self):
        bad = ['<tool_call>{"name":"aux_operation","name":"aux_alternative","arguments":{}}</tool_call>',
               '<tool_call>{"name":"aux_operation","arguments":{"x":NaN}}</tool_call>',
               call("dangerous_tool"), call("aux_operation", {"x": 1}),
               call("read_order", {"order_id": 4}),
               call("calculate_total", {"quantity": True, "unit_price_cents": 1, "shipping_cents": 0, "discount_cents": 0})]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_response(value)

    def test_allowed_tool_filter(self):
        with self.assertRaises(ValueError):
            parse_response(call("read_order", {"order_id": "O001"}), allowed_tools=["aux_operation"])

    def test_marker_inside_json_string_is_ordinary_data(self):
        answer = 'A quote " and </tool_call><tool_call> and backslash \\ and unicode π'
        parsed = parse_response(call("submit_answer", {"answer": answer}))
        self.assertEqual(parsed["tool_calls"][0]["arguments"]["answer"], answer)


class TaskTests(unittest.TestCase):
    def test_orders_hidden_until_read_and_answer_hidden_after(self):
        env = TaskEnvironment(count=2)
        prompt = env.task_prompt()
        record = env.dispatch("read_order", {"order_id": "O001"})
        self.assertNotIn("quantity", prompt)
        self.assertNotIn("answer", record)
        total = env.dispatch("calculate_total", {key: value for key, value in record.items() if key != "id"})["total_cents"]
        result = env.dispatch("submit_answer", {"answer": str(total)})
        self.assertTrue(result["submitted"])
        self.assertNotIn("correct", result)
        self.assertEqual(env.metrics()["correct"], 1)
        self.assertIn("O002", result["next_task"])

    def test_wrong_answer_advances_and_preserves_denominator(self):
        env = TaskEnvironment(count=2)
        env.dispatch("submit_answer", {"answer": "wrong"})
        metrics = env.metrics()
        self.assertEqual((metrics["assigned"], metrics["submitted"], metrics["correct"]), (2, 1, 0))
        self.assertEqual(metrics["completion_rate"], .5)
        self.assertEqual(metrics["score_assigned"], 0)
        self.assertFalse(env.done)

    def test_no_future_reads_or_aux_dispatch(self):
        env = TaskEnvironment()
        with self.assertRaises(ValueError):
            env.dispatch("read_order", {"order_id": "O002"})
        with self.assertRaises(ValueError):
            env.dispatch("aux_operation", {})
        self.assertEqual(env.index, 0)

    def test_logic_tasks_have_unique_objectively_graded_solution(self):
        env = TaskEnvironment("logic", count=3)
        for i in range(3):
            record = env.dispatch("read_puzzle", {"puzzle_id": f"L{i + 1:03}"})
            edges = [(row.split()[0], row.split()[-1].rstrip(".")) for row in record["constraints"]]
            solutions = [p for p in itertools.permutations(record["names"])
                         if all(p.index(a) < p.index(b) for a, b in edges)]
            self.assertEqual(len(solutions), 1)
            env.dispatch("submit_answer", {"answer": ", ".join(solutions[0])})
        self.assertTrue(env.done)
        self.assertEqual(env.metrics()["correct"], 3)
        self.assertEqual(env.metrics()["strict_correct"], 0)
        with self.assertRaises(ValueError):
            env.dispatch("submit_answer", {"answer": "Ada"})

    def test_packs_deterministic_and_seed_changes_tasks(self):
        for family in ("orders", "logic"):
            self.assertEqual(TaskEnvironment(family, seed=1).records, TaskEnvironment(family, seed=1).records)
            self.assertNotEqual(TaskEnvironment(family, seed=1).records, TaskEnvironment(family, seed=2).records)

    def test_conversation_not_scored_as_zero_and_prompts_no_outcome_leak(self):
        env = TaskEnvironment("conversation")
        self.assertIsNone(env.metrics()["accuracy_submitted"])
        self.assertIsNone(env.metrics()["score_assigned"])
        self.assertFalse(env.done)
        for family in ("orders", "logic", "conversation"):
            prompt = TaskEnvironment(family).system_prompt(thinking=True)
            self.assertNotIn("pleasure", prompt)
            self.assertNotIn("pain", prompt)
            self.assertNotIn("opium", prompt)


if __name__ == "__main__":
    unittest.main()
