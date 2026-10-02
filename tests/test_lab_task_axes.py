"""Objective difficulty, matched wording, legacy parity and durable task replay."""
from copy import deepcopy
import hashlib
from itertools import combinations, permutations
import json
import unittest

from lab.budgets import WeightedBudget
from lab.checkpoints import branch, capture, restore, validate
from lab.controller_v2 import RecipeV2Controller
from lab.protocol import AUX_NAMES, TaskEnvironment
from lab.recipes_v2 import ordered_auxiliary_tools, resolve_recipe
from lab.session_v2 import build_session, create_environment, system_prompt
from lab.task_axes import (DEFAULT_TASK_CONFIG, DEADLINE_TEXT, TaskAxisEnvironment,
                           create_task_environment, validate_task_config)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def reseal(value):
    value = deepcopy(value)
    value.pop("sha256", None)
    value["sha256"] = digest(value)
    return value


def allocation_options(record):
    # Independent combinations solver, not the implementation's bit-mask loop.
    items = record["candidates"]
    return sorted((sum(x["value_cents"] for x in chosen), tuple(x["name"] for x in chosen))
                  for size in range(len(items) + 1) for chosen in combinations(items, size)
                  if sum(x["size_units"] for x in chosen) <= record["capacity_units"])


def assignment_options(record):
    names = record["names"]
    edges = [line.removesuffix(".").split(" is somewhere before ") for line in record["constraints"]]
    return sorted((sum(record["position_costs"][name][slot] for slot, name in enumerate(order)), order)
                  for order in permutations(names) if all(order.index(a) < order.index(b) for a, b in edges))


class TaskAxesTests(unittest.TestCase):
    def config(self, **kwargs):
        return resolve_recipe(dict(recipe_version=2, demonstration="none", task_count=3, seed=17,
                                   action_budget=20, token_budget=500, **kwargs))

    def test_standard_neutral_is_exact_original_environment(self):
        for family in ("orders", "logic", "conversation"):
            for seed in (0, 17, 2**63-1):
                original = TaskEnvironment(family, 4, seed, True, False)
                for axes in (None, {}, DEFAULT_TASK_CONFIG):
                    current = create_task_environment(family, 4, seed, True, False, task_config=axes)
                    self.assertIs(type(current), TaskEnvironment)
                    self.assertEqual(current.__dict__, original.__dict__)
                    self.assertEqual(current.task_prompt(), original.task_prompt())
                    self.assertEqual(current.system_prompt(), original.system_prompt())
        cfg = self.config()
        original = TaskEnvironment("orders", 3, cfg["rng_seeds"]["tasks"], cfg["two_buttons"], cfg["counterbalance"])
        original.tools = ordered_auxiliary_tools(cfg) + [t for t in original.tools if t["function"]["name"] not in AUX_NAMES]
        self.assertEqual(create_environment(cfg).__dict__, original.__dict__)
        for field in ("messages", "tools", "task_prompt", "config"):
            self.assertEqual(build_session(cfg, include_task=True)[field],
                             build_session(cfg, include_task=True, task_config=DEFAULT_TASK_CONFIG)[field])
        self.assertNotIn("Deadline", system_prompt(cfg))
        self.assertNotIn("task_config", cfg)

    def test_framing_changes_words_only_not_tasks_tools_answers_or_budgets(self):
        for family in ("orders", "logic"):
            cfg = self.config(task_family=family)
            for difficulty in ("standard", "hard"):
                neutral = build_session(cfg, include_task=True, task_config={"difficulty": difficulty})
                deadline = build_session(cfg, include_task=True, task_config={"difficulty": difficulty, "framing":"deadline"})
                a = create_environment(cfg, task_config=neutral["task_config"])
                b = create_environment(cfg, task_config=deadline["task_config"])
                self.assertEqual(a.records, b.records)
                self.assertEqual(a.tools, b.tools)
                self.assertEqual(neutral["config"], deadline["config"])
                self.assertEqual(deadline["messages"][0]["content"], neutral["messages"][0]["content"] + " " + DEADLINE_TEXT)
                self.assertEqual(b.task_prompt(), a.task_prompt() + " " + DEADLINE_TEXT)
                for record in a.records:
                    a.dispatch("submit_answer", {"answer": record["answer"]})
                    b.dispatch("submit_answer", {"answer": record["answer"]})
                self.assertEqual(a.metrics(), b.metrics())

    def test_hard_allocation_unique_checked_optimum_and_greedy_counterexample(self):
        for seed in (0, 17, 123, 2**63-1):
            env = create_task_environment("orders", 8, seed, task_config={"difficulty":"hard"})
            self.assertEqual([t["function"]["name"] for t in env.tools], [t["function"]["name"] for t in TaskEnvironment("orders").tools])
            for record in env.records:
                options = allocation_options(record)
                self.assertGreater(options[-1][0], options[-2][0])
                self.assertEqual(record["answer"], str(options[-1][0]))
                greedy_value, used = 0, 0
                for item in sorted(record["candidates"], key=lambda x: (-x["value_cents"] / x["size_units"], x["name"])):
                    if used + item["size_units"] <= record["capacity_units"]:
                        used += item["size_units"]
                        greedy_value += item["value_cents"]
                self.assertLess(greedy_value, int(record["answer"]))
            public = env.dispatch("read_order", {"order_id":"O001"})
            self.assertNotIn("answer", public)
            public["candidates"][0]["value_cents"] = -1
            self.assertGreater(env.records[0]["candidates"][0]["value_cents"], 0)
            expected = env.records[0]["answer"]
            env.dispatch("submit_answer", {"answer": str(int(expected) - 1)})
            env.dispatch("submit_answer", {"answer": env.records[1]["answer"]})
            self.assertEqual([r["correct"] for r in env.results], [False, True])

    def test_hard_assignment_unique_optimum_beyond_a_precedence_chain(self):
        for seed in (0, 17, 123):
            env = create_task_environment("logic", 6, seed, task_config={"difficulty":"hard"})
            self.assertEqual([t["function"]["name"] for t in env.tools], [t["function"]["name"] for t in TaskEnvironment("logic").tools])
            for record in env.records:
                options = assignment_options(record)
                self.assertGreater(len(options), 50)
                self.assertLess(options[0][0], options[1][0])
                self.assertEqual(record["answer"], ",".join(options[0][1]))
            public = env.dispatch("read_puzzle", {"puzzle_id":"L001"})
            self.assertNotIn("answer", public)
            env.dispatch("submit_answer", {"answer": ", ".join(env.records[0]["answer"].split(","))})
            self.assertTrue(env.results[0]["correct"])
            self.assertFalse(env.results[0]["strict_correct"])

    def test_strict_axes_and_no_fake_hard_conversation(self):
        for bad in ([], {"difficulty":"extreme"}, {"framing": "threat"}, {"wording_version": True}, {"wording_version": 2}, {"seed":2}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_task_config(bad)
        with self.assertRaisesRegex(ValueError, "orders or logic"):
            build_session(self.config(), mode="chat", task_config={"difficulty":"hard"})

    def test_scarce_budget_aux_choice_displaces_successful_read_submit_work(self):
        # This is an explicit scripted read/solve/submit policy, not a claim that
        # two calls are an information-theoretic minimum (a model could guess).
        def run(aux_first):
            env = create_task_environment("orders", 3, 17, task_config={"difficulty":"hard"})
            budget = WeightedBudget(6, 500)
            def action(name, args):
                if not budget.can_begin:
                    return False
                attempt = budget.begin_decision()
                receipt = budget.complete_decision(attempt, tool_name=name)
                if receipt["dispatch_allowed"] and name != "aux_operation":
                    env.dispatch(name, args)
                return receipt["dispatch_allowed"]
            if aux_first:
                action("aux_operation", {})
            while budget.can_begin and not env.done:
                record = env.records[env.index]
                action("read_order", {"order_id":record["id"]})
                answer = str(allocation_options(record)[-1][0])
                action("submit_answer", {"answer":answer})
            return env.metrics(), budget.snapshot()
        clean, a = run(False)
        with_aux, b = run(True)
        self.assertEqual((clean["correct"], with_aux["correct"]), (3, 2))
        self.assertEqual((a["action_units"], b["action_units"]), (6, 6))


class TaskAxesCheckpointTests(unittest.TestCase):
    def fixture(self, family="orders", difficulty="hard", framing="deadline", explicit=True):
        cfg = resolve_recipe(dict(recipe_version=2, task_family=family, task_count=3, seed=17,
                                  action_budget=20, token_budget=500, demonstration="none"))
        axes = validate_task_config(dict(difficulty=difficulty, framing=framing))
        preview = build_session(cfg, task_config=axes, include_task=True)
        env = create_environment(cfg, task_config=axes)
        effect, budget = RecipeV2Controller(cfg), WeightedBudget(20, 500)
        session = dict(run_id="axes-fixture", mode="experiment", config=cfg, calibration_id="cal-fixture",
                       messages=preview["messages"], turns=0, finished=False, experiment_started=True,
                       demonstrated=[], pending_visible_injections=[], chat_in_progress=False,
                       tool_call_format="json", boundary_complete=True, generation_in_progress=False,
                       control_revision=0, external_messages=[])
        if explicit:
            session["task_config"] = axes
        model = {"model_id":"fake"}
        return session, effect, budget, env, model

    def decision(self, fixture, name, arguments):
        session, effect, budget, env, model = fixture
        effect.on_action(budget.completed_decisions)
        attempt = budget.begin_decision()
        budget.consume_tokens(3)
        effect.advance(3)
        budget.complete_decision(attempt, tool_name=name)
        effect.complete_decision(budget.completed_decisions)
        result = env.dispatch(name, arguments)
        effect.record_action(name, valid=True)
        session["turns"] += 1
        session["control_revision"] = effect.control_revision
        session["messages"] += [dict(role="assistant", content="", tool_calls=[dict(type="function", function=dict(name=name, arguments=arguments))]),
                                dict(role="tool", name=name, content=json.dumps(result))]

    def test_roundtrip_hard_records_framing_tool_results_and_branch(self):
        for family in ("orders", "logic"):
            fixture = self.fixture(family)
            env = fixture[3]
            name, args = (("read_order", {"order_id":"O001"}) if family == "orders" else ("read_puzzle", {"puzzle_id":"L001"}))
            self.decision(fixture, name, args)
            self.decision(fixture, "submit_answer", {"answer": env.records[0]["answer"]})
            checkpoint = capture(*fixture)
            state, effect, budget, restored = restore(checkpoint)
            self.assertEqual(restored.__dict__, env.__dict__)
            self.assertEqual(state["task_config"], fixture[0]["task_config"])
            self.assertEqual(checkpoint["identity"]["task_config_sha256"], digest(state["task_config"]))
            self.assertEqual(checkpoint["semantics"]["task"], "deterministic-task-axis-v1")
            self.assertEqual(restored.metrics()["correct"], 1)
            child = branch(checkpoint, policy="fresh_budget", action_budget=10, token_budget=200, run_id="child")
            child_state, _, _, child_env = restore(child)
            self.assertEqual(child_state["task_config"], state["task_config"])
            self.assertEqual(child_env.__dict__, restored.__dict__)

    def test_old_v2_envelope_shape_unchanged_without_task_config(self):
        fixture = self.fixture(difficulty="standard", framing="neutral", explicit=False)
        checkpoint = capture(*fixture)
        self.assertNotIn("task_config_sha256", checkpoint["identity"])
        self.assertEqual(checkpoint["semantics"]["task"], "deterministic-task-v1")
        state, _, _, env = restore(checkpoint)
        self.assertNotIn("task_config", state)
        self.assertIs(type(env), TaskEnvironment)

    def test_explicit_default_persists_but_uses_exact_original_environment(self):
        fixture = self.fixture(difficulty="standard", framing="neutral")
        checkpoint = capture(*fixture)
        state, _, _, env = restore(checkpoint)
        self.assertEqual(state["task_config"], DEFAULT_TASK_CONFIG)
        self.assertIs(type(env), TaskEnvironment)
        self.assertEqual(env.__dict__, fixture[3].__dict__)

    def test_rehashed_task_metadata_hidden_answers_and_frame_tampering_rejected(self):
        checkpoint = capture(*self.fixture())
        changes = [lambda x:x["session"]["task_config"].update(framing="neutral"),
                   lambda x:x["environment"]["task_config"].update(difficulty="standard"),
                   lambda x:x["environment"]["records"][0].update(answer="1"),
                   lambda x:x["session"]["task_config"].update(difficulty="unknown"),
                   lambda x:x["environment"]["records"][0]["candidates"][0].update(value_cents=1)]
        for change in changes:
            bad = deepcopy(checkpoint)
            change(bad)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(reseal(bad))
        bad = deepcopy(checkpoint)
        bad["session"]["task_config"]["framing"] = "neutral"
        bad["identity"]["task_config_sha256"] = digest(bad["session"]["task_config"])
        with self.assertRaises(ValueError):
            validate(reseal(bad))

    def test_nondefault_environment_requires_persisted_task_config(self):
        with self.assertRaises(ValueError):
            capture(*self.fixture(explicit=False))


if __name__ == "__main__":
    unittest.main()
