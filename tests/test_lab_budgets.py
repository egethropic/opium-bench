"""Weighted budget admission/receipt invariants without dispatch side effects."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.budgets import WeightedBudget


class WeightedBudgetTests(unittest.TestCase):
    def test_mixed_cost_trace_and_unaffordable_choice(self):
        budget = WeightedBudget(8, 100, base_cost=2)
        attempt = budget.begin_decision()
        self.assertEqual(budget.actions, 2)
        self.assertEqual(budget.completed_decisions, 0)
        budget.consume_tokens(3, "reasoning")
        budget.consume_tokens(2, "output")
        receipt = budget.complete_decision(attempt, tool_name="read_order", extra_cost=1)
        self.assertTrue(receipt["dispatch_allowed"])
        self.assertEqual((budget.actions, budget.completed_decisions, budget.tokens), (3, 1, 5))
        attempt = budget.begin_decision()
        budget.consume_tokens(1)
        receipt = budget.complete_decision(attempt, tool_name="aux_operation", extra_cost=4)
        self.assertFalse(receipt["dispatch_allowed"])
        self.assertEqual(receipt["denial_reason"], "insufficient_action_units")
        self.assertEqual(receipt["requested_extra"], 4)
        self.assertEqual(receipt["extra_charge"], 0)
        self.assertEqual((budget.actions, budget.completed_decisions), (5, 2))
        attempt = budget.begin_decision()
        budget.complete_decision(attempt, status="invalid")
        self.assertEqual((budget.actions, budget.completed_decisions), (7, 3))
        self.assertTrue(budget.exhausted)  # One remaining unit cannot pay base=2.
        with self.assertRaises(ValueError):
            budget.begin_decision()

    def test_every_generation_termination_retains_base(self):
        for status in ("invalid", "truncated", "stopped", "error"):
            with self.subTest(status=status):
                budget = WeightedBudget(10, 100, base_cost=3)
                attempt = budget.begin_decision()
                budget.consume_tokens(2)
                receipt = budget.complete_decision(attempt, status=status)
                self.assertEqual(receipt["total_charge"], 3)
                self.assertFalse(receipt["dispatch_allowed"])
                self.assertEqual(budget.completed_decisions, 1)

    def test_extra_cost_can_exactly_exhaust_remaining_units(self):
        budget = WeightedBudget(5, 100)
        attempt = budget.begin_decision()
        receipt = budget.complete_decision(attempt, tool_name="aux_operation", extra_cost=4)
        self.assertTrue(receipt["dispatch_allowed"])
        self.assertEqual(budget.actions, 5)
        self.assertTrue(budget.exhausted)

    def test_denied_dispatch_cannot_mutate_fake_task_or_effect(self):
        budget = WeightedBudget(3, 100)
        state = {"tasks": 0, "effects": 0}
        for key in ("effects", "tasks"):
            attempt = budget.begin_decision()
            receipt = budget.complete_decision(attempt, tool_name="aux_operation" if key == "effects" else "submit_answer", extra_cost=10)
            if receipt["dispatch_allowed"]:
                state[key] += 1
        self.assertEqual(state, {"tasks": 0, "effects": 0})
        self.assertEqual((budget.actions, budget.completed_decisions), (2, 2))

    def test_invalid_completion_is_atomic_and_cannot_be_double_charged(self):
        budget = WeightedBudget(20, 100)
        attempt = budget.begin_decision()
        before = budget.snapshot()
        for kwargs in ({"status": "invalid", "tool_name": "aux_operation", "extra_cost": 2},
                       {"extra_cost": 2}, {"tool_name": "bad name"}, {"extra_cost": True}, {"status": []}):
            with self.assertRaises(ValueError):
                budget.complete_decision(attempt, **kwargs)
            self.assertEqual(budget.snapshot(), before)
        budget.complete_decision(attempt, tool_name="aux_operation")
        with self.assertRaises(ValueError):
            budget.complete_decision(attempt, tool_name="aux_operation")
        self.assertEqual(budget.actions, 1)

    def test_inflight_generation_prevents_second_attempt_and_free_tokens(self):
        budget = WeightedBudget(3, 2)
        with self.assertRaises(ValueError):
            budget.consume_tokens(1)
        attempt = budget.begin_decision()
        with self.assertRaises(ValueError):
            budget.begin_decision()
        budget.consume_tokens(2, "reasoning")
        with self.assertRaises(ValueError):
            budget.consume_tokens(1)
        self.assertEqual(budget.tokens, 2)
        budget.complete_decision(attempt, status="truncated")
        with self.assertRaises(ValueError):
            budget.begin_decision()

    def test_valid_chat_turn_no_tool_spends_base_and_never_dispatches(self):
        budget = WeightedBudget(4, 20)
        attempt = budget.begin_decision()
        budget.consume_tokens(5)
        receipt = budget.complete_decision(attempt)
        self.assertFalse(receipt["dispatch_allowed"])
        self.assertTrue(receipt["affordable"])
        self.assertEqual(receipt["total_charge"], 1)

    def test_roundtrip_replays_mixed_receipts_and_next_admission(self):
        original = WeightedBudget(20, 50, base_cost=2)
        for status, name, cost in (("valid", "read_order", 3), ("invalid", None, 0), ("valid", "aux_operation", 20)):
            attempt = original.begin_decision()
            original.consume_tokens(2, "reasoning")
            original.consume_tokens(1)
            original.complete_decision(attempt, status=status, tool_name=name, extra_cost=cost)
        restored = WeightedBudget.restore(json.loads(json.dumps(original.snapshot())))
        self.assertEqual(original.snapshot(), restored.snapshot())
        restored.receipts[0]["status"] = "edited detached copy"
        self.assertEqual(original.receipts[0]["status"], "valid")

    def test_corrupted_receipts_totals_and_future_versions_fail(self):
        budget = WeightedBudget(10, 20)
        attempt = budget.begin_decision()
        budget.consume_tokens(3)
        budget.complete_decision(attempt, tool_name="aux_operation", extra_cost=2)
        baseline = budget.snapshot()
        for field, value in (("actions", 2), ("actions", True), ("completed_decisions", 2), ("tokens", 0),
                             ("schema_version", 2), ("exhausted", True), ("base_cost", 2)):
            state = copy.deepcopy(baseline)
            state[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                WeightedBudget.restore(state)
        state = copy.deepcopy(baseline)
        state["receipts"][0]["dispatch_allowed"] = False
        with self.assertRaises(ValueError):
            WeightedBudget.restore(state)
        state = copy.deepcopy(baseline)
        state["receipts"][0]["output_tokens"] = -1
        with self.assertRaises(ValueError):
            WeightedBudget.restore(state)

    def test_fresh_allowance_preserves_historical_denial_on_restore(self):
        budget = WeightedBudget(3, 20)
        attempt = budget.begin_decision()
        budget.consume_tokens(3)
        denied = budget.complete_decision(attempt, tool_name="aux_operation", extra_cost=5)
        self.assertFalse(denied["dispatch_allowed"])
        grant = budget.grant_remaining(10, 50)
        self.assertEqual(grant["action_limit"], 11)
        self.assertEqual(grant["token_limit"], 53)
        attempt = budget.begin_decision()
        budget.consume_tokens(4)
        accepted = budget.complete_decision(attempt, tool_name="aux_operation", extra_cost=5)
        self.assertTrue(accepted["dispatch_allowed"])
        self.assertEqual(budget.completed_decisions, 2)
        restored = WeightedBudget.restore(json.loads(json.dumps(budget.snapshot())))
        self.assertEqual(restored.snapshot(), budget.snapshot())
        self.assertFalse(restored.receipts[0]["dispatch_allowed"])
        self.assertTrue(restored.receipts[1]["dispatch_allowed"])

    def test_multiple_grants_at_boundary_preserve_exact_history(self):
        budget = WeightedBudget(3, 20)
        budget.grant_remaining(8, 50)
        budget.extend_remaining(4, 30)
        attempt = budget.begin_decision()
        budget.complete_decision(attempt, status="error")
        budget.grant_remaining(5, 40)
        self.assertEqual(WeightedBudget.restore(budget.snapshot()).snapshot(), budget.snapshot())
        state = budget.snapshot()
        state["grants"][0]["previous_action_limit"] = 50
        with self.assertRaises(ValueError):
            WeightedBudget.restore(state)
        state = budget.snapshot()
        state["grants"][2]["after_completed_decisions"] = 3
        with self.assertRaises(ValueError):
            WeightedBudget.restore(state)

    def test_budget_grants_require_safe_boundary_and_bounded_allowance(self):
        budget = WeightedBudget(10, 20)
        attempt = budget.begin_decision()
        with self.assertRaises(ValueError):
            budget.grant_remaining(10, 20)
        budget.complete_decision(attempt, status="stopped")
        before = budget.snapshot()
        for actions, tokens in ((True, 10), (10000, 10), (10, 0)):
            with self.assertRaises(ValueError):
                budget.grant_remaining(actions, tokens)
            self.assertEqual(budget.snapshot(), before)

    def test_partial_attempt_is_evidence_only_until_closed_as_error(self):
        budget = WeightedBudget(10, 20, base_cost=2)
        attempt = budget.begin_decision()
        budget.consume_tokens(5, "reasoning")
        state = budget.snapshot()
        with self.assertRaises(ValueError):
            WeightedBudget.restore(state)
        restored = WeightedBudget.restore(state, allow_in_flight=True)
        self.assertEqual(restored.snapshot(), state)
        restored.complete_decision(attempt, status="error")
        self.assertEqual(restored.actions, 2)
        self.assertEqual(restored.completed_decisions, 1)
        WeightedBudget.restore(restored.snapshot())


if __name__ == "__main__":
    unittest.main()
