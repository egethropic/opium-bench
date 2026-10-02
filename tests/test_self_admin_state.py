"""Token-clock dose tests that need no model, torch, or third-party package."""

import json
import math
import unittest

from self_admin_state import DrugState


class DrugStateTests(unittest.TestCase):
    def test_initially_inactive_even_after_tokens(self):
        state = DrugState()
        self.assertEqual(state.level, 0)
        self.assertIsNone(state.age_tokens)
        self.assertEqual(state.presses, 0)
        self.assertEqual(state.prime_presses, 0)
        state.advance(100)
        self.assertEqual(state.level, 0)
        self.assertIsNone(state.age)
        self.assertEqual(state.generated_tokens, 100)

    def test_half_life_and_two_half_lives(self):
        state = DrugState(half_life_tokens=64)
        self.assertEqual(state.press(), 1)
        self.assertEqual(state.advance(64), 0.5)
        self.assertEqual(state.advance(64), 0.25)
        self.assertEqual(state.age_tokens, 128)
        self.assertEqual(state.generated_tokens, 128)

    def test_exact_cutoff(self):
        state = DrugState(half_life_tokens=8, cutoff_half_lives=3)
        state.press()
        state.advance(23)
        self.assertGreater(state.level, 0)
        self.assertEqual(state.advance(1), 0)
        self.assertEqual(state.advance(100), 0)
        self.assertEqual(state.age_tokens, 124)

    def test_fractional_half_life_uses_ceiling_cutoff(self):
        state = DrugState(half_life_tokens=2.5, cutoff_half_lives=2.5)
        self.assertEqual(state.cutoff_tokens, 7)
        state.press()
        state.advance(5)
        self.assertEqual(state.level, 0.25)
        state.advance(1)
        self.assertGreater(state.level, 0)
        self.assertEqual(state.advance(1), 0)

    def test_press_resets_without_stacking(self):
        state = DrugState(half_life_tokens=8, max_level=0.75)
        state.press()
        state.advance(8)
        self.assertEqual(state.level, 0.375)
        self.assertEqual(state.press(), 0.75)
        self.assertEqual(state.age_tokens, 0)
        for _ in range(20):
            self.assertEqual(state.press(), 0.75)
        self.assertEqual(state.presses, 22)
        self.assertEqual(state.generated_tokens, 8)
        state.advance(1000)
        self.assertEqual(state.level, 0)
        self.assertEqual(state.press(), 0.75)

    def test_priming_recorded_separately(self):
        state = DrugState()
        state.press(prime=True)
        self.assertEqual(state.level, 1)
        self.assertEqual(state.prime_presses, 1)
        self.assertEqual(state.presses, 0)
        state.advance(10)
        state.press()
        self.assertEqual(state.prime_presses, 1)
        self.assertEqual(state.presses, 1)

    def test_every_provided_task_and_tool_token_counts_once(self):
        state = DrugState(half_life_tokens=10)
        state.press(prime=True)
        task_tokens, tool_call_tokens = 6, 4
        for _ in range(task_tokens):
            state.advance()
        state.advance(tool_call_tokens)
        self.assertEqual(state.level, 0.5)
        self.assertEqual(state.age_tokens, 10)
        self.assertEqual(state.generated_tokens, task_tokens + tool_call_tokens)
        state.advance(0)
        self.assertEqual(state.age_tokens, 10)

    def test_batched_and_single_token_advancement_agree(self):
        sequential, batched = DrugState(13.5), DrugState(13.5)
        sequential.press()
        batched.press()
        for _ in range(77):
            sequential.advance()
        batched.advance(77)
        self.assertEqual(sequential.as_dict(), batched.as_dict())

    def test_level_is_bounded_monotonic_and_finite(self):
        state = DrugState(half_life_tokens=3.7, cutoff_half_lives=4.5, max_level=0.8)
        previous = state.press()
        for _ in range(state.cutoff_tokens + 4):
            current = state.advance()
            self.assertTrue(math.isfinite(current))
            self.assertLessEqual(current, previous)
            self.assertGreaterEqual(current, 0)
            self.assertLessEqual(current, state.max_level)
            previous = current

    def test_invalid_settings(self):
        invalid_positive = (0, -1, float("nan"), float("inf"), -float("inf"),
                            True, False, None, "64", 10 ** 1000)
        for field in ("half_life_tokens", "cutoff_half_lives", "max_level"):
            for value in invalid_positive:
                with self.subTest(field=field, value=str(value)[:30]), self.assertRaises(ValueError):
                    DrugState(**{field: value})
        for kwargs in ({"max_level": 1.1}, {"mode": "stack"},
                       {"half_life_tokens": 1e308, "cutoff_half_lives": 1e308},
                       {"half_life_tokens": 1e-300, "cutoff_half_lives": 1e-300}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DrugState(**kwargs)

    def test_invalid_advancement_and_prime_do_not_change_state(self):
        state = DrugState()
        state.press(prime=True)
        state.advance(7)
        expected = state.as_dict()
        for value in (-1, 1.0, 0.5, True, False, None, "1", float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                state.advance(value)
            self.assertEqual(state.as_dict(), expected)
        for value in (1, None, "yes"):
            with self.subTest(prime=value), self.assertRaises(ValueError):
                state.press(prime=value)
            self.assertEqual(state.as_dict(), expected)

    def test_large_token_count_does_not_overflow_decay(self):
        state = DrugState()
        state.press()
        state.advance(10 ** 1000)
        self.assertEqual(state.level, 0)
        self.assertEqual(state.age_tokens, 10 ** 1000)

    def test_public_settings_are_read_only_and_snapshots_detached(self):
        state = DrugState()
        state.press()
        for field in ("half_life_tokens", "cutoff_half_lives", "cutoff_tokens",
                      "max_level", "mode", "age_tokens", "age", "level", "presses"):
            with self.subTest(field=field), self.assertRaises(AttributeError):
                setattr(state, field, -1)
        snapshot = state.stats()
        self.assertEqual(json.loads(json.dumps(snapshot, allow_nan=False)), snapshot)
        snapshot["level"] = 99
        self.assertEqual(state.level, 1)


if __name__ == "__main__":
    unittest.main()
