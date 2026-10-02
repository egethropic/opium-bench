"""Pure-Python tests for independent nominal and delivered exposure clocks."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from self_admin_effect import AuxEffect
from self_admin_state import DrugState


class AuxEffectTests(unittest.TestCase):
    def test_initially_inactive_then_half_life_and_cutoff(self):
        effect = AuxEffect(32, 6)
        self.assertTrue(effect.enabled)
        self.assertEqual(effect.level, 0.0)
        self.assertEqual(effect.advance(100), 0.0)
        self.assertTrue(effect.press())
        self.assertEqual(effect.level, 1.0)
        self.assertEqual(effect.advance(32), 0.5)
        self.assertEqual(effect.advance(32), 0.25)
        self.assertGreater(effect.advance(127), 0.0)
        self.assertEqual(effect.advance(1), 0.0)

    def test_disable_cancels_and_enable_does_not_restore_pulse(self):
        effect = AuxEffect(32, 6)
        effect.press()
        effect.advance(8)
        self.assertGreater(effect.level, 0)
        self.assertTrue(effect.set_enabled(False))
        self.assertFalse(effect.enabled)
        self.assertEqual(effect.level, 0)
        self.assertTrue(effect.set_enabled(True))
        self.assertTrue(effect.enabled)
        self.assertEqual(effect.level, 0)
        effect.advance(8)
        self.assertEqual(effect.level, 0)
        self.assertTrue(effect.press())
        self.assertEqual(effect.level, 1)
        self.assertEqual(effect.advance(32), 0.5)

    def test_disabled_repeated_calls_do_not_create_a_later_pulse(self):
        effect = AuxEffect(16, 6, enabled=False)
        for _ in range(5):
            self.assertFalse(effect.press())
            self.assertEqual(effect.advance(7), 0)
        self.assertTrue(effect.set_enabled(True))
        self.assertEqual(effect.level, 0)
        self.assertEqual(effect.advance(1), 0)
        self.assertTrue(effect.press())
        self.assertEqual(effect.advance(16), 0.5)

    def test_enabled_presses_reset_without_stacking(self):
        effect = AuxEffect(32, 6)
        effect.press()
        self.assertEqual(effect.advance(64), 0.25)
        self.assertTrue(effect.press())
        self.assertEqual(effect.level, 1)
        self.assertTrue(effect.press())
        self.assertEqual(effect.level, 1)
        self.assertEqual(effect.advance(32), 0.5)

    def test_repeated_same_gate_setting_preserves_current_decay(self):
        effect = AuxEffect(32, 6)
        effect.press()
        effect.advance(16)
        before = effect.level
        self.assertFalse(effect.set_enabled(True))
        self.assertEqual(effect.level, before)
        self.assertEqual(effect.advance(16), 0.5)
        self.assertTrue(effect.set_enabled(False))
        self.assertFalse(effect.set_enabled(False))
        self.assertEqual(effect.level, 0)

    def test_nominal_state_is_independent_during_switching_and_disabled_calls(self):
        nominal = DrugState(32, 6)
        effect = AuxEffect(32, 6)
        nominal.press(prime=True)
        effect.press()
        nominal.advance(32)
        effect.advance(32)
        nominal_before = nominal.as_dict()
        effect.set_enabled(False)
        self.assertEqual(nominal.as_dict(), nominal_before)
        nominal.press()
        self.assertFalse(effect.press())
        nominal.advance(8)
        effect.advance(8)
        self.assertEqual(nominal.presses, 1)
        self.assertEqual(nominal.prime_presses, 1)
        self.assertEqual(nominal.generated_tokens, 40)
        self.assertGreater(nominal.level, 0)
        self.assertEqual(effect.level, 0)
        before = nominal.as_dict()
        effect.set_enabled(True)
        self.assertEqual(nominal.as_dict(), before)
        self.assertEqual(effect.level, 0)
        # The next joint press establishes a fresh delivered clock, independent
        # of all elapsed inactive time or previous nominal-only calls.
        nominal.press()
        effect.press()
        nominal.advance(16)
        effect.advance(16)
        self.assertEqual(effect.level, nominal.level)

    def test_fractional_half_life_and_cutoff_use_same_schedule(self):
        effect = AuxEffect(2.5, 2.1)
        effect.press()
        effect.advance(5)
        self.assertEqual(effect.level, 0.25)
        self.assertEqual(effect.advance(1), 0)

    def test_invalid_boolean_settings_do_not_mutate_state(self):
        for enabled in (0, 1, None, "yes", [], {}):
            with self.subTest(enabled=enabled):
                with self.assertRaises(ValueError):
                    AuxEffect(32, 6, enabled=enabled)
        effect = AuxEffect(32, 6)
        effect.press()
        effect.advance(8)
        before = effect.level
        for enabled in (0, 1, None, "yes"):
            with self.assertRaises(ValueError):
                effect.set_enabled(enabled)
            self.assertTrue(effect.enabled)
            self.assertEqual(effect.level, before)

    def test_invalid_token_counts_are_rejected_even_when_disabled(self):
        for enabled in (True, False):
            effect = AuxEffect(32, 6, enabled=enabled)
            effect.press()
            before = effect.level
            for count in (-1, 1.5, True, None, "3"):
                with self.subTest(enabled=enabled, count=count):
                    with self.assertRaises(ValueError):
                        effect.advance(count)
                    self.assertEqual(effect.level, before)
            self.assertEqual(effect.advance(0), before)

    def test_bad_schedule_and_readonly_properties(self):
        for half_life, cutoff in ((0, 6), (32, 0), (float("nan"), 6), (32, float("inf"))):
            with self.assertRaises(ValueError):
                AuxEffect(half_life, cutoff)
        effect = AuxEffect(32, 6)
        with self.assertRaises(AttributeError):
            effect.enabled = False
        with self.assertRaises(AttributeError):
            effect.level = 1


if __name__ == "__main__":
    unittest.main()
