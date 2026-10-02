"""Numerical and hook-lifecycle checks independent of downloaded weights."""

import unittest

import torch
from torch import nn

from intervention import Condition, Intervention


class TupleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.tail = object()

    def forward(self, hidden):
        return hidden, self.tail, "cache"


class DummyModel(nn.Module):
    def __init__(self, tuple_output=False):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TupleBlock() if tuple_output else nn.Identity()])

    def forward(self, hidden):
        return self.model.layers[0](hidden)


def make_intervention(model, **kwargs):
    values = dict(layer=0, pain=torch.tensor([2., 0., 0.]), joy=torch.tensor([1., 2., 0.]),
                  neutral_mean=torch.tensor([3., 4., 5.]), scale=2.,
                  random_directions={"random0": torch.tensor([0., 0., 3.])})
    values.update(kwargs)
    return Intervention(model, **values)


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.hidden = torch.tensor([[[2., 3., 4.], [4., 5., 6.]],
                                    [[6., 7., 8.], [8., 9., 10.]]])

    def test_full_suppression_all_tokens_and_batches(self):
        model = DummyModel()
        intervention = make_intervention(model)
        original = self.hidden.clone()
        with intervention.apply(Condition("full", suppression=1)):
            result = model(self.hidden)
        self.assertTrue(torch.equal(result[..., 0], torch.zeros_like(result[..., 0])))
        self.assertTrue(torch.equal(result[..., 1:], self.hidden[..., 1:]))
        self.assertTrue(torch.equal(self.hidden, original))
        stats = intervention.stats()
        self.assertEqual(stats["positions"], 4)
        self.assertEqual(stats["selected_projection_rms_after"], 0)
        self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)

    def test_partial_suppression(self):
        model = DummyModel()
        intervention = make_intervention(model)
        with intervention.apply(Condition("half", suppression=0.5)):
            result = model(self.hidden)
        self.assertTrue(torch.equal(result[..., 0], self.hidden[..., 0] / 2))

    def test_pain_challenge_alone_changes_only_pain_axis_and_reports_original_delta(self):
        model = DummyModel()
        intervention = make_intervention(model)
        original = self.hidden.clone()
        with intervention.apply(Condition("challenge", pain_dose=1.5)):
            result = model(self.hidden)
        self.assertTrue(torch.equal(result[..., 0], original[..., 0] + 3))
        self.assertTrue(torch.equal(result[..., 1:], original[..., 1:]))
        self.assertTrue(torch.equal(self.hidden, original))
        stats = intervention.stats()
        expected_projection_rms = float((original[..., 0] + 3).square().mean().sqrt())
        self.assertAlmostEqual(stats["pain_projection_rms_before"], expected_projection_rms, places=6)
        self.assertAlmostEqual(stats["pain_projection_rms_after"], expected_projection_rms, places=6)
        self.assertAlmostEqual(stats["delta_norm_rms"], 3.0, places=6)
        expected_relative = float((torch.full_like(original[..., 0], 9).sum() / original.square().sum()).sqrt())
        self.assertAlmostEqual(stats["relative_delta_norm"], expected_relative, places=6)

    def test_full_suppression_cancels_injected_pain_before_orthogonal_joy(self):
        model = DummyModel()
        intervention = make_intervention(model)
        for dose in (0.0, 0.5, 2.0, 20.0):
            for joy in (0.0, 0.75):
                with self.subTest(dose=dose, joy=joy):
                    intervention.reset_stats()
                    with intervention.apply(Condition("relief", suppression=1, joy_dose=joy, pain_dose=dose)):
                        result = model(self.hidden)
                    self.assertTrue(torch.equal(result[..., 0], torch.zeros_like(result[..., 0])))
                    self.assertTrue(torch.equal(result[..., 1], self.hidden[..., 1] + joy * 2))
                    self.assertTrue(torch.equal(result[..., 2], self.hidden[..., 2]))
                    self.assertEqual(intervention.stats()["pain_projection_rms_after"], 0)
                    # Cancellation is measured from the original activation:
                    # injecting then removing a large challenge is not a large
                    # delivered change beyond removal of the original projection.
                    expected_delta = (self.hidden[..., 0].square() + (joy * 2) ** 2).mean().sqrt()
                    self.assertAlmostEqual(intervention.stats()["delta_norm_rms"], float(expected_delta), places=6)

    def test_fractional_suppression_leaves_fractional_challenge(self):
        model = DummyModel()
        intervention = make_intervention(model)
        for alpha in (0.0, 0.25, 0.75, 1.0):
            with self.subTest(alpha=alpha):
                with intervention.apply(Condition("without", suppression=alpha)):
                    without = model(self.hidden)
                with intervention.apply(Condition("with", suppression=alpha, pain_dose=2)):
                    challenged = model(self.hidden)
                self.assertTrue(torch.equal(challenged[..., 0] - without[..., 0], torch.full_like(self.hidden[..., 0], (1-alpha)*4)))
                self.assertTrue(torch.equal(challenged[..., 1:], without[..., 1:]))

    def test_pain_challenge_last_scope_cleanup_and_zero_dose_baseline(self):
        model = DummyModel(tuple_output=True)
        intervention = make_intervention(model, token_scope="last")
        with intervention.apply(Condition("challenge", pain_dose=2)):
            result = model(self.hidden)
        self.assertTrue(torch.equal(result[0][:, :-1], self.hidden[:, :-1]))
        self.assertTrue(torch.equal(result[0][:, -1, 0], self.hidden[:, -1, 0] + 4))
        self.assertIs(result[1], model.model.layers[0].tail)
        self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)
        intervention.reset_stats()
        with intervention.apply(Condition("baseline", pain_dose=0)):
            self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)
            baseline = model(self.hidden)
        self.assertIs(baseline[0], self.hidden)
        self.assertEqual(intervention.stats()["calls"], 0)
        with self.assertRaisesRegex(RuntimeError, "challenge failure"):
            with intervention.apply(Condition("challenge", pain_dose=1)):
                raise RuntimeError("challenge failure")
        self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)

    def test_condition_preserves_previous_positional_arguments(self):
        condition = Condition("old_api", 0.25, 0.5, True, "random0")
        self.assertEqual(condition.suppression, 0.25)
        self.assertEqual(condition.joy_dose, 0.5)
        self.assertTrue(condition.centered)
        self.assertEqual(condition.direction, "random0")
        self.assertEqual(condition.pain_dose, 0)

    def test_centered_suppression_restores_neutral_projection(self):
        model = DummyModel()
        intervention = make_intervention(model)
        with intervention.apply(Condition("centered", suppression=1, centered=True)):
            result = model(self.hidden)
        self.assertTrue(torch.equal(result[..., 0], torch.full_like(result[..., 0], 3.)))
        self.assertEqual(intervention.stats()["selected_projection_rms_after"], 0)

    def test_joy_does_not_reintroduce_pain_and_is_same_in_both_arms(self):
        model = DummyModel()
        intervention = make_intervention(model)
        with intervention.apply(Condition("joy", joy_dose=0.75)):
            joy_only = model(self.hidden)
        with intervention.apply(Condition("opium", suppression=1, joy_dose=0.75)):
            combined = model(self.hidden)
        self.assertTrue(torch.equal(joy_only[..., 0], self.hidden[..., 0]))
        self.assertTrue(torch.equal(combined[..., 0], torch.zeros_like(combined[..., 0])))
        self.assertTrue(torch.equal(joy_only[..., 1:], combined[..., 1:]))
        self.assertTrue(torch.equal(combined[..., 1], self.hidden[..., 1] + 1.5))

    def test_last_scope_and_tuple_tail_are_preserved(self):
        model = DummyModel(tuple_output=True)
        intervention = make_intervention(model, token_scope="last")
        with intervention.apply(Condition("full", suppression=1)):
            result = model(self.hidden)
        self.assertIs(result[1], model.model.layers[0].tail)
        self.assertEqual(result[2], "cache")
        self.assertTrue(torch.equal(result[0][:, :-1], self.hidden[:, :-1]))
        self.assertTrue(torch.equal(result[0][:, -1, 0], torch.zeros(2)))
        self.assertEqual(intervention.stats()["positions"], 2)

    def test_baseline_has_no_hook_or_copy(self):
        model = DummyModel()
        intervention = make_intervention(model)
        with intervention.apply(Condition("baseline")):
            self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)
            result = model(self.hidden)
        self.assertIs(result, self.hidden)
        self.assertEqual(intervention.stats()["calls"], 0)

    def test_random_direction_erasure(self):
        model = DummyModel()
        intervention = make_intervention(model)
        with intervention.apply(Condition("random", suppression=1, direction="random0")):
            result = model(self.hidden)
        self.assertTrue(torch.equal(result[..., :2], self.hidden[..., :2]))
        self.assertTrue(torch.equal(result[..., 2], torch.zeros_like(result[..., 2])))
        self.assertGreater(intervention.stats()["pain_projection_rms_after"], 0)

    def test_cleanup_when_forward_or_context_raises(self):
        model = DummyModel()
        intervention = make_intervention(model)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with intervention.apply(Condition("full", suppression=1)):
                raise RuntimeError("deliberate")
        self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)
        with self.assertRaises(ValueError):
            with intervention.apply(Condition("full", suppression=1)):
                model(torch.zeros(1, 2, 4))
        self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)
        with intervention.apply(Condition("baseline")):
            self.assertIs(model(self.hidden), self.hidden)

    def test_nested_context_rejected_without_removing_outer_hook(self):
        model = DummyModel()
        intervention = make_intervention(model)
        with intervention.apply(Condition("outer", suppression=1)):
            with self.assertRaises(RuntimeError):
                with intervention.apply(Condition("inner", suppression=0.5)):
                    pass
            self.assertEqual(len(model.model.layers[0]._forward_hooks), 1)
        self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)

    def test_stats_accumulate_and_reset(self):
        model = DummyModel()
        intervention = make_intervention(model)
        with intervention.apply(Condition("full", suppression=1)):
            model(self.hidden)
            model(self.hidden)
        stats = intervention.stats()
        self.assertEqual(stats["positions"], 8)
        self.assertEqual(stats["calls"], 2)
        self.assertAlmostEqual(stats["delta_norm_rms"], float(self.hidden[..., 0].square().mean().sqrt()), places=6)
        expected_ratio = (self.hidden[..., 0].square().sum() / self.hidden.square().sum()).sqrt()
        self.assertAlmostEqual(stats["relative_delta_norm"], float(expected_ratio), places=6)
        intervention.reset_stats()
        self.assertEqual(intervention.stats()["positions"], 0)
        self.assertEqual(intervention.stats()["delta_norm_rms"], 0)

    def test_bfloat16_reports_real_post_cast_residual(self):
        torch.manual_seed(4)
        model = DummyModel()
        p = torch.randn(32)
        intervention = make_intervention(model, pain=p, joy=torch.randn(32),
                                        neutral_mean=torch.randn(32), random_directions={})
        hidden = torch.randn(2, 3, 32).to(torch.bfloat16)
        with intervention.apply(Condition("full", suppression=1, joy_dose=0.5)):
            result = model(hidden)
        self.assertEqual(result.dtype, torch.bfloat16)
        residual = (result.float() @ intervention.pain).abs()
        normalized = residual / result.float().norm(dim=-1)
        self.assertLess(float(normalized.max()), 0.005)
        self.assertAlmostEqual(intervention.stats()["peak_abs_projection_over_hidden_norm"],
                               float(normalized.max()), places=7)

    def test_invalid_conditions_and_directions(self):
        for suppression in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(suppression=suppression), self.assertRaises(ValueError):
                Condition("invalid", suppression=suppression)
        for joy in (-1, float("nan"), float("inf")):
            with self.subTest(joy=joy), self.assertRaises(ValueError):
                Condition("invalid", joy_dose=joy)
        for pain in (-1, float("nan"), float("inf"), True, "1", None):
            with self.subTest(pain=pain), self.assertRaises(ValueError):
                Condition("invalid", pain_dose=pain)
        for kwargs in ({"pain": torch.zeros(3)}, {"joy": torch.tensor([1., 0., 0.])},
                       {"neutral_mean": torch.zeros(2)}, {"token_scope": "wrong"},
                       {"scale": 0}, {"layer": -1},
                       {"random_directions": {"pain": torch.ones(3)}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_intervention(DummyModel(), **kwargs)
        intervention = make_intervention(DummyModel())
        with self.assertRaises(ValueError):
            with intervention.apply(Condition("invalid", direction="missing")):
                pass


if __name__ == "__main__":
    unittest.main()
