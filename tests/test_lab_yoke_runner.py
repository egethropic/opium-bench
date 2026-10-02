"""Runtime-boundary yoke dispatch, independent choices, durable continuation."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

from lab.controller_v2 import RecipeV2Controller
from lab.effects import EffectScheduler, content_hash, validate_preset
from lab.exposure import AXES, build_yoke
from lab.yoke_runner import YokeDriver


def effect(clock="tokens", shape="constant"):
    return validate_preset(dict(id="joy", gains={"joy":1}, decay=dict(clock=clock, shape=shape, half_life=2, cutoff=4)))


def coefficients(value=0):
    return {axis: value if axis == "joy_orthogonal" else 0. for axis in AXES}


def token(index, decision=0, value=1, phase="output"):
    return dict(type="token", generation_index=index, phase=phase,
        dose=dict(schema_version=2, generated_token_index=index, completed_decisions=decision,
                  phase=phase, effective=coefficients(value), applied=bool(value)),
        measurements=dict(delivered_edit_norm=.5))


def make_schedule(events=None, *, presets=None, clock="tokens", horizon=4, decisions=2):
    presets = [effect()] if presets is None else presets
    if events is None:
        scheduler = EffectScheduler(presets)
        events = [scheduler.inject("joy")] + [token(i, i // 2) for i in range(horizon)]
    raw = json.dumps(events, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    sha = hashlib.sha256(raw).hexdigest()
    return build_yoke(events, presets, clock=clock, pair_id="pair", source=dict(run_id="source",
        events_sha256=sha, prefix_sha256=sha, prefix_event_count=len(events), config_sha256="a"*64,
        model_fingerprint_sha256="b"*64, end_tokens=horizon, end_decisions=decisions))


def recipient(schedule, **config):
    c = RecipeV2Controller(dict(recipe_version=2, effect_presets=schedule["presets"],
        auxiliary_tools=[dict(id="aux", name="aux_operation", preset_id="joy")], demonstration="none", **config))
    return c, YokeDriver(c, schedule)


def reseal(state):
    value = deepcopy(state)
    value.pop("sha256", None)
    value["sha256"] = content_hash(value)
    return value


class YokeRunnerTests(unittest.TestCase):
    def test_exact_token_dispatch_clocks_and_exposure(self):
        c, d = recipient(make_schedule())
        for i in range(4):
            events = d.before_forward()
            self.assertEqual(len(events), int(i == 0))
            self.assertEqual(c.generated_tokens, i)
            self.assertEqual(c.snapshot()["phase_coefficients"]["output"]["joy_orthogonal"], 1)
            self.assertEqual(d.before_forward(), [])
            d.record_event(token(i, i // 2))
            c.advance()
        end = d.before_forward()[0]
        self.assertEqual(end["command"]["kind"], "end")
        self.assertFalse(c.scheduler.pulses)
        d.record_event(token(4, 2, value=0))
        c.advance()
        report = d.finish()["report"]
        self.assertTrue(report["observed_coefficients_equal"])
        self.assertEqual(report["delivered_pulses"], 1)
        self.assertEqual(report["outside_source_positions"], 1)
        self.assertEqual(report["all_recipient_button_additional_effects"], 0)
        RecipeV2Controller.restore(c.snapshot())

    def test_own_button_validates_acknowledges_and_has_no_effect_or_draw(self):
        c, d = recipient(make_schedule())
        d.before_forward()
        before = c.scheduler.snapshot()
        with patch.object(c, "press", side_effect=AssertionError("must bypass ordinary press")):
            for actor in ("model", "human", "demonstration", "schedule"):
                event = d.press(actor, "aux_operation")
                self.assertFalse(event["delivered"])
                self.assertEqual(event["acknowledgment"], "Operation completed.")
                self.assertEqual(event["actor"], actor)
                self.assertIsNone(event["draw"])
        self.assertEqual(c.scheduler.snapshot(), before)
        self.assertEqual(d.report()["own_button_calls"], {"aux_operation":1})
        self.assertEqual(d.report()["external_button_calls"]["human"], {"aux_operation":1})
        self.assertEqual(c.counts["sham"], 4)
        RecipeV2Controller.restore(c.snapshot())
        YokeDriver.restore(d.snapshot(), RecipeV2Controller.restore(c.snapshot()))
        old = d.snapshot()
        for kwargs in ({"tool":"missing"},{"arguments":{"unexpected":1}},{"actor":"other"}):
            with self.assertRaises(ValueError):d.press(**kwargs)
            self.assertEqual(d.snapshot(),old)

    def test_hidden_recipient_tool_only_external_actor_can_press(self):
        schedule=make_schedule()
        c=RecipeV2Controller(dict(recipe_version=2,effect_presets=schedule["presets"],demonstration="none",
            auxiliary_tools=[dict(id="aux",name="hidden_aux",preset_id="joy",visible=False)]))
        d=YokeDriver(c,schedule)
        with self.assertRaises(ValueError):d.press(tool="hidden_aux")
        d.press(actor="human",tool="hidden_aux")
        self.assertEqual(c.counts["human"],1)
        self.assertFalse(c.scheduler.pulses)

    def test_decision_clock_does_not_age_on_tokens_and_ages_before_injection(self):
        p=effect("decisions","exponential");s=EffectScheduler([p]);s.complete_decision(1)
        schedule=make_schedule([s.inject("joy")],presets=[p],clock="decisions",decisions=4)
        c,d=recipient(schedule)
        d.on_action(0);self.assertEqual(d.before_forward(),[])
        c.advance(10);self.assertEqual(d.before_forward(),[])
        c.complete_decision(1);d.press();d.on_action(1)
        self.assertEqual(d.before_forward()[0]["command"]["at"],1)
        self.assertEqual(c.level,1)
        c.advance(500);self.assertEqual(c.level,1)
        c.complete_decision(2);d.on_action(2);d.before_forward()
        self.assertAlmostEqual(c.level,2**-.5)
        self.assertEqual(c.generated_tokens,510)

    def test_source_cancellation_has_authority_over_target_phase_policy(self):
        p=effect();s=EffectScheduler([p]);events=[s.inject("joy")];s.complete_decision(2);events.append(s.cancel(actor="schedule"))
        schedule=make_schedule(events,presets=[p],clock="decisions",decisions=3)
        c,d=recipient(schedule,conditions=["test"],condition="test",mapping_policy="explicit",
            mapping_schedule=[dict(after_decisions=0,label="first",mappings={"aux_operation":[dict(preset_id="joy",probability=1)]}),
                dict(after_decisions=1,label="second",mappings={"aux_operation":[dict(preset_id=None,probability=1)]})],transition_policy="cancel")
        d.on_action(0);d.before_forward();self.assertEqual(c.level,1)
        c.complete_decision(1);event=d.on_action(1);d.before_forward()
        self.assertEqual(event["transition_policy"],"source_yoke_only")
        self.assertIsNone(event["cancellation"]);self.assertEqual(c.level,1)
        c.complete_decision(2);d.on_action(2)
        self.assertEqual(d.before_forward()[0]["command"]["kind"],"cancel")
        self.assertEqual(c.level,0)
        RecipeV2Controller.restore(c.snapshot())

    def test_late_pulse_is_skipped_and_end_cancels_even_when_late(self):
        c,d=recipient(make_schedule())
        c.advance(2)
        row=d.before_forward()[0]
        self.assertFalse(row["receipt"]["delivered"])
        self.assertEqual(row["receipt"]["index_error"],2)
        self.assertEqual(row["receipt"]["reason"],"missed_source_index")
        self.assertFalse(c.scheduler.pulses)
        c.advance(5);end=d.before_forward()[0]
        self.assertTrue(end["receipt"]["delivered"])
        self.assertEqual(end["command"]["kind"],"end")
        self.assertEqual(d.report()["absolute_index_error"],2)
        YokeDriver.restore(d.snapshot(),RecipeV2Controller.restore(c.snapshot()))

    def test_multiple_missed_commands_emit_monotonic_sequence_before_end(self):
        p=effect();s=EffectScheduler([p]);events=[s.inject("joy")];s.advance_tokens(2);events.append(s.cancel(actor="schedule"))
        c,d=recipient(make_schedule(events,presets=[p]))
        c.advance(5);rows=d.before_forward()
        self.assertEqual([row["command"]["kind"] for row in rows],["inject","cancel","end"])
        self.assertEqual([row["sequence"] for row in rows],sorted(row["sequence"] for row in rows))
        self.assertEqual([row["receipt"]["delivered"] for row in rows],[False,False,True])
        YokeDriver.restore(d.snapshot(),RecipeV2Controller.restore(c.snapshot()))

    def test_disabled_target_records_failed_source_delivery_without_later_revival(self):
        c,d=recipient(make_schedule(),aux_enabled=False)
        event=d.before_forward()[0]
        self.assertEqual(event["receipt"]["reason"],"gate_disabled")
        self.assertEqual(c.counts["disabled"],1)
        c.set_controls(enabled=True)
        self.assertEqual(d.before_forward(),[])
        self.assertFalse(c.scheduler.pulses)
        self.assertEqual(d.report()["failed_or_missed_pulses"],1)
        YokeDriver.restore(d.snapshot(),RecipeV2Controller.restore(c.snapshot()))

    def test_failed_scheduler_admission_rolls_back_and_never_retries(self):
        c,d=recipient(make_schedule())
        with patch.object(c.scheduler,"inject",side_effect=ValueError("capacity limit")):
            event=d.before_forward()[0]
        self.assertFalse(event["receipt"]["delivered"])
        self.assertEqual(event["receipt"]["reason"],"dispatch_error: capacity limit")
        self.assertEqual(c.injection_index,0)
        self.assertFalse(c.scheduler.pulses)
        self.assertEqual(d.before_forward(),[])
        YokeDriver.restore(d.snapshot(),RecipeV2Controller.restore(c.snapshot()))

    def test_early_termination_never_delivers_a_pulse_without_a_next_forward(self):
        c,d=recipient(make_schedule())
        finished=d.finish("stopped")
        self.assertEqual(c.counts["delivered"],0)
        self.assertEqual(finished["deliveries"][0]["receipt"]["reason"],"target_finished_before_next_forward")
        self.assertTrue(finished["report"]["terminated_before_source_horizon"])
        self.assertEqual(finished["report"]["delivered_pulses"],0)
        with self.assertRaises(ValueError):d.before_forward()
        with self.assertRaises(ValueError):d.finish()
        YokeDriver.restore(d.snapshot(),RecipeV2Controller.restore(c.snapshot()))

    def test_midrun_finish_cancels_active_pulses_and_reports_unreached_future(self):
        p=effect();s=EffectScheduler([p]);events=[s.inject("joy")];s.advance_tokens(3);events.append(s.inject("joy"))
        c,d=recipient(make_schedule(events,presets=[p]))
        d.before_forward();c.advance(2)
        result=d.finish("budget_exhausted")
        self.assertEqual(result["report"]["unreached_source_pulses"],1)
        self.assertFalse(c.scheduler.pulses)
        self.assertEqual(c.counts["delivered"],1)
        RecipeV2Controller.restore(c.snapshot())

    def test_real_exposure_mismatch_and_missing_telemetry_remain_explicit(self):
        c,d=recipient(make_schedule());d.before_forward()
        d.record_event(token(0,value=.5,phase="reasoning"))
        d.record_event(dict(type="token",phase="output"))
        report=d.report()
        self.assertFalse(report["observed_coefficients_equal"])
        self.assertGreater(report["coefficient_l1_error"]["joy_orthogonal"],0)
        self.assertEqual(report["target_missing_measurement_events"],1)
        self.assertEqual(report["coverage"],"partial")
        c.advance()
        with self.assertRaisesRegex(ValueError,"before advancing"):d.record_event(token(0))
        with self.assertRaisesRegex(ValueError,"acknowledged"):d.record_event(token(1))

    def test_prefill_observations_do_not_advance_any_clock(self):
        c,d=recipient(make_schedule());d.before_forward()
        event=dict(type="prefill",positions=3,dose=dict(schema_version=2,effective=coefficients(),applied=False,
            generated_token_index=0,completed_decisions=0,phase="prefill",prefill_position="other"),
            measurements=dict(delivered_edit_norms=[0,0,0]))
        d.record_event(event)
        self.assertEqual(c.generated_tokens,0);self.assertEqual(c.actions,0)
        self.assertEqual(d.report()["by_phase"]["prefill_other"]["target_positions"],3)

    def test_restore_resumes_future_pulse_and_decays_without_replay(self):
        p=effect(shape="exponential");s=EffectScheduler([p]);events=[s.inject("joy")];s.advance_tokens(3);events.append(s.inject("joy"))
        c,d=recipient(make_schedule(events,presets=[p],horizon=6));d.before_forward()
        c.advance(2);d.before_forward();d.press(actor="human")
        state=d.snapshot();resumed_c=RecipeV2Controller.restore(c.snapshot());resumed=YokeDriver.restore(state,resumed_c)
        self.assertEqual(resumed.before_forward(),[])
        self.assertEqual(resumed.snapshot(),state)
        for original,driver in ((c,d),(resumed_c,resumed)):
            original.advance();event=driver.before_forward()[0];self.assertEqual(event["command"]["at"],3)
            self.assertEqual(original.level,1)
        self.assertEqual(c.snapshot(),resumed_c.snapshot());self.assertEqual(d.snapshot(),resumed.snapshot())

    def test_strict_restore_rejects_hash_shape_clock_and_receipt_mismatches(self):
        c,d=recipient(make_schedule());d.before_forward();d.press();state=d.snapshot()
        bad=deepcopy(state);bad["missing_measurement_events"]+=1
        with self.assertRaises(ValueError):YokeDriver.restore(bad,c)
        bad=deepcopy(state);bad["unknown"]=1
        with self.assertRaises(ValueError):YokeDriver.restore(reseal(bad),c)
        bad=deepcopy(state);bad["external_calls"]["human"]["aux_operation"]=1
        with self.assertRaisesRegex(ValueError,"counts"):YokeDriver.restore(reseal(bad),c)
        c.advance()
        with self.assertRaisesRegex(ValueError,"paired controller"):YokeDriver.restore(state,c)

    def test_source_presets_cannot_be_replaced_or_changed_and_live_controller_rejected(self):
        schedule=make_schedule();c,d=recipient(schedule)
        c.scheduler.presets["joy"]["gains"]["joy"]=.2
        with self.assertRaisesRegex(ValueError,"frozen source"):d.before_forward()
        wrong=RecipeV2Controller(dict(recipe_version=2))
        with self.assertRaises(ValueError):YokeDriver(wrong,schedule)
        c,_=recipient(schedule);c.advance()
        with self.assertRaisesRegex(ValueError,"fresh controller"):YokeDriver(c,schedule)


if __name__=="__main__":unittest.main()
