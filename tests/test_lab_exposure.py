"""Delivered-only yokes, exact clocks, bounded coverage and strict cursor replay."""
from copy import deepcopy
import hashlib
import json
import unittest

from lab.controller_v2 import RecipeV2Controller
from lab.effects import EffectScheduler, content_hash, validate_preset
from lab.exposure import AXES, YokedExposure, build_yoke, exposure_from_event, validate_yoke


def preset():
    return validate_preset(dict(id="joy",gains={"joy":1},decay={"shape":"constant"}))


def coefficients(**changes):
    return {axis:float(changes.get(axis,0)) for axis in AXES}


def token(index,decision=0,phase="output",value=1,norm=.5):
    dose=dict(schema_version=2,effective=coefficients(joy_orthogonal=value),applied=bool(value),
              generated_token_index=index,completed_decisions=decision,phase=phase)
    result=dict(type="token",dose=dose,generation_index=index)
    if norm is not None:result["measurements"]={"delivered_edit_norm":norm}
    return result


def source(events,end_tokens=4,end_decisions=2,prefix_count=None,raw=False):
    count=len(events) if prefix_count is None else prefix_count
    encoded=("".join(json.dumps(event)+"\n" for event in events)).encode() if raw else events
    full=encoded if raw else json.dumps(events,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    prefix=b"".join(full.splitlines(keepends=True)[:count]) if raw else json.dumps(events[:count],sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    metadata=dict(run_id="source",events_sha256=hashlib.sha256(full).hexdigest(),prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        prefix_event_count=count,config_sha256="a"*64,model_fingerprint_sha256="b"*64,end_tokens=end_tokens,end_decisions=end_decisions)
    return encoded,metadata


def build(events=None,clock="tokens",end_tokens=4,end_decisions=2,raw=False,presets=None,prefix_count=None):
    if events is None:
        scheduler=EffectScheduler([preset()]);events=[scheduler.inject("joy")]+[token(i,i//2) for i in range(4)]
    evidence,metadata=source(events,end_tokens,end_decisions,prefix_count,raw)
    return build_yoke(evidence,[preset()] if presets is None else presets,clock=clock,source=metadata,pair_id="pair"),evidence


def reseal(value):
    value=deepcopy(value);value.pop("sha256",None);value["sha256"]=content_hash(value);return value


def acknowledge_all(cursor,index):
    issued=cursor.due(index)
    for command in issued:cursor.acknowledge(command["id"],delivered=True)
    return issued


class ExposureTests(unittest.TestCase):
    def test_only_delivered_real_v2_calls_schedule_and_coverage_retains_sham_disabled(self):
        controller=RecipeV2Controller(dict(recipe_version=2,two_buttons=True,counterbalance=False))
        active=controller.press(tool="aux_operation")
        sham=controller.press(tool="aux_alternative")
        controller.set_controls(enabled=False)
        disabled=controller.press(tool="aux_operation")
        events=[dict(type="tool",intervention=event) for event in (active,sham,disabled)]
        schedule,_=build(events,presets=controller.recipe["effect_presets"])
        self.assertEqual([c["kind"] for c in schedule["commands"]],["inject","end"])
        self.assertEqual([a["delivery_status"] for a in schedule["delivery_audit"]],["delivered","sham","disabled"])
        report=YokedExposure(schedule).report()
        self.assertEqual(report["source_requested_pulses"],1)
        self.assertEqual(report["source_undelivered_attempts"],2)
        self.assertEqual(report["coverage"],"partial")

    def test_duplicate_wrapped_pulses_are_not_reinjected_or_double_counted(self):
        controller=RecipeV2Controller(dict(recipe_version=2))
        event=controller.press()
        schedule,_=build([event["intervention"],dict(type="tool",intervention=event)],presets=controller.recipe["effect_presets"])
        self.assertEqual(len(schedule["delivery_audit"]),1)
        self.assertEqual(len(schedule["commands"]),2)
        bad=deepcopy(event["intervention"]);bad["generated_tokens"]=1
        with self.assertRaisesRegex(ValueError,"identity changed"):
            build([event["intervention"],bad],presets=controller.recipe["effect_presets"])

    def test_disabled_and_sham_scheduler_attempts_remain_undelivered(self):
        s=EffectScheduler([preset()]);events=[s.inject("joy",active=False)]
        s.set_enabled(False);events.append(s.inject("joy"))
        schedule,_=build(events)
        self.assertEqual(len(schedule["commands"]),1)
        self.assertEqual(len(schedule["delivery_audit"]),2)
        self.assertTrue(all(not row["scheduled"] for row in schedule["delivery_audit"]))

    def test_clock_is_explicit_and_ignores_weighted_action_units(self):
        s=EffectScheduler([preset()]);s.advance_tokens(3);s.complete_decision(1)
        event=s.inject("joy");event["actions"]=999
        by_tokens,_=build([event],clock="tokens")
        by_decisions,_=build([event],clock="decisions")
        self.assertEqual(by_tokens["commands"][0]["at"],3)
        self.assertEqual(by_decisions["commands"][0]["at"],1)
        del event["completed_decisions"]
        with self.assertRaises(ValueError):build([event],clock="decisions")
        with self.assertRaises(ValueError):build([event],clock="actions")

    def test_raw_bytes_and_prefix_provenance_tampering(self):
        s=EffectScheduler([preset()]);events=[s.inject("joy"),token(0),dict(type="outside-prefix")]
        schedule,evidence=build(events,end_tokens=1,end_decisions=1,raw=True,prefix_count=2)
        self.assertEqual(schedule["source"]["hash_encoding"],"raw-jsonl")
        self.assertEqual(validate_yoke(schedule,source_events=evidence),schedule)
        changed=deepcopy(schedule);changed["commands"][0]["at"]=1;changed=reseal(changed)
        validate_yoke(changed) # A self-consistent hash is integrity, not authenticity.
        with self.assertRaisesRegex(ValueError,"verified source"):
            validate_yoke(changed,source_events=evidence)
        with self.assertRaises(ValueError):validate_yoke(schedule,source_events=evidence+b" ")
        data,meta=source(events);meta["prefix_sha256"]="c"*64
        with self.assertRaisesRegex(ValueError,"prefix hash"):
            build_yoke(data,[preset()],clock="tokens",source=meta,pair_id="p")
        with self.assertRaises(ValueError):build_yoke(b'{"type":"x","type":"y"}\n',[preset()],clock="tokens",source=meta,pair_id="p")

    def test_preset_hash_and_source_delivery_identity_fail_closed(self):
        s=EffectScheduler([preset()]);event=s.inject("joy")
        for change in (lambda e:e.update(preset_hash="c"*64),lambda e:e.update(generated_tokens=9),lambda e:e.update(run_id="other"),lambda e:e.update(delivered="true")):
            bad=deepcopy(event);change(bad)
            with self.subTest(bad=bad),self.assertRaises(ValueError):build([bad])
        with self.assertRaises(ValueError):build([dict(type="aux_call",delivered=True,actor="model")])
        s,_=build();bad=deepcopy(s);bad["presets"][0]["gains"]["joy"]=.5
        with self.assertRaises(ValueError):validate_yoke(reseal(bad))

    def test_delivery_is_once_acknowledged_and_safe_checkpoint_restores(self):
        schedule,_=build();cursor=YokedExposure(schedule)
        issued=cursor.due(0);self.assertEqual(len(issued),1);self.assertEqual(cursor.due(0),[])
        with self.assertRaises(ValueError):cursor.snapshot()
        with self.assertRaises(ValueError):cursor.due(1)
        with self.assertRaises(ValueError):cursor.acknowledge(issued[0]["id"],delivered=True,actual_index=1)
        cursor.acknowledge(issued[0]["id"],delivered=True)
        with self.assertRaises(ValueError):cursor.acknowledge(issued[0]["id"],delivered=True)
        cursor.record_exposure(0,"output",coefficients(joy_orthogonal=1),delivered_edit_norm_sum=.5)
        restored=YokedExposure.restore(json.loads(json.dumps(cursor.snapshot())))
        self.assertEqual(restored.snapshot(),cursor.snapshot())
        self.assertEqual(restored.due(0),[])
        restored.recipient_call("aux_operation")
        self.assertEqual(cursor.own_calls,{})

    def test_target_choices_authorize_no_extra_effect_and_horizon_cancels(self):
        schedule,_=build();cursor=YokedExposure(schedule)
        for i in range(4):
            acknowledge_all(cursor,i)
            cursor.record_exposure(i,"output",coefficients(joy_orthogonal=1),delivered_edit_norm_sum=.5)
            choice=cursor.recipient_call("aux_operation")
            self.assertFalse(choice["additional_active_effect"])
            self.assertFalse(choice["requested_active"])
        ending=acknowledge_all(cursor,4)
        self.assertEqual([c["kind"] for c in ending],["end"])
        result=cursor.record_exposure(4,"output",coefficients(),delivered_edit_norm_sum=0)
        self.assertFalse(result["covered"])
        report=cursor.finish(5)
        self.assertEqual(report["coverage"],"observed_source_covered")
        self.assertTrue(report["schedule_complete"])
        self.assertTrue(report["observed_coefficients_equal"])
        self.assertEqual(report["outside_source_positions"],1)
        self.assertEqual(report["own_button_calls"],{"aux_operation":4})
        self.assertEqual(report["own_button_additional_effects"],0)
        self.assertEqual(report["coefficient_l1_error"]["joy_orthogonal"],0)
        self.assertEqual(report["by_phase"]["output"]["source_delivered_edit_norm_sum"],2)
        self.assertEqual(YokedExposure.restore(cursor.snapshot()).report(),report)
        with self.assertRaises(ValueError):cursor.due(5)

    def test_late_source_pulse_is_skipped_not_shifted_and_late_end_still_cancels(self):
        schedule,_=build();cursor=YokedExposure(schedule)
        self.assertEqual(cursor.due(1),[])
        self.assertEqual(cursor.receipts[0]["reason"],"missed_source_index")
        self.assertEqual(cursor.report()["failed_or_missed_pulses"],1)
        self.assertEqual(cursor.report()["absolute_index_error"],1)
        self.assertEqual([c["kind"] for c in acknowledge_all(cursor,8)],["end"])
        report=cursor.finish(8)
        self.assertEqual(report["delivered_pulses"],0)
        self.assertEqual(report["coverage"],"partial")
        YokedExposure.restore(cursor.snapshot())

    def test_early_finish_and_phase_difference_cannot_claim_matching_exposure(self):
        schedule,_=build();cursor=YokedExposure(schedule)
        acknowledge_all(cursor,0)
        cursor.record_exposure(0,"reasoning",coefficients(joy_orthogonal=.5),delivered_edit_norm_sum=.25)
        report=cursor.finish(1)
        self.assertEqual(report["coverage"],"partial")
        self.assertFalse(report["observed_coefficients_equal"])
        self.assertTrue(report["terminated_before_source_horizon"])
        self.assertEqual(report["uncovered_source_positions"],4)
        self.assertEqual(report["excess_target_positions"],1)
        self.assertEqual(report["coefficient_l1_error"]["joy_orthogonal"],4.5)
        self.assertEqual(report["uncovered_source_intervals"],[dict(start=1,end_exclusive=4)])

    def test_source_cancellations_are_preserved_and_baseline_controls_marked_omitted(self):
        s=EffectScheduler([preset()]);events=[s.inject("joy")];s.advance_tokens(1)
        cancellation=s.cancel(channel="auxiliary")
        events += [dict(type="control",intervention=cancellation),dict(type="token",dose={})]
        schedule,_=build(events)
        self.assertEqual([row["kind"] for row in schedule["commands"]],["inject","cancel","end"])
        self.assertEqual(schedule["source_missing_measurement_events"],1)
        self.assertEqual(len(schedule["omitted_controls"]),1)
        self.assertEqual(schedule["source_observation_gaps"],[dict(start=0,end_exclusive=4)])

    def test_prefill_measurements_are_separate_from_generated_phase_and_norms_optional(self):
        event=token(0);event.update(type="prefill",positions=2,measurements={"delivered_edit_norms":[.2,.3]})
        event["dose"].update(phase="prefill",prefill_position="other")
        sample=exposure_from_event(event,clock="tokens")
        self.assertEqual((sample["phase"],sample["positions"],sample["delivered_edit_norm_sum"]),("prefill_other",2,.5))
        schedule,_=build([event,token(0,norm=None)],end_tokens=1,end_decisions=1)
        rows={row["phase"]:row for row in schedule["source_exposure_bins"]}
        self.assertEqual(rows["prefill_other"]["cumulative_coefficients"]["joy_orthogonal"],2)
        self.assertEqual(rows["output"]["measured_norm_positions"],0)
        self.assertEqual(rows["output"]["delivered_edit_norm_sum"],0)
        event["positions"]=3
        with self.assertRaises(ValueError):exposure_from_event(event,clock="tokens")
        event=token(0);event["generation_index"]=1
        with self.assertRaises(ValueError):exposure_from_event(event,clock="tokens")

    def test_decision_matching_aggregates_actual_token_counts_not_equal_effect_assumption(self):
        schedule,_=build(clock="decisions")
        cursor=YokedExposure(schedule)
        for decision in range(2):
            acknowledge_all(cursor,decision)
            cursor.record_exposure(decision,"output",coefficients(joy_orthogonal=1),positions=1)
        acknowledge_all(cursor,2);report=cursor.finish(2)
        self.assertTrue(report["schedule_complete"])
        self.assertEqual(report["delivered_pulses"],1)
        self.assertEqual(report["uncovered_source_positions"],2)
        self.assertEqual(report["coefficient_l1_error"]["joy_orthogonal"],2)
        self.assertEqual(report["coverage"],"partial")

    def test_rehashed_impossible_cursor_and_schedule_are_rejected(self):
        schedule,_=build();cursor=YokedExposure(schedule);acknowledge_all(cursor,0)
        cursor.record_exposure(0,"output",coefficients(joy_orthogonal=1))
        snapshot=cursor.snapshot()
        edits=[lambda x:x.update(cursor=0),lambda x:x.update(last_index=-1),lambda x:x["receipts"][0].update(actual_index=2,index_error=2),
            lambda x:x["receipts"][0].update(delivered=1),lambda x:x.update(own_calls={"aux_operation":-1}),
            lambda x:x["exposure_bins"][0]["cumulative_coefficients"].update(joy_orthogonal=5),lambda x:x.update(schema_version=999)]
        for edit in edits:
            bad=deepcopy(snapshot);edit(bad)
            with self.subTest(edit=edit),self.assertRaises(ValueError):YokedExposure.restore(reseal(bad))
        for edit in (lambda x:x.update(recipient_button_effect="active"),lambda x:x["delivery_audit"][0].update(delivered=False),
                     lambda x:x["commands"][-1].update(source_event_sha256="c"*64),lambda x:x.update(source_observation_gaps=[])):
            bad=deepcopy(schedule);edit(bad)
            if bad==schedule:continue
            with self.subTest(edit=edit),self.assertRaises(ValueError):validate_yoke(reseal(bad))


if __name__ == "__main__":unittest.main()
