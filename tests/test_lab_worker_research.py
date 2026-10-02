"""Worker wiring for isolated diagnostics, durable yokes, task and runtime controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from lab.checkpoints import restore, validate
from lab.discovery import digest
from lab.effects import EffectScheduler
from lab.exposure import AXES, build_yoke
from lab.recipes_v2 import resolve_recipe
from lab.storage import atomic_json
from lab.worker import Worker
from lab.yoke_runner import YokeDriver
from tests.test_lab_discovery import spec
from tests.test_lab_exposure import source
from tests.test_lab_worker import wrapped


class ResearchRuntime:
    def __init__(self):self.calls=[];self.text=wrapped();self.tokens=2;self.after_token=None
    def generate(self,messages,tools,config,calibration_dir,control,emit,should_stop,**kwargs):
        self.calls.append(dict(messages=deepcopy(messages),tools=deepcopy(tools),config=deepcopy(config),kwargs=deepcopy(kwargs)))
        ids=[]
        for i in range(min(self.tokens,config["max_new_tokens"])):
            if should_stop():break
            state=control();pulse=state["phase_coefficients"]["output"];baseline=state["baseline_by_phase"]["output"]
            effective={**pulse,**{f"baseline_{key}":value for key,value in baseline.items()}}
            dose=dict(schema_version=2,phase="output",prefill_position=None,coefficients=pulse,baseline=baseline,effective=effective,
                generated_token_index=state["generated_tokens"],completed_decisions=state["completed_decisions"],applied=any(effective.values()))
            emit(dict(type="token",phase="output",token_id=i,text="x",dose=dose,measurements={"delivered_edit_norm":.5}))
            ids.append(i)
            if self.after_token:self.after_token(i)
        return dict(content=self.text,raw_text=self.text,reasoning="",token_ids=ids,truncated=False,finish_reason="eos")


class WorkerResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)
        self.events=[];self.runtime=ResearchRuntime();self.worker=Worker(self.path/"cache",self.runtime,lambda e:self.events.append(deepcopy(e)))
        fingerprint=dict(model_id="CPU/fixture",revision="pinned",template="fixture")
        self.worker.model_info=dict(model_id="CPU/fixture",fingerprint=fingerprint,fingerprint_sha256=digest(fingerprint),tool_call_format="json")
        self.cal=self.path/"cal";self.cal.mkdir();atomic_json(self.cal/"calibration.json",{"fixture":True});(self.cal/"vectors.npz").write_bytes(b"CPU fixture identity only")
        self.cfg=resolve_recipe(dict(recipe_version=2,two_buttons=True,counterbalance=False,demonstration="none",task_count=1,action_budget=2,token_budget=32,turn_token_limit=8))

    def start(self,**extra):
        payload=dict(run_id="source",mode="experiment",config=self.cfg,calibration_id="cal",calibration_dir=str(self.cal),out_dir=str(self.path/"source"),**extra)
        self.worker.start_session(payload)
        return payload

    def schedule(self):
        s=EffectScheduler(self.cfg["effect_presets"]);events=[s.inject("opium")]
        for index in range(4):
            pulse=s.coefficients("output")["coefficients"]
            effective={**pulse,**{f"baseline_{key}":0. for key in ("pain","joy_raw","joy_orthogonal")}}
            events.append(dict(type="token",generation_index=index,dose=dict(schema_version=2,phase="output",effective=effective,
                generated_token_index=index,completed_decisions=index//2,applied=any(effective.values())),measurements={"delivered_edit_norm":.5}))
            s.advance_tokens(1)
        encoded,meta=source(events,end_tokens=4,end_decisions=2)
        return build_yoke(encoded,self.cfg["effect_presets"],clock="tokens",source=meta,pair_id="pair")

    def test_yoked_choices_have_no_extra_effect_and_last_checkpoint_remains_resumable(self):
        self.start(yoke_schedule=self.schedule())
        self.worker.run_experiment()
        summary=[e["summary"] for e in self.events if e["type"]=="session_finished"][-1]
        self.assertEqual((summary["voluntary_calls"],summary["completed_decisions"],summary["tokens"]),(2,2,4))
        self.assertEqual(summary["yoke"]["delivered_pulses"],1)
        self.assertEqual(summary["yoke"]["own_button_calls"],{"aux_operation":2})
        self.assertEqual(summary["yoke"]["own_button_additional_effects"],0)
        self.assertEqual(summary["yoke"]["coverage"],"observed_source_covered")
        self.assertTrue(summary["yoke"]["observed_coefficients_equal"])
        calls=[e for e in self.events if e["type"]=="tool" and e.get("actor")=="model"]
        self.assertTrue(all(not e["intervention"]["delivered"] for e in calls))
        saved=json.loads((self.path/"source"/"checkpoint.json").read_text());validate(saved)
        session,effect,_,_=restore(saved)
        driver=YokeDriver.restore(session["yoke_state"],effect)
        self.assertIsNone(driver.cursor.finished)
        self.assertEqual(driver.index,4)
        terminal=json.loads((self.path/"source"/"yoke-final.json").read_text())
        self.assertIsNotNone(terminal["cursor"]["finished"])
        self.assertEqual(len([e for e in self.events if e["type"]=="yoke_finished"]),1)

    def test_yoke_manual_and_demo_presses_cannot_inject_extra_active_pulses(self):
        self.start(yoke_schedule=self.schedule())
        self.worker.receive({"command":"inject","payload":{"tool":"aux_operation"}})
        self.worker._demonstrate_v2("aux_alternative",{},"manual-demo")
        self.assertEqual(self.worker.effect.counts["delivered"],0)
        self.assertEqual(self.worker.effect.counts["sham"],2)
        self.assertEqual(self.worker.yoke.report()["external_button_calls"]["human"],{"aux_operation":1})
        validate(json.loads((self.path/"source"/"checkpoint.json").read_text()))

    def test_task_axes_reach_actual_environment_and_checkpoint(self):
        self.start(task_config={"difficulty":"hard","framing":"deadline"})
        self.assertIn("candidates",self.worker.environment.records[0])
        self.assertEqual(self.worker.session["task_config"]["framing"],"deadline")
        saved=json.loads((self.path/"source"/"checkpoint.json").read_text())
        session,_,_,env=restore(saved)
        self.assertEqual(session["task_config"],self.worker.session["task_config"])
        self.assertEqual(env.records,self.worker.environment.records)

    def test_runtime_control_envelope_is_validated_forwarded_and_saved(self):
        controls={"random_norm_match":{"target_preset_id":"opium","reference":"same_unedited_position","relative_tolerance":.01,"absolute_tolerance":1e-6}}
        self.start(runtime_controls=controls)
        self.worker.run_experiment()
        self.assertEqual(self.runtime.calls[0]["kwargs"]["runtime_controls"],controls)
        saved=json.loads((self.path/"source"/"checkpoint.json").read_text())
        self.assertEqual(validate(saved)["session"]["runtime_controls"],controls)
        with self.assertRaises(ValueError):self.start(runtime_controls={"arbitrary_callback":"eval"})

    def test_diagnostics_preserve_finished_parent_and_child_events_have_own_identity(self):
        self.start();self.worker.finish("stopped","test_boundary")
        saved=json.loads((self.path/"source"/"checkpoint.json").read_text())
        before=deepcopy(self.worker.session);effect=deepcopy(self.worker.effect.snapshot());budget=deepcopy(self.worker.budget.snapshot())
        self.runtime.text='{"choice":"aux_operation","confidence":0.8}'
        result=self.worker.run_diagnostics(dict(run_id="diagnostic",checkpoint=saved,spec=spec(2),answer_key={f"context-{i}":{"answer":"aux_operation","control":"active_sham"} for i in range(2)},
            pair_id="pair",calibration_id="cal",calibration_dir=str(self.cal),out_dir=str(self.path/"diagnostic"),token_budget=8))
        self.assertEqual(result["scoring"]["by_arm"]["naive"]["correct"],2)
        self.assertEqual(self.worker.session,before);self.assertEqual(self.worker.effect.snapshot(),effect);self.assertEqual(self.worker.budget.snapshot(),budget)
        self.assertEqual(len([e for e in self.events if e["type"]=="session_finished" and e.get("run_id")=="diagnostic"]),1)
        self.assertTrue(all(not call["tools"] for call in self.runtime.calls))
        self.assertTrue(all(call["messages"][:-2]==saved["session"]["messages"] for call in self.runtime.calls))
        self.assertEqual(self.runtime.calls[0]["messages"][:-2],self.runtime.calls[1]["messages"][:-2])

    def test_full_visible_history_transfer_resets_new_task_budget_and_effects(self):
        self.start()
        self.runtime.text = wrapped("submit_answer", {"answer": self.worker.environment.records[0]["answer"]})
        self.worker.run_experiment()
        saved=json.loads((self.path/"source"/"checkpoint.json").read_text())
        original=(self.path/"source"/"checkpoint.json").read_bytes()
        target_cfg=resolve_recipe(dict(self.cfg,seed=self.cfg["seed"]+1))
        payload=dict(run_id="target",mode="experiment",config=target_cfg,calibration_id="cal",calibration_dir=str(self.cal),
                     out_dir=str(self.path/"target"),history_source_checkpoint=saved)
        self.worker.start_session(payload)
        self.assertEqual(self.worker.session["messages"][:len(saved["session"]["messages"])],saved["session"]["messages"])
        self.assertNotIn("history_source_checkpoint",self.worker.session)
        self.assertEqual((self.worker.environment.index,self.worker.budget.actions,self.worker.budget.tokens,self.worker.effect.counts["total"]),(0,0,0,0))
        target=json.loads((self.path/"target"/"checkpoint.json").read_text());validate(target)
        self.runtime.text=wrapped("submit_answer",{"answer":self.worker.environment.records[0]["answer"]})
        self.worker.run_experiment()
        target=json.loads((self.path/"target"/"checkpoint.json").read_text())
        state,_,budget,env=restore(target)
        self.assertEqual((budget.actions,env.index,env.metrics()["correct"]),(1,1,1))
        self.assertEqual((self.path/"source"/"checkpoint.json").read_bytes(),original)
        with self.assertRaisesRegex(ValueError,"supplied tool history"):
            self.worker.start_session(dict(payload,history_prefix=state["history_prefix"]))

    def test_diagnostic_model_identity_mismatch_fails_before_child_generation(self):
        self.start();self.worker.finish("stopped","test_boundary")
        saved=json.loads((self.path/"source"/"checkpoint.json").read_text())
        self.worker.model_info=dict(model_id="different")
        with self.assertRaisesRegex(ValueError,"fingerprint"):
            self.worker.run_diagnostics(dict(checkpoint=saved,calibration_dir=str(self.cal)))
        self.assertFalse(self.runtime.calls)


if __name__=="__main__":unittest.main()
