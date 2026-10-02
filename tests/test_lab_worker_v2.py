"""V2 end-to-end worker decisions, checkpoints and model-visible previews."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest

from lab.checkpoints import restore, validate
from lab.effects import COEFFICIENT_AXES
from lab.recipes_v2 import resolve_recipe
from lab.session_v2 import build_session, parse_session_response
from lab.tool_definitions import format_call
from lab.worker import Worker
from tests.test_lab_worker import FakeRuntime, wrapped


class WorkerV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.events = []
        self.runtime = FakeRuntime()
        self.worker = Worker(str(self.path / "cache"), runtime=self.runtime, output=lambda e: self.events.append(copy.deepcopy(e)))
        self.worker.model_info = {"model_id": "Qwen/fake", "device": "cuda", "tool_call_format": "json"}

    def start(self, mode="experiment", **overrides):
        cfg = resolve_recipe({"recipe_version": 2, "demonstration": "none", "task_count": 1,
                              "action_budget": 12, "token_budget": 128, "turn_token_limit": 32, **overrides})
        self.worker.start_session({"run_id": "run-v2", "mode": mode, "config": cfg, "calibration_id": "cal-v2",
             "calibration_dir": str(self.path / "cal-v2"), "out_dir": str(self.path / "run-v2")})
        return cfg

    def finished(self):
        return [e for e in self.events if e["type"] == "session_finished"][-1]

    def checkpoint(self):
        return json.loads((self.path / "run-v2" / "checkpoint.json").read_text())

    def test_custom_costly_aux_and_real_task_grade_checkpoint(self):
        cfg = self.start(base_decision_cost=2, task_tool_costs={"submit_answer": 1},
                         auxiliary_tools=[{"id": "button", "name": "neutral_button", "cost": 3}])
        record = self.worker.environment.records[0]
        self.runtime.scripts = [{"raw_text": wrapped("neutral_button")}, {"raw_text": wrapped("submit_answer", {"answer": record["answer"]})}]
        self.worker.run_experiment()
        summary = self.finished()["summary"]
        self.assertEqual((summary["correct"], summary["voluntary_calls"], summary["completed_decisions"], summary["actions"]), (1, 1, 2, 8))
        saved = validate(self.checkpoint())
        self.assertEqual(saved["effect"]["recorded_decisions"], 2)
        self.assertEqual(saved["budget"]["receipts"][0]["extra_charge"], 3)
        system = self.runtime.calls[0]["messages"][0]["content"]
        self.assertIn("Every generation attempt costs 2", system)
        self.assertNotIn("Every assistant turn uses one action", system)
        self.assertEqual(self.runtime.calls[0]["params"]["seed"], cfg["rng_seeds"]["generation"])

    def test_unaffordable_aux_and_task_calls_never_mutate_task_or_effect(self):
        self.start(action_budget=3, auxiliary_tools=[{"id": "expensive", "name": "expensive", "cost": 10}], task_tool_costs={"submit_answer": 10})
        record = self.worker.environment.records[0]
        self.runtime.scripts = [{"raw_text": wrapped("expensive")}, {"raw_text": wrapped("submit_answer", {"answer": record["answer"]})}, {"raw_text": "invalid"}]
        self.worker.run_experiment()
        summary = self.finished()["summary"]
        self.assertEqual((summary["voluntary_calls"], summary["submitted"], summary["cost_denials"], summary["aux_attempts"], summary["aux_cost_denials"]), (0, 0, 2, 1, 1))
        self.assertEqual(self.worker.environment.work_calls, 0)
        self.assertEqual(summary["actions"], 3)
        validate(self.checkpoint())

    def test_both_grammars_dispatch_custom_arguments(self):
        for grammar in ("json", "qwen_xml"):
            with self.subTest(grammar=grammar):
                if self.worker.session:
                    self.worker.finish("stopped", "test_next")
                self.worker.model_info["tool_call_format"] = grammar
                cfg = self.start(action_budget=2, auxiliary_tools=[{"id": "custom", "name": "neutral", "parameters": {
                    "type": "object", "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 2}}, "required": ["n"]}}])
                tool = cfg["auxiliary_tools"][0]
                aux = format_call("neutral", {"n": 2}, tool["parameters"], tool_call_format=grammar)
                record = self.worker.environment.records[0]
                submit_schema = next(t["function"]["parameters"] for t in self.worker.environment.tools if t["function"]["name"] == "submit_answer")
                task = format_call("submit_answer", {"answer": record["answer"]}, submit_schema, tool_call_format=grammar)
                self.runtime.scripts = [{"raw_text": aux}, {"raw_text": task}]
                self.worker.run_experiment()
                self.assertEqual(self.finished()["summary"]["correct"], 1)
                self.assertEqual(self.finished()["summary"]["voluntary_calls"], 1)
                validate(self.checkpoint())

    def test_custom_schema_on_legacy_aux_name_cannot_bypass_validation(self):
        cfg = self.start(action_budget=1, auxiliary_tools=[{"id": "aux", "name": "aux_operation", "parameters": {
            "type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}])
        self.runtime.scripts = [{"raw_text": wrapped()}]
        self.worker.run_experiment()
        self.assertEqual(self.finished()["summary"]["voluntary_calls"], 0)
        self.assertEqual(self.worker.budget.receipts[0]["status"], "invalid")
        validate(self.checkpoint())

    def test_reasoning_tool_mentions_remain_inert_and_truncation_spends_base(self):
        self.start(thinking=True, action_budget=4, base_decision_cost=2)
        expected = self.worker.environment.records[0]["answer"]
        self.runtime.scripts = [{"raw_text": "<think>" + wrapped(), "truncated": True},
                               {"raw_text": "<think>Consider " + wrapped() + "</think>" + wrapped("submit_answer", {"answer": expected})}]
        self.worker.run_experiment()
        self.assertEqual(self.finished()["summary"]["correct"], 1)
        self.assertEqual(self.finished()["summary"]["voluntary_calls"], 0)
        self.assertEqual([r["status"] for r in self.worker.budget.receipts], ["truncated", "valid"])
        self.assertEqual(self.worker.budget.actions, 4)
        validate(self.checkpoint())

    def test_initial_demonstration_and_preview_have_exact_first_visible_context(self):
        cfg = self.start(demonstration="initial", action_budget=1, budget_visibility="per_decision")
        expected = build_session(cfg, "experiment", "json", include_task=True)
        self.runtime.scripts = [{"raw_text": wrapped()}]
        self.worker.run_experiment()
        self.assertEqual(self.runtime.calls[0]["messages"], expected["messages"])
        self.assertEqual(self.runtime.calls[0]["tools"], expected["tools"])
        self.assertEqual(self.worker.effect.counts["demonstration"], 1)
        self.assertEqual(len(self.worker.session["external_messages"]), 1)
        validate(self.checkpoint())

    def test_zero_boundary_demo_is_in_exact_first_prompt_preview(self):
        cfg = self.start(demonstration="at_decisions", demonstration_decisions=[0], action_budget=1)
        preview = build_session(cfg, "experiment", "json", include_task=True)
        self.runtime.scripts = [{"raw_text": wrapped()}]
        self.worker.run_experiment()
        self.assertEqual(self.runtime.calls[0]["messages"], preview["messages"])
        self.assertEqual(self.worker.effect.counts["demonstration"], 1)

    def test_conversation_experiment_accepts_text_and_finishes_honestly(self):
        self.start(task_family="conversation")
        self.runtime.scripts = [{"raw_text": "A conversational reply."}]
        self.worker.run_experiment()
        self.assertEqual(self.finished()["summary"]["termination"], "conversation_complete")
        self.assertEqual(self.worker.budget.completed_decisions, 1)
        self.assertFalse(self.worker.budget.exhausted)
        validate(self.checkpoint())

    def test_manual_midturn_injection_keeps_arguments_and_next_turn_provenance(self):
        cfg = self.start(auxiliary_tools=[{"id": "custom", "name": "neutral", "parameters": {"type": "object", "properties": {"tag": {"type": "string"}}, "required": ["tag"]}}])
        record = self.worker.environment.records[0]
        self.runtime.scripts = [{"raw_text": wrapped("read_order", {"order_id": record["id"]})}, {"raw_text": wrapped("submit_answer", {"answer": record["answer"]})}]
        def inject(index):
            if index == 0:
                self.worker.receive({"command": "inject", "id": "human", "payload": {"tool": "neutral", "arguments": {"tag": "manual"}}})
                self.runtime.after_token = None
        self.runtime.after_token = inject
        self.worker.run_experiment()
        self.assertFalse(any(m.get("name") == "neutral" for m in self.runtime.calls[0]["messages"]))
        external = self.worker.session["external_messages"][0]
        assistant = self.worker.session["messages"][external["index"]]
        self.assertEqual(assistant["tool_calls"][0]["function"]["arguments"], {"tag": "manual"})
        self.assertEqual(external["actor"], "human")
        self.assertEqual((self.worker.effect.counts["human"], self.worker.effect.counts["model"]), (1, 0))
        validate(self.checkpoint())

    def test_control_revision_and_unsupported_manual_mutation(self):
        self.start("chat")
        self.worker.receive({"command": "control", "id": "c", "payload": {"pain": 2, "joy_suppression": .5}})
        self.assertEqual(self.worker.session["control_revision"], 1)
        before = self.worker.effect.snapshot()
        self.worker.receive({"command": "inject", "id": "bad", "payload": {"half_life_tokens": 4}})
        self.assertEqual(self.worker.effect.snapshot(), before)
        self.assertTrue(any(e.get("command_id") == "bad" and e["type"] == "error" for e in self.events))
        validate(self.checkpoint())

    def test_runtime_token_and_prefill_exposure_record_actual_stages_without_extra_age(self):
        self.start("chat")
        self.worker.budget.begin_decision()
        c = {axis: 0. for axis in COEFFICIENT_AXES}
        c["joy_raw"] = .25
        prefill_dose = {"schema_version": 2, "coefficients": c, "baseline": {"pain": 0., "joy_raw": 0., "joy_orthogonal": 0.}}
        self.worker.token_event({"type": "prefill", "phase": "prefill", "positions": 7, "prefill_position": "other", "dose": prefill_dose})
        self.assertEqual((self.worker.budget.tokens, self.worker.effect.generated_tokens), (0, 0))
        output_dose = copy.deepcopy(prefill_dose)
        output_dose["baseline"]["pain"] = 2
        self.worker.token_event({"type": "token", "phase": "reasoning", "text": "x", "is_eos": True, "dose": output_dose})
        self.assertEqual((self.worker.budget.tokens, self.worker.effect.generated_tokens), (1, 1))
        self.assertEqual(self.worker.effect.scheduler.exposure["prefill"]["joy_raw"], 1.75)
        self.assertEqual(self.worker.effect.baseline_exposure["reasoning"]["pain"], 2)
        self.assertEqual(self.worker.effect.scheduler.exposure["reasoning"]["positions"], 1)

    def test_error_and_stopped_generation_preserve_base_charge_without_dispatch(self):
        self.start("chat", base_decision_cost=2)
        def fail(*args, **kwargs):
            kwargs = kwargs
            raise RuntimeError("inference failure")
        self.runtime.generate = fail
        with self.assertRaises(RuntimeError):
            self.worker.chat("hello")
        self.assertEqual((self.worker.budget.actions, self.worker.budget.completed_decisions), (2, 1))
        self.assertEqual(self.worker.budget.receipts[0]["status"], "error")
        self.assertIsNone(self.worker.budget.in_flight)
        self.assertFalse(self.worker.session["boundary_complete"])
        self.assertEqual(self.worker.effect.recorded_decisions, 1)

    def test_stop_on_final_token_prevents_aux_and_charges_attempt(self):
        self.start(base_decision_cost=2)
        self.runtime.scripts = [{"raw_text": wrapped(), "tokens": 1}]
        self.runtime.after_token = lambda _: self.worker.receive({"command": "stop"})
        self.worker.active = True
        self.worker.run_experiment()
        self.assertEqual(self.worker.effect.counts["model"], 0)
        self.assertEqual(self.worker.budget.receipts[0]["status"], "stopped")
        self.assertEqual(self.worker.budget.actions, 2)
        self.assertFalse(self.worker.session["boundary_complete"])

    def test_decision_clock_transition_does_not_use_spent_cost_units(self):
        cfg = self.start(condition="joy_to_pain", conditions=["joy_to_pain"], phase_actions=[1, 2], base_decision_cost=2, action_budget=4)
        for preset in cfg["effect_presets"]:
            preset["decay"]["clock"] = "decisions"
        # Start a detached new resolved session with the changed schedule.
        self.worker.finish("stopped", "reconfigure")
        payload = {"run_id": "run-v2", "mode": "experiment", "config": cfg, "calibration_id": "cal-v2", "calibration_dir": str(self.path / "cal-v2"), "out_dir": str(self.path / "run-v2")}
        self.worker.start_session(payload)
        self.runtime.scripts = [{"raw_text": wrapped()}, {"raw_text": wrapped()}]
        self.worker.run_experiment()
        calls = [e["intervention"] for e in self.events if e.get("type") == "tool" and e.get("actor") == "model"]
        self.assertEqual([e["requested_preset_id"] for e in calls], ["joy", "pain"])
        self.assertEqual(self.worker.effect.actions, 2)
        self.assertEqual(self.worker.budget.actions, 4)
        self.assertEqual(self.worker.session["control_revision"], 1)
        validate(self.checkpoint())

    def test_pause_resume_v2_matches_uninterrupted_state(self):
        self.start(action_budget=8)
        record = self.worker.environment.records[0]
        scripts = [{"raw_text": wrapped("read_order", {"order_id": record["id"]})}, {"raw_text": wrapped("submit_answer", {"answer": record["answer"]})}]
        self.runtime.scripts = copy.deepcopy(scripts)
        requested = threading.Event()
        def pause(_):
            if not requested.is_set():
                requested.set()
                self.worker.receive({"command": "pause"})
        self.runtime.after_token = pause
        self.worker.active = True
        self.worker.active_command = "start_session"
        thread = threading.Thread(target=self.worker.run_experiment, daemon=True)
        thread.start()
        self.addCleanup(lambda: (self.worker.receive({"command": "stop"}), thread.join(2)))
        import time
        for _ in range(100):
            if self.worker.paused:
                break
            time.sleep(.01)
        self.assertTrue(self.worker.paused)
        paused = self.worker.budget.snapshot()
        validate(self.checkpoint())
        self.worker.receive({"command": "pause"})
        self.assertEqual(self.worker.budget.snapshot(), paused)
        self.worker.receive({"command": "resume"})
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.finished()["summary"]["correct"], 1)
        self.assertEqual(self.worker.budget.completed_decisions, 2)
        validate(self.checkpoint())

    def test_actual_template_preview_does_not_generate_or_change_session(self):
        cfg = self.start("chat")
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return json.dumps({"messages": messages, **kwargs}, sort_keys=True)
        self.runtime.tokenizer = Tokenizer()
        before = copy.deepcopy(self.worker.session)
        preview = self.worker.preview_session({"config": cfg, "mode": "experiment", "include_task": True}, "preview")
        self.assertIn("rendered_prompt", preview)
        self.assertEqual(before, self.worker.session)
        self.assertEqual(self.runtime.calls, [])
        self.assertEqual([e for e in self.events if e["type"] == "session_preview"][-1]["command_id"], "preview")

    def test_validate_calibration_job_uses_separate_runtime_method(self):
        calls = []
        def validate_runtime(config, out_dir, emit, stop):
            calls.append((config, str(out_dir)))
            emit({"type": "calibration_progress", "stage": "validation"})
            return {"schema_version": 2, "validation_status": "no_detectable_effect"}
        self.runtime.validate_calibration = validate_runtime
        self.worker.jobs.put({"command": "validate_calibration", "id": "validation", "payload": {
            "config": {"calibration_dir": "source"}, "out_dir": str(self.path / "new-validation"), "calibration_id": "cal-new"}})
        self.worker.jobs.put(None)
        self.worker.loop()
        self.assertEqual(len(calls), 1)
        event = next(e for e in self.events if e["type"] == "calibration_complete")
        self.assertTrue(event["validation"])
        self.assertEqual(event["calibration_id"], "cal-new")


if __name__ == "__main__":
    unittest.main()
