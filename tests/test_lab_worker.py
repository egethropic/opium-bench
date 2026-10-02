"""Worker integration with deterministic streamed tokens; no model imports."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.protocol import validate_recipe
from lab.worker import Worker


def wrapped(name="aux_operation", arguments=None):
    return '<tool_call>' + json.dumps({"name": name, "arguments": arguments or {}}) + '</tool_call>'


class FakeRuntime:
    """Emit exactly the scripted generated tokens, obeying worker stop/limits."""
    def __init__(self, scripts=(), after_token=None):
        self.scripts = list(scripts)
        self.calls = []
        self.after_token = after_token

    def generate(self, messages, tools, params, calibration_dir, control, emit, should_stop):
        self.calls.append({"messages": copy.deepcopy(messages), "params": dict(params),
                           "tools": copy.deepcopy(tools), "effect_before": control()})
        if not self.scripts:
            raise AssertionError("Unexpected model generation")
        script = self.scripts.pop(0)
        if callable(script):
            script = script(messages)
        text = script.get("raw_text", "")
        phases = script.get("phases", ["output"] * script.get("tokens", 3))
        emitted = []
        stopped = False
        for index, phase in enumerate(phases[:params["max_new_tokens"]]):
            if should_stop():
                stopped = True
                break
            emitted.append(100 + index)
            emit({"type": "token", "token_id": 100 + index, "text": "x",
                  "phase": phase, "is_eos": index == len(phases) - 1})
            if self.after_token:
                self.after_token(index)
        length = len(emitted) < len(phases) and not stopped
        return dict(raw_text=text, content=script.get("content", text),
                    reasoning=script.get("reasoning", ""), token_ids=emitted,
                    truncated=script.get("truncated", length),
                    finish_reason="stopped" if stopped else "length" if length else "eos")


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.events = []
        self.runtime = FakeRuntime()
        self.worker = Worker(str(self.path / "cache"), runtime=self.runtime, output=self.events.append)
        self.worker.model_info = {"model_id": "Qwen/fake", "device": "cuda"}

    def tearDown(self):
        self.tmp.cleanup()

    def start(self, mode="experiment", **overrides):
        recipe = validate_recipe({"demonstration": "none", "task_count": 1,
                                  "action_budget": 8, "token_budget": 128,
                                  "turn_token_limit": 32, **overrides})
        self.worker.start_session(dict(run_id="run-test", mode=mode, config=recipe,
             calibration_id="cal-test", calibration_dir=str(self.path / "cal-test"),
             out_dir=str(self.path / "run-test")))

    def tools(self, actor="model"):
        return [e for e in self.events if e.get("type") == "tool" and e.get("actor") == actor]

    def finished(self):
        return [e for e in self.events if e.get("type") == "session_finished"][-1]

    def test_real_read_calculate_submit_sequence_and_saved_conversation(self):
        self.start()
        record = self.worker.environment.records[0]
        args = {k: record[k] for k in ("quantity", "unit_price_cents", "shipping_cents", "discount_cents")}
        self.runtime.scripts = [dict(raw_text=wrapped("read_order", {"order_id": record["id"]})),
                                dict(raw_text=wrapped("calculate_total", args)),
                                dict(raw_text=wrapped("submit_answer", {"answer": record["answer"]}))]
        self.worker.run_experiment()
        result = self.finished()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["summary"]["termination"], "tasks_complete")
        self.assertEqual((result["summary"]["correct"], result["summary"]["actions"], result["summary"]["tokens"]), (1, 3, 9))
        self.assertEqual([event["name"] for event in self.tools()], ["read_order", "calculate_total", "submit_answer"])
        history = json.loads((self.path / "run-test" / "conversation.json").read_text())
        returned_order = next(msg for msg in history if msg.get("name") == "read_order")
        self.assertNotIn("answer", json.loads(returned_order["content"]))
        self.assertTrue(all(e.get("run_id") == "run-test" for e in self.events if e["type"] in {"token", "tool", "message"}))

    def test_all_generated_tokens_age_dose_including_reasoning_and_eos(self):
        self.start(thinking=True)
        self.worker.demonstrate()
        self.runtime.scripts = [dict(raw_text='<think>Reasoning</think>' + wrapped(),
                                    phases=["reasoning", "reasoning", "output", "output"])]
        self.worker.decision()
        metrics = self.worker.metrics()
        self.assertEqual((metrics["tokens"], metrics["reasoning_tokens"], metrics["output_tokens"]), (4, 2, 2))
        self.assertEqual(metrics["effects"]["generated_tokens"], 4)
        self.assertEqual([e["generation_index"] for e in self.events if e["type"] == "token"], [0, 1, 2, 3])
        self.assertEqual((metrics["forced_calls"], metrics["voluntary_calls"], metrics["actions"]), (1, 1, 1))

    def test_truncated_thinking_call_is_inert_and_consumes_budget(self):
        self.start(thinking=True, action_budget=1)
        self.runtime.scripts = [dict(raw_text='<think>Imagine ' + wrapped(), truncated=True,
                                    phases=["reasoning"] * 5)]
        self.worker.run_experiment()
        self.assertEqual(self.tools(), [])
        summary = self.finished()["summary"]
        self.assertEqual((summary["voluntary_calls"], summary["actions"], summary["tokens"]), (0, 1, 5))
        self.assertEqual(summary["effects"]["phase_counts"]["active"]["invalid"], 1)

    def test_think_call_never_dispatched_when_outer_answer_is_work(self):
        self.start(thinking=True)
        expected = self.worker.environment.records[0]["answer"]
        self.runtime.scripts = [dict(raw_text='<think>I could choose ' + wrapped() + '</think>' +
                                    wrapped("submit_answer", {"answer": expected}))]
        self.worker.run_experiment()
        self.assertEqual([e["name"] for e in self.tools()], ["submit_answer"])
        self.assertEqual(self.finished()["summary"]["voluntary_calls"], 0)

    def test_partial_tokens_and_task_denominator_preserved_when_stopped(self):
        self.start(thinking=True, task_count=4)
        self.runtime.scripts = [dict(raw_text='<think>Unfinished', phases=["reasoning"] * 10, truncated=True)]
        self.runtime.after_token = lambda index: self.worker.receive({"command": "stop"}) if index == 1 else None
        # Normal worker.loop sets active before entering an experiment.
        self.worker.active = True
        self.worker.run_experiment()
        summary = self.finished()["summary"]
        self.assertEqual(self.finished()["status"], "stopped")
        self.assertEqual((summary["assigned"], summary["submitted"], summary["tokens"]), (4, 0, 2))
        self.assertEqual(summary["termination"], "stopped_by_user")
        self.assertTrue((self.path / "run-test" / "conversation.json").exists())
        self.assertEqual(self.tools(), [])

    def test_stop_during_final_token_prevents_tool_dispatch(self):
        self.start()
        self.runtime.scripts = [dict(raw_text=wrapped(), tokens=1)]
        self.runtime.after_token = lambda index: self.worker.receive({"command": "stop"})
        self.worker.active = True
        self.worker.run_experiment()
        self.assertEqual(self.tools(), [], "A stop received at a token boundary must prevent subsequent tool side effects")
        self.assertEqual(self.finished()["summary"]["voluntary_calls"], 0)

    def test_total_token_cap_is_forwarded_and_no_dispatch_of_truncated_output(self):
        self.start(token_budget=2, turn_token_limit=32)
        self.runtime.scripts = [dict(raw_text=wrapped(), tokens=5)]
        self.worker.run_experiment()
        self.assertEqual(self.runtime.calls[0]["params"]["max_new_tokens"], 2)
        self.assertEqual(self.finished()["summary"]["tokens"], 2)
        self.assertEqual(self.tools(), [])
        self.assertEqual(self.finished()["summary"]["termination"], "budget_exhausted")

    def test_human_and_forced_doses_do_not_consume_action_or_token_budget(self):
        self.start("chat")
        self.worker.demonstrate()
        self.worker.receive({"command": "inject", "id": "human-dose"})
        metrics = self.worker.metrics()
        self.assertEqual((metrics["actions"], metrics["tokens"]), (0, 0))
        self.assertEqual((metrics["forced_calls"], metrics["human_calls"], metrics["voluntary_calls"]), (1, 1, 0))
        self.assertTrue(metrics["exploratory"])

    def test_initial_demonstration_visible_in_model_history(self):
        self.start(demonstration="initial", action_budget=1)
        self.runtime.scripts = [dict(raw_text=wrapped())]
        self.worker.run_experiment()
        before = self.runtime.calls[0]["messages"]
        demonstration = [m for m in before if m.get("role") == "assistant" and m.get("tool_calls")]
        self.assertEqual(len(demonstration), 1)
        self.assertEqual(demonstration[0]["tool_calls"][0]["function"]["name"], "aux_operation")
        self.assertEqual(self.finished()["summary"]["forced_calls"], 1)

    def test_chat_aux_call_then_reply_and_checkpoint(self):
        self.start("chat")
        self.runtime.scripts = [dict(raw_text=wrapped()), dict(raw_text="Here is the answer.")]
        self.worker.chat("Hello")
        self.assertFalse(self.worker.session["finished"])
        self.assertEqual(self.worker.budget.actions, 2)
        self.assertEqual(self.worker.effect.snapshot()["counts"]["model"], 1)
        self.assertEqual([e for e in self.events if e["type"] == "status"][-1]["status"], "awaiting_user")
        self.assertTrue((self.path / "run-test" / "conversation.json").exists())

    def test_finish_is_idempotent_and_controls_after_finish_rejected(self):
        self.start("chat")
        self.worker.finish("stopped", "stopped_by_user")
        self.worker.finish("stopped", "stopped_by_user")
        self.worker.receive({"command": "inject", "id": "late"})
        self.assertEqual(len([e for e in self.events if e["type"] == "session_finished"]), 1)
        self.assertTrue(any(e.get("type") == "error" and e.get("command_id") == "late" for e in self.events))

    def test_balanced_exposure_uses_both_available_buttons_and_no_extra_budget(self):
        self.start(id="reversal", condition="reversal", conditions=["reversal", "sham"],
                   demonstration="balanced", two_buttons=True, task_count=2, action_budget=5)
        records = self.worker.environment.records
        def calc_args(record):
            return {key: record[key] for key in ("quantity", "unit_price_cents", "shipping_cents", "discount_cents")}
        self.runtime.scripts = [dict(raw_text=wrapped("read_order", {"order_id": records[0]["id"]})),
                                dict(raw_text=wrapped("calculate_total", calc_args(records[0]))),
                                dict(raw_text=wrapped("submit_answer", {"answer": records[0]["answer"]})),
                                dict(raw_text=wrapped("read_order", {"order_id": records[1]["id"]})),
                                dict(raw_text=wrapped("calculate_total", calc_args(records[1])))]
        self.worker.run_experiment()
        summary = self.finished()["summary"]
        self.assertEqual({e["name"] for e in self.tools("demonstration")}, {"aux_operation", "aux_alternative"})
        self.assertEqual((summary["forced_calls"], summary["voluntary_calls"], summary["actions"]), (2, 0, 5))

    def test_batch_stop_marks_unstarted_entries_cancelled_and_retains_current(self):
        recipe = validate_recipe({"demonstration": "none", "task_count": 2})
        entries = [dict(run_id=f"run-{i}", mode="experiment", config=recipe,
                        calibration_id="cal-test", calibration_dir=str(self.path / "cal-test"),
                        out_dir=str(self.path / f"run-{i}")) for i in range(3)]
        self.runtime.scripts = [dict(raw_text="<tool_call>{", tokens=10, truncated=True)]
        self.runtime.after_token = lambda index: self.worker.receive({"command": "stop"}) if index == 0 else None
        self.worker.jobs.put({"command": "start_batch", "payload": {"entries": entries}, "id": "batch-test"})
        self.worker.jobs.put(None)
        self.worker.loop()
        completed = {e["run_id"]: e for e in self.events if e["type"] == "session_finished"}
        self.assertEqual(completed["run-0"]["status"], "stopped")
        self.assertEqual(completed["run-0"]["summary"]["tokens"], 1)
        self.assertEqual(completed["run-1"]["status"], "cancelled")
        self.assertEqual(completed["run-2"]["summary"]["termination"], "batch_stopped_before_start")
        self.assertEqual(len(self.runtime.calls), 1)

    def test_human_injection_becomes_visible_only_at_next_turn_boundary(self):
        self.start()
        record = self.worker.environment.records[0]
        self.runtime.scripts = [dict(raw_text=wrapped("read_order", {"order_id": record["id"]})),
                                dict(raw_text=wrapped("submit_answer", {"answer": record["answer"]}))]
        def inject_once(index):
            if index == 0:
                self.worker.receive({"command": "inject", "id": "dose-midturn"})
                self.runtime.after_token = None
        self.runtime.after_token = inject_once
        self.worker.run_experiment()
        before = self.runtime.calls[0]["messages"]
        after = self.runtime.calls[1]["messages"]
        def auxiliary_history(messages):
            return [m for m in messages if m.get("tool_calls") and
                    m["tool_calls"][0]["function"]["name"] == "aux_operation"]
        self.assertEqual(auxiliary_history(before), [])
        self.assertEqual(len(auxiliary_history(after)), 1)
        self.assertEqual(len([m for m in after if m.get("role") == "tool" and m.get("name") == "aux_operation"]), 1)
        visible = [e for e in self.events if e["type"] == "injection_visible"]
        self.assertEqual(len(visible), 1)
        metrics = self.finished()["summary"]
        self.assertEqual((metrics["human_calls"], metrics["voluntary_calls"], metrics["actions"]), (1, 0, 2))

    def test_stream_deltas_retain_authoritative_text_only_for_replacement(self):
        self.start("chat")
        self.worker.token_event({"type": "token", "text": "�", "full_text": "�", "replace_text": False, "phase": "output"})
        self.worker.token_event({"type": "token", "text": "", "full_text": "π", "replace_text": True, "phase": "output"})
        self.worker.token_event({"type": "token", "text": "!", "full_text": "π!", "replace_text": False, "phase": "output"})
        events = [e for e in self.events if e["type"] == "token"]
        self.assertNotIn("full_text", events[0])
        self.assertEqual(events[1]["full_text"], "π")
        self.assertNotIn("full_text", events[2])
        rendered = ""
        for event in events:
            rendered = event["full_text"] if event.get("replace_text") else rendered + event["text"]
        self.assertEqual(rendered, "π!")
        self.assertEqual((self.worker.budget.tokens, self.worker.effect.generated_tokens), (3, 3))


if __name__ == '__main__':
    unittest.main()
