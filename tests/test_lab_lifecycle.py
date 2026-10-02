"""Safe-boundary controls preserve actual task, budget and conversation state."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest

from lab.protocol import validate_recipe
from lab.worker import Worker
from tests.test_lab_worker import FakeRuntime, wrapped
from tests.test_lab_service import RecordingService
from lab.resources import ResourceGuard, Volume
from unittest.mock import patch


class PauseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def make_worker(self, mode="experiment", final_only=False):
        events, paused = [], threading.Event()

        def emit(event):
            events.append(copy.deepcopy(event))
            if event.get("type") == "status" and event.get("status") == "paused":
                paused.set()

        runtime = FakeRuntime()
        worker = Worker(self.temp.name, runtime=runtime, output=emit)
        worker.model_info = {"model_id": "Qwen/fake", "device": "cuda"}
        config = validate_recipe(dict(task_count=1, demonstration="after_two_work_calls",
                                      thinking=False, action_budget=8, token_budget=128))
        worker.start_session(dict(run_id="run-fixture", mode=mode, config=config,
            calibration_id="cal-fixture", calibration_dir=self.temp.name,
            out_dir=str(Path(self.temp.name) / "run-fixture")))
        if mode != "chat":
            record = worker.environment.records[0]
            args = {k: record[k] for k in ("quantity", "unit_price_cents", "shipping_cents", "discount_cents")}
            runtime.scripts = [dict(raw_text=wrapped("read_order", {"order_id": record["id"]})),
                dict(raw_text=wrapped("calculate_total", args)),
                dict(raw_text=wrapped("submit_answer", {"answer": record["answer"]}))]
            if final_only:
                runtime.scripts = runtime.scripts[-1:]
        return worker, runtime, events, paused

    def run_until_pause(self, final_only=False):
        worker, runtime, events, paused = self.make_worker(final_only=final_only)
        requested = threading.Event()

        def after_token(index):
            if not requested.is_set():
                requested.set()
                worker.receive(dict(command="pause", id="pause-1"))

        runtime.after_token = after_token
        worker.active = True
        worker.active_command = "start_session"
        thread = threading.Thread(target=worker.run_experiment, daemon=True)
        thread.start()
        self.addCleanup(lambda: (worker.receive(dict(command="stop")), thread.join(2)))
        self.assertTrue(paused.wait(2), events)
        self.assertEqual(worker.budget.tokens, 3)  # Completes the turn before pausing.
        self.assertEqual(worker.budget.actions, 1)
        return worker, runtime, events, thread

    def test_pause_resume_matches_uninterrupted_choices_context_and_budget(self):
        baseline, runtime0, _, _ = self.make_worker()
        baseline.run_experiment()
        worker, runtime, events, thread = self.run_until_pause()
        before = (worker.budget.snapshot(), worker.effect.snapshot())
        worker.receive(dict(command="pause", id="pause-duplicate"))
        self.assertEqual(before, (worker.budget.snapshot(), worker.effect.snapshot()))
        self.assertEqual(len([e for e in events if e.get("type") == "pause"]), 1)
        worker.receive(dict(command="resume", id="resume"))
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(worker.session["messages"], baseline.session["messages"])
        self.assertEqual(worker.budget.snapshot(), baseline.budget.snapshot())
        self.assertEqual(worker.effect.snapshot(), baseline.effect.snapshot())
        self.assertEqual([c["params"] for c in runtime.calls], [c["params"] for c in runtime0.calls])
        self.assertEqual(len([e for e in events if e.get("actor") == "demonstration"]), 1)

    def test_stop_wakes_pause_without_another_token_or_action(self):
        worker, runtime, events, thread = self.run_until_pause()
        worker.receive(dict(command="stop"))
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual((worker.budget.tokens, worker.budget.actions), (3, 1))
        finished = [e for e in events if e.get("type") == "session_finished"]
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["status"], "stopped")

    def test_final_task_can_pause_at_its_completed_boundary(self):
        worker, _, events, thread = self.run_until_pause(final_only=True)
        self.assertTrue(worker.environment.done)
        self.assertFalse(worker.session["finished"])
        worker.receive(dict(command="resume"))
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual([e for e in events if e.get("type") == "session_finished"][-1]["status"], "complete")

    def test_idle_chat_pause_resumes_waiting_without_generating(self):
        worker, runtime, events, paused = self.make_worker(mode="chat")
        worker.receive(dict(command="pause"))
        self.assertTrue(paused.is_set())
        worker.receive(dict(command="resume"))
        self.assertEqual([e for e in events if e.get("type") == "status"][-1]["status"], "awaiting_user")
        self.assertEqual(runtime.calls, [])
        self.assertEqual(worker.budget.tokens, 0)

    def test_non_session_job_cannot_be_marked_paused(self):
        worker, _, events, _ = self.make_worker(mode="chat")
        worker.active, worker.active_command = True, "calibrate"
        worker.receive(dict(command="pause"))
        self.assertFalse(worker.pause_requested.is_set())
        self.assertEqual(events[-1]["status"], "failed")

    def test_saved_boundary_continuation_matches_uninterrupted_and_keeps_parent_bytes(self):
        root = Path(self.temp.name)
        service = RecordingService(root)
        self.addCleanup(service.close)
        service.resources = ResourceGuard(dict(data=service.store.root),
            telemetry=lambda _: dict(free_bytes=100 * 2**30, total_bytes=200 * 2**30),
            resolver=lambda _: [Volume("test", self.temp.name)])
        fingerprint = dict(model_id="Qwen/fake", revision="test", template_sha256="a" * 64)
        model = dict(model_id="Qwen/fake", fingerprint=fingerprint,
                     fingerprint_sha256=hashlib.sha256(json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
        service.worker.update(status="ready", model=model)
        calibration = service.store.calibrations / "cal-fixture"
        calibration.mkdir()
        (calibration / "calibration.json").write_text('{"schema_version":1}')
        (calibration / "vectors.npz").write_bytes(b"fake-runtime-only")
        baseline, runtime0, _, _ = self.make_worker()
        baseline.run_experiment()
        with patch("lab.service.preflight"):
            service.command("start_session", dict(mode="experiment", calibration_id="cal-fixture", config=baseline.session["config"]))
        payload = service.sent[-1][1]
        first_runtime = FakeRuntime()
        first_runtime.scripts = [dict(raw_text=wrapped("read_order", {"order_id": baseline.environment.records[0]["id"]}))]
        worker = Worker(self.temp.name, runtime=first_runtime, output=service.event)
        worker.model_info = model
        worker.start_session(payload)
        worker.active, worker.active_command = True, "start_session"
        paused = threading.Event()
        original_emit = worker.output
        def emit(event):
            original_emit(event)
            if event.get("type") == "status" and event.get("status") == "paused":
                paused.set()
        worker.output = emit
        first_runtime.after_token = lambda _: worker.receive(dict(command="pause"))
        thread = threading.Thread(target=worker.run_experiment, daemon=True)
        thread.start()
        self.addCleanup(lambda: (worker.receive(dict(command="stop")), thread.join(2)))
        self.assertTrue(paused.wait(2))
        worker.receive(dict(command="stop"))
        thread.join(2)
        self.assertFalse(thread.is_alive())
        parent = Path(payload["out_dir"])
        before = {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
        with patch("lab.service.preflight"):
            accepted = service.command("branch", dict(run_id=payload["run_id"], policy="continue_state"))
        restored_payload = service.sent[-1][1]
        self.assertEqual(service.sent[-1][0], "restore_session")
        next_runtime = FakeRuntime()
        record = baseline.environment.records[0]
        next_runtime.scripts = [dict(raw_text=wrapped("calculate_total", {k:record[k] for k in ("quantity", "unit_price_cents", "shipping_cents", "discount_cents")})), dict(raw_text=wrapped("submit_answer", {"answer":record["answer"]}))]
        continuation = Worker(self.temp.name, runtime=next_runtime, output=service.event)
        continuation.model_info = model
        continuation.restore_session(restored_payload)
        continuation.run_experiment()
        self.assertEqual(continuation.session["messages"], baseline.session["messages"])
        self.assertEqual(continuation.budget.snapshot(), baseline.budget.snapshot())
        self.assertEqual(continuation.effect.snapshot(), baseline.effect.snapshot())
        # Service normalization adds generation defaults; all shared sampled parameters match.
        for resumed, original in zip(next_runtime.calls, runtime0.calls[1:]):
            for key in original["params"]:
                self.assertEqual(resumed["params"][key], original["params"][key])
        after = {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(accepted["parent"]["parent_prefix_kind"], "events")
        self.assertEqual(accepted["parent"]["inherited_actions"], 1)


if __name__ == "__main__":
    unittest.main()
