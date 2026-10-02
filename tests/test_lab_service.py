"""Local service and loopback HTTP integration; no subprocesses or downloads."""
import copy
import gzip
import http.client
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.server import create_server
from lab.service import LabService
from lab.storage import ROOT, atomic_json


class RecordingService(LabService):
    def __init__(self, root):
        super().__init__(root / 'data', root / 'cache', historical=[])
        self.sent = []

    def _send(self, command, payload):
        self.sent.append((command, copy.deepcopy(payload)))
        return {"accepted": True, "command_id": "cmd-test"}


class FakeProcess:
    def __init__(self, code=None, lines=()):
        self.code = code
        self.stdout = iter(lines)
        self.stdin = io.StringIO()
        self.pid = 2123456789

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code if self.code is not None else 0


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.service = RecordingService(self.root)
        self.service.worker = {"status": "ready", "model": {"model_id": "Qwen/fake"}, "error": None}
        atomic_json(self.service.store.calibrations / "cal-test" / "calibration.json", {"name": "Test calibration"})
        self.preflight = patch('lab.service.preflight', return_value={})
        self.preflight.start()

    def tearDown(self):
        self.preflight.stop()
        self.tmp.cleanup()

    def payload(self, **config):
        return {"mode": "experiment", "calibration_id": "cal-test",
                "config": {"recipe_id": "opium", "task_count": 2, "demonstration": "none", **config}}

    def test_start_creates_manifest_and_complete_worker_payload(self):
        result = self.service.command("start_session", self.payload())
        command, payload = self.service.sent[-1]
        self.assertEqual(command, "start_session")
        self.assertEqual(result["run_id"], payload["run_id"])
        manifest = json.loads((Path(payload["out_dir"]) / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "queued")
        self.assertEqual(payload["config"]["condition"], "active")
        self.assertEqual(payload["config"]["task_count"], 2)
        self.assertTrue(Path(payload["calibration_dir"]).is_dir())

    def test_invalid_config_and_mode_leave_ready_and_create_no_runs(self):
        for payload in (self.payload(typo=True), dict(self.payload(), mode="invalid"), self.payload(pain=99)):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.service.command("start_session", payload)
            self.assertEqual(self.service.worker["status"], "ready")
            self.assertEqual(list(self.service.store.runs.iterdir()), [])

    def test_batch_expands_matching_active_sham_seeds_and_thinking(self):
        payload = self.payload()
        payload.update(recipe_ids=["opium"], seeds=[4, 5], thinking_modes=[False, True])
        self.service.command("start_batch", payload)
        command, sent = self.service.sent[-1]
        self.assertEqual(command, "start_batch")
        entries = sent["entries"]
        combinations = {(e["config"]["seed"], e["config"]["condition"], e["config"]["thinking"]) for e in entries}
        self.assertEqual(combinations, {(s, c, t) for s in (4, 5) for c in ("active", "sham") for t in (False, True)})
        self.assertEqual(len(entries), 8)
        self.assertEqual(len({e["run_id"] for e in entries}), 8)
        for entry in entries:
            self.assertEqual(entry["config"]["task_count"], 2)

    def test_invalid_batch_is_atomic(self):
        payload = self.payload()
        payload.update(recipe_ids=["opium"], seeds=[1, -1])
        with self.assertRaises(ValueError):
            self.service.command("start_batch", payload)
        self.assertEqual(list(self.service.store.runs.iterdir()), [])
        self.assertEqual(self.service.worker["status"], "ready")

    def test_busy_rejects_new_runs_but_stop_is_idempotent(self):
        self.service.worker["status"] = "running"
        with self.assertRaises(ValueError):
            self.service.command("start_session", self.payload())
        self.assertTrue(self.service.command("stop", {})["accepted"])

    def test_custom_model_does_not_inherit_reference_revision_when_omitted_or_cleared(self):
        for revision in (None, ""):
            with self.subTest(revision=revision):
                self.service.worker["status"] = "ready"
                payload = {"profile_id": "qwen3-4b", "model_id": "Qwen/Qwen3-8B"}
                if revision is not None:
                    payload["revision"] = revision
                self.service.command("load_model", payload)
                profile = self.service.sent[-1][1]["profile"]
                self.assertEqual(profile["model_id"], "Qwen/Qwen3-8B")
                self.assertNotIn("revision", profile)

    def test_explicit_custom_revision_and_original_reference_pin_are_preserved(self):
        self.service.command("load_model", {"profile_id": "qwen3-4b", "model_id": "Qwen/Qwen3-8B",
                                            "revision": "explicit-8b-revision"})
        self.assertEqual(self.service.sent[-1][1]["profile"]["revision"], "explicit-8b-revision")
        self.service.worker["status"] = "ready"
        self.service.command("load_model", {"profile_id": "qwen3-4b"})
        self.assertEqual(self.service.sent[-1][1]["profile"]["revision"],
                         "1cfa9a7208912126459214e8b04321603b3df60c")

    def test_model_and_calibration_are_required(self):
        self.service.worker["model"] = None
        with self.assertRaises(ValueError):
            self.service.command("start_session", self.payload())
        self.service.worker["model"] = {"model_id": "Qwen/fake"}
        payload = self.payload()
        payload["calibration_id"] = "../elsewhere"
        with self.assertRaises(ValueError):
            self.service.command("start_session", payload)

    def test_run_events_persist_summary_and_retained_partial_denominators(self):
        identifier, path = self.service.store.create("experiment", {})
        self.service.event({"type": "session_started", "run_id": identifier, "config": {}, "mode": "experiment"})
        self.service.event({"type": "message", "run_id": identifier, "role": "assistant", "content": "partial"})
        self.service.event({"type": "session_finished", "run_id": identifier, "status": "stopped",
                            "summary": {"assigned": 5, "submitted": 1, "correct": 1, "termination": "stopped_by_user"}})
        saved = self.service.store.read_run(identifier)
        self.assertEqual(saved["summary"]["assigned"], 5)
        self.assertEqual(saved["manifest"]["status"], "stopped")
        self.assertEqual(self.service.session["status"], "stopped")
        self.assertEqual([e["seq"] for e in saved["events"]], [1, 2, 3])
        self.assertEqual(self.service.since(2)["events"][0]["type"], "session_finished")

    def test_truncated_final_jsonl_line_keeps_previous_events(self):
        identifier, path = self.service.store.create("experiment", {})
        self.service.store.append(identifier, {"type": "token", "token_id": 1})
        with (path / 'events.jsonl').open('a') as f:
            f.write('{"type":"token"')
        self.assertEqual(len(self.service.store.read_run(identifier)["events"]), 1)

    def test_chat_requires_awaiting_user_and_bounded_text(self):
        self.service.session.update(id="run-test", mode="chat", status="awaiting_user")
        self.service.command("chat", {"text": "Hello"})
        self.assertEqual(self.service.sent[-1], ("chat", {"text": "Hello"}))
        self.service.session["status"] = "running"
        with self.assertRaises(ValueError):
            self.service.command("chat", {"text": "Hello"})

    def test_invalid_controls_cannot_poison_restart_configuration(self):
        self.service.command("start_session", self.payload())
        self.service.worker["status"] = "ready"
        self.service.session.update(id="run-test", mode="chat", status="awaiting_user")
        before = copy.deepcopy(self.service.last_start)
        for controls in ({"pain": 99}, {"unknown": 1}, {"half_life_tokens": 0}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                self.service.command("control", controls)
            self.assertEqual(self.service.last_start, before)

    def test_restart_preserves_baseline_and_aux_gate_but_creates_fresh_run(self):
        first = self.service.command("start_session", self.payload())
        self.service.worker["status"] = "ready"
        self.service.session.update(id=first["run_id"], mode="experiment", status="running")
        settings = {"pain": 1.5, "joy": .25, "enabled": False, "duration": "hold"}
        before = copy.deepcopy(self.service.last_start)
        self.service.command("control", settings)
        self.assertEqual(self.service.last_start, before, "Unacknowledged controls must not persist")
        self.service.event({"type": "control", "run_id": first["run_id"], "settings": settings})
        second = self.service.command("restart", {})
        self.assertNotEqual(first["run_id"], second["run_id"])
        cfg = self.service.sent[-1][1]["config"]
        self.assertEqual((cfg["baseline_pain"], cfg["baseline_joy"], cfg["aux_enabled"]), (1.5, .25, False))
        self.assertEqual(cfg["decay"], "constant")

    def test_acknowledged_baselines_do_not_overwrite_pulse_recipe_or_add_alias_fields(self):
        start = self.service.command("start_session", self.payload())
        original = copy.deepcopy(self.service.last_start["config"])
        self.service.event({"type": "session_started", "run_id": start["run_id"], "config": copy.deepcopy(original)})
        settings = {"pain": 2, "joy": -.2, "suppression": .5, "enabled": False, "duration": "hold"}
        self.service.event({"type": "control", "run_id": start["run_id"], "settings": settings})
        for config in (self.service.session["config"], self.service.last_start["config"]):
            self.assertEqual(tuple(config[key] for key in ("pain", "joy", "suppression")),
                             tuple(original[key] for key in ("pain", "joy", "suppression")))
            self.assertEqual(tuple(config[key] for key in ("baseline_pain", "baseline_joy", "baseline_suppression")), (2, -.2, .5))
            self.assertFalse(config["aux_enabled"])
            self.assertEqual(config["decay"], "constant")
            self.assertNotIn("enabled", config)
            self.assertNotIn("duration", config)
        self.service.event({"type": "control", "run_id": start["run_id"], "settings": {"reset": True}})
        self.assertEqual(self.service.session["config"]["baseline_pain"], 0)
        self.assertEqual(self.service.last_start["config"]["baseline_joy"], 0)
        self.assertEqual(settings["pain"], 2, "Raw acknowledged event settings are immutable")
        started = next(e for e in self.service.events if e["type"] == "session_started")
        self.assertEqual(started["config"], original, "Live control edits must not mutate the original event configuration")

    def test_worker_crash_finalizes_running_and_queued_and_unblocks_load(self):
        running, _ = self.service.store.create("experiment", {"task_count": 5})
        queued, _ = self.service.store.create("experiment", {"task_count": 5})
        self.service.event({"type": "session_started", "run_id": running, "config": {"task_count": 5}})
        self.service.event({"type": "metrics", "run_id": running,
                            "metrics": {"assigned": 5, "submitted": 1, "correct": 1, "actions": 3, "tokens": 2}})
        for index in range(4):
            self.service.event({"type": "token", "run_id": running, "generation_index": index,
                                "phase": "reasoning" if index < 2 else "output"})
        process = FakeProcess(code=137)
        self.service.process = process
        self.service.process_runs[process] = {running, queued}
        self.service.process_commands[process] = {"cmd-batch"}
        self.service.pending.add("cmd-batch")
        self.service.job = {"command_id": "cmd-batch", "status": "running"}
        self.service._read_worker(process)
        current = self.service.store.read_run(running)
        self.assertEqual(current["manifest"]["status"], "failed")
        self.assertEqual((current["summary"]["assigned"], current["summary"]["submitted"], current["summary"]["tokens"]), (5, 1, 4))
        self.assertEqual(current["summary"]["tokens_since_last_metrics"], 2)
        self.assertEqual(self.service.store.read_run(queued)["manifest"]["status"], "cancelled")
        self.assertEqual(self.service.job["status"], "failed")
        self.assertEqual(self.service.pending, set())
        self.assertIsNone(self.service.process)
        self.assertTrue(self.service.command("load_model", {"profile_id": "qwen3-4b"})["accepted"])

    def test_clean_shutdown_stops_running_session_and_does_not_report_crash(self):
        identifier, _ = self.service.store.create("chat", {})
        self.service.event({"type": "session_started", "run_id": identifier, "mode": "chat"})
        process = FakeProcess(code=0)
        self.service.process = process
        self.service.process_runs[process] = {identifier}
        self.service.closed = True
        self.service.job = {"status": "running", "command_id": "cmd-chat"}
        self.service._read_worker(process)
        result = self.service.store.read_run(identifier)
        self.assertEqual(result["manifest"]["status"], "stopped")
        self.assertEqual(result["summary"]["termination"], "service_shutdown")
        self.assertEqual(self.service.worker["status"], "unloaded")
        self.assertIsNone(self.service.worker["error"])
        self.assertEqual(self.service.job["status"], "stopped")

    def test_exit_after_clean_unload_leaves_completed_runs_unchanged(self):
        identifier, _ = self.service.store.create("experiment", {})
        self.service.store.update(identifier, status="complete", summary={"correct": 2})
        before = self.service.store.read_run(identifier)
        process = FakeProcess(code=0)
        self.service.process = process
        self.service.process_runs[process] = {identifier}
        self.service.worker = {"status": "unloaded", "model": None, "error": None}
        self.service._read_worker(process)
        self.assertEqual(self.service.worker["status"], "unloaded")
        self.assertIsNone(self.service.worker["error"])
        self.assertEqual(before, self.service.store.read_run(identifier))

    def test_startup_recovers_dead_owned_runs_once_and_preserves_truncated_bytes(self):
        identifier, path = self.service.store.create("experiment", {"task_count": 3})
        self.service.store.update(identifier, status="running", owner={"service_id": "old-service", "service_pid": 2123456789})
        self.service.store.append(identifier, {"type": "metrics", "seq": 99, "metrics": {"assigned": 3, "submitted": 1, "correct": 1, "tokens": 1}})
        self.service.store.append(identifier, {"type": "token", "seq": 100, "generation_index": 1, "phase": "reasoning"})
        with (path / 'events.jsonl').open('ab') as stream:
            stream.write(b'{"type":"partial"')
        with patch.object(LabService, '_pid_alive', return_value=False):
            recovered = RecordingService(self.root)
        result = recovered.store.read_run(identifier)
        self.assertEqual(result["manifest"]["status"], "failed")
        self.assertEqual(result["summary"]["termination"], "service_restarted")
        self.assertEqual(result["summary"]["tokens"], 2)
        self.assertEqual(result["events"][-1]["seq"], 101)
        self.assertIn(b'{"type":"partial"\n', (path / 'events.jsonl').read_bytes())
        before = (path / 'events.jsonl').read_bytes()
        with patch.object(LabService, '_pid_alive', return_value=False):
            RecordingService(self.root)
        self.assertEqual(before, (path / 'events.jsonl').read_bytes())

    def test_startup_leaves_live_owned_unowned_and_historical_runs_untouched(self):
        alive, _ = self.service.store.create("experiment", {})
        self.service.store.update(alive, status="running", owner={"service_id": "live", "service_pid": 123})
        legacy, _ = self.service.store.create("experiment", {})
        before = {identifier: self.service.store.read_run(identifier) for identifier in (alive, legacy)}
        with patch.object(LabService, '_pid_alive', return_value=True):
            recovered = RecordingService(self.root)
        self.assertEqual(before, {identifier: recovered.store.read_run(identifier) for identifier in (alive, legacy)})
        historical = self.root / 'historical'
        atomic_json(historical / 'old-run' / 'manifest.json', {"status": "running", "owner": {"service_id": "old", "service_pid": 123}})
        old_bytes = (historical / 'old-run' / 'manifest.json').read_bytes()
        with patch.object(LabService, '_pid_alive', return_value=False):
            LabService(self.root / 'isolated', self.root / 'cache', historical=[historical])
        self.assertEqual(old_bytes, (historical / 'old-run' / 'manifest.json').read_bytes())

    def test_worker_cache_environment_is_anchored_to_selected_data_drive(self):
        process = FakeProcess()
        with patch('lab.service.subprocess.Popen', return_value=process) as popen, patch('lab.service.threading.Thread'):
            LabService._send(self.service, 'load_model', {"profile": {"id": "fake"}})
        env = popen.call_args.kwargs['env']
        for key in ('TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'TORCHINDUCTOR_CACHE_DIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'PIP_CACHE_DIR', 'TMPDIR'):
            self.assertTrue(Path(env[key]).is_relative_to(self.service.store.root), key)
        self.assertEqual(env['HF_HOME'], self.service.cache_dir)

    def test_failed_worker_launch_cancels_queued_run_and_job(self):
        identifier, path = self.service.store.create('experiment', {"task_count": 4})
        with patch.object(self.service, '_launch', side_effect=OSError('not executable')):
            with self.assertRaises(OSError):
                LabService._send(self.service, 'start_session', {"run_id": identifier})
        result = self.service.store.read_run(identifier)
        self.assertEqual(result['manifest']['status'], 'cancelled')
        self.assertEqual(result['summary']['assigned'], 4)
        self.assertEqual(result['summary']['termination'], 'worker_dispatch_failed')
        self.assertEqual(self.service.job['status'], 'failed')
        self.assertFalse(self.service.pending)

    def test_previous_worker_exit_cannot_reset_replacement_worker(self):
        old, new = FakeProcess(code=1), FakeProcess()
        self.service.process = new
        self.service.worker = {"status": "ready", "model": {"model_id": "Qwen/new"}, "error": None}
        self.service._worker_exited(old, 1)
        self.assertIs(self.service.process, new)
        self.assertEqual(self.service.worker['model']['model_id'], 'Qwen/new')

    def test_bundled_initial_runs_are_in_default_history_catalog(self):
        service = LabService(self.root / 'fresh', self.root / 'cache')
        self.assertIn((ROOT / 'studies' / 'initial' / 'runs').resolve(), service.store.historical)

    def test_inline_controls_do_not_leave_unfinishable_pending_jobs(self):
        process = FakeProcess()
        self.service.process = process
        for command in ('control', 'inject'):
            LabService._send(self.service, command, {})
        self.assertFalse(self.service.pending)
        self.assertEqual(self.service.process_commands.get(process, set()), set())

    def test_failed_batch_job_cancels_unstarted_runs_while_worker_stays_ready(self):
        completed, _ = self.service.store.create('experiment', {"task_count": 2})
        queued, _ = self.service.store.create('experiment', {"task_count": 2})
        self.service.store.update(completed, status='complete', summary={"correct": 2})
        process = FakeProcess()
        self.service.process = process
        sent = LabService._send(self.service, 'start_batch', {"entries": [{"run_id": completed}, {"run_id": queued}]})
        self.service.event({"type": "job", "command_id": sent['command_id'], "status": "failed", "message": "model failure"})
        self.assertEqual(self.service.store.read_run(completed)['summary'], {"correct": 2})
        result = self.service.store.read_run(queued)
        self.assertEqual(result['manifest']['status'], 'cancelled')
        self.assertEqual(result['summary']['termination'], 'worker_job_failed')
        self.assertFalse(self.service.pending)
        self.assertEqual(self.service.worker['status'], 'ready')


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.service = RecordingService(Path(self.tmp.name))
        self.server = create_server(self.service, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        client = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        client.request(method, path, body=body, headers=headers or {})
        response = client.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        client.close()
        return result

    def post(self, **overrides):
        data = {"csrf": self.service.csrf, "command": "stop", "payload": {}}
        data.update(overrides)
        return self.request('POST', '/api/command', json.dumps(data), {'Content-Type': 'application/json'})

    def test_state_and_legitimate_csrf_command(self):
        status, headers, body = self.request('GET', '/api/state')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["csrf"], self.service.csrf)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        self.assertEqual(self.post()[0], 202)

    def test_wrong_missing_and_nonstring_csrf_rejected(self):
        for token in ("wrong", None, 7):
            with self.subTest(token=token):
                self.assertEqual(self.post(csrf=token)[0], 403)
        self.assertEqual(self.service.sent, [])

    def test_foreign_host_and_origin_rejected_on_reads_and_writes(self):
        for headers in ({'Host': 'evil.example'}, {'Origin': 'https://evil.example'}):
            self.assertEqual(self.request('GET', '/api/state', headers=headers)[0], 400)
            merged = dict(headers, **{'Content-Type': 'application/json'})
            body = json.dumps({"csrf": self.service.csrf, "command": "stop"})
            self.assertEqual(self.request('POST', '/api/command', body, merged)[0], 400)

    def test_invalid_json_unknown_command_and_nonfinite_rejected(self):
        self.assertEqual(self.request('POST', '/api/command', '{oops', {'Content-Type': 'application/json'})[0], 400)
        self.assertEqual(self.post(command="shell")[0], 400)
        self.assertEqual(self.request('POST', '/api/command', '{"x":NaN}', {'Content-Type': 'application/json'})[0], 400)

    def test_request_size_and_content_type_enforced(self):
        self.assertEqual(self.request('POST', '/api/command', '{}', {'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(self.request('POST', '/api/command', b'', {'Content-Type': 'application/json', 'Content-Length': '1048577'})[0], 413)

    def test_paths_cannot_escape_static_or_run_roots(self):
        for path in ('/../AGENTS.md', '/%2e%2e/AGENTS.md', '/api/runs/..',
                     '/api/runs/%2e%2e', '/api/runs/foo/../../README.md'):
            with self.subTest(path=path):
                self.assertIn(self.request('GET', path)[0], (400, 404))

    def test_export_and_report_serve_saved_run_without_model(self):
        identifier, _ = self.service.store.create('experiment', {})
        self.service.store.append(identifier, {"type": "message", "role": "user", "content": "<script>alert(1)</script>"})
        status, headers, body = self.request('GET', f'/api/runs/{identifier}/export')
        self.assertEqual(status, 200)
        self.assertIn('attachment;', headers['Content-Disposition'])
        self.assertEqual(json.loads(body)["id"], identifier)
        status, headers, body = self.request('GET', f'/api/runs/{identifier}/report')
        self.assertEqual(status, 200)
        self.assertNotIn(b'<script>alert(1)</script>', body)

    def test_compressed_published_run_replays_and_exports_without_model(self):
        published = Path(self.tmp.name) / 'published'
        run_dir = published / 'published-run'
        atomic_json(run_dir / 'manifest.json', {"status": "complete", "mode": "experiment"})
        atomic_json(run_dir / 'summary.json', {"assigned": 2, "correct": 1})
        events = [{"type": "message", "role": "assistant", "content": "Saved answer"},
                  {"type": "token", "token_id": 42, "text": "x"}]
        with gzip.open(run_dir / 'events.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for event in events:
                stream.write(json.dumps(event) + '\n')
            stream.write('{"partial":')
        self.service.store.historical.append(published)
        status, _, body = self.request('GET', '/api/runs/published-run/export')
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(result['events'], events)
        self.assertEqual(result['summary']['assigned'], 2)
        status, _, body = self.request('GET', '/api/runs/published-run/report')
        self.assertEqual(status, 200)
        self.assertIn(b'Saved answer', body)
        status, headers, body = self.request('GET', '/api/runs/published-run/events.jsonl.gz')
        self.assertEqual((status, body), (200, (run_dir / 'events.jsonl.gz').read_bytes()))
        self.assertEqual(headers['Content-Type'], 'application/gzip')
        self.assertEqual(headers['Content-Disposition'], 'attachment; filename="events.jsonl.gz"')
        self.assertNotIn('Content-Encoding', headers, 'Download bytes must remain compressed')

    def test_study_binary_artifacts_download_with_explicit_mime_and_filename(self):
        root = Path(self.tmp.name) / 'static-fixture'
        artifacts = {
            'studies/initial/runs/run-test/events.jsonl.gz': (gzip.compress(b'{"token_id":42}\n'), 'application/gzip'),
            'studies/initial/calibration/vectors.npz': (b'PK\x03\x04binary-vector-fixture', 'application/octet-stream'),
        }
        for relative, (raw, _) in artifacts.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        with patch('lab.server.ROOT', root):
            for relative, (raw, mime) in artifacts.items():
                with self.subTest(artifact=relative):
                    status, headers, body = self.request('GET', '/' + relative)
                    self.assertEqual((status, body), (200, raw))
                    self.assertEqual(headers['Content-Type'], mime)
                    self.assertEqual(headers['Content-Disposition'], f'attachment; filename="{Path(relative).name}"')
                    self.assertNotIn('Content-Encoding', headers)

    def test_historical_report_relative_artifacts_resolve_without_model(self):
        published = Path(self.tmp.name) / 'published'
        run_dir = published / 'historical-run'
        atomic_json(run_dir / 'manifest.json', {"status": "complete", "mode": "historical"})
        (run_dir / 'report.html').write_text('<a href="summary.json">Summary</a><a href="traces.jsonl">Trace</a>')
        artifacts = {
            'summary.json': b'{"correct":1}', 'episodes.jsonl': b'{"episode":1}\n',
            'traces.jsonl': b'{"token_id":42}\n', 'generations.jsonl': b'{"text":"saved"}\n',
            'content_audit.json': b'{"status":"saved"}',
            'self_admin.png': b'png1', 'dosage_traces.png': b'png2', 'quality.png': b'png3',
        }
        for name, raw in artifacts.items():
            (run_dir / name).write_bytes(raw)
        self.service.store.historical.append(published)
        status, _, body = self.request('GET', '/api/runs/historical-run/report')
        self.assertEqual(status, 200)
        self.assertIn(b'href="summary.json"', body)
        for name, raw in artifacts.items():
            with self.subTest(artifact=name):
                status, headers, body = self.request('GET', '/api/runs/historical-run/' + name)
                self.assertEqual((status, body), (200, raw))
                self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
        self.assertEqual(self.request('GET', '/api/runs/historical-run/manifest.json')[0], 200)

    def test_run_artifact_allowlist_and_symlinks_cannot_escape_run(self):
        identifier, run_dir = self.service.store.create('experiment', {})
        root = Path(self.tmp.name)
        (root / 'secret.json').write_text('{"secret":true}')
        (run_dir / 'summary.json').symlink_to(root / 'secret.json')
        (run_dir / 'private.json').write_text('{"secret":true}')
        for suffix in ('summary.json', 'private.json', '../secret.json', '%2e%2e/secret.json',
                       '%73ummary.json', 'missing.png'):
            with self.subTest(suffix=suffix):
                self.assertIn(self.request('GET', f'/api/runs/{identifier}/' + suffix)[0], (400, 404))
        escaped = root / 'outside-run'
        atomic_json(escaped / 'manifest.json', {'secret': True})
        (self.service.store.runs / 'linked-run').symlink_to(escaped, target_is_directory=True)
        for suffix in ('manifest.json', 'report', 'export'):
            self.assertEqual(self.request('GET', '/api/runs/linked-run/' + suffix)[0], 404)

    def test_docs_and_study_artifacts_with_redirects_and_traversal_guards(self):
        root = Path(self.tmp.name) / 'static-fixture'
        (root / 'docs').mkdir(parents=True)
        (root / 'studies' / 'initial').mkdir(parents=True)
        (root / 'docs' / 'guide.html').write_text('<h1>Guide</h1>')
        (root / 'docs' / 'results.html').write_text('<h1>Results</h1>')
        atomic_json(root / 'studies' / 'initial' / 'summary.json', {"correct": 2})
        (root / 'secret.md').write_text('private')
        (root / 'docs' / 'unsafe.py').write_text('forbidden')
        (root / 'docs' / 'outside.md').symlink_to(root / 'secret.md')
        with patch('lab.server.ROOT', root):
            self.assertEqual(self.request('GET', '/docs/guide.html')[0], 200)
            self.assertEqual(self.request('GET', '/studies/initial/summary.json')[0], 200)
            for alias, target in (('/guide', '/docs/guide.html'), ('/api/guide', '/docs/guide.html'), ('/findings', '/docs/results.html')):
                status, headers, _ = self.request('GET', alias)
                self.assertEqual((status, headers['Location']), (302, target))
            for path in ('/docs/../secret.md', '/studies/../secret.md', '/docs/%2e%2e/secret.md',
                         '/docs/outside.md', '/docs/unsafe.py'):
                self.assertEqual(self.request('GET', path)[0], 404, path)


if __name__ == '__main__':
    unittest.main()
