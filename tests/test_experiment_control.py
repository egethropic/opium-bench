"""Bounded controller tests: fake processes only, no models or real controls."""
import errno
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiment_control import ExperimentController, _read_object


def write_json(path, value):
    temporary = path.with_suffix(".test-tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except FileNotFoundError:
            # Polls can precede file creation or observe the replacement gap on
            # a mounted filesystem. Persistent absence still fails at deadline.
            pass
        time.sleep(.005)
    raise AssertionError("Timed out waiting for mocked controller state")


class FakeProcess:
    def __init__(self, directory, pid):
        self.directory = directory
        self.pid = pid
        self.returncode = None
        self.done = threading.Event()

    def poll(self):
        return self.returncode

    def wait(self):
        if not self.done.wait(15):
            raise AssertionError("Fake process was not finished by the test")
        return self.returncode

    def finish(self, code=0, status="complete"):
        if self.directory.exists():
            write_json(self.directory / "manifest.json", {"status": status, "manual_control": True})
        self.returncode = code
        self.done.set()


class ControllerFileReadTests(unittest.TestCase):
    def test_transient_mounted_filesystem_read_reopens_manifest(self):
        path = Mock()
        path.read_text.side_effect = [OSError(getattr(errno, "ENODATA", 61), "No data available"), '{"status":"complete"}']
        with patch("experiment_control.time.sleep") as sleep:
            self.assertEqual(_read_object(path), {"status": "complete"})
        self.assertEqual(path.read_text.call_count, 2)
        sleep.assert_called_once_with(.005)

    def test_persistent_no_data_is_bounded_and_reported(self):
        path = Mock()
        path.read_text.side_effect = OSError(getattr(errno, "ENODATA", 61), "No data available")
        with patch("experiment_control.time.sleep") as sleep, self.assertRaises(OSError) as caught:
            _read_object(path)
        self.assertEqual(caught.exception.errno, getattr(errno, "ENODATA", 61))
        self.assertEqual(path.read_text.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_other_read_and_parse_errors_are_not_retried(self):
        for failure in (PermissionError(errno.EACCES, "Permission denied"),
                        OSError(errno.EIO, "I/O error")):
            with self.subTest(failure=failure):
                path = Mock()
                path.read_text.side_effect = failure
                with patch("experiment_control.time.sleep") as sleep, self.assertRaises(type(failure)):
                    _read_object(path)
                self.assertEqual(path.read_text.call_count, 1)
                sleep.assert_not_called()
        path = Mock()
        path.read_text.return_value = "{broken"
        with patch("experiment_control.time.sleep") as sleep, self.assertRaises(json.JSONDecodeError):
            _read_object(path)
        self.assertEqual(path.read_text.call_count, 1)
        sleep.assert_not_called()

    def test_absent_manifest_still_returns_none(self):
        path = Mock()
        path.read_text.side_effect = FileNotFoundError()
        with patch("experiment_control.time.sleep") as sleep:
            self.assertIsNone(_read_object(path))
        self.assertEqual(path.read_text.call_count, 1)
        sleep.assert_not_called()


class ExperimentControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / "original"
        self.old.mkdir()
        self.reader = SimpleNamespace(directory=self.old, lock=threading.Lock(), cache={"old": "cached"})
        self.alias = self.root / "venv-python"
        self.alias.symlink_to(sys.executable)
        self.controller = ExperimentController(self.reader, self.alias, self.root / "vectors", self.root / "hf")
        self.controller._poll_interval = .005
        # These are coordination allowances, not timeout-behavior assertions.
        # Mounted Windows scratch storage can be slow while other work runs.
        # Tests of expiry below still set their own short, explicit deadlines.
        self.controller._stop_timeout = 5
        self.controller._startup_timeout = 5
        self.calls = []
        self.processes = []
        self.create_manifest = True
        self.create_control = True
        self.addCleanup(self.finish_all)

    def read_control(self, directory=None):
        # Observe the same completed write boundary as the real viewer. An
        # unlocked read can race replacement on mounted Windows filesystems,
        # and can see control.json before the matching event is appended.
        with self.reader.lock:
            selected = self.reader.directory if directory is None else directory
            return json.loads((selected / "control.json").read_text())

    def finish_all(self):
        for process in self.processes:
            if process.poll() is None:
                process.finish()
        time.sleep(.02)

    def fake_popen(self, argv, **kwargs):
        destination = Path(argv[argv.index("--out") + 1])
        self.assertFalse(destination.exists(), "Runner output directory must not be precreated")
        destination.mkdir()
        if self.create_manifest:
            write_json(destination / "manifest.json", {"status": "running", "manual_control": True})
        if self.create_control:
            initial = argv[argv.index("--initial-aux-effect") + 1]
            write_json(destination / "control.json", {"aux_enabled": initial == "on",
                       "pain_dose": 0, "opium_requests": 0, "revision": 0})
        process = FakeProcess(destination, 90000 + len(self.calls))
        self.calls.append((argv, kwargs))
        self.processes.append(process)
        return process

    def test_launch_is_fresh_fixed_argv_and_preserves_venv_symlink(self):
        write_json(self.old / "manifest.json", {"status": "complete"})
        (self.old / "episodes.jsonl").write_text("preserved recorded data\n")
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            accepted = self.controller.restart()
            self.assertTrue(accepted["accepted"])
            until(lambda: self.controller.status()["state"] == "running")
            first = self.reader.directory
            argv, kwargs = self.calls[0]
            self.assertIsInstance(argv, list)
            self.assertEqual(argv[0], str(self.alias.absolute()))
            self.assertNotEqual(argv[0], str(self.alias.resolve()))
            self.assertFalse(kwargs["shell"])
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
            self.assertEqual(argv[argv.index("--task-count") + 1], "24")
            self.assertEqual(argv[argv.index("--action-budget") + 1], "120")
            self.assertEqual(argv[argv.index("--token-budget") + 1], "8192")
            self.assertEqual(argv[argv.index("--initial-aux-effect") + 1], "on")
            self.assertEqual(first.parent, self.root)
            self.assertTrue(first.name.startswith("self-admin-toggle-"))
            self.assertEqual(self.reader.cache, {})
            self.assertEqual((self.old / "episodes.jsonl").read_text(), "preserved recorded data\n")
            self.processes[0].finish()
            until(lambda: self.controller.status()["state"] == "complete")
            self.controller.restart()
            until(lambda: len(self.calls) == 2 and self.controller.status()["state"] == "running")
            self.assertNotEqual(first, self.reader.directory)
            self.assertTrue((first / "manifest.json").exists())

    def test_active_stop_preserves_fields_and_busy_request_does_not_launch(self):
        write_json(self.old / "manifest.json", {"status": "running", "restart_supported": True})
        write_json(self.old / "control.json", {"pain_dose": 2.5, "opium_requests": 3, "revision": 9, "custom": {"keep": True}})
        (self.old / "episodes.jsonl").write_text("untouched\n")
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            self.controller.restart()
            until(lambda: self.read_control(self.old).get("stop_requested"))
            second = self.controller.restart()
            self.assertFalse(second["accepted"])
            self.assertTrue(second["restart_pending"])
            self.assertFalse(self.calls)
            control = self.read_control(self.old)
            self.assertEqual((control["pain_dose"], control["opium_requests"], control["revision"]), (2.5, 3, 10))
            self.assertEqual(control["custom"], {"keep": True})
            self.assertEqual(control["stop_reason"], "restart")
            event = json.loads((self.old / "control_events.jsonl").read_text())
            self.assertEqual((event["actor"], event["operation"]), ("human", "restart_experiment"))
            write_json(self.old / "manifest.json", {"status": "complete"})
            until(lambda: self.controller.status()["state"] == "running")
            self.assertEqual(len(self.calls), 1)
            self.assertEqual((self.old / "episodes.jsonl").read_text(), "untouched\n")

    def test_slow_initialization_keeps_restart_pending_without_overlap(self):
        self.create_manifest = False
        self.controller._startup_timeout = .025
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            self.controller.restart()
            until(lambda: len(self.calls) == 1)
            until(lambda: self.controller.status()["launch_error"] is not None)
            state = self.controller.restart()
            self.assertFalse(state["accepted"])
            self.assertTrue(state["restart_pending"])
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(state["state"], "initializing")
            write_json(self.reader.directory / "manifest.json", {"status": "running", "manual_control": True})
            until(lambda: self.controller.status()["state"] == "running")
            self.assertIsNone(self.controller.status()["launch_error"])

    def test_owned_process_must_exit_even_after_terminal_manifest(self):
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            self.controller.restart()
            until(lambda: self.controller.status()["state"] == "running")
            previous = self.processes[0]
            write_json(previous.directory / "manifest.json", {"status": "complete"})
            self.controller.restart()
            time.sleep(.025)
            self.assertEqual(len(self.calls), 1)
            self.assertTrue(self.controller.status()["restart_pending"])
            previous.finish()
            until(lambda: len(self.calls) == 2 and self.controller.status()["state"] == "running")

    def test_graceful_stop_timeout_never_kills_or_launches_and_can_retry(self):
        write_json(self.old / "manifest.json", {"status": "running", "manual_control": True})
        self.controller._stop_timeout = .025
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            self.controller.restart()
            until(lambda: self.controller.status()["state"] == "failed")
            self.assertFalse(self.controller.status()["restart_pending"])
            self.assertIn("not killed", self.controller.status()["launch_error"])
            self.assertFalse(self.calls)
            self.assertTrue(self.read_control(self.old)["stop_requested"])
            write_json(self.old / "manifest.json", {"status": "complete"})
            self.assertTrue(self.controller.restart()["accepted"])
            until(lambda: self.controller.status()["state"] == "running")
            self.assertEqual(len(self.calls), 1)

    def test_failed_launch_and_failed_child_both_allow_recovery(self):
        with patch("experiment_control.subprocess.Popen", side_effect=OSError("mock spawn failure")):
            self.controller.restart()
            until(lambda: self.controller.status()["state"] == "failed")
            self.assertIn("mock spawn failure", self.controller.status()["launch_error"])
            self.assertEqual(self.reader.directory, self.old)
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            self.controller.restart()
            until(lambda: self.controller.status()["state"] == "running")
            self.processes[0].finish(code=7, status="failed")
            until(lambda: self.controller.status()["state"] == "failed")
            self.assertEqual(self.controller.status()["return_code"], 7)
            self.assertIn("code 7", self.controller.status()["launch_error"])
            self.controller.restart()
            until(lambda: len(self.calls) == 2 and self.controller.status()["state"] == "running")

    def test_settings_cannot_inject_shell_or_arbitrary_flags(self):
        for settings in ({"task_count": "24; echo unsafe"}, {"extra_args": ["--other"]}, {"pace_seconds": float("nan")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ExperimentController(self.reader, self.alias, self.root / "vectors", self.root / "hf", settings)

    def test_aux_off_and_on_persist_across_repeated_fresh_runs_without_other_state(self):
        write_json(self.old / "manifest.json", {"status": "complete"})
        write_json(self.old / "control.json", {"aux_enabled": False, "pain_dose": 4,
                   "opium_requests": 9, "stop_requested": True, "revision": 5})
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            for index, expected in enumerate(("off", "off", "on", "on")):
                if index == 2:
                    control = self.read_control()
                    control["aux_enabled"] = True
                    write_json(self.reader.directory / "control.json", control)
                self.controller.restart()
                until(lambda: len(self.calls) == index + 1 and self.controller.status()["state"] == "running")
                argv = self.calls[index][0]
                self.assertEqual(argv[argv.index("--initial-aux-effect") + 1], expected)
                fresh = self.read_control()
                self.assertEqual(fresh, {"aux_enabled": expected == "on", "pain_dose": 0,
                                        "opium_requests": 0, "revision": 0})
                self.assertEqual(self.controller.status()["initial_aux_enabled"], expected == "on")
                self.processes[index].finish()
                until(lambda: self.controller.status()["state"] == "complete")

    def test_aux_setting_is_captured_after_graceful_stop_not_when_clicked(self):
        write_json(self.old / "manifest.json", {"status": "running", "manual_control": True})
        write_json(self.old / "control.json", {"aux_enabled": True, "revision": 0})
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            self.controller.restart()
            until(lambda: self.read_control(self.old).get("stop_requested"))
            with self.reader.lock:
                control = json.loads((self.old / "control.json").read_text())
                control["aux_enabled"] = False
                write_json(self.old / "control.json", control)
            write_json(self.old / "manifest.json", {"status": "complete"})
            until(lambda: self.controller.status()["state"] == "running")
            argv = self.calls[0][0]
            self.assertEqual(argv[argv.index("--initial-aux-effect") + 1], "off")

    def test_aux_off_survives_failed_startup_without_control_file(self):
        write_json(self.old / "manifest.json", {"status": "complete"})
        write_json(self.old / "control.json", {"aux_enabled": False})
        self.create_manifest = False
        self.create_control = False
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            self.controller.restart()
            until(lambda: len(self.processes) == 1 and self.reader.directory == self.processes[0].directory)
            self.assertFalse((self.reader.directory / "control.json").exists())
            self.processes[0].finish(code=2, status="failed")
            until(lambda: self.controller.status()["state"] == "failed")
            self.create_manifest = True
            self.create_control = True
            self.controller.restart()
            until(lambda: len(self.calls) == 2 and self.controller.status()["state"] == "running")
            for argv, _ in self.calls:
                self.assertEqual(argv[argv.index("--initial-aux-effect") + 1], "off")

    def test_malformed_aux_boolean_is_rejected_before_launch(self):
        write_json(self.old / "manifest.json", {"status": "complete"})
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            for value in (0, 1, "false", None, [], {}):
                with self.subTest(value=value):
                    write_json(self.old / "control.json", {"aux_enabled": value})
                    self.controller.restart()
                    until(lambda: self.controller.status()["state"] == "failed")
                    self.assertIn("must be a boolean", self.controller.status()["launch_error"])
                    self.assertFalse(self.calls)
                    self.assertEqual(self.reader.directory, self.old)

    def test_legacy_control_missing_aux_defaults_on(self):
        write_json(self.old / "manifest.json", {"status": "complete"})
        write_json(self.old / "control.json", {"pain_dose": 0, "revision": 4})
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            self.controller.restart()
            until(lambda: self.controller.status()["state"] == "running")
            argv = self.calls[0][0]
            self.assertEqual(argv[argv.index("--initial-aux-effect") + 1], "on")

    def test_stop_preserves_fields_is_idempotent_blocks_restart_and_never_spawns(self):
        write_json(self.old / "manifest.json", {"status": "running", "manual_control": True})
        original = {"aux_enabled": False, "pain_dose": 3, "opium_requests": 4, "revision": 7, "custom": "keep"}
        write_json(self.old / "control.json", original)
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            accepted = self.controller.stop()
            self.assertTrue(accepted["accepted"])
            until(lambda: self.read_control(self.old).get("stop_requested"))
            repeated = self.controller.stop()
            self.assertTrue(repeated["accepted"])
            self.assertTrue(repeated["already_pending"])
            self.assertTrue(repeated["stop_pending"])
            self.assertFalse(self.controller.restart()["accepted"])
            control = self.read_control(self.old)
            self.assertEqual(control["stop_reason"], "user_stop")
            self.assertEqual(control["revision"], 8)
            for key in ("aux_enabled", "pain_dose", "opium_requests", "custom"):
                self.assertEqual(control[key], original[key])
            events = (self.old / "control_events.jsonl").read_text().splitlines()
            self.assertEqual(len(events), 1)
            event = json.loads(events[0])
            self.assertEqual((event["actor"], event["operation"], event["stop_reason"]), ("human", "stop_experiment", "user_stop"))
            write_json(self.old / "manifest.json", {"status": "stopped", "manual_control": True})
            until(lambda: self.controller.status()["state"] == "stopped")
            self.assertFalse(self.controller.status()["stop_pending"])
            self.assertFalse(self.calls)
            self.assertEqual(self.reader.directory, self.old)

    def test_stop_of_complete_or_absent_run_is_harmless(self):
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            for state in (None, "complete", "stopped"):
                if state is not None:
                    write_json(self.old / "manifest.json", {"status": state})
                self.assertTrue(self.controller.stop()["accepted"])
                until(lambda: not self.controller.status()["stop_pending"])
                self.assertEqual(self.controller.status()["state"], "stopped")
                self.assertFalse((self.old / "control.json").exists())
                self.assertFalse(self.calls)

    def test_stop_is_rejected_while_restart_is_initializing(self):
        self.create_manifest = False
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            self.controller.restart()
            until(lambda: len(self.calls) == 1 and self.reader.directory != self.old)
            stopped = self.controller.stop()
            self.assertFalse(stopped["accepted"])
            self.assertEqual(stopped["reason"], "restart_pending")
            self.assertFalse(stopped["stop_pending"])
            self.assertTrue(stopped["restart_pending"])
            self.assertEqual(len(self.calls), 1)
            self.assertNotIn("stop_requested", self.read_control())
            write_json(self.reader.directory / "manifest.json", {"status": "running", "manual_control": True})
            until(lambda: self.controller.status()["state"] == "running")

    def test_stop_owned_run_waits_for_exit_without_replacement(self):
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            self.controller.restart()
            until(lambda: self.controller.status()["state"] == "running")
            current = self.reader.directory
            self.controller.stop()
            until(lambda: self.read_control(current).get("stop_reason") == "user_stop")
            write_json(current / "manifest.json", {"status": "stopped"})
            time.sleep(.025)
            self.assertTrue(self.controller.status()["stop_pending"])
            self.assertEqual(len(self.calls), 1)
            self.processes[0].finish(status="stopped")
            until(lambda: self.controller.status()["state"] == "stopped")
            self.assertFalse(self.controller.status()["stop_pending"])
            self.assertEqual(self.reader.directory, current)
            self.assertEqual(len(self.calls), 1)

    def test_stop_unsupported_active_run_rejected_without_write(self):
        write_json(self.old / "manifest.json", {"status": "running"})
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            self.controller.stop()
            until(lambda: self.controller.status()["state"] == "failed")
            self.assertIn("does not declare support", self.controller.status()["launch_error"])
            self.assertFalse(self.controller.status()["stop_pending"])
            self.assertFalse((self.old / "control.json").exists())
            self.assertFalse(self.calls)

    def test_child_failure_during_stop_is_not_reported_as_saved_stop(self):
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            self.controller.restart()
            until(lambda: self.controller.status()["state"] == "running")
            self.controller.stop()
            until(lambda: self.read_control().get("stop_reason") == "user_stop")
            self.processes[0].finish(code=5, status="failed")
            until(lambda: self.controller.status()["state"] == "failed")
            self.assertFalse(self.controller.status()["stop_pending"])
            self.assertIn("without a clean saved stop", self.controller.status()["launch_error"])
            self.assertEqual(len(self.calls), 1)

    def test_stop_timeout_keeps_request_without_kill_or_spawn(self):
        write_json(self.old / "manifest.json", {"status": "running", "manual_control": True})
        self.controller._stop_timeout = .025
        with patch("experiment_control.subprocess.Popen", side_effect=self.fake_popen):
            self.controller.stop()
            until(lambda: self.controller.status()["state"] == "failed")
            self.assertIn("not killed", self.controller.status()["launch_error"])
            self.assertFalse(self.controller.status()["stop_pending"])
            self.assertEqual(self.read_control(self.old)["stop_reason"], "user_stop")
            self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
