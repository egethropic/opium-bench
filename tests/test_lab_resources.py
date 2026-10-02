"""Resource limits with fake capacity and bounded owned-process fixtures."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from lab.resources import (EmergencyMetadata, ResourceGuard, ResourceStop, Volume,
                           cancel_owned_process, resolve_volumes, supervise_process)


class ResourceGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.free = 1000
        self.calls = 0
        self.now = 0
        self.guard = ResourceGuard({"data": self.root / "data", "cache": self.root / "cache"},
                                  reserve_bytes=100, emergency_bytes=10, max_write_bytes=64,
                                  telemetry=self.telemetry, resolver=self.resolver,
                                  clock=lambda: self.now)

    def resolver(self, path):
        return [Volume("shared", str(self.root))]

    def telemetry(self, path):
        self.calls += 1
        return dict(free_bytes=self.free, total_bytes=2000)

    def test_shared_volume_reservations_are_summed_not_double_claimed(self):
        self.guard.reserve("generation", {self.root / "data": 400})
        self.guard.reserve("download", {self.root / "cache": 400})
        with self.assertRaises(ResourceStop) as error:
            self.guard.reserve("export", {self.root / "export": 100})
        self.assertEqual(error.exception.to_dict()["status"], "resource_stopped")
        self.assertNotIn("export", self.guard.reservations)
        self.assertEqual(len(self.guard.status()["volumes"]), 1)
        self.guard.release("download")
        self.guard.reserve("export", {self.root / "export": 100})

    def test_chunk_completion_releases_only_consumed_reservation(self):
        path = self.root / "data"
        self.guard.reserve("run", {path: 300})
        self.guard.before_write("run", path, 60)
        self.free -= 60
        self.guard.written("run", path, 60)
        self.assertEqual(self.guard.status()["reservations"]["run"], 240)
        self.assertEqual(self.guard.check(force=True)["volumes"][0]["available_bytes"], 590)
        with self.assertRaises(ValueError):
            self.guard.before_write("run", path, 65)
        with self.assertRaises(ValueError):
            self.guard.written("run", path, 241)
        with self.assertRaises(ValueError):
            self.guard.before_write("run", self.root / "wrong", 1)

    def test_external_drop_stops_before_next_write_and_retains_evidence(self):
        path = self.root / "evidence.jsonl"
        self.guard.reserve("run", {path: 100})
        with self.guard.write(path, 10, operation="run"):
            path.write_bytes(b"{}\n{}\n{}\n ")
        before = path.read_bytes()
        self.free = 180
        with self.assertRaises(ResourceStop):
            with self.guard.write(path, 10, operation="run"):
                path.write_bytes(b"should not execute")
        self.assertEqual(path.read_bytes(), before)

    def test_monitoring_is_bounded_and_chunk_checks_are_forced(self):
        self.guard.check()
        calls = self.calls
        for _ in range(50):
            self.guard.check()
        self.assertEqual(self.calls, calls)
        self.now = 2
        self.guard.check()
        self.assertEqual(self.calls, calls + 1)
        for index in range(100):
            self.now = index + 3
            self.guard.check()
        self.assertEqual(len(self.guard.status()["trend"]), 60)

    def test_wsl_guest_and_shared_host_pool_are_both_reserved(self):
        resolver = lambda path: ([Volume("guest", "/fake/guest"), Volume("host", "/fake/host", "wsl_backing")]
                                 if path.endswith("guest") else [Volume("host", "/fake/host", "host_volume")])
        guard = ResourceGuard(reserve_bytes=100, emergency_bytes=0, resolver=resolver,
                              telemetry=lambda path: dict(free_bytes=1000, total_bytes=2000))
        guard.reserve("calibrate", {self.root / "guest": 500})
        with self.assertRaises(ResourceStop) as caught:
            guard.reserve("download", {self.root / "host": 401})
        self.assertEqual([v["id"] for v in caught.exception.volumes], ["host"])

    def test_resolver_follows_destination_symlink_and_accounts_wsl_backing(self):
        target = self.root / "target"
        target.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(target, target_is_directory=True)
        with patch("lab.resources.wsl_backing_volume", return_value=("/mnt/d", "registry")):
            pools = resolve_volumes(alias / "not-yet-created")
        self.assertEqual(pools[0].path, str(target))
        if target.stat().st_dev == Path("/").stat().st_dev:
            self.assertEqual(pools[-1].id, "windows:d")
        direct = resolve_volumes("/mnt/d/cache/example")
        self.assertEqual([p.id for p in direct], ["windows:d"])

    def test_large_writes_to_c_or_c_backed_wsl_are_rejected_before_admission(self):
        guard = ResourceGuard(reserve_bytes=0, emergency_bytes=0,
                              resolver=lambda path: [Volume("windows:c", "/fake/c", "wsl_backing")],
                              telemetry=lambda path: dict(free_bytes=2**40, total_bytes=2**41))
        with self.assertRaisesRegex(ResourceStop, "cannot use C:"):
            guard.reserve("weights", {self.root: 64 * 1024**2})
        self.assertEqual(guard.reservations, {})

    def test_streaming_download_reconciliation_does_not_double_charge_written_bytes(self):
        path = self.root / "cache"
        self.guard.preflight({path: 500})
        self.guard.reserve("download", {path: 500})
        self.free -= 400
        self.guard.update_remaining("download", {path: 100})
        self.assertEqual(self.guard.status()["volumes"][0]["available_bytes"], 390)
        with self.assertRaises(ValueError):
            self.guard.update_remaining("download", {path: 101})

    def test_emergency_allowance_yields_small_final_record_without_rewriting_evidence(self):
        slot = EmergencyMetadata(self.root, allowance_bytes=1024).allocate()
        self.assertEqual(slot.path.stat().st_size, 1024)
        evidence = self.root / "evidence.json"
        evidence.write_text('{"intact":true}')
        result = slot.finalize(ResourceStop("reserve", operation="test"))
        self.assertFalse(slot.path.exists())
        self.assertEqual(json.loads(result.read_text())["status"], "resource_stopped")
        self.assertEqual(evidence.read_text(), '{"intact":true}')

    def test_blocked_child_is_actually_terminated_on_resource_drop(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(lambda: cancel_owned_process(process, grace_seconds=.2))
        self.free = 0
        result = supervise_process(process, self.guard, poll_seconds=.01, grace_seconds=.2)
        self.assertEqual(result["status"], "resource_stopped")
        self.assertIsNotNone(process.poll())

    def test_user_cancel_terminates_child_and_is_not_completion(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(lambda: cancel_owned_process(process, grace_seconds=.2))
        cancelled = threading.Event()
        cancelled.set()
        result = supervise_process(process, self.guard, cancel_event=cancelled, poll_seconds=.01, grace_seconds=.2)
        self.assertEqual(result["status"], "cancelled")
        self.assertIsNotNone(process.poll())

    def test_kill_fallback_for_nonresponsive_owned_child(self):
        class Child:
            returncode = None
            terminated = False
            killed = False
            def poll(self): return self.returncode
            def terminate(self): self.terminated = True
            def kill(self): self.killed = True
            def wait(self, timeout):
                if not self.killed:
                    raise subprocess.TimeoutExpired("fixture", timeout)
                self.returncode = -9
                return -9
        child = Child()
        self.assertEqual(cancel_owned_process(child), -9)
        self.assertTrue(child.terminated and child.killed)


if __name__ == "__main__":
    unittest.main()
