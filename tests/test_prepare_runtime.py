"""No installs/downloads: tiny owned subprocesses and fake volume telemetry."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from lab.resources import ResourceGuard, Volume
from prepare_runtime import child_environment, command_plan, run_command


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.low=False;self.guard=ResourceGuard({"root":self.root},reserve_bytes=1000000,emergency_bytes=65536,
            forbid_large_c_writes=False,resolver=lambda path:[Volume("shared",str(self.root))],
            telemetry=lambda path:{"free_bytes":0 if self.low else 8*2**30,"total_bytes":16*2**30})
    def plan(self,code="print('tiny fixture')",**kwargs):
        return command_plan([sys.executable,"-c",code],environment_dir=self.root/"venv",cache_dir=self.root/"cache",temp_dir=self.root/"tmp",
            planned_bytes=dict(environment=1000000,cache=1000000,temp=1000000),**kwargs)
    def test_plan_resolves_storage_but_does_not_create_or_run(self):
        plan=self.plan();self.assertFalse((self.root/"venv").exists())
        self.assertFalse(plan["shell"]);self.assertFalse(plan["allow_network"])
        for name,path in plan["destinations"].items():self.assertTrue(Path(path).is_absolute())
        env=child_environment(plan,{"PATH":"/usr/bin","HF_HOME":"/wrong"})
        self.assertEqual(env["PIP_NO_INDEX"],"1");self.assertEqual(env["HF_HUB_OFFLINE"],"1")
        self.assertEqual(env["TMPDIR"],str(self.root/"tmp"))
        self.assertTrue(env["HF_HOME"].startswith(str(self.root/"cache")))
        self.assertEqual(env["HF_HUB_DISABLE_XET"],"1")
        self.assertNotIn("PIP_NO_INDEX",child_environment(self.plan(allow_network=True),{"PIP_NO_INDEX":"1"}))
    def test_small_child_completes_with_bounded_output_and_receipt(self):
        plan=self.plan("print('x'*10000)")
        result=run_command(plan,guard=self.guard,record_dir=self.root/"records",poll_seconds=.01,log_limit=128)
        self.assertEqual(result["status"],"complete")
        self.assertEqual(result["log_bytes"],128)
        self.assertEqual(result["discarded_log_bytes"],10001-128)
        self.assertEqual(Path(result["log"]).stat().st_size,128)
        self.assertEqual(json.loads(Path(result["record"]).read_text())["command_plan_sha256"],plan["sha256"])
        self.assertEqual(self.guard.reservations,{})
    def test_plan_mutation_rejected_before_launch(self):
        plan=self.plan();plan["command"][2]="raise RuntimeError()"
        with patch("prepare_runtime.subprocess.Popen") as launch,self.assertRaisesRegex(ValueError,"changed"):
            run_command(plan,guard=self.guard)
        launch.assert_not_called()
    def test_cancelled_before_start_never_runs_child(self):
        event=threading.Event();event.set()
        with patch("prepare_runtime.subprocess.Popen") as launch:
            result=run_command(self.plan(),guard=self.guard,cancel_event=event,record_dir=self.root/"records")
        self.assertEqual(result["status"],"cancelled");launch.assert_not_called()
    def test_external_capacity_drop_kills_only_owned_blocked_child_and_saves_record(self):
        timer=threading.Timer(.12,lambda:setattr(self,"low",True));timer.start();self.addCleanup(timer.cancel)
        result=run_command(self.plan("import time; time.sleep(30)"),guard=self.guard,record_dir=self.root/"records",poll_seconds=.01)
        self.assertEqual(result["status"],"resource_stopped")
        self.assertIsNotNone(result["returncode"])
        self.assertEqual(json.loads(Path(result["record"]).read_text())["status"],"resource_stopped")
        self.assertEqual(self.guard.reservations,{})
    def test_shared_destinations_admission_aggregates_estimates(self):
        plan=command_plan([sys.executable,"-c","pass"],environment_dir=self.root/"same",cache_dir=self.root/"same",temp_dir=self.root/"same",
            planned_bytes=dict(environment=4*2**30,cache=4*2**30,temp=1))
        with patch("prepare_runtime.subprocess.Popen") as launch:
            result=run_command(plan,guard=self.guard,record_dir=self.root/"records")
        self.assertEqual(result["status"],"resource_stopped");launch.assert_not_called()


if __name__=="__main__":unittest.main()
