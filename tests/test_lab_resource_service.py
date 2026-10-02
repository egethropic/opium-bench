"""Owned-worker supervision and emergency evidence under simulated capacity loss."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lab.resources import ResourceGuard,ResourceStop,Volume,EmergencyMetadata
from lab.service import LabService
from test_lab_service import RecordingService,FakeProcess


class ResourceServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.service=RecordingService(self.root);self.free=10000000
        guard=ResourceGuard({'data':self.service.store.root},reserve_bytes=100000,emergency_bytes=10000,
            resolver=lambda path:(Volume('fixture',str(self.root)),),telemetry=lambda path:dict(free_bytes=self.free,total_bytes=10000000))
        self.service.resources=guard;self.service.store.guard=guard
        self.run,path=self.service.store.create('experiment',{'task_count':1})
        (path/'checkpoint.json').write_bytes(b'last-safe-boundary')
        self.path=path
    def own(self,process):
        self.service.process=process;self.service.process_runs[process]={self.run}
        self.service.session=dict(id=self.run,status='running',mode='experiment',metrics={'tokens':1},config={})
        self.service.store.update(self.run,status='running')
    def test_reserve_failure_during_output_preserves_checkpoint_and_finalizes_run(self):
        process=FakeProcess(lines=[json.dumps({'type':'token','run_id':self.run,'text':'next','seq':1})+'\n'])
        self.own(process);before=(self.path/'manifest.json').read_bytes();self.free=0
        self.service._read_worker(process)
        saved=self.service.store.read_run(self.run)
        self.assertEqual(saved['manifest']['status'],'resource_stopped')
        self.assertEqual((self.path/'manifest.json').read_bytes(),before)
        self.assertEqual((self.path/'checkpoint.json').read_bytes(),b'last-safe-boundary')
        self.assertEqual(saved['summary']['termination'],'storage_reserve')
        self.assertEqual(self.service.job['status'],'resource_stopped')
    def test_watchdog_marks_abort_before_cancelling_owned_process(self):
        process=FakeProcess();process.returncode=-15;self.own(process)
        slot=EmergencyMetadata(self.root/'emergency').allocate(self.service.resources)
        self.free=0
        def cancel(owned,**kwargs):
            self.assertIs(owned,process)
            self.assertEqual(self.service.store.read_run(self.run)['manifest']['status'],'resource_stopped')
            self.assertTrue(kwargs['process_group'])
        with patch('lab.resources.cancel_owned_process',side_effect=cancel) as cancel_call:
            self.service._supervise_worker(process,'fixture-op',slot)
        cancel_call.assert_called_once()
        self.assertTrue(list((self.root/'emergency').glob('resource-stop-*.json')))
        self.assertFalse(slot.path.exists())
    def test_prelaunch_refusal_never_spawns_or_leaves_queued_claim(self):
        self.free=0
        with patch('lab.service.subprocess.Popen') as popen:
            with self.assertRaises(ResourceStop):
                LabService._send(self.service,'start_session',{'run_id':self.run})
        popen.assert_not_called()
        self.assertEqual(self.service.store.read_run(self.run)['manifest']['status'],'resource_stopped')
        self.assertFalse(self.service.pending)


if __name__=='__main__':unittest.main()
