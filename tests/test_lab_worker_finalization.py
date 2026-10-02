"""A failed boundary must settle once without poisoning Stop or later sessions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lab.checkpoints import validate
from lab.effects import content_hash
from lab.recipes_v2 import resolve_recipe
from lab.resources import ResourceStop
from lab.worker import Worker
from tests.test_lab_worker import FakeRuntime,wrapped


class VersionString(str):
    """CPU-independent equivalent of the strict-JSON TorchVersion failure."""


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name);self.events=[];self.runtime=FakeRuntime()
        self.worker=Worker(str(self.path/'cache'),runtime=self.runtime,output=lambda e:self.events.append(deepcopy(e)))
        self.worker.model_info={'model_id':'Qwen/fixture','tool_call_format':'json'}
    def payload(self,id='run-first'):
        return dict(run_id=id,mode='experiment',config=resolve_recipe({'recipe_version':2,'demonstration':'none','task_count':1,
            'action_budget':1,'token_budget':64,'turn_token_limit':32}),calibration_id='cal-fixture',
            calibration_dir=str(self.path/'cal-fixture'),out_dir=str(self.path/id))
    def bad_version(self,value=None):
        fp={'model_id':'Qwen/fixture','torch':VersionString('2.8.0+cu128') if value is None else value}
        self.worker.model_info.update(fingerprint=fp,fingerprint_sha256=content_hash(fp))
    def terminal(self,id='run-first'):
        return [e for e in self.events if e.get('type')=='session_finished' and e.get('run_id')==id]
    def test_nonserializable_version_preserves_last_good_checkpoint_and_stop_next_session(self):
        self.worker.start_session(self.payload());file=self.path/'run-first'/'checkpoint.json'
        before=file.read_bytes();self.bad_version()
        self.worker.receive({'command':'stop','id':'stop-test'})
        self.assertTrue(self.worker.session['finished']);self.assertFalse(self.worker.paused)
        terminal=self.terminal();self.assertEqual(len(terminal),1);self.assertEqual(terminal[0]['status'],'failed')
        summary=terminal[0]['summary'];self.assertEqual(summary['termination'],'checkpoint_failed')
        self.assertFalse(summary['checkpoint_error']['latest_boundary_saved'])
        self.assertTrue(summary['checkpoint_error']['previous_checkpoint_present'])
        self.assertEqual(summary['requested_termination'],'stopped_by_user')
        self.assertEqual(file.read_bytes(),before);validate(json.loads(before))
        self.assertTrue(any(e['type']=='job' and e.get('command_id')=='stop-test' and e['status']=='stopped' for e in self.events))
        self.worker.finish('stopped','again');self.assertEqual(len(self.terminal()),1)
        self.worker.model_info['fingerprint']['torch']=str(self.worker.model_info['fingerprint']['torch'])
        self.worker.stop.clear();self.worker.start_session(self.payload('run-next'))
        record=self.worker.environment.records[0]
        self.runtime.scripts=[{'raw_text':wrapped('submit_answer',{'answer':record['answer']})}]
        self.worker.run_experiment()
        self.assertEqual(self.terminal('run-next')[0]['status'],'complete')
        self.assertEqual(self.terminal('run-next')[0]['summary']['correct'],1)
    def test_real_torch_version_initial_failure_is_terminal_and_does_not_kill_worker_loop(self):
        try:from torch.torch_version import TorchVersion
        except ImportError:self.skipTest('Torch unavailable; equivalent plain-string subclass covered above')
        self.bad_version(TorchVersion('2.8.0+cu128'))
        old=self.worker.output
        def output(event):
            old(event)
            if event.get('type')=='session_finished' and event.get('run_id')=='run-first':
                self.worker.model_info['fingerprint']['torch']=str(self.worker.model_info['fingerprint']['torch'])
        self.worker.output=output;self.runtime.scripts=[{'raw_text':'invalid'}]
        for id in ('run-first','run-next'):
            self.worker.jobs.put({'id':id,'command':'start_session','payload':self.payload(id)})
        self.worker.jobs.put(None)
        with patch('lab.worker.traceback.print_exc'):self.worker.loop()
        self.assertEqual(self.terminal()[0]['status'],'failed')
        self.assertFalse(self.terminal()[0]['summary']['checkpoint_error']['previous_checkpoint_present'])
        self.assertEqual(self.terminal('run-next')[0]['status'],'complete')
        self.assertEqual(len(self.runtime.calls),1);self.assertFalse(self.worker.active)
    def test_resource_failure_stays_on_emergency_path(self):
        self.worker.start_session(self.payload())
        with patch.object(self.worker,'checkpoint',side_effect=ResourceStop('full')):
            with self.assertRaises(ResourceStop):self.worker.finish('complete','tasks_complete')
        self.assertFalse(self.worker.session['finished']);self.assertEqual(self.terminal(),[])
        self.assertFalse(any(e['type']=='checkpoint_error' for e in self.events))
    def test_resource_failure_during_exception_cleanup_does_not_kill_loop(self):
        with patch.object(self.worker,'checkpoint',side_effect=[ValueError('primary failure'),ResourceStop('reserve during cleanup')]),patch('lab.worker.traceback.print_exc'):
            self.worker.jobs.put({'id':'start','command':'start_session','payload':self.payload()});self.worker.jobs.put(None)
            self.worker.loop()
        self.assertEqual(self.terminal()[0]['status'],'resource_stopped')
        self.assertTrue(self.worker.session['finished']);self.assertFalse(self.worker.active)
        self.assertTrue(any(e['type']=='job' and e.get('command_id')=='start' and e['status']=='resource_stopped' for e in self.events))
    def test_idle_stop_storage_error_uses_emergency_terminal(self):
        self.worker.start_session(self.payload())
        with patch.object(self.worker,'checkpoint',side_effect=OSError('disk full')):
            self.worker.receive({'command':'stop','id':'stop-test'})
        self.assertEqual(self.terminal()[0]['status'],'resource_stopped')
        self.assertTrue(self.worker.session['finished'])
        self.assertEqual([e for e in self.events if e['type']=='job'][-1]['status'],'resource_stopped')
    def test_failed_final_checkpoint_never_claims_success(self):
        self.worker.start_session(self.payload());self.bad_version()
        self.worker.finish('complete','tasks_complete')
        result=self.terminal()[0];self.assertEqual(result['status'],'failed')
        self.assertEqual(result['summary']['requested_status'],'complete')
        self.assertTrue(self.worker.session['finished'])
