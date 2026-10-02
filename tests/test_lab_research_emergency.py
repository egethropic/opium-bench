"""Durable job-level failure evidence when ordinary writes are unavailable."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from unittest.mock import patch

from lab.protocol_runner import ProtocolRunner
from lab.research_jobs import (prepare_dispatch,bind_dispatch_worker,dispatch_stop,
    record_dispatch_stop,release_dispatch,recover_dispatches,read_job,catalog,
    load_plan,save_receipt,analysis_records)
from lab.resources import ResourceGuard,ResourceStop,Volume
from lab.service import LabService
from lab.storage import read_json
from tests.test_lab_protocol_runner import RunnerFixture,document
from test_lab_service import RecordingService,FakeProcess


class EmergencyTests(RunnerFixture):
    def guard(self):
        self.free=10_000_000
        self.store.guard=ResourceGuard({'data':self.store.root},reserve_bytes=100_000,emergency_bytes=10_000,
            resolver=lambda path:(Volume('fixture',str(self.store.root)),),
            telemetry=lambda path:dict(free_bytes=self.free,total_bytes=10_000_000))
    def failure_job(self):
        return self.job(document([self.calibration_stage()],arms=('active','sham')),{'calibration_id':'cal-test'})
    def test_reserve_loss_after_main_keeps_primary_bytes_and_partial_denominators(self):
        preview,receipt,entries=self.failure_job();self.guard();saved={}
        def fail(item,spec,entry,target,receipt):
            path=self.store.run_path(entry['run_id']);saved['manifest']=(path/'manifest.json').read_bytes()
            (path/'checkpoint.json').write_bytes(b'last-safe-boundary');saved['receipt']=(self.store.root/'research'/receipt['id']/'receipt.json').read_bytes()
            directory=self.store.root/'research'/receipt['id']/'dispatches'/receipt['active_execution_id']
            self.assertEqual(len(list(directory.glob('.emergency-*'))),2)
            self.free=0;raise ResourceStop('capacity lost during diagnostic')
        self.backend.run_stage=fail
        with self.assertRaises(ResourceStop):self.runjob(preview,receipt)
        raw=load_plan(self.store,receipt['id'])[0];self.assertEqual(raw['status'],'running')
        self.assertEqual((self.store.root/'research'/receipt['id']/'receipt.json').read_bytes(),saved['receipt'])
        path=self.store.run_path(entries[0]['run_id'])
        self.assertEqual((path/'manifest.json').read_bytes(),saved['manifest'])
        self.assertEqual((path/'checkpoint.json').read_bytes(),b'last-safe-boundary')
        state=read_job(self.store,receipt['id'],verify=True)
        self.assertEqual(state['receipt']['status'],'resource_stopped');self.assertFalse(state['all_complete'])
        self.assertEqual(state['status_counts'],{'queued':1,'resource_stopped':1})
        units=state['states'][0]['units'];self.assertEqual([u['status'] for u in units],['complete','resource_stopped'])
        self.assertTrue(units[1]['summary']['consumption_unknown'])
        self.assertEqual(catalog(self.store)[0]['status'],'resource_stopped')
        records=analysis_records(self.store,receipt['id']);self.assertEqual(len(records),2)
        self.assertEqual(records[0]['status'],'partial');self.assertEqual(records[0]['outcomes']['task_accuracy'],{'numerator':1,'denominator':1})
    def test_transient_resource_error_also_halts_instead_of_running_later_episodes(self):
        preview,receipt,_=self.failure_job()
        def fail(*args):raise ResourceStop('reserve reached')
        self.backend.run_stage=fail
        with self.assertRaises(ResourceStop):self.runjob(preview,receipt)
        self.assertEqual(len(self.backend.calls),1)
        state=read_job(self.store,receipt['id'],verify=True)
        self.assertEqual(state['receipt']['status'],'resource_stopped')
        self.assertEqual(state['states'][0]['units'][1]['status'],'resource_stopped')
    def test_independent_actor_slots_are_atomic_idempotent_and_strict(self):
        _,receipt,_=self.job();execution=prepare_dispatch(self.store,receipt['id']);self.guard();self.free=0
        def record(actor):return record_dispatch_stop(self.store,receipt['id'],execution,'resource_stopped','capacity',actor=actor)
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(record,['worker','supervisor']*8))
        directory=self.store.root/'research'/receipt['id']/'dispatches'/execution
        paths=list(directory.glob('resource-stop-*.json'));self.assertEqual(len(paths),2)
        before={p.name:p.read_bytes() for p in paths};record('worker');record('supervisor')
        self.assertEqual(before,{p.name:p.read_bytes() for p in paths})
        self.assertEqual(dispatch_stop(self.store,receipt['id'],execution)['status'],'resource_stopped')
        paths[0].write_text('{}')
        with self.assertRaisesRegex(ValueError,'integrity'):read_job(self.store,receipt['id'])
    def test_normal_completion_releases_headroom_only_after_durable_terminal(self):
        preview,receipt,_=self.job();execution=prepare_dispatch(self.store,receipt['id'])
        with self.assertRaisesRegex(ValueError,'no durable'):release_dispatch(self.store,receipt['id'],execution,actor='all')
        ProtocolRunner(self.store,self.backend).run(receipt['id'],preview['expansion_sha256'],execution_id=execution)
        directory=self.store.root/'research'/receipt['id']/'dispatches'/execution
        self.assertEqual(len(list(directory.glob('.emergency-*'))),1) # Supervisor still owns its reserved slot.
        release_dispatch(self.store,receipt['id'],execution,actor='all')
        self.assertFalse(list(directory.glob('.emergency-*')));self.assertIsNone(dispatch_stop(self.store,receipt['id'],execution))
        self.assertIsNone(record_dispatch_stop(self.store,receipt['id'],execution,'failed','late process exit'))
        self.assertTrue(read_job(self.store,receipt['id'],verify=True)['all_complete'])
    def test_retry_keeps_historical_stop_and_new_execution_can_complete(self):
        preview,receipt,_=self.failure_job();original=self.backend.run_stage
        self.backend.run_stage=lambda *args:(_ for _ in ()).throw(ResourceStop('fixture'))
        with self.assertRaises(ResourceStop):self.runjob(preview,receipt)
        old=load_plan(self.store,receipt['id'])[0]['active_execution_id'];self.backend.run_stage=original
        self.runjob(preview,receipt,resume=True,retry_failed=True)
        state=read_job(self.store,receipt['id'],verify=True)
        self.assertTrue(state['all_complete']);self.assertNotEqual(state['receipt']['active_execution_id'],old)
        self.assertEqual(state['states'][0]['status'],'resource_stopped')
        self.assertEqual(dispatch_stop(self.store,receipt['id'],old)['status'],'resource_stopped')
        self.assertEqual(catalog(self.store)[0]['status'],'complete')
    def test_startup_recovery_requires_both_owners_gone(self):
        _,receipt,_=self.job();execution=prepare_dispatch(self.store,receipt['id'])
        bind_dispatch_worker(self.store,receipt['id'],execution,987654)
        recover_dispatches(self.store,lambda pid:pid==987654)
        self.assertIsNone(dispatch_stop(self.store,receipt['id'],execution))
        recover_dispatches(self.store,lambda pid:False)
        self.assertEqual(read_job(self.store,receipt['id'])['receipt']['status'],'failed')
        directory=self.store.root/'research'/receipt['id']/'dispatches'/execution
        self.assertFalse(list(directory.glob('.emergency-*')))


class ServiceEmergencyTests(RunnerFixture):
    def setUp(self):
        super().setUp();self.service=RecordingService(Path(self.tmp.name)/'service')
        self.service.store=self.store;self.process=FakeProcess();self.service.process=self.process
        self.service.worker={'status':'ready','model':None,'error':None}
    def dispatch(self,receipt,entries):
        with patch.object(self.service,'_launch'):
            result=LabService._send(self.service,'run_research_job',{'job_id':receipt['id'],'entries':entries})
        payload=json.loads(self.process.stdin.getvalue().splitlines()[-1])['payload']
        return result['command_id'],payload['execution_id']
    def make_inflight(self):
        preview,receipt,entries=self.job(document([self.calibration_stage()]),{'calibration_id':'cal-test'})
        command,execution=self.dispatch(receipt,entries)
        def killed(*args):raise KeyboardInterrupt('simulate process death')
        # Construct real durable main evidence and a running extra stage, then
        # suppress worker reporting to model an abrupt process kill.
        self.backend.run_stage=killed
        with patch('lab.protocol_runner.record_dispatch_stop'),patch('lab.protocol_runner.release_dispatch'):
            with self.assertRaises(KeyboardInterrupt):ProtocolRunner(self.store,self.backend).run(receipt['id'],preview['expansion_sha256'],execution_id=execution)
        return receipt,entries,command,execution
    def test_watchdog_and_exit_cover_job_after_main_is_complete(self):
        receipt,entries,command,execution=self.make_inflight();path=self.store.run_path(entries[0]['run_id'])
        before=(path/'manifest.json').read_bytes()
        self.service._resource_failure(self.process,ResourceStop('storage lost'))
        self.assertEqual(read_job(self.store,receipt['id'])['receipt']['status'],'resource_stopped')
        self.service._worker_exited(self.process,-15);self.service._worker_exited(self.process,-15)
        self.assertEqual((path/'manifest.json').read_bytes(),before)
        self.assertNotIn(command,self.service.research_commands)
        directory=self.store.root/'research'/receipt['id']/'dispatches'/execution
        self.assertEqual(len(list(directory.glob('resource-stop-*.json'))),1);self.assertFalse(list(directory.glob('.emergency-*')))
    def test_plain_worker_exit_produces_terminal_failed_receipt(self):
        receipt,_,_,_=self.make_inflight();self.service._worker_exited(self.process,1)
        self.assertEqual(read_job(self.store,receipt['id'])['receipt']['status'],'failed')
        self.assertEqual(catalog(self.store)[0]['status'],'failed')
    def test_premature_success_event_cannot_discard_unfinished_job(self):
        _,receipt,entries=self.job();command,execution=self.dispatch(receipt,entries)
        self.service.event({'type':'job','command_id':command,'status':'complete'})
        self.assertEqual(read_job(self.store,receipt['id'])['receipt']['status'],'failed')
        self.assertIn('without a durable',dispatch_stop(self.store,receipt['id'],execution)['reason'])
    def test_dispatch_failure_is_reserved_before_process_write(self):
        _,receipt,entries=self.job()
        with patch.object(self.service,'_launch',side_effect=OSError('spawn failed')):
            with self.assertRaises(OSError):LabService._send(self.service,'run_research_job',{'job_id':receipt['id'],'entries':entries})
        self.assertEqual(read_job(self.store,receipt['id'])['receipt']['status'],'failed')
        self.assertFalse(self.service.research_commands)
    def test_normal_service_completion_preserves_success_and_releases_slots(self):
        preview,receipt,entries=self.job();command,execution=self.dispatch(receipt,entries)
        ProtocolRunner(self.store,self.backend).run(receipt['id'],preview['expansion_sha256'],execution_id=execution)
        self.service.event({'type':'job','command_id':command,'status':'complete'})
        self.assertTrue(read_job(self.store,receipt['id'],verify=True)['all_complete'])
        self.assertFalse(list((self.store.root/'research'/receipt['id']/'dispatches'/execution).glob('.emergency-*')))
