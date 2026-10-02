"""Manual CLI frozen launch, explicit retries, polling and review-only outputs."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_research_protocol import main


class Client:
    def __init__(self):self.commands=[];self.gets=[];self.responses=[]
    def command(self,name,payload):
        self.commands.append((name,deepcopy(payload)))
        return {'accepted':True,'command_id':'fixture','research_job_id':'research-fixture'}
    def get(self,path):
        self.gets.append(path)
        return self.responses.pop(0)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.client=Client();self.urls=[]
    def factory(self,url):self.urls.append(url);return self.client
    def call(self,*args):
        with redirect_stdout(io.StringIO()) as output:main(list(args),self.factory)
        return output.getvalue()
    def preview(self):
        path=self.root/'preview.json'
        self.call('dry-run','--protocol','task_pressure','--seeds','17','--output',str(path))
        return path
    def test_list_and_preview_never_contact_server(self):
        catalog=json.loads(self.call('list'));self.assertTrue(any(row['id']=='task_pressure' for row in catalog))
        preview=json.loads(self.preview().read_text());self.assertEqual(preview['seeds'],[17]);self.assertEqual(self.urls,[])
    def test_launch_exact_preview_and_preserve_bindings(self):
        path=self.preview();bindings=self.root/'bindings.json';bindings.write_text(json.dumps({'source_run_id':'recorded-run'}))
        self.call('--url','http://127.0.0.1:8888','launch','--preview',str(path),'--calibration','cal-id','--bindings',str(bindings))
        command,payload=self.client.commands[0]
        self.assertEqual(command,'start_protocol');self.assertEqual(payload['bindings'],{'source_run_id':'recorded-run'})
        self.assertEqual(payload['expansion_sha256'],json.loads(path.read_text())['expansion_sha256'])
        self.assertEqual(self.urls,['http://127.0.0.1:8888'])
    def test_tampered_preview_and_existing_output_refuse_before_launch(self):
        path=self.preview();value=json.loads(path.read_text());value['episodes'][0]['recipe']['action_budget']+=1;path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'preview differs'):self.call('launch','--preview',str(path),'--calibration','cal-id')
        with self.assertRaisesRegex(ValueError,'Output exists'):self.call('resume','job','--expansion-sha256','a'*64,'--output',str(path))
        self.assertEqual(self.client.commands,[]);self.assertEqual(self.urls,[])
    def test_resume_retry_is_explicit_and_analysis_declares_attempt_policy(self):
        self.call('resume','research-fixture','--expansion-sha256','a'*64)
        self.assertFalse(self.client.commands[-1][1]['retry_failed'])
        self.call('resume','research-fixture','--expansion-sha256','a'*64,'--retry-failed')
        self.assertTrue(self.client.commands[-1][1]['retry_failed'])
        self.call('analysis','research-fixture','--endpoint','task_accuracy','--arm-a','arm=active','--arm-b','arm=sham','--attempt-policy','latest')
        name,payload=self.client.commands[-1];self.assertEqual(name,'analyze_protocol');self.assertEqual(payload['attempt_policy'],'latest')
    def test_poll_stops_on_resource_failure_and_keeps_last_receipt(self):
        self.client.responses=[{'settled':False,'receipt':{'status':'running'},'status_counts':{'running':1}},
            {'settled':False,'receipt':{'status':'resource_stopped'},'status_counts':{'resource_stopped':1}}]
        target=self.root/'status.json'
        with patch('run_research_protocol.time.sleep') as sleep:self.call('poll','research-fixture','--wait','--output',str(target))
        self.assertEqual(sleep.call_count,1);self.assertEqual(json.loads(target.read_text())['receipt']['status'],'resource_stopped')
    def test_poll_timeout_does_not_stop_job(self):
        self.client.responses=[{'receipt':{'status':'running'}}]
        with patch('run_research_protocol.time.monotonic',side_effect=[0,2]),self.assertRaisesRegex(TimeoutError,'continues'):
            self.call('poll','research-fixture','--wait','--timeout','1')
        self.assertEqual(self.client.commands,[])


if __name__=='__main__':unittest.main()
