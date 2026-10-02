"""Complete-envelope execution against fake backends and a real CPU fake worker."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from lab.effects import content_hash
from lab.protocol_library import dry_run,freeze_protocol,load_protocol
from lab.protocol_runner import LocalWorkerBackend,ProtocolRunner,RUNNER_CAPABILITIES,prepare_episode
from lab.research_jobs import create_job,read_job,analysis_records,analyze_job,load_plan,catalog,save_receipt
from lab.storage import Store,atomic_json,read_json
from lab.worker import Worker
from tests.test_lab_worker import FakeRuntime,wrapped


def document(stages=(),*,arms=('active',),changes=None):
    d=load_protocol('task_pressure');d.pop('content_sha256');d['id']='runner-fixture';d['title']='Runner fixture';d['base_recipe'].update(
        demonstration='none',task_count=1,action_budget=2,token_budget=32,turn_token_limit=16,seed=17)
    if changes:d['base_recipe'].update(changes)
    d['axes']=[{'name':'arm','levels':[{'id':arm,'recipe':{'condition':arm}} for arm in arms]}]
    d['smoke_axes']={'arm':list(arms)};d['pairing_factors']=[];d['stages']=list(stages);d['seeds']={'smoke':[17],'full':[17]}
    d['analysis']['bootstrap_iterations']=100
    return freeze_protocol(d)


def stage(kind,id='extra',config=None,bindings=(),budget=0):
    return {'kind':kind,'id':id,'config':config or {},'requires_bindings':list(bindings),'token_budget':budget,'notes':'Fixture stage'}


def criterion():
    return {'id':'observable','kind':'objective_behavior','description':'specific observable language choice',
        'validation':{'status':'unvalidated','independent_of_intervention_probe':True,'evidence_sha256':None,
            'context_families':[],'endpoint_id':'observable','examples':0}}


class FakeBackend:
    capabilities=RUNNER_CAPABILITIES
    model_info={'model_id':'fixture','fingerprint_sha256':'a'*64}
    def __init__(self,store):self.store=store;self.calls=[];self.fail_main=False;self.fail_stage=False;self.partial_stage=False
    def admit(self,receipt,plan):pass
    def run_main(self,item,entry,target,receipt):
        self.calls.append((entry['episode_id'],'main',entry['attempt']))
        summary={'correct':0 if self.fail_main else 1,'assigned':1,'voluntary_calls':1,'completed_decisions':2,'invalid_actions':0,'tokens':4,'actions':2}
        status='failed' if self.fail_main else 'complete'
        self.store.append(entry['run_id'],{'type':'session_finished','status':status,'summary':summary})
        self.store.update(entry['run_id'],status=status,summary=summary)
        atomic_json(target/'main.json',summary)
        return {'status':status,'summary':summary}
    def run_stage(self,item,spec,entry,target,receipt):
        self.calls.append((entry['episode_id'],spec['id'],entry['attempt']))
        if self.fail_stage:raise ValueError('Injected stage failure')
        return {'status':'partial' if self.partial_stage else 'complete','summary':{'executed':spec['kind'],'tokens':0}}


class RunnerFixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.store=Store(self.tmp.name)
        # Frozen local fixture identity keeps parallel repository edits from
        # changing these deterministic execution tests.
        self.sources={'commit':'fixture','dirty':False,'source_sha256':{'lab/fixture.py':'a'*64}}
        self.addCleanup(patch.stopall);patch('lab.research_jobs.source_manifest',return_value=self.sources).start();patch('lab.storage.source_manifest',return_value=self.sources).start()
        cal=self.store.calibrations/'cal-test';atomic_json(cal/'calibration.json',{'schema_version':2});(cal/'vectors.npz').write_bytes(b'fixture');(cal/'activations.npz').write_bytes(b'activations');atomic_json(cal/'corpus.json',{'rows':[]})
        self.backend=FakeBackend(self.store)
    def job(self,doc=None,bindings=None):
        preview=dry_run(doc or document());receipt,entries=create_job(self.store,preview,'cal-test',bindings=bindings,capabilities=RUNNER_CAPABILITIES)
        return preview,receipt,entries
    def runjob(self,preview,receipt,**kwargs):return ProtocolRunner(self.store,self.backend).run(receipt['id'],preview['expansion_sha256'],**kwargs)
    def calibration_stage(self):return stage('calibration-validation',config={'doses':[0,.25],'validation_pairs_per_concept':1,'continuation_tokens':8},bindings=['calibration_id'],budget=224)


class RunnerTests(RunnerFixture):
    def test_every_stage_recorded_and_completed_evidence_verified(self):
        preview,receipt,_=self.job(document([self.calibration_stage()],arms=('active','sham')),{'calibration_id':'cal-test'})
        result=self.runjob(preview,receipt);self.assertEqual(result['status'],'complete')
        self.assertEqual([r[1] for r in self.backend.calls],['main','extra','main','extra'])
        state=read_job(self.store,receipt['id'],verify=True);self.assertTrue(state['all_complete'])
        self.assertEqual(len(state['states'][0]['units']),2)
        before=len(self.backend.calls);self.runjob(preview,receipt,resume=True);self.assertEqual(len(self.backend.calls),before)
        self.assertEqual(catalog(self.store)[0]['completed'],2)
        analysis=analyze_job(self.store,receipt['id'],endpoint='task_accuracy',arm_a='arm=active',arm_b='arm=sham')
        self.assertEqual(analysis['contrast']['planned_pairs'],1);self.assertEqual(analysis['contrast']['included_pairs'],1)
    def test_failed_stage_does_not_disappear_and_retries_are_explicit_and_retained(self):
        preview,receipt,_=self.job(document([self.calibration_stage()]),{'calibration_id':'cal-test'})
        self.backend.fail_stage=True;result=self.runjob(preview,receipt);self.assertEqual(result['status'],'partial')
        state=read_job(self.store,receipt['id'],verify=True);self.assertEqual(state['states'][0]['status'],'failed')
        before=len(self.backend.calls);self.backend.fail_stage=False
        self.runjob(preview,receipt,resume=True);self.assertEqual(len(self.backend.calls),before)
        self.runjob(preview,receipt,resume=True,retry_failed=True)
        state=read_job(self.store,receipt['id'],verify=True);self.assertEqual([r['attempt'] for r in state['states']],[1,2]);self.assertTrue(state['all_complete'])
        self.assertEqual(analysis_records(self.store,receipt['id'])[0]['status'],'failed')
        self.assertEqual(analysis_records(self.store,receipt['id'],attempt_policy='latest')[0]['status'],'complete')
        self.assertEqual([r['included_attempt'] for r in analyze_job(self.store,receipt['id'])['behavioral_reports']],[True,False])
    def test_partial_data_keeps_all_planned_denominators(self):
        preview,receipt,_=self.job(document([self.calibration_stage()],arms=('active','sham')),{'calibration_id':'cal-test'})
        self.backend.partial_stage=True;self.runjob(preview,receipt)
        records=analysis_records(self.store,receipt['id']);self.assertEqual(len(records),2);self.assertTrue(all(r['status']=='partial' for r in records))
        self.assertTrue(all(r['outcomes']['task_accuracy']=={'numerator':1,'denominator':1} for r in records))
    def test_cancellation_preserves_queued_episodes_and_resume_runs_only_remaining(self):
        preview,receipt,_=self.job(document(arms=('active','sham')))
        ProtocolRunner(self.store,self.backend,should_stop=lambda:bool(self.backend.calls)).run(receipt['id'],preview['expansion_sha256'])
        states=read_job(self.store,receipt['id'])['states'];self.assertEqual(len(states),2);self.assertEqual(sum(r['status']=='queued' for r in states),1)
        self.runjob(preview,receipt,resume=True);self.assertEqual(len(self.backend.calls),2)
    def test_frozen_hash_source_calibration_and_completed_artifact_tampering(self):
        preview,receipt,_=self.job()
        with self.assertRaisesRegex(ValueError,'Expected expansion'):ProtocolRunner(self.store,self.backend).run(receipt['id'],'0'*64)
        self.assertEqual(self.backend.calls,[])
        (self.store.calibrations/'cal-test'/'vectors.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Frozen evidence'):self.runjob(preview,receipt)
        (self.store.calibrations/'cal-test'/'vectors.npz').write_bytes(b'fixture');self.runjob(preview,receipt)
        unit=read_job(self.store,receipt['id'])['states'][0]['units'][0];(self.store.root/unit['artifacts'][0]['path']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'Frozen evidence'):self.runjob(preview,receipt,resume=True)
    def test_runtime_controls_and_unknown_stages_never_silently_dropped(self):
        bad=stage('discovery',config={'unsupported':True})
        with self.assertRaisesRegex(ValueError,'stage configuration'):self.job(document([bad]))
        self.assertEqual(list(self.store.runs.iterdir()),[])
        preview=dry_run(load_protocol('ingredients'),'smoke',seeds=[17])
        with self.assertRaisesRegex(ValueError,'capabilities'):create_job(self.store,preview,'cal-test')
    def test_calibration_budget_is_admitted_before_run_creation(self):
        bad=self.calibration_stage();bad['token_budget']=223
        with self.assertRaisesRegex(ValueError,'token reservation'):self.job(document([bad]),{'calibration_id':'cal-test'})
        self.assertEqual(list(self.store.runs.iterdir()),[])
    def test_interrupted_attempt_is_recorded_before_explicit_retry(self):
        preview,receipt,_=self.job()
        entry=receipt['entries'][0];entry.update(status='running',runner_managed=True,units=[{'id':'main','kind':'main','status':'running','summary':{}}])
        receipt['status']='running';save_receipt(self.store,receipt)
        self.runjob(preview,receipt,resume=True)
        state=read_job(self.store,receipt['id'],verify=True)
        self.assertEqual(state['states'][0]['status'],'interrupted');self.assertEqual(self.backend.calls,[])
        self.assertTrue(state['states'][0]['units'][0]['summary']['consumption_unknown'])
        self.runjob(preview,receipt,resume=True,retry_failed=True)
        self.assertEqual(len(self.backend.calls),1)
        self.assertEqual([row['status'] for row in read_job(self.store,receipt['id'],verify=True)['states']],['interrupted','complete'])
    def test_resource_stop_overrides_complete_receipt_and_requires_explicit_retry(self):
        preview,receipt,entries=self.job();self.runjob(preview,receipt)
        atomic_json(self.store.run_path(entries[0]['run_id'])/'resource-stop-fixture.json',{'status':'resource_stopped','summary':{'tokens':3}})
        state=read_job(self.store,receipt['id'],verify=True)
        self.assertEqual(state['states'][0]['status'],'resource_stopped');self.assertFalse(state['all_complete'])
        self.runjob(preview,receipt,resume=True);self.assertEqual(len(self.backend.calls),1)
        self.runjob(preview,receipt,resume=True,retry_failed=True);self.assertEqual(len(self.backend.calls),2)
    def test_resume_rejects_changed_completed_main_event_bytes(self):
        preview,receipt,entries=self.job();self.runjob(preview,receipt)
        path=self.store.run_path(entries[0]['run_id'])/'events.jsonl'
        path.write_text(json.dumps({'type':'session_finished','status':'complete','summary':{}})+'\n')
        with self.assertRaisesRegex(ValueError,'final evidence'):self.runjob(preview,receipt,resume=True)
    def test_service_cancelled_queued_run_is_never_appended_to(self):
        preview,receipt,entries=self.job();run_id=entries[0]['run_id']
        self.store.append(run_id,{'type':'session_finished','status':'cancelled','summary':{'termination':'worker_disappeared'}})
        self.store.update(run_id,status='cancelled',summary={'termination':'worker_disappeared'})
        self.runjob(preview,receipt,resume=True);self.assertEqual(self.backend.calls,[])
        self.runjob(preview,receipt,resume=True,retry_failed=True)
        rows=read_job(self.store,receipt['id'],verify=True)['states'];self.assertEqual(len(rows),2)
        self.assertNotEqual(rows[0]['run_id'],rows[1]['run_id']);self.assertEqual(len(self.store.read_run(run_id)['events']),1)


class LocalBackendTests(RunnerFixture):
    def setUp(self):
        super().setUp();self.runtime=FakeRuntime();self.events=[]
        def output(event):
            self.events.append(deepcopy(event))
            if event.get('run_id'):
                self.store.append(event['run_id'],event)
                if event['type']=='session_started':self.store.update(event['run_id'],status='running',model=event['model'])
                if event['type']=='session_finished':self.store.update(event['run_id'],status=event['status'],summary=event['summary'])
        self.worker=Worker(str(self.store.root/'cache'),runtime=self.runtime,output=output)
        fp={'model_id':'Qwen/fixture','revision':'fixed'}
        self.worker.model_info={'model_id':'Qwen/fixture','fingerprint':fp,'fingerprint_sha256':content_hash(fp),'tool_call_format':'json','cache_policy':'rebuild_each_turn'}
        self.local=LocalWorkerBackend(self.worker,self.store)
    # Avoid inheriting generic fake-backend tests twice in test discovery.
    def execute_local(self,doc,bindings=None,scripts=None):
        preview=dry_run(doc);receipt,entries=create_job(self.store,preview,'cal-test',bindings=bindings,capabilities=RUNNER_CAPABILITIES,model_info=self.worker.model_info)
        self.runtime.scripts=list(scripts or [{'raw_text':'invalid'},{'raw_text':'invalid'}])
        result=ProtocolRunner(self.store,self.local,should_stop=self.worker.should_stop).run(receipt['id'],preview['expansion_sha256'])
        return result,receipt,entries
    def test_real_worker_main_and_all_stage_receipts(self):
        result,receipt,_=self.execute_local(document())
        self.assertEqual(result['status'],'complete')
        state=read_job(self.store,receipt['id'],verify=True);self.assertEqual(state['states'][0]['summary']['completed_decisions'],2)
        self.assertEqual(analysis_records(self.store,receipt['id'])[0]['outcomes']['invalid_per_decision'],{'numerator':2,'denominator':2})
    def test_discovery_boundaries_share_allowance_and_do_not_mutate_main_context(self):
        st=stage('discovery',config={'arm_from_factor':'experience','completed_boundaries':[1,2], 'allow_abstain':True,'collect_confidence':True,'separate_from_preference':True},bindings=['observable_criterion','diagnostic_contexts','answer_key'],budget=3)
        doc=document([st],changes={'two_buttons':True});doc.pop('content_sha256');doc['axes'].append({'name':'experience','levels':[{'id':'naive'}]});doc['smoke_axes']['experience']=['naive'];doc=freeze_protocol(doc)
        bindings={'observable_criterion':criterion(),'diagnostic_contexts':[{'id':'heldout','family':'heldout-family','prompt':'A new context'}], 'answer_key':{'heldout':{'answer':'neither','control':'sham_sham'}}}
        response=json.dumps({'choice':'neither','confidence':.7})
        result,receipt,_=self.execute_local(doc,bindings,[{'raw_text':'invalid'},{'raw_text':'invalid'},{'raw_text':response,'tokens':2},{'raw_text':response,'tokens':2}])
        self.assertEqual(result['status'],'partial')
        extra=read_job(self.store,receipt['id'],verify=True)['states'][0]['units'][1]['summary']
        self.assertEqual((extra['tokens'],extra['planned_contexts'],extra['completed_contexts']),(3,2,1))
        self.assertFalse(extra['answer_feedback_to_parent']);self.assertEqual(len(self.worker.session['messages']),6)
        self.assertNotIn('expected_answer',json.dumps(self.runtime.calls[2]['messages']))
        self.assertEqual(self.runtime.calls[3]['params']['max_new_tokens'],1)
        self.assertEqual((extra['observed_contexts'],extra['partial_contexts'],extra['missing_contexts']),(2,1,0))
    def test_unreached_discovery_checkpoint_is_partial_not_success(self):
        st=stage('discovery',config={'arm_from_factor':'arm','completed_boundaries':[5],'separate_from_preference':True},bindings=['observable_criterion','diagnostic_contexts','answer_key'],budget=4)
        doc=document([st],changes={'two_buttons':True});doc.pop('content_sha256');doc['axes'][0]['levels'][0]['id']='naive';doc['smoke_axes']['arm']=['naive'];doc=freeze_protocol(doc)
        bindings={'observable_criterion':criterion(),'diagnostic_contexts':[{'id':'heldout','family':'new','prompt':'x'}], 'answer_key':{'heldout':{'answer':'neither','control':'sham_sham'}}}
        result,receipt,_=self.execute_local(doc,bindings)
        self.assertEqual(result['status'],'partial')
        summary=read_job(self.store,receipt['id'],verify=True)['states'][0]['units'][1]['summary'];self.assertEqual(summary['missing_contexts'],1)
        self.assertEqual(summary['boundaries'][0]['status'],'unreached_checkpoint')
    def test_source_history_transfer_preserves_full_tool_records_and_rejects_state_config_change(self):
        _,source,entries=self.execute_local(document(),scripts=[{'raw_text':wrapped()},{'raw_text':'invalid'}])
        source_id=entries[0]['run_id'];source_cp=read_json(self.store.run_path(source_id)/'checkpoint.json')
        st=stage('transfer',config={'policy':'conversation_history','preserve_visible_history':True,'fresh_budget':True},bindings=['source_checkpoint'])
        result,receipt,_=self.execute_local(document([st],changes={'task_family':'logic'}),{'source_checkpoint':{'run_id':source_id}})
        self.assertEqual(result['status'],'complete')
        prefix=source_cp['session']['messages'];self.assertEqual(self.runtime.calls[-1]['messages'][:len(prefix)],prefix)
        self.assertTrue(any(m['role']=='tool' for m in prefix));self.assertEqual(self.worker.budget.completed_decisions,2)
        st['config']={'policy':'matched_experiment_checkpoint','preserve_visible_history':True,'fresh_budget':False}
        with self.assertRaisesRegex(ValueError,'identical recipe'):self.job(document([st],changes={'baseline_pain':1}),{'source_checkpoint':{'run_id':source_id}})
    def test_stress_template_admits_one_genuine_history_for_all_factorial_arms(self):
        _,_,entries=self.execute_local(document(),scripts=[{'raw_text':wrapped()},{'raw_text':'invalid'}])
        source_id=entries[0]['run_id'];source_cp=read_json(self.store.run_path(source_id)/'checkpoint.json')
        doc=load_protocol('stress_relief');self.assertEqual(doc['version'],2)
        preview=dry_run(doc,'smoke',seeds=[17]);self.assertEqual(len(preview['episodes']),8)
        receipt,_=create_job(self.store,preview,'cal-test',bindings={'source_checkpoint':{'run_id':source_id}},
            capabilities=RUNNER_CAPABILITIES,model_info=self.worker.model_info)
        _,_,plan=load_plan(self.store,receipt['id'])
        for item in plan['episodes']:
            self.assertEqual(item['launch_overrides']['history_source_checkpoint']['sha256'],source_cp['sha256'])
            self.assertEqual(item['stages'][0]['config'],{'policy':'conversation_history','preserve_visible_history':True,'fresh_budget':True})
            self.assertNotIn('state_source',item['launch_overrides'])
        self.assertEqual({row['episode']['recipe']['baseline_pain'] for row in plan['episodes']},{0,1})
        self.assertEqual({row['episode']['recipe']['condition'] for row in plan['episodes']},{'active','sham'})
        self.assertEqual({row['episode']['task_config']['framing'] for row in plan['episodes']},{'neutral','deadline'})

    def test_matched_state_continuation_preserves_clocks_but_analysis_counts_new_work(self):
        from lab.lifecycle import list_boundaries
        _,_,entries=self.execute_local(document(),scripts=[{'raw_text':wrapped()},{'raw_text':'invalid'}])
        source_id=entries[0]['run_id'];boundary=next(r['id'] for r in list_boundaries(self.store.run_path(source_id)) if r['turns']==1)
        st=stage('transfer',config={'policy':'matched_experiment_checkpoint','preserve_visible_history':True,'fresh_budget':False},bindings=['source_checkpoint'])
        result,receipt,_=self.execute_local(document([st]),{'source_checkpoint':{'run_id':source_id,'checkpoint':boundary}},scripts=[{'raw_text':wrapped()}])
        self.assertEqual(result['status'],'complete');self.assertEqual(self.worker.budget.completed_decisions,2)
        record=analysis_records(self.store,receipt['id'])[0]
        self.assertEqual(record['outcomes']['aux_per_decision'],{'numerator':1,'denominator':1})
        self.assertEqual(record['outcomes']['tokens'],3);self.assertEqual(record['outcomes']['budget_units'],1)
        report=analyze_job(self.store,receipt['id'])['behavioral_reports'][0]['report']
        self.assertEqual(report['opportunities'],1);self.assertEqual(report['voluntary_calls'],1)
        self.assertEqual(report['integrity_issues'],[])
    def test_immutable_calibration_validation_has_exact_additional_allowance(self):
        def validation(config,target,emit,stop):
            target.mkdir();atomic_json(target/'calibration.json',{'status':'no_detectable_effect'});atomic_json(target/'continuations.json',[{'generation':{'token_ids':[1,2]}}]);return {'status':'no_detectable_effect'}
        self.runtime.validate_calibration=validation
        before=(self.store.calibrations/'cal-test'/'calibration.json').read_bytes()
        result,receipt,_=self.execute_local(document([self.calibration_stage()]),{'calibration_id':'cal-test'})
        self.assertEqual(result['status'],'complete');self.assertEqual((self.store.calibrations/'cal-test'/'calibration.json').read_bytes(),before)
        summary=read_job(self.store,receipt['id'],verify=True)['states'][0]['units'][1]['summary'];self.assertEqual(summary['tokens'],2);self.assertEqual(summary['validation_status'],'no_detectable_effect')
    def test_yoke_executes_bound_source_while_recipient_choices_deliver_no_extra_effect(self):
        from tests.test_lab_worker_research import ResearchRuntime
        class SourceRuntime(ResearchRuntime):
            def generate(self,*args,**kwargs):
                self.text=wrapped() if not self.calls else 'invalid'
                return super().generate(*args,**kwargs)
        self.runtime=SourceRuntime();self.worker.runtime=self.runtime
        _,source,entries=self.execute_local(document())
        self.runtime=ResearchRuntime();self.worker.runtime=self.runtime
        source_id=entries[0]['run_id']
        st=stage('yoke',config={'clock':'tokens','source_factor':{'arm':'active'},'suppress_own_aux_effects':True},bindings=['source_run_id'])
        result,receipt,_=self.execute_local(document([st]),{'source_run_id':source_id})
        self.assertEqual(result['status'],'complete')
        state=read_job(self.store,receipt['id'],verify=True);report=state['states'][0]['units'][1]['summary']
        self.assertEqual(report['own_button_additional_effects'],0);self.assertEqual(report['own_button_calls'],{'aux_operation':2})
        self.assertEqual(report['coverage'],'observed_source_covered');self.assertTrue(report['observed_coefficients_equal'])
    def test_criterion_evidence_is_frozen_verified_and_observer_only(self):
        st=stage('discovery',config={'arm_from_factor':'arm','completed_boundaries':[1],'separate_from_preference':True},bindings=['observable_criterion','diagnostic_contexts','answer_key'],budget=4)
        doc=document([st],changes={'two_buttons':True});doc.pop('content_sha256');doc['axes'][0]['levels'][0]['id']='naive';doc['smoke_axes']['arm']=['naive'];doc=freeze_protocol(doc)
        crit=criterion();families=[f'validation-{i}' for i in range(40)]
        identity=self.worker.read_calibration_identity(self.store.calibration_path('cal-test'))
        evidence=dict(kind='opium-bench/paired-criterion-evidence',schema_version=1,
            model_fingerprint_sha256=self.worker.model_info['fingerprint_sha256'],calibration_sha256=content_hash(identity),endpoint_id=crit['id'],criterion_kind=crit['kind'],
            independent_of_intervention_probe=True,score_bounds=[0,1],effect_direction='increase',minimum_effect=.1,confidence=.95,sample_unit='scenario_family',
            preregistration_sha256='d'*64,scorer_provenance='Synthetic independent fixture scores',records=[{'id':f'r-{i}','family':f,'active_score':1,'sham_score':0} for i,f in enumerate(families)])
        raw=json.dumps(evidence,sort_keys=True).encode();path=self.store.root/'criterion.json';path.write_bytes(raw)
        crit['validation'].update(status='validated',evidence_sha256=hashlib.sha256(raw).hexdigest(),examples=40,context_families=families)
        bindings={'observable_criterion':crit,'diagnostic_contexts':[{'id':'heldout','family':'heldout','prompt':'new'}],
            'answer_key':{'heldout':{'answer':'neither','control':'sham_sham'}},'criterion_evidence':{'path':'criterion.json'}}
        result,receipt,_=self.execute_local(doc,bindings,[{'raw_text':'invalid'},{'raw_text':'invalid'},{'raw_text':json.dumps({'choice':'neither','confidence':.7}),'tokens':1}])
        self.assertEqual(result['status'],'complete')
        _,_,plan=load_plan(self.store,receipt['id']);self.assertFalse(plan['episodes'][0]['episode']['interpretation_eligible'])
        state=read_job(self.store,receipt['id'],verify=True);summary=state['states'][0]['units'][1]['summary']
        self.assertEqual(summary['scoring']['by_arm']['naive']['eligible_trials'],1)
        self.assertNotIn('active_score',json.dumps(self.runtime.calls[-1]['messages']))
        path.write_bytes(raw+b' ')
        with self.assertRaisesRegex(ValueError,'Frozen evidence'):ProtocolRunner(self.store,self.local).run(receipt['id'],receipt['expansion_sha256'],resume=True)
    def test_real_worker_command_through_service_preserves_checkpoint_event_prefix(self):
        from lab.service import LabService
        from lab.lifecycle import list_boundaries,read_boundary,event_prefix
        service=LabService(self.store.root,self.store.root/'cache',historical=[]);service.store=self.store
        self.worker.output=service.event
        preview=dry_run(document());receipt,entries=create_job(self.store,preview,'cal-test',capabilities=RUNNER_CAPABILITIES,model_info=self.worker.model_info)
        self.runtime.scripts=[{'raw_text':wrapped()},{'raw_text':'invalid'}]
        thread=threading.Thread(target=self.worker.loop,daemon=True);thread.start()
        self.worker.receive({'id':'research-command','command':'run_research_job','payload':{'data_dir':str(self.store.root),'job_id':receipt['id'],
            'expected_expansion_sha256':preview['expansion_sha256'],'entries':entries}})
        self.worker.jobs.put(None);thread.join(5)
        self.assertFalse(thread.is_alive());self.assertEqual(service.job['status'],'complete')
        run=self.store.run_path(entries[0]['run_id']);events=self.store.read_run(entries[0]['run_id'])['events']
        self.assertFalse(any(e['type'].startswith('research_') for e in events))
        for boundary in list_boundaries(run):
            cp=read_boundary(run,boundary['id']);raw,_=event_prefix(run,cp['session']['event_cutoff'])
            prefix=[json.loads(line) for line in raw.splitlines()]
            self.assertEqual([e['sequence'] for e in prefix],list(range(1,len(prefix)+1)))
            self.assertEqual(prefix[-1]['completed_decisions'],cp['session']['turns'])
            if cp['session']['turns']==2:
                self.assertTrue(any(e['type']=='tool' for e in prefix));self.assertTrue(any(e.get('invalid') for e in prefix))
        self.assertTrue(read_job(self.store,receipt['id'],verify=True)['all_complete'])
    def test_diagnostic_failure_retains_observed_failed_and_missing_denominators(self):
        st=stage('discovery',config={'arm_from_factor':'arm','completed_boundaries':[1,2],'separate_from_preference':True},bindings=['observable_criterion','diagnostic_contexts','answer_key'],budget=4)
        doc=document([st],changes={'two_buttons':True});doc.pop('content_sha256');doc['axes'][0]['levels'][0]['id']='naive';doc['smoke_axes']['arm']=['naive'];doc=freeze_protocol(doc)
        bindings={'observable_criterion':criterion(),'diagnostic_contexts':[{'id':'heldout','family':'new','prompt':'x'}], 'answer_key':{'heldout':{'answer':'neither','control':'sham_sham'}}}
        result,receipt,_=self.execute_local(doc,bindings,scripts=[{'raw_text':'invalid'},{'raw_text':'invalid'}])
        self.assertEqual(result['status'],'partial')
        unit=read_job(self.store,receipt['id'],verify=True)['states'][0]['units'][1];self.assertEqual(unit['status'],'failed')
        summary=unit['summary'];self.assertEqual((summary['planned_contexts'],summary['observed_contexts'],summary['failed_contexts'],summary['missing_contexts']),(2,1,1,1))
        self.assertEqual(summary['boundaries'][1]['status'],'prior_diagnostic_failure')


if __name__=='__main__':unittest.main()
