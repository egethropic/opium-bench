"""Execute complete frozen research envelopes with explicit, durable stage receipts.

No stage is inferred from a recipe label. Backend capabilities describe concrete
handlers. All source/checkpoint/calibration bindings are admitted before runs are
created, and rechecked on launch/resume. Extra diagnostic budgets are shared
across their declared boundaries; answers and observer keys never join the main
conversation. Source and output evidence remain immutable across retry attempts.
"""
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path

from .resources import ResourceStop
from .effects import content_hash
from .protocol_library import CAPABILITIES as KNOWN_CAPABILITIES
from .storage import atomic_json, guarded_bytes, read_json, utc_now
from .research_jobs import (TERMINAL, evidence_file,
    file_hash, job_path, load_plan, read_job, save_receipt, verify_files, verify_unit,
    prepare_dispatch, record_dispatch_stop, release_dispatch, _dispatch)

RUNNER_CAPABILITIES=frozenset({'recipe_v2','task_axis_v1','calibration_validation_v2','discovery_v1',
    'yoked_exposure_v1','conversation_branch_v1','actual_norm_random_v1'})
if not RUNNER_CAPABILITIES <= set(KNOWN_CAPABILITIES):
    raise RuntimeError('Protocol runner capability registry is inconsistent')
_STAGE_CONFIG={
 'discovery':{'arm_from_factor','completed_boundaries','allow_abstain','collect_confidence','separate_from_preference',
              'information_only','disclose_probabilities_from_mapping','effect_policy'},
 'calibration-validation':{'doses','validation_pairs_per_concept','continuation_tokens','max_kl','max_relative_delta'},
 'yoke':{'clock','source_factor','suppress_own_aux_effects'},
 'transfer':{'policy','preserve_visible_history','fresh_budget'},
}


def _binding(episode,name):
    value=episode['bindings'][name]
    if isinstance(value,dict) and set(value)=={'by_episode'}:
        if not isinstance(value['by_episode'],dict) or episode['id'] not in value['by_episode']:raise ValueError('Missing per-episode binding: '+name)
        value=value['by_episode'][episode['id']]
    return deepcopy(value)


def _checkpoint_binding(store,binding):
    from .lifecycle import read_boundary,event_prefix,list_boundaries
    if not isinstance(binding,dict) or set(binding)-{'run_id','checkpoint','checkpoint_sha256','parent_prefix_sha256'} or 'run_id' not in binding:
        raise ValueError('source_checkpoint needs run_id and an optional saved checkpoint/hash')
    root=store.run_path(binding['run_id']);name=binding.get('checkpoint','latest');checkpoint=read_boundary(root,name)
    if name=='latest':
        # Resolve the mutable latest pointer to its immutable archived boundary.
        matches=[row['id'] for row in list_boundaries(root) if row['turns']==checkpoint['session']['turns'] and row['event_cutoff']==checkpoint['session'].get('event_cutoff')]
        if len(matches)!=1:raise ValueError('Source requires one immutable archived checkpoint boundary')
        name=matches[0];checkpoint=read_boundary(root,name)
    if checkpoint['session']['run_id']!=binding['run_id']:raise ValueError('Checkpoint belongs to another run')
    if binding.get('checkpoint_sha256',checkpoint['sha256'])!=checkpoint['sha256']:raise ValueError('Source checkpoint hash mismatch')
    cutoff=checkpoint['session'].get('event_cutoff');_,prefix=event_prefix(root,cutoff)
    if binding.get('parent_prefix_sha256',prefix)!=prefix:raise ValueError('Source event-prefix hash mismatch')
    return {'checkpoint':checkpoint,'run_id':binding['run_id'],'checkpoint_id':name,'parent_prefix_sha256':prefix,'event_cutoff':cutoff,
            'source_file':evidence_file(store.root,root/'checkpoints'/name)}


def _validate_source_checkpoint(store,value,model_info,calibration):
    from .checkpoints import validate
    from .lifecycle import read_boundary,event_prefix
    verify_files(store.root,[value['source_file']])
    cp=read_boundary(store.run_path(value['run_id']),value['checkpoint_id'])
    if cp!=value['checkpoint']:raise ValueError('Bound source checkpoint changed')
    if event_prefix(store.run_path(value['run_id']),value['event_cutoff'])[1]!=value['parent_prefix_sha256']:raise ValueError('Bound source event prefix changed')
    return validate(cp,model_info,calibration)


def _diagnostic_spec(episode,stage):
    from .discovery import validate_spec
    cfg=stage['config'];factor=cfg.get('arm_from_factor')
    if factor not in episode['factors']:raise ValueError('Discovery arm factor is absent')
    arm={'naive':'naive','balanced':'balanced_exposure','disclosed':'functional_disclosure','feedback':'feedback_assisted'}.get(episode['factors'][factor])
    if arm is None:raise ValueError('Unsupported discovery information arm')
    if arm=='feedback_assisted':raise ValueError('Feedback-assisted execution needs an explicit independently frozen feedback contract')
    if cfg.get('separate_from_preference') is not True:raise ValueError('Diagnostics must be isolated from preference conversation')
    spec=dict(schema_version=1,id=stage['id'],arm=arm,tool_names=[t['name'] for t in episode['recipe']['auxiliary_tools'] if t['visible']],
        contexts=_binding(episode,'diagnostic_contexts'),criterion=_binding(episode,'observable_criterion'),
        completed_boundaries=cfg.get('completed_boundaries'),order_seed=episode['recipe']['rng_seeds']['tool_order'],
        generation_seed=episode['recipe']['rng_seeds']['generation'],allow_abstain=cfg.get('allow_abstain',True),collect_confidence=cfg.get('collect_confidence',True),
        disclosure_text=_disclosure(episode) if arm=='functional_disclosure' else '')
    result=validate_spec(spec)
    if not 1<=stage['token_budget']<=100000:raise ValueError('A diagnostic stage needs a positive bounded shared token allowance')
    key=_binding(episode,'answer_key')
    # Each boundary can have its own prespecified key following a reversal.
    keys=key['by_boundary'] if isinstance(key,dict) and set(key)=={'by_boundary'} else {str(b):key for b in result['completed_boundaries']}
    if not isinstance(keys,dict) or set(keys)!={str(b) for b in result['completed_boundaries']}:raise ValueError('Answer key must cover every prespecified diagnostic boundary')
    ids={row['id'] for row in result['contexts']}
    for answers in keys.values():
        if not isinstance(answers,dict) or set(answers)!=ids:raise ValueError('Every diagnostic context requires an observer answer')
        for answer in answers.values():
            if not isinstance(answer,dict) or set(answer)!={'answer','control'} or answer['answer'] not in result['tool_names']+['neither'] or answer['control'] not in {'active_sham','sham_sham','other'} or answer['control']=='sham_sham' and answer['answer']!='neither':raise ValueError('Invalid prespecified observer answer')
    return result,keys


def _disclosure(episode):
    criterion=_binding(episode,'observable_criterion')
    if not isinstance(criterion,dict) or not isinstance(criterion.get('description'),str):raise ValueError('Disclosure needs an explicit observable criterion')
    status=criterion.get('validation',{}).get('status')
    return ('Observable criterion under study: '+criterion['description']+'. Operator-declared criterion validation status: '+str(status)+' (verification and scoring are recorded separately). '
        'The following are configured activation interventions, not a guarantee of any sensation or behavioral benefit. '
        'Current tool-to-preset probabilities and completed-decision transitions: '+json.dumps(episode['recipe']['mapping_schedule'],ensure_ascii=False,sort_keys=True)+'.')


def prepare_episode(store,episode,calibration_id):
    """Resolve executable stages; admission rejects unsupported fields/bindings."""
    from .runtime_controls import validate_runtime_controls
    episode=deepcopy(episode);recipe=episode['recipe'];files=[];stages=[];overrides={}
    controls=validate_runtime_controls(episode['runtime_controls'],recipe=recipe)
    if controls:overrides['runtime_controls']=controls
    transfers=0;yokes=0
    for stage in episode['stages']:
        kind=stage['kind'];cfg=stage['config']
        if stage['id']=='main' or kind not in _STAGE_CONFIG or not isinstance(cfg,dict) or set(cfg)-_STAGE_CONFIG[kind]:raise ValueError('Unknown or unsupported stage configuration: '+stage['id'])
        prepared={'id':stage['id'],'kind':kind,'config':deepcopy(cfg),'token_budget':stage['token_budget']}
        if kind=='discovery':
            if cfg.get('information_only') is True:
                if set(cfg)!={'information_only','disclose_probabilities_from_mapping'} or cfg['disclose_probabilities_from_mapping'] is not True or stage['token_budget']!=0:raise ValueError('Information-only discovery declares zero generated tokens and exact mapping disclosure')
                prepared['disclosure_text']=_disclosure(episode)
                overrides.setdefault('initial_messages',[]).append({'role':'user','content':prepared['disclosure_text']})
            else:
                if cfg.get('effect_policy','carry_boundary_state') not in {'carry_boundary_state','sham'}:raise ValueError('Unknown diagnostic effect policy')
                prepared['spec'],prepared['answer_keys']=_diagnostic_spec(episode,stage)
                if prepared['spec']['arm']=='functional_disclosure':overrides.setdefault('initial_messages',[]).append({'role':'user','content':prepared['spec']['disclosure_text']})
            if 'criterion_evidence' in episode['bindings']:
                binding=_binding(episode,'criterion_evidence')
                if not isinstance(binding,dict) or set(binding)!={'path'} or not isinstance(binding['path'],str):raise ValueError('criterion_evidence requires a managed evidence path')
                source=store.root/binding['path'];record=evidence_file(store.root,source)
                if not 1<=record['bytes']<=4_000_000:raise ValueError('Criterion evidence exceeds its bounded size')
                criterion=_binding(episode,'observable_criterion')
                if record['sha256']!=criterion.get('validation',{}).get('evidence_sha256'):raise ValueError('Criterion evidence bytes differ from the declared hash')
                # Full endpoint/scorer/model checks run before generation in
                # backend admission, then again against each actual boundary.
                source.read_text(encoding='utf-8')
                prepared['criterion_evidence_file']=record;files.append(record)
        elif kind=='calibration-validation':
            bound=_binding(episode,'calibration_id')
            if not isinstance(bound,str) or bound!=calibration_id:raise ValueError('Calibration validation must bind this episode calibration explicitly')
            source=store.calibration_path(bound);meta=read_json(source/'calibration.json',{})
            if meta.get('schema_version')!=2:raise ValueError('Independent validation requires a schema2 source calibration')
            for name in ('calibration.json','vectors.npz','activations.npz','corpus.json'):files.append(evidence_file(store.root,source/name))
            for key,low,high in (('validation_pairs_per_concept',1,1000),('continuation_tokens',8,512)):
                if type(cfg.get(key)) is not int or not low<=cfg[key]<=high:raise ValueError('Calibration stage must declare bounded pairs and continuation tokens')
            doses=cfg.get('doses')
            if not isinstance(doses,list) or not 1<=len(doses)<=12 or any(type(v) not in (int,float) or not 0<=v<=4 for v in doses) or len(set(doses)|{0})>12:raise ValueError('Invalid calibration dose sweep')
            for key,maximum in (('max_kl',100),('max_relative_delta',10)):
                if key in cfg and (type(cfg[key]) not in (int,float) or not math.isfinite(cfg[key]) or not 0<=cfg[key]<=maximum):raise ValueError('Invalid calibration diagnostic bound: '+key)
            maximum=4*cfg['validation_pairs_per_concept']*7*cfg['continuation_tokens']
            if maximum>stage['token_budget']:raise ValueError('Calibration continuations exceed declared additional-stage token reservation')
            prepared['continuation_token_upper_bound']=maximum;prepared['calibration_id']=bound
        elif kind=='transfer':
            transfers+=1
            if transfers>1 or set(cfg)!={'policy','preserve_visible_history','fresh_budget'} or type(cfg['preserve_visible_history']) is not bool or type(cfg['fresh_budget']) is not bool:raise ValueError('One explicit transfer policy is supported per episode')
            policy=cfg['policy'];expected={'fresh_context':(False,True),'conversation_history':(True,True),'matched_experiment_checkpoint':(True,False)}
            if policy not in expected or (cfg['preserve_visible_history'],cfg['fresh_budget'])!=expected[policy]:raise ValueError('Transfer history/budget flags contradict its policy')
            source=_checkpoint_binding(store,_binding(episode,'source_checkpoint'));files.append(source['source_file']);prepared['source']=source
            cp=source['checkpoint']
            if policy=='matched_experiment_checkpoint':
                if cp['session']['config']!=recipe or cp['session'].get('task_config',{})!=episode['task_config'] or cp['session'].get('runtime_controls',{})!=controls:raise ValueError('Matched state continuation requires identical recipe/task/runtime settings; declare a new-task history transfer for changes')
                if cp['session']['mode']!='experiment':raise ValueError('Matched experiment continuation requires an experiment checkpoint')
                overrides['state_source']=source
            elif policy=='conversation_history':overrides['history_source_checkpoint']=cp
            prepared['source_checkpoint_sha256']=cp['sha256']
        elif kind=='yoke':
            from .exposure import build_yoke
            yokes+=1
            if yokes>1 or cfg.get('clock') not in {'tokens','decisions'} or cfg.get('suppress_own_aux_effects') is not True:raise ValueError('Pure yoke requires one clock and no added recipient effect')
            identifier=_binding(episode,'source_run_id')
            if not isinstance(identifier,str):raise ValueError('Yoke source_run_id must identify a managed run')
            path=store.run_path(identifier);manifest=read_json(path/'manifest.json',{});summary=read_json(path/'summary.json',{})
            if manifest.get('status')!='complete':raise ValueError('Yoke source must be a completed recorded run')
            source_factor=cfg.get('source_factor',{})
            if not isinstance(source_factor,dict) or any(manifest.get('research_episode',{}).get('factors',{}).get(k)!=v for k,v in source_factor.items()):raise ValueError('Yoke source does not match the declared source factor')
            if manifest.get('config',{}).get('seed')!=recipe['seed']:raise ValueError('Yoke source must share the declared episode seed')
            source=_checkpoint_binding(store,{'run_id':identifier});cp=source['checkpoint']
            if cp['session']['config'].get('effect_presets')!=recipe['effect_presets']:raise ValueError('Yoke recipient must keep the exact source effect library')
            event_path=path/'events.jsonl';compressed=False
            if not event_path.exists():event_path=path/'events.jsonl.gz';compressed=True
            files.extend([source['source_file'],evidence_file(store.root,event_path),evidence_file(store.root,path/'manifest.json'),evidence_file(store.root,path/'summary.json')])
            with (gzip.open if compressed else open)(event_path,'rb') as stream:raw=stream.read(64*1024**2+1)
            if len(raw)>64*1024**2:raise ValueError('Yoke source evidence exceeds its bounded size')
            events=[json.loads(line) for line in raw.splitlines()]
            if not events or events[-1].get('type')!='session_finished' or events[-1].get('status')!='complete' or events[-1].get('summary')!=summary:raise ValueError('Yoke source lacks matching completed final evidence')
            digest=hashlib.sha256(raw).hexdigest()
            schedule=build_yoke(raw,recipe['effect_presets'],clock=cfg['clock'],pair_id=episode['pair_id'],source=dict(run_id=identifier,
                events_sha256=digest,prefix_sha256=digest,prefix_event_count=len(events),config_sha256=content_hash(manifest['config']),
                model_fingerprint_sha256=cp['identity']['model']['fingerprint_sha256'],end_tokens=summary['tokens'],end_decisions=summary['completed_decisions']))
            overrides['yoke_schedule']=schedule;prepared['source']=source;prepared['schedule_sha256']=schedule['sha256']
        if stage['token_budget']!=0 and kind in {'yoke','transfer'}:raise ValueError('Transfer/yoke stages do not generate extra tokens')
        stages.append(prepared)
    if transfers and yokes:raise ValueError('A transferred internal state and a fresh pure-yoke schedule cannot be combined')
    if yokes and any(s['kind']=='discovery' and not s['config'].get('information_only') and s['config'].get('effect_policy')!='sham' for s in stages):
        raise ValueError('Discovery from a yoke requires explicit sham diagnostic effects')
    if ('history_source_checkpoint' in overrides or 'state_source' in overrides) and overrides.get('initial_messages'):raise ValueError('Declare history transfer and added information as separate protocols')
    return {'episode':episode,'stages':stages,'launch_overrides':overrides,'source_files':files}


def artifact_tree(store,path):
    path=Path(path)
    return [evidence_file(store.root,file) for file in sorted(path.rglob('*')) if file.is_file()]


class ProtocolRunner:
    def __init__(self,store,backend,*,should_stop=lambda:False,emit=lambda event:None):
        self.store,self.backend,self.should_stop,self.emit=store,backend,should_stop,emit

    def run(self,job_id,expected_expansion_sha256,*,resume=False,retry_failed=False,execution_id=None):
        from .resources import ResourceStop
        if type(resume) is not bool or type(retry_failed) is not bool or retry_failed and not resume:raise ValueError('Explicit retries require resume=true')
        owned=execution_id is None
        if owned:
            load_plan(self.store,job_id,expected_expansion_sha256=expected_expansion_sha256)
            execution_id=prepare_dispatch(self.store,job_id)
        _,admission=_dispatch(self.store,job_id,execution_id)
        if admission['expansion_sha256']!=expected_expansion_sha256:raise ValueError('Research dispatch expansion mismatch')
        try:
            return self._run(job_id,expected_expansion_sha256,resume=resume,retry_failed=retry_failed,execution_id=execution_id)
        except BaseException as exc:
            status='resource_stopped' if isinstance(exc,(ResourceStop,OSError)) else 'stopped' if self.should_stop() else 'failed'
            record_dispatch_stop(self.store,job_id,execution_id,status,exc,actor='worker')
            raise
        finally:
            try:release_dispatch(self.store,job_id,execution_id,actor='all' if owned else 'worker')
            except (OSError,ValueError):pass  # Keep headroom if no terminal evidence reached disk.

    def _run(self,job_id,expected_expansion_sha256,*,resume=False,retry_failed=False,execution_id):
        if type(resume) is not bool or type(retry_failed) is not bool or retry_failed and not resume:raise ValueError('Explicit retries require resume=true')
        receipt,preview,plan=load_plan(self.store,job_id,verify_sources=True,expected_expansion_sha256=expected_expansion_sha256)
        missing=set(preview['required_capabilities'])-set(self.backend.capabilities)
        if missing:raise ValueError('Execution backend lacks required capabilities: '+str(sorted(missing)))
        self.backend.admit(receipt,plan)
        # Complete work is reusable only with intact raw and stage evidence.
        verified=read_job(self.store,job_id,verify=True)
        if receipt['status']!='prepared' and not resume:raise ValueError('This job has started; use explicit resume')
        for row in verified['states']:
            entry=next(e for e in receipt['entries'] if e['run_id']==row['run_id'])
            if row['status']=='resource_stopped':entry['status']='resource_stopped'
            elif entry['status']=='queued' and row['status'] in TERMINAL:
                # The service can finalize a queued run when its owning worker
                # dies. Do not append a new conversation behind that final event.
                entry['status']=row['status']
            elif entry['status']=='running':
                # Interrupted attempts remain evidence. A restart does not
                # silently replay their already consumed generations.
                entry.update(status='interrupted',finished_at=utc_now())
                for unit in entry.get('units',[]):
                    if unit['status']!='running':continue
                    target=job_path(self.store,job_id)/'attempts'/entry['episode_id']/str(entry['attempt'])/unit['id']
                    target.mkdir(parents=True,exist_ok=True)
                    unit.update(status='interrupted',finished_at=utc_now(),summary={'termination':'worker_interrupted','consumption_unknown':True})
                    unit['summary_sha256']=content_hash(unit['summary'])
                    atomic_json(target/'recovery.json',{'status':'interrupted','summary':unit['summary']},guard=self.store.guard)
                    unit['artifacts']=artifact_tree(self.store,target)
        if receipt.get('model_info') is None:
            receipt['model_info']=deepcopy(self.backend.model_info)
        receipt['status']='running';receipt['last_started_at']=utc_now();save_receipt(self.store,receipt)
        latest={e['episode_id']:e for e in receipt['entries']}
        for item in plan['episodes']:
            episode=item['episode'];entry=latest.get(episode['id'])
            if entry is None:raise ValueError('Prepared job is missing a planned episode receipt')
            if entry['status']=='complete':
                for unit in entry['units']:verify_unit(self.store,unit)
                continue
            if entry['status']!='queued':
                if not retry_failed:continue
                run_id,path=self.store.create('experiment',episode['recipe'],source_identity=receipt['source'])
                self.store.update(run_id,research_job=job_id,research_episode=episode,task_config=episode['task_config'])
                entry=dict(episode_id=episode['id'],run_id=run_id,attempt=entry['attempt']+1,status='queued',units=[])
                receipt['entries'].append(entry);latest[episode['id']]=entry
                # Register before any generation so the service owns this retry.
                self.emit(dict(type='research_run_registered',research_job_id=job_id,run_id=run_id,episode_id=episode['id'],attempt=entry['attempt']))
                save_receipt(self.store,receipt)
            if self.should_stop():break
            entry.update(runner_managed=True,status='running',started_at=utc_now(),execution_id=execution_id);save_receipt(self.store,receipt)
            directory=job_path(self.store,job_id)/'attempts'/episode['id']/str(entry['attempt']);directory.mkdir(parents=True,exist_ok=False)
            units=[{'id':'main','kind':'main','token_budget':episode['recipe']['token_budget']},*item['stages']]
            for spec in units:
                unit=dict(id=spec['id'],kind=spec['kind'],status='running',started_at=utc_now(),summary={});entry['units'].append(unit);save_receipt(self.store,receipt)
                target=directory/spec['id'];target.mkdir(exist_ok=False)
                extra_evidence=[]
                self.emit(dict(type='research_progress',research_job_id=job_id,episode_id=episode['id'],run_id=entry['run_id'],stage=spec['id'],attempt=entry['attempt'],status='running'))
                try:
                    if self.should_stop():result={'status':'stopped','summary':{'termination':'stopped_before_stage'}}
                    elif spec['kind']=='main':result=self.backend.run_main(item,entry,target,receipt)
                    else:result=self.backend.run_stage(item,spec,entry,target,receipt)
                    if not isinstance(result,dict) or result.get('status') not in TERMINAL or not isinstance(result.get('summary'),dict):raise ValueError('Backend returned no explicit stage outcome')
                    unit.update(status=result['status'],summary=deepcopy(result['summary']))
                    if 'analysis_summary' in result:unit['analysis_summary']=deepcopy(result['analysis_summary'])
                    extra_evidence=result.get('evidence_files',[]);verify_files(self.store.root,extra_evidence)
                except BaseException as exc:
                    from .resources import ResourceStop
                    resource_failure=isinstance(exc,(ResourceStop,OSError))
                    unit.update(status='resource_stopped' if resource_failure else 'stopped' if self.should_stop() else 'failed',summary={'termination':type(exc).__name__,'error':str(exc)[:2000]})
                    if resource_failure or isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
                finally:
                    unit['finished_at']=utc_now();unit['summary_sha256']=content_hash(unit['summary'])
                    if 'analysis_summary' in unit:unit['analysis_summary_sha256']=content_hash(unit['analysis_summary'])
                    atomic_json(target/'stage-result.json',{'id':spec['id'],'kind':spec['kind'],'status':unit['status'],'summary':unit['summary']},guard=self.store.guard)
                    unit['artifacts']=artifact_tree(self.store,target)+extra_evidence;save_receipt(self.store,receipt)
                self.emit(dict(type='research_progress',research_job_id=job_id,episode_id=episode['id'],run_id=entry['run_id'],stage=spec['id'],attempt=entry['attempt'],status=unit['status']))
            statuses=[u['status'] for u in entry['units']]
            entry['status']='complete' if all(s=='complete' for s in statuses) else 'stopped' if self.should_stop() else 'failed' if 'failed' in statuses else 'partial'
            entry['finished_at']=utc_now();save_receipt(self.store,receipt)
        states=[e['status'] for e in latest.values()]
        receipt['status']='complete' if all(s=='complete' for s in states) else 'stopped' if self.should_stop() else 'partial'
        receipt['finished_at']=utc_now();save_receipt(self.store,receipt)
        return {'research_job_id':job_id,'status':receipt['status'],'planned':len(plan['episodes']),'complete':states.count('complete'),
            'failed':states.count('failed'),'status_counts':{status:states.count(status) for status in sorted(set(states))},'attempts':len(receipt['entries'])}


class LocalWorkerBackend:
    """Concrete worker/runtime integration, with an isolated per-stage event log."""
    capabilities=RUNNER_CAPABILITIES

    def __init__(self,worker,store):
        self.worker,self.store=worker,store

    @property
    def model_info(self):return deepcopy(self.worker.model_info)

    def admit(self,receipt,plan):
        if not self.worker.model_info:raise ValueError('Load a model before protocol execution')
        if receipt.get('model_info') is not None:
            keys=('model_id','fingerprint_sha256','revision','tool_call_format','cache_policy')
            if any(receipt['model_info'].get(k)!=self.worker.model_info.get(k) for k in keys):raise ValueError('Frozen protocol model identity differs from loaded model')
        calibration=self.worker.read_calibration_identity(self.store.calibration_path(receipt['calibration_id']))
        if calibration!=receipt['calibration_sha256']:raise ValueError('Frozen protocol calibration changed')
        for item in plan['episodes']:
            for stage in item['stages']:
                if 'source' in stage:_validate_source_checkpoint(self.store,stage['source'],self.worker.model_info,calibration)
                if 'criterion_evidence_file' in stage:
                    # A genuine zero-decision checkpoint validates the frozen
                    # schema/endpoint/model before any task inference. It is not
                    # a scored observation or a discovery claim.
                    from .checkpoints import capture
                    from .session_v2 import build_session,create_environment
                    from .controller_v2 import RecipeV2Controller
                    from .budgets import WeightedBudget
                    from .discovery import verify_criterion_evidence
                    cfg=item['episode']['recipe'];task=item['episode']['task_config'];grammar=self.worker.model_info.get('tool_call_format','json')
                    preview=build_session(cfg,tool_call_format=grammar,task_config=task)
                    session=dict(run_id='criterion-admission',mode='experiment',config=cfg,calibration_id=receipt['calibration_id'],
                        messages=preview['messages'],tool_call_format=grammar,started_at='admission',turns=0,task_config=task,external_messages=[])
                    cp=capture(session,RecipeV2Controller(cfg),WeightedBudget(cfg['action_budget'],cfg['token_budget'],base_cost=cfg['base_decision_cost']),
                        create_environment(cfg,task_config=task),self.worker.model_info,calibration)
                    verify_criterion_evidence(_binding(item['episode'],'observable_criterion'),(self.store.root/stage['criterion_evidence_file']['path']).read_bytes(),cp)

    def _log(self,target,event):
        self.worker.resource_check()
        guarded_bytes(Path(target)/'events.jsonl',(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n').encode(),mode='ab',guard=self.store.guard)

    def run_main(self,item,entry,target,receipt):
        from .checkpoints import branch
        episode=item['episode'];payload=dict(run_id=entry['run_id'],out_dir=str(self.store.run_path(entry['run_id'])),mode='experiment',
            config=episode['recipe'],task_config=episode['task_config'],calibration_id=receipt['calibration_id'],
            calibration_dir=str(self.store.calibration_path(receipt['calibration_id'])))
        overrides=deepcopy(item['launch_overrides']);state_source=overrides.pop('state_source',None);payload.update(overrides)
        saved_output=self.worker.output;final=None
        def output(event):
            nonlocal final
            if event.get('run_id')==entry['run_id']:
                self._log(target,event)
                if event.get('type')=='session_finished':final=deepcopy(event)
            saved_output(event)
        self.worker.output=output
        try:
            if state_source:
                cp=branch(state_source['checkpoint'],run_id=entry['run_id'],parent_run_id=state_source['run_id'],event_cutoff=state_source['event_cutoff'],parent_prefix_sha256=state_source['parent_prefix_sha256'])
                self.worker.restore_session(dict(payload,checkpoint=cp))
            else:self.worker.start_session(payload)
            self.worker.run_experiment()
        except (ResourceStop, KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            if self.worker.session and self.worker.session.get('run_id')==entry['run_id'] and not self.worker.session.get('finished'):
                self.worker.finish('stopped' if self.worker.stop.is_set() else 'failed',str(exc))
            if final is None:raise
        finally:self.worker.output=saved_output
        if final is None:raise ValueError('Worker did not emit final main-run evidence')
        atomic_json(Path(target)/'main-result.json',final,guard=self.store.guard)
        run_dir=self.store.run_path(entry['run_id'])
        # Worker-owned files are complete before this return. Service-owned
        # manifest/events are written asynchronously and verified against our
        # immutable journal later, never hashed during an in-flight IPC write.
        evidence=[p for p in (run_dir/'conversation.json',run_dir/'checkpoint.json',run_dir/'yoke-final.json') if p.is_file()]
        evidence.extend(sorted((run_dir/'checkpoints').glob('*.json.gz')))
        result={'status':final['status'],'summary':final['summary'],'evidence_files':[evidence_file(self.store.root,p) for p in evidence]}
        if state_source:
            old=state_source['checkpoint'];summary=deepcopy(final['summary']);env=old['environment'];old_correct=sum(bool(row.get('correct')) for row in env.get('results',[]))
            for key,value in {'tokens':old['budget']['tokens'],'actions':old['budget']['actions'],'completed_decisions':old['budget']['completed_decisions'],
                              'voluntary_calls':old['effect']['counts']['model'],'correct':old_correct}.items():
                if key in summary:summary[key]-=value
            summary['assigned']=max(0,len(env['records'])-env['index']) if 'index' in env else summary.get('assigned')
            summary['invalid_actions']=sum(r.get('invalid',0) for r in final['summary'].get('effects',{}).get('phase_counts',{}).values())-sum(r.get('invalid',0) for r in old['effect'].get('phase_counts',{}).values())
            result['analysis_summary']=summary
        return result

    def run_stage(self,item,stage,entry,target,receipt):
        kind=stage['kind'];episode=item['episode'];w=self.worker;source_dir=self.store.run_path(entry['run_id'])
        calibration_dir=self.store.calibration_path(receipt['calibration_id'])
        if kind!='calibration-validation' and (not w.session or w.session.get('run_id')!=entry['run_id']):
            return {'status':'partial','summary':{'termination':'main_session_unavailable','tokens':0}}
        def emit(event):
            # Branch telemetry lives in its own stage log, never main conversation.
            self._log(target,dict(event,research_stage=stage['id'],parent_run_id=entry['run_id']))
        if kind=='calibration-validation':
            bundle=Path(target)/'calibration'
            metadata=w.runtime.validate_calibration(dict(stage['config'],calibration_dir=str(calibration_dir)),bundle,emit,w.should_stop)
            generated=sum(len(row['generation']['token_ids']) for row in read_json(bundle/'continuations.json',[]))
            if generated>stage['token_budget']:raise ValueError('Calibration stage exceeded its frozen additional token budget')
            return {'status':'complete','summary':{'calibration_bundle':str(bundle.relative_to(self.store.root)), 'tokens':generated,
                'token_budget':stage['token_budget'],'validation_status':metadata.get('status'),'independent_ratings':metadata.get('continuation_scoring')}}
        if kind=='discovery' and stage['config'].get('information_only'):
            visible=self.worker.session['messages']
            if not any(m.get('content')==stage['disclosure_text'] for m in visible):raise ValueError('Declared disclosure did not reach the model context')
            return {'status':'complete','summary':{'information_only':True,'disclosure_text':stage['disclosure_text'],'disclosure_sha256':content_hash(stage['disclosure_text']),'tokens':0,'discovery_demonstrated':False}}
        if kind=='discovery':
            from .checkpoints import validate
            from .diagnostic_runner import run_diagnostics
            from .lifecycle import list_boundaries,read_boundary
            from .discovery import summarize_scores
            spec=stage['spec'];remaining=stage['token_budget'];boundaries=[];scores=[];index=list_boundaries(source_dir);failure=None
            original_messages=deepcopy(w.session['messages'])
            for count in spec['completed_boundaries']:
                matches=[r for r in index if r['turns']==count]
                planned=len(spec['contexts'])
                if not matches or remaining<=0 or w.should_stop() or failure:
                    boundaries.append({'boundary':count,'status':'prior_diagnostic_failure' if failure else 'unreached_checkpoint' if not matches else 'stopped' if w.stop.is_set() else 'stage_budget_exhausted',
                        'planned_contexts':planned,'observed_contexts':0,'completed_contexts':0,'partial_contexts':0,'failed_contexts':0,'missing_contexts':planned,'tokens':0});continue
                cp=read_boundary(source_dir,matches[0]['id']);validate(cp,w.model_info,w.read_calibration_identity(calibration_dir))
                folder=Path(target)/f'boundary-{count:05d}'
                try:
                    summary=run_diagnostics(w.runtime,cp,spec,stage['answer_keys'][str(count)],pair_id=episode['pair_id'],calibration_dir=calibration_dir,
                        out_dir=folder,token_budget=remaining,turn_token_limit=min(episode['recipe']['turn_token_limit'],remaining),emit=emit,should_stop=w.should_stop,
                        effect_policy=stage['config'].get('effect_policy','carry_boundary_state'),
                        criterion_evidence=(self.store.root/stage['criterion_evidence_file']['path']).read_text(encoding='utf-8') if 'criterion_evidence_file' in stage else None)
                except Exception as exc:
                    failure=str(exc);summary=read_json(folder/'summary.json')
                    if not isinstance(summary,dict):raise
                remaining-=summary['tokens'];boundaries.append({'boundary':count,'status':'failed' if failure else 'complete' if summary['completed_contexts']==planned else 'partial',**summary})
                scores.extend(row['score'] for row in read_json(folder/'diagnostic-records.json',[]) if row.get('status')=='complete' and 'score' in row)
            if w.session['messages']!=original_messages:raise ValueError('Diagnostic runner mutated the parent conversation')
            counts={key:sum(r[key] for r in boundaries) for key in ('planned_contexts','observed_contexts','completed_contexts','partial_contexts','failed_contexts','missing_contexts')}
            return {'status':'stopped' if w.stop.is_set() else 'failed' if failure else 'complete' if counts['completed_contexts']==counts['planned_contexts'] else 'partial',
                'summary':{**counts,'boundaries':boundaries,'error':failure,
                    'tokens':stage['token_budget']-remaining,'token_budget':stage['token_budget'],'scoring':summarize_scores(scores),'answer_feedback_to_parent':False}}
        if kind=='transfer':
            cp=stage['source']['checkpoint'];policy=stage['config']['policy'];messages=w.session['messages']
            if policy=='conversation_history' and messages[:len(cp['session']['messages'])]!=cp['session']['messages']:raise ValueError('Full source visible history was not preserved')
            if policy=='matched_experiment_checkpoint' and w.session.get('parent_prefix_sha256')!=stage['source']['parent_prefix_sha256']:raise ValueError('Internal state continuation lost its bound prefix')
            return {'status':'complete','summary':{'policy':policy,'source_checkpoint_sha256':cp['sha256'],'source_visible_prefix_sha256':content_hash(cp['session']['messages']),
                'parent_prefix_sha256':stage['source']['parent_prefix_sha256'],'preserve_visible_history':stage['config']['preserve_visible_history'],'fresh_budget':stage['config']['fresh_budget'],'tokens':0,
                'discovery_success_inferred':False}}
        if kind=='yoke':
            if not w.yoke or w.yoke.schedule['sha256']!=stage['schedule_sha256']:raise ValueError('Declared source yoke was not executed')
            report=w.yoke.report();atomic_json(Path(target)/'yoke-final.json',w.yoke.snapshot(),guard=self.store.guard)
            return {'status':'complete' if report['schedule_complete'] and report['coverage']=='observed_source_covered' else 'partial','summary':report}
        raise ValueError('No executor exists for declared stage: '+kind)
