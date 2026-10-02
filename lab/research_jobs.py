"""Frozen research plans, durable attempt receipts and all-planned denominators."""
from copy import deepcopy
import hashlib
import gzip
import json
import os
from itertools import zip_longest
from pathlib import Path
import re
import threading

from .effects import content_hash
from .protocol_library import execution_requests
from .storage import ID, atomic_json, new_id, read_json, source_manifest, utc_now

# The legacy start_batch route can execute only these envelopes. Full protocol
# capability advertisements come from the explicitly wired protocol backend.
CAPABILITIES = frozenset({'recipe_v2', 'task_axis_v1'})
TERMINAL = {'complete', 'failed', 'partial', 'stopped', 'cancelled', 'resource_stopped', 'interrupted'}
_EMERGENCY_LOCK=threading.RLock()
_EMERGENCY_BYTES=64*1024


def _dispatch(store,identifier,execution_id):
    if not isinstance(execution_id,str) or not ID.fullmatch(execution_id):raise ValueError('Invalid research execution ID')
    directory=job_path(store,identifier)/'dispatches'/execution_id
    if directory.is_symlink() or not directory.is_dir():raise ValueError('Research dispatch was not admitted')
    admission=read_json(directory/'admission.json',{})
    expected={'schema_version','job_id','execution_id','expansion_sha256','created_at','slots','owner','sha256'}
    if set(admission)!=expected or admission['schema_version']!=1 or admission['job_id']!=identifier or admission['execution_id']!=execution_id:
        raise ValueError('Invalid research dispatch admission')
    if content_hash({k:v for k,v in admission.items() if k!='sha256'})!=admission['sha256']:raise ValueError('Research dispatch admission hash mismatch')
    owner=admission['owner']
    if not isinstance(owner,dict) or not {'service_pid'}<=set(owner)<={'service_pid','worker_pid'} or any(type(pid) is not int or pid<=0 for pid in owner.values()):raise ValueError('Invalid research dispatch owner')
    if not isinstance(admission['slots'],dict) or set(admission['slots'])!={'supervisor','worker'} or any(not re.fullmatch(r'\.emergency-[a-f0-9]{32}',str(v)) for v in admission['slots'].values()):raise ValueError('Invalid research emergency slot')
    return directory,admission


def prepare_dispatch(store,identifier):
    """Physically reserve separate worker/supervisor records BEFORE dispatch.

    Each actor owns its own slot, so a worker exception and simultaneous watchdog
    stop never race to consume the same disk headroom. Admission is durable and
    can be reopened after a process restart without allocating more space.
    """
    from .resources import EmergencyMetadata
    receipt,_,_=load_plan(store,identifier)
    execution_id=new_id('execution');directory=job_path(store,identifier)/'dispatches'/execution_id
    slots={}
    try:
        for actor in ('supervisor','worker'):slots[actor]=EmergencyMetadata(directory,allowance_bytes=_EMERGENCY_BYTES).allocate(store.guard)
        admission=dict(schema_version=1,job_id=identifier,execution_id=execution_id,expansion_sha256=receipt['expansion_sha256'],created_at=utc_now(),slots={k:v.path.name for k,v in slots.items()},owner={'service_pid':os.getpid()})
        admission['sha256']=content_hash(admission);atomic_json(directory/'admission.json',admission,guard=store.guard)
        receipt['active_execution_id']=execution_id;save_receipt(store,receipt)
    except BaseException:
        for slot in slots.values():slot.release()
        raise
    return execution_id


def bind_dispatch_worker(store,identifier,execution_id,worker_pid):
    if type(worker_pid) is not int or worker_pid<=0:raise ValueError('Invalid owned worker process')
    directory,admission=_dispatch(store,identifier,execution_id)
    admission['owner']['worker_pid']=worker_pid;admission.pop('sha256');admission['sha256']=content_hash(admission)
    atomic_json(directory/'admission.json',admission,guard=store.guard)


def recover_dispatches(store,pid_alive):
    """Settle only abandoned executions whose recorded owners are both gone."""
    root=store.root/'research'
    if not root.is_dir() or root.is_symlink():return
    for directory in root.iterdir():
        if not directory.is_dir() or directory.is_symlink():continue
        receipt=read_json(directory/'receipt.json',{});execution_id=receipt.get('active_execution_id')
        if not execution_id or receipt.get('status') in TERMINAL:continue
        _,admission=_dispatch(store,directory.name,execution_id)
        if any(pid_alive(pid) for pid in admission['owner'].values()):continue
        record_dispatch_stop(store,directory.name,execution_id,'failed','research_owner_exited',actor='supervisor')
        release_dispatch(store,directory.name,execution_id,actor='all')


def dispatch_stop(store,identifier,execution_id):
    directory,admission=_dispatch(store,identifier,execution_id);records=[]
    for path in directory.glob('resource-stop-*.json'):
        if path.is_symlink() or path.stat().st_size>_EMERGENCY_BYTES:raise ValueError('Invalid research emergency record')
        row=read_json(path,{})
        if (row.get('kind')!='opium-bench/research-emergency' or row.get('schema_version')!=1 or row.get('job_id')!=identifier
                or row.get('execution_id')!=execution_id or row.get('expansion_sha256')!=admission['expansion_sha256']
                or row.get('status') not in {'resource_stopped','failed','stopped'} or row.get('actor') not in {'supervisor','worker'}
                or content_hash({k:v for k,v in row.items() if k!='sha256'})!=row.get('sha256')):raise ValueError('Research emergency integrity failed')
        records.append(row)
    if len(records)>2 or len({r['actor'] for r in records})!=len(records):raise ValueError('Duplicate research emergency actor')
    rank={'resource_stopped':3,'failed':2,'stopped':1}
    return max(records,key=lambda r:(rank[r['status']],r['time'])) if records else None


def record_dispatch_stop(store,identifier,execution_id,status,error,*,actor='supervisor'):
    """Write bounded terminal metadata from reserved space without normal writes."""
    from .resources import EmergencyMetadata
    if status not in {'resource_stopped','failed','stopped'} or actor not in {'supervisor','worker'}:raise ValueError('Invalid research stop status/actor')
    with _EMERGENCY_LOCK:
        directory,admission=_dispatch(store,identifier,execution_id)
        current=read_json(job_path(store,identifier)/'receipt.json',{})
        if current.get('active_execution_id')!=execution_id or current.get('status') in TERMINAL:return dispatch_stop(store,identifier,execution_id)
        existing=dispatch_stop(store,identifier,execution_id)
        slot=EmergencyMetadata(directory,allowance_bytes=_EMERGENCY_BYTES);slot.path=directory/admission['slots'][actor]
        if not slot.path.is_file() or slot.path.is_symlink():return existing
        detail=error.to_dict() if hasattr(error,'to_dict') else error
        # Large exception payloads must never exhaust the final metadata slot.
        reason=str(detail.get('reason',detail.get('error',detail))) if isinstance(detail,dict) else str(detail)
        record=dict(kind='opium-bench/research-emergency',schema_version=1,job_id=identifier,execution_id=execution_id,
            expansion_sha256=admission['expansion_sha256'],status=status,actor=actor,time=utc_now(),reason=reason[:4000],
            receipt_sha256=current.get('receipt_sha256'),accounting='Last durable stage evidence retained; uncommitted generation consumption may be unknown.')
        record['sha256']=content_hash(record)
        raw=(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n').encode()
        if len(raw)>_EMERGENCY_BYTES:raise ValueError('Research stop record exceeds its reserved space')
        # Reuse already allocated blocks; publish only after fsync, so readers
        # never observe a half-written error JSON during watchdog races.
        with slot.path.open('r+b') as stream:
            stream.write(raw);stream.truncate();stream.flush();os.fsync(stream.fileno())
        slot.path.replace(directory/f'resource-stop-{actor}.json')
        return dispatch_stop(store,identifier,execution_id)


def release_dispatch(store,identifier,execution_id,*,actor='supervisor'):
    """Release unused headroom only after a durable terminal receipt/overlay."""
    if actor not in {'supervisor','worker','all'}:raise ValueError('Unknown research emergency actor')
    directory,admission=_dispatch(store,identifier,execution_id)
    receipt=read_json(job_path(store,identifier)/'receipt.json',{})
    if not (receipt.get('active_execution_id')==execution_id and receipt.get('status') in TERMINAL) and not dispatch_stop(store,identifier,execution_id):
        raise ValueError('Research execution has no durable terminal outcome')
    for name in ('supervisor','worker') if actor=='all' else (actor,):(directory/admission['slots'][name]).unlink(missing_ok=True)


def _stop_for(store,identifier,execution_id,cache):
    if not execution_id:return None
    if execution_id not in cache:cache[execution_id]=dispatch_stop(store,identifier,execution_id)
    return cache[execution_id]


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def evidence_file(root,path):
    root=Path(root).resolve();path=Path(path)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError('Evidence must be an ordinary file in the managed data root')
    return {'path':str(path.resolve().relative_to(root)).replace('\\','/'),'sha256':file_hash(path),'bytes':path.stat().st_size}


def verify_files(root,records):
    if not isinstance(records,list):raise ValueError('Evidence file list is missing')
    for row in records:
        if not isinstance(row,dict) or set(row)!={'path','sha256','bytes'}:raise ValueError('Invalid evidence file receipt')
        path=Path(root)/row['path']
        if evidence_file(root,path)!=row:raise ValueError('Frozen evidence file changed: '+str(row['path']))


def job_path(store,identifier):
    if not isinstance(identifier,str) or not ID.fullmatch(identifier):raise ValueError('Invalid research job ID')
    root=store.root/'research';path=root/identifier
    if root.is_symlink() or path.is_symlink() or not path.is_dir():raise FileNotFoundError('Research job not found')
    return path


def save_receipt(store,receipt):
    value=deepcopy(receipt);value.pop('receipt_sha256',None)
    value['receipt_sha256']=content_hash(value)
    atomic_json(job_path(store,receipt['id'])/'receipt.json',value,guard=store.guard)
    # Keep active entry/unit references attached while persisting a snapshot.
    receipt['receipt_sha256']=value['receipt_sha256']


def load_plan(store,identifier,*,verify_sources=False,expected_expansion_sha256=None):
    directory=job_path(store,identifier)
    receipt=read_json(directory/'receipt.json');preview=read_json(directory/'preview.json');plan=read_json(directory/'plan.json')
    if not isinstance(receipt,dict) or type(receipt.get('schema_version')) is not int or receipt['schema_version']!=1 or receipt.get('id')!=identifier:raise ValueError('Invalid research receipt')
    check=deepcopy(receipt);checksum=check.pop('receipt_sha256',None)
    if checksum!=content_hash(check):raise ValueError('Research receipt integrity failed')
    if not isinstance(preview,dict) or not isinstance(plan,dict):raise ValueError('Frozen research plan missing')
    payload=deepcopy(preview);recorded=payload.pop('expansion_sha256',None)
    if content_hash(payload)!=recorded or receipt.get('expansion_sha256')!=recorded:raise ValueError('Frozen research expansion integrity failed')
    if expected_expansion_sha256 is not None and expected_expansion_sha256!=recorded:raise ValueError('Expected expansion hash does not match frozen job')
    if content_hash(plan)!=receipt.get('plan_sha256'):raise ValueError('Frozen execution plan integrity failed')
    expected={e['id']:e for e in preview['episodes']}
    if set(expected)!={e['episode']['id'] for e in plan['episodes']} or len(expected)!=len(plan['episodes']):raise ValueError('Execution plan episodes differ from preview')
    for row in plan['episodes']:
        envelope=deepcopy(row['episode'])
        for key in ('bindings','bound_execution_sha256','criterion_validation_status','interpretation_eligible'):envelope.pop(key,None)
        if envelope!=expected[envelope['id']]:raise ValueError('Bound episode changed its frozen expansion')
        if row['episode']['bound_execution_sha256']!=content_hash({'execution':row['episode']['execution_sha256'],'bindings':row['episode']['bindings']}):raise ValueError('Bound episode hash mismatch')
    if verify_sources:
        if receipt['source']['source_sha256']!=source_manifest()['source_sha256']:raise ValueError('Source code hashes changed; start a new job rather than silently resuming another implementation')
        verify_files(store.root,plan['source_files'])
    return receipt,preview,plan


def create_job(store,preview,calibration_id,*,bindings=None,guard=None,capabilities=None,model_info=None):
    requests=execution_requests(preview,set(CAPABILITIES if capabilities is None else capabilities),bindings)
    bindings={} if bindings is None else bindings
    extra=set(bindings)-set(preview['unresolved_bindings'])-{'criterion_evidence'}
    if extra:raise ValueError('Unknown execution bindings: '+str(sorted(extra)))
    for request in requests:
        if 'criterion_evidence' in bindings:
            if 'observable_criterion' not in request['bindings']:raise ValueError('Criterion evidence needs an observable criterion stage')
            request['bindings']['criterion_evidence']=deepcopy(bindings['criterion_evidence'])
        # Declared validation status is not evidence verification. Eligibility
        # is established only against each actual diagnostic checkpoint.
        if request.get('interpretation_eligible') is not None:request['interpretation_eligible']=False
        request['bound_execution_sha256']=content_hash({'execution':request['execution_sha256'],'bindings':request['bindings']})
    if len(requests)>1000:raise ValueError('One local job may contain at most 1000 episodes; split the prespecified matrix explicitly')
    from .protocol_runner import prepare_episode
    calibration=store.calibration_path(calibration_id)
    calibration_files=[evidence_file(store.root,calibration/name) for name in ('calibration.json','vectors.npz')]
    plans=[prepare_episode(store,request,calibration_id) for request in requests]
    all_files={row['path']:row for row in calibration_files}
    for episode in plans:
        for row in episode['source_files']:
            if row['path'] in all_files and all_files[row['path']]!=row:raise ValueError('Bound source changed during job admission')
            all_files[row['path']]=row
    plan={'schema_version':1,'episodes':plans,'source_files':[all_files[key] for key in sorted(all_files)]}
    if guard:guard.preflight({store.root:preview['estimates']['storage_reservation_bytes']})
    identifier=new_id('research');directory=store.root/'research'/identifier;directory.mkdir(parents=True,exist_ok=False)
    receipt=dict(schema_version=1,id=identifier,created_at=utc_now(),expansion_sha256=preview['expansion_sha256'],plan_sha256=content_hash(plan),
        calibration_id=calibration_id,calibration_sha256={Path(r['path']).name:r['sha256'] for r in calibration_files},
        source=source_manifest(),model_info=deepcopy(model_info),status='prepared',entries=[],protocol_id=preview['protocol_id'],title=preview['protocol_id'].replace('_',' ').title(),planned=len(requests))
    atomic_json(directory/'preview.json',preview,guard=store.guard);atomic_json(directory/'plan.json',plan,guard=store.guard);save_receipt(store,receipt)
    entries=[]
    try:
        for episode in requests:
            run_id,path=store.create('experiment',episode['recipe'],source_identity=receipt['source'])
            store.update(run_id,research_job=identifier,research_episode=episode,task_config=episode['task_config'])
            receipt['entries'].append(dict(episode_id=episode['id'],run_id=run_id,attempt=1,status='queued',units=[]))
            save_receipt(store,receipt)
            entries.append(dict(run_id=run_id,out_dir=str(path),mode='experiment',config=episode['recipe'],task_config=episode['task_config'],
                calibration_id=calibration_id,calibration_dir=str(calibration),research_episode_id=episode['id']))
    except Exception:
        receipt['status']='preparation_failed';save_receipt(store,receipt)
        for entry in entries:store.update(entry['run_id'],status='cancelled',summary={'termination':'research_job_preparation_failed'})
        raise
    return receipt,entries


def verify_unit(store,unit):
    if unit.get('status') in TERMINAL:
        if not unit.get('artifacts'):raise ValueError('Settled stage is missing artifact evidence')
        verify_files(store.root,unit['artifacts'])
        if content_hash(unit.get('summary'))!=unit.get('summary_sha256'):raise ValueError('Stage summary hash mismatch')
        if 'analysis_summary' in unit and content_hash(unit['analysis_summary'])!=unit.get('analysis_summary_sha256'):raise ValueError('Stage analysis summary hash mismatch')


def verify_main_events(store,path,unit):
    """Compare service records with the immutable worker-side stage journal.

    Service timestamps/global sequence numbers are additive metadata. Every
    actual worker event and its full content must otherwise match in order.
    """
    logs=[row for row in unit.get('artifacts',[]) if row['path'].endswith('/main/events.jsonl')]
    if not logs:return  # Generic non-worker backends have their own evidence.
    if len(logs)!=1:raise ValueError('Main stage has ambiguous raw event evidence')
    source=path/'events.jsonl';compressed=not source.exists()
    if compressed:source=path/'events.jsonl.gz'
    if not source.is_file() or source.is_symlink():raise ValueError('Main event evidence is missing')
    with (store.root/logs[0]['path']).open(encoding='utf-8') as recorded,(gzip.open if compressed else open)(source,'rt',encoding='utf-8') as actual:
        for expected_line,actual_line in zip_longest(recorded,actual):
            if expected_line is None or actual_line is None:raise ValueError('Main event count differs from the frozen worker evidence')
            expected=json.loads(expected_line);value=json.loads(actual_line)
            if set(value)-set(expected)-{'seq','time'} or any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Main event content differs from the frozen worker evidence')


def read_job(store,identifier,*,verify=False):
    receipt,preview,plan=load_plan(store,identifier)
    stops={};job_stop=_stop_for(store,identifier,receipt.get('active_execution_id'),stops)
    if job_stop:
        receipt=dict(receipt,stored_status=receipt['status'],status=job_stop['status'],emergency_stop=job_stop)
    expected={e['id']:e for e in preview['episodes']};planned={p['episode']['id']:p for p in plan['episodes']}
    states=[];seen={};latest={}
    for entry in receipt['entries']:
        episode_id=entry.get('episode_id');attempt=entry.get('attempt')
        if episode_id not in expected or type(attempt) is not int or attempt!=seen.get(episode_id,0)+1:raise ValueError('Unexpected or duplicate research receipt episode attempt')
        seen[episode_id]=attempt;episode=expected[episode_id]
        path=store.run_path(entry['run_id']);manifest=read_json(path/'manifest.json',{})
        if content_hash(manifest.get('config'))!=content_hash(episode['recipe']) or manifest.get('task_config')!=episode['task_config']:raise ValueError('Recorded episode differs from its frozen task/recipe')
        summary=read_json(path/'summary.json',{})
        resource=store.resource_stop_record(path);status='resource_stopped' if resource else manifest.get('status','unknown')
        if resource:summary=resource.get('summary',{})
        if entry.get('runner_managed'):
            status='resource_stopped' if resource else entry['status'];main=next((u for u in entry['units'] if u['id']=='main'),None)
            if main and not resource:summary=main.get('summary',{})
            unit_ids=[u['id'] for u in entry['units']]
            expected_ids=['main']+[s['id'] for s in episode['stages']]
            if len(unit_ids)!=len(set(unit_ids)) or not set(unit_ids)<=set(expected_ids):raise ValueError('Unexpected/duplicate execution stage receipt')
            if status=='complete' and (set(unit_ids)!=set(expected_ids) or any(u['status']!='complete' for u in entry['units'])):raise ValueError('Episode was marked complete with ignored stages')
            if verify:
                for unit in entry['units']:verify_unit(store,unit)
                if main and main.get('status')=='complete':verify_main_events(store,path,main)
        elif episode['stages'] or episode['runtime_controls']:
            if status=='complete':raise ValueError('Extra-stage episode has no executed envelope receipt')
        if verify and status=='complete':
            run=store.read_run(entry['run_id']);final=[event for event in run['events'] if event.get('type')=='session_finished']
            if not final or final[-1].get('status')!='complete' or final[-1].get('summary')!=summary:raise ValueError('Completed episode is missing matching final evidence')
        row=dict(entry,status=status,factors=episode['factors'],pair_id=episode['pair_id'],summary=summary)
        entry_stop=_stop_for(store,identifier,entry.get('execution_id'),stops)
        if entry_stop and status!='complete':
            row.update(status=entry_stop['status'] if status not in TERMINAL else status,emergency_stop=entry_stop,units=deepcopy(entry.get('units',[])))
            for unit in row['units']:
                if unit['status']=='running':unit.update(stored_status='running',status=entry_stop['status'],emergency_stop=entry_stop,
                    summary=dict(unit.get('summary',{}),termination=entry_stop['status'],consumption_unknown=True))
        states.append(row);latest[episode_id]=row
    missing=sorted(set(expected)-set(seen));counts={status:sum(row['status']==status for row in latest.values()) for status in sorted({row['status'] for row in latest.values()})}
    return dict(receipt=receipt,protocol_id=preview['protocol_id'],mode=preview['mode'],planned=len(expected),missing_episode_ids=missing,
        states=states,latest_states=list(latest.values()),status_counts=counts,settled=not missing and all(row['status'] in TERMINAL for row in latest.values()),
        all_complete=not job_stop and not missing and bool(latest) and all(row['status']=='complete' for row in latest.values()))


def analysis_records(store,identifier,*,attempt_policy='first'):
    if attempt_policy not in {'first','latest'}:raise ValueError('Choose an explicit first or latest attempt analysis')
    job=read_job(store,identifier,verify=True);_,preview,_=load_plan(store,identifier)
    rows={}
    for row in job['states']:
        if attempt_policy=='latest' or row['episode_id'] not in rows:rows[row['episode_id']]=row
    records=[]
    for episode in preview['episodes']:
        row=rows.get(episode['id'],{});summary=row.get('summary',{})
        main=next((unit for unit in row.get('units',[]) if unit['id']=='main'),{})
        summary=main.get('analysis_summary',summary)
        if 'invalid_actions' not in summary and isinstance(summary.get('effects',{}).get('phase_counts'),dict):
            summary=dict(summary,invalid_actions=sum(v.get('invalid',0) for v in summary['effects']['phase_counts'].values()))
        def rate(numerator,denominator):
            return {'numerator':summary[numerator],'denominator':summary[denominator]} if numerator in summary and denominator in summary else None
        status=row.get('status','pending');status='partial' if status in {'resource_stopped','interrupted'} else 'pending' if status in {'queued','running','unknown'} else status
        records.append(dict(id=row.get('run_id',episode['id']),episode_id=episode['id'],attempt=row.get('attempt'),attempt_policy=attempt_policy,
            pair_id=episode['pair_id'],family=episode['family'],arm=' | '.join(f'{k}={v}' for k,v in sorted(episode['factors'].items())),status=status,
            outcomes={'task_accuracy':rate('correct','assigned'),'aux_per_decision':rate('voluntary_calls','completed_decisions'),
                'invalid_per_decision':rate('invalid_actions','completed_decisions'),'tokens':summary.get('tokens'),'budget_units':summary.get('actions')}))
    return records


def analyze_job(store,identifier,*,endpoint=None,arm_a=None,arm_b=None,attempt_policy='first'):
    from .statistics import paired_bootstrap,task_benefit_bound
    from .behavioral_analysis import analyze_behavior
    records=analysis_records(store,identifier,attempt_policy=attempt_policy);job=read_job(store,identifier)
    _,preview,_=load_plan(store,identifier)
    result={'schema_version':1,'job_id':identifier,'expansion_sha256':preview['expansion_sha256'],'attempt_policy':attempt_policy,
        'planned_episodes':len(records),'total_recorded_attempts':len(job['states']),'status_counts':job['status_counts'],
        'observed_task_outcomes':sum(r['outcomes']['task_accuracy'] is not None for r in records),'records':records,
        'stages':[{'episode_id':row['episode_id'],'attempt':row['attempt'],'units':row.get('units',[])} for row in job['states']],
        'interpretation':'Exploratory episode-level evidence. Retries are retained; first-attempt analysis is the default.'}
    selected_ids={row['id'] for row in records};episodes={row['id']:row for row in preview['episodes']}
    result['behavioral_reports']=[]
    for row in job['states']:
        events=store.read_run(row['run_id'])['events']
        result['behavioral_reports'].append({'episode_id':row['episode_id'],'run_id':row['run_id'],'attempt':row['attempt'],
            'status':row['status'],'included_attempt':row['run_id'] in selected_ids,
            'report':analyze_behavior(events,recipe=episodes[row['episode_id']]['recipe'],recorded_summary=row['summary']) if events else None,
            'observation_status':'observed_events' if events else 'no_recorded_events'})
    if any(value is not None for value in (endpoint,arm_a,arm_b)):
        if not all(isinstance(value,str) and value for value in (endpoint,arm_a,arm_b)):raise ValueError('A contrast requires endpoint and both arms')
        options=preview['analysis']
        selected=[e['pair_id'] for e in preview['episodes'] if ' | '.join(f'{k}={v}' for k,v in sorted(e['factors'].items())) in {arm_a,arm_b}]
        result['contrast']=paired_bootstrap(records,endpoint,arm_a,arm_b,unit=options['unit'],seed=options['seed'],iterations=options['bootstrap_iterations'],
            planned_pair_ids=sorted(set(selected)),endpoint_bounds=[0,1] if endpoint in {'task_accuracy','aux_per_decision','invalid_per_decision'} else None)
        if endpoint=='task_accuracy':result['task_benefit_bound']=task_benefit_bound(result['contrast'],preview['practical_task_benefit_margin'])
    return result


def catalog(store):
    """Small metadata-only catalog for polling; full verification is explicit."""
    root=store.root/'research';rows=[]
    if root.is_symlink():return rows
    for directory in sorted(root.glob('*'),reverse=True) if root.exists() else []:
        if directory.is_symlink() or not directory.is_dir():continue
        receipt=read_json(directory/'receipt.json',{})
        if receipt.get('schema_version')!=1 or receipt.get('id')!=directory.name:continue
        latest={entry.get('episode_id'):entry for entry in receipt.get('entries',[]) if isinstance(entry,dict)}
        stops={};stop=_stop_for(store,directory.name,receipt.get('active_execution_id'),stops)
        latest={key:dict(entry) for key,entry in latest.items()}
        for entry in latest.values():
            event=_stop_for(store,directory.name,entry.get('execution_id'),stops)
            if event and entry.get('status') not in TERMINAL:entry['status']=event['status']
        counts={status:sum(e.get('status')==status for e in latest.values()) for status in {'complete','partial','failed','queued','running','stopped','resource_stopped','interrupted'}}
        rows.append({'id':directory.name,'status':stop['status'] if stop else receipt.get('status'),'protocol_id':receipt.get('protocol_id'),
            'title':receipt.get('title',receipt.get('protocol_id',directory.name)),'expansion_sha256':receipt.get('expansion_sha256'),
            'planned':receipt.get('planned',len(latest)),'completed':counts['complete'],'status_counts':counts,
            'attempts':len(receipt.get('entries',[])),'created_at':receipt.get('created_at')})
    return rows
