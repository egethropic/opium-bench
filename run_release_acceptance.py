#!/usr/bin/env python3
"""Explicit, bounded acceptance stages through the running local lab API.

Planning is offline. Execution requires --execute, a matching already-loaded
model and a NEW output directory. No downloads, automatic retries or model
switches occur. A stage receipt is engineering evidence, never a feeling score.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote, urlparse
from urllib.request import urlopen

from lab.effects import content_hash
from lab.protocol_library import dry_run, freeze_protocol, validate_protocol
from lab.recipes_v2 import resolve_recipe
from lab.resources import EmergencyMetadata, ResourceGuard, ResourceStop
from lab.storage import guarded_bytes, utc_now
from run_lab_experiments import Client

ROOT = Path(__file__).resolve().parent
DEFAULT_PLAN = ROOT / 'protocols' / 'acceptance' / 'release-v1.json'
STAGES = ('extract', 'validate', 'protocol', 'diagnostic', 'yoke', 'lifecycle', 'portable')
TERMINAL = {'complete', 'failed', 'stopped', 'resource_stopped', 'partial', 'cancelled'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()


def selected_corpus(source, selection):
    """Deterministic matched pairs; keep original text and metadata unchanged."""
    raw = Path(source).read_bytes()
    if sha(raw) != selection['source_sha256']:
        raise ValueError('Frozen research corpus bytes changed')
    document = json.loads(raw)
    groups = defaultdict(lambda: defaultdict(list))
    for row in document['rows']:
        groups[(row['split'], row['context'], row['concept'])][row['pair_id']].append(row)
    chosen = []
    for key in sorted(groups):
        for pair_id in sorted(groups[key])[:selection['pairs_per_stratum']]:
            pair = groups[key][pair_id]
            if len(pair) != 2 or {r['label'] for r in pair} != {0, 1}:
                raise ValueError('Incomplete source pair')
            chosen.extend(sorted(pair, key=lambda row: row['id']))
    if len(chosen) != selection['expected_rows'] or [r['id'] for r in chosen] != selection['row_ids']:
        raise ValueError('Frozen stratified corpus selection changed')
    document['rows'] = chosen
    document['version'] += '-acceptance-160-v1'
    document['provenance'] += ' Acceptance subset: first two sorted pair IDs within each split/context/concept stratum; source SHA256 ' + selection['source_sha256'] + '.'
    document['limitations'] = list(document.get('limitations', [])) + ['Small deterministic engineering subset; not the full research calibration or a powered study.']
    return document


def load_plan(path=DEFAULT_PLAN):
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    payload = deepcopy(plan)
    expected = payload.pop('sha256', None)
    if plan.get('schema_version') != 1 or plan.get('kind') != 'opium-bench-release-acceptance' or content_hash(payload) != expected:
        raise ValueError('Unsupported or altered acceptance plan')
    for profile in plan['profiles'].values():
        validate_protocol(profile['protocol'])
        preview = dry_run(profile['protocol'], 'smoke', order_seed=plan['order_seed'])
        if preview['expansion_sha256'] != profile['expansion_sha256']:
            raise ValueError('Resolver output changed; preserve this plan and explicitly version a new one')
    selected_corpus(ROOT / 'corpora' / 'research-v2.json', plan['corpus_selection'])
    return plan


class Evidence:
    def __init__(self, directory, guard=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.guard = guard or ResourceGuard({'acceptance': self.directory})
        self.files = []
        self.emergency = EmergencyMetadata(self.directory).allocate(self.guard)

    def write(self, name, value, *, raw=False):
        path = self.directory / name
        if path.resolve().parent != self.directory or path.exists():
            raise ValueError('Evidence names must be new local basenames')
        data = value if raw else json_bytes(value)
        guarded_bytes(path, data, mode='xb', guard=self.guard)
        self.files.append({'file': name, 'bytes': len(data), 'sha256': sha(data)})
        return path


class Acceptance:
    def __init__(self, plan, profile, client, evidence, *, timeout=14400, interval=1):
        self.plan, self.profile = plan, plan['profiles'][profile]
        self.profile_id, self.client, self.evidence = profile, client, evidence
        self.timeout, self.interval = timeout, interval
        self.sequence = 0
        self.owned_commands = set()

    def command(self, name, payload):
        self.sequence += 1
        prefix = f'{self.sequence:03d}-{name}'
        self.evidence.write(prefix + '-request.json', payload)
        response = self.client.command(name, payload)
        if response.get('command_id'):
            self.owned_commands.add(response['command_id'])
        self.evidence.write(prefix + '-accepted.json', response)
        return response

    def poll(self, getter, done):
        started = time.monotonic()
        while True:
            value = getter()
            if done(value):
                return value
            if time.monotonic() - started >= self.timeout:
                raise TimeoutError('Acceptance wait expired; no automatic retry was started')
            time.sleep(self.interval)

    def wait(self, accepted, allowed=('complete',)):
        command_id = accepted['command_id']
        result = self.poll(lambda: self.client.get('/api/state'),
            lambda state: (state.get('job') or {}).get('command_id') == command_id and (state.get('job') or {}).get('status') in TERMINAL)
        job = result['job']
        self.evidence.write(f'{self.sequence:03d}-job-result.json', job)
        if job['status'] not in allowed:
            raise RuntimeError('Command ended ' + job['status'] + ': ' + str(job.get('message', '')))
        return result

    def admit(self, *, needs_model=True):
        state = self.client.get('/api/state')
        snapshot = {key: state.get(key) for key in ('worker', 'session', 'job', 'storage')}
        self.evidence.write('admission.json', snapshot)
        if needs_model:
            model = state.get('worker', {}).get('model') or {}
            fingerprint = model.get('fingerprint', {})
            expected = self.profile['model']
            if model.get('model_id', fingerprint.get('model_id')) != expected['model_id'] or fingerprint.get('revision') != expected['revision']:
                raise ValueError('Load the pinned profile using its matching worker interpreter first')
            if state.get('session', {}).get('status') in {'running', 'generating', 'awaiting_user', 'paused', 'queued'}:
                raise ValueError('Stop existing work before starting an acceptance stage')
        return state

    def cancel_owned(self):
        state = self.client.get('/api/state')
        job = state.get('job') or {}
        if job.get('command_id') in self.owned_commands and (job.get('status') == 'running' or state.get('session', {}).get('status') in {'running', 'generating', 'paused', 'awaiting_user'}):
            # Cancellation must still reach the service when local evidence storage fails.
            accepted = self.client.command('stop', {})
            stopped = self.poll(lambda: self.client.get('/api/state'), lambda s:
                s.get('job', {}).get('command_id') == accepted.get('command_id') and s.get('job', {}).get('status') in TERMINAL)
            try: self.evidence.write('owned-cancellation.json', {'accepted': accepted, 'job': stopped['job']})
            except ResourceStop: pass
            return True
        return False

    def calibrate(self, source_id=None):
        if self.profile_id != '4b':
            raise ValueError('27B acceptance uses its existing matching pilot calibration')
        before = {c['id'] for c in self.client.get('/api/state')['calibrations']}
        if source_id:
            accepted = self.command('validate_calibration', {'calibration_id': source_id, **self.plan['validation']})
        else:
            config = deepcopy(self.plan['extraction'])
            config['corpus'] = selected_corpus(ROOT / 'corpora/research-v2.json', self.plan['corpus_selection'])
            self.evidence.write('selected-corpus.json', config['corpus'])
            accepted = self.command('calibrate', config)
        state = self.wait(accepted)
        added = [c['id'] for c in state['calibrations'] if c['id'] not in before]
        if len(added) != 1:
            raise ValueError('Expected one identifiable immutable calibration output')
        identifier = added[0]
        for name in (('calibration.json', 'diagnostics.json', 'scoring-sheet.json') if source_id else ('calibration.json', 'progress.json')):
            self.evidence.write(name, self.client.get('/api/calibrations/' + quote(identifier, safe='') + '/' + name))
        return {'calibration_id': identifier, 'independent_semantic_ratings': 'not_performed', 'source_calibration_id': source_id}

    def protocol(self, calibration, document=None, bindings=None):
        document = deepcopy(document or self.profile['protocol'])
        preview = dry_run(document, 'smoke', order_seed=self.plan['order_seed'])
        self.evidence.write('protocol.json', document)
        self.evidence.write('preview.json', preview)
        accepted = self.command('start_protocol', {'document': document, 'mode': 'smoke', 'seeds': preview['seeds'],
            'order_seed': preview['order_seed'], 'expansion_sha256': preview['expansion_sha256'], 'calibration_id': calibration, 'bindings': bindings or {}})
        job_id = accepted['research_job_id']
        result = self.poll(lambda: self.client.get('/api/research/' + quote(job_id, safe='')),
            lambda job: job.get('settled') or job.get('receipt', {}).get('status') in TERMINAL | {'preparation_failed'})
        self.evidence.write('research-receipt.json', result)
        self.wait(accepted)
        analysis = self.command('analyze_protocol', {'research_job_id': job_id, 'attempt_policy': 'first'})
        self.evidence.write('analysis.json', analysis)
        if not result.get('all_complete'):
            raise RuntimeError('Research protocol retained partial/failed/missing stages; inspect the receipt')
        return {'research_job_id': job_id, 'planned_episodes': len(preview['episodes']),
            'run_ids': [row['run_id'] for row in result['latest_states']], 'expansion_sha256': preview['expansion_sha256']}

    def boundary(self, source_id, *, first=False):
        rows = self.client.get('/api/runs/' + quote(source_id, safe='') + '/checkpoints')['checkpoints']
        rows = [row for row in rows if row['turns'] >= 1]
        if not rows:
            raise ValueError('Source has no completed generated-turn boundary')
        return sorted(rows, key=lambda row: (row['turns'], row['event_cutoff']), reverse=not first)[0]

    def diagnostic(self, calibration, source_id):
        source = self.client.get('/api/runs/' + quote(source_id, safe=''))
        config = source['manifest']['config']
        if config.get('condition') != 'sham':
            raise ValueError('This frozen acceptance diagnostic requires a sham/sham source; its key is neither')
        boundary = self.boundary(source_id, first=True)
        names = [tool['name'] for tool in config['auxiliary_tools'] if tool['visible']]
        if names != self.plan['diagnostic']['spec']['tool_names']:
            raise ValueError('Source tools differ from the frozen diagnostic')
        payload = deepcopy(self.plan['diagnostic'])
        payload['spec']['completed_boundaries'] = [boundary['turns']]
        payload.update(run_id=source_id, checkpoint=boundary['id'], calibration_id=calibration)
        parent_hash = content_hash(source)
        accepted = self.command('run_diagnostics', payload)
        self.wait(accepted)
        result = self.client.get('/api/runs/' + quote(accepted['run_id'], safe=''))
        self.evidence.write('diagnostic-run.json', result)
        if content_hash(self.client.get('/api/runs/' + quote(source_id, safe=''))) != parent_hash:
            raise ValueError('Diagnostic changed the parent evidence')
        return {'run_id': accepted['run_id'], 'source_run_id': source_id, 'source_checkpoint': boundary,
            'source_unchanged': True, 'interpretation_eligible': False, 'criterion_status': 'unvalidated'}

    def yoke(self, calibration, source_id):
        source = self.client.get('/api/runs/' + quote(source_id, safe=''))
        config = deepcopy(source['manifest']['config'])
        if config.get('recipe_version') != 2 or config.get('condition') != 'active':
            raise ValueError('Use a completed active v2 source for this acceptance yoke')
        clocks = {p['decay']['clock'] for p in config['effect_presets']}
        if len(clocks) != 1:
            raise ValueError('Acceptance yoke requires one declared source clock')
        config['demonstration'] = 'none'
        document = deepcopy(self.profile['protocol'])
        document.update(id='acceptance-yoke-v1', title='Bounded source-exposure acceptance', base_recipe=config,
            axes=[{'name': 'case', 'levels': [{'id': 'recipient', 'label': 'Pure recipient'}]}],
            smoke_axes={'case': ['recipient']}, seeds={'smoke': [config['seed']], 'full': [config['seed']]}, pairing_factors=[],
            stages=[{'id': 'yoke-check', 'kind': 'yoke', 'token_budget': 0, 'requires_bindings': ['source_run_id'],
                'config': {'clock': next(iter(clocks)), 'source_factor': source['manifest'].get('research_episode', {}).get('factors', {}),
                    'suppress_own_aux_effects': True}, 'notes': 'No extrapolation or recipient-added active dose; partial source coverage remains visible.'}])
        result = self.protocol(calibration, freeze_protocol(document), {'source_run_id': source_id})
        result.update(source_run_id=source_id, coverage_interpretation='Inspect measured coverage; execution completion does not imply full source exposure was covered.')
        return result

    def lifecycle(self, calibration):
        config = deepcopy(self.profile['protocol']['base_recipe'])
        config.update(task_family='conversation', task_tool_costs={}, demonstration='none', thinking=False, action_budget=10, token_budget=768, turn_token_limit=128)
        config = resolve_recipe(config)
        accepted = self.command('start_session', {'mode': 'chat', 'config': config, 'calibration_id': calibration})
        source_id = accepted['run_id']
        self.wait(accepted)
        self.wait(self.command('chat', {'text': 'Reply with the single word READY.'}))
        self.wait(self.command('pause', {}))
        paused = self.poll(lambda: self.client.get('/api/state'), lambda s: s.get('session', {}).get('status') == 'paused')
        self.evidence.write('paused.json', paused['session'])
        boundary = self.boundary(source_id)
        self.wait(self.command('resume', {}))
        self.poll(lambda: self.client.get('/api/state'), lambda s: s.get('session', {}).get('status') == 'awaiting_user')
        self.wait(self.command('stop', {}), allowed=('complete', 'stopped'))
        source = self.client.get('/api/runs/' + quote(source_id, safe=''))
        self.evidence.write('source-run.json', source)
        branches = []
        for policy in ('continue_state', 'fresh_budget'):
            payload = dict(run_id=source_id, checkpoint=boundary['id'], calibration_id=calibration, policy=policy)
            if policy == 'fresh_budget':
                payload.update(action_budget=4, token_budget=256)
            response = self.command('branch', payload)
            self.wait(response)
            self.wait(self.command('chat', {'text': 'Reply with the single word CONTINUED.'}))
            self.wait(self.command('stop', {}), allowed=('complete', 'stopped'))
            branch = self.client.get('/api/runs/' + quote(response['run_id'], safe=''))
            self.evidence.write(policy + '-run.json', branch)
            branches.append({'policy': policy, 'run_id': response['run_id']})
        if content_hash(self.client.get('/api/runs/' + quote(source_id, safe=''))) != content_hash(source):
            raise ValueError('Branches changed source evidence')
        return {'source_run_id': source_id, 'source_checkpoint': boundary, 'branches': branches, 'pause_resume': True, 'source_unchanged': True}

    def portable(self, source_id):
        from lab.portability import import_bundle
        path = '/api/runs/' + quote(source_id, safe='') + '/bundle'
        raw = self.client.download(path, 128 * 1024**2)
        archive = self.evidence.write('source.zip', raw, raw=True)
        imported = import_bundle(archive, self.evidence.directory / 'imported-runs', guard=self.evidence.guard)
        self.evidence.write('import.json', imported)
        source = self.client.get('/api/runs/' + quote(source_id, safe=''))
        original = json.loads((Path(imported['path']) / 'manifest.json').read_text())
        if original != source['manifest']:
            raise ValueError('Imported manifest differs from the source bytes')
        return {'source_run_id': source_id, 'archive_sha256': sha(raw), 'import': imported,
            'route': 'live HTTP bundle export -> isolated local guarded importer', 'resume_attempted': False}


class LocalClient(Client):
    def __init__(self, url):
        parsed = urlparse(url)
        if parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1', '::1'} or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Acceptance targets the local loopback lab only')
        super().__init__(url)

    def download(self, path, maximum):
        with urlopen(self.url + path, timeout=120) as response:
            raw = response.read(maximum + 1)
        if len(raw) > maximum:
            raise ValueError('Evidence download exceeds its bound')
        return raw


def run_stage(plan, profile, stage, output, client, *, calibration=None, source_run=None, timeout=14400, guard=None):
    if stage not in STAGES or profile not in plan['profiles']:
        raise ValueError('Unknown acceptance stage/profile')
    evidence = Evidence(output, guard=guard)
    runner = Acceptance(plan, profile, client, evidence, timeout=timeout)
    source_bytes = Path(__file__).read_bytes()
    identity = dict(schema_version=1, kind='opium-bench-acceptance-stage', plan_sha256=plan['sha256'], profile=profile,
        stage=stage, calibration_id=calibration, source_run_id=source_run, started_at=utc_now(), automatic_retries=False,
        runner_sha256=sha(source_bytes))
    result = dict(identity, status='failed')
    try:
        evidence.write('runner.py', source_bytes, raw=True)
        evidence.write('started.json', identity)
        evidence.write('plan.json', plan)
        runner.admit(needs_model=stage != 'portable')
        if stage in {'validate', 'protocol', 'diagnostic', 'yoke', 'lifecycle'} and not calibration:
            raise ValueError('This stage needs an explicit compatible --calibration')
        if stage in {'diagnostic', 'yoke', 'portable'} and not source_run:
            raise ValueError('This stage needs an explicit --source-run')
        if stage == 'extract': details = runner.calibrate()
        elif stage == 'validate': details = runner.calibrate(calibration)
        elif stage == 'protocol': details = runner.protocol(calibration)
        elif stage == 'diagnostic': details = runner.diagnostic(calibration, source_run)
        elif stage == 'yoke': details = runner.yoke(calibration, source_run)
        elif stage == 'lifecycle': details = runner.lifecycle(calibration)
        else: details = runner.portable(source_run)
        result.update(status='complete', details=details)
    except (Exception, KeyboardInterrupt) as error:
        result.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'resource_stopped' if isinstance(error, ResourceStop) else 'failed',
            error={'type': type(error).__name__, 'message': str(error)[:4096]})
        try: result['owned_work_stopped'] = runner.cancel_owned()
        except Exception as stop_error: result['stop_error'] = str(stop_error)
    result.update(finished_at=utc_now(), artifacts=deepcopy(evidence.files), interpretation='Engineering acceptance only; no independent semantic or subjective-experience conclusion.')
    try:
        evidence.write('result.json', result)
    except ResourceStop as error:
        result.update(status='resource_stopped', resource=error.to_dict())
        result['emergency_record'] = str(evidence.emergency.finalize(result))
    finally:
        evidence.emergency.release()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--profile', choices=('4b', '27b'), default='4b')
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--calibration')
    parser.add_argument('--source-run')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=float, default=14400)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 86400:
        parser.error('Timeout must be positive and at most one day')
    plan = load_plan(args.plan)
    if not args.execute:
        profile = plan['profiles'][args.profile]
        print(json.dumps({'plan_sha256': plan['sha256'], 'status': 'plan_only', 'profile': args.profile, 'stage': args.stage,
            'model': profile['model'], 'stages': list(STAGES), 'corpus_rows': plan['corpus_selection']['expected_rows'],
            'preview': dry_run(profile['protocol'], 'smoke', order_seed=plan['order_seed']), 'limitations': plan['limitations']}, indent=2))
        return 0
    if not args.stage or not args.output:
        parser.error('--execute requires --stage and a new --output directory')
    result = run_stage(plan, args.profile, args.stage, args.output, LocalClient(args.url),
        calibration=args.calibration, source_run=args.source_run, timeout=args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
