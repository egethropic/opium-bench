#!/usr/bin/env python3
"""Supplemental first-run verification; default is an offline plan, never inference.

Run against a clean, separately created local clone. --execute owns two temporary
local services but keeps all their data/evidence. Existing dependencies and model
cache are reused, not installed or redownloaded. No automatic job retries.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

sys.dont_write_bytecode = True
CASES = ('active-tokens', 'sham-tokens')
STEPS = ('runtime_check', 'load', 'extract', 'pair', 'http_export', 'http_import_replay')
MAX_BUNDLE = 128 * 1024**2


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.STDOUT).decode().strip()


def modules(repo):
    sys.path.insert(0, str(repo))
    return importlib.import_module('run_release_acceptance')


def build_plan(repo, api):
    source = api.load_plan(repo / 'protocols/acceptance/release-v1.json')
    document = deepcopy(source['profiles']['4b']['protocol'])
    if len(document['axes']) != 1 or document['axes'][0]['name'] != 'case':
        raise ValueError('Frozen source protocol no longer has the expected case axis')
    levels = {row['id']: row for row in document['axes'][0]['levels']}
    document.update(id='fresh-clone-4b-v1', title='Fresh-clone first-run active/sham engineering check',
        question='Can a clean source clone load existing weights, extract a new calibration, run two bounded cases and round-trip their evidence?',
        axes=[{'name': 'case', 'levels': [levels[name] for name in CASES]}], smoke_axes={'case': list(CASES)},
        controls=['Active versus sham only, identical two-task and 768-token budgets',
                  'No reward for auxiliary calls; demonstrations are not voluntary choices',
                  'All planned cases and failures retained; no automatic retries'],
        limitations=['One seed, two cases, fresh unvalidated extraction; engineering workflow evidence only.',
                     'The .25 intervention is inherited from the frozen engineering plan, not a selected safe or effective semantic dose.',
                     'No independent continuation scores, powered behavioral claim, sensation claim, or full installation validation.'])
    document = api.freeze_protocol(document)
    preview = api.dry_run(document, 'smoke', order_seed=source['order_seed'])
    original = {row['factors']['case']: row for row in api.dry_run(source['profiles']['4b']['protocol'], 'smoke', order_seed=source['order_seed'])['episodes']}
    if len(preview['episodes']) != 2:
        raise ValueError('Expected exactly two preregistered cases')
    for episode in preview['episodes']:
        before = original[episode['factors']['case']]
        for key in ('recipe', 'task_config', 'runtime_controls', 'stages'):
            if episode[key] != before[key]:
                raise ValueError('Derived first-run case changed frozen source field: ' + key)
        recipe = episode['recipe']
        if recipe['thinking'] or recipe['task_count'] != 2 or recipe['token_budget'] != 768:
            raise ValueError('First-run scope must remain direct, two tasks and 768 tokens')
    corpus = api.selected_corpus(repo / 'corpora/research-v2.json', source['corpus_selection'])
    if len(corpus['rows']) != 160:
        raise ValueError('First-run extraction must use the exact frozen 160 rows')
    dirty = git(repo, 'status', '--porcelain', '--untracked-files=all')
    source_files = {}
    for name in git(repo, 'ls-files').splitlines():
        if name.endswith(('.py', '.txt')) or name in ('corpora/research-v2.json', 'protocols/acceptance/release-v1.json'):
            source_files[name] = api.sha((repo / name).read_bytes())
    plan = dict(schema_version=1, kind='opium-bench-fresh-clone-verification', version=1,
        source=dict(repository=str(repo), commit=git(repo, 'rev-parse', 'HEAD'), tree=git(repo, 'rev-parse', 'HEAD^{tree}'),
                    dirty=bool(dirty), status_porcelain=dirty, tracked_source_sha256=source_files),
        source_acceptance_sha256=source['sha256'], corpus_selection=source['corpus_selection'],
        extraction=source['extraction'], model=source['profiles']['4b']['model'], protocol=document,
        preview=preview, order_seed=source['order_seed'], export_case='sham-tokens',
        bounds=dict(corpus_rows=160, planned_cases=2, tasks_per_case=2, total_generated_token_cap=1536,
                    max_http_bundle_bytes=MAX_BUNDLE, retries=0, downloads=False, installs=False),
        inclusion='Every step is complete, failed, resource_stopped, cancelled or not_attempted; every planned episode remains in its raw receipt. Export uses the predeclared sham case only.',
        limitations=document['limitations'] + ['Fresh data and cache namespaces reuse existing pinned cached weights and the existing 4B environment; this is not a clean dependency installation.'])
    plan['sha256'] = api.content_hash(plan)
    return source, plan


class OwnedServer:
    def __init__(self, repo, python, data, cache, port, evidence, api, label):
        self.repo, self.data, self.cache, self.evidence, self.api = repo, data, cache, evidence, api
        self.client = api.LocalClient(f'http://127.0.0.1:{port}')
        self.process = None
        self.log_name = label + '-server.log'
        self.log_bytes, self.log_dropped, self.log_error = 0, 0, None
        self.command = [sys.executable, '-B', str(repo / 'launch_lab.py'), '--data-dir', str(data), '--cache-dir', str(cache), '--python', python, '--port', str(port)]
        self.port = port

    def start(self):
        # Admission only, not a way to take over another service's port.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', self.port))
        self.data.mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_HOME=str(self.cache), OPIUM_DATA_DIR=str(self.data), TMPDIR=str(self.data / 'tmp'),
                   TEMP=str(self.data / 'tmp'), TMP=str(self.data / 'tmp'))
        self.process = subprocess.Popen(self.command, cwd=self.repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        self.reader = threading.Thread(target=self._drain, daemon=True)
        self.reader.start()
        deadline = time.monotonic() + 45
        while True:
            self.evidence.guard.check(force=True)
            if self.process.poll() is not None:
                raise RuntimeError('Owned service exited before readiness: ' + self.log_name)
            try:
                state = self.client.get('/api/state')
                if state['storage']['data']['path'] != str(self.data.resolve()):
                    raise RuntimeError('Port returned another service; refusing to send commands')
                if state['worker'].get('model') or state['calibrations']:
                    raise RuntimeError('Expected new empty data and an unloaded worker')
                return state
            except URLError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Owned service did not become ready; no restart attempted')
                time.sleep(.25)

    def _drain(self):
        try:
            while block := self.process.stdout.read(4096):
                retained = block[:max(0, 8*1024**2 - self.log_bytes)]
                self.log_dropped += len(block) - len(retained)
                if retained:
                    self.api.guarded_bytes(self.evidence.directory / self.log_name, retained, mode='ab', guard=self.evidence.guard)
                    self.log_bytes += len(retained)
        except Exception as error:
            self.log_error = str(error)
            # Do not leave the child blocked on a full stdout pipe.
            while self.process.stdout.read(4096):
                pass

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGINT)  # launcher finally closes its worker
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    from lab.resources import cancel_owned_process
                    cancel_owned_process(self.process, process_group=True)
            self.reader.join(timeout=5)
        return dict(command=self.command, exit_code=None if self.process is None else self.process.returncode,
                    log_bytes=self.log_bytes, omitted_log_bytes=self.log_dropped, log_error=self.log_error)


def import_and_verify(client, raw, source, evidence):
    state = client.get('/api/state')
    if state['worker'].get('model') is not None:
        raise ValueError('Replay service must remain unloaded')
    request = Request(client.url + '/api/import', data=raw, headers={'Content-Type': 'application/zip', 'X-CSRF-Token': state['csrf']})
    with urlopen(request, timeout=120) as response:
        imported = json.load(response)
    evidence.write('http-import.json', imported)
    if imported.get('partial_events') or not imported.get('replay_only') or imported.get('resume_eligible'):
        raise ValueError('Expected a complete replay-only import')
    replay = client.get('/api/runs/' + quote(imported['id'], safe=''))
    evidence.write('replay.json', replay)
    keys = ('manifest', 'summary', 'conversation', 'events', 'parent_events')
    comparisons = {key: source.get(key, []) == replay.get(key, []) for key in keys}
    evidence.write('replay-comparison.json', comparisons)
    if not all(comparisons.values()):
        raise ValueError('Imported replay differs from source scientific evidence')
    html = client.download('/api/runs/' + quote(imported['id'], safe='') + '/report', maximum=4*1024**2)
    evidence.write('replay-report.html', html, raw=True)
    final = client.get('/api/state')
    evidence.write('replay-state.json', final)
    if final['worker'].get('model') is not None or final['worker'].get('status') != 'unloaded':
        raise ValueError('Replay unexpectedly started or loaded a worker')
    return dict(run_id=imported['id'], compared_fields=keys, report_bytes=len(html), replay_only=True, worker_unloaded=True)


def execute(args, source, plan, api):
    if plan['source']['dirty']:
        raise ValueError('Execution requires a clean clone; commit/copy no files automatically')
    if not (args.repo / '.git').is_dir():
        raise ValueError('Use a separately created local clone, not an attached worktree')
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(args.repo):
        raise ValueError('Output must be new and outside the source clone')
    hub = args.cache_source.resolve() / 'hub'
    snapshot = hub / 'models--Qwen--Qwen3-4B' / 'snapshots' / plan['model']['revision']
    if not (snapshot / 'config.json').is_file():
        raise ValueError('Pinned cached Qwen3-4B snapshot is absent; downloads are forbidden')
    if not Path(args.python).is_file():
        raise ValueError('Existing explicit worker interpreter is missing')
    guard = api.ResourceGuard({'verification': output, 'cache_source': hub})
    guard.preflight({output: 1024**3})
    evidence = api.Evidence(output, guard=guard)
    result = dict(kind=plan['kind'], schema_version=1, plan_sha256=plan['sha256'], status='running',
                  started_at=api.utc_now(), steps={key: {'status': 'not_attempted'} for key in STEPS})
    servers, runner = [], None
    current = None
    try:
        evidence.write('plan.json', plan)
        evidence.write('started.json', result)
        evidence.write('source-acceptance-plan.json', source)
        evidence.write('verification-runner.py', Path(__file__).read_bytes(), raw=True)
        evidence.write('invocation.json', dict(repo=str(args.repo), output=str(output), python=args.python,
            cache_source=str(args.cache_source.resolve()), source_port=args.port, replay_port=args.port+1,
            timeout_seconds=args.timeout, execute=True, no_downloads=True, reused_environment=True))
        caches = [output / 'source-cache', output / 'replay-cache']
        for cache in caches:
            cache.mkdir(); (cache / 'hub').symlink_to(hub, target_is_directory=True)
        evidence.write('cache-layout.json', {'mode': 'new_cache_namespaces_existing_hub_symlinks', 'cache_roots': list(map(str, caches)), 'actual_model_hub': str(hub)})
        current = 'runtime_check'
        check = subprocess.run([sys.executable, '-B', str(args.repo / 'check_runtime.py'), '--python', args.python,
            '--profile', '4b', '--data-dir', str(output / 'source-data'), '--cache-dir', str(caches[0]), '--check'],
            cwd=args.repo, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, timeout=90)
        evidence.write('runtime-check-stdout.json', check.stdout, raw=True)
        evidence.write('runtime-check-stderr.txt', check.stderr, raw=True)
        if check.returncode:
            raise RuntimeError('Runtime metadata/reserve check failed; inference not started')
        result['steps'][current] = {'status': 'complete', 'scope': 'metadata/reserve only; no CUDA import requested'}
        current = 'load'
        server = OwnedServer(args.repo, args.python, output/'source-data', caches[0], args.port, evidence, api, 'source')
        servers.append(server); server.start()
        # Guard the whole duration, not just writes. This changes no task behavior.
        class GuardedAcceptance(api.Acceptance):
            def poll(self, getter, done, *, timeout=None):
                def checked():
                    guard.check(force=True)
                    return getter()
                return super().poll(checked, done, timeout=timeout)
        runner = GuardedAcceptance(source, '4b', server.client, evidence, timeout=args.timeout)
        loaded = runner.wait(runner.command('load_model', dict(profile_id='qwen3-4b', **plan['model'], allow_download=False, quantization='none', dtype='bfloat16')))
        runner.admit()
        evidence.write('loaded-model.json', loaded['worker']['model'])
        result['steps'][current] = {'status': 'complete', 'model': loaded['worker']['model']}
        current = 'extract'
        calibration = runner.calibrate()
        result['steps'][current] = {'status': 'complete', **calibration, 'corpus_rows': 160, 'semantic_validation': 'not_performed'}
        current = 'pair'
        pair = runner.protocol(calibration['calibration_id'], plan['protocol'])
        result['steps'][current] = {'status': 'complete', **pair}
        # Select by preregistered condition, never by correctness or effect magnitude.
        current = 'http_export'
        rows = [server.client.get('/api/runs/' + quote(identifier, safe='')) for identifier in pair['run_ids']]
        selected = [row for row in rows if row['manifest'].get('research_episode', {}).get('factors', {}).get('case') == plan['export_case']]
        if len(selected) != 1 or selected[0]['summary'].get('status') not in (None, 'complete'):
            raise ValueError('Predeclared completed sham export source not found')
        run = selected[0]; identifier = run['manifest']['id']
        evidence.write('export-source.json', run)
        raw = server.client.download('/api/runs/' + quote(identifier, safe='') + '/bundle', maximum=MAX_BUNDLE)
        evidence.write('source.zip', raw, raw=True)
        result['steps'][current] = {'status': 'complete', 'run_id': identifier, 'bytes': len(raw), 'sha256': api.sha(raw)}
        current = 'http_import_replay'
        replay = OwnedServer(args.repo, args.python, output/'replay-data', caches[1], args.port+1, evidence, api, 'replay')
        servers.append(replay); replay.start()
        result['steps'][current] = {'status': 'complete', **import_and_verify(replay.client, raw, run, evidence)}
        if git(args.repo, 'status', '--porcelain', '--untracked-files=all'):
            raise ValueError('Clone changed during verification; inspect before claiming a clean-source receipt')
        result['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as error:
        status = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'resource_stopped' if isinstance(error, api.ResourceStop) else 'failed'
        result.update(status=status, error={'type': type(error).__name__, 'message': str(error)[:4096]})
        if current:
            result['steps'][current] = {'status': status, 'error': result['error']}
        if runner:
            try: result['owned_work_stopped'] = runner.cancel_owned()
            except Exception as stop_error: result['stop_error'] = str(stop_error)[:4096]
    finally:
        result['services'] = []
        for server in reversed(servers):
            try: result['services'].append(server.close())
            except Exception as error:
                result['services'].append({'close_error': str(error)[:4096]})
                result['status'] = 'failed'
        result.update(finished_at=api.utc_now(), artifacts=deepcopy(evidence.files), interpretation=plan['limitations'])
        try:
            files = {}
            for path in sorted(output.rglob('*')):
                if path.is_file() and not path.is_symlink() and not path.name.startswith('.emergency-'):
                    h = hashlib.sha256()
                    with path.open('rb') as stream:
                        while block := stream.read(1024**2): h.update(block)
                    files[str(path.relative_to(output))] = {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
            evidence.write('artifact-index.json', files)
            evidence.write('result.json', result)
        except Exception as error:
            result.update(status='failed', finalization_error=str(error)[:4096])
            evidence.emergency.finalize(result)
        finally:
            evidence.emergency.release()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='Clean local clone; never edited by this script')
    parser.add_argument('--python', required=True, help='Existing 4B venv interpreter (symlink preserved)')
    parser.add_argument('--cache-source', type=Path, required=True, help='Existing HF cache root containing hub/')
    parser.add_argument('--output', type=Path, required=True, help='NEW external D-drive evidence/data directory')
    parser.add_argument('--port', type=int, default=8780, help='Uses this and next port; neither may already be bound')
    parser.add_argument('--timeout', type=float, default=1200, help='Per-command timeout, at most 3600 seconds')
    parser.add_argument('--execute', action='store_true', help='Explicitly allow local inference after freezing the plan')
    args = parser.parse_args(argv)
    args.repo = args.repo.resolve()
    if not 1024 <= args.port < 65535 or not 0 < args.timeout <= 3600:
        parser.error('Use a valid consecutive port pair and a timeout in (0,3600]')
    api = modules(args.repo)
    source, plan = build_plan(args.repo, api)
    if not args.execute:
        print(json.dumps({'status': 'plan_only', 'plan': plan}, indent=2, allow_nan=False)); return 0
    result = execute(args, source, plan, api)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
