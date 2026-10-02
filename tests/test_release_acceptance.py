"""Acceptance-driver CPU fixtures: no HTTP listener, model, download or GPU."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from lab.effects import content_hash
from lab.protocol_library import dry_run
from lab.resources import ResourceGuard, Volume
from lab.runtime_controls import validate_runtime_controls
from run_release_acceptance import Acceptance, Evidence, LocalClient, ROOT, load_plan, run_stage, selected_corpus


class FakeClient:
    def __init__(self, plan, profile='4b'):
        self.plan = plan
        model = deepcopy(plan['profiles'][profile]['model'])
        self.state = {'worker': {'status': 'ready', 'model': {'model_id': model['model_id'], 'fingerprint': model}},
            'session': {}, 'job': {}, 'calibrations': [{'id': 'cal-source'}], 'storage': {}}
        self.calls, self.runs, self.commands = [], {}, 0
        self.fail = None
        self.protocol_result = {'settled': True, 'all_complete': True, 'latest_states': [{'run_id': 'run-protocol'}], 'receipt': {'status': 'complete'}}

    def get(self, path):
        if path == '/api/state': return deepcopy(self.state)
        if path.startswith('/api/calibrations/'): return {'artifact': path}
        if path.startswith('/api/research/'): return deepcopy(self.protocol_result)
        if path.startswith('/api/runs/'):
            parts = path.split('/')
            if len(parts) == 5 and parts[-1] == 'checkpoints':
                return {'checkpoints': [{'id': 'turn-1.json.gz', 'turns': 1, 'event_cutoff': 10}]}
            return deepcopy(self.runs[parts[3]])
        raise AssertionError(path)

    def command(self, name, payload):
        self.calls.append((name, deepcopy(payload)))
        if self.fail == name: raise RuntimeError('fixture command failure')
        if name == 'chat':
            if set(payload) != {'text'} or not isinstance(payload['text'], str) or not payload['text'].strip():
                raise ValueError('Message must contain 1–24,000 characters using the service text field')
        self.commands += 1
        identifier = 'cmd-' + str(self.commands)
        self.state['job'] = {'command_id': identifier, 'status': 'complete'}
        accepted = {'accepted': True, 'command_id': identifier}
        if name in {'calibrate', 'validate_calibration'}:
            self.state['calibrations'].append({'id': 'cal-output'})
        if name == 'start_protocol': accepted['research_job_id'] = 'research-fixture'
        if name == 'analyze_protocol': return {'accepted': True, 'analysis': {'planned': 14}}
        if name == 'start_session':
            run_id = 'run-' + str(self.commands)
            self.runs[run_id] = {'manifest': {'id': run_id, 'config': deepcopy(payload['config']), 'status': 'awaiting_user'}, 'events': [], 'summary': {}}
            self.state['session'] = {'id': run_id, 'status': 'awaiting_user'}
            accepted['run_id'] = run_id
        if name == 'pause': self.state['session']['status'] = 'paused'
        if name == 'resume': self.state['session']['status'] = 'awaiting_user'
        if name == 'stop':
            self.state['session']['status'] = 'stopped'
            self.runs[self.state['session']['id']]['manifest']['status'] = 'stopped'
        if name == 'branch':
            run_id = 'run-' + str(self.commands)
            self.runs[run_id] = deepcopy(self.runs[payload['run_id']])
            self.runs[run_id]['manifest'].update(id=run_id, status='awaiting_user')
            self.state['session'] = {'id': run_id, 'status': 'awaiting_user'}
            accepted['run_id'] = run_id
        if name == 'run_diagnostics':
            run_id = 'diagnostic-fixture'
            self.runs[run_id] = {'manifest': {'id': run_id, 'status': 'complete'}, 'events': [], 'summary': {'interpretation_eligible': False}}
            accepted['run_id'] = run_id
        return accepted


class AcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.plan = load_plan()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.guard = ResourceGuard({'output': self.root}, reserve_bytes=1024, emergency_bytes=1024,
            forbid_large_c_writes=False, resolver=lambda _: [Volume('fixture', str(self.root))],
            telemetry=lambda _: {'free_bytes': 2**30, 'total_bytes': 2**31})

    def tearDown(self): self.tmp.cleanup()

    def run_stage(self, stage, client=None, **kwargs):
        return run_stage(self.plan, '4b', stage, self.root/'stage', client or FakeClient(self.plan), guard=self.guard, **kwargs)

    def test_plan_preserves_pairs_splits_and_frozen_expansions(self):
        corpus = selected_corpus(ROOT/'corpora/research-v2.json', self.plan['corpus_selection'])
        self.assertEqual(len(corpus['rows']), 160)
        groups = {}
        for row in corpus['rows']: groups.setdefault((row['split'], row['context'], row['concept']), []).append(row)
        self.assertEqual(len(groups), 40)
        self.assertTrue(all(len(rows) == 4 and sum(r['label'] for r in rows) == 2 for rows in groups.values()))
        for profile, count in [('4b', 14), ('27b', 2)]:
            preview = dry_run(self.plan['profiles'][profile]['protocol'], 'smoke', order_seed=self.plan['order_seed'])
            self.assertEqual(len(preview['episodes']), count)
            self.assertEqual(preview['expansion_sha256'], self.plan['profiles'][profile]['expansion_sha256'])
            for episode in preview['episodes']:
                validate_runtime_controls(episode['runtime_controls'], recipe=episode['recipe'])
                for tool in episode['recipe']['auxiliary_tools']:
                    self.assertEqual(tool['parameters']['required'], ['mode'])
                    self.assertEqual(tool['cost'], 2)

    def test_altered_plan_and_corpus_fail_closed(self):
        edited = deepcopy(self.plan)
        edited['validation']['continuation_tokens'] += 1
        target = self.root/'altered.json'; target.write_text(json.dumps(edited))
        with self.assertRaisesRegex(ValueError, 'altered'): load_plan(target)
        source = self.root/'corpus.json'; source.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'bytes changed'): selected_corpus(source, self.plan['corpus_selection'])

    def test_extraction_archives_exact_runner_and_has_no_diagnostic_fetch(self):
        client = FakeClient(self.plan)
        result = self.run_stage('extract', client)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['details']['calibration_id'], 'cal-output')
        self.assertEqual((self.root/'stage/runner.py').read_bytes(), (ROOT/'run_release_acceptance.py').read_bytes())
        self.assertTrue((self.root/'stage/progress.json').exists())
        self.assertFalse((self.root/'stage/diagnostics.json').exists())
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0][1]['corpus']['rows']), 160)

    def test_validation_exports_unrated_sheet_and_keeps_source_id(self):
        result = self.run_stage('validate', calibration='cal-source')
        self.assertEqual(result['status'], 'complete')
        self.assertTrue((self.root/'stage/scoring-sheet.json').exists())
        self.assertEqual(result['details']['independent_semantic_ratings'], 'not_performed')
        self.assertEqual(result['details']['source_calibration_id'], 'cal-source')

    def test_mismatched_model_stops_before_any_command_and_retains_failure(self):
        client = FakeClient(self.plan, '27b')
        result = self.run_stage('extract', client)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(client.calls, [])
        self.assertTrue((self.root/'stage/result.json').exists())

    def test_complete_protocol_freezes_all_14_cases_without_retries(self):
        client = FakeClient(self.plan)
        result = self.run_stage('protocol', client, calibration='cal-source')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['details']['planned_episodes'], 14)
        self.assertEqual([name for name, _ in client.calls], ['start_protocol', 'analyze_protocol'])
        self.assertEqual(client.calls[0][1]['expansion_sha256'], self.plan['profiles']['4b']['expansion_sha256'])

    def test_partial_protocol_stays_failed_with_complete_denominator_receipt(self):
        client = FakeClient(self.plan)
        client.protocol_result.update(all_complete=False, status_counts={'failed': 1, 'complete': 13})
        result = self.run_stage('protocol', client, calibration='cal-source')
        self.assertEqual(result['status'], 'failed')
        saved = json.loads((self.root/'stage/research-receipt.json').read_text())
        self.assertEqual(saved['status_counts']['failed'], 1)
        self.assertNotIn('resume_protocol', [name for name, _ in client.calls])

    def test_lifecycle_pauses_resumes_and_branches_independently(self):
        client = FakeClient(self.plan)
        result = self.run_stage('lifecycle', client, calibration='cal-source')
        self.assertEqual(result['status'], 'complete', result.get('error'))
        self.assertTrue(result['details']['source_unchanged'])
        branches = [payload for name, payload in client.calls if name == 'branch']
        self.assertEqual([b['policy'] for b in branches], ['continue_state', 'fresh_budget'])
        self.assertNotIn('token_budget', branches[0])
        self.assertEqual(branches[1]['token_budget'], 256)
        config = next(payload['config'] for name, payload in client.calls if name == 'start_session')
        self.assertEqual(config['task_tool_costs'], {})

    def test_diagnostic_requires_real_sham_source_and_no_parent_mutation(self):
        client = FakeClient(self.plan)
        episode = next(e for e in dry_run(self.plan['profiles']['4b']['protocol'], 'smoke')['episodes'] if e['factors']['case'] == 'sham-tokens')
        client.runs['source'] = {'manifest': {'id': 'source', 'config': episode['recipe'], 'status': 'complete'}, 'events': [], 'summary': {}}
        before = deepcopy(client.runs['source'])
        result = self.run_stage('diagnostic', client, calibration='cal-source', source_run='source')
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(result['details']['interpretation_eligible'])
        self.assertEqual(client.runs['source'], before)
        payload = client.calls[0][1]
        self.assertEqual(payload['spec']['completed_boundaries'], [1])
        self.assertEqual(payload['effect_policy'], 'sham')

    def test_yoke_uses_managed_source_binding_exact_presets_and_declared_clock(self):
        client = FakeClient(self.plan)
        episode = next(e for e in dry_run(self.plan['profiles']['4b']['protocol'], 'smoke')['episodes'] if e['factors']['case'] == 'active-decisions')
        client.runs['source'] = {'manifest': {'id': 'source', 'config': episode['recipe'], 'status': 'complete', 'research_episode': {'factors': episode['factors']}}, 'events': [], 'summary': {}}
        result = self.run_stage('yoke', client, calibration='cal-source', source_run='source')
        self.assertEqual(result['status'], 'complete', result.get('error'))
        payload = client.calls[0][1]
        self.assertEqual(payload['bindings'], {'source_run_id': 'source'})
        document = payload['document']
        self.assertEqual(document['base_recipe']['effect_presets'], episode['recipe']['effect_presets'])
        self.assertEqual(document['stages'][0]['config']['clock'], 'decisions')
        self.assertTrue(document['stages'][0]['config']['suppress_own_aux_effects'])

    def test_output_directory_cannot_be_reused(self):
        Evidence(self.root/'new', self.guard).write('data.json', {'original': True})
        with self.assertRaises(FileExistsError): Evidence(self.root/'new', self.guard)
        self.assertEqual(json.loads((self.root/'new/data.json').read_text()), {'original': True})

    def test_resource_failure_after_dispatch_cancels_only_owned_work_and_keeps_emergency_receipt(self):
        client = FakeClient(self.plan)
        original = client.command
        def dispatch(name, payload):
            response = original(name, payload)
            if name == 'calibrate':
                client.state['job']['status'] = 'running'
                client.state['session'] = {'id': 'run-owned', 'status': 'running'}
                client.runs['run-owned'] = {'manifest': {'status': 'running'}}
                self.guard.telemetry = lambda _: {'free_bytes': 0, 'total_bytes': 2**31}
                self.guard.last_check = float('-inf')
            return response
        client.command = dispatch
        result = self.run_stage('extract', client)
        self.assertEqual(result['status'], 'resource_stopped')
        self.assertTrue(result['owned_work_stopped'])
        self.assertEqual([name for name, _ in client.calls], ['calibrate', 'stop'])
        self.assertTrue(Path(result['emergency_record']).is_file())
        self.assertEqual(list((self.root/'stage').glob('.emergency-*')), [])

    def test_portable_roundtrip_without_a_model_preserves_original_bytes(self):
        from lab.portability import export_bundle
        directory = self.root/'original'/'run-source'; directory.mkdir(parents=True)
        contents = {'manifest.json': b'{"id":"run-source","format_version":2,"status":"complete"}\n',
            'events.jsonl': b'{"type":"message","content":"original"}\n', 'summary.json': b'{"correct":1}\n'}
        for name, raw in contents.items(): (directory/name).write_bytes(raw)
        archive = self.root/'fixture.zip'; export_bundle(directory, archive)
        client = FakeClient(self.plan); client.state['worker']['model'] = None
        client.runs['run-source'] = {'manifest': json.loads(contents['manifest.json'])}
        client.download = lambda path, maximum: archive.read_bytes()
        result = self.run_stage('portable', client, source_run='run-source')
        self.assertEqual(result['status'], 'complete', result.get('error'))
        target = Path(result['details']['import']['path'])
        for name, raw in contents.items(): self.assertEqual((target/name).read_bytes(), raw)
        self.assertFalse(result['details']['resume_attempted'])
        self.assertEqual(client.calls, [])

    def test_remote_targets_and_credentials_are_rejected_without_network(self):
        for url in ('https://example.com', 'http://example.com', 'http://user:pass@localhost:8766'):
            with self.assertRaises(ValueError): LocalClient(url)
        self.assertEqual(LocalClient('http://127.0.0.1:8766').url, 'http://127.0.0.1:8766')


if __name__ == '__main__': unittest.main()
