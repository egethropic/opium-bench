"""Review-only research scoring and explicit recipe service boundaries."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lab.scoring import RUBRIC, digest, score_blinded_sheet
from lab.service import normalize_config
from lab.storage import atomic_json
from lab.resources import ResourceStop
from test_lab_service import RecordingService


class ResearchServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.service = RecordingService(self.root)
        self.sample = dict(blind_id='blind-a', prompt='Describe a person.', continuation='A person looked pleased.')
        self.sheet = dict(rubric=deepcopy(RUBRIC), rubric_sha256=digest(RUBRIC), samples=[dict(self.sample, ratings={key: None for key in RUBRIC['fields']})])
        self.key = dict(rubric_sha256=digest(RUBRIC), records=[dict(blind_id='blind-a', content_sha256=digest(self.sample), condition='sham', split='heldout')])
        self.directory = self.service.store.calibrations / 'cal-research'
        atomic_json(self.directory / 'scoring-key.json', self.key)
        raw = (self.directory / 'scoring-key.json').read_bytes()
        atomic_json(self.directory / 'calibration.json', dict(schema_version=2, evidence_sha256={'scoring-key.json': hashlib.sha256(raw).hexdigest()}))
        self.source = {p.name: p.read_bytes() for p in self.directory.iterdir()}

    def tearDown(self):
        self.tmp.cleanup()

    def score(self, **kwargs):
        return self.service.command('score_calibration', dict(calibration_id='cal-research', sheet=deepcopy(self.sheet), scorer='Rater A', **kwargs))

    def test_review_only_persists_ratings_and_preserves_source(self):
        self.assertIsNone(self.service.worker['model'])
        result = self.score()
        saved = json.loads(self.service.rating_path(result['rating_id']).read_text())
        self.assertEqual(saved['scores']['conditions']['sham']['fields']['coherence']['missing'], 1)
        self.assertEqual(self.service.state()['ratings'][0]['id'], result['rating_id'])
        self.assertEqual(self.source, {p.name: p.read_bytes() for p in self.directory.iterdir()})
        self.assertEqual(self.service.sent, [])

    def test_tampered_key_is_refused(self):
        atomic_json(self.directory / 'scoring-key.json', dict(self.key, rubric_sha256='changed'))
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.score()
        self.assertEqual(self.service.rating_catalog(), [])

    def test_malformed_or_edited_scoring_never_persists(self):
        for replacement in (None, [], {'samples': None}, dict(self.sheet, samples=[None]), dict(self.sheet, samples=[])):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                score_blinded_sheet(replacement, self.key)
        for change in ({'prompt':'changed'}, {'ratings':{}}):
            sheet = deepcopy(self.sheet); sheet['samples'][0].update(change)
            with self.assertRaises(ValueError):
                self.service.command('score_calibration', dict(calibration_id='cal-research', sheet=sheet))
        self.assertEqual(self.service.rating_catalog(), [])

    def test_reserve_failure_leaves_no_published_or_partial_rating(self):
        with patch.object(self.service.resources, 'write', side_effect=ResourceStop('reserve breached')):
            with self.assertRaises(ResourceStop): self.score()
        self.assertEqual(list((self.service.store.root/'ratings').iterdir()), [])

    def test_recipe_preview_without_model_and_exact_preview_gate(self):
        result = self.service.command('preview_recipe', dict(config={'recipe_version':2}))
        self.assertEqual(result['preview']['config']['recipe_version'], 2)
        self.assertFalse(result['exact_template'])
        self.assertTrue(result['preview']['messages'])
        with self.assertRaisesRegex(ValueError, 'Load a model'):
            self.service.command('preview_session', dict(config={'recipe_version':2}))

    def test_explicit_version_one_matches_legacy_and_unknown_versions_reject(self):
        self.assertEqual(normalize_config({}), normalize_config({'recipe_version':1}))
        for version in (True, 0, 3, '2'):
            with self.assertRaises(ValueError): normalize_config({'recipe_version':version})

    def test_validation_creates_new_destination_and_strips_untrusted_path(self):
        self.service.worker.update(model={'model_id':'fake'}, status='ready')
        with patch('lab.service.preflight'):
            self.service.command('validate_calibration', dict(calibration_id='cal-research', doses=[0,.2]))
        _, job = self.service.sent[-1]
        self.assertNotEqual(Path(job['out_dir']), self.directory)
        self.assertEqual(job['config']['calibration_dir'], str(self.directory))
        with self.assertRaises(ValueError):
            self.service.command('validate_calibration', dict(calibration_id='cal-research', calibration_dir='/untrusted'))

    def test_frozen_protocol_dispatches_full_runner_and_review_works_without_model(self):
        (self.directory/'vectors.npz').write_bytes(b'fixture-vector-evidence')
        self.service.worker.update(model={'model_id':'fake', 'fingerprint_sha256':'a'*64}, status='ready')
        preview=self.service.command('preview_protocol',{'protocol_id':'task_pressure','mode':'smoke'})['preview']
        # The recording service allocates tiny receipts but never starts the
        # advertised GPU workloads; reserve enforcement has separate fault tests.
        with patch.object(self.service.resources,'preflight'):
            result=self.service.command('start_protocol',dict(protocol_id='task_pressure',mode='smoke',
                expansion_sha256=preview['expansion_sha256'],calibration_id='cal-research'))
        name,payload=self.service.sent[-1]
        self.assertEqual(name,'run_research_job')
        self.assertEqual(payload['expected_expansion_sha256'],preview['expansion_sha256'])
        self.assertFalse(payload['resume']);self.assertFalse(payload['retry_failed'])
        self.assertEqual(len(payload['entries']),len(preview['episodes']))
        from lab.research_jobs import read_job
        job=read_job(self.service.store,result['research_job_id'])
        self.assertEqual(job['receipt']['model_info']['model_id'],'fake')
        self.assertEqual(job['planned'],len(preview['episodes']))
        self.service.worker.update(model=None,status='unloaded')
        analysis=self.service.command('analyze_protocol',dict(research_job_id=result['research_job_id']))['analysis']
        self.assertEqual(analysis['planned_episodes'],len(preview['episodes']))
        self.assertEqual(analysis['observed_task_outcomes'],0)
        self.assertTrue(all(row['outcomes']['task_accuracy'] is None for row in analysis['records']))

    def test_unresolved_stages_and_changed_hash_fail_before_run_allocation(self):
        (self.directory/'vectors.npz').write_bytes(b'fixture-vector-evidence')
        self.service.worker.update(model={'model_id':'fake'},status='ready')
        preview=self.service.command('preview_protocol',{'protocol_id':'discovery_reversal','mode':'smoke'})['preview']
        payload=dict(protocol_id='discovery_reversal',mode='smoke',calibration_id='cal-research',expansion_sha256=preview['expansion_sha256'])
        with self.assertRaisesRegex(ValueError,'bindings'):self.service.command('start_protocol',payload)
        with self.assertRaisesRegex(ValueError,'expansion hash'):self.service.command('start_protocol',dict(payload,expansion_sha256='changed'))
        self.assertEqual(self.service.store.catalog(),[])

    def test_retry_registration_is_owned_before_any_streamed_evidence(self):
        identifier,_=self.service.store.create('experiment',{'recipe_version':2})
        self.service.event(dict(type='research_run_registered',run_id=identifier,command_id='cmd-research'))
        self.assertIn(identifier,self.service.command_runs['cmd-research'])
        self.assertEqual(self.service.store.read_run(identifier)['events'],[])
        self.service.event(dict(type='research_progress',run_id=identifier,command_id='cmd-research',stage='main',status='running'))
        self.assertEqual(self.service.store.read_run(identifier)['events'],[])


if __name__ == '__main__': unittest.main()
