"""Bounded acceptance publication from immutable local CPU fixtures."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from acceptance_publication import MAX_FILE, Collector, encode, publish, public_value, sha
from lab.effects import content_hash
from lab.resources import ResourceGuard, Volume
from lab.storage import Store

ROOT = Path(__file__).resolve().parents[1]


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.data = self.root/'data'; self.data.mkdir()
        self.plan_path = ROOT/'protocols/acceptance/release-v1.json'
        self.plan = json.loads(self.plan_path.read_text())
        self.guard = ResourceGuard({'publication': self.root}, reserve_bytes=1024, emergency_bytes=1024,
            forbid_large_c_writes=False, resolver=lambda _: [Volume('fixture', str(self.root))],
            telemetry=lambda _: {'free_bytes': 2**30, 'total_bytes': 2**31})

    def tearDown(self): self.temp.cleanup()

    def stage(self, name='attempt', *, stage='validate', status='complete', details=None, error=None, final=True):
        directory = self.root/name; directory.mkdir()
        runner = b'# archived fixture driver\n'
        started = dict(schema_version=1, kind='opium-bench-acceptance-stage', plan_sha256=self.plan['sha256'],
            profile='4b', stage=stage, started_at='2026-10-02T00:00:00Z', runner_sha256=sha(runner), automatic_retries=False)
        files = {'runner.py': runner, 'started.json': encode(started), 'plan.json': self.plan_path.read_bytes()}
        for filename, raw in files.items(): (directory/filename).write_bytes(raw)
        result = dict(started, status=status, finished_at='2026-10-02T00:01:00Z', details=details or {}, error=error,
            artifacts=[dict(file=filename, bytes=len(raw), sha256=sha(raw)) for filename, raw in files.items()])
        if final: (directory/'result.json').write_bytes(encode(result))
        return directory

    def calibration(self):
        directory = self.data/'calibrations/cal-fixture'; directory.mkdir(parents=True)
        vectors = b'numeric archive fixture\0bytes'
        meta = {'schema_version': 2, 'status': 'validated_for_enumerated_tests', 'layer': 12, 'pooling': 'mean',
            'selected_dose': 0., 'operating_range': {'eligible_doses': [0.], 'rule': {'max_relative_delta': .3}},
            'vectors_sha256': sha(vectors), 'model_fingerprint': {'model_id': 'Qwen/fixture', 'local_path': '/mnt/d/owner/model'},
            'continuation_scoring': {'status': 'awaiting_independent_human_ratings'}}
        (directory/'calibration.json').write_bytes(encode(meta)); (directory/'vectors.npz').write_bytes(vectors)
        sheet = {'samples': [{'blind_id': 'a', 'ratings': {'joy': None, 'coherence': None}}]}
        (directory/'scoring-sheet.json').write_bytes(encode(sheet))
        (directory/'scoring-key.json').write_bytes(encode({'private': 'observer answer'}))
        return directory

    def source_run(self):
        directory = self.data/'runs/run-fixture'; directory.mkdir(parents=True)
        parent = b'{"type":"message","content":"original parent"}\n'
        manifest = {'id': 'run-fixture', 'status': 'complete', 'mode': 'experiment', 'config': {'condition': 'active', 'thinking': False},
            'parent': {'parent_prefix_sha256': sha(parent)}, 'model': {'local_path': '/mnt/d/owner/model'}}
        files = {'manifest.json': encode(manifest), 'summary.json': encode({'correct': 1, 'assigned': 2, 'tokens': 4, 'voluntary_calls': 1}),
            'events.jsonl': b'{"type":"message","content":"/mnt/d/source/path remains scientifically bound"}\n',
            'conversation.json': b'[{"role":"assistant","content":"original"}]\n',
            'parent-events.jsonl.gz': gzip.compress(parent, mtime=0),
            'checkpoint.json': b'{"fixture":"identity-bound bytes /mnt/d/original"}\n'}
        for name, raw in files.items(): (directory/name).write_bytes(raw)
        return directory, files

    def publish(self, stages, output='publication'):
        return publish(stages, data_dir=self.data, output_dir=self.root/output, plan_path=self.plan_path, guard=self.guard)

    def test_complete_partial_missing_unscored_and_zero_dose_are_explicit(self):
        self.calibration()
        stages = [self.stage(details={'calibration_id': 'cal-fixture'}),
            self.stage('failed', stage='protocol', status='failed', error={'message': '<script>invalid</script>'})]
        result = self.publish(stages)
        report = json.loads((self.root/'publication/acceptance.json').read_text())
        self.assertEqual(result['attempts'], 2)
        self.assertTrue(any(r['status']=='missing' and r['profile']=='27b' for r in report['stage_matrix']))
        self.assertEqual(report['calibrations'][0]['unscored_samples'], 1)
        html = (self.root/'publication/index.html').read_text()
        self.assertIn('No tested nonzero dose met', html)
        self.assertIn('1 / 1 unscored', html)
        self.assertIn('&lt;script&gt;invalid&lt;/script&gt;', html)
        self.assertNotIn('<script>invalid</script>', html)
        self.assertNotIn('feature complete.</strong>', html)

    def test_scientific_bytes_and_parent_prefix_are_preserved_and_store_can_replay(self):
        source, files = self.source_run(); cal = self.calibration()
        stage = self.stage(stage='portable', details={'source_run_id':'run-fixture', 'calibration_id':'cal-fixture'})
        self.publish([stage]); output = self.root/'publication'
        for name, raw in files.items():
            target = output/'runs/run-fixture'/name
            if name == 'events.jsonl': self.assertEqual(gzip.decompress(target.with_suffix('.jsonl.gz').read_bytes()), raw)
            else: self.assertEqual(target.read_bytes(), raw)
            self.assertEqual((source/name).read_bytes(), raw)
        self.assertEqual((output/'calibrations/cal-fixture/vectors.npz').read_bytes(), (cal/'vectors.npz').read_bytes())
        self.assertEqual((output/'calibrations/cal-fixture/calibration.json').read_bytes(), (cal/'calibration.json').read_bytes())
        run = Store(self.root/'reader-data', historical=[output/'runs']).read_run('run-fixture')
        self.assertEqual(run['summary']['correct'], 1)
        self.assertEqual(run['events'][0]['content'], '/mnt/d/source/path remains scientifically bound')
        self.assertEqual(run['parent_events'][0]['content'], 'original parent')

    def test_administrative_redaction_is_distinct_and_private_key_is_omitted(self):
        self.calibration(); stage = self.stage(details={'calibration_id':'cal-fixture'})
        (stage/'extra.json').write_bytes(encode({'csrf': 'private-secret', 'output': '/mnt/d/owner/private'}))
        (stage/'admission.json').write_bytes(encode({'previous_session': 'unrelated text'}))
        self.publish([stage]); output = self.root/'publication'
        records = json.loads((output/'sha256-manifest.json').read_text())['files']
        admin = next(r for r in records if r.get('source','').endswith('/extra.json'))
        self.assertTrue(admin['metadata_redacted'])
        self.assertNotEqual(admin['source_sha256'], admin['expanded_sha256'])
        self.assertNotIn('private-secret', gzip.decompress((output/admin['file']).read_bytes()).decode())
        self.assertFalse(next(r for r in records if r.get('source','').endswith('/admission.json'))['published'])
        self.assertFalse((output/'calibrations/cal-fixture/scoring-key.json').exists())

    def test_all_attempts_retained_and_missing_final_is_not_passed(self):
        stages=[self.stage('first',stage='protocol',status='failed'),self.stage('second',stage='protocol'),self.stage('unfinished',stage='yoke',final=False)]
        self.publish(stages)
        report=json.loads((self.root/'publication/acceptance.json').read_text())
        row=next(r for r in report['stage_matrix'] if r['profile']=='4b' and r['stage']=='protocol')
        self.assertEqual(row['status'],'mixed_attempts'); self.assertEqual(row['attempts'],2)
        self.assertEqual(report['attempts'][2]['status'],'no_final_receipt')

    def test_supervisor_recovery_keeps_missing_driver_result_and_failed_job_identity(self):
        stage=self.stage(stage='protocol',final=False)
        snapshot={'receipt':{'id':'research-failed'},'status_counts':{'failed':14}}
        raw=encode(snapshot);(stage/'research-receipt.json').write_bytes(raw)
        recovery={'schema_version':1,'kind':'operator-supervised-acceptance-recovery','status':'failed',
            'research_job_id':'research-failed','research_receipt_sha256':sha(raw),'research_status_counts':{'failed':14},
            'primary_failure':'setup failed before tokens','driver_result':'No final driver receipt'}
        (stage/'supervisor-recovery.json').write_bytes(encode(recovery))
        self.publish([stage]);output=self.root/'publication'
        report=json.loads((output/'acceptance.json').read_text())
        self.assertEqual(report['attempts'][0]['status'],'no_final_receipt')
        self.assertFalse(report['attempts'][0]['final_driver_receipt_present'])
        self.assertEqual(report['attempts'][0]['supervisor_recovery']['research_status_counts'],{'failed':14})
        self.assertTrue(any(r['id']=='research-failed' for r in report['missing_evidence']))
        html=(output/'index.html').read_text();self.assertIn('Started receipt only',html);self.assertIn('setup failed before tokens',html)

    def test_supervisor_recovery_hash_mismatch_is_rejected(self):
        stage=self.stage(stage='protocol',final=False);(stage/'research-receipt.json').write_bytes(encode({'receipt':{'id':'research-failed'}}))
        (stage/'supervisor-recovery.json').write_bytes(encode({'kind':'operator-supervised-acceptance-recovery','research_receipt_sha256':'a'*64}))
        with self.assertRaisesRegex(ValueError,'Supervisor recovery'):self.publish([stage])

    def test_tampered_driver_artifact_rejected_without_partial_publication(self):
        stage=self.stage(); (stage/'runner.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'runner hash'):self.publish([stage])
        self.assertFalse((self.root/'publication').exists())
        self.assertEqual(list(self.root.glob('.publication-staging-*')),[])

    def test_duplicate_inputs_and_overwrite_are_rejected(self):
        stage=self.stage()
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.publish([stage,stage])
        self.publish([stage]); before=(self.root/'publication/index.html').read_bytes()
        with self.assertRaises(FileExistsError):self.publish([stage])
        self.assertEqual((self.root/'publication/index.html').read_bytes(),before)

    def test_vector_hash_mismatch_is_rejected(self):
        directory=self.calibration();(directory/'vectors.npz').write_bytes(b'wrong')
        stage=self.stage(details={'calibration_id':'cal-fixture'})
        with self.assertRaisesRegex(ValueError,'vector hash'):self.publish([stage])

    def test_real_schema_probe_float_specificity_and_sham_truncation_are_reported(self):
        directory=self.calibration();path=directory/'calibration.json';meta=json.loads(path.read_text())
        meta.update(supported_thinking_modes=[],extraction_policy={'generation_transfer':'unvalidated'},
            heldout={'sites':{'probe':{'pain':{'n':20,'auc':1.,'uncertainty':{'families':2,'resamples':20,'auc_95':[1.,1.]}},
                'cross_concept_score_correlation':.849316,'cross_concept_discrimination':{'pain':{'joy':{'auc':.98}}}}}})
        path.write_bytes(encode(meta))
        (directory/'continuations.json').write_bytes(encode([{'condition':'zero','split':'heldout','generation':{'token_ids':[1]*16,'finish_reason':'length'}}]))
        self.publish([self.stage(details={'calibration_id':'cal-fixture'})])
        row=json.loads((self.root/'publication/acceptance.json').read_text())['calibrations'][0]
        self.assertEqual(row['specificity']['probe']['cross_concept_score_correlation'],.849316)
        self.assertEqual(row['truncated_continuations'],1);self.assertEqual(row['continuations'][0]['tokens'],16)
        html=(self.root/'publication/index.html').read_text();self.assertIn('0.8493',html);self.assertIn('0.98',html)
        self.assertIn('Validated thinking modes:</strong> []',html)

    def test_missing_declared_evidence_and_run_files_remain_visible(self):
        directory=self.calibration();path=directory/'calibration.json';meta=json.loads(path.read_text())
        meta['evidence_sha256']={'diagnostics.json':'a'*64};path.write_bytes(encode(meta))
        run,_=self.source_run();(run/'summary.json').unlink();(run/'events.jsonl').unlink();(run/'checkpoint.json').unlink()
        self.publish([self.stage(details={'calibration_id':'cal-fixture','run_id':'run-fixture'})])
        report=json.loads((self.root/'publication/acceptance.json').read_text())
        missing={item.get('file') for item in report['missing_evidence']}
        self.assertTrue({'diagnostics.json','summary.json','events.jsonl or events.jsonl.gz'}<=missing)
        self.assertFalse(report['runs'][0]['checkpoint_bytes_preserved'])

    def test_declared_scoring_sheet_hash_is_checked(self):
        directory=self.calibration();path=directory/'calibration.json';meta=json.loads(path.read_text())
        meta['evidence_sha256']={'scoring-sheet.json':'a'*64};path.write_bytes(encode(meta))
        with self.assertRaisesRegex(ValueError,'Calibration evidence hash'):
            self.publish([self.stage(details={'calibration_id':'cal-fixture'})])

    def test_research_binding_outside_run_tree_is_archived_and_verified(self):
        rating=self.data/'ratings/criterion.json';rating.parent.mkdir();rating.write_bytes(b'{"endpoint":"fixture"}\n')
        directory=self.data/'research/research-fixture';directory.mkdir(parents=True)
        execution={'source_files':[{'path':'ratings/criterion.json','bytes':rating.stat().st_size,'sha256':sha(rating.read_bytes())}]}
        preview={'episodes':[]};preview['expansion_sha256']=content_hash(preview)
        receipt={'id':'research-fixture','status':'complete','planned':0,'entries':[],
            'plan_sha256':content_hash(execution),'expansion_sha256':preview['expansion_sha256']}
        receipt['receipt_sha256']=content_hash(receipt)
        for name,value in [('plan.json',execution),('preview.json',preview),('receipt.json',receipt)]:
            (directory/name).write_bytes(encode(value))
        self.publish([self.stage(stage='protocol',details={'research_job_id':'research-fixture'})])
        self.assertEqual((self.root/'publication/ratings/criterion.json').read_bytes(),rating.read_bytes())

    def test_scientific_symlinks_are_rejected(self):
        directory,files=self.source_run();(directory/'evil.json').symlink_to(self.plan_path)
        stage=self.stage(details={'run_id':'run-fixture'})
        with self.assertRaisesRegex(ValueError,'symlink'):self.publish([stage])

    def test_manifest_hashes_every_published_file_and_preserves_license_bytes(self):
        self.publish([self.stage()]);output=self.root/'publication'
        manifest=json.loads((output/'sha256-manifest.json').read_text());check=deepcopy(manifest);expected=check.pop('content_sha256')
        self.assertEqual(content_hash(check),expected)
        for record in manifest['files']:
            if 'file' in record:
                raw=(output/record['file']).read_bytes();self.assertEqual(sha(raw),record['sha256']);self.assertEqual(len(raw),record['bytes'])
        for name in ['LICENSE','NOTICE','UPSTREAM_LICENSE']:
            self.assertEqual((output/name).read_bytes(),(ROOT/name).read_bytes())


if __name__=='__main__':unittest.main()
