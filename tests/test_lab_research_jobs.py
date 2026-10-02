"""Frozen matrices produce reviewable receipts and fail closed on extra stages."""
from pathlib import Path
import tempfile
import unittest

from lab.effects import content_hash
from lab.protocol_library import dry_run,load_protocol
from lab.research_jobs import create_job,read_job,analysis_records
from lab.storage import Store,atomic_json


class ResearchJobsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.store=Store(self.tmp.name)
        cal=self.store.calibrations/'cal-test';atomic_json(cal/'calibration.json',{'schema_version':2});(cal/'vectors.npz').write_bytes(b'fixture')
    def preview(self): return dry_run(load_protocol('task_pressure'),'smoke',seeds=[17])
    def test_receipt_matches_resolved_tasks_and_analysis_denominators(self):
        preview=self.preview();receipt,entries=create_job(self.store,preview,'cal-test')
        self.assertEqual(len(entries),len(preview['episodes']))
        self.assertTrue(all('task_config' in entry for entry in entries))
        for entry in entries:
            summary={'correct':1,'assigned':2,'voluntary_calls':3,'completed_decisions':10,'tokens':32,'actions':13}
            self.store.append(entry['run_id'],{'type':'session_finished','status':'complete','summary':summary})
            self.store.update(entry['run_id'],status='complete',summary=summary)
        job=read_job(self.store,receipt['id'],verify=True)
        self.assertTrue(job['all_complete'])
        records=analysis_records(self.store,receipt['id'])
        self.assertEqual(records[0]['outcomes']['task_accuracy'],{'numerator':1,'denominator':2})
        self.assertEqual(records[0]['outcomes']['aux_per_decision'],{'numerator':3,'denominator':10})
    def test_recipe_tampering_and_missing_completion_evidence_are_rejected(self):
        receipt,entries=create_job(self.store,self.preview(),'cal-test');entry=entries[0]
        self.store.update(entry['run_id'],status='complete',summary={'correct':1})
        with self.assertRaisesRegex(ValueError,'final evidence'):read_job(self.store,receipt['id'],verify=True)
        self.store.update(entry['run_id'],config={})
        with self.assertRaisesRegex(ValueError,'frozen'):read_job(self.store,receipt['id'])
    def test_missing_capability_refuses_before_any_run_is_created(self):
        with self.assertRaisesRegex(ValueError,'capabilities'):
            create_job(self.store,dry_run(load_protocol('discovery_reversal')),'cal-test')
        self.assertEqual(list(self.store.runs.iterdir()),[])


if __name__=='__main__':unittest.main()
