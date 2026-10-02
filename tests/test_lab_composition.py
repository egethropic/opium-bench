"""Comprehensive views preserve source provenance and avoid duplicate controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from compose_lab_results import ROOT, build, compose_4b, episode_key
from lab.storage import atomic_json
from publish_lab_study import read_json


class CompositionTests(unittest.TestCase):
    def test_real_primary_selection_includes_pain_without_double_counting_controls(self):
        result=compose_4b(ROOT/'studies/initial',ROOT/'studies/core-pain-4b')
        self.assertEqual(result['recorded_episodes'],54)
        self.assertEqual(result['composition']['total_recorded_episodes'],70)
        self.assertEqual(result['composition']['verification_repeats'],16)
        self.assertEqual(result['composition']['exact_verification_repeats'],16)
        self.assertEqual(len({episode_key(row) for row in result['runs']}),54)
        self.assertEqual(sum(row['stage']=='core' and row['condition']=='pain' for row in result['runs']),8)
        self.assertEqual(result['totals']['assigned'],210)
        self.assertEqual(sum(group['episodes'] for group in result['groups']),54)
        for row in result['runs']:
            expected='studies/core-pain-4b' if row['stage']=='core' else 'studies/initial'
            self.assertEqual(result['run_sources'][row['run_id']],expected)

    def test_comprehensive_builder_preserves_every_source_checksum_and_uses_source_links(self):
        archives=[ROOT/'studies/initial',ROOT/'studies/core-pain-4b']
        before={str(path):path.read_bytes() for archive in archives for path in archive.rglob('*') if path.is_file()}
        with TemporaryDirectory() as name:
            root=Path(name)
            result=build(*archives,root/'composed',root/'results.html',skip_figures=True)
            html=(root/'results.html').read_text()
            self.assertIn('54 primary episodes',html)
            self.assertIn('70 total recorded',html)
            self.assertNotIn('Follow-up study',html)
            self.assertNotIn('COMPLETE FROZEN PILOT',html)
            self.assertIn('comprehensive condition set',html)
            self.assertIn('core-pain-4b/runs/',html)
            self.assertIn('initial/runs/',html)
            self.assertIn('<strong>pain</strong>',html)
            self.assertTrue((root/'composed/composition.json').exists())
            self.assertEqual(result['composition']['primary_episodes'],54)
        self.assertEqual(before,{str(path):path.read_bytes() for archive in archives for path in archive.rglob('*') if path.is_file()})

    def test_model_or_calibration_mismatch_cannot_be_presented_as_one_4b_configuration(self):
        old=read_json(ROOT/'studies/initial/results.json')
        fresh=read_json(ROOT/'studies/core-pain-4b/results.json')
        for key,field in (('model','fingerprint_sha256'),('calibration','vectors_sha256')):
            with self.subTest(key=key), TemporaryDirectory() as name:
                root=Path(name); left=root/'studies/initial';right=root/'studies/core'
                changed=deepcopy(fresh);changed[key][field]='different'
                for folder,value in ((left,old),(right,changed)):
                    atomic_json(folder/'results.json',value)
                    atomic_json(folder/'checksums.json',{'results.json':hashlib.sha256((folder/'results.json').read_bytes()).hexdigest()})
                with patch('compose_lab_results.ROOT',root),self.assertRaisesRegex(ValueError,'different'):
                    compose_4b(left,right)

    def test_replication_archive_is_protected_and_partial_data_cannot_claim_completion(self):
        with TemporaryDirectory() as name:
            root=Path(name);replication=root/'replication';output=root/'composed';page=root/'page.html'
            current=deepcopy(read_json(ROOT/'studies/core-pain-4b/results.json'))
            current['status']='partial'
            atomic_json(replication/'results.json',current)
            raw=(replication/'results.json').read_bytes()
            atomic_json(replication/'checksums.json',{'results.json':hashlib.sha256(raw).hexdigest()})
            sources=(ROOT/'studies/initial',ROOT/'studies/core-pain-4b')
            with self.assertRaisesRegex(ValueError,'immutable source'):
                build(*sources,replication,page,replication=replication,skip_figures=True)
            with self.assertRaisesRegex(ValueError,'immutable source'):
                build(*sources,output,replication/'results.json',replication=replication,skip_figures=True)
            with self.assertRaisesRegex(ValueError,'complete, verified replication'):
                build(*sources,output,page,replication=replication,skip_figures=True)
            self.assertEqual(raw,(replication/'results.json').read_bytes())
            self.assertFalse(output.exists())
            self.assertFalse(page.exists())


if __name__=='__main__':
    unittest.main()
