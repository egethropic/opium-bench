"""Portable evidence lookup gives local records precedence and confines symlinks."""
from pathlib import Path
import tempfile
import unittest

from lab.storage import Store, atomic_json


class EvidenceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bundled = self.root / 'bundled'
        self.store = Store(self.root / 'data', [self.bundled])

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, base, identifier, correct):
        atomic_json(base / identifier / 'manifest.json', {'status': 'complete', 'created_at': '2026-10-02'})
        atomic_json(base / identifier / 'summary.json', {'correct': correct})

    def test_catalog_deduplicates_local_and_bundled_copy_with_local_precedence(self):
        self.save(self.store.runs, 'same-run', 2)
        self.save(self.bundled, 'same-run', 1)
        self.save(self.bundled, 'other-run', 3)
        rows = self.store.catalog()
        self.assertEqual(len(rows), 2)
        selected = next(row for row in rows if row['id'] == 'same-run')
        self.assertEqual(selected['summary']['correct'], 2)
        self.assertFalse(selected['historical'])
        self.assertEqual(self.store.read_run('same-run')['summary'], selected['summary'])
        self.assertTrue(next(row for row in rows if row['id'] == 'other-run')['historical'])

    def test_duplicate_historical_roots_and_incomplete_local_copy_match_lookup(self):
        self.save(self.bundled, 'same-run', 1)
        self.store.historical.append(self.bundled)
        self.assertEqual(len(self.store.catalog()), 1)
        # An incomplete local record must not silently display another copy's metadata.
        (self.store.runs / 'same-run').mkdir()
        self.assertEqual(self.store.catalog(), [])
        self.assertEqual(self.store.read_run('same-run')['manifest'], {})

    def test_run_directory_symlink_cannot_escape_evidence_root(self):
        escaped = self.root / 'outside'
        atomic_json(escaped / 'manifest.json', {'secret': True})
        (self.store.runs / 'linked-run').symlink_to(escaped, target_is_directory=True)
        self.assertEqual(self.store.catalog(), [])
        for writable in (False, True):
            with self.subTest(writable=writable), self.assertRaises(FileNotFoundError):
                self.store.run_path('linked-run', writable=writable)


if __name__ == '__main__':
    unittest.main()
