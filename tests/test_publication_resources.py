"""Guarded publication preserves exact source/gzip bytes and old files on refusal."""
import gzip
import io
import os
from pathlib import Path
import tempfile
import unittest

from lab.resources import ResourceGuard, ResourceStop, Volume
from lab.storage import managed_writes
from publish_lab_study import deterministic_gzip, publish, _copy_publication_file
from tests import test_lab_analysis as fixtures


class PublicationResourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.free=2**30;self.calls=[]
        self.guard=ResourceGuard({"publication":self.root},reserve_bytes=1024**2,emergency_bytes=65536,max_write_bytes=65536,
            forbid_large_c_writes=False,resolver=lambda path:[Volume("shared",str(self.root))],
            telemetry=lambda path:{"free_bytes":self.free,"total_bytes":2**31})
        original=self.guard.before_write
        def before(operation,path,count):self.calls.append(count);return original(operation,path,count)
        self.guard.before_write=before
    def test_guarded_gzip_is_byte_identical_to_historical_stream_algorithm(self):
        raw=os.urandom(160000);expected=io.BytesIO()
        with gzip.GzipFile(filename="",fileobj=expected,mode="wb",mtime=0) as zipped:zipped.write(raw)
        target=self.root/"events.jsonl.gz"
        with managed_writes(self.guard):deterministic_gzip(target,raw)
        self.assertEqual(target.read_bytes(),expected.getvalue())
        self.assertLessEqual(max(self.calls),65536)
    def test_rejected_copy_preserves_previous_published_bytes(self):
        source=self.root/"source.npz";source.write_bytes(b"s"*150000)
        target=self.root/"existing.npz";target.write_bytes(b"preserve previous publication")
        original=self.guard.before_write
        def before(operation,path,count):
            if self.calls:self.free=0
            return original(operation,path,count)
        self.guard.before_write=before
        with managed_writes(self.guard),self.assertRaises(ResourceStop):_copy_publication_file(source,target)
        self.assertEqual(target.read_bytes(),b"preserve previous publication")
        self.assertFalse(list(self.root.glob("*.publish-*")))
    def test_complete_publication_uses_guard_and_keeps_source_bytes(self):
        data,output,receipt=fixtures.PublicationTests().fixture(self.root)
        original=(data/"calibrations/cal-fixture/vectors.npz").read_bytes()
        result=publish(receipt,data,output,dashboard=self.root/"results.html",skip_figures=True,guard=self.guard)
        self.assertEqual(result["status"],"complete")
        self.assertEqual((output/"calibration/vectors.npz").read_bytes(),original)
        self.assertGreater(len(self.calls),10)
        self.assertFalse(list(self.root.glob(".emergency-*")))
    def test_preflight_refusal_leaves_existing_evidence_and_emergency_record(self):
        data,output,receipt=fixtures.PublicationTests().fixture(self.root)
        frozen=(output/"protocol.json").read_bytes()
        # Enough for the physical small metadata slot, not the publication plan.
        self.free=2*1024**2
        with self.assertRaises(ResourceStop):publish(receipt,data,output,dashboard=self.root/"results.html",skip_figures=True,guard=self.guard)
        self.assertEqual((output/"protocol.json").read_bytes(),frozen)
        self.assertFalse((output/"receipt.json").exists())
        self.assertEqual(len(list(self.root.glob("resource-stop-*.json"))),1)


if __name__=="__main__":unittest.main()
