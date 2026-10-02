"""Resource-stop integration with fake capacity, small files, and CPU runtimes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lab.resources import ResourceGuard, ResourceStop, Volume
from lab.storage import Store, atomic_json, guarded_bytes, guarded_copyfile, guarded_npz, managed_writes
from lab.worker import Worker
from lab.protocol import validate_recipe
from tests.test_lab_worker import FakeRuntime, wrapped


class ManagedStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.free=4*2**30;self.tick=0;self.writes=[]
        def clock():self.tick+=1;return self.tick
        self.guard=ResourceGuard({"data":self.root},reserve_bytes=1024**2,emergency_bytes=65536,max_write_bytes=65536,
            forbid_large_c_writes=False,resolver=lambda path:[Volume("test",str(self.root))],
            telemetry=lambda path:{"free_bytes":self.free,"total_bytes":8*2**30},clock=clock)
        self.original_before=self.guard.before_write
        def before(operation,path,count):
            self.writes.append(count)
            return self.original_before(operation,path,count)
        self.guard.before_write=before

    def tearDown(self):self.tmp.cleanup()

    def test_bounded_chunks_and_context_do_not_leak_or_change_json_encoding(self):
        target=self.root/"payload.bin";data=b"a"*200000
        with managed_writes(self.guard):
            guarded_bytes(target,data)
            atomic_json(self.root/"unicode.json",{"value":"é"},ensure_ascii=True)
        self.assertEqual(target.read_bytes(),data)
        self.assertEqual((self.root/"unicode.json").read_text(),json.dumps({"value":"é"},indent=2,allow_nan=False)+"\n")
        self.assertLessEqual(max(self.writes),65536)
        self.assertEqual(self.guard.reservations,{})
        self.free=0
        atomic_json(self.root/"unmanaged.json",{"outside":"context"})
        with managed_writes(self.guard),self.assertRaises(ResourceStop):
            atomic_json(self.root/"blocked.json",{})
        self.assertFalse((self.root/"blocked.json").exists())

    def test_rejected_atomic_update_preserves_old_bytes_and_removes_only_owned_temp(self):
        target=self.root/"checkpoint.json";atomic_json(target,{"safe":"earlier"})
        earlier=target.read_bytes();self.free=0
        with self.assertRaises(ResourceStop):atomic_json(target,{"partial":"new"},guard=self.guard)
        self.assertEqual(target.read_bytes(),earlier)
        self.assertEqual(list(self.root.glob("*.tmp-*")),[])
        self.assertEqual(self.guard.reservations,{})

    def test_drop_mid_write_stops_at_chunk_boundary_without_disk_filling(self):
        target=self.root/"partial.bin"
        def before(operation,path,count):
            if len(self.writes)==1:self.free=0
            self.writes.append(count)
            return self.original_before(operation,path,count)
        self.guard.before_write=before
        with self.assertRaises(ResourceStop):guarded_bytes(target,b"x"*150000,guard=self.guard)
        self.assertEqual(target.stat().st_size,65536)
        self.assertEqual(self.guard.reservations,{})

    def test_guarded_copy_keeps_identical_bytes(self):
        source=self.root/"original";source.write_bytes(bytes(range(256))*700)
        destination=self.root/"copied"
        guarded_copyfile(source,destination,guard=self.guard)
        self.assertEqual(source.read_bytes(),destination.read_bytes())
        self.assertLessEqual(max(self.writes),65536)

    def test_numpy_stream_writes_are_bounded_numeric_and_readable(self):
        try:import numpy as np
        except ImportError:self.skipTest("Optional NumPy runtime not installed")
        data=np.arange(30000,dtype=np.float32)
        target=self.root/"vectors.npz"
        guarded_npz(target,guard=self.guard,values=data,scale=np.array(2.))
        with np.load(target,allow_pickle=False) as archive:self.assertTrue(np.array_equal(archive["values"],data))
        self.assertLessEqual(max(self.writes),65536)
        with self.assertRaises(ValueError):guarded_npz(self.root/"bad.npz",guard=self.guard,values=np.array([{}],dtype=object))
        self.assertFalse((self.root/"bad.npz").exists())

    def test_store_emergency_overlay_preserves_manifest_events_and_safe_checkpoint(self):
        store=Store(self.root/"store",guard=self.guard)
        with patch("lab.storage.source_manifest",return_value={}):identifier,path=store.create("experiment",{})
        atomic_json(path/"checkpoint.json",{"safe":"boundary"},guard=self.guard)
        store.append(identifier,{"type":"token","text":"partial"})
        originals={name:(path/name).read_bytes() for name in ("manifest.json","events.jsonl","checkpoint.json")}
        self.assertIn(identifier,store.emergency)
        self.free=0
        with self.assertRaises(ResourceStop):store.append(identifier,{"type":"token","text":"blocked"})
        error=ResourceStop("fake reserve breach")
        saved=store.resource_stop(identifier,error,{"tokens":1})
        self.assertEqual(store.resource_stop(identifier,error),saved)
        self.assertEqual(store.catalog()[0]["status"],"resource_stopped")
        run=store.read_run(identifier)
        self.assertEqual(run["manifest"]["status"],"resource_stopped")
        self.assertEqual(run["summary"]["tokens"],1)
        for name,raw in originals.items():self.assertEqual((path/name).read_bytes(),raw)
        self.assertEqual(len(list(path.glob("resource-stop-*.json"))),1)
        self.assertNotIn(identifier,store.emergency)

    def test_worker_resource_loss_stops_generation_with_last_safe_checkpoint(self):
        events=[]
        runtime=FakeRuntime([dict(raw_text=wrapped(),tokens=8)],after_token=lambda i:setattr(self,"free",0) if i==0 else None)
        worker=Worker(self.root/"cache",runtime,events.append,resources=self.guard,data_dir=self.root)
        worker.model_info={"model_id":"fake"}
        config=validate_recipe(dict(demonstration="none",task_count=1,action_budget=3,token_budget=64))
        payload=dict(run_id="resource-run",mode="experiment",config=config,calibration_id="fake",calibration_dir=str(self.root/"calibration"),out_dir=str(self.root/"run"))
        worker.jobs.put(dict(command="start_session",id="cmd",payload=payload));worker.jobs.put(None)
        worker.loop()
        finished=[event for event in events if event["type"]=="session_finished"]
        self.assertEqual(len(finished),1)
        self.assertEqual(finished[0]["status"],"resource_stopped")
        self.assertEqual(finished[0]["summary"]["tokens"],1)
        self.assertEqual(json.loads((self.root/"run"/"checkpoint.json").read_text())["session"]["turns"],0)
        self.assertEqual([e["status"] for e in events if e["type"]=="job"][-1],"resource_stopped")
        self.assertEqual(len(list((self.root/"resource-stops").glob("resource-stop-*.json"))),1)
        self.assertFalse(worker.active)
        self.assertEqual(self.guard.reservations,{})

    def test_worker_preflight_failure_writes_emergency_without_loading_model(self):
        events=[];worker=Worker(self.root/"cache",FakeRuntime(),events.append,resources=self.guard,data_dir=self.root)
        worker.jobs.put(dict(command="load_model",id="cmd",payload={"profile":{"model_id":"Qwen/27B","allow_download":True}}));worker.jobs.put(None)
        worker.loop()
        self.assertEqual([e["status"] for e in events if e["type"]=="job"][-1],"resource_stopped")
        self.assertEqual(len(list((self.root/"resource-stops").glob("resource-stop-*.json"))),1)


if __name__=="__main__":unittest.main()
