"""Untrusted evidence import, provenance, bounded writes and atomic installation."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import stat
import struct
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from lab.portability import (FORMAT, Limits, export_bundle, import_bundle,
                             import_json_export, _rename_no_replace)
from lab.resources import ResourceGuard, ResourceStop, Volume


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "source" / "run-fixture"
        self.run.mkdir(parents=True)
        (self.run / "manifest.json").write_bytes(b'{"id":"run-fixture","format_version":2,"status":"complete","source":{"commit":"abc"}}\n')
        (self.run / "events.jsonl").write_bytes(b'{"type":"message","content":"<script>not code</script>"}\n')
        (self.run / "conversation.json").write_bytes(b'[{"role":"assistant","tool_calls":[{"function":{"name":"aux","arguments":"{}"}}]}]\n')
        (self.run / "summary.json").write_bytes(b'{"correct":1}\n')
        self.archive = self.root / "export.zip"
        self.imports = self.root / "imports"

    def bundle(self):
        export_bundle(self.run, self.archive)
        return self.archive.read_bytes()

    def modify(self, raw, transform):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            files = [(info, archive.read(info)) for info in archive.infolist()]
        changed = transform(files)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in changed:
                archive.writestr(name, content)
        return buffer.getvalue()

    def assertEmpty(self):
        self.assertFalse(self.imports.exists() and list(self.imports.iterdir()))

    def test_roundtrip_preserves_every_source_byte_and_is_replay_only(self):
        raw = self.bundle()
        result = import_bundle(raw, self.imports)
        self.assertTrue(result["replay_only"])
        self.assertFalse(result["resume_eligible"])
        self.assertFalse(result["checkpoint_candidate"])
        self.assertEqual(result["source_sha256"], hashlib.sha256(raw).hexdigest())
        target = Path(result["path"])
        for original in self.run.iterdir():
            self.assertEqual(original.read_bytes(), (target / original.name).read_bytes())
        metadata = json.loads(next((target / "_portable").glob("import-*.json")).read_text())
        self.assertFalse(metadata["source_bundle"]["weights_included"])
        self.assertEqual(metadata["source_bundle"]["provenance"]["manifest_sha256"], hashlib.sha256((self.run / "manifest.json").read_bytes()).hexdigest())

    def test_checkpoint_presence_is_only_a_candidate_not_automatic_resume(self):
        (self.run / "checkpoint.json").write_text('{"schema_version":1,"boundary":"completed_turn"}')
        result = import_bundle(self.bundle(), self.imports)
        self.assertTrue(result["checkpoint_candidate"])
        self.assertFalse(result["resume_eligible"])

    def test_existing_json_export_imports_offline_preserving_original(self):
        raw = b'{"id":"legacy-run","manifest":{"format_version":2},"summary":{"correct":1},"events":[{"type":"message","content":"hello"}],"historical":false}\n'
        result = import_json_export(raw, self.imports)
        self.assertTrue(result["replay_only"])
        self.assertEqual((Path(result["path"]) / "_portable/source-export.json").read_bytes(), raw)
        self.assertFalse((Path(result["path"]) / "conversation.json").exists())

    def test_optional_calibration_remains_embedded_and_never_loaded(self):
        calibration = self.root / "cal-fixture"
        calibration.mkdir()
        (calibration / "calibration.json").write_text('{"schema_version":1,"model":"fixture"}')
        export_bundle(self.run, self.archive, calibration_dir=calibration)
        result = import_bundle(self.archive, self.imports)
        self.assertEqual(result["embedded_calibration"], "cal-fixture")
        self.assertTrue((Path(result["path"]) / "_portable/calibration/cal-fixture/calibration.json").exists())

    def test_corruption_or_inventory_mismatch_installs_nothing(self):
        raw = self.bundle()
        corrupt = self.modify(raw, lambda files: [(i, b'{}' if i.filename.endswith("summary.json") else data) for i, data in files])
        with self.assertRaisesRegex(ValueError, "hash"):
            import_bundle(corrupt, self.imports)
        self.assertEmpty()
        missing = self.modify(raw, lambda files: [(i, d) for i, d in files if not i.filename.endswith("summary.json")])
        with self.assertRaisesRegex(ValueError, "inventory"):
            import_bundle(missing, self.imports)
        self.assertEmpty()

    def test_path_escape_symlink_duplicates_and_case_collisions_rejected(self):
        raw = self.bundle()
        for name in ["../escape", "/absolute", "C:/escape", "run\\escape", "run/CON.txt", "run/x."]:
            with self.subTest(name=name):
                bad = self.modify(raw, lambda files: files + [(name, b"x")])
                with self.assertRaises(ValueError):
                    import_bundle(bad, self.imports)
                self.assertEmpty()
        info = zipfile.ZipInfo("run/run-fixture/linked.json")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        bad = self.modify(raw, lambda files: files + [(info, b"/etc/passwd")])
        with self.assertRaisesRegex(ValueError, "ordinary"):
            import_bundle(bad, self.imports)
        for name in ["bundle.json", "BUNDLE.JSON"]:
            with self.subTest(duplicate=name):
                with self.assertWarns(UserWarning) if name == "bundle.json" else self.subTest():
                    bad = self.modify(raw, lambda files: files + [(name, b"{}")])
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    import_bundle(bad, self.imports)
        self.assertEmpty()

    def test_json_duplicates_nonfinite_depth_and_future_versions_rejected(self):
        base = '{"id":"bad","manifest":{},"events":[],"summary":{}}'
        values = [base.replace('"events":[]', '"events":[],"events":[]'),
                  base.replace('"summary":{}', '"summary":{"x":NaN}'),
                  base.replace('"summary":{}', '"summary":{"x":1e999}'),
                  base.replace('"manifest":{}', '"manifest":{"format_version":99}'),
                  base.replace('"manifest":{}', '"manifest":"bad"')]
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                import_json_export(value.encode(), self.imports)
        with self.assertRaises(ValueError):
            import_json_export(base.replace('"summary":{}', '"summary":{"x":[[[[0]]]]}').encode(), self.imports, limits=Limits(json_depth=3))
        self.assertEmpty()

    def test_compressed_expanded_member_count_and_nested_gzip_bounds(self):
        raw = self.bundle()
        for limits in [Limits(compressed_bytes=10), Limits(expanded_bytes=10), Limits(member_bytes=10), Limits(members=1)]:
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                import_bundle(raw, self.imports, limits=limits)
        (self.run / "events.jsonl.gz").write_bytes(gzip.compress(b'{"x":"' + b"z" * 10000 + b'"}\n'))
        with self.assertRaisesRegex(ValueError, "Nested"):
            export_bundle(self.run, self.root / "bomb.zip", limits=Limits(member_bytes=5000))
        self.assertEmpty()

    def test_conflicts_never_overwrite_existing_run_or_export(self):
        raw = self.bundle()
        result = import_bundle(raw, self.imports)
        target = Path(result["path"])
        before = (target / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            import_bundle(raw, self.imports)
        self.assertEqual((target / "manifest.json").read_bytes(), before)
        with self.assertRaises(FileExistsError):
            export_bundle(self.run, self.archive)
        self.assertEqual(self.archive.read_bytes(), raw)

    def test_atomic_install_does_not_replace_empty_racing_directory(self):
        source, destination = self.root / "stage", self.root / "target"
        source.mkdir()
        destination.mkdir()
        with self.assertRaises(FileExistsError):
            _rename_no_replace(source, destination)
        self.assertTrue(source.exists() and destination.exists())

    def test_source_symlink_weights_and_pickle_are_rejected(self):
        for name in ["model.safetensors", "state.pkl", "pytorch_model.bin"]:
            file = self.run / name
            file.write_bytes(b"untrusted")
            with self.subTest(name=name), self.assertRaises(ValueError):
                export_bundle(self.run, self.archive)
            file.unlink()
        (self.run / "escape.json").symlink_to(self.run / "manifest.json")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            export_bundle(self.run, self.archive)

    def test_partial_final_event_is_preserved_and_prevents_resume_candidate(self):
        (self.run / "events.jsonl").write_bytes(b'{"type":"message"}\n{"type":')
        (self.run / "checkpoint.json").write_text('{}')
        result = import_bundle(self.bundle(), self.imports)
        self.assertTrue(result["partial_events"])
        self.assertFalse(result["checkpoint_candidate"])
        self.assertEqual((Path(result["path"]) / "events.jsonl").read_bytes(), (self.run / "events.jsonl").read_bytes())

    def test_nonfinite_final_event_is_not_misclassified_as_partial_write(self):
        (self.run / "events.jsonl").write_bytes(b'{"x":NaN}')
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.bundle()

    def test_numeric_npz_is_inert_and_object_pickle_arrays_are_rejected(self):
        def npz(dtype, shape, payload):
            header = repr(dict(descr=dtype, fortran_order=False, shape=shape)).encode() + b"\n"
            array = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + payload
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("vector.npy", array)
            return buffer.getvalue()
        vector = self.run / "vectors.npz"
        vector.write_bytes(npz("<f4", (2,), struct.pack("<ff", .1, .2)))
        result = import_bundle(self.bundle(), self.imports)
        self.assertEqual((Path(result["path"]) / "vectors.npz").read_bytes(), vector.read_bytes())
        vector.write_bytes(npz("|O8", (1,), b"pickle!"))
        with self.assertRaisesRegex(ValueError, "Object"):
            export_bundle(self.run, self.root / "object.zip")

    def test_future_checkpoint_version_is_rejected_before_install(self):
        (self.run / "checkpoint.json").write_text('{"schema_version":99}')
        with self.assertRaisesRegex(ValueError, "checkpoint version"):
            self.bundle()

    def test_multiple_nested_compressed_files_share_one_expansion_budget(self):
        for index in range(3):
            (self.run / f"events-{index}.jsonl.gz").write_bytes(gzip.compress(b'{"x":"' + b"z" * 2000 + b'"}\n'))
        with self.assertRaisesRegex(ValueError, "combined expanded"):
            export_bundle(self.run, self.archive, limits=Limits(expanded_bytes=5000, member_bytes=3000))

    def test_reserve_drop_cleans_owned_stage_and_preserves_unrelated_files(self):
        raw = self.bundle()
        self.imports.mkdir()
        unrelated = self.imports / "unrelated"
        unrelated.mkdir()
        (unrelated / "evidence.txt").write_text("retain")
        calls = 0
        def telemetry(path):
            nonlocal calls
            calls += 1
            return dict(free_bytes=100000 if calls < 3 else 0, total_bytes=100000)
        guard = ResourceGuard(reserve_bytes=0, emergency_bytes=0, max_write_bytes=64,
                              resolver=lambda path: [Volume("fixture", str(self.root))], telemetry=telemetry)
        with self.assertRaises(ResourceStop):
            import_bundle(raw, self.imports, guard=guard)
        self.assertEqual([p.name for p in self.imports.iterdir()], ["unrelated"])
        self.assertEqual((unrelated / "evidence.txt").read_text(), "retain")
        self.assertEqual(guard.reservations, {})


if __name__ == "__main__":
    unittest.main()
