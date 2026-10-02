"""Real local HTTP evidence transfers require no inference process."""
from contextlib import contextmanager
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

from lab.portability import export_bundle
from lab.resources import ResourceGuard, Volume
from lab.server import create_server
from lab.service import LabService


class PortableHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.service = LabService(self.root / "data", self.root / "cache", historical=[])
        self.free = 100 * 2**30
        self.service.resources = ResourceGuard(dict(data=self.service.store.root),
            telemetry=lambda p: dict(free_bytes=self.free, total_bytes=200 * 2**30),
            resolver=lambda p: [Volume("fixture", str(self.root))], interval_seconds=.001)
        self.server = create_server(self.service, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temp.cleanup()

    def request(self, method, url, data=None, headers=None):
        client = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        client.request(method, url, body=data, headers=headers or {})
        reply = client.getresponse()
        result = reply.status, dict(reply.getheaders()), reply.read()
        client.close()
        return result

    def upload(self, raw, mime="application/json", **headers):
        return self.request("POST", "/api/import", raw, dict(
            {"Content-Type": mime, "X-CSRF-Token": self.service.csrf}, **headers))

    def source(self, identifier="run-transfer", **manifest):
        return json.dumps(dict(id=identifier, manifest=dict(id=identifier, mode="chat",
            format_version=2, status="complete", config={}, **manifest), summary={},
            events=[dict(type="message", role="user", content="<script>window.executed=true</script>")])).encode()

    def test_json_upload_replay_and_zip_download_without_model(self):
        original = self.source()
        status, _, raw = self.upload(original)
        self.assertEqual(status, 201, raw)
        self.assertTrue(json.loads(raw)["replay_only"])
        self.assertIsNone(self.service.process)
        path = self.service.store.runs / "run-transfer"
        self.assertEqual((path / "_portable/source-export.json").read_bytes(), original)
        status, _, replay = self.request("GET", "/api/runs/run-transfer")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(replay)["imported"])
        export_finished = threading.Event()
        original_export = self.service.portable_export

        @contextmanager
        def observed_export(identifier):
            with original_export(identifier) as archive_path:
                yield archive_path
            export_finished.set()

        with patch.object(self.service, "portable_export", observed_export):
            status, headers, bundle = self.request("GET", "/api/runs/run-transfer/bundle")
            # Receiving the final response byte need not mean the handler has
            # finished its context-manager cleanup, especially on mounted NTFS.
            self.assertTrue(export_finished.wait(5), "Export cleanup did not finish")
        self.assertEqual(status, 200, bundle[:200])
        self.assertIn("attachment", headers["Content-Disposition"])
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            self.assertEqual(archive.read("run/run-transfer/_portable/source-export.json"), original)
        self.assertEqual(list((self.service.store.root / "tmp").iterdir()), [])

    def test_imported_html_and_dead_owner_are_inert(self):
        source = self.root / "source" / "run-inert"
        source.mkdir(parents=True)
        manifest = dict(id="run-inert", mode="chat", format_version=2, status="running",
                        owner=dict(service_id="old", service_pid=2123456789))
        raw = json.dumps(manifest).encode()
        (source / "manifest.json").write_bytes(raw)
        (source / "report.html").write_text("<script>dangerousImport()</script>")
        export_bundle(source, self.root / "inert.zip")
        status, _, result = self.upload((self.root / "inert.zip").read_bytes(), "application/zip")
        self.assertEqual(status, 201, result)
        with patch.object(self.service, "_pid_alive", return_value=False):
            self.service._recover_abandoned_runs()
        self.assertEqual((self.service.store.runs / "run-inert/manifest.json").read_bytes(), raw)
        status, _, report = self.request("GET", "/api/runs/run-inert/report")
        self.assertEqual(status, 200)
        self.assertNotIn(b"dangerousImport", report)

    def test_duplicate_including_bundled_ids_cannot_shadow_evidence(self):
        history = self.root / "history"
        target = history / "run-transfer"
        target.mkdir(parents=True)
        raw = b'{"id":"run-transfer","status":"complete"}'
        (target / "manifest.json").write_bytes(raw)
        self.service.store.historical.append(history)
        status, _, _ = self.upload(self.source())
        self.assertEqual(status, 409)
        self.assertEqual((target / "manifest.json").read_bytes(), raw)
        self.assertFalse((self.service.store.runs / "run-transfer").exists())

    def test_csrf_corruption_and_resource_failure_install_nothing(self):
        status, _, _ = self.upload(self.source(), **{"X-CSRF-Token": "invalid"})
        self.assertEqual(status, 403)
        status, _, _ = self.upload(b"not a zip", "application/zip")
        self.assertEqual(status, 400)
        self.free = 1
        status, _, raw = self.upload(self.source())
        self.assertEqual(status, 507, raw)
        self.assertEqual(json.loads(raw)["resource"]["status"], "resource_stopped")
        self.assertEqual(list(self.service.store.runs.iterdir()), [])
        self.assertIsNone(self.service.process)


if __name__ == "__main__":
    unittest.main()
