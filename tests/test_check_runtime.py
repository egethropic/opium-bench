"""Read-only runtime inspection: package metadata, explicit import, fake storage."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import check_runtime as checker
from lab.resources import ResourceGuard, Volume


class RuntimeCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def guard(self, destinations, free=20 * 2**30):
        return ResourceGuard(destinations, resolver=lambda path: [Volume("shared", str(self.root))],
                             telemetry=lambda path: dict(free_bytes=free, total_bytes=100 * 2**30))

    def observed(self, python, packages, cuda=False):
        pins = checker.requirements(checker.ROOT / "requirements-27b.txt")
        versions = {row["package"]: row["expected"] for row in pins}
        versions["torch"] = "2.8.0+cu128"
        return dict(executable=str(python), prefix="test", python="3.12.3", python_minor=[3, 12],
                    implementation="CPython", system="Linux", machine="x86_64",
                    packages={name: versions[name] for name in packages},
                    cuda=dict(status="checked", available=True, device_count=1,
                              torch_cuda_version="12.8", cxx11_abi=True) if cuda else dict(status="not_checked"))

    def run_check(self, **kwargs):
        return checker.inspect_runtime(python=sys.executable, profile="27b", data_dir=self.root / "data",
                                       cache_dir=self.root / "cache", guard_factory=self.guard,
                                       probe=kwargs.pop("probe", self.observed), **kwargs)

    def test_requirements_include_exact_local_wheel_pin_and_no_network(self):
        rows = checker.requirements(checker.ROOT / "requirements-27b.txt")
        wheel = next(row for row in rows if row["package"] == "causal-conv1d")
        self.assertEqual(wheel["expected"], "1.7.0")
        self.assertEqual(wheel["wheel_filename_version"], "1.7.0+cu12torch2.8cxx11abiTRUE")
        self.assertEqual(len(wheel["wheel_sha256"]), 64)
        self.assertTrue(checker.matching_version("2.8.0+cu128", "2.8.0"))
        self.assertTrue(checker.matching_version("1.7.0", wheel["expected"]))
        self.assertFalse(checker.matching_version("1.7.0", wheel["wheel_filename_version"]))
        self.assertFalse(checker.matching_version("2.8.0rc1", "2.8.0"))
        req = self.root / "unknown.txt"
        req.write_text("torch>=2\n")
        with self.assertRaisesRegex(ValueError, "Unsupported requirement"):
            checker.requirements(req)

    def test_success_checks_reserve_without_creating_any_destinations(self):
        result = self.run_check(check_cuda=True)
        self.assertEqual(result["status"], "checks_passed")
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(len(result["storage"]["volumes"]), 1)
        self.assertEqual(result["storage"]["reserve_bytes"], 10 * 2**30)
        self.assertEqual(result["storage"]["reservations"], {})
        self.assertEqual(result["destinations"]["temp"], str(self.root / "data" / "tmp"))

    def test_missing_mismatch_and_wrong_profile_are_not_passes(self):
        def bad(*args):
            result = self.observed(*args)
            result["packages"]["transformers"] = "4.57.6"
            result["packages"]["fla-core"] = None
            result["python_minor"] = [3, 11]
            result["system"] = "Windows"
            return result
        result = self.run_check(probe=bad)
        self.assertEqual(result["status"], "issues_found")
        self.assertEqual(len(result["issues"]), 4)
        self.assertIn("version_mismatch", result["issues"][0])
        self.assertTrue(any("fla-core: missing" in issue for issue in result["issues"]))

    def test_cuda_unavailable_wrong_build_and_abi_reported(self):
        def bad(*args):
            result = self.observed(*args)
            result["cuda"].update(available=False, torch_cuda_version=None, cxx11_abi=False)
            return result
        result = self.run_check(check_cuda=True, probe=bad)
        self.assertEqual(len(result["issues"]), 3)
        self.assertEqual(result["status"], "issues_found")

    def test_low_shared_volume_is_failure_without_writes(self):
        with patch.object(self, "guard", side_effect=lambda paths: RuntimeCheckTests.guard(self, paths, 9 * 2**30)):
            result = self.run_check()
        self.assertEqual(result["storage"]["status"], "resource_stopped")
        self.assertEqual(result["status"], "issues_found")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_package_probe_never_imports_torch_without_explicit_flag(self):
        # The fake package throws if imported. No real CUDA or torch import occurs.
        (self.root / "torch.py").write_text("raise RuntimeError('explicit-import-fixture')\n")
        with patch.dict(os.environ, {"PYTHONPATH": str(self.root)}):
            result = checker.probe_python(sys.executable, ["opium-impossible-missing-fixture"], False)
            self.assertEqual(result["cuda"]["status"], "not_checked")
            self.assertIsNone(result["packages"]["opium-impossible-missing-fixture"])
            explicit = checker.probe_python(sys.executable, [], True)
        self.assertEqual(explicit["cuda"]["status"], "error")
        self.assertIn("explicit-import-fixture", explicit["cuda"]["error"])
        self.assertFalse((self.root / "__pycache__").exists())

    def test_selected_executable_symlink_and_spaces_not_resolved_or_shell(self):
        selected = self.root / "worker python"
        selected.symlink_to(sys.executable)
        result = checker.probe_python(selected, [], False)
        self.assertEqual(result["executable"], str(selected))
        self.assertEqual(result["requested_executable"], str(selected))

    def test_unavailable_interpreter_still_reports_storage(self):
        def missing(*args):
            raise FileNotFoundError("missing-worker-fixture")
        result = self.run_check(probe=missing)
        self.assertEqual(result["status"], "issues_found")
        self.assertIn("missing-worker-fixture", result["issues"])
        self.assertEqual(len(result["storage"]["volumes"]), 1)


if __name__ == "__main__":
    unittest.main()
