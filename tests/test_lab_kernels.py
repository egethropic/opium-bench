"""Kernel dispatch contracts and provenance without Torch imports or GPU work."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lab.runtime import _configure_qwen35_kernels


class LocalKernelTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def fused_recurrent_gated_delta_rule(q, k, v, g=None, beta=None, scale=None,
                initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False,
                state_v_first=False, cu_seqlens=None, **kwargs):
            self.calls.append(("delta", q, k, v, g, beta, scale, initial_state,
                               output_final_state, use_qk_l2norm_in_kernel, state_v_first, cu_seqlens, kwargs))
            return "output", "state"

        def causal_conv1d_fn(x, weight, bias=None, activation=None):
            self.calls.append(("prefill_conv", x, weight, bias, activation))
            return "convolved"

        def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None):
            self.calls.append(("decode_conv", x, conv_state, weight, bias, activation))
            return "updated"

        self.delta = SimpleNamespace(fused_recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule,
                                     chunk_gated_delta_rule=fused_recurrent_gated_delta_rule)
        self.conv = SimpleNamespace(causal_conv1d_fn=causal_conv1d_fn, causal_conv1d_update=causal_conv1d_update)
        self.model = SimpleNamespace(torch_recurrent_gated_delta_rule=self.original,
                                    torch_chunk_gated_delta_rule=self.original,
                                    causal_conv1d_fn=self.original, causal_conv1d_update=self.original)
        self.imports = []

    @staticmethod
    def original(*args, **kwargs):
        return "original"

    def import_module(self, name):
        self.imports.append(name)
        return {"fla.ops.gated_delta_rule": self.delta, "causal_conv1d": self.conv,
                "transformers.models.qwen3_5.modeling_qwen3_5": self.model}[name]

    def configure(self, enabled):
        with patch("importlib.metadata.version", side_effect=lambda name: "fixture-" + name), \
                patch("importlib.import_module", side_effect=self.import_module):
            return _configure_qwen35_kernels(enabled)

    def test_local_dispatch_preserves_normalization_scale_and_recurrent_state(self):
        module, info = self.configure(True)
        self.assertIs(module, self.model)
        self.assertEqual(self.imports[:2], ["fla.ops.gated_delta_rule", "causal_conv1d"])
        self.assertFalse(hasattr(self.delta, "recurrent_gated_delta_rule"), "Do not mutate installed FLA")
        for name in ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"):
            result = getattr(module, name)(1, 2, 3, g=4, beta=5, scale=.125,
                initial_state="previous", output_final_state=True, use_qk_l2norm_in_kernel=True,
                state_v_first=False, cu_seqlens="offsets", attention_mask="irrelevant")
            self.assertEqual(result, ("output", "state"))
            self.assertEqual(self.calls[-1], ("delta", 1, 2, 3, 4, 5, .125, "previous", True, True, False, "offsets", {}))
        self.assertEqual(module.causal_conv1d_fn(1, 2, 3, activation="silu", unused=True), "convolved")
        self.assertEqual(self.calls[-1], ("prefill_conv", 1, 2, 3, "silu"))
        self.assertEqual(module.causal_conv1d_update(1, "cache", 2, 3, "silu"), "updated")
        self.assertEqual(self.calls[-1], ("decode_conv", 1, "cache", 2, 3, "silu"))
        self.assertEqual(info["mode"], "explicit_local_v1")
        self.assertFalse(info["hub_kernels"])
        self.assertEqual(info["packages"]["fla-core"], "fixture-fla-core")
        identity = info["implementations"]["torch_recurrent_gated_delta_rule"]
        self.assertTrue(identity["callable"].endswith("fused_recurrent_gated_delta_rule"))
        self.assertEqual(len(identity["source_sha256"]), 64)

    def test_disable_restores_original_bindings_and_records_that_implementation(self):
        self.configure(True)
        module, info = self.configure(False)
        self.assertIs(module.torch_recurrent_gated_delta_rule, self.original)
        self.assertEqual(module.torch_recurrent_gated_delta_rule(), "original")
        self.assertEqual(info["mode"], "transformers_default")
        self.assertTrue(info["implementations"]["torch_recurrent_gated_delta_rule"]["callable"].endswith("LocalKernelTests.original"))

    def test_incompatible_normalization_or_state_contract_fails_before_rebinding(self):
        def incompatible(q, k, v, g=None, beta=None, **kwargs):
            raise AssertionError("Never execute incompatible implementation")
        self.delta.fused_recurrent_gated_delta_rule = incompatible
        with self.assertRaisesRegex(RuntimeError, "argument contract"):
            self.configure(True)
        self.assertNotIn("transformers.models.qwen3_5.modeling_qwen3_5", self.imports)
        self.assertIs(self.model.torch_recurrent_gated_delta_rule, self.original)

    def test_missing_optional_backend_fails_explicitly_without_modifying_model(self):
        with patch("importlib.import_module", side_effect=ImportError("missing FLA")):
            with self.assertRaisesRegex(ImportError, "missing FLA"):
                _configure_qwen35_kernels(True)
        self.assertFalse(hasattr(self.model, "_opium_bench_original_kernels"))


if __name__ == "__main__":
    unittest.main()
