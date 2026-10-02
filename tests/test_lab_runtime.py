"""Local numerical/protocol checks; no model downloads or GPU allocation."""
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.torch_version import TorchVersion

from lab.calibration_data import rows, validate_rows
from lab.runtime import (Cancelled, Runtime, _digest, normalized_control,
                         prepare_messages, resolve_blocks, split_reasoning, DEFAULT_REVISION, tokenizer_identity,
                         _quantization_load_kwargs)


class Tokenizer:
    eos_token_id = 0
    all_special_ids = [0, 1, 3]
    chat_template = "fixture template"
    pieces = {0: "<eos>", 1: "<think>", 2: "Let me reason.", 3: "</think>",
              4: '<tool_call>{"name":"aux_operation","arguments":{}}</tool_call>',
              5: "Hello", 6: " world", 7: "�", 8: "é"}

    def __init__(self):
        self.corpus = {row["text"]: {"pain": 10, "joy": 20, "neutral": 30}[row["label"]]
                       for row in rows()}
        self.template_kwargs = None

    def __call__(self, text, return_tensors=None):
        return SimpleNamespace(input_ids=torch.tensor([[self.corpus.get(text, 30), 30]]))

    def get_vocab(self):
        return {piece: token_id for token_id, piece in self.pieces.items()}

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        self.messages = messages
        return "prompt"

    def decode(self, ids, skip_special_tokens=False):
        if ids == [7, 8]:
            return "é"
        return "".join(self.pieces.get(t, f"[{t}]") for t in ids)


class Block(nn.Module):
    def forward(self, value):
        return value * 1.1, "untouched tail"


class Model(nn.Module):
    def __init__(self, tokens=(5, 6, 0), fail_at=None):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block() for _ in range(4)])
        self.tokens, self.calls, self.fail_at = tokens, [], fail_at
        self.generation_config = SimpleNamespace(eos_token_id=0)

    def forward(self, input_ids, past_key_values=None, use_cache=True, logits_to_keep=0):
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("simulated model failure")
        self.calls.append({"ids": input_ids.clone(), "cache": past_key_values,
                           "logits_to_keep": logits_to_keep})
        label = int(input_ids[0, 0])
        value = {10: [4., 1., 2., 1.], 20: [1., 4., 2., 1.]}.get(label, [1., 1., 2., 1.])
        hidden = torch.tensor(value).reshape(1, 1, 4).repeat(1, input_ids.shape[1], 1)
        for block in self.model.layers:
            hidden = block(hidden)[0]
        # Nontrivial logits let calibration measure a real distribution change.
        logits = torch.zeros((1, 1, 40))
        logits[0, 0, 10:14] = hidden[0, -1]
        step = 0 if past_key_values is None else past_key_values
        logits[0, 0, self.tokens[min(step, len(self.tokens)-1)]] = 12
        return SimpleNamespace(logits=logits, past_key_values=step+1 if use_cache else None)


def runtime(tokens=(5, 6, 0), fail_at=None):
    result = Runtime()
    result.model, result.tokenizer, result.device = Model(tokens, fail_at), Tokenizer(), "cpu"
    result.blocks = result.model.model.layers
    fingerprint = {"model_id": "test", "revision": "1", "architecture": "qwen3",
                   "adapter": "qwen3", "dtype": "float32", "quantization": "none",
                   "hidden_size": 4, "layers": 4}
    result.info = {"status": "loaded", "hidden_size": 4, "fingerprint": fingerprint,
                   "fingerprint_sha256": _digest(fingerprint), "max_position_embeddings": 2048}
    result._configure_forward()
    return result


def package():
    return {"metadata": {"layer": 1, "downstream_layer": 3}, "vectors": {
        "pain": np.array([1., 0., 0., 0.]), "joy": np.array([0., 1., 0., 0.]),
        "random": np.array([0., 0., 0., 1.]), "scale": np.array(2.),
        "probe_center": np.zeros(4), "probe_scale": np.array(2.),
        "probe_pain": np.array([1., 0., 0., 0.]), "probe_joy": np.array([0., 1., 0., 0.]),
        "downstream_center": np.zeros(4), "downstream_scale": np.array(2.),
        "downstream_pain": np.array([1., 0., 0., 0.]), "downstream_joy": np.array([0., 1., 0., 0.]),
    }}


class RuntimeNumericalTests(unittest.TestCase):
    def test_controls_bound_signed_axes_and_no_second_level_multiplier(self):
        value = normalized_control({"pain": -2, "joy": 1.5, "suppression": .8, "level": .2}, "output")
        self.assertEqual(value["effective"], {"pain": -2., "joy": 1.5, "suppression": .8, "random_gain": 0.})
        for invalid in (True, float("nan"), float("inf"), 4.1, -4.1, "1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalized_control({"pain": invalid}, "output")

    def test_phase_scopes_and_measurement_only_gate_every_edit(self):
        for setting in ({"measurement_only": True}, {"phase_scope": "reasoning"}):
            result = normalized_control({"pain": 1, "joy": 1, "suppression": 1, **setting}, "output")
            self.assertFalse(result["applied"])
            self.assertFalse(any(result["effective"].values()))

    def test_aux_gate_never_erases_independent_baseline_challenge(self):
        result = normalized_control({"enabled": False, "pain": 2, "joy": 0, "suppression": 0}, "output")
        self.assertEqual(result["effective"]["pain"], 2)
        self.assertFalse(result["enabled"])

    def test_passive_probes_leave_model_bitwise_identical(self):
        rt = runtime()
        ids = torch.tensor([[30, 30]])
        baseline = rt._forward(input_ids=ids).logits
        state = {"dose": normalized_control({}, "output")}
        with rt._hooks(package(), state):
            measured = rt._forward(input_ids=ids).logits
        self.assertTrue(torch.equal(baseline, measured))
        self.assertEqual(state["measurements"]["pre"], state["measurements"]["post"])
        self.assertIsNotNone(state["measurements"]["downstream"])
        self.assertEqual(state["measurements"]["relative_delta"], 0.)
        self.assertTrue(all(not block._forward_hooks for block in rt.blocks))

    def test_full_suppression_cancels_challenge_with_joy_preserved(self):
        rt = runtime()
        state = {"dose": normalized_control({"pain": 4, "suppression": 1, "joy": .5}, "output")}
        with rt._hooks(package(), state):
            rt._forward(input_ids=torch.tensor([[30, 30]]))
        self.assertAlmostEqual(state["measurements"]["post"]["pain"], 0., places=6)
        self.assertAlmostEqual(state["measurements"]["post"]["joy"]-state["measurements"]["pre"]["joy"], .5, places=6)

    def test_hooks_are_removed_on_forward_failure(self):
        rt = runtime(fail_at=0)
        state = {"dose": normalized_control({"joy": 1}, "output")}
        with self.assertRaisesRegex(RuntimeError, "simulated"), rt._hooks(package(), state):
            rt._forward(input_ids=torch.tensor([[30]]))
        self.assertTrue(all(not block._forward_hooks for block in rt.blocks))

    def test_independent_random_gain_changes_orthogonal_dimension(self):
        rt = runtime()
        before = rt._forward(input_ids=torch.tensor([[30, 30]])).logits
        state = {"dose": normalized_control({"random_gain": 1}, "output")}
        with rt._hooks(package(), state):
            after = rt._forward(input_ids=torch.tensor([[30, 30]])).logits
        self.assertTrue(torch.equal(before[..., 10:13], after[..., 10:13]))
        self.assertGreater(float(after[..., 13]), float(before[..., 13]))

    def test_dedicated_adapter_rejects_unrecognized_model_architecture(self):
        model = Model()
        self.assertEqual(resolve_blocks(model, "qwen3")[1], "model.layers")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            resolve_blocks(model, "arbitrary_remote_code")


class RuntimeGenerationTests(unittest.TestCase):
    def test_cache_rebuilt_each_turn_and_logits_last_only(self):
        rt = runtime()
        for _ in range(2):
            result = rt.generate([{"role": "user", "content": "Hi"}], [], {"temperature": 0}, None)
            self.assertEqual(result["content"], "Hello world")
            self.assertEqual(result["finish_reason"], "eos")
        self.assertIsNone(rt.model.calls[0]["cache"])
        self.assertIsNone(rt.model.calls[3]["cache"])
        self.assertTrue(all(call["logits_to_keep"] == 1 for call in rt.model.calls))
        self.assertEqual([call["ids"].shape[1] for call in rt.model.calls], [2, 1, 1, 2, 1, 1])

    def test_thinking_phase_clock_counts_closing_marker_and_eos(self):
        rt = runtime((2, 3, 4, 0))
        rt._package = lambda path: package()
        events, clock = [], [0]

        def control():
            return {"joy": 1/(2**clock[0]), "level": 1/(2**clock[0]), "phase_scope": "reasoning"}

        def emit(event):
            events.append(event)
            clock[0] += 1

        result = rt.generate([{"role": "user", "content": "Work"}], [],
                             {"thinking": True, "temperature": 0}, Path("fake"), control, emit)
        self.assertEqual([event["phase"] for event in events], ["reasoning", "reasoning", "output", "output"])
        self.assertEqual([event["dose"]["effective"]["joy"] for event in events], [1., .5, 0., 0.])
        self.assertEqual(result["reasoning_tokens"], 2)
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(result["reasoning"], "Let me reason.")
        self.assertTrue(result["content"].startswith("<tool_call>"))
        self.assertEqual(clock[0], 4)
        self.assertTrue(events[-1]["is_eos"])

    def test_unfinished_reasoning_is_never_returned_as_tool_output(self):
        rt = runtime((2, 4))
        result = rt.generate([{"role": "user", "content": "Work"}], [],
                             {"thinking": True, "max_new_tokens": 2, "temperature": 0}, None)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["content"], "")
        self.assertIn("<tool_call>", result["reasoning"])

    def test_cancellation_returns_partial_and_cleans_hooks(self):
        rt = runtime()
        rt._package = lambda path: package()
        events = []
        result = rt.generate([{"role": "user", "content": "Hi"}], [], {"temperature": 0},
                             Path("fake"), lambda: {}, events.append, lambda: len(events) >= 1)
        self.assertEqual(result["finish_reason"], "stopped")
        self.assertEqual(result["token_ids"], [5])
        self.assertFalse(result["truncated"])
        self.assertTrue(all(not block._forward_hooks for block in rt.blocks))

    def test_callback_failure_cleans_hooks(self):
        rt = runtime()
        rt._package = lambda path: package()

        def fail(event):
            raise RuntimeError("observer disconnected")

        with self.assertRaisesRegex(RuntimeError, "observer" ):
            rt.generate([{"role": "user", "content": "Hi"}], [], {}, Path("fake"), emit=fail)
        self.assertTrue(all(not block._forward_hooks for block in rt.blocks))

    def test_utf8_decode_revision_has_authoritative_full_text(self):
        rt = runtime((7, 8, 0))
        events = []
        rt.generate([{"role": "user", "content": "Hi"}], [], {"temperature": 0}, None, emit=events.append)
        self.assertEqual(events[0]["full_text"], "�")
        self.assertEqual(events[1]["full_text"], "é")
        self.assertTrue(events[1]["replace_text"])

    def test_nonzero_effect_requires_calibration(self):
        rt = runtime()
        with self.assertRaisesRegex(ValueError, "calibration"):
            rt.generate([{"role": "user", "content": "Hi"}], [], {}, None, lambda: {"joy": 1})
        self.assertEqual(rt.model.calls, [])

    def test_context_budget_refuses_silent_truncation(self):
        rt = runtime()
        with self.assertRaisesRegex(ValueError, "context budget"):
            rt.generate([{"role": "user", "content": "Hi"}], [],
                        {"max_new_tokens": 200, "max_context_tokens": 128}, None)

    def test_history_drop_is_explicit_and_does_not_mutate_original(self):
        messages = [{"role": "assistant", "reasoning_content": "private external-model trace",
                     "content": "<think>trace</think>Answer"}]
        prepared = prepare_messages(messages, "drop")
        self.assertEqual(prepared[0]["content"], "Answer")
        self.assertNotIn("reasoning_content", prepared[0])
        self.assertIn("reasoning_content", messages[0])


class CalibrationTests(unittest.TestCase):
    def test_authored_corpus_has_disjoint_families_and_all_labels(self):
        data = validate_rows(rows())
        self.assertEqual(len(data), 72)
        family_splits = {}
        for row in data:
            family_splits.setdefault(row["family"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in family_splits.values()))

    def test_duplicate_text_and_cross_split_families_rejected(self):
        for field, value in (("text", rows()[0]["text"]), ("family", rows()[0]["family"])):
            data = rows()
            data[-1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_rows(data)

    def test_calibration_saves_real_separate_probes_and_fingerprint(self):
        rt = runtime()
        events = []
        with TemporaryDirectory() as directory:
            target = Path(directory)/"calibration"
            result = rt.calibrate({"layers": [1, 2], "doses": [0, .5]}, target, events.append)
            stored = rt._package(target)
            self.assertEqual(result["counts"], {"train": 24, "probe": 18, "selection": 12, "heldout": 18})
            self.assertIn("probe_pain", stored["vectors"])
            self.assertIn("downstream_joy", stored["vectors"])
            self.assertEqual(result["model_fingerprint_sha256"], rt.info["fingerprint_sha256"])
            self.assertEqual(len(result["dose_validation"]), 2)
            self.assertGreater(result["dose_validation"][1]["mean_relative_delta"], 0)
            self.assertEqual(result["dose_validation"][0]["mean_relative_delta"], 0)
            self.assertTrue(all(not block._forward_hooks for block in rt.blocks))
            with self.assertRaisesRegex(ValueError, "immutable"):
                rt.calibrate({}, target)

    def test_heldout_changes_do_not_change_layer_selection_or_vectors(self):
        rt = runtime()
        with TemporaryDirectory() as directory:
            one, two = Path(directory)/"one", Path(directory)/"two"
            a = rt.calibrate({"layers": [1, 2], "doses": [0]}, one)
            data = rows()
            for row in data:
                if row["split"] == "heldout":
                    rt.tokenizer.corpus[row["text"]] = 30
            b = rt.calibrate({"layers": [1, 2], "doses": [0], "corpus": data}, two)
            self.assertEqual(a["layer"], b["layer"])
            self.assertEqual(a["candidate_layers"], b["candidate_layers"])
            with np.load(one/"vectors.npz") as va, np.load(two/"vectors.npz") as vb:
                for name in va.files:
                    self.assertTrue(np.array_equal(va[name], vb[name]))
            self.assertNotEqual(a["heldout"], b["heldout"])

    def test_fingerprint_and_checksum_mismatch_refused(self):
        rt = runtime()
        with TemporaryDirectory() as directory:
            target = Path(directory)/"calibration"
            rt.calibrate({"layers": [1], "doses": [0]}, target)
            rt.info["fingerprint_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                rt._package(target)
            rt.info["fingerprint_sha256"] = _digest(rt.info["fingerprint"])
            with (target/"vectors.npz").open("ab") as file:
                file.write(b"unexpected")
            with self.assertRaisesRegex(ValueError, "checksum"):
                rt._package(target)

    def test_cancelled_extraction_never_saves_completed_package(self):
        rt = runtime()
        events = []
        with TemporaryDirectory() as directory:
            target = Path(directory)/"calibration"
            with self.assertRaises(Cancelled):
                rt.calibrate({"layers": [1]}, target, events.append, lambda: len(events) >= 2)
            self.assertFalse((target/"calibration.json").exists())
            self.assertTrue(all(not block._forward_hooks for block in rt.blocks))

    def test_zero_norm_direction_is_rejected(self):
        rt = runtime()
        rt.tokenizer.corpus = {key: 30 for key in rt.tokenizer.corpus}
        with TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "degenerate"):
            rt.calibrate({"layers": [1]}, Path(directory)/"calibration")


class RuntimeLoadingTests(unittest.TestCase):
    def test_loaded_version_subclasses_keep_fingerprint_hash_and_v2_checkpoint_roundtrip(self):
        import sys
        import transformers
        from lab.budgets import WeightedBudget
        from lab.checkpoints import capture, restore
        from lab.controller_v2 import RecipeV2Controller
        from lab.session_v2 import build_session, create_environment

        # Use the installed torch version object, which caused the real failure.
        self.assertIsInstance(torch.__version__, TorchVersion)
        torch_version = torch.__version__
        transformers_version = TorchVersion(transformers.__version__)
        bnb_version = TorchVersion("0.50.2")
        for quantized in (False, True):
            with self.subTest(quantized=quantized), TemporaryDirectory() as directory, ExitStack() as stack:
                model, tokenizer = Model(), Tokenizer()
                model.config = SimpleNamespace(model_type="qwen3", hidden_size=4,
                                               max_position_embeddings=2048, _commit_hash=DEFAULT_REVISION)
                profile = {"device": "cpu"}
                if quantized:
                    # Exercise the optional metadata path with a mocked loader
                    # and CUDA APIs; no GPU allocation or kernels are executed.
                    profile = {"device": "cuda", "quantization": "4bit"}
                    model.config.quantization_config = dict(quant_method="bitsandbytes", load_in_4bit=True, bnb_4bit_quant_type="nf4")
                    stack.enter_context(patch.dict(sys.modules, {"bitsandbytes": SimpleNamespace(__version__=bnb_version)}))
                    stack.enter_context(patch("torch.cuda.is_available", return_value=True))
                    stack.enter_context(patch("torch.cuda.empty_cache"))
                    stack.enter_context(patch.object(Runtime, "_gpu_memory", return_value=None))
                    stack.enter_context(patch.object(Runtime, "_numerical_environment", return_value={"device": "mock_cuda"}))
                stack.enter_context(patch("transformers.__version__", transformers_version))
                stack.enter_context(patch("transformers.AutoConfig.from_pretrained", return_value=model.config))
                stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer))
                stack.enter_context(patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model))
                rt = Runtime()
                info = rt.load(profile, directory)
                fingerprint = info["fingerprint"]
                for key in ("torch", "transformers") + (("bitsandbytes",) if quantized else ()):
                    self.assertIs(type(fingerprint[key]), str)
                if not quantized:
                    self.assertIsNone(fingerprint["bitsandbytes"])
                prior_fingerprint = dict(fingerprint, torch=torch_version, transformers=transformers_version,
                                         bitsandbytes=bnb_version if quantized else None)
                self.assertEqual(json.dumps(fingerprint, sort_keys=True), json.dumps(prior_fingerprint, sort_keys=True))
                self.assertEqual(info["fingerprint_sha256"], _digest(prior_fingerprint))

                preview = build_session(dict(recipe_version=2, demonstration="none", task_count=1,
                                             action_budget=8, token_budget=128))
                config = preview["config"]
                environment = create_environment(config)
                effect = RecipeV2Controller(config)
                budget = WeightedBudget(config["action_budget"], config["token_budget"], base_cost=config["base_decision_cost"])
                session = dict(run_id="run-runtime-version", mode="experiment", config=config,
                               calibration_id="cal-runtime-version", messages=preview["messages"],
                               tool_call_format="json", turns=0, external_messages=[])
                calibration = dict(schema_version=2, metadata_sha256="a" * 64, vectors_sha256="b" * 64)
                checkpoint = capture(session, effect, budget, environment, info, calibration)
                restored, restored_effect, restored_budget, restored_env = restore(json.loads(json.dumps(checkpoint)), info, calibration)
                self.assertEqual(restored["messages"], preview["messages"])
                self.assertEqual(restored_effect.snapshot(), effect.snapshot())
                self.assertEqual(restored_budget.snapshot(), budget.snapshot())
                self.assertEqual(restored_env.metrics(), environment.metrics())
                self.assertEqual(checkpoint["identity"]["model"]["fingerprint_sha256"], info["fingerprint_sha256"])
                rt.unload()

    def test_prequantized_nf4_keeps_checkpoint_skip_modules(self):
        import transformers
        stored = {"quant_method": "bitsandbytes", "load_in_4bit": True,
                  "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16",
                  "llm_int8_skip_modules": ["model.visual", "lm_head", "mtp"]}
        config = SimpleNamespace(quantization_config=stored)
        with patch("transformers.BitsAndBytesConfig") as constructor:
            self.assertEqual(_quantization_load_kwargs(config, "4bit", torch.bfloat16, transformers), {})
            constructor.assert_not_called()
        self.assertEqual(config.quantization_config["llm_int8_skip_modules"], ["model.visual", "lm_head", "mtp"])
        with self.assertRaises(ValueError):
            _quantization_load_kwargs(config, "none", torch.bfloat16, transformers)
        for method, kind in (("awq", "nf4"), ("bitsandbytes", "fp4")):
            config.quantization_config = dict(stored, quant_method=method, bnb_4bit_quant_type=kind)
            with self.assertRaises(ValueError):
                _quantization_load_kwargs(config, "4bit", torch.bfloat16, transformers)

    def test_unquantized_nf4_still_constructs_explicit_recipe(self):
        import transformers
        with patch("transformers.BitsAndBytesConfig", return_value="recipe") as constructor:
            self.assertEqual(_quantization_load_kwargs(SimpleNamespace(), "4bit", torch.bfloat16, transformers),
                             {"quantization_config": "recipe"})
            self.assertEqual(constructor.call_args.kwargs["bnb_4bit_compute_dtype"], torch.bfloat16)
            self.assertTrue(constructor.call_args.kwargs["bnb_4bit_use_double_quant"])

    def test_tokenizer_template_and_backend_changes_have_distinct_identities(self):
        tokenizer = Tokenizer()
        first = tokenizer_identity(tokenizer)
        tokenizer.chat_template = "a different template"
        second = tokenizer_identity(tokenizer)
        self.assertNotEqual(first["chat_template_sha256"], second["chat_template_sha256"])
        self.assertEqual(first["backend_sha256"], second["backend_sha256"])
        tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: '{"normalizer":"A","merges":[]}')
        third = tokenizer_identity(tokenizer)
        tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: '{"normalizer":"B","merges":[]}')
        fourth = tokenizer_identity(tokenizer)
        self.assertNotEqual(third["backend_sha256"], fourth["backend_sha256"])
        self.assertEqual(fourth["backend_identity_source"], "fast_backend_serialization")

    def test_local_only_default_and_resolved_fingerprint(self):
        model, tokenizer = Model(), Tokenizer()
        model.config = SimpleNamespace(model_type="qwen3", hidden_size=4,
                                       max_position_embeddings=2048, _commit_hash=DEFAULT_REVISION)
        with TemporaryDirectory() as directory, \
                patch("transformers.AutoConfig.from_pretrained", return_value=model.config) as config_loader, \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as tokenizer_loader, \
                patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as model_loader:
            rt = Runtime()
            info = rt.load({"device": "cpu"}, directory)
            for loader in (config_loader, tokenizer_loader, model_loader):
                self.assertTrue(loader.call_args.kwargs["local_files_only"])
                self.assertFalse(loader.call_args.kwargs["trust_remote_code"])
                self.assertEqual(loader.call_args.kwargs["cache_dir"], str(Path(directory)/"hub"))
            self.assertEqual(model_loader.call_args.kwargs["device_map"], {"": "cpu"})
            self.assertEqual(info["fingerprint"]["revision"], DEFAULT_REVISION)
            self.assertEqual(info["fingerprint"]["tokenizer"], tokenizer_identity(tokenizer))
            self.assertEqual(info["fingerprint"]["model_config_sha256"], _digest(vars(model.config)))
            self.assertIsNone(info["fingerprint"]["quantization_config"])
            self.assertEqual(info["numerical_environment"]["device"], "cpu")
            self.assertIn("deterministic_algorithms", info["numerical_environment"])
            self.assertEqual(len(info["source_sha256"]["runtime.py"]), 64)
            self.assertEqual(info["layer_count"], 4)
            self.assertEqual(rt.unload(), {"status": "unloaded"})
            self.assertIsNone(rt.model)

    def test_wrong_revision_does_not_leave_loaded_state(self):
        model = Model()
        model.config = SimpleNamespace(model_type="qwen3", hidden_size=4, _commit_hash="bad-revision")
        with TemporaryDirectory() as directory, \
                patch("transformers.AutoConfig.from_pretrained", return_value=model.config), \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()), \
                patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model):
            rt = Runtime()
            with self.assertRaisesRegex(RuntimeError, "revision"):
                rt.load({"device": "cpu"}, directory)
            self.assertIsNone(rt.model)
            self.assertEqual(rt.info["status"], "unloaded")

    def test_q4_cpu_and_unsupported_architecture_fail_explicitly(self):
        rt = Runtime()
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                rt.load({"device": "cpu", "quantization": "4bit"}, directory)
            with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(model_type="moe")):
                with self.assertRaisesRegex(ValueError, "Unsupported architecture"):
                    rt.load({"device": "cpu"}, directory)

    def test_stop_before_load_never_contacts_model_hub(self):
        with TemporaryDirectory() as directory, patch("transformers.AutoConfig.from_pretrained") as loader:
            with self.assertRaises(Cancelled):
                Runtime().load({}, directory, should_stop=lambda: True)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
