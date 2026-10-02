"""Frozen-weight local Qwen runtime, calibrated probes and streamed steering.

The hook/extraction/cache design follows LynnColeArt/ai-hotbox's
``impossible_states/engine.py`` (MIT; see UPSTREAM_LICENSE). Only final-position
block activations are captured; no all-layer hidden-state tensors are retained.
Dependencies are imported lazily so the viewer and replay need no ML runtime.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import re
import sys

from .calibration_data import CORPUS_VERSION, rows as default_rows, validate_rows

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
SCHEMA_VERSION = 1


class Cancelled(RuntimeError):
    """Cooperative cancellation; registered hooks are always removed."""


def _check_stop(should_stop):
    if should_stop():
        raise Cancelled("Operation stopped by the user")


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return float(value)


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _config_identity(value):
    """A JSON-compatible snapshot of official model/quantization config data."""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif not isinstance(value, (dict, list, str, int, float, bool, type(None))):
        value = vars(value)
    # Official config dictionaries occasionally contain torch.dtype objects;
    # their stable textual names are sufficient identity, without importing a
    # second serializer or relying on an object address.
    return json.loads(json.dumps(value, sort_keys=True, default=str, allow_nan=False))


def tokenizer_identity(tokenizer):
    """Fingerprint actual tokenization data, not merely the checkpoint name.

    All supported real Qwen tokenizers are fast tokenizers. A get_vocab fallback
    is retained for CPU fixtures and future slow adapters and is marked clearly.
    The serialized fast backend includes merges, normalization and added-token
    behavior. Template and decode controls have separate readable identities.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        encoded = backend.to_str().encode("utf-8")
        source = "fast_backend_serialization"
        vocabulary_hash = hashlib.sha256(encoded).hexdigest()
    elif hasattr(tokenizer, "get_vocab"):
        source = "vocabulary_only_slow_adapter_requires_validation"
        vocabulary_hash = _digest(tokenizer.get_vocab())
    else:
        raise ValueError("Tokenizer does not expose auditable vocabulary data")
    return {"class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
            "backend_identity_source": source, "backend_sha256": vocabulary_hash,
            "chat_template_sha256": _digest(getattr(tokenizer, "chat_template", None)),
            "special_token_ids": sorted(set(getattr(tokenizer, "all_special_ids", []))),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "padding_side": getattr(tokenizer, "padding_side", None),
            "truncation_side": getattr(tokenizer, "truncation_side", None),
            "clean_up_tokenization_spaces": getattr(tokenizer, "clean_up_tokenization_spaces", None)}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def normalized_control(value, phase):
    """Coefficients are already dosed by the caller; ``level`` is telemetry.

    ``enabled`` reports the caller's aux-pulse gate; it must not gate already
    composed coefficients, because baseline sliders can remain active while
    aux delivery is disabled. ``measurement_only`` gates every edit.
    phase is the generation phase before sampling: the closing think marker is
    generated under the reasoning phase, and its following token under output.
    """
    if not isinstance(value, dict):
        raise ValueError("control must return an object")
    result = {k: _number(value.get(k, default), k, low, high)
              for k, default, low, high in (("pain", 0, -4, 4), ("joy", 0, -4, 4),
                                            ("suppression", 0, 0, 1), ("random_gain", 0, -4, 4), ("level", 0, 0, 1))}
    result["enabled"] = _boolean(value.get("enabled", True), "enabled")
    result["measurement_only"] = _boolean(value.get("measurement_only", False), "measurement_only")
    scope = value.get("phase_scope", "all")
    if scope not in ("all", "reasoning", "output"):
        raise ValueError("phase_scope must be all, reasoning, or output")
    direction = value.get("direction", "pain")
    if direction not in ("pain", "random"):
        raise ValueError("direction must be pain or random")
    result.update(phase_scope=scope, direction=direction, phase=phase)
    result["applied"] = bool(not result["measurement_only"]
                             and scope in ("all", phase))
    result["effective"] = {k: result[k] if result["applied"] else 0.0
                           for k in ("pain", "joy", "suppression", "random_gain")}
    return result


def split_reasoning(raw_text, thinking=False):
    """Separate externally generated Qwen reasoning without executing its text.

    The native thinking template may already contain the opening marker. An
    unfinished thinking segment stays reasoning, never executable output.
    """
    if not isinstance(raw_text, str):
        raise ValueError("raw_text must be text")
    if "</think>" in raw_text:
        reasoning, content = raw_text.split("</think>", 1)
        reasoning = reasoning.split("<think>", 1)[-1]
        return reasoning.strip(), content.strip()
    if thinking or "<think>" in raw_text:
        return raw_text.split("<think>", 1)[-1].strip(), ""
    return "", raw_text.strip()


def prepare_messages(messages, reasoning_history="template"):
    if reasoning_history not in ("template", "drop"):
        raise ValueError("reasoning_history must be template or drop")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    result = deepcopy(messages)
    for message in result:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError("invalid conversation message")
        if reasoning_history == "drop" and message["role"] == "assistant":
            message.pop("reasoning_content", None)
            if isinstance(message.get("content"), str) and "</think>" in message["content"]:
                message["content"] = split_reasoning(message["content"])[1]
    return result


def resolve_blocks(model, adapter):
    """Explicit dense Qwen adapters; never guess an arbitrary model layout."""
    paths = {
        "qwen3": ("model.layers",),
        "qwen3_5": ("model.language_model.layers", "model.language_model.model.layers",
                     "language_model.model.layers", "model.layers"),
    }
    if adapter not in paths:
        raise ValueError(f"Unsupported model architecture {adapter!r}; use a Qwen3 or Qwen3.5 dense model")
    for path in paths[adapter]:
        value = model
        for part in path.split("."):
            value = getattr(value, part, None)
            if value is None:
                break
        if value is not None and hasattr(value, "__len__") and len(value) >= 2:
            return value, path
    raise ValueError(f"Installed {adapter} model does not expose the expected decoder blocks")


def _hidden(output):
    import torch
    hidden = output[0] if isinstance(output, tuple) and output else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError("Expected a decoder block tensor [1, tokens, hidden]")
    return hidden


def _unit(array, name):
    import numpy as np
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    norm = float(np.linalg.norm(array))
    if norm < 1e-8:
        raise ValueError(f"{name} is degenerate; choose another corpus or layer")
    return array / norm


def _quantization_load_kwargs(config, quantization, dtype, transformers):
    """Keep a prequantized checkpoint's module inventory and stored NF4 recipe."""
    stored = _config_identity(getattr(config, "quantization_config", None))
    if stored is not None:
        if (quantization != "4bit" or stored.get("quant_method") != "bitsandbytes"
                or not stored.get("load_in_4bit", stored.get("_load_in_4bit", False))
                or stored.get("bnb_4bit_quant_type") != "nf4"):
            raise ValueError("Prequantized checkpoints require a matching 4bit NF4 profile; other formats are unsupported")
        # Passing a newly created BitsAndBytesConfig can override load attributes
        # or skipped modules. The checkpoint config is authoritative on reload.
        return {}
    if quantization == "4bit":
        return {"quantization_config": transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True)}
    return {}


class Runtime:
    def __init__(self):
        self.model = self.tokenizer = self.blocks = None
        self.info = {"status": "unloaded"}
        self.device = None
        self._logits_kwarg = None
        self._calibration_cache = {}

    def _require_loaded(self):
        if self.model is None:
            raise RuntimeError("Load a model before calibrating or generating")

    def load(self, profile, cache_dir, emit=lambda event: None, should_stop=lambda: False):
        import torch
        import transformers
        if not isinstance(profile, dict):
            raise ValueError("model profile must be an object")
        _check_stop(should_stop)
        model_id = profile.get("model_id", profile.get("model", DEFAULT_MODEL))
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a nonempty string")
        revision = profile.get("revision") or (DEFAULT_REVISION if model_id == DEFAULT_MODEL else "main")
        if not isinstance(revision, str):
            raise ValueError("revision must be a string")
        quantization = profile.get("quantization", "none")
        if quantization == "nf4":
            quantization = "4bit"
        if quantization not in ("none", "4bit"):
            raise ValueError("quantization must be none or 4bit (NF4)")
        device = profile.get("device", "cuda")
        if device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or explicit cpu; offloading is not automatic")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; install a compatible CUDA PyTorch runtime or explicitly choose CPU")
        if quantization == "4bit" and device != "cuda":
            raise ValueError("The 4-bit adapter requires CUDA; CPU offloading is not implemented")
        dtype_name = profile.get("dtype", "bfloat16" if device == "cuda" else "float32")
        if dtype_name not in ("bfloat16", "float16", "float32"):
            raise ValueError("dtype must be bfloat16, float16, or float32")
        dtype = getattr(torch, dtype_name)
        allow_download = _boolean(profile.get("allow_download", False), "allow_download")
        attention = profile.get("attention_implementation", "sdpa")
        if attention not in ("sdpa", "eager", "flash_attention_2"):
            raise ValueError("unsupported attention implementation")
        torch.set_num_threads(_integer(profile.get("cpu_threads", 8), "cpu_threads", 1, 256))
        torch.backends.cuda.matmul.allow_tf32 = False
        cache_path = Path(cache_dir).expanduser().resolve() / "hub"
        cache_path.mkdir(parents=True, exist_ok=True)
        common = {"revision": revision, "cache_dir": str(cache_path),
                  "local_files_only": not allow_download, "trust_remote_code": False}
        emit({"type": "status", "status": "loading", "model": model_id,
              "message": "Loading local frozen weights"})
        try:
            config = transformers.AutoConfig.from_pretrained(model_id, **common)
        except (KeyError, ValueError) as exc:
            raise RuntimeError("This model config is unsupported by the installed Transformers version. "
                               "Qwen3.5/3.8 profiles need a compatible official Transformers adapter; "
                               "the lab will not silently load a different architecture.") from exc
        kind = getattr(config, "model_type", "")
        if kind == "qwen3":
            adapter, model_class = "qwen3", transformers.AutoModelForCausalLM
        elif kind in ("qwen3_5", "qwen3_5_text"):
            adapter = "qwen3_5"
            class_name = "Qwen3_5ForConditionalGeneration" if kind == "qwen3_5" else "Qwen3_5ForCausalLM"
            model_class = getattr(transformers, class_name, None)
            if model_class is None:
                raise RuntimeError(f"Installed Transformers {transformers.__version__} lacks {class_name}; "
                                   "install the model's documented compatible version explicitly")
        else:
            raise ValueError(f"Unsupported architecture {kind!r}; only dense Qwen3 and Qwen3.5 adapters are provided")
        kwargs = dict(common, dtype=dtype, device_map={"": device}, attn_implementation=attention)
        if quantization == "4bit":
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("4-bit loading requires optional bitsandbytes. Install it in the lab runtime; "
                                   "the lab never falls back to a larger unquantized model.") from exc
        kwargs.update(_quantization_load_kwargs(config, quantization, dtype, transformers))
        if profile.get("max_memory_gib") is not None:
            maximum = _number(profile["max_memory_gib"], "max_memory_gib", 1, 1024)
            kwargs["max_memory"] = {0 if device == "cuda" else "cpu": int(maximum * 1024**3)}
        self.unload()
        model = tokenizer = None
        try:
            _check_stop(should_stop)
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **common)
            model = model_class.from_pretrained(model_id, **kwargs).eval().requires_grad_(False)
            _check_stop(should_stop)
            blocks, block_path = resolve_blocks(model, adapter)
            if device == "cuda" and any(p.device.type != "cuda" for p in model.parameters()):
                raise RuntimeError("The model is not fully GPU-resident; loading refused rather than silently offloading")
            if profile.get("max_memory_gib") is not None and device == "cuda":
                if torch.cuda.memory_allocated() > float(profile["max_memory_gib"]) * 1024**3:
                    raise RuntimeError("Loaded GPU allocations exceed max_memory_gib; the explicit residency limit was not met")
            text_config = getattr(config, "text_config", config)
            resolved = getattr(model.config, "_commit_hash", None) or getattr(config, "_commit_hash", None)
            if re.fullmatch(r"[0-9a-f]{40}", revision) and resolved != revision:
                raise RuntimeError("Resolved model revision does not match the requested pinned revision")
            model_config = _config_identity(model.config)
            quantization_config = _config_identity(getattr(model.config, "quantization_config", None))
            if quantization_config is None and quantization == "4bit":
                quantization_config = _config_identity(kwargs.get("quantization_config"))
            tokenizer_data = tokenizer_identity(tokenizer)
            fingerprint = {"model_id": model_id, "revision": resolved or revision,
                           "architecture": kind, "adapter": adapter, "dtype": dtype_name,
                           "quantization": quantization, "layers": len(blocks),
                           "hidden_size": int(text_config.hidden_size), "transformers": transformers.__version__,
                           "torch": torch.__version__, "attention_implementation": attention,
                           "model_config_sha256": _digest(model_config),
                           "tokenizer": tokenizer_data, "quantization_config": quantization_config,
                           "bitsandbytes": getattr(sys.modules.get("bitsandbytes"), "__version__", None)
                               if quantization_config is not None else None}
            tool_call_format = "qwen_xml" if adapter == "qwen3_5" else "json"
            if adapter == "qwen3_5":
                # Preserve existing Qwen3 calibration identities. New adapters
                # additionally record the native grammar selected by the worker.
                fingerprint["tool_call_format"] = tool_call_format
            self.model, self.tokenizer, self.blocks, self.device = model, tokenizer, blocks, device
            self._configure_forward()
            self.info = {"status": "loaded", "fingerprint": fingerprint,
                         "fingerprint_sha256": _digest(fingerprint), "model_id": model_id,
                         "revision": resolved or revision, "device": device, "adapter": adapter,
                         "tool_call_format": tool_call_format,
                         "layer_count": len(blocks), "hidden_size": int(text_config.hidden_size),
                         "quantization": quantization, "dtype": dtype_name,
                         "block_path": block_path, "cache_policy": "rebuild_each_turn",
                         "profile_validation": "reference" if (model_id == DEFAULT_MODEL and resolved == DEFAULT_REVISION
                                    and quantization == "none" and dtype_name == "bfloat16" and device == "cuda") else "requires_local_validation",
                         "max_position_embeddings": getattr(text_config, "max_position_embeddings", None),
                         "gpu_memory": self._gpu_memory(), "numerical_environment": self._numerical_environment(),
                         "model_config": model_config,
                         "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                           for name in ("runtime.py", "calibration_data.py")}}
            emit({"type": "model_loaded", **self.info})
            return deepcopy(self.info)
        except BaseException:
            self.unload()
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def _configure_forward(self):
        params = inspect.signature(self.model.forward).parameters
        self._logits_kwarg = "logits_to_keep" if "logits_to_keep" in params else (
            "num_logits_to_keep" if "num_logits_to_keep" in params else None)

    def _forward(self, **kwargs):
        if self._logits_kwarg:
            kwargs[self._logits_kwarg] = 1
        return self.model(**kwargs)

    def _gpu_memory(self):
        import torch
        if self.device != "cuda" or not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {"name": torch.cuda.get_device_name(), "free_bytes": free, "total_bytes": total,
                "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()}

    def _numerical_environment(self):
        import torch
        environment = {"python": platform.python_version(), "platform": platform.platform(),
                       "device": self.device, "cuda_runtime": torch.version.cuda,
                       "cudnn_version": torch.backends.cudnn.version(),
                       "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                       "cudnn_deterministic": torch.backends.cudnn.deterministic,
                       "cudnn_benchmark": torch.backends.cudnn.benchmark,
                       "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                       "float32_matmul_precision": torch.get_float32_matmul_precision(),
                       "cpu_threads": torch.get_num_threads()}
        if self.device == "cuda":
            properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            environment["gpu"] = {"name": properties.name, "compute_capability": [properties.major, properties.minor],
                                  "total_memory": properties.total_memory, "multiprocessor_count": properties.multi_processor_count}
        return environment

    def unload(self):
        self.model = self.tokenizer = self.blocks = None
        self.device = None
        self._calibration_cache.clear()
        self.info = {"status": "unloaded"}
        gc.collect()
        # Do not import the GPU stack merely to open a replay-only viewer.
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return dict(self.info)

    def _extract(self, data, layers, emit, should_stop, max_input_tokens):
        import numpy as np
        import torch
        activations = np.empty((len(data), len(layers), self.info["hidden_size"]), dtype=np.float32)
        current, handles = [0], []
        try:
            for j, layer in enumerate(layers):
                def capture(module, inputs, output, j=j):
                    activations[current[0], j] = _hidden(output)[0, -1].detach().float().cpu().numpy()
                handles.append(self.blocks[layer].register_forward_hook(capture))
            with torch.inference_mode():
                for i, row in enumerate(data):
                    _check_stop(should_stop)
                    current[0] = i
                    ids = self.tokenizer(row["text"], return_tensors="pt").input_ids.to(self.device)
                    if ids.shape[1] > max_input_tokens:
                        raise ValueError(f"Corpus row {row['id']} exceeds max_input_tokens; no text is silently truncated")
                    output = self._forward(input_ids=ids, use_cache=False)
                    del output
                    emit({"type": "calibration_progress", "stage": "extract", "completed": i + 1,
                          "total": len(data), "row_id": row["id"]})
        finally:
            for handle in handles:
                handle.remove()
        return activations

    def calibrate(self, config, out_dir, emit=lambda event: None, should_stop=lambda: False):
        import numpy as np
        self._require_loaded()
        _check_stop(should_stop)
        name = config.get("name", "Pain and joy calibration")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
            raise ValueError("Calibration name must contain 1–100 characters")
        data = validate_rows(deepcopy(config.get("corpus", default_rows())))
        maximum = _integer(config.get("max_input_tokens", 256), "max_input_tokens", 16, 4096)
        count = len(self.blocks)
        default_layers = sorted({min(count - 2, max(0, int(count * fraction))) for fraction in (.35, .55, .7)})
        layers = config.get("layers", default_layers)
        if not isinstance(layers, list) or not layers or len(layers) > 12:
            raise ValueError("layers must contain 1 to 12 block indices")
        layers = sorted({_integer(layer, "layer", 0, count - 2) for layer in layers})
        downstream_layer = _integer(config.get("downstream_layer", count - 1), "downstream_layer", max(layers) + 1, count - 1)
        doses = config.get("doses", [0, .25, .5, 1.0])
        if not isinstance(doses, list) or not doses or len(doses) > 12:
            raise ValueError("doses must contain 1 to 12 nonnegative values")
        doses = sorted({_number(dose, "dose", 0, 4) for dose in doses} | {0.0})
        out_dir = Path(out_dir)
        if out_dir.exists() and any(out_dir.iterdir()):
            raise ValueError("Calibration output must be a new or empty directory; existing packages are immutable")
        out_dir.mkdir(parents=True, exist_ok=True)
        all_layers = layers + [downstream_layer]
        activations = self._extract(data, all_layers, emit, should_stop, maximum)

        def subset(split, label, j):
            return activations[[i for i, row in enumerate(data) if row["split"] == split and row["label"] == label], j]

        def means(split, j):
            return {label: subset(split, label, j).mean(axis=0) for label in ("pain", "joy", "neutral")}

        def make_probe(j):
            m = means("probe", j)
            result = {"center": m["neutral"], "scale": max(float(np.linalg.norm(subset("probe", "neutral", j), axis=1).mean()) / 4, 1e-8)}
            for label in ("pain", "joy"):
                result[label] = _unit(m[label] - m["neutral"], f"{label} probe")
                result[f"{label}_threshold"] = float(((m[label] + m["neutral"]) / 2) @ result[label])
            return result

        def report_probe(j, probe, split):
            report = {}
            for label in ("pain", "joy"):
                positive = subset(split, label, j) @ probe[label]
                negative = subset(split, "neutral", j) @ probe[label]
                threshold = probe[f"{label}_threshold"]
                # Pairwise AUC handles ties explicitly; balanced accuracy uses only probe-fit threshold.
                auc = float(((positive[:, None] > negative).mean() + .5 * (positive[:, None] == negative).mean()))
                accuracy = float(((positive > threshold).mean() + (negative <= threshold).mean()) / 2)
                report[label] = {"auc": auc, "balanced_accuracy": accuracy,
                                 "mean_positive_projection": float(positive.mean()),
                                 "mean_neutral_projection": float(negative.mean()),
                                 "positive_count": len(positive), "neutral_count": len(negative)}
            return report

        candidates, probes = [], []
        for j, layer in enumerate(all_layers):
            probe = make_probe(j)
            probes.append(probe)
            if layer in layers:
                report = report_probe(j, probe, "selection")
                candidates.append({"layer": layer, "selection": report,
                                   "score": sum(report[label]["auc"] for label in ("pain", "joy")) / 2})
        # Fixed tie break: prefer the earliest candidate block.
        selected = max(candidates, key=lambda value: (value["score"], -value["layer"]))
        layer, j = selected["layer"], all_layers.index(selected["layer"])
        train = means("train", j)
        pain = _unit(train["pain"] - train["neutral"], "pain direction")
        joy_raw = _unit(train["joy"] - train["neutral"], "joy direction")
        joy = _unit(joy_raw - float(joy_raw @ pain) * pain, "orthogonal joy direction")
        scale = float(np.linalg.norm(subset("train", "neutral", j), axis=1).mean()) / 4
        if not math.isfinite(scale) or scale < 1e-8:
            raise ValueError("Neutral activation scale is degenerate")
        random = _unit(np.random.default_rng(20261001).normal(size=pain.shape), "random control direction")
        vectors = {"pain": pain, "joy": joy, "joy_raw": joy_raw, "neutral": train["neutral"],
                   "scale": np.asarray(scale, dtype=np.float32), "random": random}
        for prefix, probe in (("probe", probes[j]), ("downstream", probes[-1])):
            vectors.update({f"{prefix}_{key}": np.asarray(probe[key], dtype=np.float32)
                            for key in ("pain", "joy", "center", "scale")})
        metadata = {"schema_version": SCHEMA_VERSION, "kind": "split_calibration", "name": name.strip(),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "model_fingerprint": deepcopy(self.info["fingerprint"]),
                    "model_fingerprint_sha256": self.info["fingerprint_sha256"],
                    "numerical_environment": deepcopy(self.info.get("numerical_environment")),
                    "source_sha256": deepcopy(self.info.get("source_sha256")),
                    "layer": layer, "downstream_layer": downstream_layer,
                    "corpus_version": CORPUS_VERSION if "corpus" not in config else "custom",
                    "corpus_sha256": _digest(data), "rows": data, "candidate_layers": candidates,
                    "selection_rule": "Maximum mean pain/joy selection AUC; ties choose earlier block",
                    "counts": {split: sum(row["split"] == split for row in data)
                               for split in ("train", "probe", "selection", "heldout")},
                    "heldout": {"edit_layer": report_probe(j, probes[j], "heldout"),
                                "downstream_layer": report_probe(len(all_layers)-1, probes[-1], "heldout")},
                    "direction_cosine_before_orthogonalization": float(pain @ joy_raw),
                    "probe_training": "Independent probe families; never used to extract intervention vectors",
                    "context_scope": "Raw authored sentences, final token; transfer to generated conversation is unvalidated",
                    "interpretation": "Concept-association probes, not validated measures of felt emotion. Small convenience corpus.",
                    "dose_validation": []}
        package = {"metadata": metadata, "vectors": vectors}
        selection_rows = [row for row in data if row["split"] == "selection" and row["label"] == "neutral"]
        metadata["dose_validation"] = self._validate_doses(package, selection_rows, doses, emit, should_stop)
        acceptable = [row["dose"] for row in metadata["dose_validation"]
                      if row["mean_next_token_kl"] <= .5 and row["mean_relative_delta"] <= .3]
        metadata["selected_dose"] = max(acceptable, default=0)
        metadata["dose_selection_rule"] = "Largest tested combined dose with selection next-token KL <= 0.5 nats and mean relative edit <= 0.3; not a validated efficacy or safety threshold"
        _check_stop(should_stop)
        np.savez(out_dir / "vectors.npz", **vectors)
        metadata["vectors_sha256"] = hashlib.sha256((out_dir / "vectors.npz").read_bytes()).hexdigest()
        _write_json(out_dir / "calibration.json", metadata)
        emit({"type": "calibration_complete", "path": str(out_dir), "layer": layer,
              "selected_dose": metadata["selected_dose"], "heldout": metadata["heldout"]})
        return metadata

    def _validate_doses(self, package, rows, doses, emit, should_stop):
        import torch
        records, baselines = [], []
        with torch.inference_mode():
            for row in rows:
                _check_stop(should_stop)
                ids = self.tokenizer(row["text"], return_tensors="pt").input_ids.to(self.device)
                baseline = self._forward(input_ids=ids, use_cache=False).logits[0, -1].float().log_softmax(-1)
                baselines.append((ids, baseline.detach()))
            for dose in doses:
                kls, changes = [], []
                for ids, baseline in baselines:
                    _check_stop(should_stop)
                    # Explicit recipe: joy=dose, suppression=min(dose, 1), no challenge.
                    state = {"dose": normalized_control({"joy": dose, "suppression": min(dose, 1)}, "output")}
                    with self._hooks(package, state):
                        actual = self._forward(input_ids=ids, use_cache=False).logits[0, -1].float().log_softmax(-1)
                    kls.append(max(0.0, float((baseline.exp() * (baseline - actual)).sum())))
                    changes.append(state["measurements"]["relative_delta"])
                record = {"dose": dose, "recipe": {"joy": dose, "suppression": min(dose, 1)},
                          "mean_next_token_kl": sum(kls)/len(kls),
                          "mean_relative_delta": sum(changes)/len(changes), "examples": len(rows)}
                records.append(record)
                emit({"type": "calibration_progress", "stage": "dose_validation", **record})
        return records

    def _package(self, calibration_dir):
        import numpy as np
        directory = Path(calibration_dir).resolve()
        manifest_path = directory / "calibration.json"
        vector_path = directory / "vectors.npz"
        key = (str(directory), manifest_path.stat().st_mtime_ns, vector_path.stat().st_mtime_ns)
        if key in self._calibration_cache:
            return self._calibration_cache[key]
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported calibration schema")
        if metadata.get("model_fingerprint_sha256") != self.info["fingerprint_sha256"]:
            raise ValueError("Calibration fingerprint differs from the loaded model/runtime; recalibrate this exact profile")
        if metadata.get("model_fingerprint") != self.info["fingerprint"]:
            raise ValueError("Calibration fingerprint payload is inconsistent")
        if hashlib.sha256(vector_path.read_bytes()).hexdigest() != metadata.get("vectors_sha256"):
            raise ValueError("Calibration vector checksum mismatch")
        _integer(metadata.get("layer"), "calibration layer", 0, len(self.blocks)-1)
        downstream = metadata.get("downstream_layer")
        if downstream is not None:
            _integer(downstream, "downstream layer", metadata["layer"]+1, len(self.blocks)-1)
        with np.load(vector_path, allow_pickle=False) as stored:
            vectors = {name: stored[name].copy() for name in stored.files}
        required = {"pain", "joy", "neutral", "scale", "random", "probe_pain", "probe_joy", "probe_center", "probe_scale"}
        if downstream is not None:
            required.update({"downstream_pain", "downstream_joy", "downstream_center", "downstream_scale"})
        if not required <= vectors.keys():
            raise ValueError("Calibration package lacks required vectors")
        for name, value in vectors.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite calibration vector {name}")
            if name.endswith("scale") or name == "scale":
                if value.size != 1 or float(value) <= 1e-8:
                    raise ValueError(f"Invalid calibration scale {name}")
            elif value.shape != (self.info["hidden_size"],):
                raise ValueError(f"Wrong hidden dimension for {name}")
        for name in ("pain", "joy", "random", "probe_pain", "probe_joy"):
            vectors[name] = _unit(vectors[name], name)
        self._calibration_cache = {key: {"metadata": metadata, "vectors": vectors}}
        return self._calibration_cache[key]

    @contextmanager
    def _hooks(self, package, state):
        import torch
        if package is None:
            state["measurements"] = {"pre": None, "post": None, "downstream": None, "relative_delta": 0.0}
            yield
            return
        meta, vectors = package["metadata"], package["vectors"]
        cached, handles = {}, []

        def tensors(device):
            if device not in cached:
                cached[device] = {name: torch.as_tensor(value, device=device, dtype=torch.float32)
                                  for name, value in vectors.items()}
            return cached[device]

        def measure(hidden, v, prefix):
            centered = hidden - v[f"{prefix}_center"]
            scale = v[f"{prefix}_scale"]
            return {label: float(centered @ v[f"{prefix}_{label}"] / scale) for label in ("pain", "joy")}

        def edit(module, inputs, output):
            hidden = _hidden(output)
            v = tensors(hidden.device)
            before = hidden[0, -1].float()
            coefficients = state["dose"]["effective"]
            actual = before
            changed = None
            if any(coefficients.values()):
                direction = v["random"] if state["dose"]["direction"] == "random" else v["pain"]
                challenged = before + coefficients["pain"] * v["scale"] * direction
                after = challenged - coefficients["suppression"] * (challenged @ direction) * direction
                after = after + coefficients["joy"] * v["scale"] * (v["random"] if state["dose"]["direction"] == "random" else v["joy"])
                after = after + coefficients["random_gain"] * v["scale"] * v["random"]
                changed = hidden.clone()
                changed[0, -1] = after.to(hidden.dtype)
                actual = changed[0, -1].float()
            state["measurements"] = {"pre": measure(before, v, "probe"), "post": measure(actual, v, "probe"),
                                     "downstream": None,
                                     "relative_delta": float((actual-before).norm() / before.norm().clamp_min(1e-12))}
            if changed is not None:
                return (changed,) + output[1:] if isinstance(output, tuple) else changed
            return None  # Preserve bitwise baseline; read-only probes never replace model outputs.

        def downstream(module, inputs, output):
            hidden = _hidden(output)[0, -1].float()
            state["measurements"]["downstream"] = measure(hidden, tensors(hidden.device), "downstream")

        try:
            handles.append(self.blocks[meta["layer"]].register_forward_hook(edit))
            if meta.get("downstream_layer") is not None:
                handles.append(self.blocks[meta["downstream_layer"]].register_forward_hook(downstream))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def generate(self, messages, tools, config, calibration_dir, control=lambda: {},
                 emit=lambda event: None, should_stop=lambda: False):
        import torch
        self._require_loaded()
        limit = _integer(config.get("max_new_tokens", 512), "max_new_tokens", 1, 32768)
        max_context = _integer(config.get("max_context_tokens", 8192), "max_context_tokens", 128, 262144)
        temperature = _number(config.get("temperature", .6), "temperature", 0, 2)
        top_p = _number(config.get("top_p", .95), "top_p", .001, 1)
        top_k = _integer(config.get("top_k", 20), "top_k", 0, 1000)
        seed = _integer(config.get("seed", 20261001), "seed", 0, 2**63-1)
        thinking = _boolean(config.get("thinking", False), "thinking")
        history = config.get("reasoning_history", "template")
        prepared = prepare_messages(messages, history)
        if not isinstance(tools, list):
            raise ValueError("tools must be a list")
        _check_stop(should_stop)
        formatted = self.tokenizer.apply_chat_template(prepared, tools=tools or None,
                    tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        input_ids = self.tokenizer(formatted, return_tensors="pt").input_ids.to(self.device)
        prompt_tokens = int(input_ids.shape[1])
        model_context = self.info.get("max_position_embeddings")
        if isinstance(model_context, int):
            max_context = min(max_context, model_context)
        if prompt_tokens + limit > max_context:
            raise ValueError(f"Conversation ({prompt_tokens} tokens) plus output allowance ({limit}) exceeds "
                             f"context budget ({max_context}); shorten the conversation or raise the explicit budget")
        package = self._package(calibration_dir) if calibration_dir is not None else None
        eos = self.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        if self.tokenizer.eos_token_id is not None:
            eos_ids.add(self.tokenizer.eos_token_id)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        emitted, raw, streamed = [], "", ""
        cache, phase = None, "reasoning" if thinking else "output"
        reasoning_tokens, finish = 0, "length"
        state = {}
        try:
            with torch.inference_mode(), self._hooks(package, state):
                for index in range(limit):
                    if should_stop():
                        finish = "stopped"
                        break
                    state["dose"] = normalized_control(control(), phase)
                    if package is None and any(state["dose"]["effective"].values()):
                        raise ValueError("A calibration package is required for nonzero effects")
                    output = self._forward(input_ids=input_ids, past_key_values=cache, use_cache=True)
                    cache = output.past_key_values
                    logits = output.logits[0, -1].float()
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("Model emitted nonfinite logits; generation stopped")
                    if temperature == 0:
                        token = int(logits.argmax())
                    else:
                        logits = logits / temperature
                        values, indices = logits.sort(descending=True)
                        if top_k:
                            values, indices = values[:top_k], indices[:top_k]
                        remove = values.softmax(-1).cumsum(-1) > top_p
                        remove[1:] = remove[:-1].clone()
                        remove[0] = False
                        values[remove] = -float("inf")
                        chosen = torch.multinomial(values.softmax(-1), 1, generator=generator)
                        token = int(indices[chosen].item())
                    del output, logits
                    emitted.append(token)
                    if phase == "reasoning":
                        reasoning_tokens += 1
                    decoded_ids = emitted[:-1] if token in eos_ids else emitted
                    raw = self.tokenizer.decode(decoded_ids, skip_special_tokens=False)
                    # Decode the whole prefix: byte-fallback tokens can revise a
                    # trailing replacement character. full_text is authoritative.
                    text = raw[len(streamed):] if raw.startswith(streamed) else ""
                    replace = not raw.startswith(streamed)
                    streamed = raw
                    event = {"type": "token", "text": text, "full_text": raw,
                             "replace_text": replace, "token_id": token, "index": index,
                             "phase": phase, "is_eos": token in eos_ids,
                             "measurements": deepcopy(state["measurements"]),
                             "dose": deepcopy(state["dose"]),
                             "measurement_alignment": "final input position producing this sampled output token"}
                    # The caller owns all dose clocks and advances exactly once here.
                    emit(event)
                    if token in eos_ids:
                        finish = "eos"
                        break
                    if "</think>" in raw:
                        phase = "output"
                    elif "<think>" in raw and not thinking:
                        phase = "reasoning"
                    input_ids = torch.tensor([[token]], device=self.device)
        finally:
            del cache
        reasoning, content = split_reasoning(raw, thinking)
        return {"raw_text": raw, "reasoning": reasoning, "content": content,
                "token_ids": emitted, "reasoning_tokens": reasoning_tokens,
                "output_tokens": len(emitted)-reasoning_tokens, "generated_tokens": len(emitted),
                "truncated": finish == "length", "finish_reason": finish,
                "prompt_tokens": prompt_tokens, "prompt_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
                "thinking": thinking, "reasoning_history": history, "cache_policy": "rebuild_each_turn",
                "tool_call_format": self.info.get("tool_call_format", "json"),
                "sampling": {"temperature": temperature, "top_p": top_p, "top_k": top_k, "seed": seed},
                "intervention_scope": "last_position",
                "model_fingerprint_sha256": self.info["fingerprint_sha256"],
                "calibration_sha256": package["metadata"].get("vectors_sha256") if package else None}

    def import_legacy_calibration(self, source_dir, out_dir):
        """Explicit import of this repository's original pilot; no heldout probe claims."""
        import numpy as np
        self._require_loaded()
        source_dir, out_dir = Path(source_dir), Path(out_dir)
        manifest = json.loads((source_dir / "manifest.json").read_text())
        fingerprint = self.info["fingerprint"]
        if (manifest.get("model") != fingerprint["model_id"] or manifest.get("revision") != fingerprint["revision"]
                or fingerprint["quantization"] != "none" or fingerprint["dtype"] != "bfloat16"
                or fingerprint["adapter"] != "qwen3"):
            raise ValueError("Legacy vectors only match their original unquantized BF16 Qwen3 profile")
        if out_dir.exists() and any(out_dir.iterdir()):
            raise ValueError("Calibration output must be new or empty")
        with np.load(source_dir / "vectors.npz", allow_pickle=False) as stored:
            pain = _unit(stored["pain"], "pain")
            raw = _unit(stored["joy_raw"], "joy")
            joy = _unit(raw - (raw @ pain)*pain, "orthogonal joy")
            center, scale = stored["neutral"].copy(), float(stored["scale"])
        vectors = {"pain": pain, "joy": joy, "joy_raw": raw, "neutral": center,
                   "scale": np.asarray(scale, dtype=np.float32), "probe_pain": pain, "probe_joy": joy,
                   "probe_center": center, "probe_scale": np.asarray(scale, dtype=np.float32),
                   "random": _unit(np.random.default_rng(20261001).normal(size=pain.shape), "random")}
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "vectors.npz", **vectors)
        metadata = {"schema_version": SCHEMA_VERSION, "kind": "legacy_pilot_import",
                    "model_fingerprint": fingerprint, "model_fingerprint_sha256": self.info["fingerprint_sha256"],
                    "layer": manifest["layer"], "downstream_layer": None,
                    "source": str(source_dir), "heldout": None, "selected_dose": None,
                    "interpretation": "Legacy projections reuse the intervention directions. They are not independent probes; no downstream calibration or split validation is available.",
                    "vectors_sha256": hashlib.sha256((out_dir / "vectors.npz").read_bytes()).hexdigest()}
        _write_json(out_dir / "calibration.json", metadata)
        return metadata
