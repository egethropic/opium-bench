"""Frozen-weight local Qwen runtime, calibrated probes and streamed steering.

The hook/extraction/cache design follows LynnColeArt/ai-hotbox's
``impossible_states/engine.py`` (MIT-style; see UPSTREAM_LICENSE). Historical
extraction captures final positions; opt-in research extraction also supports
masked mean and declared-span pooling without retaining all-layer token tensors.
Dependencies are imported lazily so the viewer and replay need no ML runtime.
"""
from contextlib import contextmanager
from copy import deepcopy
from .storage import guarded_npz, guarded_copyfile
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
    from .storage import atomic_json
    atomic_json(path, value, ensure_ascii=True)


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


def _kernel_callable_identity(function):
    """Describe the callable selected by Transformers' frozen fallback wrapper."""
    seen = set()
    while inspect.isfunction(function) and id(function) not in seen:
        seen.add(id(function))
        implementation = inspect.getclosurevars(function).nonlocals.get("implementation")
        if not callable(implementation):
            break
        function = implementation
    function = inspect.unwrap(function)
    source = inspect.getsourcefile(function)
    return {"callable": function.__module__ + "." + function.__qualname__,
            "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None}


def _configure_qwen35_kernels(enabled):
    """Select local kernels explicitly; never fetch Hub kernels or mutate FLA.

    Transformers 5.15.1 expects ``recurrent_gated_delta_rule`` while fla-core
    0.5.2 exports ``fused_recurrent_gated_delta_rule``. Both take q/k/v followed
    by the same named g/beta/state/normalization arguments at the model callsite.
    Qwen's forward reads these four module globals, including cached decoding.
    """
    import importlib
    from importlib import metadata

    names = ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule",
             "causal_conv1d_fn", "causal_conv1d_update")
    implementations = None
    if enabled:
        # Import before modeling so its optional-backend decorators also see
        # installed packages. Missing/broken packages fail instead of silently
        # changing the selected numerical implementation.
        delta = importlib.import_module("fla.ops.gated_delta_rule")
        conv = importlib.import_module("causal_conv1d")
        recurrent = getattr(delta, "recurrent_gated_delta_rule", None)
        if recurrent is None:
            recurrent = delta.fused_recurrent_gated_delta_rule
        implementations = (recurrent, delta.chunk_gated_delta_rule,
                           conv.causal_conv1d_fn, conv.causal_conv1d_update)
        for function in implementations[:2]:
            params = inspect.signature(function).parameters
            required = {"g", "beta", "scale", "initial_state", "output_final_state",
                        "use_qk_l2norm_in_kernel", "state_v_first", "cu_seqlens"}
            if (not required.issubset(params) or tuple(params)[:3] != ("q", "k", "v")
                    or params["scale"].default is not None or params["state_v_first"].default is not False):
                raise RuntimeError("Installed FLA kernel has an unsupported Qwen3.5 argument contract")
        for function, required in ((implementations[2], {"x", "weight", "bias", "activation"}),
                                   (implementations[3], {"x", "conv_state", "weight", "bias", "activation"})):
            if not required.issubset(inspect.signature(function).parameters):
                raise RuntimeError("Installed causal-conv1d kernel has an unsupported Qwen3.5 argument contract")

    module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    if not hasattr(module, "_opium_bench_original_kernels"):
        module._opium_bench_original_kernels = {name: getattr(module, name) for name in names}
    originals = module._opium_bench_original_kernels

    def bind(implementation):
        params = tuple(inspect.signature(implementation).parameters)

        def dispatch(*args, **kwargs):
            # Match the Transformers fallback decorator: unrelated model
            # kwargs are dropped, but state/layout options are preserved.
            return implementation(*args, **{key: value for key, value in kwargs.items() if key in params})

        return dispatch

    for index, name in enumerate(names):
        setattr(module, name, bind(implementations[index]) if enabled else originals[name])
    versions = {}
    for package in ("transformers", "torch", "triton", "fla-core", "causal-conv1d", "einops"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    selected = {name: _kernel_callable_identity(getattr(module, name)) for name in names}
    return module, {"mode": "explicit_local_v1" if enabled else "transformers_default",
                    "hub_kernels": False, "packages": versions, "implementations": selected}

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
        kernel_backend = None
        if kind == "qwen3":
            adapter, model_class = "qwen3", transformers.AutoModelForCausalLM
        elif kind in ("qwen3_5", "qwen3_5_text"):
            adapter = "qwen3_5"
            class_name = "Qwen3_5ForConditionalGeneration" if kind == "qwen3_5" else "Qwen3_5ForCausalLM"
            module, kernel_backend = _configure_qwen35_kernels(
                _boolean(profile.get("local_kernels", False), "local_kernels"))
            model_class = getattr(module, class_name, None)
            if model_class is None:
                raise RuntimeError(f"Installed Transformers {transformers.__version__} lacks {class_name}; "
                                   "install the model's documented compatible version explicitly")
        else:
            raise ValueError(f"Unsupported architecture {kind!r}; only dense Qwen3 and Qwen3.5 adapters are provided")
        kwargs = dict(common, dtype=dtype, device_map={"": device}, attn_implementation=attention)
        if adapter == "qwen3_5":
            kwargs["use_kernels"] = False  # Local selected implementations only; no Hub fetch.
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
                fingerprint["kernel_backend"] = kernel_backend
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
        if config.get("schema_version") == 2 or config.get("preset") == "research":
            return self._calibrate_research(config, out_dir, emit, should_stop)
        if config.get("schema_version", 1) != 1:
            raise ValueError("Unsupported calibration schema")
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
        guarded_npz(out_dir / "vectors.npz", **vectors)
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

    def _research_tokens(self, text, maximum, span=None):
        """Tokenize without truncation; optional declared character-span mask."""
        import torch
        kwargs = {"return_tensors": "pt"}
        if span is not None:
            kwargs["return_offsets_mapping"] = True
        try:
            encoded = self.tokenizer(text, **kwargs)
        except (TypeError, NotImplementedError) as error:
            if span is not None:
                raise ValueError("Span pooling requires a tokenizer with character offset mapping") from error
            raise
        ids = encoded.input_ids.to(self.device)
        if ids.ndim != 2 or ids.shape[0] != 1 or not ids.shape[1] or ids.shape[1] > maximum:
            raise ValueError("Research calibration input exceeds max_input_tokens or has invalid shape; no truncation is allowed")
        mask = getattr(encoded, "attention_mask", None)
        mask = torch.ones_like(ids) if mask is None else mask.to(self.device)
        if mask.shape != ids.shape or not bool(mask.bool().any()):
            raise ValueError("Research tokenizer returned an invalid attention mask")
        span_mask = None
        if span is not None:
            offsets = getattr(encoded, "offset_mapping", None)
            if offsets is None:
                raise ValueError("Span pooling requires tokenizer offset_mapping")
            offsets = offsets[0].tolist() if hasattr(offsets, "tolist") else offsets[0]
            if len(offsets) != ids.shape[1]:
                raise ValueError("Tokenizer offsets do not match input tokens")
            start, end = span
            valid = [(max(start, a), min(end, b)) for (a, b), active in zip(offsets, mask[0].tolist()) if active and b > a and a < end and b > start]
            covered = set(i for a, b in valid for i in range(a, b))
            if not valid or any(i not in covered for i in range(start, end) if not text[i].isspace()):
                raise ValueError("Declared span is incomplete in tokenizer offsets; no truncation/fallback is allowed")
            span_mask = torch.tensor([[bool(active and b > a and a < end and b > start) for (a, b), active in zip(offsets, mask[0].tolist())]], device=self.device)
        return ids, mask, span_mask

    def _research_forward(self, ids, mask, **kwargs):
        # Single-row unpadded tokenization avoids unnecessary model-specific
        # kwargs. Explicit padding, if supplied by an adapter, retains its mask.
        if not bool(mask.bool().all()):
            kwargs["attention_mask"] = mask
        return self._forward(input_ids=ids, **kwargs)

    def _extract_research(self, corpus, config, emit, should_stop):
        import numpy as np
        import torch
        from .calibration import pool_hidden
        data = corpus["rows"]
        layers = config["layers"] + [config["downstream_layer"]]
        arrays = {f"{layer}:{policy}": np.empty((len(data), self.info["hidden_size"]), dtype=np.float32)
                  for layer in layers for policy in config["poolings"]}
        state, handles = {}, []
        try:
            for layer in layers:
                def capture(module, inputs, output, layer=layer):
                    for policy in config["poolings"]:
                        pooled = pool_hidden(_hidden(output), state["mask"], policy, state["span_mask"])
                        arrays[f"{layer}:{policy}"][state["index"]] = pooled[0].detach().float().cpu().numpy()
                handles.append(self.blocks[layer].register_forward_hook(capture))
            with torch.inference_mode():
                for i, row in enumerate(data):
                    _check_stop(should_stop)
                    if "span" in config["poolings"] and "span" not in row:
                        raise ValueError(f"Span pooling requires a declared span in corpus row {row['id']}")
                    ids, mask, span_mask = self._research_tokens(row["text"], config["max_input_tokens"], row.get("span") if "span" in config["poolings"] else None)
                    state.update(index=i, mask=mask, span_mask=span_mask)
                    output = self._research_forward(ids, mask, use_cache=False)
                    del output
                    emit({"type": "calibration_progress", "stage": "research_extract", "completed": i + 1,
                          "total": len(data), "row_id": row["id"], "sites": len(arrays)})
        finally:
            for handle in handles:
                handle.remove()
        return arrays

    def _calibrate_research(self, config, out_dir, emit, should_stop):
        import numpy as np
        from .calibration import (corpus_summary, digest, evaluate_fitted, fit_calibration,
                                  load_research_corpus, validate_config)
        config = validate_config(config, len(self.blocks))
        corpus = config.pop("corpus", None) or load_research_corpus()
        out_dir = Path(out_dir)
        if out_dir.exists() and any(out_dir.iterdir()):
            raise ValueError("Calibration output must be a new or empty directory; existing packages are immutable")
        out_dir.mkdir(parents=True, exist_ok=True)
        progress = {"schema_version": 2, "status": "extracting", "corpus": corpus_summary(corpus), "config": config}
        _write_json(out_dir / "progress.json", progress)
        try:
            emit({"type": "calibration_progress", "stage": "research_plan", "rows": len(corpus["rows"]),
                  "sites": len(config["poolings"]) * (len(config["layers"]) + 1),
                  "activation_bytes": len(corpus["rows"]) * len(config["poolings"]) * (len(config["layers"]) + 1) * self.info["hidden_size"] * 4,
                  "message": "Extraction and heldout discrimination only; independent intervention validation is a separate immutable job"})
            arrays = self._extract_research(corpus, config, emit, should_stop)
            _check_stop(should_stop)
            guarded_npz(out_dir / "activations.npz", **arrays)
            _write_json(out_dir / "corpus.json", corpus)
            emit({"type": "calibration_progress", "stage": "research_fit"})
            package = fit_calibration(arrays, corpus, config, lambda: _check_stop(should_stop), emit)
            _check_stop(should_stop)
            metadata = package["metadata"]
            metadata["heldout"] = evaluate_fitted(package, arrays, corpus, checkpoint=lambda: _check_stop(should_stop))
            _check_stop(should_stop)
            metadata.update(name=config["name"], created_utc=datetime.now(timezone.utc).isoformat(),
                            model_fingerprint=deepcopy(self.info["fingerprint"]),
                            model_fingerprint_sha256=self.info["fingerprint_sha256"],
                            tokenizer_identity=tokenizer_identity(self.tokenizer),
                            numerical_environment=deepcopy(self.info.get("numerical_environment")),
                            source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                           for name in ("runtime.py", "calibration.py", "calibration_validation.py")},
                            extraction_policy={"site": "post_block_residual", "input": "raw authored corpus text",
                                               "poolings": config["poolings"], "half_precision_pooling_accumulator": "float32", "span_offsets": "character overlap; complete nonwhitespace coverage required",
                                               "prompt_truncation": False, "probe_score": "affine score, zero classification threshold",
                                               "generation_transfer": "unvalidated; generation samples final input position regardless of extraction pooling",
                                               "attenuation_reference": "residual_origin",
                                               "attenuation_basis": "symmetric_lowdin_orthogonalization"},
                            supported_contexts=sorted({r["context"] for r in corpus["rows"]}),
                            supported_thinking_modes=[], selected_dose=0.,
                            limitations=["Heldout text discrimination is not an independent causal or experiential validation.",
                                         "Authored reasoning contexts do not establish transfer to generated reasoning.",
                                         "Run immutable intervention validation and independently score its blinded continuations before causal language claims."])
            metadata["runtime_compatibility_sha256"] = digest({"fit": metadata["compatibility_sha256"],
                "model": metadata["model_fingerprint_sha256"], "tokenizer": metadata["tokenizer_identity"],
                "extraction": metadata["extraction_policy"]})
            guarded_npz(out_dir / "vectors.npz", **package["vectors"])
            metadata["vectors_sha256"] = hashlib.sha256((out_dir / "vectors.npz").read_bytes()).hexdigest()
            metadata["evidence_sha256"] = {name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
                                          for name in ("corpus.json", "activations.npz")}
            _check_stop(should_stop)
            _write_json(out_dir / "calibration.json", metadata)
            progress["status"] = "complete"
            _write_json(out_dir / "progress.json", progress)
            emit({"type": "calibration_complete", "path": str(out_dir), "layer": metadata["layer"],
                  "selected_dose": 0., "status": "unvalidated", "heldout": metadata["heldout"]})
            return metadata
        except BaseException as error:
            progress.update(status="cancelled" if isinstance(error, Cancelled) else "failed", error=str(error))
            _write_json(out_dir / "progress.json", progress)
            raise

    def _package_research(self, directory, metadata):
        import numpy as np
        from .calibration import corpus_summary, digest, validate_config, validate_corpus
        if metadata.get("model_fingerprint_sha256") != self.info["fingerprint_sha256"] or metadata.get("model_fingerprint") != self.info["fingerprint"]:
            raise ValueError("Research calibration fingerprint differs from the loaded model/runtime")
        if metadata.get("tokenizer_identity") != tokenizer_identity(self.tokenizer):
            raise ValueError("Research calibration tokenizer/template differs from the loaded model")
        config = validate_config(metadata.get("config"), len(self.blocks))
        if config != metadata["config"]:
            raise ValueError("Research calibration config is not its canonical resolved identity")
        evidence = metadata.get("evidence_sha256")
        allowed = {"corpus.json", "activations.npz", "diagnostics.json", "continuations.json", "scoring-sheet.json", "scoring-key.json"}
        if not isinstance(evidence, dict) or not {"corpus.json", "activations.npz"} <= set(evidence) or set(evidence) - allowed:
            raise ValueError("Research calibration evidence manifest is incomplete or unsafe")
        for name, checksum in evidence.items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Research calibration evidence checksum mismatch: {name}")
        corpus = validate_corpus(json.loads((directory / "corpus.json").read_text(encoding="utf-8")))
        if corpus_summary(corpus) != metadata.get("corpus"):
            raise ValueError("Research calibration corpus/split identity mismatch")
        selected, downstream = metadata.get("selected", {}), metadata.get("downstream_selected", {})
        if selected.get("layer") not in config["layers"] or selected.get("pooling") not in config["poolings"] or downstream.get("layer") != config["downstream_layer"] or downstream.get("pooling") not in config["poolings"]:
            raise ValueError("Research calibration selected site is outside its declared candidates")
        if metadata.get("layer") != selected["layer"] or metadata.get("downstream_layer") != downstream["layer"]:
            raise ValueError("Research calibration hook sites differ from fitted sites")
        fit_identity = digest({"config": config, "corpus": metadata["corpus"], "selected": selected, "downstream": downstream})
        runtime_identity = digest({"fit": fit_identity, "model": self.info["fingerprint_sha256"],
                                   "tokenizer": metadata["tokenizer_identity"], "extraction": metadata["extraction_policy"]})
        if fit_identity != metadata.get("compatibility_sha256") or runtime_identity != metadata.get("runtime_compatibility_sha256"):
            raise ValueError("Research calibration compatibility identity mismatch")
        vector_path = directory / "vectors.npz"
        if hashlib.sha256(vector_path.read_bytes()).hexdigest() != metadata.get("vectors_sha256"):
            raise ValueError("Calibration vector checksum mismatch")
        with np.load(vector_path, allow_pickle=False) as stored:
            vectors = {name: stored[name].copy() for name in stored.files}
        required = {"pain", "joy", "joy_raw", "neutral", "scale", "random"}
        for prefix in ("probe", "downstream"):
            required.update({f"{prefix}_center", f"{prefix}_scale"})
            for concept in ("pain", "joy"):
                required.update({f"{prefix}_{concept}", f"{prefix}_{concept}_weight", f"{prefix}_{concept}_bias",
                                 f"{prefix}_{concept}_fit_center", f"{prefix}_{concept}_fit_scale"})
        if not required <= vectors.keys():
            raise ValueError("Research calibration lacks required readout/intervention arrays")
        for name, value in vectors.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite research calibration array {name}")
            scalar = name == "scale" or (name.endswith("_scale") and not name.endswith("_fit_scale")) or name.endswith("_bias")
            if value.shape != (() if scalar else (self.info["hidden_size"],)):
                raise ValueError(f"Wrong research calibration array dimension: {name}")
            if (name.endswith("scale") or name == "scale") and not (value > 0).all():
                raise ValueError(f"Invalid research calibration scale {name}")
        for name in ("pain", "joy_raw", "random"):
            if abs(float(np.linalg.norm(vectors[name])) - 1.) > 1e-4:
                raise ValueError(f"Research intervention is not unit normalized: {name}")
        if metadata.get("orthogonal_joy_available") and abs(float(np.linalg.norm(vectors["joy"])) - 1.) > 1e-4:
            raise ValueError("Research orthogonal joy vector is inconsistent")
        return {"metadata": metadata, "vectors": vectors}

    @contextmanager
    def _research_diagnostic_hooks(self, package, spec, state):
        """Actual single-final-position diagnostic, independent of live controls.

        All targets are computed from the same unedited state. Random matching
        accounts for dtype rounding; requested and delivered norms are logged.
        """
        import numpy as np
        import torch
        from .calibration_validation import perturbation_delta
        meta, vectors = package["metadata"], package["vectors"]
        handles = []
        def measure(hidden, prefix):
            value = hidden.detach().float().cpu().numpy()
            return {c: float(value @ vectors[f"{prefix}_{c}_weight"] + vectors[f"{prefix}_{c}_bias"])
                    for c in ("pain", "joy")}
        def edit(module, inputs, output):
            hidden = _hidden(output)
            before = hidden[0, -1].float()
            raw = before.detach().cpu().numpy()
            matched = None
            if spec["operator"] == "random_matched":
                target_spec = {"operator": "combined", "joy_gain": spec["dose"], "pain_suppression": min(spec["dose"], 1.)}
                desired = perturbation_delta(raw, vectors, target_spec)
                # Match the actual representable combined edit, not a coefficient.
                rounded = (before + torch.as_tensor(desired, device=before.device, dtype=torch.float32)).to(hidden.dtype).float()
                matched = (rounded - before).detach().cpu().numpy()
            delta = perturbation_delta(raw, vectors, spec, matched)
            if np.any(delta):
                changed = hidden.clone()
                changed[0, -1] = (before + torch.as_tensor(delta, device=before.device, dtype=torch.float32)).to(hidden.dtype)
                actual = changed[0, -1].float()
            else:
                changed, actual = None, before
            requested_norm = float(np.linalg.norm(delta))
            delivered_norm = float((actual - before).norm())
            if matched is not None and requested_norm > 0 and delivered_norm > 0:
                # Quantized hidden dtype rounds individual coordinates. Refine
                # the scalar along the fixed random direction and retain the
                # closest representable edit; residual mismatch stays explicit.
                best = (abs(delivered_norm - requested_norm), actual.clone(), delivered_norm)
                gain = 1.
                for _ in range(8):
                    gain *= requested_norm / max(delivered_norm, 1e-12)
                    candidate = (before + torch.as_tensor(delta * gain, device=before.device, dtype=torch.float32)).to(hidden.dtype).float()
                    delivered_norm = float((candidate - before).norm())
                    error = abs(delivered_norm - requested_norm)
                    if error < best[0]:
                        best = (error, candidate.clone(), delivered_norm)
                    if error <= max(1e-6, .001 * requested_norm):
                        break
                actual, delivered_norm = best[1], best[2]
                changed[0, -1] = actual.to(hidden.dtype)
            state.update(pre=measure(before, "probe"), post=measure(actual, "probe"), downstream=None,
                         relative_delta=float((actual - before).norm() / before.norm().clamp_min(1e-12)),
                         requested_edit_norm=requested_norm, delivered_edit_norm=delivered_norm,
                         matched_target_norm=float(np.linalg.norm(matched)) if matched is not None else None,
                         random_norm_match_error=(abs(delivered_norm - float(np.linalg.norm(matched))) if matched is not None else None),
                         random_match_within_tolerance=(abs(delivered_norm - requested_norm) <= max(1e-6, .01 * requested_norm) if matched is not None else None),
                         random_match_tolerance={"absolute": 1e-6, "relative": .01} if matched is not None else None,
                         edited_positions=[int(hidden.shape[1]) - 1], scope="final_processed_input_position")
            if changed is not None:
                return (changed,) + output[1:] if isinstance(output, tuple) else changed
            return None
        def downstream(module, inputs, output):
            state["downstream"] = measure(_hidden(output)[0, -1].float(), "downstream")
        try:
            handles.append(self.blocks[meta["layer"]].register_forward_hook(edit))
            handles.append(self.blocks[meta["downstream_layer"]].register_forward_hook(downstream))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _research_continuation(self, package, prompt, spec, limit, maximum, should_stop):
        """Greedy raw-prefix continuation; records exact public token sequence.

        No private API reasoning is accessed. These are raw-prefix continuations,
        not a claim that both chat thinking modes were validated.
        """
        import torch
        ids, mask, _ = self._research_tokens(prompt, maximum)
        context_limit = self.info.get("max_position_embeddings")
        if isinstance(context_limit, int) and ids.shape[1] + limit > context_limit:
            raise ValueError("Research continuation exceeds the model context allowance")
        eos = self.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        if self.tokenizer.eos_token_id is not None:
            eos_ids.add(self.tokenizer.eos_token_id)
        tokens, measurements, state, cache = [], [], {}, None
        finish = "length"
        try:
            with torch.inference_mode(), self._research_diagnostic_hooks(package, spec, state):
                for _ in range(limit):
                    _check_stop(should_stop)
                    output = self._research_forward(ids, mask, past_key_values=cache, use_cache=True)
                    cache = output.past_key_values
                    logits = output.logits[0, -1].float()
                    if not bool(torch.isfinite(logits).all()):
                        raise RuntimeError("Nonfinite continuation logits")
                    token = int(logits.argmax())
                    tokens.append(token); measurements.append(deepcopy(state))
                    del output
                    if token in eos_ids:
                        finish = "eos"
                        break
                    ids = torch.tensor([[token]], device=self.device)
                    mask = torch.ones_like(ids)
            visible = tokens[:-1] if tokens and tokens[-1] in eos_ids else tokens
            return {"text": self.tokenizer.decode(visible, skip_special_tokens=False), "token_ids": tokens,
                    "finish_reason": finish, "measurements": measurements,
                    "sampling": {"temperature": 0}, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "intervention_scope": "final processed input position, including each generated-token step"}
        finally:
            del cache

    def validate_calibration(self, config, out_dir, emit=lambda event: None, should_stop=lambda: False):
        """Create a new immutable schema2 validation bundle from a fitted one.

        Config: calibration_dir plus optional doses, validation_pairs_per_concept,
        continuation_tokens, max_kl, max_relative_delta. Source bundle is never
        modified. Selection locks operating dose before heldout diagnostics.
        """
        import shutil
        import torch
        from .calibration import validate_config
        from .calibration_validation import (blinded_continuations, diagnostic_specs,
            grade_objective_continuation, select_operating_range)
        self._require_loaded()
        allowed = {"calibration_dir", "doses", "validation_pairs_per_concept", "continuation_tokens", "max_kl", "max_relative_delta"}
        if not isinstance(config, dict) or set(config) - allowed or not isinstance(config.get("calibration_dir"), str):
            raise ValueError("invalid research validation config")
        source = Path(config["calibration_dir"]).resolve()
        package = self._package(source)
        if package["metadata"].get("schema_version") != 2:
            raise ValueError("Independent research validation requires a schema2 calibration")
        cfg = deepcopy(package["metadata"]["config"])
        cfg.update({k: v for k, v in config.items() if k != "calibration_dir"})
        cfg = validate_config(cfg, len(self.blocks))
        target = Path(out_dir).resolve()
        if target == source or (target.exists() and any(target.iterdir())):
            raise ValueError("Validation output must be a new or empty directory; source calibrations are immutable")
        target.mkdir(parents=True, exist_ok=True)
        corpus = json.loads((source / "corpus.json").read_text(encoding="utf-8"))
        rows_by_split = {}
        for split in ("selection", "heldout"):
            chosen = []
            for concept in ("pain", "joy"):
                # Deterministic round-robin across families avoids selecting only
                # the first family's correlated wrappers for a bounded smoke.
                groups = {}
                for row in corpus["rows"]:
                    if row["split"] == split and row["concept"] == concept and row["label"] == 0:
                        groups.setdefault(row["family"], []).append(row)
                ordered = [group[round_] for round_ in range(max(map(len, groups.values())))
                           for _, group in sorted(groups.items()) if round_ < len(group)]
                chosen.extend(ordered[:cfg["validation_pairs_per_concept"]])
            rows_by_split[split] = chosen
        specs = diagnostic_specs(cfg["doses"], (cfg["seed"] + 200) % 2**32)
        plan = {"schema_version": 1, "status": "running", "source_calibration_sha256": hashlib.sha256((source / "calibration.json").read_bytes()).hexdigest(),
                "source_config_sha256": _digest(package["metadata"]["config"]), "validation_config": cfg,
                "rows": {split: [r["id"] for r in rows] for split, rows in rows_by_split.items()},
                "specifications": specs, "scope": "single final processed input position",
                "fixed_prefix_forwards": sum(len(rows) for rows in rows_by_split.values()) * (len(specs) + 1),
                "continuation_token_upper_bound": sum(len(rows) for rows in rows_by_split.values()) * 7 * cfg["continuation_tokens"]}
        _write_json(target / "progress.json", plan)
        emit({"type": "calibration_progress", "stage": "validation_plan", **{k: plan[k] for k in ("fixed_prefix_forwards", "continuation_token_upper_bound", "rows")}})
        diagnostics, summaries, continuations = [], [], []
        locked = None
        try:
            for split, rows in rows_by_split.items():
                baselines = {}
                for row in rows:
                    _check_stop(should_stop)
                    ids, mask, _ = self._research_tokens(row["text"], cfg["max_input_tokens"])
                    with torch.inference_mode():
                        baseline = self._research_forward(ids, mask, use_cache=False).logits[0, -1].float().log_softmax(-1)
                    baselines[row["id"]] = baseline.detach().cpu()
                for spec in specs:
                    observations = []
                    for row in rows:
                        _check_stop(should_stop)
                        ids, mask, _ = self._research_tokens(row["text"], cfg["max_input_tokens"])
                        state = {}
                        with torch.inference_mode(), self._research_diagnostic_hooks(package, spec, state):
                            actual = self._research_forward(ids, mask, use_cache=False).logits[0, -1].float().log_softmax(-1)
                        baseline = baselines[row["id"]].to(actual.device)
                        kl = max(0., float((baseline.exp() * (baseline - actual)).sum()))
                        if not math.isfinite(kl):
                            raise RuntimeError("Nonfinite fixed-prefix KL")
                        record = {"split": split, "row_id": row["id"], "family": row["family"], "concept": row["concept"],
                                  "condition": spec["id"], "specification": spec, "next_token_kl": kl, **deepcopy(state)}
                        diagnostics.append(record); observations.append(record)
                        emit({"type": "calibration_progress", "stage": "research_diagnostics", "completed": len(diagnostics),
                              "total": sum(map(len, rows_by_split.values())) * len(specs), "split": split, "condition": spec["id"], "row_id": row["id"]})
                    summaries.append({"split": split, "condition": spec["id"], "operator": spec["operator"], "dose": spec["dose"],
                                      "examples": len(observations), "mean_next_token_kl": sum(r["next_token_kl"] for r in observations)/len(observations),
                                      "mean_relative_delta": sum(r["relative_delta"] for r in observations)/len(observations)})
                if split == "selection":
                    locked = select_operating_range([r for r in summaries if r["split"] == "selection"], cfg["max_kl"], cfg["max_relative_delta"])
                    _write_json(target / "diagnostics.json", {"status": "selection_locked", "plan": plan, "operating_range": locked, "summaries": summaries, "records": diagnostics})
                    emit({"type": "calibration_progress", "stage": "operating_range_locked", **locked})
                # Use precisely the already-locked operating dose for both sets;
                # all signed/attenuation sweeps remain in fixed-prefix evidence.
                selected_specs = [s for s in specs if s["operator"] == "sham" or (s["dose"] == locked["selected_dose"] and s["operator"] in {"add", "combined", "random_matched"})]
                for row in rows:
                    prompt = row["text"] + "\nContinue the description in one or two sentences."
                    for spec in selected_specs:
                        _check_stop(should_stop)
                        generated = self._research_continuation(package, prompt, spec, cfg["continuation_tokens"], cfg["max_input_tokens"], should_stop)
                        identifier = f"{split}-{row['id']}-{spec['id']}"
                        continuations.append({"record_id": identifier, "prompt": prompt, "continuation": generated["text"],
                                              "condition": spec["id"], "split": split, "row_id": row["id"],
                                              "generation": generated, "objective_grade": grade_objective_continuation(generated["text"]),
                                              "semantic_ratings": None})
                        emit({"type": "calibration_progress", "stage": "research_continuations", "completed": len(continuations),
                              "total": sum(map(len, rows_by_split.values())) * len(selected_specs), "split": split, "condition": spec["id"]})
            # Empty completions are evidence too. The sheet accepts their text
            # as empty and labels all semantic scores missing until a rater acts.
            sheet, key = blinded_continuations(continuations, cfg["seed"] + 300)
            _write_json(target / "diagnostics.json", {"status": "complete", "plan": plan, "operating_range": locked, "summaries": summaries, "records": diagnostics})
            _write_json(target / "continuations.json", continuations)
            _write_json(target / "scoring-sheet.json", sheet)
            _write_json(target / "scoring-key.json", key)
            for name in ("corpus.json", "activations.npz", "vectors.npz"):
                _check_stop(should_stop)
                guarded_copyfile(source / name, target / name)
            metadata = deepcopy(package["metadata"])
            metadata.update(created_utc=datetime.now(timezone.utc).isoformat(), parent_calibration_sha256=plan["source_calibration_sha256"],
                            validation_config=cfg, validation_plan=plan, selected_dose=locked["selected_dose"],
                            operating_range=locked, status="validated_for_enumerated_tests",
                            validated_tests=["family-split heldout text discrimination", "fixed-prefix signed gain/attenuation/sham/random perturbation measurements"],
                            continuation_scoring={"status": "awaiting_independent_human_ratings", "examples": len(continuations),
                                                  "rubric_sha256": sheet["rubric_sha256"], "scored": 0,
                                                  "public_sheet": "scoring-sheet.json", "private_key": "scoring-key.json"})
            unmatched = sum(r.get("random_match_within_tolerance") is False for r in diagnostics)
            metadata["random_control_norm_audit"] = {"mismatched_positions": unmatched, "tolerance": {"absolute": 1e-6, "relative": .01}}
            if unmatched:
                metadata["status"] = "unvalidated"
                metadata["limitations"].append("Some random controls could not match delivered edit norm within 1% or 1e-6 absolute after dtype rounding; inspect diagnostics before interpreting comparisons.")
            if not any(r["next_token_kl"] > 1e-7 for r in diagnostics if r["split"] == "heldout" and r["specification"]["operator"] != "sham"):
                metadata["status"] = "no_detectable_effect"
                metadata["no_detectable_effect_scope"] = "heldout fixed-prefix next-token KL at the tested doses; no conclusion about unscored semantic effects"
            metadata["evidence_sha256"] = {name: hashlib.sha256((target / name).read_bytes()).hexdigest()
                for name in ("corpus.json", "activations.npz", "diagnostics.json", "continuations.json", "scoring-sheet.json", "scoring-key.json")}
            _check_stop(should_stop)
            _write_json(target / "calibration.json", metadata)
            plan["status"] = "complete"; _write_json(target / "progress.json", plan)
            emit({"type": "calibration_complete", "path": str(target), "layer": metadata["layer"],
                  "selected_dose": metadata["selected_dose"], "status": metadata["status"], "continuation_scoring": metadata["continuation_scoring"]})
            return metadata
        except BaseException as error:
            plan.update(status="cancelled" if isinstance(error, Cancelled) else "failed", error=str(error))
            _write_json(target / "progress.json", plan)
            _write_json(target / "diagnostics.json", {"status": plan["status"], "plan": plan, "operating_range": locked, "summaries": summaries, "records": diagnostics})
            _write_json(target / "continuations.json", continuations)
            raise

    def _package(self, calibration_dir):
        import numpy as np
        directory = Path(calibration_dir).resolve()
        manifest_path = directory / "calibration.json"
        vector_path = directory / "vectors.npz"
        evidence_names = ("corpus.json", "activations.npz", "diagnostics.json", "continuations.json", "scoring-sheet.json", "scoring-key.json")
        evidence_stats = tuple((name, (directory / name).stat().st_mtime_ns, (directory / name).stat().st_size)
                               for name in evidence_names if (directory / name).exists())
        key = (str(directory), manifest_path.stat().st_mtime_ns, vector_path.stat().st_mtime_ns, vector_path.stat().st_size, evidence_stats)
        if key in self._calibration_cache:
            return self._calibration_cache[key]
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") == 2:
            package = self._package_research(directory, metadata)
            self._calibration_cache = {key: package}
            return package
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
            if meta.get("schema_version") == 2:
                return {label: float(hidden @ v[f"{prefix}_{label}_weight"] + v[f"{prefix}_{label}_bias"])
                        for label in ("pain", "joy")}
            centered = hidden - v[f"{prefix}_center"]
            scale = v[f"{prefix}_scale"]
            return {label: float(centered @ v[f"{prefix}_{label}"] / scale) for label in ("pain", "joy")}

        def edit(module, inputs, output):
            hidden = _hidden(output)
            v = tensors(hidden.device)
            before = hidden[0, -1].float()
            coefficients = state["dose"]["effective"]
            if meta.get("schema_version") == 2 and coefficients["joy"] and not meta.get("orthogonal_joy_available"):
                raise ValueError("This calibration has no independent orthogonal joy direction; choose a supported direction or recalibrate")
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

    def _prepare_control_v2(self, value, package, runtime_controls=None):
        """Validate the actual controller snapshot, never UI display scalars."""
        import numpy as np
        from .effects import (ATTENUATION_AXES, COEFFICIENT_AXES, GAIN_AXES,
                              joint_attenuation, validate_preset)
        if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 2:
            raise ValueError("An active v2 recipe requires v2 controller snapshots throughout generation")
        if package is None:
            raise ValueError("A compatible calibration is required for a v2 effect recipe")
        meta, vectors = package["metadata"], package["vectors"]
        axes = {"pain": vectors["pain"], "joy_raw": vectors.get("joy_raw"),
                "joy_orthogonal": vectors["joy"], "random_gain": vectors["random"]}
        def require_axis(axis):
            vector = axes[axis]
            if vector is None or np.shape(vector) != (self.info["hidden_size"],) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-8:
                raise ValueError(f"Calibration does not support the requested {axis} direction")
            if axis == "joy_orthogonal" and meta.get("schema_version") == 2 and not meta.get("orthogonal_joy_available"):
                raise ValueError("Calibration has no independent orthogonal joy direction")
        config = value.get("config")
        if not isinstance(config, dict) or not isinstance(config.get("effect_presets"), list) or not config["effect_presets"]:
            raise ValueError("V2 control must retain its resolved effect presets")
        prefill_positions = set()
        for raw in config["effect_presets"]:
            preset = validate_preset(raw, capabilities={"phases": ["prefill", "reasoning", "output"],
                "prefill_positions": ["last", "all"], "sites": ["residual_post"], "layers": [meta["layer"]]})
            if preset["site"]["layer"] not in ("calibrated", meta["layer"]):
                raise ValueError("Preset edit layer differs from the calibrated site")
            if "prefill" in preset["phases"]:
                prefill_positions.add("last")
                if preset["prefill_positions"] == "all":
                    prefill_positions.add("other")
            if preset["gains"]["pain"] or preset["attenuation"]["pain"]:
                require_axis("pain")
            if preset["gains"]["joy"] or preset["attenuation"]["joy"]:
                require_axis("joy_" + preset["joy_direction"])
            if preset["gains"]["random"]:
                require_axis("random_gain")
        if value.get("baseline_policy") != "generated-phases-only":
            raise ValueError("Unsupported v2 baseline processing policy")
        phase_keys = ("reasoning", "output", "prefill_last", "prefill_other")
        baselines, coefficients = value.get("baseline_by_phase"), value.get("phase_coefficients")
        if not isinstance(baselines, dict) or not isinstance(coefficients, dict) or set(baselines) != set(phase_keys) or set(coefficients) != set(phase_keys):
            raise ValueError("V2 control must contain every declared processing phase")
        prepared = {"schema_version": 2, "control_revision": _integer(value.get("control_revision"), "control revision", 0, 10**12),
                    "generated_tokens": _integer(value.get("generated_tokens"), "generated token index", 0, 10**12),
                    "completed_decisions": _integer(value.get("completed_decisions"), "completed decisions", 0, 10**12),
                    "enabled": _boolean(value.get("enabled"), "enabled"), "level": _number(value.get("level"), "level", 0, 1),
                    "recipe_hash": value.get("recipe_hash"), "site": {"location": "residual_post", "layer": meta["layer"]},
                    "preset_identity": _digest(config["effect_presets"]), "prefill_positions": sorted(prefill_positions),
                    "phase_coefficients": {}, "baseline_by_phase": {}}
        for phase in phase_keys:
            pulse, base = coefficients[phase], baselines[phase]
            if not isinstance(pulse, dict) or set(pulse) != set(COEFFICIENT_AXES) or not isinstance(base, dict) or set(base) != {"pain", "joy_raw", "joy_orthogonal"}:
                raise ValueError("V2 phase coefficients have missing/unknown axes")
            pulse = {axis: _number(pulse[axis], axis, -4 if axis in GAIN_AXES else 0, 4 if axis in GAIN_AXES else 1) for axis in COEFFICIENT_AXES}
            base = {axis: _number(base[axis], f"baseline {axis}", -4, 4) for axis in base}
            if phase.startswith("prefill_") and any(base.values()):
                raise ValueError("V2 held baselines cannot apply during prompt prefill")
            if phase.startswith("prefill_") and phase.split("_", 1)[1] not in prefill_positions and any(pulse.values()):
                raise ValueError("Undeclared prefill coefficients are not supported")
            for axis in GAIN_AXES:
                if pulse[axis] or base.get(axis, 0):
                    require_axis(axis)
            names = [axis.removesuffix("_attenuation") for axis in ATTENUATION_AXES if pulse[axis]]
            for axis in names:
                require_axis(axis)
            if names:
                # Fail before forward if active directions are dependent. This
                # is repeated at control boundaries because a manual change can
                # activate a previously inactive combination midway through text.
                joint_attenuation(np.zeros(self.info["hidden_size"]), [axes[n] for n in names], [pulse[n + "_attenuation"] for n in names])
            prepared["phase_coefficients"][phase] = pulse
            prepared["baseline_by_phase"][phase] = base
        if runtime_controls:
            from .runtime_controls import prepare_random_match
            match = prepare_random_match(value, runtime_controls)
            require_axis("random_gain")
            for target in match["target_by_phase"].values():
                names = [axis.removesuffix("_attenuation") for axis in ATTENUATION_AXES if target[axis]]
                if names:
                    joint_attenuation(np.zeros(self.info["hidden_size"]), [axes[n] for n in names], [target[n + "_attenuation"] for n in names])
            prepared["random_norm_match"] = match
        return prepared

    def _dose_v2(self, prepared, phase_key):
        pulse = deepcopy(prepared["phase_coefficients"][phase_key])
        baseline = deepcopy(prepared["baseline_by_phase"][phase_key])
        effective = {**pulse, **{f"baseline_{key}": value for key, value in baseline.items()}}
        result = {"schema_version": 2, "phase": "prefill" if phase_key.startswith("prefill_") else phase_key,
                "prefill_position": phase_key.split("_", 1)[1] if phase_key.startswith("prefill_") else None,
                "coefficients": pulse, "baseline": baseline, "effective": effective,
                "applied": any(effective.values()), "enabled": prepared["enabled"], "level": prepared["level"],
                "control_revision": prepared["control_revision"], "generated_token_index": prepared["generated_tokens"],
                "completed_decisions": prepared["completed_decisions"], "site": deepcopy(prepared["site"]),
                "operation_order": ["baseline_challenge", "joint_attenuation", "addition"],
                "attenuation_reference": "residual_origin",
                "attenuation_basis": "symmetric_lowdin_orthogonalization", "recipe_hash": prepared["recipe_hash"]}
        if "random_norm_match" in prepared:
            match = prepared["random_norm_match"]
            result["random_norm_match"] = {key: deepcopy(value) for key, value in match.items() if key != "target_by_phase"}
            result["random_norm_match"].update(target_coefficients=deepcopy(match["target_by_phase"][phase_key]),
                                               nominal_random_gain=pulse["random_gain"])
        return result

    @contextmanager
    def _hooks_v2(self, package, state):
        """Explicit prefill stages followed by the sampled-token input stage.

        The first forward remains a single full-prompt pass. If both scopes are
        requested, prefill is applied first; the last position then receives the
        generation-phase operation. Later passes receive only generation-phase
        operations. Each stage is recorded separately, with no prompt clock age.
        """
        import numpy as np
        import torch
        from .effects import ATTENUATION_AXES, GAIN_AXES
        meta, vectors = package["metadata"], package["vectors"]
        cached, bases, handles = {}, {}, []
        def tensors(device):
            if device not in cached:
                cached[device] = {name: torch.as_tensor(value, device=device, dtype=torch.float32) for name, value in vectors.items()}
            return cached[device]
        def measure(hidden, v, prefix):
            if meta.get("schema_version") == 2:
                return {c: hidden @ v[f"{prefix}_{c}_weight"] + v[f"{prefix}_{c}_bias"] for c in ("pain", "joy")}
            return {c: ((hidden - v[f"{prefix}_center"]) @ v[f"{prefix}_{c}"]) / v[f"{prefix}_scale"] for c in ("pain", "joy")}
        def serialize_measure(values):
            return {key: value.detach().float().cpu().tolist() for key, value in values.items()}
        def summary(values):
            return {key: float(value.mean()) for key, value in values.items()}
        def axis_arrays(v):
            return {"pain": v["pain"], "joy_raw": v.get("joy_raw"), "joy_orthogonal": v["joy"], "random_gain": v["random"]}
        def lowdin(names, device):
            key = (tuple(names), device)
            if key not in bases:
                raw = {"pain": vectors["pain"], "joy_raw": vectors.get("joy_raw"), "joy_orthogonal": vectors["joy"]}
                d = np.stack([raw[name] / np.linalg.norm(raw[name]) for name in names], axis=1).astype(np.float64)
                eigenvalues, eigenvectors = np.linalg.eigh(d.T @ d)
                if eigenvalues.min() <= 1e-7 * eigenvalues.max():
                    raise ValueError("Active attenuation directions are rank deficient")
                q = d @ ((eigenvectors * (1 / np.sqrt(eigenvalues))) @ eigenvectors.T)
                bases[key] = torch.as_tensor(q, device=device, dtype=torch.float32)
            return bases[key]
        def apply(before, dose, v):
            axes = axis_arrays(v)
            challenged = before
            for axis, coefficient in dose["baseline"].items():
                if coefficient:
                    challenged = challenged + coefficient * v["scale"] * axes[axis]
            names = [axis.removesuffix("_attenuation") for axis in ATTENUATION_AXES if dose["coefficients"][axis]]
            after = challenged
            if names:
                q = lowdin(names, before.device)
                fractions = torch.tensor([dose["coefficients"][name + "_attenuation"] for name in names], device=before.device, dtype=torch.float32)
                after = after - ((after @ q) * fractions) @ q.T
            for axis in GAIN_AXES:
                coefficient = dose["coefficients"][axis]
                if coefficient:
                    after = after + coefficient * v["scale"] * axes[axis]
            return after
        def requested_edit(before, dose, v, dtype):
            if "random_norm_match" not in dose:
                return apply(before, dose, v) if dose["applied"] else before
            from .runtime_controls import match_rounded_random, RuntimeControlUnavailable
            match = dose["random_norm_match"]
            target_dose = {**dose, "coefficients": match["target_coefficients"]}
            target = apply(before, target_dose, v).to(dtype).float()
            nominal = match["nominal_random_gain"]
            if nominal == 0:
                if bool((target != before).any()):
                    raise RuntimeControlUnavailable("Counterfactual target is active without a random source pulse")
                match.update(status="inactive", target_edit_norms=0., delivered_edit_norms=0., absolute_errors=0.,
                             norm_ratios=1., effective_random_coefficients=0., rounding_dtype=str(dtype))
                return before
            actual, gains, evidence = match_rounded_random(before, target, v["random"], v["scale"],
                1 if nominal > 0 else -1, dtype, relative_tolerance=match["relative_tolerance"],
                absolute_tolerance=match["absolute_tolerance"])
            match.update(evidence)
            # A multi-position prefill group exposes every individual gain and
            # its mean for the existing positions-weighted exposure ledger.
            coefficient = float(gains.mean())
            dose["coefficients"]["random_gain"] = coefficient
            dose["effective"]["random_gain"] = coefficient
            dose["applied"] = any(dose["effective"].values())
            dose["numeric_nonzero"] = bool((actual != before).any())
            dose["coefficient_aggregation"] = "mean_over_positions" if gains.ndim else "single_position"
            return actual
        def edit(module, inputs, output):
            hidden = _hidden(output)
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError("V2 intervention adapter supports one sequence at a time")
            v = tensors(hidden.device)
            working, changed = hidden, False
            state["prefill_events"] = []
            if state["first_forward"]:
                groups = []
                if "other" in state["prepared"]["prefill_positions"] and hidden.shape[1] > 1:
                    groups.append(("other", list(range(hidden.shape[1]-1))))
                if "last" in state["prepared"]["prefill_positions"]:
                    groups.append(("last", [hidden.shape[1]-1]))
                for position, indices in groups:
                    dose = self._dose_v2(state["prepared"], "prefill_" + position)
                    before = working[0, indices].float()
                    requested = requested_edit(before, dose, v, hidden.dtype)
                    actual = requested.to(hidden.dtype).float()
                    if dose["applied"]:
                        if not changed:
                            working = hidden.clone(); changed = True
                        working[0, indices] = actual.to(hidden.dtype)
                    pre, post = measure(before, v, "probe"), measure(actual, v, "probe")
                    norms = (actual-before).norm(dim=-1)
                    state["prefill_events"].append({"type": "prefill", "phase": "prefill", "prefill_position": position,
                        "positions": len(indices), "position_indices": indices, "dose": dose,
                        "control_revision": dose["control_revision"], "generated_token_index": dose["generated_token_index"],
                        "completed_decisions": dose["completed_decisions"], "site": dose["site"],
                        "measurements": {"pre": summary(pre), "post": summary(post), "downstream": None,
                                         "relative_delta": float((norms / before.norm(dim=-1).clamp_min(1e-12)).mean()),
                                         "requested_edit_norms": (requested-before).norm(dim=-1).detach().cpu().tolist(),
                                         "delivered_edit_norms": norms.detach().cpu().tolist(),
                                         "per_position": {"pre": serialize_measure(pre), "post": serialize_measure(post), "downstream": None}},
                        "measurement_alignment": "prompt input positions at the edit site; generated-token clock does not advance",
                        "composition": "prefill stage precedes generation-stage edit of the final prompt position"})
            before = working[0, -1].float().clone()
            dose = state["dose"]
            requested = requested_edit(before, dose, v, hidden.dtype)
            actual = requested.to(hidden.dtype).float()
            if dose["applied"]:
                if not changed:
                    working = hidden.clone(); changed = True
                working[0, -1] = actual.to(hidden.dtype)
            state["measurements"] = {"pre": summary(measure(before, v, "probe")), "post": summary(measure(actual, v, "probe")),
                "downstream": None, "relative_delta": float((actual-before).norm() / before.norm().clamp_min(1e-12)),
                "requested_edit_norm": float((requested-before).norm()), "delivered_edit_norm": float((actual-before).norm()),
                "site": dose["site"], "phase": dose["phase"], "position_indices": [int(hidden.shape[1])-1],
                "control_revision": dose["control_revision"],
                "units": "independent_affine_readout" if meta.get("schema_version") == 2 else "projection_divided_by_reference_scale",
                "composition": "generation edit follows explicit prefill edit on the first final input position" if state["prefill_events"] else "generation input position only"}
            if changed:
                return (working,) + output[1:] if isinstance(output, tuple) else working
            return None
        def downstream(module, inputs, output):
            hidden = _hidden(output).float()
            v = tensors(hidden.device)
            state["measurements"]["downstream"] = summary(measure(hidden[0, -1], v, "downstream"))
            for event in state.get("prefill_events", []):
                measured = measure(hidden[0, event["position_indices"]], v, "downstream")
                event["measurements"]["downstream"] = summary(measured)
                event["measurements"]["per_position"]["downstream"] = serialize_measure(measured)
                event["downstream_alignment"] = "includes all first-forward prefill and generation edits, not an isolated prefill causal readout"
        try:
            handles.append(self.blocks[meta["layer"]].register_forward_hook(edit))
            if meta.get("downstream_layer") is not None:
                handles.append(self.blocks[meta["downstream_layer"]].register_forward_hook(downstream))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def generate(self, messages, tools, config, calibration_dir, control=lambda: {},
                 emit=lambda event: None, should_stop=lambda: False, runtime_controls=None):
        import torch
        from .runtime_controls import validate_runtime_controls
        runtime_controls = validate_runtime_controls(runtime_controls)
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
        initial_control = control()
        if isinstance(initial_control, dict) and "schema_version" in initial_control and (type(initial_control["schema_version"]) is not int or initial_control["schema_version"] not in (1, 2)):
            raise ValueError("Unsupported control snapshot schema")
        v2 = isinstance(initial_control, dict) and initial_control.get("schema_version") == 2
        def control_failure(exc, snapshot):
            if runtime_controls:
                snapshot = snapshot if isinstance(snapshot, dict) else {}
                emit({"type": "runtime_control_unavailable", "runtime_controls": deepcopy(runtime_controls),
                      "control_revision": snapshot.get("control_revision"),
                      "generated_token_index": snapshot.get("generated_tokens"),
                      "completed_decisions": snapshot.get("completed_decisions"),
                      "evidence": deepcopy(getattr(exc, "evidence", {"status": "unavailable", "reason": str(exc)}))})
        try:
            if runtime_controls and not v2:
                raise ValueError("Actual-norm matching requires recipe-v2 controller snapshots")
            prepared_initial = self._prepare_control_v2(initial_control, package, runtime_controls) if v2 else None
        except ValueError as exc:
            control_failure(exc, initial_control)
            raise
        eos = self.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        if self.tokenizer.eos_token_id is not None:
            eos_ids.add(self.tokenizer.eos_token_id)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        emitted, raw, streamed = [], "", ""
        cache, phase = None, "reasoning" if thinking else "output"
        reasoning_tokens, finish = 0, "length"
        state = {}
        requested_control = initial_control
        try:
            hook_context = self._hooks_v2(package, state) if v2 else self._hooks(package, state)
            with torch.inference_mode(), hook_context:
                for index in range(limit):
                    if should_stop():
                        finish = "stopped"
                        break
                    requested_control = initial_control if index == 0 else control()
                    if v2:
                        prepared = prepared_initial if index == 0 else self._prepare_control_v2(requested_control, package, runtime_controls)
                        if prepared["preset_identity"] != prepared_initial["preset_identity"] or prepared["recipe_hash"] != prepared_initial["recipe_hash"]:
                            raise ValueError("A running generation cannot change its frozen recipe/presets")
                        state.update(prepared=prepared, first_forward=index == 0)
                        state["dose"] = self._dose_v2(prepared, phase)
                    else:
                        if isinstance(requested_control, dict) and "schema_version" in requested_control and (type(requested_control["schema_version"]) is not int or requested_control["schema_version"] != 1):
                            raise ValueError("A generation cannot switch to another control schema midway")
                        state["dose"] = normalized_control(requested_control, phase)
                    if package is None and any(state["dose"]["effective"].values()):
                        raise ValueError("A calibration package is required for nonzero effects")
                    output = self._forward(input_ids=input_ids, past_key_values=cache, use_cache=True)
                    if v2:
                        for prefill_event in state.get("prefill_events", []):
                            emit(deepcopy(prefill_event))
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
        except ValueError as exc:
            control_failure(exc, requested_control)
            raise
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
                "intervention_scope": "explicit_prefill_then_last_generation_position" if v2 else "last_position",
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
        guarded_npz(out_dir / "vectors.npz", **vectors)
        metadata = {"schema_version": SCHEMA_VERSION, "kind": "legacy_pilot_import",
                    "model_fingerprint": fingerprint, "model_fingerprint_sha256": self.info["fingerprint_sha256"],
                    "layer": manifest["layer"], "downstream_layer": None,
                    "source": str(source_dir), "heldout": None, "selected_dose": None,
                    "interpretation": "Legacy projections reuse the intervention directions. They are not independent probes; no downstream calibration or split validation is available.",
                    "vectors_sha256": hashlib.sha256((out_dir / "vectors.npz").read_bytes()).hexdigest()}
        _write_json(out_dir / "calibration.json", metadata)
        return metadata
