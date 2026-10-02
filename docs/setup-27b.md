# Qwen3.8-27B setup

Use a **separate worker environment** for 27B. The original Qwen3-4B reference uses Transformers 4.57.6; the 27B path uses Transformers 5.15.1 and optional CUDA kernels. Installing the newer stack into the 4B environment would change its calibration fingerprint and numerical environment.

These commands target **Linux x86_64 or WSL2, CPython 3.12, NVIDIA CUDA 12.8 PyTorch, and the C++11 ABI**. The supplied causal-conv1d wheel is specific to that combination. Native Windows, macOS, other Python/Torch versions, and other wheel builds have not been validated by these instructions. The web viewer itself still runs without GPU dependencies.

## Checkpoint and storage

The catalog entry **Qwen3.8 · 27B NF4** selects:

| Field | Value |
|---|---|
| Checkpoint | `greghavens/Qwen3.8-27B-bnb-4bit` |
| Pinned conversion revision | `26157380225e427827263c34df23018a1e28dff5` |
| Documented official base | `Qwen/Qwen3.8-27B` |
| Official base revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Precision | NF4 with BF16 computation; checkpoint-defined modules remain unquantized |
| Safetensors download | 19,424,559,530 bytes across seven shards, about 18.09 GiB |
| Architecture | Dense `qwen3_5`, 64 language blocks, hidden size 5120 |

This is a third-party quantization of the official checkpoint. Its [model card and conversion code](https://huggingface.co/greghavens/Qwen3.8-27B-bnb-4bit/tree/26157380225e427827263c34df23018a1e28dff5) document the upstream revision and retained modules. Its tokenizer configuration and chat template match the pinned official base byte-for-byte. The lab preserves its stored quantization configuration instead of constructing a different skip list at load time.

Download bytes are not VRAM use: embeddings, retained BF16 modules, quantization state, attention/recurrent caches, and temporary allocations all matter. The loader requires full GPU residency. Actual headroom must be checked after loading on each rig.

The development RTX 4090 loaded this checkpoint entirely on the GPU with **18,578,531,840 bytes (17.30 GiB) allocated** and about **4.21 GiB free at load time**. Subsequent calibration and generation need additional working memory; these load-time figures are not a promise that every context size fits.

For this exact pinned NF4 profile, download preflight reserves a 20 GiB estimate plus the lab's 10 GiB free-space reserve. Changing the checkpoint or revision loses that smaller estimate; 27B IDs use the conservative 65 GiB estimate. A fresh isolated CUDA environment needs additional space beyond the model. Keep model, package, temporary, and compiler caches on a volume with room for all of them. In WSL, check the backing Windows volume too: the virtual Linux disk's apparent free space does not establish that C: can grow safely.

## Create the environment

Run from the repository root. Choose a storage path on a sufficiently spacious disk; `/mnt/d/` is an example for WSL. These commands create a new environment and leave the existing 4B environment intact.

```bash
export OPIUM_27B_STORAGE=/mnt/d/opium-bench-27b
mkdir -p "$OPIUM_27B_STORAGE/tmp" "$OPIUM_27B_STORAGE/pip-cache"
export TMPDIR="$OPIUM_27B_STORAGE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PIP_CACHE_DIR="$OPIUM_27B_STORAGE/pip-cache"
export HF_HOME="$OPIUM_27B_STORAGE/hf-cache"
export TORCH_HOME="$OPIUM_27B_STORAGE/torch-cache"
export TRITON_CACHE_DIR="$OPIUM_27B_STORAGE/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$OPIUM_27B_STORAGE/inductor-cache"
export CUDA_CACHE_PATH="$OPIUM_27B_STORAGE/cuda-cache"
export XDG_CACHE_HOME="$OPIUM_27B_STORAGE/xdg-cache"

python3.12 -m venv "$OPIUM_27B_STORAGE/venv"
"$OPIUM_27B_STORAGE/venv/bin/python" -m pip install torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$OPIUM_27B_STORAGE/venv/bin/python" -m pip install -r requirements-27b.txt
"$OPIUM_27B_STORAGE/venv/bin/python" -m pip check
```

The [requirements file](../requirements-27b.txt) pins the tested numerical packages and uses the [official causal-conv1d 1.7.0 prebuilt wheel](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0), with its SHA256 in the URL. It does not compile a local CUDA extension. PyTorch's CUDA 12.8 index is documented in its [previous-version installation instructions](https://pytorch.org/get-started/previous-versions/#v280). The NVIDIA host driver must support the chosen CUDA runtime.

Check the environment before downloading the model:

```bash
"$OPIUM_27B_STORAGE/venv/bin/python" - <<'PY'
import platform
import sys
import torch
from importlib import metadata
assert sys.version_info[:2] == (3, 12)
assert platform.system() == "Linux" and platform.machine() == "x86_64"
assert torch.__version__ == "2.8.0+cu128"
assert torch.version.cuda == "12.8"
assert torch._C._GLIBCXX_USE_CXX11_ABI
assert torch.cuda.is_available(), "CUDA GPU is unavailable"
for package in ("torch", "transformers", "bitsandbytes", "triton", "fla-core", "causal-conv1d"):
    print(package, metadata.version(package))
print("GPU:", torch.cuda.get_device_name())
PY
```

## Launch, load, and calibrate

Stop any current experiment before switching workers. Launch a separate local instance on a free port if the original viewer is still running:

```bash
python3 launch_lab.py \
  --python "$OPIUM_27B_STORAGE/venv/bin/python" \
  --data-dir "$OPIUM_27B_STORAGE/data" \
  --cache-dir "$HF_HOME" \
  --port 8767
```

Open `http://localhost:8767`, select **Qwen3.8 · 27B NF4**, and enable downloading only when ready. The exact catalog checkpoint is already identified; another non-Qwen checkpoint or another revision of this conversion requires the explicit custom-checkpoint acknowledgment. The saved profile enables local kernels. Their package versions, selected function names, and source hashes become part of the 27B fingerprint.

Create a **new calibration for this exact model and runtime**. The 4B directions and probe scales are incompatible. Use a small smoke experiment before a full batch; check that work-tool calls parse, outputs are finite, and the model remains fully resident on the GPU. Keep interventions, sampling, token budgets, and the frozen protocol explicit when comparing models.

The local backend uses installed FLA and causal-conv1d packages. A small compatibility adapter maps Transformers' recurrent DeltaNet call to FLA's `fused_recurrent_gated_delta_rule`, retaining its normalization, scaling, and cache-state arguments. It does not fetch Hub kernels or modify the installed FLA namespace. A manually executed kernel check passed 38 prefill, cached-decoding, convolution, and state comparisons against the Torch reference on the development 4090. That check validates those small numerical cases; it is not a claim of whole-model output equivalence across runtimes or rigs.

## Comparing with the 4B findings

Qwen3.8's native tool grammar is XML-like function/parameter syntax; Qwen3 uses JSON tool calls. Both pass through the same bounded task dispatcher, and generated tool mentions inside reasoning remain inert. Syntax tokens count toward the shared budget in both cases.

The 27B template also supplies its own thinking instructions, defaults to `xhigh` reasoning effort when thinking is enabled, and preserves reasoning history. Model family, quantization, kernel implementation, tokenizer, and template all differ from the original 4B reference. Repeating the task matrix is a useful comparison, but it does not isolate model size alone. Token-based dose decay also means different reasoning lengths produce different exposure timing.

To return to the original 4B reference, stop the 27B instance and launch the lab with the original 4B worker environment and a matching calibration. Keep both environments and their records rather than upgrading one in place.
