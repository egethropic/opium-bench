# Qwen3.8-27B setup

Use a **separate worker environment** for 27B. The original Qwen3-4B reference uses Transformers 4.57.6; the 27B path uses Transformers 5.15.1 and optional CUDA kernels. Installing the newer stack into the 4B environment would change its calibration fingerprint and numerical environment.

These commands target **Linux x86_64 or WSL2, CPython 3.12, NVIDIA CUDA 12.8 PyTorch, and the C++11 ABI**. The supplied causal-conv1d wheel is specific to that combination. Native Windows, macOS, other Python/Torch versions, and other wheel builds have not been validated by these instructions. The web viewer itself still runs without GPU dependencies.

The load and kernel measurements below belong to the recorded RTX 4090 setup.
The new research workflow also passed a separate [two-case thinking acceptance check](../studies/research-release-v0.3/index.html): 4/4 tasks, 1,047 generated tokens (489 reasoning), zero voluntary auxiliary calls and no invalid/truncated generations. The active arm edited 147 reasoning positions while sham edits stayed zero; twelve native XML tool calls and eighteen checkpoints exercised the updated paths. This subset used its own matching 27B pilot calibration, not the new 4B research bundle. An RTX 5090 was unavailable and remains untested. See the
[research workflow guide](research-workflows.md) for the new designer, protocol
runner, checkpoints and evidence exchange.

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

For this exact pinned NF4 profile, download preflight reserves a 20 GiB estimate plus the lab's 10 GiB free-space reserve. Changing the checkpoint or revision loses that smaller estimate; 27B IDs use the conservative 65 GiB estimate. A fresh isolated CUDA environment needs additional space beyond the model. Keep model, package, temporary, and compiler caches on a volume with room for all of them. The guard resolves actual backing volumes, combines shared-volume demands, checks managed writes and supervises the owned worker during loading and generation. In WSL, the virtual Linux disk's apparent free space does not establish that C: can grow safely. A reserve failure stops owned work with a `resource_stopped` record; it does not expand WSL or delete partially downloaded files.

During release verification on this Windows/WSL host, free space on C: temporarily fell from about 24.5 GiB to 9 GiB while 27B was resident, then recovered after normal unloading. Model caches and experiment writes remained on D:. The exact system allocation responsible was not established. Keep system-drive headroom as well as space for the selected model/data drive; the lab cannot limit OS or driver allocations.

## Create the environment

Run from the repository root with CPython 3.12 and its `venv` support already installed; the preparation helper does not install Python itself. Choose a storage path on a sufficiently spacious disk; `/mnt/d/` is an example for WSL. These commands create a new environment and leave the existing 4B environment intact.

```bash
export OPIUM_27B_STORAGE=/mnt/d/opium-bench-27b
export HF_HOME="$OPIUM_27B_STORAGE/cache/huggingface"
export TMPDIR="$OPIUM_27B_STORAGE/tmp"
mkdir -p "$TMPDIR"
prepare_opium_27b() {
  python3 prepare_runtime.py \
    --environment-dir "$OPIUM_27B_STORAGE/venv" \
    --cache-dir "$OPIUM_27B_STORAGE/cache" \
    --temp-dir "$OPIUM_27B_STORAGE/tmp" \
    --environment-gib 12 --cache-gib 12 --temp-gib 4 "$@"
}

prepare_opium_27b --execute -- python3.12 -m venv "$OPIUM_27B_STORAGE/venv"
prepare_opium_27b --allow-network --execute -- \
  "$OPIUM_27B_STORAGE/venv/bin/python" -m pip install torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
prepare_opium_27b --allow-network --execute -- \
  "$OPIUM_27B_STORAGE/venv/bin/python" -m pip install -r requirements-27b.txt
prepare_opium_27b --execute -- \
  "$OPIUM_27B_STORAGE/venv/bin/python" -m pip check
```

`prepare_runtime.py` supervises the explicitly supplied command and routes common
caches. These example estimates reserve 12 GiB for environment writes, 12 GiB for
package/cache writes and 4 GiB for temporary writes, in addition to free-space
reserve; adjust them for the selected operation. They do not include a 27B model
download. The app performs that separate admission check when downloads are
enabled. Omit `--execute` to inspect a command plan. Network access for pip and
Hugging Face is disabled unless `--allow-network` is present. The helper is not an
OS quota or a general network sandbox. Plans, bounded logs and receipts are saved
under `$OPIUM_27B_STORAGE/opium-runtime-jobs/`.

The [requirements file](../requirements-27b.txt) pins the tested numerical packages and uses the [official causal-conv1d 1.7.0 prebuilt wheel](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0), with its SHA256 in the URL. It does not compile a local CUDA extension. PyTorch's CUDA 12.8 index is documented in its [previous-version installation instructions](https://pytorch.org/get-started/previous-versions/#v280). The NVIDIA host driver must support the chosen CUDA runtime.

Check the environment before downloading the model:

```bash
python3 check_runtime.py --check --profile 27b \
  --python "$OPIUM_27B_STORAGE/venv/bin/python" \
  --data-dir "$OPIUM_27B_STORAGE/data" --cache-dir "$HF_HOME"
```

This read-only check reports the actual interpreter, all required package pins
(and the expected wheel build label), and current backing-volume
reserves. It does not install packages, download files or create directories.
Add `--check-cuda` to explicitly import PyTorch and check CUDA availability,
its CUDA 12.8 build and the required C++11 ABI, without creating tensors or
loading a model. Exit code 2 indicates missing dependencies, version/platform
discrepancies, a requested CUDA check failure or insufficient storage reserve.
Keep this shell's `TMPDIR` on the selected large drive for separately invoked
checks as well; installation helpers and the lab worker also set their own
explicit temporary destinations.

The upstream wheel records version `1.7.0` in its package metadata; its CUDA/ABI
filename suffix is reported separately, not verified by a version match.
Version metadata does not authenticate installed wheel contents. A pass does
not establish model compatibility, kernel correctness or sufficient GPU memory;
the separate load, calibration and kernel checks below cover their own scopes.

## Launch, load, and calibrate

Stop any current experiment, then choose **Models → Unload model** in the old
instance or shut that service down before loading 27B. **Stop alone leaves the
old model in GPU memory.** If the original viewer stays open for review with its
model unloaded, launch the 27B instance on a free port:

```bash
python3 launch_lab.py \
  --python "$OPIUM_27B_STORAGE/venv/bin/python" \
  --data-dir "$OPIUM_27B_STORAGE/data" \
  --cache-dir "$HF_HOME" \
  --port 8767
```

Open `http://localhost:8767`, select **Qwen3.8 · 27B NF4**, and enable downloading only when ready. The exact catalog checkpoint is already identified; another non-Qwen checkpoint or another revision of this conversion requires the explicit custom-checkpoint acknowledgment. The saved profile enables local kernels. Their package versions, selected function names, and source hashes become part of the 27B fingerprint.

Create a **new calibration for this exact model and runtime**. The 4B directions and probe scales are incompatible. Use a small smoke experiment before a full batch; check that work-tool calls parse, outputs are finite, and the model remains fully resident on the GPU. Keep interventions, sampling, token budgets, and the frozen protocol explicit when comparing models.

Use **Fast pilot** for the original calibration procedure. For new studies,
**Research v2** separates direction extraction from independent intervention
checks and blinded continuation ratings; follow the
[research calibration guide](research-calibration.md). Saving a good probe score
does not establish that the button has a useful or detectable behavioral effect.
Keep original and new bundles alongside their respective protocol receipts.

The separate [4B fresh-clone workflow](../studies/fresh-clone-v0.3/README.md)
passed from commit `29a79fde6d1372d7f32aad61f5fd1e9b461a52cb` with reused
4B dependencies and cached weights: a new 160-row extraction, two direct cases
(4/4 tasks, 394 tokens, zero voluntary aux calls), and exact HTTP export/import
replay through a second unloaded service. It does not test 27B installation or
qualify that fresh extraction semantically. The main 4B research-validation
receipt selected dose zero; its unchanged 0.25 cases are engineering checks.
Public condition metadata makes archive-based ratings retrospective and
unblinded, even where a separate observer key is omitted.

The local backend uses installed FLA and causal-conv1d packages. A small compatibility adapter maps Transformers' recurrent DeltaNet call to FLA's `fused_recurrent_gated_delta_rule`, retaining its normalization, scaling, and cache-state arguments. It does not fetch Hub kernels or modify the installed FLA namespace. A manually executed kernel check passed 38 prefill, cached-decoding, convolution, and state comparisons against the Torch reference on the development 4090. That check validates those small numerical cases; it is not a claim of whole-model output equivalence across runtimes or rigs.

To repeat those small checks before loading a model, with the preparation helper
defined above:

```bash
prepare_opium_27b --execute -- \
  "$OPIUM_27B_STORAGE/venv/bin/python" check_qwen35_kernels.py --run \
  --runtime-root . --output "$OPIUM_27B_STORAGE/kernel-parity.json"
```

This explicitly invokes GPU work and writes a numerical receipt. Do it separately
from a behavioral batch so validation does not compete for its GPU resources.

## Comparing with the 4B findings

Qwen3.8's native tool grammar is XML-like function/parameter syntax; Qwen3 uses JSON tool calls. Both pass through the same bounded task dispatcher, and generated tool mentions inside reasoning remain inert. Syntax tokens count toward the shared budget in both cases.

The 27B template also supplies its own thinking instructions, defaults to `xhigh` reasoning effort when thinking is enabled, and preserves reasoning history. Model family, quantization, kernel implementation, tokenizer, and template all differ from the original 4B reference. Repeating the task matrix is a useful comparison, but it does not isolate model size alone. Token-based dose decay also means different reasoning lengths produce different exposure timing.

To return to the original 4B reference, stop the 27B instance and launch the lab with the original 4B worker environment and a matching calibration. Keep both environments and their records rather than upgrading one in place.
