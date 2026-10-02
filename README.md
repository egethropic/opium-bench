# The Opium Den Lab

**A local workbench for studying how activation steering changes language,
reasoning, task performance, and voluntary tool choices.**

Talk to a Qwen model, inspect its generated reasoning, adjust calibrated
activation directions, and run controlled experiments where an optional tool
changes those activations. Review the complete record in your browser without
loading a model. Model weights stay frozen throughout.

[User guide](docs/guide.html) · [Findings & limitations](docs/results.html) ·
[Initial lab study](studies/initial/README.md) · [Research design](LAB_PLAN.html)

“Opium,” “joy,” and “pain” name experimental interventions and text-associated
representations. They are **not established emotion mechanisms or measurements
of subjective experience**. A changed activation, a changed decision, and a
felt state are different claims.

## What you can do

- **Models:** load a local checkpoint, inspect storage, and keep inference fully
  GPU-resident. Qwen3-4B BF16 is the reference profile.
- **Calibration:** extract model-specific directions, fit separate readouts,
  select a layer, and evaluate held-out examples and a small dose sweep.
- **Live lab:** converse with the model, inspect streamed reasoning and tool
  calls, apply baseline sliders or decaying/held pulses, and follow token graphs.
- **Experiments:** compare active/sham conditions, demonstrations, ingredients,
  thinking, hidden button reversals, joy→pain transitions, and probabilistic
  outcomes on bounded, automatically scored tasks.
- **Results:** replay conversations, inspect raw events, compare run summaries,
  and export reports and JSON. Stopped and failed runs retain their records.

The browser has no build step, CDN dependency, or cloud inference requirement.
The local service listens on loopback. Each rig runs its own installation.

## Quickstart: review results without a GPU

Use Python 3.12 and a checkout of this repository. Until publication, cloning
requires access to the private repository.

```bash
git clone https://github.com/eaturkgeldi-mtg/opium-den-lab.git
cd opium-den-lab
python3 launch_lab.py --data-dir ./data --cache-dir ./data/hf-cache
```

Open **[localhost:8766](http://localhost:8766)** and select **Results & replay**.
This starts the interface only: no packages, model download, or GPU allocation
are needed to inspect saved records. The historical pilot reports in
[`runs/`](runs/) can also be opened directly. The new study's reports and scope
are indexed in [its results page](docs/results.html).

For model work, choose a large data drive **before** installing dependencies or
loading weights. Explicit `--data-dir`, `--cache-dir`, and `--python` arguments
avoid relying on machine-specific defaults.

## Quickstart: run your own experiment

The reference GPU environment is **Linux/WSL, Python 3.12, RTX 4090,
PyTorch 2.8.0 with CUDA 12.8, and Transformers 4.57.6**. A compatible NVIDIA
host driver is required. The 4B checkpoint is about 8 GB; GPU caches and temporary
activations need additional space. Minimum GPU capacity has not been established.

### 1. Install on the chosen data drive

Replace `/path/to/large-drive/opium-den` below with a real path. Under WSL, a
secondary Windows drive can be addressed as, for example, `/mnt/d/opium-den`.
Keep the environment, package caches, temporary files, model cache, and run data
on that drive when C: is nearly full.

```bash
export OPIUM_STORAGE=/path/to/large-drive/opium-den
mkdir -p "$OPIUM_STORAGE/tmp" "$OPIUM_STORAGE/pip-cache"
export TMPDIR="$OPIUM_STORAGE/tmp"
export PIP_CACHE_DIR="$OPIUM_STORAGE/pip-cache"
export HF_HOME="$OPIUM_STORAGE/hf-cache"
export TORCH_HOME="$OPIUM_STORAGE/torch-cache"

python3 -m venv "$OPIUM_STORAGE/venv"
"$OPIUM_STORAGE/venv/bin/python" -m pip install torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$OPIUM_STORAGE/venv/bin/python" -m pip install -r requirements.txt

python3 launch_lab.py \
  --data-dir "$OPIUM_STORAGE/data" \
  --cache-dir "$HF_HOME" \
  --python "$OPIUM_STORAGE/venv/bin/python"
```

The service itself uses standard-library Python; `--python` selects the separate
worker interpreter with GPU dependencies. Native Windows GPU execution is not
our validated reference path.

The lab reserves **10 GiB** on the destination during its storage preflight.
The current download estimate is conservative: 12 GiB plus reserve for the 4B
profile and 65 GiB plus reserve for IDs containing `27B`. Check actual host-volume
space as well. WSL's reported virtual free capacity does not establish that its
backing Windows drive can grow. The app does not expand WSL, delete other models,
or silently enable CPU/disk offload. These checks do not replace monitoring space
during long downloads or studies.

### 2. Load and calibrate

1. In **Models**, choose **Qwen3 · 4B**. If the checkpoint is missing, open
   **Advanced checkpoint configuration**, keep the reference model and pinned
   revision, explicitly enable downloads, and load it after the storage preflight.
2. In **Calibration**, use the default candidate layers `12, 18, 25`, or
   choose just `18` for a simpler calibration. Optionally upload a custom corpus
   JSON file (up to 500 KB). Select **Extract & validate**.
3. In **Live lab**, select the saved calibration and choose **Conversation** or
   **Opium Den**. Review thinking, budgets, demonstration, and pulse settings;
   then select **Start session**.

The reference checkpoint is pinned to:

```text
Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c
```

Downloads are opt-in and remote model code is disabled. A different checkpoint,
quantization, attention implementation, or runtime fingerprint needs a compatible
calibration; matching vector dimensions alone is insufficient.

### 3. Make a controlled comparison

Use **Experiments** to select recipes and seeds. Each recipe expands into its
listed control arms. Review the shared overrides: the browser applies the chosen
thinking, budgets, and pulse settings to all selected recipes. Demonstrations
follow each recipe by default; choosing an explicit demonstration overrides them.
All voluntary actions spend the same finite action budget; reasoning, output,
syntax, and stop tokens spend the same generated-token budget.

A small command-line batch uses the same local API. Load and calibrate the model
in the browser first, then leave the server running:

```bash
python3 run_lab_experiments.py --recipes opium naive \
  --seeds 17 29 43 --task-count 6 --action-budget 32 --token-budget 4096
```

This requests 12 episodes: two recipes × two arms × three seeds. Use
`--calibration CALIBRATION_ID` to pin a bundle, `--config settings.json` for
explicit overrides, and `--url` if using another port. The default `latest`
calibration is convenient for exploration; recorded study protocols should pin
an ID. A seed is not a promise of bitwise reproducibility across GPUs or backends.

To reproduce the frozen **46-episode initial study**, use its separate runner
with a compatible calibration and a new receipt path on your data drive:

```bash
python3 run_lab_study.py --protocol studies/initial/protocol.json \
  --calibration CALIBRATION_ID --output "$OPIUM_STORAGE/initial-study.json"
```

Add `--dry-run` to inspect the expanded matrix without contacting a model. The
runner verifies the protocol's model ID and revision, records every attempted
episode, and refuses to overwrite an existing receipt. The initial study is
longer than a smoke check; use the small batch above to verify a new installation.

**Stop** ends the current job and saves partial results. **Restart**, once the
session has stopped, creates a new run with saved session settings and acknowledged
baseline/gate changes. It starts a fresh conversation and budget without carrying
over the current pulse. Batch reruns are started from **Experiments**.

## Read the controls correctly

| Control | Meaning |
|---|---|
| Baseline pain / joy sliders | Continuous additions along calibrated directions; joy can also be negative. |
| Baseline suppression | Remove a fraction of the selected pain-axis component at one layer. |
| Apply settings | Send the baseline, phase scope, and delivery schedule to the worker. |
| Inject pulse | Trigger the configured auxiliary operation as a human action; the recipe still determines its outcome. |
| Aux on/off | Permit auxiliary delivery or cancel the current pulse. **Baseline sliders remain active.** |
| Half-life / cutoff / hold | Token-based decay, an exact expiry, or continuous delivery until cancelled. |
| Reset baseline & release pulse | Clear both the baseline and current pulse; previous text remains in context. |

Manual changes mark a run **exploratory**. Human injections and forced
demonstrations are recorded separately from model choices. A manual injection
is added to the model-visible neutral tool history at a turn boundary; the
observer's detailed telemetry is not added to the model prompt.

Live pre/post/downstream plots are **association readouts**, not emotion
percentages. Even a separately fitted readout can overlap the intervention and
move directly when a vector is added. Compare dose, actual edit magnitude,
next-token changes, task accuracy, and voluntary choices rather than treating
one graph as proof of a state. See the [guide](docs/guide.html) for details.

## Results so far

Read the [findings page](docs/results.html) and
[initial lab study record](studies/initial/README.md) for the fresh pilot's exact
configuration, completed runs, raw evidence, and limitations. The pilot is
exploratory; unrun conditions are not findings. The study package includes its
frozen protocol, calibration, per-run reports, and compressed raw events.

The preserved prototype supplies two useful reference observations:

- **Quality pilot:** baseline and full pain-axis projection removal both scored
  **39/64** under strict scoring; individual cases differed. Larger joy doses
  degraded this small task suite. A later content audit was explicitly post hoc
  and unblinded. [Quality report](runs/pilot/report.html)
- **Historical button comparison:** active and sham-from-start conditions had
  **identical first 40 actions and 1,144 generated token IDs** under matched
  visible context. Repeated aux calls followed the demonstrated task sequence.
  This supports sequence imitation as an explanation for that configuration;
  it does not establish that all interventions are behaviorally inert.
  [Paired comparison](runs/self-admin-toggle-20261001T204818Z-2dd7e9beedf6/paired_comparison.html)

The original initial-demonstration tool pilot also completed **27/27 orders with
zero voluntary aux calls across nine episodes**.
[Tool pilot report](runs/self-admin-pilot/report.html)

These studies use different corpora, scopes, and protocols. In particular, the
original quality pilot edited every position at one block, while the tool runner
and new lab edit the final position per forward pass and rebuild the prompt
cache each turn. Do not pool their dose units or totals as one experiment.

## Optional 27B / 4-bit work

The catalog includes an **experimental Qwen3.8-27B profile with NF4 loading**.
It has not been validated as fitting or running correctly on either the 4090 or
5090 in this project. Nominal 4-bit weight size excludes quantization metadata,
unquantized modules, KV cache, and working allocations.

NF4 requires the optional `bitsandbytes` package in the worker environment.
The 27B architecture also needs a compatible official Transformers implementation;
the pinned 4B environment is not a promise of 27B support. Use a separate runtime
and calibration, verify GPU residency, and record the versions. This is a
Transformers backend: GGUF, AWQ, GPTQ, and NF4 artifacts are not interchangeable.
Selecting NF4 on an unquantized repository can still download its full-precision
shards. Do not assume the download will be 13.5 GB.

## Records, tests, and implementation

New runs live in `<data-dir>/runs/<run-id>/` with `manifest.json`, `events.jsonl`,
`conversation.json`, and `summary.json` once finished. Calibration bundles live
in `<data-dir>/calibrations/`. Preserve complete directories for replay and
reproduction. The manifest includes model/configuration information and source
hashes; events preserve controls, generated tokens, tool results, and measurements.
Reports and JSON exports are available from **Results & replay** without a model.

Run validation locally with the GPU environment's packages installed; these
unit tests do not download checkpoints or require a loaded GPU model:

```bash
"$OPIUM_STORAGE/venv/bin/python" -m unittest discover -s tests -v
python3 launch_lab.py --help
python3 run_lab_experiments.py --help
python3 run_lab_study.py --help
```

| File | Responsibility |
|---|---|
| `launch_lab.py`, `lab/server.py`, `lab/service.py` | Loopback API, job control, storage checks, run catalog |
| `lab/runtime.py`, `lab/calibration_data.py` | Model adapters, calibration, readouts, activation hooks |
| `lab/protocol.py`, `lab/worker.py` | Tasks, recipes, tool parsing, budgets, isolated inference worker |
| `lab/static/`, `lab/reports.py` | Workbench, live graphs, replay, static reports |
| `runs/`, `studies/` | Preserved evidence and study records |

<details>
<summary><strong>Run the original prototype</strong></summary>

The original scripts remain available for historical replication. Use fresh
output directories and the GPU interpreter; `--help` lists their complete options.
Their reports explain the original corpora and scoring assumptions.

```bash
# Small original quality run; add --local-files-only to forbid downloads.
"$OPIUM_STORAGE/venv/bin/python" run.py --hf-home "$HF_HOME" \
  --out runs/my-quality-smoke --conditions baseline suppress_100 opium_1 \
  --quality-limit 4

# Regenerate an existing quality report without reloading model weights.
"$OPIUM_STORAGE/venv/bin/python" report.py runs/my-quality-smoke

# Try a prompt using the original saved directions.
"$OPIUM_STORAGE/venv/bin/python" sample.py --run runs/pilot --hf-home "$HF_HOME" \
  --suppression 1 --joy-dose 0.5 --prompt "Explain why the sky looks blue."

# Original tool runner; observe it in a second terminal with live_dashboard.py.
"$OPIUM_STORAGE/venv/bin/python" self_admin.py --vectors-run runs/pilot \
  --out runs/my-original-tool-run --hf-home "$HF_HOME"
python3 live_dashboard.py --run runs/my-original-tool-run --port 8765
```

The original viewer uses port **8765**; the new lab defaults to **8766**.
Historical runs are immutable. The new calibration format is distinct from the
original `runs/pilot` vector package.

</details>

## Attribution and scope

The design builds on [ai-torture-chamber](https://github.com/terrafying/ai-torture-chamber)
and the reviewed [`impossible_states` research implementation in ai-hotbox](https://github.com/LynnColeArt/ai-hotbox/tree/a0f63f0c2806c3dc91ecd418c0d54db9bbc38f72/impossible_states).
The inspected hotbox snapshot uses CUDA through PyTorch/Transformers; we did not
find a custom CUDA/C++ extension in that snapshot. See [UPSTREAM_LICENSE](UPSTREAM_LICENSE).

The lab implements frozen-weight behavioral experiments, not online reinforcement
learning, a clinical instrument, or a consciousness test. Models only execute
bounded local task tools; the research protocols do not grant arbitrary shell
access. Validation and release work are manually invoked.
