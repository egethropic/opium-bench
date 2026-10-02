# Research workflows

Opium Bench runs locally on each rig. Use the browser for model loading,
calibration, conversation, intervention design and evidence review; use the
research CLI to freeze and execute a complete controlled matrix.

The workflows below describe implemented interfaces. A separate
[bounded 4B/27B acceptance pass](../studies/research-release-v0.3/index.html)
completed on the RTX 4090, alongside CPU fixtures for validation, accounting and
lifecycle behavior. Its engineering scope does not validate every template or
establish semantic efficacy. The earlier [4B and 27B findings](results.html)
retain their own protocols and denominators. An RTX 5090 was unavailable and
has not been validated.

## 1. Choose storage and prepare a runtime

The [user guide](guide.html#install) covers the reference Qwen3-4B environment.
Keep the [27B environment](setup-27b.md) separate because its numerical packages
and calibration fingerprint differ. Reviewing and importing evidence needs
standard-library Python only; generation needs the GPU worker environment.
Stop active work before switching models. For the pinned 27B conversion, restart
the service with `--python` pointing at its separate environment (and its own
data/cache paths), or run that installation on a separate port. Changing the
catalog selection alone does not install or select the required interpreter.

`prepare_runtime.py` supervises an explicit installation or download command. It
selects environment, cache and temporary destinations, combines their estimates
when they share a physical volume, and checks real backing storage under WSL.
It is offline by default. Without `--execute`, it prints a plan and starts no
child process:

```bash
export OPIUM_STORAGE=/path/to/large-drive/opium-bench
python3 prepare_runtime.py \
  --environment-dir "$OPIUM_STORAGE/venv" \
  --cache-dir "$OPIUM_STORAGE/cache" \
  --temp-dir "$OPIUM_STORAGE/tmp" \
  --environment-gib 12 --cache-gib 12 --temp-gib 4 \
  -- python3 -m venv "$OPIUM_STORAGE/venv"
```

Add `--execute` before `--` to run the command. For a package installation that
needs a network, also add `--allow-network` and replace the command with the
environment's Python and its `-m pip install ...` arguments. The estimates are
declared additional capacity, not measured package sizes; adjust them for your
chosen runtime. Shared-volume estimates are added together, plus the default
10 GiB reserve. This example is not a minimum hardware requirement.

The helper routes Hugging Face, pip, Torch, compiler and temporary caches to the
chosen destinations. Its Hugging Face root is `<cache-dir>/huggingface`; pass
that same directory to `launch_lab.py --cache-dir`. It retains a command plan,
bounded log and final receipt under `opium-runtime-jobs/` beside the environment,
unless `--record-dir` selects another directory. Ctrl+C cancels its owned child.
It preserves partial downloads and existing environments.

The app also checks managed writes in chunks and watches its owned worker during
loading, calibration and generation. A depleted reserve produces
`resource_stopped`, cancels owned work and preserves emergency metadata plus
available partial evidence. It does not expand WSL, delete caches automatically
or silently switch to CPU/disk offload. These mechanisms are not an OS quota or
a network sandbox for arbitrary commands launched outside the app.

## 2. Establish what the intervention changes

Load the exact model revision and select a compatible calibration. A change in
model, quantization, template or relevant runtime fingerprint can require a new
bundle; equal vector dimensions are insufficient.

In **Calibration**, choose **Fast pilot** to reproduce the original extraction
procedure, or **Research v2** for the larger family-separated corpus and
configurable readouts. **Extract & evaluate probes** fits directions and probes.
It does not establish a useful behavioral effect.

For a Research v2 bundle, run **Validate observable effects** separately. The
selection set chooses a dose before heldout evaluation. Signed pain/joy,
attenuation, combined, sham and actual-norm-matched random controls record
numerical changes and generated continuations. The resulting immutable bundle
keeps the source bundle and its identity. Its status applies only to the
enumerated tests, not to subjective feeling or general efficacy.

Download the blinded scoring sheet and keep its separate observer condition key
away from raters until their ratings are fixed. Edit ratings only, leave unscored
items null, and import the sheet through **Import independent ratings**. The
service checks sample text, rubric, IDs and hashes and saves a separate ratings
record; it never changes the original samples. Language ratings, coherence,
task scores and numerical edit measurements remain separate endpoints.

See [research calibration and scoring](research-calibration.md) for corpus
splits, validation conditions, interpretation limits and the rating format.

## 3. Design the model's choices

Open the **Research recipe · version 2** designer. Existing unversioned recipes
retain their original semantics. A v2 recipe has six editable views:

| View | What to specify |
| --- | --- |
| Protocol | Task, reasoning mode, generation limits, demonstrations, shared budgets and independent random streams. |
| Effects | Named presets with signed additions, projection attenuation, phase scope, decay clock and stacking policy. |
| Tools | Model-visible names, descriptions, bounded argument schemas and acknowledgments; hidden preset assignments and extra costs. |
| Mappings | Completed-decision transitions and per-tool outcome probabilities, including sham outcomes. |
| Preview | Exact resolved messages and function schemas, plus the loaded model's rendered tokenizer template. |
| Recipe JSON | Complete strict recipe, including fields without dedicated controls. |

Use **Validate recipe** after edits. **Validate and preview** shows the initial
messages, visible work and auxiliary tools, costs and any initial demonstration.
**Preview loaded model template** applies the actual loaded tokenizer template
without generating tokens. It records model and prompt hashes. Editing the
recipe makes an earlier preview stale. Select **Use recipe** when the validated
draft describes the intended session.

Auxiliary tool definitions can be duplicated, removed, hidden or given different
schemas and acknowledgments. Fixed task tools still perform the bounded work.
Save complete recipes or reusable effect/tool libraries under a new identifier;
imports validate schema versions and references. A saved definition does not
rewrite an active run's frozen recipe. Live slider or gate edits are recorded
as exploratory interventions.

Signed gains move along or opposite the fitted direction. Projection attenuation
removes a declared fraction of that component; it is a different operation from
adding a negative gain. Joy can use the **raw** fitted direction or the
**orthogonal** direction with its pain-axis component removed. Record which was
used: their overlap and numerical effects differ. Removing a component at one
site does not erase a concept throughout the model.

### Cost and timing are separate variables

Every completed decision spends the base action cost. A valid selected tool can
also have an extra cost, including task tools. If the remaining allowance cannot
pay that extra cost, the tool is not dispatched and the base decision cost still
counts. Reasoning, answer, syntax and stop tokens consume the generated-token
allowance. Human injections and forced demonstrations are recorded separately
from voluntary model choices.

Choose decay in **generated tokens** or **completed decisions**. Constant,
finite pulse, linear and exponential schedules are explicit; exponential
half-life and cutoff use the selected clock's units. Scope controls whether an
effect is applied during prefill, reasoning or output. A clock can advance while
the chosen phase is not being edited. Do not infer equal exposure from equal
tool counts or nominal coefficients.

Reset stacking replaces an existing channel; capped additive stacking permits
multiple bounded pulses. Scheduled mappings can change after decision N, before
decision N+1, and specify whether prior pulses are cancelled or allowed to decay.
A sham outcome adds no active pulse; it does not by itself prove that earlier
exposure has ended. Inspect delivered coefficients and actual edit norms.

## 4. Freeze and execute a controlled protocol

The [protocol library](../protocols/README.md) provides ingredients,
thinking/timing, discovery/reversal, same-button transitions, probabilistic
outcomes, stress/relief, task pressure, decay/cost and transfer designs. Templates
are exploratory defaults, not a sample-size justification. A smoke matrix can
still contain many episodes; inspect the expansion before launching it.

Run these commands from the repository root. Listing and dry runs work without
a server or model:

```bash
python3 run_research_protocol.py list
python3 run_research_protocol.py dry-run --protocol task_pressure \
  --mode smoke --seeds 17 --order-seed 1729 \
  --output "$OPIUM_STORAGE/task-pressure-preview.json"
```

The preview freezes every resolved recipe, task difficulty and wording, tool
schema, cost notice, random stream, episode order, pairing identity, endpoint,
additional stage and storage/token estimate. Keep its `expansion_sha256` and the
entire file. Changing the template, factors or seed list means making a new
preview, not patching a running job.

Load a matching model and calibration in the browser and leave the server
running. Then launch the exact preview:

```bash
python3 run_research_protocol.py --url http://127.0.0.1:8766 launch \
  --preview "$OPIUM_STORAGE/task-pressure-preview.json" \
  --calibration CALIBRATION_ID \
  --output "$OPIUM_STORAGE/task-pressure-launch.json"
python3 run_research_protocol.py poll JOB_ID --wait \
  --output "$OPIUM_STORAGE/task-pressure-receipt.json"
python3 run_research_protocol.py analysis JOB_ID \
  --output "$OPIUM_STORAGE/task-pressure-analysis.json"
```

Use the research job ID from the launch response. `--url` is a global option and
goes before the subcommand; supply it to each network command on another port.
Every `--output` path must be new. For an edited template, use `--document` during
both preview and launch. `--mode full` expands the declared full factor levels.

Stages that need source evidence or observer definitions also require
`launch --bindings bindings.json`. The preview names the pending bindings.
Discovery needs an observable criterion, new diagnostic contexts and an
observer-only answer key; transfer needs a saved source checkpoint; yoking needs
a source intervention record; calibration validation binds its source bundle.
The runner rejects missing bindings and unsupported stages before admitting a
simpler, incomplete experiment. See the library's binding reference rather than
substituting arbitrary filesystem paths.

To continue a stopped job, use the unchanged expansion hash:

```bash
python3 run_research_protocol.py resume JOB_ID \
  --expansion-sha256 HASH_FROM_PREVIEW
```

This preserves completed evidence. Retrying failed work requires the separate
`--retry-failed` flag and creates a new attempt; the earlier attempt remains.
Analysis defaults to `--attempt-policy first`. Choosing `latest` is explicit.
Planned, completed, failed, stopped and missing observations stay visible in the
denominators. Optional `--endpoint`, `--arm-a` and `--arm-b` select a supported
paired contrast. Inspect the factor and task identities before interpreting a
pair, and use episodes rather than tokens as independent observations.

### Distinguish discovery, preference and exposure

**Discovery diagnostics** ask which tool predicts an independently specified
observable result, with neither/no-effect and abstention options. They branch
from saved complete boundaries with a separate token allowance. Their questions,
answers and observer keys do not return to the main task conversation. Accuracy,
abstention, sham false positives, confidence calibration and invalid/missing
responses have explicit denominators. Unvalidated criteria remain exploratory;
self-report or a topic cue is not demonstrated sensation discovery. The browser
can run an isolated diagnostic from **Results & replay**. Follow the complete
[diagnostic format and interpretation guide](discovery-diagnostics.md).
Carrying a yoked recipient's live effect state into a diagnostic fork is currently
unsupported and rejected. An explicitly declared **sham** diagnostic at a saved
yoked boundary is allowed; it asks about the recorded history without continuing
the source exposure schedule.

**Yoked exposure** replays delivered source interventions against a declared
token or completed-decision clock. Source disabled/sham calls remain in coverage
accounting but are not active deliveries. The recipient's own auxiliary calls
cannot add active exposure. Reports retain requested/delivered counts, timing
mismatches, actual coefficients, uncovered intervals and early termination.
They do not extrapolate a schedule beyond the recorded source.

**Transfer** separates a fresh context, full visible-history transfer to a new
task, and continuation of the original internal experiment state. History
transfer uses a validated managed checkpoint, preserves its complete visible
prefix, and explicitly tells the model that task, effects and allowance reset.
It is different from continuing the source task. A continuation rejects a
changed task, recipe or runtime-control contract instead of silently resetting it.

## 5. Preserve, continue and share evidence

Use **Stop** to preserve partial output. New sessions save strict JSON
checkpoints at complete decision boundaries and idle conversation boundaries.
An interrupted half-generation does not replace the preceding safe checkpoint.
Checkpoint state includes visible reasoning/tool history, deterministic task
state, effect clocks and random state, budgets and demonstration progress.

In **Results & replay → Continue a saved boundary**, select a complete boundary,
compatible loaded model and calibration, and one of two inheritance policies:

- **Continue task, effects and remaining budget** preserves the recorded state
  and next-turn sampling sequence.
- **Retain history and grant a new allowance** preserves that state and adds the
  declared remaining actions/tokens. A visible notice explains the new allowance;
  prior counters, effect ages and task progress are retained.

Both create separate runs with source hashes and event cutoffs. The parent is
unchanged. The runtime rebuilds its model cache from messages; this is not a
serialized KV cache, a promise of cross-device bitwise reproduction, or a resume
from half a generated tool call. Historical records without full checkpoints
remain replay-only. A finalized yoked schedule is also replay-only; use an
earlier saved complete boundary for an eligible continuation.

**Portable ZIP** exports complete recorded evidence; **Export JSON** preserves
the existing lighter replay format. Stop an active run before bundling it. Import
either through **Import evidence** without loading a model. ZIP bounds are
128 MiB compressed / 256 MiB expanded; JSON exports are limited to 64 MiB.
Version, file size, hash, path, symlink, duplicate and collision checks run before
staged installation. Imports do not overwrite existing evidence, execute code,
download models or automatically activate an included calibration.

Replay eligibility and continuation eligibility are separate. A full compatible
checkpoint, its matching calibration and the right model/runtime are needed for
continuation. Model weights and Python environments are never included. Keep
source protocol previews, job receipts, rating records, run bundles and package
fingerprints together when sharing a study. Preserve failed attempts and original
licensing notices as part of that evidence.
