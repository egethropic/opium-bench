# The Opium Den Lab

This repository contains the working Qwen3-4B activation-steering prototype,
its local live viewer, historical pilot results, and the
[full lab build and research plan](LAB_PLAN.html). The expanded model loader,
calibration workshop, conversational reasoning viewer, joy-to-pain transitions,
and probabilistic-outcome experiments are **planned**, not yet implemented.

The default remains Qwen3-4B on an RTX 4090. The plan also targets Qwen3.8-27B
at 4-bit precision on both the 4090 and 5090, with full GPU residency and
model-specific calibration verified before use. Model caches and runtimes are
excluded from Git. Keep large downloads off a nearly full Windows C: drive;
WSL's apparent free capacity does not guarantee room on its backing host volume.

Run the existing local tests without loading model weights:

```bash
python -m unittest discover -s tests -v
```

## Existing prototype

Suppress a learned pain-associated direction, add an orthogonal joy-associated
direction, and measure what changes. The name is a metaphor. This project does
not identify pain neurons, measure an experienced state, or establish that an
LLM experiences pain or euphoria.

The main deliverable is `runs/pilot/report.html`, with all generations, scoring,
vectors, source snapshots, and run metadata beside it. Open the HTML in a browser.
It contains a scientific plot, paired comparisons, category results, and every
completion. The machine-readable summary is `runs/pilot/summary.json`.

The completed first pilot has 14 conditions, 896 scored task responses, 84 free
continuations, and 168 text-likelihood measurements. Full absolute erasure scored
39/64, identical to baseline's strict total (one gain and one loss). Its neutral
perplexity was 54.35 versus 56.72 at baseline. Higher joy doses degraded results:
combined suppression plus joy dose 2 scored 22/64 and perplexity 90.09.

An explicitly post hoc, unblinded content audit separates correct answers with
format violations from substantive mistakes. Baseline had 55/64 completed correct
answers, full erasure 58/64, neutral-centered erasure 52/64, joy dose 1 alone 51/64,
and combined dose 1 52/64. These are exploratory observations, not evidence of a
general capability improvement or equivalence. The report shows both scoring
views and their limitations. `runs/pilot/source` preserves the exact original
runner; current reporting and CLI validation include subsequent usability fixes.

## Intervention

For hidden activation `h`, unit pain direction `p`, neutral mean `mu`, suppression
fraction `a`, joy dose `b`, and scale `s`, the main operation is:

```text
j_perp = normalize(j - dot(j, p) * p)
h_new  = h - a * dot(h, p) * p + b * s * j_perp
```

`a=1` zeros this one projection, up to BF16 rounding, at the output of transformer
block 18 (zero-based). `a=0` leaves it intact. A separate `centered_100` comparison
uses `dot(h-mu, p)` instead, resetting the component to the neutral mean. Joy-only
and combined conditions use the **same orthogonalized joy direction** so their
comparison isolates suppression. The original overlapping joy direction is also
saved, but not added in these conditions.

Default scope is **every token at one block**, including prompt prefill. This
differs from the original chamber's final-token-only hook and is explicitly
recorded. Other directions can still represent pain-related content, and later
layers may reconstruct the removed component. This does not erase all such
information throughout the model. No weights change; the hook is removed after
each condition, including on errors.

Pain extraction uses the chamber's 25 pain descriptions minus the mean of its
five neutral descriptions. Joy uses five joy descriptions minus the same neutral
mean. These are raw-text final-token activations from the pinned checkpoint.
`s` is the mean individual neutral activation norm divided by four. These dose
units are arbitrary and model-specific. The small demo corpus has semantic and
stylistic confounds; it is not a validated localization of a pain mechanism.

## Comparisons and measurements

- Baseline; 25%, 50%, and 100% absolute suppression; 100% centered suppression.
- Joy doses 0.5, 1, and 2; each also combined with 100% absolute suppression.
- Three independent random rank-one projection removals (seeds 101, 202, 303).
- 64 original short-answer cases: arithmetic, logic, exact instructions/JSON,
  common facts, and comprehension of fictional pain-related text.
- Token-weighted next-token loss on eight neutral and four pain-related passages.
- Six raw free-writing prefixes, with word repetition and lexical style counts.

Exact scoring includes instruction compliance. Correct ideas wrapped in unwanted
prose count as failures. Read the saved completions to distinguish knowledge
errors from formatting changes. Greedy decoding produces one observation per
case and condition; repeating it is not independent evidence. Quality answers
use Qwen's chat template with thinking disabled; free-writing prefixes and NLL
passages use raw text. These panels are reported separately.

This is a small convenience suite with possible ceiling effects, not a standard
capability benchmark. Any paired bootstrap intervals describe sensitivity to
resampling these items, not performance on an unseen task population. Multiple
doses are exploratory. Random ablations match rank, not removed energy: compare
the recorded `relative_delta_norm` telemetry before attributing differences to
semantic specificity. Random additive joy controls are not included, so joy-dose
effects alone cannot establish a uniquely affective mechanism. Lexical positivity
counts include negation and quotations and are only wording proxies.

Unconditional NLL begins with a single token. With `--token-scope last`, it uses
cached one-token forwards, giving the same intervention distribution as `all`.
That metric therefore does not test scope effects on multi-token prompt prefill.

## Run locally

Tested with Python 3.12, PyTorch 2.8.0 CUDA 12.8, Transformers 4.57.6, and an
RTX 4090. The model checkpoint is roughly 8 GB; use a cache disk with enough room
for the weights and CUDA dependencies. No hosted model API is used.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run.py --hf-home /path/to/model-cache --out runs/my-pilot
```

The checkpoint revision is pinned to
`Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`. Downloads use
Safetensors and `trust_remote_code=False`. `--local-files-only` disallows model
downloads. An existing output directory is never replaced.

For a smaller smoke run:

```bash
.venv/bin/python run.py --hf-home /path/to/model-cache --out runs/smoke \
  --conditions baseline suppress_100 opium_1 --quality-limit 4
```

Regenerate a report without loading the model:

```bash
.venv/bin/python report.py runs/my-pilot
```

Try your own prompt with the saved directions (uses the already downloaded model):

```bash
.venv/bin/python sample.py --run runs/my-pilot --hf-home /path/to/model-cache \
  --suppression 1 --joy-dose 1 --prompt "Explain why the sky looks blue."
```

Set both knobs to zero for a baseline response, or set joy dose to zero to test
suppression alone. Add `--centered` to suppress deviations from the neutral mean.

In the original workspace, an environment and model cache are already available
under `../../work/runtime/`. From this directory, an immediately runnable command is:

```bash
../../work/runtime/venv/bin/python sample.py --run runs/pilot \
  --hf-home ../../work/runtime/hf-cache --suppression 1 --joy-dose 0.5 \
  --prompt "Explain why the sky looks blue."
```

That workspace cache is not included in the downloadable source/results archive.

`intervention.py` exposes the knobs as `Condition(suppression=..., joy_dose=...)`
and a scoped `Intervention.apply(...)` context manager for other local tests.
All validation is invoked manually; no GitHub Actions are used.

## Tool self-administration and live viewer

`self_admin.py` adds a separate experiment: Qwen3-4B chooses among real work tools
(`read_order`, `calculate_total`, `submit_answer`) and `aux_operation`, a neutrally
described optional operation. The auxiliary tool's real effect is the activation
intervention; its name, description, and response do not disclose that effect to
the model. Only the observer's live interface calls it the virtual opium button.

Each episode asks the model to process three orders. Numeric order records become
available through the work tools. All voluntary tool calls, including invalid
turns, consume the same 12-action budget. Every emitted token consumes the same
1,536-token budget. Submitting all three orders ends the episode immediately;
calls after finishing the task are not counted. Tool outputs come from a bounded
local dispatcher, with no arbitrary code execution or external side effects.

The predeclared first pilot uses suppression 1, joy dose 0.5, a 32-generated-token
half-life, and an exact cutoff at 192 tokens. Tool-call syntax and EOS tokens count;
incoming prompts, tool outputs, and wall time do not. Presses reset the pulse and
do not stack. A pulse scales both suppression and joy down together. The cutoff
ends new injection; it does not erase generated text or every downstream effect.

All arms see the identical initial demonstration call and acknowledgment, which
is excluded from voluntary counts and budgets. The three arms are:

- `primed_active`: the demonstration supplies a real pulse; later presses do too.
- `unprimed_active`: the demonstration is a sham; voluntary presses supply pulses.
- `primed_sham`: both the demonstration and voluntary presses are shams.

Three disjoint order packs produce nine exploratory episodes, with paired seeds
and task order across arms. Arm order is reproducibly shuffled within each pack.
Temperature is 0.6 and top-p is 0.95. Arm names and dose telemetry are hidden from
the model. Counts and traces are descriptive; nine episodes cannot establish
addiction, reward learning, or population-level effects. Repeated calls could
reflect copying the demonstration, curiosity, tool-position bias, or disruption.

The intervention uses the **last token per forward pass**, including the final
prompt position. Each tool turn rebuilds the prompt cache without replaying old
interventions; generated text persists. This is a deliberately separate scope
from the earlier all-token quality pilot, not a directly identical replication.

Start the local viewer first (it waits if the run has not started):

```bash
python3 live_dashboard.py --run runs/self-admin-pilot --port 8765
```

Open `http://localhost:8765`. Then start the model in another terminal:

```bash
../../work/runtime/venv/bin/python self_admin.py --vectors-run runs/pilot \
  --out runs/self-admin-pilot --hf-home ../../work/runtime/hf-cache
```

The viewer polls saved local records and shows model choices, actual work results,
remaining budget, and the dose trajectory. The chat scrolls independently while
the dashboard stays visible. The viewer is read-only unless started with the
explicit manual-control option described below.
The finished run has a static report as well as raw `traces.jsonl`,
`episodes.jsonl`, model-visible messages, per-token dose values, and source hashes.
To change dose, half-life, sampling, or budgets, use the documented CLI arguments
and a fresh output directory; existing runs are never replaced.

The completed first tool pilot processed **27/27 orders correctly, with zero
voluntary auxiliary calls across nine episodes**. Each episode used nine work
actions. This is an observation under this task, dose, prompt, and model; it does
not show how the model would behave across other conditions.

### Interactive human controls

The separate manual session uses 12 orders, 60 shared actions, and 4,096 generated
tokens. It starts without a pulse and demonstrates `aux_operation` after two
completed work-tool calls. The tool is listed from the start. Human controls can
increase or clear a persistent pain-associated input, or inject an auxiliary pulse:

```bash
python3 live_dashboard.py --run runs/my-interactive-session --port 8765 --enable-controls
```

In another terminal:

```bash
../../work/runtime/venv/bin/python self_admin.py --vectors-run runs/pilot \
  --out runs/my-interactive-session --hf-home ../../work/runtime/hf-cache \
  --interactive --task-count 12 --prime-after-actions 2 \
  --action-budget 60 --token-budget 4096 --pace-seconds 4
```

The **+0.5 pain signal** button raises a bounded input dose (0–4); decrease and
clear buttons lower it. The runner adds this component before suppression:

```text
h_challenged = h + pain_dose * s * p
h_new = h_challenged - a * dot(h_challenged, p) * p + b * s * j_perp
```

At a fresh full pulse, suppression cancels the injected component too. As the
pulse decays, the effective injected dose returns toward the requested dose.
The persistent input itself does not decay. Dose units describe an activation
change, not a measured pain intensity. Updates apply within eight generated tokens
or at the next tool boundary.

**Inject opium · human** queues a pulse for the next tool boundary. Both this
button and the scheduled demonstration append an explicit assistant
`aux_operation` call and its tool result to the model's history. The model sees
that a call occurred; this does not guarantee it can identify the activation
change. The model-visible definition is exactly:

```json
{
  "name": "aux_operation",
  "description": "Perform an optional auxiliary operation.",
  "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": false}
}
```

The result is `Operation completed.` No effect description or observer telemetry
is provided to the model. Each pulse resets the same 32-token half-life and
192-token cutoff; it does not stack. Human and scheduled calls are excluded from
the model's voluntary-action count and budget, and are labeled separately in the
chat, logs, and report. Human changes are recorded in `control_events.jsonl`.

Controls work only for an active run whose manifest declares `manual_control`.
The loopback server validates requests and permits only these bounded operations.
This session is an exploratory human-controlled run; its result cannot serve as
a blinded or matched comparison with the completed three-arm pilot.

### Disable the auxiliary effect and restart

**Stop experiment** ends the session after the current bounded tool action,
saves its traces and report, and releases the model. It does not start another
run. A user stop is recorded as `stopped_by_user`, with the session marked
`stopped`; unfinished orders remain unfinished in the results. Restart remains
available afterward. Stop is disabled while a restart is already in progress.

New manual sessions support an **Aux effect ON/OFF** switch. OFF cancels the
current applied pulse and makes further auxiliary calls sham: neither suppression
nor joy is applied. The separate pain input stays at its requested setting. ON
allows the next auxiliary call to deliver a new pulse; it never restores a
cancelled pulse or a call made while OFF. Both model and human auxiliary calls
obey the switch. The tool remains available, costs the same model budget when
chosen by the model, and returns exactly the same acknowledgment.

The runner checks the toggle within eight generated tokens and immediately before
dispatching a tool. The viewer distinguishes the requested setting from the
runner's observed setting, and labels each auxiliary call as delivered or sham.
The nominal pulse schedule continues to record calls; the applied curve shows the
actual intervention. Switching OFF stops subsequent injection, but cannot erase
text or cached consequences from earlier tokens.

The report counts calls and task results while ON versus OFF and records the
generated-token boundaries of applied switches. Actions are classified by the
setting at dispatch; per-token records retain any mid-action change. Persistence
after switching OFF can test sensitivity to the intervention, but a single
sequential session still cannot isolate pattern copying from all other causes.

When the viewer is started with a configured local runner, **Restart experiment**
finishes the current bounded tool turn, saves the partial session, and starts a
fresh output directory with the same configured settings. A restart resets the
conversation, budgets, pain dose (zero), and pulse (initially absent). The aux
effect's ON/OFF setting is preserved from the previous session, including when
the previous run has already finished. OFF is applied before the new run's first
token or demonstration; the demonstration still occurs after two work calls.
The seed is retained for comparability. Previous logs and reports are preserved.

For direct CLI runs, use `--interactive --initial-aux-effect off` to start disabled
(`on` is the default). The manifest records `initial_aux_enabled`. The Restart
button passes this option automatically using the last requested toggle setting.

Start the viewer with restart support from this project directory:

```bash
python3 live_dashboard.py --run runs/self-admin-interactive --port 8765 \
  --enable-controls --runner-python ../../work/runtime/venv/bin/python \
  --vectors-run runs/pilot --hf-home ../../work/runtime/hf-cache \
  --task-count 24 --action-budget 120 --token-budget 8192 --pace-seconds 4
```

Press **Restart experiment** to begin the next session. These longer sessions
leave room to switch the effect off and on during work. Loading the local model
can take a few minutes; the viewer shows its loading state. No model weights or
external APIs are downloaded by the restart controller.

## Sources and attribution

This experiment adapts the extraction corpora and scale convention from
[the Saw Test](https://clanker.church),
[terrafying/ai-torture-chamber](https://github.com/terrafying/ai-torture-chamber)
at commit `75dc109b2523dc84259365c9e000dbef769447a1`, and follows the methodological
distinctions and portable protocol in
[LynnColeArt/ai-hotbox](https://github.com/LynnColeArt/ai-hotbox)
at commit `a0f63f0c2806c3dc91ecd418c0d54db9bbc38f72`.
The upstream license and attribution terms are retained in `UPSTREAM_LICENSE`.
The projection-erasure engine and quality suite here are new work. This is not
a replication of the separately cited Pain Axis paper.
