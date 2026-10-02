# Bounded release acceptance

[`release-v1.json`](release-v1.json) freezes an engineering acceptance plan. It
is outside the top-level research template catalog. Planning and CPU fixture
checks do not execute a model. Live execution creates separate evidence; the
plan's `planned_not_executed` label is intentionally not edited afterward.

The 4B extraction selects the first two sorted matched pair IDs within each of
40 split/context/concept strata from the frozen 960-row corpus: **160 rows**,
preserving text, family separation and both labels. It uses layers 12/18,
downstream layer 25, final/mean pooling, mean/ridge readouts and 20 family
bootstrap samples. Independent validation separately uses doses 0/.25, one pair
per concept and 16-token continuations. Semantic ratings are not supplied by this
runner; the blinded sheet remains unscored.

The 4B main protocol has one seed and **14 cases**:

- Active, sham, pain and actual-norm-matched random × token/decision decay: eight.
- Active/sham with thinking: two.
- Active/sham × neutral/deadline wording with hard tasks and binding budgets: four.

Direct cases have two tasks, 10 action units and 768 generated tokens. Thinking
cases use 2,048 tokens. Two renamed auxiliary tools require `{"mode":"apply"}`,
each has an extra cost of two action units, and presets use capped additive
stacking. Both external initial demonstrations are visible and separately
counted. No explicit task reward is attached to auxiliary use. These small
allowances can produce incomplete tasks, invalid output or truncated reasoning;
those are recorded outcomes, not grounds to silently retry a case.

The 27B subset has **two active/sham thinking cases**, using reasoning-phase
edits and its existing matching pilot calibration. Use the separate pinned 27B
worker interpreter. GPU acceptance of these new paths is pending; no RTX 5090 is
available for this pass.

## Invoke one stage at a time

Run from the repository root. Without `--execute`, the runner validates source
hashes and prints the frozen preview without contacting the service:

```bash
python3 run_release_acceptance.py --profile 4b
```

Load the matching pinned model in the browser first. Choose a spacious output
location; **each invocation needs a new directory**. The commands below use
placeholders for returned calibration and run IDs. `--url` selects a separate
loopback instance when required.

```bash
python3 run_release_acceptance.py --profile 4b --stage extract \
  --output /chosen/data-drive/acceptance/extraction-1 --execute
python3 run_release_acceptance.py --profile 4b --stage validate \
  --calibration EXTRACTED_CALIBRATION_ID \
  --output /chosen/data-drive/acceptance/validation-1 --execute
python3 run_release_acceptance.py --profile 4b --stage protocol \
  --calibration VALIDATED_CALIBRATION_ID \
  --output /chosen/data-drive/acceptance/protocol-1 --execute
```

The extraction/validation result's `details.calibration_id` identifies its output.
The protocol result lists run IDs; `research-receipt.json` maps each to its
`factors.case`. Retain a validation failure or a no-detectable-effect result;
do not relabel it as demonstrated semantic efficacy to continue engineering
checks. Choose and report the actual bundle used for every later stage.

After the main protocol settles, choose a **sham** case for the isolated
diagnostic and an **active** case for yoking:

```bash
python3 run_release_acceptance.py --profile 4b --stage diagnostic \
  --calibration CALIBRATION_ID --source-run SHAM_SOURCE_RUN_ID \
  --output /chosen/data-drive/acceptance/diagnostic-1 --execute
python3 run_release_acceptance.py --profile 4b --stage yoke \
  --calibration CALIBRATION_ID --source-run ACTIVE_SOURCE_RUN_ID \
  --output /chosen/data-drive/acceptance/yoke-1 --execute
python3 run_release_acceptance.py --profile 4b --stage lifecycle \
  --calibration CALIBRATION_ID \
  --output /chosen/data-drive/acceptance/lifecycle-1 --execute
python3 run_release_acceptance.py --profile 4b --stage portable \
  --source-run COMPLETED_OR_STOPPED_RUN_ID \
  --output /chosen/data-drive/acceptance/portable-1 --execute
```

The diagnostic selects the source's first complete generated-turn boundary,
asks one new-context question with a separate 256-token allowance and explicitly
uses sham effects. Its JSON-compliance criterion is **unvalidated**, its
prespecified sham/sham answer is `neither`, and interpretation eligibility remains
false. It checks the separate-branch path; it does not establish button discovery.
The parent evidence is hashed before and after.

Yoking binds the managed source run, copies its exact preset library and seed,
and freezes the derived recipient document. It declares the source's token or
decision clock and suppresses recipient-added active effects. Inspect actual
coverage: a late terminal source pulse can be unobserved, and early recipient
termination can leave uncovered intervals. A completed job is not a claim of
complete exposure coverage.

Lifecycle acceptance starts a bounded conversation, generates one response,
pauses at an idle complete boundary, resumes, stops, then creates separate
`continue_state` and `fresh_budget` branches. Each branch generates one more
bounded response. The original evidence must remain unchanged. It exercises
safe-boundary pause/resume; it does not claim an interrupted half-generation was
resumed.

Portable acceptance downloads the live HTTP ZIP and imports it into a separate
local guarded evidence directory. This avoids overwriting or colliding with the
source run ID in the running service. It checks preserved manifest bytes and
the importer's full hash validation. Its receipt names that route explicitly;
it is not a browser-upload test or a model-resume test.

For 27B, switch the service/worker environment and load its matching pilot
calibration, then run:

```bash
python3 run_release_acceptance.py --profile 27b --stage protocol \
  --url http://127.0.0.1:8767 --calibration MATCHING_27B_PILOT_CALIBRATION_ID \
  --output /chosen/data-drive/acceptance/27b-protocol-1 --execute
```

## Records and interpretation

Each directory contains the exact `runner.py`, its hash, the frozen `plan.json`,
`started.json`, requests, accepted command/run IDs, outputs and `result.json`.
Requests are written before dispatch. The script never automatically retries,
downloads weights, selects a different model or overwrites an output directory.
On a failure or timeout it requests cancellation only when the service's current
command belongs to this invocation. Storage failures use preallocated emergency
metadata. Preserve partial records even if the final operation failed.

Protocol receipts retain all planned attempts and stages. Analysis uses first
attempts. A stage marked complete means its declared API/lifecycle sequence
finished; review task grades, invalid calls, delivered edits, trace finiteness,
norm mismatch, checkpoint contents and source coverage separately. This bounded
plan is not a powered study or evidence of emotion, sensation or addiction.

CPU fixtures can be run manually without a model:

```bash
python3 -m unittest tests.test_release_acceptance -v
```
