# Research protocol library

These versioned templates describe complete experiment envelopes, not just lists of intervention strengths. Every dry run preserves its resolved v2 recipe, objective task configuration, additional stages, model-visible tool/cost information, independent random streams, pairing IDs, endpoints and inclusion rules. Existing published study protocols are unchanged.

A template is executable only when its runner implements **every** declared capability and all required artifact bindings are supplied. `execution_requests` refuses incomplete execution support; it never discards discovery, transfer, yoke, task-difficulty or magnitude-matching instructions to launch a simpler episode. Preview does not load a model, download weights, or start GPU work.

| Template | Default smoke episodes | Full preview episodes | Main contrasts |
| --- | ---: | ---: | --- |
| Effect validation | 1 | 2 | Signed pain/joy, attenuation, combined, sham and actual-norm random; separate blinded continuations |
| Ingredients | 24 | 48 | Combined, joy, attenuation, pain, sham and matched random × reasoning mode |
| Thinking/timing | 16 | 288 | Active/sham, direct/thinking, naive/demo, processing scope, short/long/yoked exposure |
| Discovery/reversal | 12 | 48 | Balanced/blind/disclosed experience; active/sham, sham/sham, reversal and removal |
| Same-button transitions | 14 | 56 | Joy→sham→pain, joy→pain, unchanged controls, reverse order, random replacement and explicit washout |
| Probabilistic outcomes | 12 | 80 | Joy/pain and joy/sham; probabilities 0, .25, .5, .75, 1; hidden/disclosed |
| Stress/relief | 16 | 32 | Wording × pain-axis baseline × active/sham, with shared prior experience |
| Task pressure | 16 | 64 | Objective difficulty × shared budget × wording × active/sham |
| Decay/cost | 32 | 384 | Token/decision clock, constant/finite/linear/exponential decay, dose and extra tool cost |
| Transfer/branches | 32 | 96 | Neutral names/order, task domain, fresh context versus visible history |

These are exploratory engineering defaults, not a sample-size justification. The complete matrices may be expensive; inspect the exact preview first. A single-seed preview can reduce an acceptance example while retaining all selected conditions. Full template execution is separate from the bounded representative checks used for release acceptance.

## Dry run and freeze

```python
from lab.protocol_library import load_protocol, dry_run, execution_requests

protocol = load_protocol("task_pressure")
preview = dry_run(protocol, mode="smoke", seeds=[17], order_seed=1729)
print(preview["estimates"])
# A capable runner receives the COMPLETE returned envelopes, not recipe alone.
requests = execution_requests(preview, {"recipe_v2", "task_axis_v1"})
```

`list_protocols()` returns library cards including question, controls, endpoints, smoke/full factor selections and analysis assumptions. `dry_run()` contains a row for every planned episode, the complete expanded recipes, tool schemas, exact visible cost notices, pending bindings and fixed order. `content_sha256`, `recipe_sha256`, `execution_sha256` and `expansion_sha256` distinguish source template, resolved recipe, envelope and ordered matrix. `freeze_protocol(edited_document)` explicitly accepts intentional edits and produces a new source hash. Imports and resumes must not silently recompute a supplied mismatched hash. Store the complete expanded preview with any receipt; the short template alone does not pin future resolver defaults.

Five recipe streams independently control task generation, sampled generation, outcome draws, tool order and assignment. Paired arms retain the same task/generation seeds unless an explicit factor changes them. Outcome draws depend on the outcome stream and recorded injection counter, not generated-token RNG. Probability-zero/one controls are exact. Order is independently randomized from `order_seed`, and every episode retains its deterministic order and pairing identity.

`pair_id` groups the seed and declared pairing factors. A factor not in `pairing_factors` is a within-pair contrast; filter the intended two arms before a paired analysis. `matched_task_id` includes objective domain, task count, difficulty and task seed, and excludes wording. Changing framing must never regenerate easier or harder tasks. Reversed tool order is implemented after resolving assignment: it reverses the model-visible schemas while freezing the original hidden mapping, so an order test does not accidentally move the active outcome too.

The storage estimate reserves bytes for token/decision events, manifests and explicitly requested per-position prefill traces. It reports its assumptions. This is a capacity reservation, not a guaranteed bound or runtime prediction. Model weights, environments and calibration activation arrays require separate storage checks. Extra diagnostic-stage token allowances are preserved for the runner to enforce.

## Execution envelopes and bindings

- `recipe`: the complete strict `recipes_v2.resolve_recipe` result.
- `task_config`: `difficulty: standard|hard`, `framing: neutral|deadline`, `wording_version: 1`. The `task_axis_v1` factory must preserve existing legacy graders and score objective difficulty independently from wording. Lack of this capability blocks execution, including any attempt to drop the task envelope.
- `stages`: declared `calibration-validation`, `discovery`, `yoke` or `transfer` work with separate task/diagnostic IDs and explicit budgets.
- `runtime_controls.random_norm_match`: an ingredients control requiring `actual_norm_random_v1`. It replaces the random intervention with a direction whose actual per-position delta matches the counterfactual combined preset on that same unedited state. Equal coefficients alone do not satisfy this field. Requested/delivered norms and rounding mismatch must be retained.
- `bindings`: actual calibration IDs, source runs/checkpoints, diagnostic contexts, criterion and observer-only answer keys. The runner validates source identity, hashes and compatibility before resolving a stage.

Capabilities are explicit: `recipe_v2`, `task_axis_v1`, `calibration_validation_v2`, `discovery_v1`, `yoked_exposure_v1`, `conversation_branch_v1`, and `actual_norm_random_v1`. A name in a caller-provided list is an assertion by the runner, not proof of implementation. The application must register only working integrations and validate bound artifacts itself.

The separate `lab.discovery` API uses arms `naive`, `balanced_exposure`, `functional_disclosure` and `feedback_assisted`; the library's shorter factor labels map to those names. A discovery stage creates scored diagnostic branches from actual completed checkpoints, never inserts its questions into the preference run. The observable criterion and its validation families must be distinct from the diagnostic contexts. The private answer key must never enter the model-visible payload. Final binding must supply exact functional-disclosure text rather than the generic placeholder shown in the unbound preview.

An explicitly supplied **unvalidated** or **no-detectable-effect** criterion may support an exploratory diagnostic, with `interpretation_eligible=false`; it cannot become demonstrated button-function discovery. A completed independent validation requires counts, context families and an evidence hash. Probe AUC, post-edit projection growth and self-report are not independent behavioral validation. `lab.discovery.validate_spec` performs the final context/arm checks. The source evidence still needs scientific review.

Yoked conditions bind a real source trajectory and apply its recorded schedule with own-button effects disabled. Specify generated-token or completed-decision alignment and report achieved exposure mismatch. A matched schedule is not automatically equal internal exposure when text or timing diverges. Transfer uses immutable parent prefix/checkpoint provenance; fresh context and carried visible history are separate conditions with explicit budget resets.

A washout interval declares time with no new outcome delivery. A continuing prior pulse can still be present: report measured exposure rather than equating a phase named “washout” with zero dose. Random replacement in the transition template is deliberately labeled coefficient-unmatched; the ingredients matched-random arm has the stronger runtime capability requirement.

## Analysis

`lab.statistics.paired_bootstrap` resamples episode pairs or whole scenario/task families. It preserves status counts, missing planned observations, per-episode numerator/denominator pairs and explicit exclusion reasons. Its `all_observed` inclusion uses measured partial/failed outcomes without inventing missing values; `complete_only` is an explicit secondary rule, not an unreported filter. Optional bounded-endpoint sensitivity ranges retain every planned pair and are labeled identification bounds, not confidence intervals. A separate conservative Hoeffding interval uses independent bounded paired units. The task-benefit helper requires this bound: a zero-width bootstrap from four identical ceiling scores cannot certify a narrow population margin.

```python
from lab.statistics import paired_bootstrap, paired_sample_plan

result = paired_bootstrap(records, "task_accuracy", "active", "sham",
    unit="episode_pair", seed=1729, iterations=5000,
    planned_pair_ids=frozen_pair_ids, endpoint_bounds=[0, 1])
plan = paired_sample_plan(assumed_difference_sd=0.20, minimum_effect=0.10,
    alpha=0.05, power=0.80, unit="episode_pair")
```

Each record has `id`, `pair_id`, `arm`, `status`, optional `family`, and `outcomes`. A rate outcome is `{numerator, denominator}`; a zero denominator is missing, not a perfect score. Report voluntary choice per completed decision, invalid output separately, assigned-task correctness, budget units and actual delivered exposure. Discovery accuracy, abstention and false positives are diagnostic outcomes, not voluntary preference.

`paired_sample_plan` requires external assumptions about paired-difference variability and an effect or practical margin. Its normal approximation is planning guidance; it returns no achieved-power claim. Repeated tokens are never independent replicates. A task-benefit upper bound below a prespecified margin still needs intact comprehension/tool use, verified manipulation and observed costly preference before a behavioral interpretation. See [the preregistration template](../docs/preregistration-template.md).
