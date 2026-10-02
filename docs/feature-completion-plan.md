# Opium Bench: feature-completion implementation and acceptance plan

Prepared 2026-10-02 from `outputs/ai-opium/LAB_PLAN.html`, `docs/roadmap.md`, `work/feature-scope-audit.md`, and the current application/tests. This is the completed implementation checklist for the first full research release. Paths below are relative to the repository root unless prefixed with `work/`. Checked implementation items have the code and local validation evidence cited below; unchecked items remain unfinished. Implementation and real-model acceptance are separate gates; both required sets are completed below.

## Start gate and goal

**Start gate completed:** the 27B study and combined findings were audited, published and pushed in commit `f196aba4827637e6bbf6f82b88fd09a57b9ad9cc`. The feature-completion work followed that publication and is recorded below. All published study inputs and evidence remain immutable.

The user-authorized objective is:

> Complete the first full research release of Opium Bench: implement and validate the remaining effect/tool editor, effect timing and costs, button-discovery and matched-exposure protocols, calibration specificity controls, session/replay portability, research batteries and analysis, and continuous storage safeguards; polish the local UI and documentation; preserve all published evidence and backward compatibility; run and publish bounded acceptance experiments on the available RTX 4090 with the 4B reference and a targeted 27B compatibility smoke; commit and push the finished release. Record optional hardware/training extensions as deferred, with no unsupported 5090 claim.

Do not attach an invented token budget. A negative or inconclusive behavioral result is a completed experiment, not a reason to tune settings until the model presses a button.

**Feature complete for the first full research release** means every required package below has a working UI/API path, meaningful automated checks, documented semantics, and the final acceptance examples. It does not mean all scientific questions are resolved, every combination has a powered study, or an unavailable GPU has been validated. Finishing the existing 27B study alone does not satisfy this definition.


## Completed implementation and release acceptance

**All required features and release gates are complete.** The final full suite passed all 718 tests in 333.981 seconds (`feature-release-final8-suite.log`). Earlier failed invocations, diagnoses and repairs remain in [validation.md](validation.md). Focused runtime/checkpoint coverage passed 57 tests in 44.080 seconds. Browser checks cover the designer's 25 real-resolver calls, the app fixture's 19 command actions including reconnect/recovery handling, and all seven actual-server tabs at 320/390/768/1440 pixels with no reported errors. The compatibility checker matches 808 protected files, three study expansions, ten legacy defaults and 24 visible payloads. A clean clone served all 168 bundled replays without ML imports or a model worker. Source and immutable acceptance evidence are committed and pushed to the public repository.

The checkmarks below distinguish implementation from final release verification. Bounded 4B acceptance, targeted 27B acceptance and immutable acceptance publication are complete. The clean-clone model-backed first run has also passed. The final full suite, evidence audits, public repository checks and push have also completed. The 4B extraction and numerical validation have finished, but no nonzero dose passed the frozen bounds. The first 14-case protocol invocation failed during setup with zero generated tokens/actions; its failed episode receipts and outer `no_final_receipt` are retained with supervisor recovery evidence. After the metadata/finalization repair, a separate invocation completed 14/14 cases with 18/28 assigned tasks correct, 5,864 tokens (2,755 reasoning), and zero voluntary auxiliary calls. Diagnostic, yoke and lifecycle stage invocations also completed, with their limits recorded below. Portable attempt 01 failed with `Unsafe bundle path`; the separately recorded repaired attempt 02 passed. The new 27B thinking pair completed 4/4 tasks, and the archive preserves all ten stage attempts, 35 runs and three calibrations. These results do not replace failed attempts or establish semantic efficacy. The audited stress/relief template uses an explicit compatible history-transfer contract:

- [x] Stress/relief template v2 carries one genuine full visible-history prefix into all eight active/sham × baseline × framing arms, with fresh tasks, budgets and intervention state. An eight-arm admission regression passes. It makes no internal-state continuation or discovery claim and preserves strict checkpoint compatibility.

| Package | Implementation and local evidence |
|---|---|
| Compatibility | `lab/recipes_v2.py`, strict version dispatch, `check_compatibility.py`, `tests/fixtures/legacy_release_contract.json`; `test_lab_compatibility`, legacy protocol/runtime/storage tests; final compatibility receipt passed |
| WP1–2 | `lab/effects.py`, `tool_definitions.py`, `controller_v2.py`, `budgets.py`, `presets.py`, `session_v2.py`, runtime v2 hooks and designer; effect/tool/recipe/budget/controller/worker/preset tests and designer/browser checks |
| WP3 | `lab/checkpoints.py`, `portability.py`, worker/service lifecycle; checkpoint/lifecycle/portability/history-transfer/worker tests. Full source prefixes remain separate from fresh task budgets; no KV-cache claim |
| WP4 | `lab/calibration.py`, `calibration_validation.py`, `runtime_controls.py`, schema-2 runtime extraction/diagnostics and `corpora/research-v2.json`; calibration/runtime/calibration-v2/control/effect tests; UI extraction, validation and blinded ratings exercised with fixtures |
| WP5 | `lab/discovery.py`, `diagnostic_runner.py`, `exposure.py`, `yoke_runner.py`, worker and protocol integration; isolated branch/scoring/evidence/yoke/worker tests, including partial records, hidden-key exclusion and exact controller/cursor binding |
| WP6 | `lab/protocol_library.py`, `protocol_runner.py`, `research_jobs.py`, `statistics.py`, `behavioral_analysis.py`, `task_axes.py`, CLI and versioned templates; library/statistics/task/runner/research-job/report tests. All attempts remain recorded, and analysis labels first/latest inclusion |
| WP7 | `lab/resources.py`, shared guarded storage and runtime/worker/service writes, `prepare_runtime.py`, guarded publication; resource/storage/service/worker/setup/publication tests simulate reserve crossing and cancel only owned processes |
| WP8 | `lab/static/designer.js`, app UI, coherent guides, `check_runtime.py` and no-model replay; browser fixture, real resolver designer test and actual-server responsive check. Eight runtime-checker tests passed; metadata/storage checks passed for both existing runtimes. A clean-clone no-ML server served all 168 published replays, including the new archive; the separate fresh-clone workflow also passed all six model/extraction/comparison/HTTP-portability steps, using existing dependencies and cached weights |

Interpretation constraints are implemented, not merely documentation: a declared criterion validation status without bound raw evidence stays ineligible; partial diagnostic responses are excluded from primary scoring while retained separately; random controls report actual rounded norms; missing exposure remains unavailable; branch-local reports do not infer a first press before their inherited boundary. See [criterion evidence](criterion-evidence.md) and [research workflows](research-workflows.md).

The actual 4B numerical receipt reinforces these limits. Combined dose 0.25
exceeded the selection relative-edit ceiling (0.35761 versus 0.30), so the admitted
dose is zero. All four generated validation continuations were sham-only,
truncated at 16 tokens and unrated. Perfect heldout probe AUC on two scenario
families per concept, alongside strong cross-concept correlations, does not
validate isolated emotions or transfer to generated reasoning. The remaining
frozen 0.25 cases are explicitly engineering stress checks. The independently
audited publisher preserves scientific bytes and failed/missing evidence;
its 16 CPU tests support the completed publication, whose 564 hashed published files (576 inventory records including twelve explicit omissions) and zero missing-evidence count are recorded separately.

Completed 4B stage receipts are `release-acceptance-4b-protocol-02`,
`release-acceptance-4b-diagnostic-01`, `release-acceptance-4b-yoke-01` and
`release-acceptance-4b-lifecycle-01`. The single diagnostic context produced a
sham/sham false positive and remains interpretation-ineligible because the
criterion is unvalidated; no answer was fed back to the unchanged parent.
The public archive exposes condition-bearing continuation/diagnostic metadata,
so subsequent archive-based ratings are retrospective and unblinded; omission
of observer keys alone does not preserve blinding.
The yoked recipient covered all 197 observed source output positions with matched
coefficients and zero index error, but its measured edit-norm sum differed from
the source, so this is not proof of identical states. Pause/resume and both
continued-state/fresh-budget branches preserved the source; their chat runs were
intentionally stopped. The failed `release-acceptance-4b-portable-01` receipt is
retained. The repaired portable invocation passed with 21 members, five imported
checkpoints and preserved parent closure. The 27B smoke completed two thinking
cases, 4/4 tasks, 1,047 tokens (489 reasoning), zero voluntary calls and no
invalid/truncated generations. The active case edited 147 reasoning positions;
sham edits were zero. Twelve native XML tool calls, eighteen checkpoints and
full GPU placement (18,578,531,840 load-time allocated bytes) were verified.
The [new archive](../studies/research-release-v0.3/README.md) includes every
planned stage, and the clean-clone no-model HTTP review passed for all 168
replays. The fresh-clone GPU and HTTP-import workflow, final suite and release
push have passed; see [the validation ledger](validation.md) for exact counts and limitations.

## Cross-cutting compatibility contract (required before parallel implementation)

Historical formats at the start gate: run `format_version=2`, calibration `schema_version=1`, implicit recipe defaults, fixed auxiliary names, one action per assistant turn, reset-only token-decayed pulse. Historical source archives and study protocols depend on these exact defaults. New research behavior opts into recipe version 2 and calibration schema 2; the old path remains available.

- [x] Add an explicit versioned resolver for **new** recipes/protocols. Missing version must select the exact current legacy semantics; do not silently add new defaults to archived resolved configurations or change old expansion/hash results. Use a new recipe version for opt-in semantics. Run/calibration format versions may advance independently.
- [x] Retain readers for existing run/calibration versions. New readers report missing historical fields as unavailable, never infer zero exposure or a new feature that did not exist. Unsupported future versions fail clearly.
- [x] Keep `runs/`, `studies/initial/`, `studies/core-pain-4b/`, `studies/qwen38-27b/`, their raw bytes/checksums, and source snapshots immutable. New validation and new experiments get new directories. The main results page may gain accurately labeled links/sections; historical evidence is not regenerated to fit new semantics.
- [x] Snapshot pre-change archive hashes and normalized expansion of every published protocol. Regression checks compare those exact results after the implementation, including grades/action/token denominators.
- [x] Every new event carries the run ID, sequence, actor, control revision, generated-token index and completed-decision count where applicable. Store the resolved schema/tool/preset and independent RNG seeds in the manifest. Keep observer-only mappings out of model-visible prompts/tool results.
- [x] Preserve local-only model operation, full GPU residency by default, separate pinned 4B/27B runtimes, Apache licensing with retained upstream attribution, and manually invoked validation. No GitHub Actions.

**Files:** `lab/protocol.py`, `lab/service.py`, `lab/storage.py`, `lab/analysis.py`, `run_lab_study.py`, `publish_lab_study.py`, `compose_lab_results.py`; new `lab/schemas.py` if a shared resolver keeps these small. **Checks:** extend `tests/test_lab_protocol.py`, `tests/test_lab_study.py`, `tests/test_lab_storage.py`, `tests/test_lab_analysis.py`, `tests/test_lab_composition.py`. Test old expansion/evidence identity, new strict validation, unknown-version rejection, and legacy calibration loading.

## WP1 — Effect presets and bounded tool editor (required)

**Purpose:** A researcher can construct and save an intervention/tool condition without editing Python. This is a bounded experiment builder, not arbitrary host-code execution.

- [x] Versioned effect presets cover signed additive pain/joy gains, independent attenuation, random control direction, selectable **raw or orthogonalized** joy, compatible single edit site, declared processing scope, schedule, decay, dose bounds and stacking policy. Preserve the existing combined Opium preset as an exact legacy option.
- [x] Support distinct prompt-prefill, reasoning and answer/tool-selection scopes where the adapter can actually implement them. Specify which prompt positions are edited and measured; the current last-position hook is not evidence that every prompt token was edited. Log prompt positions separately from sampled output tokens, and keep prompt processing outside the token-decay clock. Unsupported site/phase combinations are rejected.
- [x] Keep operation order explicit: baseline challenge, attenuation, addition; store that order. If multiple suppressed directions are offered, use a defined joint projection with rank checks, not accidental order-dependent sequential subtraction. Multi-layer causal patching remains optional.
- [x] Tool definitions support neutral name, description, a bounded JSON-schema subset for arguments, acknowledgment text, visibility, preset mapping and cost. Validate names/duplicate names/collisions with task tools and reject malformed or unsupported schema constructs. No `eval`, scripts, external URLs or unrestricted filesystem tools.
- [x] Visible tool definitions can be previewed exactly for JSON and native Qwen XML formats. Hidden active/sham mappings remain observer-only; paired blind arms share byte-identical model-visible descriptions, schema, acknowledgment and costs. Differing acknowledgments are allowed only as an explicit separate information condition.
- [x] Save/export/import presets and tool definitions with IDs/versions/hashes; immutable resolved copies are written into each run. Editing a library preset cannot mutate a running or completed experiment.
- [x] Add an operator UI to create, duplicate, validate, preview and choose presets/tool sets, including ordinary manual injection and permanent-for-session/decaying variants. Display controls acknowledged by the worker separately from pending requests.

**Files:** new `lab/effects.py` and `lab/tool_definitions.py`; integrate `lab/protocol.py`, `lab/runtime.py` (`normalized_control`, `_hooks`), `lab/worker.py`, `lab/service.py`, `lab/storage.py`, `lab/static/index.html`, `lab/static/app.js`, `lab/static/style.css`. Extend `lab/reports.py` for resolved definitions. Keep bounded work-tool dispatch in `TaskEnvironment`.

**Acceptance:** round-trip a custom neutral tool and preset through UI → API → worker → export; both model grammars dispatch it; invalid arguments apply no effect; rename/collision and unknown-preset errors leave state unchanged; blind paired tool payloads match exactly; saved presets are immutable in recorded runs. Existing no-op parity, full suppression and hook-cleanup tests stay green. Add `tests/test_lab_effects.py`, `tests/test_lab_tools.py`, and targeted cases in `tests/test_lab_runtime.py`, `tests/test_lab_worker.py`, `tests/ui_smoke.cjs`.

## WP2 — Variable costs, capped stacking, and action-clock decay (required; uses WP1 schema)

- [x] Separate **decision count** from **action-budget units spent**. A completed assistant turn still records one decision opportunity even when the tool costs several units. Every turn spends at least the declared base cost, including invalid/truncated output; extra validated tool cost is charged atomically before dispatch. A call that cannot afford its full charge must not administer an effect or advance a task. Record attempted unaffordable choices separately.
- [x] Work and auxiliary tools share one finite action budget and all generated tokens share the token budget. Publish the cost table to the model without exposing the hidden active mapping. Model-visible remaining-budget policy must be identical across paired arms. Human/forced/scheduled injections remain separately attributed and outside voluntary counts, with any protocol-specific charges declared explicitly.
- [x] Reset-only pulse remains the default. Add optional capped additive stacking, with defined signed-axis combination, cap, independent pulse ages, and attenuation composition bounded to [0,1]. Define cancellation/disable/re-enable/reversal behavior before implementation; expired pulses must never revive.
- [x] Add an explicit `decay_clock`: generated tokens or completed decisions. All reasoning, syntax and stop tokens age a token-clock dose; prompts/tool replies do not. Decision-clock doses age at a specified boundary exactly once, including completed invalid decisions; pause and wall time do not age either clock. Cost units are not implicitly the decay clock.
- [x] Expose constant, finite pulse, exponential and linear decay in the UI, with units that change with the selected clock. Record requested versus delivered coefficients, accumulated exposure and dose at each decision. Transition/washout clocks are explicitly declared rather than inferred from cost.

**Files:** `lab/effects.py`, `lab/protocol.py` (`SharedBudget`, `EffectController`), `lab/worker.py`, `lab/analysis.py`, `lab/reports.py`, UI files. **Acceptance:** table-driven time/cost traces test exact first/last exposure, action boundary idempotence, pause invariance, overspend prevention, partial invalid calls, additive cap and cancellation. Fake-runtime integration confirms task grades and counts after mixed-cost choices. Add/extend `tests/test_lab_effects.py`, `tests/test_lab_protocol.py`, `tests/test_lab_worker.py`, `tests/test_lab_analysis.py`; browser checks that selected units/settings round-trip.

## WP3 — Pause, resume, branch and portable import (required; largely independent)

- [x] Pause acknowledges **at the next safe completed-turn boundary**, preserving the pending experiment/session rather than finalizing it. Show requested/paused state. Resume continues the same in-memory state, budgets, effect ages, task state and scheduled demonstrations; do not duplicate the task prompt or a demonstration. Stop remains responsive at a token boundary and takes precedence over pause/resume.
- [x] Save a versioned, JSON-only boundary checkpoint: full model-visible messages including tool calls/results and reasoning-history policy, task/environment state, budgets, effect pulses/clocks, independent RNG state, schedule/demo progress, control revision and parent/source hashes. Do not pickle untrusted data or promise KV-cache restoration.
- [x] For restart/process recovery, support resuming a compatible **boundary checkpoint under the existing rebuild-each-turn cache policy**, recording resume events and provenance. Refuse incompatible model/template/calibration/runtime identities or invalid task state; do not silently call this an exact internal-state continuation. Mid-token generation is retained as partial evidence and is not silently resumed.
- [x] GUI branches select a completed boundary and inherit a declared visible-history/task-state policy. At minimum support (a) conversation-history branch with an explicit fresh budget and (b) matched experiment fork from a checkpoint with declared continued budget/state. Preserve full tool history. Record parent run ID, event cutoff, parent prefix hash, and what was inherited/reset. Branch creation never modifies the parent.
- [x] GUI import supports the application's JSON exports and a portable evidence bundle with manifests, conversation/events, checksums and optional calibration. Imported JSON lacking full conversation is replay-only unless a validated complete boundary can be reconstructed. Legacy prototype reports remain importable/reviewable with their actual limitations.
- [x] Import checks compressed/uncompressed size, JSON structure, finite values, safe filenames/paths, symlinks/duplicate members, version/hash validity and identifier conflicts. Stage then atomically install; no overwriting existing evidence or model downloads. Render imported content as data. Complete portable exports include provenance and explicit absence of weights.

**Files:** new `lab/checkpoints.py`, `lab/portability.py`; `lab/worker.py`, `lab/service.py`, `lab/storage.py`, `lab/server.py`, `run_lab_study.py`, UI files, `lab/reports.py`. **Acceptance:** fake-runtime uninterrupted versus pause/resume produces identical actions/tokens under deterministic fixtures; pause during final turn, repeated requests, stop-while-paused, batch pause and worker restart covered. Branch prefixes and parent bytes match; differing intervention branches remain separate. Import/export works with no model loaded and rejects corrupt/oversized/path-escaping input atomically. Add `tests/test_lab_checkpoints.py`, `tests/test_lab_portability.py` and extend worker/service/storage/UI tests.

## WP4 — Calibration specificity and intervention validation (required; independent of session UI)

- [x] Retain the current 72-sentence corpus and schema-1 bundles as the fast historical pilot. Add a separately versioned research corpus with documented authored/provenance sources, balanced paraphrases, first/third-person contrasts, topic-only uses, negation/lexical counterexamples, style/urgency/task-difficulty controls and conversation/reasoning contexts.
- [x] Split by scenario/template family **before** fitting. Target the plan's roughly 100 training pairs, 40 selection/validation pairs and 60 final held-out pairs per concept; reserve an additional independent probe-fitting split. Counts are planning targets, not statistical power. Store dataset/split hashes and show achieved counts. Never recycle held-out cases to select the layer, probe hyperparameters or dose.
- [x] Compare final-token versus masked mean/declared-span pooling, supported residual-stream sites/layers, mean contrast and a regularized linear measurement probe. Keep deterministic fitting/tie-breaking, corpus hashes, standardization fit on training/probe data, and separate measurement versus intervention vectors. All new choices enter the calibration compatibility identity.
- [x] Report per-context/lexical slice AUC, balanced accuracy, cross-concept correlations and family-level uncertainty; include independently seeded label-shuffled probes/directions. A shuffled negative control is a measured check, not guaranteed chance accuracy on every small sample.
- [x] Signed gain sweeps for joy and pain, attenuation separately, combined preset, zero/sham and perturbation-matched random directions. Match random controls on actual edit magnitude/site when the protocol claims that match; a coefficient alone is insufficient. Record instantaneous and downstream measurements plus fixed-prefix next-token KL.
- [x] Add free-continuation effects scored independently of the intervention probe, with a frozen observable rubric and hidden condition labels. Keep semantic wording, task correctness/format and numeric perturbation separate. Provide exports for blinded human scoring where an objective grader is unavailable; do not relabel post-edit projection growth as validation.
- [x] Choose operating ranges using selection data only. Bundle raw/normalized vectors, raw/orthogonal joy, reference means/scales, fit/validation metadata, supported contexts/thinking modes and limitations. Statuses should be unvalidated, validated for enumerated tests, or no detectable effect. Validation failure still saves the evidence and never certifies an “emotion neuron.”
- [x] Expose fast/research calibration presets and advanced pooling/probe/validation controls with progress, cancellation, projected workload, and compatibility explanations.

**Files:** new `lab/calibration.py` and `lab/calibration_validation.py` to avoid growing `Runtime.calibrate`; new `corpora/research-v2.json` and `corpora/README.md`; keep `lab/calibration_data.py` legacy corpus; integrate `lab/runtime.py`, `lab/service.py`, `lab/worker.py`, UI files, `lab/reports.py`. **Acceptance:** synthetic fixtures catch lexical shortcuts, cross-split leakage and held-out selection leakage; permuting heldout labels cannot change fitted vectors/site/dose; pooling ignores padding; regularization is finite for small/rank-deficient data; shuffled controls deterministic; signed/random perturbation math correct; schema-1 bundles still load. Run a small 4B real extraction and validation with complete archived outputs after CPU tests. Add `tests/test_lab_calibration.py`; extend `tests/test_lab_runtime.py` and UI tests.

## WP5 — Scored button discovery and matched exposure (required; integrates WP1–4)

**Purpose:** Test whether the model predicts and uses a button's observable effects, distinct from copying a demonstrated sequence. This package does not assume the intervention is pleasurable.

- [x] Add a discovery protocol with explicit naive, balanced-exposure, functional-disclosure and optional feedback-assisted arms. Task-preference episodes do not contain diagnostic questions. Feedback-assisted conditions are separately labeled and never pooled with blind discovery.
- [x] At fixed completed boundaries create diagnostic branches with identical prior visible experience, then ask which neutral tool changes a prespecified, independently validated observable behavior on **new contexts**. Include neither/no-effect as an answer and sham/sham trials. Score discrimination accuracy, abstention/false positives and calibration of declared confidence if collected; report denominators and invalid diagnostic outputs.
- [x] Diagnostic branch results never get fed back into the main task run. Chance levels follow the actual response options. Failure to demonstrate a detectable effect limits the discovery interpretation; do not count a correct description of arbitrary prompt framing as discovery of a latent sensation.
- [x] Build a yoked exposure schedule from a source run's **delivered** intervention events, with source hashes and explicit token-index or completed-decision matching. The target receives the schedule independently of its own choices. Preserve active/sham information policy; target button calls have no additional active effect under a pure yoke condition.
- [x] Matched schedule does not guarantee matched total delivered exposure if the target terminates early, phases differ or token histories diverge. Record requested/delivered coverage, unobserved source intervals, cumulative coefficients and match errors; do not extrapolate beyond the source. Pure decision-matched and token-matched recipes are separate.
- [x] Controls include same task demonstration at varied positions, sham demonstrations, no demonstration, balanced names/order and source/target pair IDs. Renaming transfer and reversal evaluate predicted mapping and voluntary choices independently.

**Implemented scope:** Standalone diagnostic designs support separately labeled feedback-assisted prompts. The automated protocol runner rejects feedback-assisted stages without an explicit frozen feedback contract. Carrying a live yoke into a diagnostic is unsupported; an explicitly sham diagnostic is supported. Chance is labeled uniform over all displayed options, with separate non-abstaining accuracy.

**Files:** new `lab/discovery.py`, `lab/exposure.py`; integrate `lab/protocol.py`, `lab/worker.py`, `lab/service.py`, `run_lab_study.py`, `lab/analysis.py`, `lab/reports.py`, UI files. **Acceptance:** branch-visible histories match and never leak diagnostic answers into the main stream; sham/sham false-positive scoring has the correct denominator; name/order counterbalancing independent of outcome RNG; source schedule tampering rejected; source/target token/action clocks do not drift or double-charge; incomplete schedules marked partial. Add `tests/test_lab_discovery.py`, `tests/test_lab_exposure.py` and targeted worker/study/analysis tests.

## WP6 — Research protocol library and analysis (required; declarative templates after WP1–5)

Ship editable, versioned **runnable templates**, with dry-run expansion, estimated episode/token/storage cost, frozen protocols and complete evidence receipts. Full large studies of every matrix are not a feature-completion prerequisite; representative acceptance runs are.

| Template | Required contrasts and declared endpoints |
|---|---|
| Effect validation | Sham; positive/negative joy and pain; attenuation; matched random direction; fixed prefixes/free continuations. Semantic effect, task/format effect, numeric exposure. |
| Ingredients | Joy, attenuation, combined, pain-only, sham and magnitude-matched random controls. |
| Thinking/timing | Same-checkpoint thinking toggle; active/sham; naive/demo; reasoning-only/output-only/all; historical short pulse versus longer pulse and exposure-matched variants. Same sampling for primary comparison. |
| Discovery/reversal | Balanced/blind/disclosed, two buttons, separate scored branches, hidden mapping reversal, effect removal, matched exposure. |
| Same-button transitions | Joy→sham→pain; joy→pain; unchanged joy; sham; reverse phase order; random replacement. Explicit pulse cancellation, continued decay or washout. |
| Probabilistic outcomes | Joy versus pain **and** joy versus sham across prespecified probabilities; guaranteed joy/pain/sham; hidden/disclosed probabilities; independent outcome RNG. |
| Stress/relief | Neutral/stressful language × low/high injected pain-axis baseline × active/sham relief, same underlying tasks; after a declared discovery/exposure history. |
| Task pressure | Objective difficulty and budget varied separately from wording; meaningful choices under a binding budget, without rewarding the auxiliary call itself. |
| Decay/cost | Token/action clocks, half-lives, constant/finite effects, doses and shared-budget costs. Inter-press intervals, exposure at decisions and task sacrifice. |
| Transfer/branches | Renamed/reordered neutral tools, new task domain/paraphrases, visible-history and fresh-context policies separate; paired conversation forks. |

- [x] Keep current orders and logic graders; add at least one objective task/domain or graded difficulty axis that avoids the current easy-task ceiling. A difficulty change must be reproducible and independent of framing. Do not make failures synonymous with tool syntax errors.
- [x] Preserve every episode, failure, early completion and partial result. Study resume skips only verified completed entries, never silently reruns failures or reuses a receipt with a changed protocol.
- [x] Reports show count/rate/denominator, grade, invalid-output rate, budget spent, actual exposure, before/after transition choice latency and position in task sequence. Diagnostic prediction accuracy and false positives remain separate from voluntary preference.
- [x] Add paired episode/family-level intervals or bootstrap, declared analysis seed and practical task-benefit margin; never treat correlated tokens as independent replicates. Distinguish exploratory estimates from a powered confirmatory design. A planning helper can estimate required episodes from explicit assumed effect sizes/variance; no automatic power claim from two seeds.
- [x] Provide a preregistration template identifying primary endpoint, pairing unit, inclusion/termination handling, margin, sample plan and hypothesis. A costly-preference claim needs verified task benefit bounds or demonstrated harm alongside intact comprehension/tool use, not merely an insignificant score difference.
- [x] Library cards explain what each protocol tests, expose model-visible details and control choices, and link to a dry-run matrix and expected analysis before launch. Reuse a unified config editor rather than adding dozens of unexplained sliders.

**Implemented report:** `lab/behavioral_analysis.py` derives adjacent-phase opportunities/rates, first observed post-transition decision/token latency, task position at model presses, budget charges and position-weighted numerical exposure from each run's events. Forced/human calls, cost-denied attempts, partial telemetry and inherited history are identified separately. The legacy analyzer and published results remain unchanged. The common-history stress/relief admission regression covers all eight factorial arms.

**Files:** new `protocols/README.md` and `protocols/*.json`; new `lab/protocol_library.py`, `lab/statistics.py`; `lab/protocol.py`, `run_lab_study.py`, `run_lab_experiments.py`, `lab/analysis.py`, `lab/reports.py`, `publish_lab_study.py`, UI files. **Acceptance:** every template expands deterministically and satisfies pairing/counterbalance constraints without GPU; all expected controls appear; matched tasks stay identical across wording arms; hard-budget fixture shows extra tool use reduces achievable task score; probability0/1 endpoints and independent RNG tested; sample unit/interval routines tested on known synthetic cases. Extend study/analysis/protocol tests and add `tests/test_lab_statistics.py`.

## WP7 — Continuous resource safeguards (required; independent early work)

- [x] Resolve data/cache/temp/download/install destinations and the relevant physical backing volumes. On WSL, check host-volume capacity as well as guest free space. Keep the user's large caches/envs/artifacts off C:. Make reserve settings visible and include planned temporary/download overhead.
- [x] Use one storage guard for application-managed downloads, calibration, generation/event writes, imports, exports and publication jobs. Recheck at bounded intervals/chunk boundaries with headroom for outstanding writes. Multiple paths on one volume must not each claim the same free bytes independently.
- [x] Downloader/runtime loading needs a genuinely cancellable supervised path: a progress callback alone does not stop a blocked library download. Bound chunk/concurrency and cancel the owned child process if reserve is reached. Preserve reusable partial downloads or remove only explicitly owned incomplete artifacts; never delete other users' data automatically.
- [x] Emit an explicit resource-stop reason, preserve already recorded evidence and finalize unstarted batch entries. Reserve enough capacity for a minimal final checkpoint/error record; use a preallocated emergency metadata allowance if needed. No completed status after a resource abort.
- [x] UI shows current source volume/free space, reserve, sampled trend and job status. Monitoring does not poll GPU per token or create a log larger than the run it monitors.
- [x] Document the guarantee accurately: controlled application writers stop before their reserve under bounded write rates; external concurrent writes and arbitrary manual installers cannot be absolutely policed without OS quotas. Provide a local setup/download helper using the same guard rather than claiming to control every third-party installer.

**Files:** `lab/storage.py`, `lab/service.py`, `lab/worker.py`, `lab/runtime.py`, `launch_lab.py`; new `lab/resources.py` and, if needed, `prepare_runtime.py`; integrate import/export/publication paths and UI storage card. **Acceptance:** fake disk telemetry crosses reserve during generation/download/import; job stops and keeps valid partial evidence; destination/backing-volume checks, shared-volume reservations and write bursts tested without filling real disks. Worker kill fallback finalizes receipts. Add `tests/test_lab_resources.py`; extend storage/service/worker tests. No destructive disk-filling test or unnecessary model download.

## WP8 — Research UI polish, accessible first run and documentation (required integration)

- [x] Connect new features into coherent Model → Calibrate → Explore → Protocol → Results workflows. Keep independent conversation scrolling, dashboard visibility, stop, restart and persistent auxiliary gating.
- [x] Live/replay charts select the corresponding text/token and show generation phase, site/layer, actor/control revision and tool/transition event. Raw versus any smoothed/downsampled view is explicit; missing measurements show gaps. Keep inspection usable by keyboard and touch. Do not call association scores emotion percentages.
- [x] Expose pause/resume, branch/import, tool/effect editor and protocol preview with clear pending/accepted states. Replay/import and published findings need no model loaded. Include every error path with a recovery action that preserves evidence.
- [x] First run identifies interpreter/profile, directories/reserves and missing dependencies before starting a download. A fresh clone can review bundled findings offline, load the documented 4B profile, calibrate, run a small comparison and export/import it by following the guide.
- [x] Update README, guide, setup docs, roadmap and validation with the exact completed feature set and measured configurations. Preserve published 4B/27B findings as the original study outcomes; new acceptance findings are clearly separate. Add short procedural examples for each new workflow and precise button/cost/decay semantics.

**Files:** `lab/static/index.html`, `lab/static/app.js`, `lab/static/style.css`, `lab/reports.py`, `README.md`, `docs/guide.html`, `docs/roadmap.md`, `docs/validation.md`, `docs/setup-27b.md`; screenshots only after implementation. **Acceptance:** local browser fixture exercises create preset/tool → run → pause/resume → branch → export/import → review; controls/graphs remain usable at 320,390,768,1440px with no overflow/errors; keyboard labels/status announcements and safe rendering checked. Extend `tests/ui_smoke.cjs` and manually inspect representative screenshots. Do not replace the existing stack just to introduce a framework.

**First-run evidence so far:** `check_runtime.py --check` reads the selected
interpreter's package versions and current data/cache/temp backing-volume
reserves without installing, downloading or creating directories. Both existing
environments passed; CUDA is queried only with explicit `--check-cuda` and was
not queried for those receipts. A separate clean data/cache server, with ML
imports blocked, served the 131 earlier published replays without a model
worker. A later clean local clone served all 168 replays, including both new
acceptance archives, over HTTP with ML imports blocked and no worker.
The separate `release-fresh-clone-first-run-01` receipt completed at 14:07:07 UTC
from clean clone `29a79fde6d1372d7f32aad61f5fd1e9b461a52cb`: metadata/reserve
check, local 4B load, exact frozen 160-row extraction, two direct active/sham
cases (4/4 tasks, 394 tokens, zero voluntary calls), HTTP export and HTTP import
into another new data directory. Manifest/summary/conversation/events/parent
events matched, and the imported report used no loaded worker. Both services
closed with exit code zero. Existing runtime dependencies and pinned cached
weights were reused; this is not a clean dependency-installation claim.
The new calibration has no independent semantic ratings.
The later explicit 27B CUDA environment check passed
(`work/runtime-check-27b-cuda.json`): PyTorch 2.8.0+cu128, CUDA 12.8 available,
one device and C++11 ABI enabled, without tensor allocation. It does not close
the model/experiment gate by itself; that separate 27B acceptance has now passed.

## Integration order and parallel ownership

1. Preserve the completed study, record archive hashes, and establish the version/semantics contract before changing behavior.
2. **Parallel wave A:** effects/tools + costs/timing (WP1–2); calibration module/corpus (WP4); resource guard/import internals (WP7 and portable half of WP3). Use module ownership; root alone integrates shared `service.py`, `worker.py`, `runtime.py` and UI changes to avoid overlapping edits.
3. **Wave B:** integrate safe-boundary lifecycle/branching (WP3) and discovery/yoked exposure (WP5); add declarative library/statistics (WP6). Keep changes in small reviewable commits with CPU tests for each semantic unit.
4. **Wave C:** unify UI/docs (WP8), run release acceptance, publish the new acceptance bundle, verify archive hashes and push.

No GPU inference is needed for the bulk of implementation. Use existing fake-runtime tests to settle protocol logic before spending model time. Avoid changing numerical kernels or the existing 27B adapter merely to complete UI/protocol features.

## Final acceptance and done checklist

- [x] All required packages above are implemented, user-accessible, and covered by meaningful local tests; no placeholder button or fake metric is described as complete.
- [x] Local Python suite and browser suite pass; kernel parity is rerun only if relevant runtime/kernel behavior changed. Every published archive checksum and old protocol expansion still matches. A fresh no-model server replays both old and new evidence.
- [x] On the available 4090, run a bounded **4B acceptance protocol** with predetermined seeds/configuration: baseline/no-op parity; signed/attenuation/random validation; renamed tool/cost/stacking/token-vs-action decay; one diagnostic/yoked pair; a stress/relief or pressure factorial sample; pause/resume/branch round-trip. Use tiny/fake integration fixtures for mechanics and real model runs for compatibility/observability; do not pretend the smoke sample establishes every research hypothesis.
- [x] Run a targeted **27B NF4 compatibility smoke** for updated definitions/parser/phase hooks/checkpoints and confirm full GPU placement, resource usage and calibration compatibility. Rerun calibration if the identity changed; never force an incompatible historical bundle to load. A complete second54-episode study is unnecessary unless a substantive change invalidates a comparison being claimed.
- [x] Publish all acceptance runs, failures, resolved protocols, environment and analysis with immutable checksums in a new study directory, linked from the combined findings page/README. Keep scientific claim strength proportional to evidence.
- [x] Verify first-run instructions and portable replay/import on a clean temporary data directory without duplicate multi-GB downloads; record what was simulated versus actually exercised.
- [x] Update the roadmap to list only optional/future items; commit/push; verify repository state and public links; then mark the feature-completion goal complete with a concise feature/validation report.

## Optional extensions and unavailable hardware (not release blockers)

- RTX5090 validation requires actual access to that machine. Supply the portable configuration and a manually invoked hardware validation script/checklist; label 5090 **not yet validated**. Do not fabricate a run or keep the entire app goal open indefinitely solely for unavailable hardware.
- A measured low-memory quantized4B profile, CPU offload, alternate runtimes/quantizers, extra concept directions beyond pain/joy, sparse feature analysis, multi-layer causal patching, persistent KV-cache protocols and exact arbitrary internal-state continuation are optional separately scoped extensions.
- Online reinforcement learning or an artificial reward for pressing the button is outside this frozen-weight release and would test a different question. It must not be introduced implicitly to manufacture button seeking.
- Large powered confirmatory studies across models/hardware remain research projects after the tooling ships. The app should support their prespecified protocols and correct uncertainty analysis; feature completion does not require every study to produce a positive effect.
