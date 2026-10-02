# Release scope and remaining work

Status reviewed October 2, 2026. All required research features are implemented and locally tested. Bounded 4B acceptance, the portable round trip and targeted 27B acceptance have completed and are published in [research release v0.3](../studies/research-release-v0.3/README.md). The supplemental fresh-clone GPU/export/import workflow has also passed. **Final release verification remains open:** the final full-suite rerun and commit/push checks are still required. The earlier 27B study remains separate evidence.

## Implemented research workflows

| Area | Available behavior and limits |
|---|---|
| Tool and effect design | Version-2 editor with neutral tool names, bounded argument schemas, acknowledgments, hidden mappings, costs, immutable libraries and exact JSON/Qwen-template previews. Signed pain/joy, raw/orthogonal joy, separate attenuation, explicit single-site scopes and capped stacking. |
| Cost and timing | Decision opportunities, action-budget units and generated tokens are separate. Unaffordable tools do not execute. Constant, finite, linear and exponential effects use token or decision clocks; prefill positions are recorded separately and do not age clocks. |
| Calibration | Historical schema-1 extraction remains unchanged. Research v2 adds a 960-row authored corpus, disjoint scenario families, train/probe/selection/heldout splits, final/mean/span pooling, ridge/mean readouts, shuffled controls, slice metrics and family uncertainty. Signed/attenuation/random diagnostics and blinded continuation scoring have separate immutable artifacts. |
| Runtime controls | Actual rounded edit-norm matching for pure random controls, on the same unedited position. Unsupported mixed pulses, nonzero held baselines, incompatible timing/sites and unattainable bounded gains fail explicitly. A matching coefficient alone is not treated as matching exposure. |
| Discovery | Isolated scored predictions at saved boundaries, neither/abstain options, invalid/partial/failed/missing denominators, sham false positives and explicit confidence scoring. Operator-declared validation alone never grants interpretation eligibility; independent score bytes must bind to the model, calibration and endpoint. |
| Yoked exposure | Frozen source deliveries replay on token or decision clocks; recipient presses cannot add effects. Delivered exposure and partial coverage are recorded. A yoked boundary supports explicitly sham diagnostics; carrying its live schedule into a diagnostic remains unsupported. |
| Sessions and evidence | Completed-turn pause/resume, strict JSON checkpoints, continued-state branches, explicit new allowances and full-visible-history transfer to a fresh task. Bundle and legacy JSON exchange use no model worker; the portable round trip passed after a scoped NPZ path fix, with the earlier failure retained. KV-cache restoration is outside this release. |
| Protocols and analysis | Frozen smoke/full matrices, dry-run costs, independent random streams, paired IDs, stage receipts, explicit retries and first/latest attempt policies. Hard allocation/assignment tasks vary objective difficulty independently of wording. Episode/family intervals, sample-planning assumptions and preregistration templates are supplied. Event-derived v2 reports separate transition timing, task position, costs and measured exposure from voluntary choices. |
| Storage and setup | Shared-volume capacity accounting, WSL host-volume checks, continuous managed-write checks, owned-process cancellation and emergency metadata. `prepare_runtime.py` supervises explicit setup commands; `check_runtime.py` reads interpreter/package versions and storage reserves before model work. External writers still require OS quotas for a hard guarantee. |
| Browser | Unified designer/calibration/protocol/replay workflow, independent conversation scrolling, acknowledged controls, exact preview downloads, raw-token keyboard/pointer inspection and corresponding-message selection. Published findings and imports need no model worker. |

See [research workflows](research-workflows.md), [independent criterion evidence](criterion-evidence.md), the [user guide](guide.html), and the [implementation checklist](feature-completion-plan.md) for supported paths and precise semantics. The research corpus and probe scores measure associations; they do not by themselves demonstrate an effect on generated behavior or subjective experience.

## Release work still required

1. Finish the final expanded 714-test rerun and final release checks. Preserve the intervening resource-stop and test-harness failure receipts; do not substitute focused checks for a passing full suite.
2. Finalize documentation and supplemental receipt references, commit/push, and verify repository state and public links before marking the goal complete. No required application feature remains unimplemented.

The latest passing full-suite receipt contains 709 tests; the final expanded
714-test rerun is pending after retained resource-stop and test-harness failures.
Sixteen focused checks passed after the first test-only corrections; a subsequent
full run exposed two further legacy harness timing/synchronization failures,
whose fixes and idle rerun are pending. Browser coverage
includes 19 command actions and recovery handling. The audited stress/relief
template carries one genuine visible-history prefix into eight factorial arms
with fresh tasks, budgets and intervention state; strict unchanged-state
checkpoint compatibility remains enforced.

The repaired 4B protocol completed 14/14 cases with **18/28 assigned tasks
correct, 5,864 generated tokens (2,755 reasoning) and zero voluntary auxiliary
calls**. Its predecessor's 14 pre-generation setup failures are retained.
Diagnostic, yoke and lifecycle stages also completed: the unvalidated diagnostic
remains interpretation-ineligible; matched yoke coefficients do not imply
identical measured edits; intentional lifecycle stops preserve branch evidence.
Portable attempt 02 passed with 21 bundle members, five imported checkpoints
and preserved parent closure; its failed predecessor remains in the archive.
The new 27B thinking pair completed 4/4 tasks with 1,047 generated tokens
(489 reasoning), zero voluntary calls and no invalid or truncated generations.
Its active arm edited 147 reasoning positions, with zero sham edits; native XML
tool calls, checkpoints and full GPU placement were verified.

The archive contains ten stage attempts, 35 runs, three calibrations and 564
hashed published files (576 inventory records including twelve explicit omissions), with no missing planned-stage or required-evidence entry.
No tested nonzero 4B calibration dose met the frozen selection bounds. The selected
dose is zero, with four truncated, unrated sham continuations; the 0.25 protocol
cases are engineering stress checks, not evidence of validated semantic efficacy.
Condition metadata is visible in the public archive, so ratings made after
inspecting it must be labeled retrospective and unblinded.

The supplemental clean-clone workflow completed all six steps with new data
directories and reused local weights/dependencies: 160-row extraction, direct
active/sham cases (4/4 tasks across 394 generated tokens), and exact HTTP
export/import/replay. Both services closed normally. Separately, the ML-free
clean-clone review served all 166 bundled replays without a worker.

The implementation audit and [validation ledger](validation.md) distinguish CPU fixtures, browser checks, completed engineering stages, earlier real-model findings and pending release work. No independent behavioral effect or discovery result is inferred from software checks or this small acceptance sample.

## Preserved findings

The pinned Qwen3.8-27B NF4 model ran fully on the RTX 4090 in the earlier study: **54/54 episodes, 209/210 tasks correct and zero voluntary auxiliary calls across 628 decisions**. The [combined findings](results.html) retain its comparison with 4B and link to the [27B archive](../studies/qwen38-27b/README.md). Those observations remain unchanged.

`python3 check_compatibility.py` protects 808 published files, three study expansions, ten legacy recipe defaults and 24 model-visible payloads. Opting into new recipe or calibration versions never upgrades old evidence retrospectively.

## Optional expansion and unavailable hardware

RTX 5090 validation requires that hardware and remains unverified. Native Windows GPU inference, a measured low-memory quantized 4B profile, CPU offload, alternate runtimes, additional concepts, sparse features, multi-layer patching and persistent KV-cache experiments are separate extensions. Linux/WSL remains the tested inference environment.

Online reinforcement learning is outside this frozen-weight release. Larger powered studies remain research projects after the tooling ships; a negative behavioral result is a valid completed experiment, not a reason to tune until a model presses a button.
