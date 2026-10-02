# Release scope and remaining work

Status reviewed October 2, 2026. The research interfaces below are implemented and locally tested. The bounded 4B protocol and diagnostic/yoke/lifecycle stages have completed. **The feature-completion goal remains open:** portable qualification, targeted 27B acceptance, the new acceptance publication, and final release verification are still required. The earlier 27B study is complete; it is separate evidence.

## Implemented research workflows

| Area | Available behavior and limits |
|---|---|
| Tool and effect design | Version-2 editor with neutral tool names, bounded argument schemas, acknowledgments, hidden mappings, costs, immutable libraries and exact JSON/Qwen-template previews. Signed pain/joy, raw/orthogonal joy, separate attenuation, explicit single-site scopes and capped stacking. |
| Cost and timing | Decision opportunities, action-budget units and generated tokens are separate. Unaffordable tools do not execute. Constant, finite, linear and exponential effects use token or decision clocks; prefill positions are recorded separately and do not age clocks. |
| Calibration | Historical schema-1 extraction remains unchanged. Research v2 adds a 960-row authored corpus, disjoint scenario families, train/probe/selection/heldout splits, final/mean/span pooling, ridge/mean readouts, shuffled controls, slice metrics and family uncertainty. Signed/attenuation/random diagnostics and blinded continuation scoring have separate immutable artifacts. |
| Runtime controls | Actual rounded edit-norm matching for pure random controls, on the same unedited position. Unsupported mixed pulses, nonzero held baselines, incompatible timing/sites and unattainable bounded gains fail explicitly. A matching coefficient alone is not treated as matching exposure. |
| Discovery | Isolated scored predictions at saved boundaries, neither/abstain options, invalid/partial/failed/missing denominators, sham false positives and explicit confidence scoring. Operator-declared validation alone never grants interpretation eligibility; independent score bytes must bind to the model, calibration and endpoint. |
| Yoked exposure | Frozen source deliveries replay on token or decision clocks; recipient presses cannot add effects. Delivered exposure and partial coverage are recorded. A yoked boundary supports explicitly sham diagnostics; carrying its live schedule into a diagnostic remains unsupported. |
| Sessions and evidence | Completed-turn pause/resume, strict JSON checkpoints, continued-state branches, explicit new allowances and full-visible-history transfer to a fresh task. Bundle and legacy JSON exchange use no model worker; the current portable acceptance failure remains open below. KV-cache restoration is outside this release. |
| Protocols and analysis | Frozen smoke/full matrices, dry-run costs, independent random streams, paired IDs, stage receipts, explicit retries and first/latest attempt policies. Hard allocation/assignment tasks vary objective difficulty independently of wording. Episode/family intervals, sample-planning assumptions and preregistration templates are supplied. Event-derived v2 reports separate transition timing, task position, costs and measured exposure from voluntary choices. |
| Storage and setup | Shared-volume capacity accounting, WSL host-volume checks, continuous managed-write checks, owned-process cancellation and emergency metadata. `prepare_runtime.py` supervises explicit setup commands; `check_runtime.py` reads interpreter/package versions and storage reserves before model work. External writers still require OS quotas for a hard guarantee. |
| Browser | Unified designer/calibration/protocol/replay workflow, independent conversation scrolling, acknowledged controls, exact preview downloads, raw-token keyboard/pointer inspection and corresponding-message selection. Published findings and imports need no model worker. |

See [research workflows](research-workflows.md), [independent criterion evidence](criterion-evidence.md), the [user guide](guide.html), and the [implementation checklist](feature-completion-plan.md) for supported paths and precise semantics. The research corpus and probe scores measure associations; they do not by themselves demonstrate an effect on generated behavior or subjective experience.

## Release work still required

1. Resolve the retained portable-attempt failure (`Unsafe bundle path`) and complete a separately recorded portable round trip. Do not replace or hide the failed invocation.
2. Run a targeted 27B NF4 smoke for the new parser, phase hooks and checkpoint paths, with compatible calibration and measured GPU residency. The previous 54-episode study does not qualify these new code paths.
3. Publish the new resolved protocols, all successful/failed/partial acceptance outputs, environment records and checksums in a separate study directory. Complete clean-clone first-run/replay checks, relevant final test and compatibility reruns, review and push before marking the goal complete.

The current full-suite receipt contains 709 passing tests; browser coverage
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
No tested nonzero calibration dose met the frozen selection bounds. The selected
dose is zero, with four truncated, unrated sham continuations; the 0.25 protocol
cases are engineering stress checks, not evidence of validated semantic efficacy.

The implementation audit and [validation ledger](validation.md) distinguish CPU fixtures, browser checks, completed engineering stages, earlier real-model findings and pending release work. No independent behavioral effect or discovery result is inferred from software checks or this small acceptance sample.

## Preserved findings

The pinned Qwen3.8-27B NF4 model ran fully on the RTX 4090 in the earlier study: **54/54 episodes, 209/210 tasks correct and zero voluntary auxiliary calls across 628 decisions**. The [combined findings](results.html) retain its comparison with 4B and link to the [27B archive](../studies/qwen38-27b/README.md). Those observations remain unchanged.

`python3 check_compatibility.py` protects 808 published files, three study expansions, ten legacy recipe defaults and 24 model-visible payloads. Opting into new recipe or calibration versions never upgrades old evidence retrospectively.

## Optional expansion and unavailable hardware

RTX 5090 validation requires that hardware and remains unverified. Native Windows GPU inference, a measured low-memory quantized 4B profile, CPU offload, alternate runtimes, additional concepts, sparse features, multi-layer patching and persistent KV-cache experiments are separate extensions. Linux/WSL remains the tested inference environment.

Online reinforcement learning is outside this frozen-weight release. Larger powered studies remain research projects after the tooling ships; a negative behavioral result is a valid completed experiment, not a reason to tune until a model presses a button.
