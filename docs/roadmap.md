# Release scope and remaining work

Status reviewed October 2, 2026. Opium Bench's core lab is usable. The broader
[research and build plan](../LAB_PLAN.html) is not fully implemented; finishing
the current 27B study does not complete every item in that plan.

## Available now

- Local Qwen loading with pinned revisions, calibration compatibility checks,
  storage preflight, and full GPU placement checks.
- Pain/joy direction extraction, separate measurement probes, layer selection,
  held-out association checks, and a numerical dose diagnostic. Custom calibration
  examples can replace the bundled corpus.
- Conversation, visible generated reasoning, independent chat scrolling, live
  measurements, baseline sliders, manual or model-triggered pulses, decay,
  effect gating, reset, stop, restart, and in-process pause/resume.
- Shared task/effect budgets; active/sham, demonstration/no-demonstration,
  thinking, ingredient, two-button reversal, joy-to-pain transition, and
  probabilistic-outcome recipes. Orders and constraint puzzles have local graders.
- Seeded batches, saved raw evidence, conversation replay, descriptive comparisons,
  reports, JSON exports, and published [4B findings](results.html).

The Qwen3.8-27B NF4 runtime has loaded fully on the RTX 4090, with separate
calibration and kernel/tool-calling checks. Its **54-episode behavioral study is
complete: 209/210 tasks correct and zero voluntary auxiliary calls**. The
[combined findings](results.html) preserve the comparison with 4B and the raw
[27B evidence](../studies/qwen38-27b/README.md). This completes that study;
the planned features below remain separate implementation work.

The [implementation and acceptance checklist](feature-completion-plan.md) tracks
the active completion goal. A frozen compatibility contract now protects 808
published evidence files, all three study expansions, ten legacy recipe defaults
and 24 model-visible payloads. Run `python3 check_compatibility.py` to verify it.

## Remaining items explicitly described in the plan

| Area | Current limit and remaining work |
|---|---|
| [Tool and effect design](../LAB_PLAN.html#workbench) | Recipes expose preset parameters, but auxiliary names, schemas, and acknowledgment are fixed. A general editor, variable per-tool costs, selectable raw/orthogonalized directions, and optional capped dose stacking remain. |
| [Button discovery](../LAB_PLAN.html#den) | Balanced exposure, disclosure, and reversal exist. Separate scored predictions of what a button does, false-positive/no-effect tests, and automatic matched-exposure controls remain. Repetition or generated explanations alone do not establish discovery. |
| [Calibration specificity](../LAB_PLAN.html#calibration) | Current calibration uses 72 authored sentences, final-token activations, mean-contrast probes, and one selected edit layer. Broader lexical/context controls, pooling comparisons, label-shuffled validation, signed calibration sweeps, and independently scored continuation effects remain. |
| [Session controls and branches](../LAB_PLAN.html#workbench) | Stop/restart and completed-turn pause/resume work. Durable checkpoint recovery is still in progress. API branches currently accept user/assistant text, without restoring tool state, cache, or RNG. A complete conversation-branch workflow remains; exact internal-state continuation is a separate protocol. |
| [Timing and research batteries](../LAB_PLAN.html#thinking) | Decay uses generated tokens; action-based decay remains. The stress/relief, cost/transfer, and broader task-pressure comparisons in the plan are not complete automated batteries. Confirmatory sample planning and uncertainty analysis also remain beyond the descriptive pilot. |
| [Storage safeguards](../LAB_PLAN.html#models) | Admission checks preserve a configured reserve. Continuous monitoring and clean cancellation before every large job exhausts that reserve remain. |

The [user guide](guide.html) documents practical limits: custom recipes have a
fixed schema; some advanced options require the API; replaying an exported run
on another installation requires copying its directory. There is no automatic
GUI import of a JSON export.

## Optional expansion and hardware validation

The plan explicitly defers additional concepts, sparse-feature investigations,
and multi-layer causal patching. Online reinforcement learning is outside this
initial frozen-weight lab. CPU offloading is a lower-priority optional path.

The RTX 5090 and a measured low-memory quantized 4B profile still need their own
validation. A successful 27B load on the 4090 does not validate another GPU,
checkpoint, quantization format, or context length. See the
[27B setup guide](setup-27b.md) for the configuration actually checked.

All current findings concern computation and observable behavior. Neither this
release nor a completed study establishes whether a model has subjective experience.
