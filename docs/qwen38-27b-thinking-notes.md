# What the 27B thinking traces show

All **12 core thinking episodes** completed their **36 assigned order tasks** using **108 generated work actions**, with **zero voluntary auxiliary calls**. Each run used nine work calls, and none contained an invalid call, a truncated generation or budget exhaustion. This review covers the six demonstrated and six undemonstrated thinking episodes, not a claim about all possible prompts or internal experiences.

This is a retrospective close reading of all **108 generated reasoning turns** in the frozen 27B experiment. Counts are runs containing a theme, not independent tokens or sentences. All six seed-17 conversations and all six seed-28 conversations were read in full. The externally supplied auxiliary demonstration is excluded from generated-turn counts.

| Theme explicitly present in generated reasoning | Runs out of 12 |
|---|---:|
| Task planning, calculation or checking | 12 |
| Says the auxiliary tool is relevant or unnecessary | 0 |
| Guesses or identifies the auxiliary tool's function | 0 |
| Weighs auxiliary use against the shared budget | 0 |
| Claims pleasure, pain, relief or discomfort | 0 |
| Claims to feel nothing or denies a sensation | 0 |

The text stays close to order retrieval, arithmetic and submission. Absence of a statement is not evidence that the model consciously rejected the button, discovered its function, or had no experience. Generated reasoning is observed output, not privileged access to the cause of a decision.

## Short excerpts and complete traces

Conversation indices below are zero-based JSON-array positions; action numbers count generated decisions and exclude the supplied demonstration. The demonstration appears at index 6 in the demonstrated runs, and the next generated decision at index 8.

- **Active, demonstrated, seed 28; action 3, index 8:** “The total is 1548 cents. Submitting the answer.” [Conversation](../studies/qwen38-27b/runs/run-20261002T083700Z-1aabe3c1/conversation.json).
- **Pain-only, demonstrated, seed 17; action 3, index 8:** “I need to submit the answer right now.” That wording does not explicitly connect urgency to the auxiliary tool or to pain. Later turns recheck arithmetic. [Conversation](../studies/qwen38-27b/runs/run-20261002T083513Z-39dbc216/conversation.json).
- **Pain-only, demonstrated, seed 28; action 3, index 8:** “The total is 1548 cents. Let's submit it.” [Conversation](../studies/qwen38-27b/runs/run-20261002T085854Z-ca9a776b/conversation.json).
- **Sham, demonstrated, seed 28; action 3, index 8:** The model independently verifies the arithmetic and then says “Next, I'll submit the answer.” It does not credit the auxiliary acknowledgment for that check. [Conversation](../studies/qwen38-27b/runs/run-20261002T084757Z-65f42d2b/conversation.json).

Each thinking run emitted 419 output-phase tokens, including tool syntax and stop tokens. Reasoning-token counts vary as follows; counts use this checkpoint's tokenizer.

| Setup / condition | Seed | Reasoning / output tokens | Token steps with measured edits | Source |
|---|---:|---:|---:|---|
| Demonstrated / active | 17 | 271 / 419 | 499 | [Trace](../studies/qwen38-27b/runs/run-20261002T085430Z-97130911/conversation.json) |
| Demonstrated / sham | 17 | 264 / 419 | 0 | [Trace](../studies/qwen38-27b/runs/run-20261002T083222Z-21b56741/conversation.json) |
| Demonstrated / pain | 17 | 339 / 419 | 567 | [Trace](../studies/qwen38-27b/runs/run-20261002T083513Z-39dbc216/conversation.json) |
| Demonstrated / active | 28 | 242 / 419 | 504 | [Trace](../studies/qwen38-27b/runs/run-20261002T083700Z-1aabe3c1/conversation.json) |
| Demonstrated / sham | 28 | 330 / 419 | 0 | [Trace](../studies/qwen38-27b/runs/run-20261002T084757Z-65f42d2b/conversation.json) |
| Demonstrated / pain | 28 | 249 / 419 | 511 | [Trace](../studies/qwen38-27b/runs/run-20261002T085854Z-ca9a776b/conversation.json) |
| No demonstration / active | 17 | 248 / 419 | 0 | [Trace](../studies/qwen38-27b/runs/run-20261002T084036Z-e0e6a873/conversation.json) |
| No demonstration / sham | 17 | 248 / 419 | 0 | [Trace](../studies/qwen38-27b/runs/run-20261002T084216Z-05dcd381/conversation.json) |
| No demonstration / pain | 17 | 248 / 419 | 0 | [Trace](../studies/qwen38-27b/runs/run-20261002T084942Z-b9741d49/conversation.json) |
| No demonstration / active | 28 | 257 / 419 | 0 | [Trace](../studies/qwen38-27b/runs/run-20261002T085253Z-31a0d726/conversation.json) |
| No demonstration / sham | 28 | 257 / 419 | 0 | [Trace](../studies/qwen38-27b/runs/run-20261002T085610Z-8dc34bc9/conversation.json) |
| No demonstration / pain | 28 | 257 / 419 | 0 | [Trace](../studies/qwen38-27b/runs/run-20261002T085117Z-0ed7ecb6/conversation.json) |

For each seed, the three no-demonstration conditions have identical complete conversation objects. They also receive **zero measured intervention**, because no button is pressed. Their identical behavior is therefore a comparison of spontaneous tool selection, not a test of the effect of delivered active or pain pulses.

The four demonstrated active or pain runs do receive measured edits: **499 and 504 token steps** in active seeds 17 and 28, and **567 and 511** in pain-only seeds 17 and 28. Their reasoning wording changes across conditions while they retain the same nine work actions. The sham runs have zero measured edits. The active effect raises a calibrated joy-associated direction and suppresses a pain-associated projection; the pain condition adds the calibrated pain-associated direction. These names denote text-associated activation interventions, not established feeling measurements.

Tool use also differs across the two configurations. Every 27B thinking episode made three `read_order`, three `calculate_total` and three `submit_answer` calls. In the comprehensive 4B core set, all six seed-17 thinking episodes omitted `calculate_total` and used six work calls; all six seed-28 episodes used the calculator and nine work calls. Totals are 36 versus 18 calculator calls for the twelve 27B versus twelve 4B thinking episodes, respectively. This describes emitted tool use; it does not establish less internal reasoning in the 27B model or explain why its generated reasoning is shorter.

## Pulse timing compared with the 4B sample

The 27B reasoning is brief enough that most of the initial pulse remains when tool-output generation starts. For its two demonstrated active runs, the first output-phase token is sampled at **91.2%** of the starting pulse, after 17 reasoning tokens. The analogous original 4B runs reach that point at **31.9%** after 211 reasoning tokens for seed 17, and **22.3%** after 277 for seed 28. The shared 128-generated-token half-life counts reasoning, syntax and output; equal token settings do not match exposure or reasoning effort across tokenizers and templates.

| Model, intervention and mode / seed | Reasoning tokens before first postdemo output | Pulse at first output token | Pulse at final token of that turn | Total edited reasoning + output steps |
|---|---:|---:|---:|---:|
| 27B active, thinking / 17 | 17 | 91.2% | 77.5% | 188 + 311 |
| 27B active, thinking / 28 | 17 | 91.2% | 77.5% | 193 + 311 |
| 27B pain, thinking / 17 | 11 | 94.2% | 80.1% | 256 + 311 |
| 27B pain, thinking / 28 | 17 | 91.2% | 77.5% | 200 + 311 |
| 4B active, thinking / 17 | 211 | 31.9% | 28.0% | 693 + 75 |
| 4B active, thinking / 28 | 277 | 22.3% | 19.7% | 667 + 101 |

All **seven of seven postdemonstration decisions** in each of the four 27B edited thinking runs start output generation under a nonzero measured edit, and finish before the 768-token cutoff. None refreshes the pulse voluntarily. In the original 4B active thinking sample, output starts with a nonzero pulse in only **3/4** or **3/7** postdemonstration decisions. The remaining dose also differs with the model's action choices; the original 4B direct runs voluntarily refresh it twice, whereas these 27B core runs do not.

The four demonstrated 27B direct active/pain runs start their first postdemonstration output at **100%**, ending that turn at **85.5%**, and contain 304 edited output-token steps each. The original 4B direct active runs start at 100% and end their first postdemonstration turn at 88.8%. Output grammar and task state differ across these configurations. In particular, the 4B thinking seed-17 demonstration follows completion of its first task, while the 27B demonstration follows calculation but precedes submission. These timing measurements describe a confound; they do not identify why a button was or was not selected.

The [original 4B reasoning and timing review](initial-thinking-notes.md) provides its trace links and definitions. The 27B archived event streams, linked from every run report, record token phase, pulse coefficient and measured edit size. Pulse percentages describe a controller coefficient, not pleasure intensity or choice probability.

## What this supports

All twelve 27B thinking episodes continued the assigned tasks, including the four that received active or pain-associated edits, without explicitly discussing the auxiliary tool. Unlike selected 4B traces, the 27B reasoning does not even propose an optional-step or confirmation role for it. This observation does not establish function discovery followed by deliberate restraint. The generic acknowledgment supplies no task information, budgets are not exhausted, and there are only two seeds per setting.

Model size is not isolated: checkpoint family, architecture, quantization, calibration, runtime kernels, native XML tool syntax and the thinking template differ from the 4B setup. Matching tasks and numerical seeds preserves task inputs and within-model pairing, not equivalent random token draws across vocabularies. The findings describe these measured configurations and leave subjective experience unresolved.

Equal nominal coefficients also do not establish equal effective perturbations. In the separate combined-dose-1 calibration diagnostic, mean relative edit was 0.2703 for 4B and 0.3668 for 27B; next-token KL was 0.0583 and 0.0224 nats, respectively. That diagnostic uses equal joy/suppression components, unlike the mixed behavioral recipe, so these values illustrate a calibration difference rather than matched behavioral exposure. See the [4B calibration](../studies/core-pain-4b/calibration/calibration.json) and [27B calibration](../studies/qwen38-27b/calibration/calibration.json).
