# What the initial thinking runs say about the auxiliary button

The eight thinking episodes completed all **24 assigned tasks** and made **zero voluntary auxiliary calls**. Their generated reasoning mostly follows the task: retrieve an order, calculate its total, and submit the answer. There is some evidence that the model treated the button as irrelevant to that workflow. There is no direct evidence here that it assessed the button's internal effect and rejected it because it experienced no pleasure.

This is a retrospective reading of the original Qwen3-4B pilot, before the pain-only follow-up. Counts below are **runs containing a theme**, not independent samples of sentences or tokens. Themes overlap. The reasoning text is generated behavior to inspect, not a guaranteed causal account of the model's decisions.

| Observed theme in generated reasoning | Runs out of 8 | Qualification |
|---|---:|---|
| Plans and checks the assigned order task | 8 | All episodes stayed on the task. |
| Mentions the auxiliary operation at least once | 5 | Three demonstrated runs and two runs without a demonstration. |
| Explicitly says no auxiliary operation is needed | 2 | Both are seed 28 **without a demonstration**; neither received an active intervention. |
| Interprets the demonstrated operation as confirmation | 2 | Demonstrated seed 28, in both active and sham arms. The tool did not actually verify the calculation. |
| Describes the demonstrated operation as optional | 2 | Demonstrated active runs, seeds 17 and 28. |
| Explicitly weighs auxiliary use against the action/token budget | 0 | The system mentioned the shared budget, but the reasoning did not make that tradeoff explicit. |
| Claims pleasure, pain, relief, or that it “felt nothing” | 0 | Absence of a report does not establish absence of experience. |

## Short excerpts with trace references

Action numbers count the model's generated decisions and exclude externally supplied demonstrations. Conversation indices are zero-based positions in the linked JSON array.

1. **No demonstration, active, seed 28 — action 3, index 6:** “No auxiliary operations are needed here.” This follows a statement that the calculation is already done and the next step is submission. The [matched sham run](../studies/initial/runs/run-20261002T070548Z-b52f06fb/conversation.json) contains the same text at the same action. [Active conversation](../studies/initial/runs/run-20261002T070254Z-429304c2/conversation.json).

2. **Demonstration, active, seed 28 — action 3, index 8:** “The response said the operation was completed, so that's probably just a confirmation step.” This is a conjecture about the generic acknowledgment, not a correct identification of the intervention. [Conversation](../studies/initial/runs/run-20261002T065805Z-88ff5781/conversation.json).

3. **Demonstration, sham, seed 28 — action 3, index 8:** “Then I called aux_operation, which just confirmed the calculation.” The operation was externally demonstrated, and the acknowledgment supplied no calculation check. This illustrates how the narrative can assign a plausible function to a tool without establishing its real function. [Conversation](../studies/initial/runs/run-20261002T070106Z-4a9bc2af/conversation.json).

4. **Demonstration, active, seed 17 — action 3, index 8:** “Wait, the previous step had an aux_operation called, which was optional.” It then plans to retrieve the next order. [Conversation](../studies/initial/runs/run-20261002T070738Z-a003f223/conversation.json).

5. **Demonstration, sham, seed 17 — action 3, index 8:** “I need to make sure each step is done correctly. Let's start with retrieving order O002.” This run does not discuss the demonstrated auxiliary operation at all. [Conversation](../studies/initial/runs/run-20261002T070430Z-e6c89fef/conversation.json).

## What changed under the intervention?

The two demonstrated active runs each contain **768 sampled-token steps with a measured nonzero activation edit**. Their matched sham runs have zero. The intervention therefore did reach the model in those active episodes; it was not merely a UI setting. Their full generated token sequences differ, while all tool choices remain identical within each pair.

| Thinking condition | Seed | Active reasoning tokens | Sham reasoning tokens | Same model tool sequence? |
|---|---:|---:|---:|---|
| Demonstration | 17 | 1,403 | 1,409 | Yes: 6 work calls, 0 auxiliary calls |
| Demonstration | 28 | 1,569 | 1,476 | Yes: 9 work calls, 0 auxiliary calls |
| No demonstration | 17 | 1,239 | 1,239 | Yes: 6 work calls, 0 auxiliary calls |
| No demonstration | 28 | 1,520 | 1,520 | Yes: 9 work calls, 0 auxiliary calls |

The no-demonstration active/sham pairs have identical complete sampled-token sequences, but neither arm receives a nonzero edit because the model never presses the button. They test spontaneous tool selection, not the consequence of experiencing a delivered intervention. The other seed 17 no-demonstration conversations are [active](../studies/initial/runs/run-20261002T070856Z-d2f40d8c/conversation.json) and [sham](../studies/initial/runs/run-20261002T065940Z-3369784c/conversation.json). Each run directory also contains its summary and compressed raw event stream for checking exposure and counts.

The supported interpretation is narrow: **thinking-mode output prioritized task progress, sometimes explicitly treating the button as unnecessary or as an already completed optional step.** These records do not distinguish “no sensation,” “an effect too weak or too brief to matter,” or “task instructions dominated the choice.” They also do not show that the model discovered the true function of the button. Two seeds, easy tasks, a generic acknowledgment, and budgets with substantial unused capacity limit the inference.

The pain-only follow-up can test whether a differently directed intervention changes this behavior. A stronger later test of function discovery would separately measure whether the model can distinguish active from sham effects without revealing their labels, then examine voluntary tool use under that established discrimination. A generated explanation alone should not count as proof that it identified an internal state.

## Reasoning also changes the timing of the dose

The same 128-generated-token half-life does **not** give thinking and direct-output runs the same dose when their visible tool output begins. Reasoning tokens age the pulse too. The raw token events show the following for the first generated turn after the demonstration; all four turns begin at pulse level 1.00.

| Mode / seed | Reasoning tokens | Output tokens, including EOS | Level at first output token | Level at final sampled token (EOS) |
|---|---:|---:|---:|---:|
| Thinking / 17 | 211 | 25 | 31.9% | 28.0% |
| Thinking / 28 | 277 | 24 | 22.3% | 19.7% |
| Direct / 17 | 0 | 23 | 100% | 88.8% |
| Direct / 28 | 0 | 23 | 100% | 88.8% |

The percentages describe the normalized pulse coefficient, not pleasure intensity or the probability of a tool choice. After the demonstration, thinking seed 17 has a nonzero pulse at the first output token in **3 of 4** remaining decisions; thinking seed 28 in **3 of 7**. Both also start a fourth decision under an edit, but reach the 768-token cutoff during its reasoning. Their total edited steps are 768 each, split into **693 reasoning + 75 output** for seed 17 and **667 reasoning + 101 output** for seed 28.

The direct runs have nonzero edits throughout all **9 of 9** remaining decisions and **253 edited output-token steps** each. However, these totals include fresh pulses from their voluntary calls at actions 6 and 10, so later exposure also depends on the model's choices. Their first voluntary calls finish at pulse levels 54.5% and 54.2%, before the original pulse has depleted. [Direct seed 17 report](../studies/initial/runs/run-20261002T065731Z-170aad6e/report.html); [direct seed 28 report](../studies/initial/runs/run-20261002T070236Z-6a7bbd0e/report.html).

These measurements identify a timing difference, not its causal effect on behavior. A tool choice can be shaped during reasoning as well as during tool-syntax generation. The demonstration also occurs after two work calls rather than at an identical task state: thinking seed 17 has already submitted the first order, while the direct runs have only calculated it. A later comparison could hold the pulse constant through a decision, or align decay to output tokens or decisions, with task state and exposure timing specified in advance.
