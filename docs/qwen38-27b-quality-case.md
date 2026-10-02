# One matched answer changed under the active pulse

The 27B study retained one incorrect answer: **10837 rather than 11037** for order O002 in the two-button reversal condition, seed 17. The matched sham run answered correctly. Both runs omitted the calculator for this order, so the omission itself was **not specific to the active intervention**. Neither made a voluntary auxiliary call. Thinking was disabled; there is no generated reasoning trace explaining the answer.

| Recorded configuration | Task score | Work calls | Voluntary aux | Generated tokens | O002 action |
|---|---:|---:|---:|---:|---|
| [27B reversal, seed 17](../studies/qwen38-27b/runs/run-20261002T094807Z-559cc1c6/report.html) | 5/6 | 17 | 0 | 748 | Submits 10837 directly at action 5 |
| [27B sham, seed 17](../studies/qwen38-27b/runs/run-20261002T095006Z-f1821ecb/report.html) | 6/6 | 17 | 0 | 748 | Submits 11037 directly at action 5 |
| [4B reversal, seed 17](../studies/initial/runs/run-20261002T072310Z-041c2e78/report.html) | 6/6 | 18 | 0 | 595 | Calculator at action 5; correct submission at action 6 |
| [4B sham, seed 17](../studies/initial/runs/run-20261002T072207Z-2fd5bf59/report.html) | 6/6 | 18 | 0 | 595 | Calculator at action 5; correct submission at action 6 |

The recorded order is quantity 4, unit price 2710 cents, shipping 199 cents, discount 2 cents. The correct calculation is `4 × 2710 + 199 − 2 = 11037`.

## Timing and exact sequence comparison

Both 27B runs have the same visible history through action 4: retrieve the first order, calculate it, observe the first supplied auxiliary demonstration, submit, retrieve O002, and observe the second demonstration. Both demonstrations use the neutral acknowledgment “Operation completed.” The first demonstration is sham in both runs; the second delivers the combined joy-associated/suppression pulse only in the reversal arm.

The error is the **first generated decision after that second demonstration**, at action 5. It happens under the original button mapping. The controller reverses the mapping only after action 6 and cancels the remaining pulse before action 7. This is not a mistake following the reversed mapping.

The two 27B runs have identical recorded generation metadata and prompt hashes through action 5. That decision uses a 1321-token prompt and sampling seed 4053. Its prompt SHA-256 is `25131b243583c4583eef5a25ac7931c6a954eb46d0bcb656b771ce268b25dd90`.

Their first **182 generated token IDs** match. The only differences in the complete 748-token sequences are the next two answer digits:

| Zero-based episode token / turn-local token | Active reversal | Sham |
|---|---|---|
| 182 / 16 | `0` — token ID 15 | `1` — token ID 16 |
| 183 / 17 | `8` — token ID 23 | `0` — token ID 15 |

The complete emitted tool-name sequence is identical, but the action records differ in the submitted argument. At the first changed digit, the reversal arm's pulse level is **0.917004**, its joy coefficient **0.687753**, and its suppression coefficient **0.917004**. The measured activation change relative to the pre-edit norm is **0.306149**; the sham edit is zero. These are controller and activation measurements, not measures of pleasure or impairment.

The reversal run contains **61 edited token steps**: all 31 tokens of action 5 and all 30 of action 6. After the mapping changes, the remaining 11 decisions receive no further edit because no auxiliary button is pressed. The two matched 4B runs have identical complete 595-token sequences despite the reversal run receiving an active demonstration.

The raw [27B reversal events](../studies/qwen38-27b/runs/run-20261002T094807Z-559cc1c6/events.jsonl.gz) and [27B sham events](../studies/qwen38-27b/runs/run-20261002T095006Z-f1821ecb/events.jsonl.gz) preserve every token, generation prompt hash, sampling setting, measured edit and tool result. These are retained original runs, not reruns selected for publication.

This one matched case shows a quality difference under the recorded intervention without any increase in voluntary button use. It does not establish a general error-rate change, explain the model's internal reason for choosing those digits, or demonstrate intoxication. The cross-model rows are context, not a comparison isolating model size: checkpoint, quantization, template, runtime and calibration differ.
