# Qwen3.8-27B NF4 replication

Status: **complete** — 54/54 recorded episodes.

[Open the results dashboard](../../docs/results-27b.html) · [Computed results](results.json) · [Execution receipt](receipt.json) · [Frozen protocol](protocol.json)

Across the recorded episodes: **209/210 assigned tasks correct**, **0/628 voluntary aux choices**, and **31868 generated tokens**. These totals mix distinct experimental conditions; use the dashboard’s matched comparisons to interpret effects.

Across model studies, checkpoint, quantization/runtime, native tool grammar or reasoning-template settings, and independently calibrated intervention directions can change together. They do not isolate model size. Matched active/sham comparisons within each study remain the primary comparisons.

The corpus and probes measure text-associated activation directions, not subjective emotion. The two planned seeds per condition are descriptive; tokens are not independent replicates. See the dashboard for zero-exposure pairs, failures, task denominators, and interpretive limits.

The calculator-assisted order tasks are easy, and the 20-action/3-order or 30-action/6-order budgets allow patterned aux use while still completing every task. A perfect task score does **not** rule out a preference that would become costly under tighter budgets. Harder tasks, binding budgets, dose sweeps, and longer learning periods remain future experiments.

Each run directory preserves its manifest, summary, conversation, deterministic compressed raw event trace, and standalone report. Calibration vectors and their split-validation manifest are in [calibration/](calibration/calibration.json). [checksums.json](checksums.json) identifies every published evidence file.

Supplementary resource sampling began partway through the study: [coverage and limitations](resource-coverage.json) · [recorded samples](resource-observations.jsonl.gz). These are not complete-study peak measurements or behavior results.


Rebuild this publication from the primary data:

```bash
python publish_lab_study.py --receipt /path/to/study-receipt.json --data-dir /path/to/lab-data --output studies/qwen38-27b --dashboard docs/results-27b.html --study-reasoning-notes studies/qwen38-27b/reasoning-notes.json
```

Use `--allow-partial` only when intentionally publishing incomplete or inconsistent evidence; such exports are prominently labeled partial. The original frozen protocol is never overwritten.
