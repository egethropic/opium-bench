# Qwen3-4B core pain-only follow-up

Status: **complete** — 24/24 recorded episodes.

[Open the results dashboard](../../docs/results-core-pain-4b.html) · [Computed results](results.json) · [Execution receipt](receipt.json) · [Frozen protocol](protocol.json)

Across the recorded episodes: **72/72 assigned tasks correct**, **12/210 voluntary aux choices**, and **23435 generated tokens**. These totals mix distinct experimental conditions; use the dashboard’s matched comparisons to interpret effects.

User-requested pain-only auxiliary arm added after reviewing the initial results; fresh active/sham controls rerun alongside pain. This is a prospective follow-up, not part of the original frozen46.

The corpus and probes measure text-associated activation directions, not subjective emotion. The two planned seeds per condition are descriptive; tokens are not independent replicates. See the dashboard for zero-exposure pairs, failures, task denominators, and interpretive limits.

The calculator-assisted order tasks are easy, and the 20-action/3-order or 30-action/6-order budgets allow patterned aux use while still completing every task. A perfect task score does **not** rule out a preference that would become costly under tighter budgets. Harder tasks, binding budgets, dose sweeps, and longer learning periods remain future experiments.

Each run directory preserves its manifest, summary, conversation, deterministic compressed raw event trace, and standalone report. Calibration vectors and their split-validation manifest are in [calibration/](calibration/calibration.json). [checksums.json](checksums.json) identifies every published evidence file.

Rebuild this publication from the primary data:

```bash
python publish_lab_study.py --receipt /path/to/study-receipt.json --data-dir /path/to/lab-data --output studies/core-pain-4b --dashboard docs/results-core-pain-4b.html
```

Use `--allow-partial` only when intentionally publishing incomplete or inconsistent evidence; such exports are prominently labeled partial. The original frozen protocol is never overwritten.
