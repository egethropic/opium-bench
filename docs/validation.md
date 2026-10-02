# Opium Bench release validation

Validated locally on 2026-10-02 (UTC). No hosted automation was used.

| Check | Result |
|---|---|
| Python test suite | 229 tests passed in 59.331 seconds |
| Browser fixture suite | 19 checks passed against an isolated fixture server |
| Responsive app | Six tabs at 320, 390, 768, and 1440 pixels; no horizontal page overflow |
| Results publication | All 154 local evidence links valid; desktop, mobile, HTTP, and local-file viewing checked |
| Fresh review-only server | Standard-library Python with ML imports blocked; 46 published and 7 historical runs available |
| Historical evidence | All 34 report artifact links resolved; local/bundled run IDs deduplicated |
| Binary evidence downloads | Gzip/NPZ MIME types and download filenames checked through the running server |
| Study integrity | 239 published file checksums verified; all 46 compressed traces match the source bytes |
| GPU pilot | 46/46 episodes completed; 186/186 assigned tasks correct; zero integrity warnings |

The real-browser observer made no model-control requests during the recorded
study or screenshots. Browser fixture tests use a separate synthetic service.
Unit tests do not download checkpoints or require a loaded GPU model.

Reference inference: Qwen3-4B BF16 on RTX 4090, Python 3.12, PyTorch 2.8.0+cu128,
Transformers 4.57.6. The model revision, tokenizer/template fingerprints,
calibration, per-generation metadata, and source hashes are preserved in the
[study records](../studies/initial/README.md).

Subsequent local validation for the pain-inclusive findings and 27B runtime:

| Check | Result |
|---|---|
| Updated Python suite | 255 tests passed in 105.415 seconds |
| Source archives | All 239 original and 129 pain-inclusive core checksums still match |
| Comprehensive 4B view | 54 primary episodes, 70 recorded episodes, 16 exact verification repeats; source archives unchanged |
| Comprehensive page | Local links and responsive layouts checked; pain appears in the same main figure and table |
| 27B kernel comparison | 38 small BF16 prefill, decoding, convolution and recurrent-state checks passed against Torch reference functions |
| 27B loading | Pinned Qwen3.8-27B NF4 conversion fully GPU-resident on RTX 4090; 17.30 GiB allocated before generation |
| 27B calibration | New corpus-split calibration completed at block 22, with downstream readout at block 63 |
| 27B engineering smoke | Direct and thinking runs each completed 1/1 task with no invalid calls; excluded from the formal study |

The [27B engineering receipt](../studies/qwen38-27b/engineering/engineering-smoke.json) identifies both smoke traces. The same archive preserves
[checkpoint hashes](../studies/qwen38-27b/engineering/checkpoint-checksums.json),
[runtime identity](../studies/qwen38-27b/engineering/loaded-model.json), and
[kernel numerical checks](../studies/qwen38-27b/engineering/kernel-parity.json). The separate frozen behavioral batch completed all 54 episodes: 209/210 tasks correct, no voluntary auxiliary calls across 628 decisions, and one retained incorrect answer in the two-button reversal condition. See the [27B evidence archive](../studies/qwen38-27b/README.md).
The 5090, native Windows GPU execution, and a minimum GPU-memory requirement
have not been established. Linux/WSL is the tested inference environment.

Final 27B publication checks:

| Check | Result |
|---|---|
| Final publication tests | 41 focused analysis/composition tests passed after the resource and incorrect-answer callouts were added |
| Behavioral study | 54/54 episodes complete; 209/210 tasks correct; 0 voluntary aux calls across 628 decisions |
| Numerical exposure | 12,481 generated token steps with measured nonzero edits; 0 integrity warnings, invalid decisions or truncated generations |
| Published evidence | All 297 archive checksums verified; all 54 copied raw runs match their originals |
| Earlier evidence | All 239 original, 129 pain-inclusive core and 14 frozen engineering files unchanged |
| Combined publication | 484 combined-page and 306 archive-page links checked; both model sections checked at 320, 390, 768 and 1440 pixels, without broken images/anchors, duplicate IDs or page overflow |
| Fresh review-only installation | 131 unique entries: 7 historical, 46 original, 24 pain-inclusive core and 54 new runs; all 54 new JSON replays and reports served successfully |
| Review dependencies | No PyTorch, Transformers or NumPy imports and no model worker in the fresh review-only check |
| Supplementary resource record | 147 samples beginning during episode 5; 21,476–21,816 MiB observed device-wide GPU usage, explicitly not a complete-study or model-only peak |

The single incorrect answer is retained and analyzed in the
[matched quality case](qwen38-27b-quality-case.md). Software validation does not
turn a behavioral error into an integrity failure, and no run was replaced to
improve its score. See [release scope](roadmap.md) for planned features that
remain beyond this core release.

To repeat the code checks, use the GPU environment's Python:

```bash
python -m unittest discover -s tests -v
node tests/ui_smoke.cjs
```

The browser check needs a separately installed Playwright module and browser;
`PLAYWRIGHT_MODULE` and `BROWSER_CHANNEL` select an existing installation. The
application itself has no Node, browser automation, or frontend build dependency.

Scientific limits and observed results are documented on the
[findings page](results.html). Passing software checks does not validate the
interventions as emotion measurements.
