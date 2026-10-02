# Opium Bench release validation

Validation ledger updated 2026-10-02. All checks are manually invoked locally. Historical GPU findings and the new research-feature checks are distinct; new GPU acceptance and release publication remain pending.

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
improve its score. These historical checks precede the research-feature implementation below. See [release scope](roadmap.md) for the pending release gates.


## Research-feature implementation checks (October 2, 2026)

These checks exercise the new versioned interfaces and accounting. Synthetic outputs test the software; they do not demonstrate that an activation edit produces the intended behavioral effect.

| Check | Recorded result |
|---|---|
| Full local Python suite | 622 tests passed in 128.432 seconds (`work/feature-release-suite.log`); this run preceded the final diagnostic/report audit additions |
| Diagnostic/runtime audit regression | 78 tests passed in 44.710 seconds (`work/diagnostic-audit-final.log`), including legacy runtime, v2 phase hooks, rounded norm controls, independent criterion evidence and partial diagnostic denominators |
| New event-derived behavioral report | 9 CPU tests passed in 0.103 seconds (`work/behavioral-analysis-tests.log`); includes unequal costs, task position, censored transition latency, branch offsets, missing telemetry and a real Worker with scripted outputs |
| Integration checks | 59 focused checks passed in 11.4 seconds (`work/feature-final-targeted.log`); final all-source rerun still belongs to release acceptance |
| Browser workflow fixture | 19 accepted command actions; all tabs at 320/390/768 pixels plus desktop; no console errors (`work/feature-browser-final.log`). Includes delayed initial-state readiness, calibration/ratings, protocol preview, paired analysis, controls, pause/resume and branches |
| Designer browser check | 25 calls to the real Python recipe resolver; 320/390/768/1280-pixel layouts, exact 63-bit JSON round trips, import/export and asynchronous preview-race checks; no reported errors |
| Real local server browser check | All seven tabs at 320/390/768/1440 pixels; actual designer validation and protocol preview; no reported errors (`work/feature-real-browser.log`). This made no model-generation request |
| Compatibility contract | 808 protected files, three historical protocol expansions, ten recipe defaults and 24 visible payloads all match (`work/feature-compatibility-final.log`) |
| New GPU acceptance | Pending. Loading the 4B model successfully is preparation, not an acceptance experiment |
| New acceptance publication | Pending. No historical study is relabeled as evidence for the new features |

`work/` logs identify local development receipts, not newly published scientific artifacts. The final release will preserve its acceptance evidence separately. See [the implementation checklist](feature-completion-plan.md) for pending gates and [release scope](roadmap.md) for supported limitations.

The new CPU coverage includes strict recipe/tool/preset schemas; finite-budget admission; token/decision clocks and prefill accounting; family-separated calibration and heldout isolation; actual rounded edit-norm matching; complete-history checkpoints and pure-yoke restore; isolated diagnostics without answer feedback; portable import/export; interrupted/failed attempt retention; managed storage exhaustion and owned-process cancellation. No test fills a real disk or downloads model weights.

Independent criterion verification checks exact operator-supplied paired-score bytes, model/calibration/endpoint identities and a conservative family-level bound. It does not authenticate those observations, prove preregistration timing, or certify subjective states. The [criterion format](criterion-evidence.md) records these assumptions and the unverified path. Completed, partial, failed and missing diagnostics stay separate. The chance null is explicitly uniform guessing over all displayed options, including abstain; conditional non-abstaining accuracy has its own denominator.

The bounded GPU pass must still demonstrate compatibility of the implemented paths on the documented hardware. It must retain negative effects, unavailable norm matches, stops and failures honestly. No RTX 5090 validation or minimum-memory claim follows from these checks.

To repeat the code checks, use the configured runtime environment's Python (the unit suite itself runs on CPU):

```bash
python -m unittest discover -s tests -v
node tests/ui_smoke.cjs
node tests/test_designer.cjs
python check_compatibility.py
```

The browser check needs a separately installed Playwright module and browser;
`PLAYWRIGHT_MODULE` and `BROWSER_CHANNEL` select an existing installation. The
application itself has no Node, browser automation, or frontend build dependency.

Scientific limits and observed results are documented on the
[findings page](results.html). Passing software checks does not validate the
interventions as emotion measurements.
