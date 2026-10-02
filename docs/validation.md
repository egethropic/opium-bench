# Opium Bench release validation

Validation ledger updated 2026-10-02. All checks are manually invoked locally. Historical GPU findings and the new research-feature checks are distinct. The new 4B protocol and diagnostic/yoke/lifecycle stages have completed; portable qualification, targeted 27B acceptance and release publication remain pending.

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
| Latest full local Python suite | 709 tests passed in 175.274 seconds (`work/feature-release-final2-suite.log`), including the runtime-version producer fix and checkpoint-finalization recovery. The earlier 622-test/128.432-second run remains a separate development receipt |
| Diagnostic/runtime audit regression | 78 tests passed in 44.710 seconds (`work/diagnostic-audit-final.log`), including legacy runtime, v2 phase hooks, rounded norm controls, independent criterion evidence and partial diagnostic denominators |
| New event-derived behavioral report | 9 CPU tests passed in 0.103 seconds (`work/behavioral-analysis-tests.log`); includes unequal costs, task position, censored transition latency, branch offsets, missing telemetry and a real Worker with scripted outputs |
| Integration checks | 59 focused checks passed in 11.4 seconds (`work/feature-final-targeted.log`); the latest 709-test suite also covers the subsequent source changes |
| Runtime identity/checkpoint regression | 57 runtime/checkpoint tests passed in 44.080 seconds (`work/runtime-version-checkpoint-suite.log`). The new test uses the installed PyTorch version type through `Runtime.load`, real v2 capture/JSON/restore, and mocked quantized metadata; serialized fingerprint bytes and hashes remain identical |
| Browser workflow fixture | 19 accepted command actions; all tabs at 320/390/768 pixels plus desktop; no console errors (`work/feature-browser-recovery.log`). Includes delayed initial-state readiness, full-history reconnect and recovery-header handling, calibration/ratings, protocol preview, paired analysis, controls, pause/resume and branches |
| Designer browser check | 25 calls to the real Python recipe resolver; 320/390/768/1280-pixel layouts, exact 63-bit JSON round trips, import/export and asynchronous preview-race checks; no reported errors |
| Real local server browser check | All seven tabs at 320/390/768/1440 pixels; actual designer validation and protocol preview; no reported errors (`work/feature-real-browser.log`). This made no model-generation request |
| Compatibility contract | 808 protected files, three historical protocol expansions, ten recipe defaults and 24 visible payloads all match (`work/feature-compatibility-final.log`) |
| Read-only first-run checker | Eight tests passed; both existing CPython 3.12.3 environments passed `check_runtime.py --check` (five 4B pins, fourteen 27B pins) and `--help`. Package metadata and selected storage reserves were inspected; CUDA was deliberately not checked (`work/runtime-check-{4b,27b}.json`) |
| Explicit 27B CUDA environment check | `work/runtime-check-27b-cuda.json` passed: PyTorch 2.8.0+cu128, CUDA 12.8 available, one device and C++11 ABI enabled. This separately requested check allocated no tensors; it does not qualify model loading, kernels or the pending 27B experiment |
| Clean review-only data/cache | A separate standard-library server with PyTorch, NumPy, Transformers and bitsandbytes imports blocked served all 131 previously published JSON replays, without a worker (`work/feature-clean-review.log`). This does not yet cover the new acceptance archive |
| Acceptance publisher | 16 publication tests passed (`tests/test_acceptance_publication.py`), including missing/failed attempt counts, all declared calibration hashes, external bound inputs, source-prefix replay and raw scientific-byte preservation |
| New 4B protocol | Repaired attempt completed 14/14 cases: 18/28 assigned tasks correct, 5,864 generated tokens including 2,755 reasoning tokens, and zero voluntary auxiliary calls. The first attempt's 14 setup failures remain retained separately |
| New 4B diagnostic/yoke/lifecycle | All three stage invocations completed; the diagnostic remains interpretation-ineligible, yoke coefficients matched observed coverage, and pause/resume plus both branch policies preserved the source |
| Portable qualification and 27B smoke | Portable attempt 01 failed with HTTP 400 / `Unsafe bundle path` and remains retained while investigated. Targeted 27B smoke is pending |
| New acceptance publication | Pending. No historical study is relabeled as evidence for the new features |

`work/` logs identify local development receipts, not newly published scientific artifacts. The final release will preserve its acceptance evidence separately. See [the implementation checklist](feature-completion-plan.md) for pending gates and [release scope](roadmap.md) for supported limitations.

### Bounded 4B acceptance: interim evidence, not a release result

The `release-acceptance-4b-extract-01` and
`release-acceptance-4b-validation-01` local receipts completed. Calibration
`cal-20261002T123642Z-46614ba5` uses block 12, mean pooling and a mean readout,
with downstream measurement at block 25. Its snapshot and managed metadata are
byte-identical; the vector file and all six declared evidence hashes verified.
Heldout AUC was 1.0 for both concepts on only 20 rows and two scenario families
per concept. The 20-resample bootstrap's degenerate AUC interval does not
establish broad generalization. Pain/joy score correlations were 0.8493 at the
probe and 0.8846 downstream; cross-concept AUCs of 0.98 and 0.93 also limit claims
of specificity. Transfer from mean-pooled authored text to generated reasoning
remains unvalidated.

The numerical validation contains 36 fixed-prefix records. All four sham records
had exactly zero KL/edit, and all four measured random controls met the declared
norm tolerance. **No tested nonzero dose met the frozen selection bounds.**
Combined dose 0.25 had selection mean relative edit 0.35761, exceeding 0.30;
its mean next-token KL of 0.15412 was below the 0.50 bound. The selected dose is
therefore zero. The four generated continuations are all sham, all ended at the
16-token length limit, and all remain unscored. They provide no active-versus-sham
semantic comparison. The frozen 0.25 protocol cases remain engineering stress
checks, not a validated operating-dose study.

The first 14-case protocol invocation, `release-acceptance-4b-protocol-01`,
retains **14 failed setup cases with zero generated tokens and zero charged
actions**. PyTorch exposed its version as a `TorchVersion` string subclass;
the strict checkpoint plain-data validator rejected that metadata before
generation. The outer invocation has no final receipt and is preserved as
`no_final_receipt`, alongside the failed research receipt and supervisor
recovery evidence. These are software failures, not model choices or task-effect
observations.

The repair normalizes version metadata at its producer to built-in strings,
preserving serialized fingerprints, and hardens checkpoint-error finalization.
The separate `release-acceptance-4b-protocol-02` invocation completed all 14
prespecified cases at 13:28:50 UTC. It recorded **18/28 assigned tasks correct,
5,864 generated tokens, 2,755 reasoning tokens and zero voluntary auxiliary
calls**. A completed case can end at its budget; completion does not mean every
assigned task was answered correctly. The first failed invocation remains
separate. This small engineering matrix does not establish task benefit,
discovery, addiction or emotional experience.

The subsequent stage receipts preserve the following evidence:

| Local stage receipt | Recorded outcome and limits |
|---|---|
| `release-acceptance-4b-diagnostic-01` | One completed context, 17 tokens, one valid but incorrect prediction on the sham/sham control and one false positive. The criterion is unvalidated and `interpretation_eligible=false`. The source checkpoint was unchanged and no diagnostic answer was fed back to its parent |
| `release-acceptance-4b-yoke-01` | One source pulse delivered; 197 observed output positions matched, no uncovered source positions, and zero coefficient/index errors. The target completed 2/2 tasks with zero voluntary calls. Measured edit-norm sums differed (source 1,156.8364; target 1,157.2239), so matched coefficients do not establish identical numerical effects or internal states |
| `release-acceptance-4b-lifecycle-01` | Pause/resume and `continue_state`/`fresh_budget` branches completed; source evidence unchanged. The source and branch chat runs were intentionally stopped. Branch totals include inherited counters and are not independent task episodes |
| `release-acceptance-4b-portable-01` | Failed with HTTP 400; the reported service error is `Unsafe bundle path`. The failed receipt is retained and the portable round trip remains unqualified while the cause is investigated |

These local receipts live under the selected data drive's
`release-acceptance-*` directories and will be included in the separate immutable
acceptance publication. Portable recovery, the new 27B compatibility smoke,
final publication, fresh-clone replay of the new archive, and final release
verification remain pending. No final gate is inferred from a stage's completed
status alone.

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
