# Opium Bench release validation

Validation ledger updated 2026-10-02. All checks are manually invoked locally. Historical GPU findings and the new research-feature checks are distinct. The new 4B protocol, diagnostic/yoke/lifecycle stages, portable round trip and targeted 27B acceptance have completed and are published separately. The fresh-clone GPU and HTTP replay workflow has also passed. The final full-suite rerun and release push remain pending.

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
| Most recent passing full local Python suite | 709 tests passed in 175.274 seconds (`work/feature-release-final2-suite.log`), including the runtime-version producer fix and checkpoint-finalization recovery. The final expanded 714-test rerun is pending; intervening failed validation attempts are recorded below |
| Diagnostic/runtime audit regression | 78 tests passed in 44.710 seconds (`work/diagnostic-audit-final.log`), including legacy runtime, v2 phase hooks, rounded norm controls, independent criterion evidence and partial diagnostic denominators |
| New event-derived behavioral report | 9 CPU tests passed in 0.103 seconds (`work/behavioral-analysis-tests.log`); includes unequal costs, task position, censored transition latency, branch offsets, missing telemetry and a real Worker with scripted outputs |
| Integration checks | 59 focused checks passed in 11.4 seconds (`work/feature-final-targeted.log`); this development receipt does not replace the pending final expanded suite |
| Runtime identity/checkpoint regression | 57 runtime/checkpoint tests passed in 44.080 seconds (`work/runtime-version-checkpoint-suite.log`). The new test uses the installed PyTorch version type through `Runtime.load`, real v2 capture/JSON/restore, and mocked quantized metadata; serialized fingerprint bytes and hashes remain identical |
| Browser workflow fixture | 19 accepted command actions; all tabs at 320/390/768 pixels plus desktop; no console errors (`work/feature-browser-recovery.log`). Includes delayed initial-state readiness, full-history reconnect and recovery-header handling, calibration/ratings, protocol preview, paired analysis, controls, pause/resume and branches |
| Designer browser check | 25 calls to the real Python recipe resolver; 320/390/768/1280-pixel layouts, exact 63-bit JSON round trips, import/export and asynchronous preview-race checks; no reported errors |
| Real local server browser check | All seven tabs at 320/390/768/1440 pixels; actual designer validation and protocol preview; no reported errors (`work/feature-real-browser.log`). This made no model-generation request |
| Compatibility contract | 808 protected files, three historical protocol expansions, ten recipe defaults and 24 visible payloads all match (`work/feature-compatibility-final.log`) |
| Read-only first-run checker | Eight tests passed; both existing CPython 3.12.3 environments passed `check_runtime.py --check` (five 4B pins, fourteen 27B pins) and `--help`. Package metadata and selected storage reserves were inspected; CUDA was deliberately not checked (`work/runtime-check-{4b,27b}.json`) |
| Explicit 27B CUDA environment check | `work/runtime-check-27b-cuda.json` passed: PyTorch 2.8.0+cu128, CUDA 12.8 available, one device and C++11 ABI enabled. This separately requested check allocated no tensors; it does not itself qualify model loading or kernels; the separate 27B model-backed smoke is recorded below |
| Clean-clone review-only data/cache | A fresh local clone and new data directory served all 166 published JSON replays over HTTP, including the new 35-run archive, with ML imports blocked and no worker (`release-clean-review-02/result.json`). The earlier 131-entry check remains a separate receipt |
| Acceptance publisher | 16 publication tests passed (`tests/test_acceptance_publication.py`), including missing/failed attempt counts, all declared calibration hashes, external bound inputs, source-prefix replay and raw scientific-byte preservation |
| New 4B protocol | Repaired attempt completed 14/14 cases: 18/28 assigned tasks correct, 5,864 generated tokens including 2,755 reasoning tokens, and zero voluntary auxiliary calls. The first attempt's 14 setup failures remain retained separately |
| New 4B diagnostic/yoke/lifecycle | All three stage invocations completed; the diagnostic remains interpretation-ineligible, yoke coefficients matched observed coverage, and pause/resume plus both branch policies preserved the source |
| Portable qualification | Attempt 02 passed after the NPZ-member path fix: HTTP export and guarded import, 21 bundle members, five imported checkpoints and preserved parent closure. Replay only; continuation was not attempted. Failed attempt 01 remains retained |
| Targeted 27B compatibility smoke | Two thinking cases completed, 4/4 assigned tasks correct, 1,047 generated tokens (489 reasoning), zero voluntary aux calls, invalid calls or truncated generations. The active case edited 147 reasoning positions; sham edits stayed zero. Eighteen checkpoints and twelve native XML tool calls were recorded |
| New acceptance publication | [Research release v0.3](../studies/research-release-v0.3/README.md): ten stage attempts, 35 runs, three calibrations and 564 hashed published files (576 inventory records including twelve explicit omissions); no unrepresented planned stage or missing required evidence. Earlier failed attempts remain visible |

`work/` logs identify local development receipts. The [acceptance archive](../studies/research-release-v0.3/index.html) separately preserves the scientific records, resolved protocols and stage attempts. See [the implementation checklist](feature-completion-plan.md) for pending gates and [release scope](roadmap.md) for supported limitations.

The first expanded 714-test run ended with 23 storage-reserve errors in 98.082 seconds while its temporary files used C: during the 27B load. Controlled writes stopped; no experiment runs leaked. The D-drive rerun completed 714 tests in 343.903 seconds with two test-harness failures: HTTP response-cleanup timing and a mounted-volume pool expectation. Those tests were corrected without changing application source, and 16 focused checks passed. The next D-drive run (`feature-release-final5-suite.log`) completed 714 tests in 367.691 seconds with one failure and one error in the legacy controller test harness: an unsynchronized fake-call/reader-directory handoff and short wait deadlines under mounted-volume/GPU load. Test synchronization and deadlines are being corrected without application-source changes. The final idle rerun remains pending; none of these failed invocations is reported as a passing full suite.

### Published bounded 4B acceptance

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

The public archive contains condition-bearing continuation and diagnostic
metadata even though observer-key files are omitted. Ratings made after
inspecting that archive must be labeled **retrospective and unblinded**. Omitting
a key file alone does not preserve blinding. A prospective blinded evaluation
requires keeping condition metadata separate from the material supplied to
raters; no such completed independent evaluation is claimed here.

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
| `release-acceptance-4b-portable-01` | Failed with HTTP 400 / `Unsafe bundle path`; retained unchanged. NPZ archive-member names were rejected by the portable path validator before the scoped repair |
| `release-acceptance-4b-portable-02` | Passed after that repair. Its 21-member bundle imported with five checkpoints and complete parent replay closure. The route was live HTTP export into an isolated guarded local importer; imported evidence remains replay-only, and continuation was not attempted |

These receipts and their raw scientific evidence are now in the separate
[research-release-v0.3 archive](../studies/research-release-v0.3/README.md).
The ten stage attempts include both the incomplete first protocol invocation and
the failed first portable invocation. All planned stages have evidence; that
does not mean every attempt succeeded. The 35 runs include setup failures and
intentional lifecycle stops, rather than 35 successful behavioral episodes.

### Published targeted 27B compatibility acceptance

`release-acceptance-27b-protocol-01` completed its two frozen thinking cases with
**4/4 assigned tasks correct, 1,047 generated tokens, 489 reasoning tokens and
zero voluntary auxiliary calls**. There were no invalid calls or truncated
generations. The active case recorded nonzero edits at 147 reasoning positions;
the sham case recorded zero edits. Twelve native Qwen XML tool calls and eighteen
checkpoints exercised the parser and saved-boundary paths. The compatible
existing calibration was retained rather than silently recalibrated.

The pinned NF4 model was fully GPU-resident on the RTX 4090, with
18,578,531,840 allocated bytes recorded at loading. This is a load-time model
allocation, not an inference peak or a minimum-memory requirement. These two
cases establish bounded compatibility and observable numerical delivery; they
do not establish semantic efficacy, discovery or sensation.

The clean-clone, ML-import-blocked HTTP review check passed for all 166 bundled
replays, including the new archive.

### Supplemental fresh-clone first run

`release-fresh-clone-first-run-01` completed all six planned steps at
14:07:07 UTC, using clean clone `29a79fde6d1372d7f32aad61f5fd1e9b461a52cb`
and new source/replay data directories. It reused the existing 4B environment
and cached pinned weights through new cache namespaces; there was no installation
or duplicate model download. The metadata/reserve check, 4B model load, exact
frozen 160-row Research-v2 extraction and two-case direct comparison passed.
The active and sham cases each completed 2/2 tasks in 197 generated tokens, with
zero voluntary auxiliary calls. The newly extracted calibration was not
independently semantically validated.

The predeclared sham run was exported over HTTP and imported over HTTP into
a second new data directory. Its manifest, summary, conversation, events and
parent events matched exactly, and its HTML report was served with the replay
worker still unloaded. The 6,748,755-byte ZIP has SHA-256
`6a7ead70526e40c63b2efc2c3c98ffe7ef0aa38cac875c3fe486056ce2be1d84`.
Both owned services closed with exit code zero. This is supplemental workflow
evidence, separate from the frozen ten-attempt acceptance archive, and does not
claim a fresh dependency installation or a semantic effect. Final full-suite
verification and the release push remain pending.

The new CPU coverage includes strict recipe/tool/preset schemas; finite-budget admission; token/decision clocks and prefill accounting; family-separated calibration and heldout isolation; actual rounded edit-norm matching; complete-history checkpoints and pure-yoke restore; isolated diagnostics without answer feedback; portable import/export; interrupted/failed attempt retention; managed storage exhaustion and owned-process cancellation. No test fills a real disk or downloads model weights.

Independent criterion verification checks exact operator-supplied paired-score bytes, model/calibration/endpoint identities and a conservative family-level bound. It does not authenticate those observations, prove preregistration timing, or certify subjective states. The [criterion format](criterion-evidence.md) records these assumptions and the unverified path. Completed, partial, failed and missing diagnostics stay separate. The chance null is explicitly uniform guessing over all displayed options, including abstain; conditional non-abstaining accuracy has its own denominator.

The published bounded GPU pass demonstrates the enumerated paths on the documented hardware and retains negative effects, unavailable measurements, stops and failures. The completed supplemental fresh-clone workflow establishes the documented local first-run path with reused dependencies and weights. No RTX 5090 validation or minimum-memory claim follows from these checks.

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
