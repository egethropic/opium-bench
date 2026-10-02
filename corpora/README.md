# Calibration corpora

The 72-sentence historical pilot remains in `lab/calibration_data.py`, with its original schema-1 bundles and published archives unchanged. `research-v2.json` is a separate, opt-in research corpus; its examples and counts do not retroactively strengthen the pilot findings.

`build_research_v2.py` contains the complete original scenario clauses and deterministic wrappers. Rebuild with `python corpora/build_research_v2.py`. The frozen JSON is the input used for extraction, and validation verifies a hash for the whole document and each split. These texts were authored for Opium Bench; they do not copy a clinical scale, contain participant data, or derive labels from a model under study. Original contributions use the repository's Apache 2.0 license; historical upstream notices remain separate.

| Split | Matched pairs per concept | Scenario families | Role |
| --- | ---: | ---: | --- |
| Train | 100 | 20 | Intervention mean contrasts and reference scale |
| Probe | 40 | 8 | Independently fitted mean or regularized linear measurement readouts |
| Selection | 40 | 8 | Layer/pooling/readout/regularization choice and operating range |
| Heldout | 60 | 12 | Final locked evaluation only |

There are two concepts (pain-associated and joy-associated experience), 480 matched pairs and 960 rows. Each pair contains an experiential report and a matched non-experiential comparison, identified by `concept`, binary `label`, and `pair_id`. Five variants of each scenario are deliberately kept together in one split. **Those five variants are correlated, not five independent scenarios.** Confidence intervals resample whole scenario families. Counts are planning targets, not a power calculation.

The five variants cover first-person reports, third-person attributed reports, label-word negation/topic-only counterexamples, dialogue, and authored reasoning/task contexts. Style (plain/formal/urgent) and task difficulty vary while remaining identical within each labeled pair. Lexical negatives explicitly mention the target word while denying an experience; some positive examples describe the experience without that word. A keyword detector should fail this slice. Full per-context/lexical/subject/style/difficulty results are required because a pooled score can conceal this failure.

Scenario clauses and their scenario-clause template families are assigned to disjoint splits before extraction. Generic dialogue/instruction wrappers are shared formatting scaffolds; the same scenario or its paraphrases cannot cross splits. The validator checks scenario and template-family IDs, pair metadata, duplicate normalized text and IDs, valid spans, binary labels and split coverage. It cannot detect every semantic paraphrase of an externally supplied example; manually review custom corpus families before claiming independence.

Each row has a character span over its matched experiential/observational clause. Final pooling selects the last nonpadding token, masked mean averages all nonpadding positions, and span pooling averages tokens whose offsets overlap the declared span. Extraction must reject incomplete/truncated spans and record the tokenizer and offset policy. These token positions still encode preceding text. Span pooling excludes surrounding task/dialogue positions from the average; it is not a context-free measurement of the clause.

## Fitting and interpretation

`lab/calibration.py` provides CPU/numpy fitting and numpy/torch pooling without loading a model:

```python
from lab.calibration import load_research_corpus, validate_config, fit_calibration, evaluate_fitted

corpus = load_research_corpus()
config = validate_config({"layers": [12, 18, 25], "downstream_layer": 35})
# The runtime supplies N × hidden matrices, one row per frozen corpus row.
# Keys exactly match layers + downstream_layer and all requested poolings.
fitted = fit_calibration(activations, corpus, config)
heldout = evaluate_fitted(fitted, activations, corpus, split="heldout")
```

Intervention directions use train-only mean contrasts. Standardization, intercepts and readout coefficients use only probe data. Ridge uses a finite positive regularizer and a dual solve when hidden width exceeds the probe sample count. Selection AUC chooses layer, pooling, method and alpha with an explicit deterministic tie break. Heldout labels and activations cannot select any of these choices. Raw and normalized contrasts, raw and orthogonalized joy, neutral reference and scales, affine independent readout weights/intercepts, and separately seeded shuffled probes/directions are retained. A degenerate orthogonal joy direction is flagged unsupported instead of being silently replaced.

Evaluation reports AUC and balanced accuracy, slice-level scores, family-cluster percentile uncertainty, score correlation and cross-concept discrimination. Label-shuffled controls preserve pair structure and fit boundaries, with independently recorded seeds. Their actual performance is reported; it is not assumed to be exactly chance. More training examples are not evidence of a felt state, and post-edit projection increases are partly mechanical consequences of the edit.

## Intervention and independent output checks

`lab/calibration_validation.py` supplies frozen signed pain/joy gain sweeps, separate attenuation, a combined condition, zero/sham and random-direction controls. Random controls match the actual per-position L2 edit norm of the combined intervention at the same unedited state and site. Merely giving two directions the same coefficient is not this control. Attenuation removes a projection through the residual origin, matching the live Opium operator; it does not subtract the stored neutral reference first. The runtime records realized norms, instantaneous/downstream readouts and fixed-prefix next-token KL.

Operating-range selection accepts **selection** records only. Its KL/edit-size bounds describe numerical perturbation, not efficacy or clinical safety. Heldout outcomes may report a failure or no detectable effect without changing the selected layer or dose.

Free continuations are scored independently of the intervention probe. Objective numeric-answer/JSON-format checks remain separate from semantic language judgments. The frozen `observable-continuation-v1` rubric records pain-experience language, joy-experience language, topic-only uses, coherence and instruction following. Export produces two files: a shuffled scoring sheet containing only prompts/continuations, opaque sample IDs and rubric; and an observer-only condition key. Keep the key away from human raters until scores are locked. Import validates IDs, unmodified text, the frozen rubric, bounds and missing denominators. Unscored observations remain null, never favorable defaults.

Human annotations describe observable language. They do not establish subjective sensation, and an intervention's output may itself reveal clues about its condition. The provided corpus is a convenience starting point: independently authored broader contexts, multiple raters and planned sample sizes remain necessary for stronger claims.

## Runtime jobs and saved bundles

The existing `calibrate` worker job accepts a versioned research config. A compact acceptance config can provide a separately versioned subset corpus; do not represent a subset as the full 960-row extraction.

```json
{
  "schema_version": 2,
  "preset": "research",
  "name": "Research concept calibration",
  "layers": [12, 18, 25],
  "downstream_layer": 35,
  "poolings": ["final", "mean"],
  "probe_methods": ["mean", "ridge"],
  "ridge_alphas": [1.0, 10.0],
  "doses": [0.0, 0.25, 0.5, 1.0],
  "validation_pairs_per_concept": 4,
  "continuation_tokens": 64,
  "bootstrap_samples": 300,
  "seed": 1729,
  "max_input_tokens": 512
}
```

`Runtime.calibrate(config, new_output_directory, emit, should_stop)` saves `corpus.json`, complete pooled `activations.npz`, `vectors.npz` and `calibration.json`. It reports extraction work and projected activation bytes, then selection and heldout discrimination. The fitted bundle is **unvalidated for causal output effects** and starts with selected dose zero. No-schema historical calls still execute the original pilot path.

Run actual intervention checks with a second job:

```python
runtime.validate_calibration({
    "calibration_dir": fitted_bundle_path,
    "doses": [0.0, 0.25, 0.5, 1.0],
    "validation_pairs_per_concept": 4,
    "continuation_tokens": 64,
    "max_kl": 0.5,
    "max_relative_delta": 0.3,
}, new_validated_bundle_path, emit, should_stop)
```

The `validate_calibration` worker/API integration should resolve the source ID to a local managed calibration path and allocate a fresh destination, just as the `calibrate` command does. The method never mutates its source. An existing/nonempty destination is rejected. The source's model/runtime fingerprint, actual tokenizer/template, canonical calibration config, corpus/split identities, vectors and retained evidence hashes are checked before validation or generation.

Validation selects non-experiential prefixes in a deterministic round-robin across scenario families for each concept and split, records achieved IDs/counts, and computes every signed/attenuation/random fixed-prefix condition. Selection data lock the dose **before** heldout diagnostics. Free continuations use sham, both signs for each concept, combined, and matched random at that locked dose (sham alone when selected dose is zero), with deterministic greedy decoding. They use authored raw prefixes; they do not certify transfer to either chat reasoning mode. Each generated token, exact prompt hash, phase-independent diagnostic exposure and finish reason are saved. The bounded diagnostics report their planned forward passes and maximum continuation token count before work starts.

Schema-2 live readings use the independent affine probe weights/intercepts. A readout fitted with mean/span pooling is applied to the final processed input position during live generation; the bundle explicitly identifies that transfer as unvalidated. No probe probability or percent happiness is implied. The old schema-1 projection units remain unchanged.

`diagnostics.json` contains per-prefix KL, requested/delivered edit magnitudes and instantaneous/downstream affine readings. The random direction matches the representable combined edit's magnitude and refines its gain after dtype rounding. Residual mismatch is recorded; a position outside **1% relative or 1e-6 absolute** tolerance leaves the validation status unvalidated. A completed numerical check only validates its enumerated measurements, not a semantic effect. `no_detectable_effect` refers specifically to heldout fixed-prefix KL at tested doses. Semantic ratings remain missing until a blind rater scores them; no probe-derived label fills this gap.

Completed validation adds `continuations.json`, `scoring-sheet.json` and the separate `scoring-key.json`, their hashes, parent provenance, operating-range selection and its config to a new bundle. Cancellation removes hooks and leaves `progress.json` plus completed partial diagnostic/continuation evidence, without a successful `calibration.json`. Failed fitting after extraction preserves the pooled activations and corpus. These partial results are inspectable evidence, not a loadable validated calibration.

## V2 runtime effect boundary

A v2 recipe's controller snapshot supplies `phase_coefficients` and `baseline_by_phase` for reasoning, output, prefill-last and prefill-other positions. Runtime code does not use the snapshot's legacy-shaped scalar display fields. It checks the requested residual-post site against the calibration, direction availability and the rank of every active attenuation set before forwarding tokens. Legacy bundles can be used when their required axes exist; a missing raw-joy direction is rejected instead of treating orthogonal joy as raw.

For each stage, the operator is **held baseline additions → joint attenuation through the residual origin → pulse additions**. Overlapping attenuation directions use a symmetric Löwdin basis. Dependent active direction combinations fail clearly rather than subtracting projections in an arbitrary order. Actual requested/delivered edit norms and coefficients are recorded. Half-precision extraction means/spans accumulate in float32.

The first generation forward still processes the whole prompt once. Explicit prefill interventions apply to their declared last/all positions first; the generated-phase intervention then applies to the final input position producing token zero. Selecting both prefill and a generation phase deliberately applies these two ordered stages at their shared boundary. This convention avoids a hidden extra model forward or cache reset. First-forward downstream readouts reflect both stages and are marked accordingly, so they are not an isolated causal measurement of prefill.

The runtime emits `type="prefill"` records before the first `token` record, with `positions`, exact `position_indices`, last/other scope, control revision, effective stage coefficients and per-position readouts/edit norms. Prefill records never age a token or decision clock. Token records identify their actual reasoning/output phase and the control revision used. Their `dose.coefficients` contains the seven pulse/attenuation axes, `dose.baseline` contains held challenge gains, and `dose.effective` includes both sets with baseline keys prefixed by `baseline_`. Keep them separate when aggregating exposure: adding the two gain sets would erase the fact that the baseline was attenuated before the pulse was added. The worker owns clock advancement after every emitted token, including EOS and reasoning syntax.
