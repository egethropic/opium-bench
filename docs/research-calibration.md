# Research calibration and independent scoring

The **Calibration** workshop has two explicitly different procedures. The pilot preserves the calibration used in the published 4B and 27B studies. Research v2 is opt-in; it does not rewrite those bundles or reinterpret their evidence.

## Extract candidate directions

Select **Research v2**, a compatible loaded model, zero-based layers, pooling and readout methods. Leave the custom corpus empty to use `corpora/research-v2.json`: 960 examples in 48 scenario families, with correlated prompt wrappers kept in their own split. The [corpus specification](../corpora/README.md) documents authorship, hashes, slices and split counts.

Intervention directions come from training contrasts. Independent readouts fit probe families. Selection families choose the layer/pooling/readout settings, and heldout families supply final discrimination measurements. Mean and ridge readouts, shuffled-label controls, specificity slices and family-level bootstrap are included. Masked mean and declared-span pooling are available; truncated examples are rejected. Span pooling requires valid offsets in the corpus.

The extraction progress reports forward-pass workload. A successful extraction is still **unvalidated as an intervention**. Generated final-position readouts fitted on mean/span pooling are an explicitly unvalidated transfer.

## Validate measurable interventions

Select the extraction bundle under **Validate an intervention**. Set dose magnitudes, pairs per concept, continuation length, next-token KL bound and relative-edit bound before running. The job creates a new bundle, preserving the input bundle.

Selection-prefix tests lock a dose before the heldout tests. Conditions include positive/negative pain and joy, independent attenuation, combined edits, sham and random edits matched to actual perturbation norm. Attenuation removes an origin projection, not a displacement from a neutral mean. Zero/random identity and tolerance outcomes are recorded. Numerical non-detection remains a useful negative result.

A `validated_for_enumerated_tests` status means the recorded numerical/probe criteria passed for those prefixes and settings. It does not establish semantic efficacy, general task benefit, sensation or safety. Increasing an edited direction's own projection is a mechanical consequence of the intervention. Inspect independent/downstream measures and behavior separately.

Cancellation preserves partial diagnostic progress and does not create successful completion metadata. Avoid choosing a setting after looking at heldout results and then reporting it as prespecified.

## Score output independently

1. Download **Blinded scoring sheet** from the validation bundle. Keep its collapsed **Observer-only condition key** away from raters until scoring is locked.
2. Edit only `ratings` for every sample. Follow the embedded rubric. Use `null` where a field cannot be scored, and explain ambiguity in `notes`. Preserve all IDs, prompts, continuations and rubric fields.
3. In **Import independent ratings**, choose the bundle, completed sheet and optional rater label. No GPU/model is required.
4. Download the saved ratings record from the persistent list. It includes the calibration hash, submitted sheet, per-field condition counts, missing denominators and scorer label. The source calibration is unchanged.

The scorer rejects changed text, duplicate/missing samples, a changed rubric or a tampered condition key. It records observable language, coherence and instruction following separately. Objective task grades and numerical perturbation are separate outcomes. Prompt content can itself reveal a condition; the export cannot guarantee blinding when that occurs.

API equivalents are `calibrate` with `schema_version: 2, preset: "research"`, `validate_calibration` with a managed `calibration_id`, and `score_calibration` with a completed `sheet`. Download artifacts from `/api/calibrations/<id>/scoring-sheet.json`, `diagnostics.json` and `scoring-key.json`; ratings from `/api/ratings/<id>`. All writes go through the local authenticated command endpoint.
