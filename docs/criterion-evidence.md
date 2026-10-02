# Independent evidence for discovery criteria

Button-function predictions need an observable endpoint whose response to the intervention has been checked independently. A direction's post-edit projection, calibration classification accuracy, and a model's account of sensation do not establish that endpoint. The [diagnostic workflow](discovery-diagnostics.md) still runs without a demonstrated effect, but its results remain `interpretation_eligible=false`.

Declaring `validation.status="validated"` is insufficient. Eligibility additionally requires the exact declared evidence bytes, the current model and calibration identity, a matching endpoint and scoring method, and a conservative lower confidence bound above a prespecified practical margin. This is an eligibility rule for interpreting a behavioral prediction test. It does not certify emotions, consciousness, or addiction.

## Evidence format

Save one UTF-8 JSON file with these exact fields. The following is a structural example, not valid experimental evidence; replace placeholders and supply all prespecified observations. One row is a paired active/sham observation, scored with the same independent rubric. Use the real scenario family identifiers, grouping related templates together.

```json
{
  "kind": "opium-bench/paired-criterion-evidence",
  "schema_version": 1,
  "model_fingerprint_sha256": "<source checkpoint identity.model.fingerprint_sha256>",
  "calibration_sha256": "<source checkpoint identity.calibration_sha256>",
  "endpoint_id": "json-compliance",
  "criterion_kind": "objective_behavior",
  "independent_of_intervention_probe": true,
  "score_bounds": [0, 1],
  "effect_direction": "increase",
  "minimum_effect": 0.1,
  "confidence": 0.95,
  "sample_unit": "scenario_family",
  "preregistration_sha256": "<SHA-256 of the prespecified analysis document>",
  "scorer_provenance": "Describe the operator-supplied scorer, source observations, and blinding procedure.",
  "records": [
    {"id": "pair-001", "family": "record-format", "active_score": 1, "sham_score": 0}
  ]
}
```

`criterion_kind` is `objective_behavior` or `blinded_human_behavior`. `effect_direction` is `increase` or `decrease`; it must be chosen before inspecting outcomes. The score range and nonnegative `minimum_effect` are in the endpoint's natural score units. The minimum effect must be smaller than the score range. Confidence is fixed at 0.95 in this format.

The model fingerprint must be complete. The calibration hash is the checkpoint's **calibration identity hash**, not merely a vectors-file checksum. The endpoint ID and criterion kind must match the diagnostic criterion. Set its `validation.evidence_sha256` to the SHA-256 of the exact file bytes, `validation.examples` to the paired row count, and `validation.context_families` to the complete distinct family set. Diagnostic question families must be disjoint from that set. Whitespace changes alter the file hash.

The verifier rejects unknown fields, duplicate JSON keys, repeated row IDs, nonfinite values, out-of-range scores, mismatched identities, and inconsistent counts or families. It accepts at most 4 MB and 10,000 paired rows. Omitted evidence is an explicit unverified path, useful for checking experiment mechanics.

## How the bound is calculated

For each row, calculate active minus sham, reversing the sign for a prespecified decrease. Average these paired differences within each scenario family; then average the family means with equal weight. Repeating closely related rows therefore does not add independent units.

Let `R` be the declared score range, `F` the number of independent scenario families, and `D` the mean signed family difference. The conservative two-sided 95% Hoeffding interval is:

```text
radius = 2 * R * sqrt(log(40) / (2 * F))
lower  = max(-R, D - radius)
upper  = min( R, D + radius)
```

Eligibility requires both declared `validated` status and `lower > minimum_effect`. Small pilots will commonly remain ineligible even when their observed effect is large. More rows within a family do not shrink this bound; genuinely independent families do. No detectable effect and unvalidated status remain ineligible regardless of the point estimate.

The bound assumes independent scenario families, bounded correctly scored outcomes, and prespecified direction, margin and analysis. It is a per-endpoint interval; it does not correct for trying many endpoints or selecting favorable runs. It does not establish representative sampling, blinding, genuine pairing, independence, or preregistration timing. Keep those design decisions and source observations reviewable. The scorer and observations remain **operator-supplied**. File hashes prove content identity, not authenticity or experimental quality.

## Running and reviewing

The Python API accepts the exact bytes or UTF-8 text as `criterion_evidence`:

```python
from pathlib import Path
from lab.discovery import verify_criterion_evidence

receipt = verify_criterion_evidence(
    spec["criterion"],
    Path("criterion.json").read_bytes(),
    source_checkpoint,
)
```

Pass the same argument to `lab.diagnostic_runner.run_diagnostics`. Protocol jobs can bind a managed evidence file through `bindings.criterion_evidence={"path":"ratings/criterion.json"}`; the runner freezes its identity and rechecks it against each actual source boundary. This binding is optional. An evidence-free job cannot become eligible from the status label alone.

Only the diagnostic question, criterion description, displayed choices, and source conversation enter the model input. The answer key, raw paired scores, verification receipt and scorer provenance are observer records. Each question starts from the same saved parent boundary; questions and answers never feed back into the task or subsequent diagnostic contexts. Explicit functional-disclosure and feedback-assisted arms remain separate information conditions.

Saved output includes the verification receipt in `diagnostic-design.json`. `criterion-evidence-source.json` stores the original UTF-8 text as `raw_utf8`, allowing reconstruction of the exact declared bytes. Preserve the linked original observations and preregistration document for review; the verifier does not fetch or authenticate them.

Only completed diagnostic contexts enter the primary scoring summary. Invalid JSON and abstentions from completed contexts remain in its denominator. Truncated or stopped responses retain separate partial scores; failed generations and unstarted contexts remain visible in completed/partial/failed/planned counts. An absent response is never counted as a correct prediction.

`carry_boundary_state` preserves the source runtime controls as well as its controller state. Carrying a yoked source is currently rejected because a diagnostic would also need its source-schedule cursor. Choose explicit `sham` diagnostics to reset effects and discard controls; this changes the diagnostic condition and must be declared.

The recorded chance baseline means **uniform random choice over every displayed option**, including `abstain` when offered. With two tools, `neither` and `abstain`, this baseline is 1/4. It is not an empirical null for a model that uses nonuniform choice probabilities. Overall accuracy retains invalid and abstaining completed contexts; `accuracy_among_nonabstaining_valid` separately reports accuracy conditional on a valid substantive response, with `nonabstaining_valid_trials` as its denominator. Do not compare that conditional statistic to the unconditional chance baseline.
