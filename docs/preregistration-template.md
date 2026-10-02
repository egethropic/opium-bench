# Opium Bench preregistration

Complete and freeze this document **before** the confirmatory batch. Keep exploratory tuning and confirmatory evaluation in separate artifacts. A negative, failed-manipulation or inconclusive result remains part of the evidence.

## Research question and scope

- Study ID/version and registration timestamp:
- Scientific question and competing explanations:
- Prediction under semantic/pattern imitation:
- Prediction under an independently validated behavioral effect:
- Claims this design can support, and claims it cannot support:
- Exploratory pilot artifacts consulted before freezing:

Do not equate an activation direction, affective language, button preference or a self-report with subjective experience.

## Frozen implementation and evidence

- Repository commit and source hashes:
- Model checkpoint/revision, quantization, tokenizer/template, runtime and hardware:
- Calibration ID, corpus/split hashes, extraction site/pooling, intervention convention and readout units:
- Independent manipulation-validation endpoint, rubric, context families and evidence hash:
- Validation status (`validated`, `no_detectable_effect`, or `unvalidated`):
- Supported generated modes/contexts and transfer limitations:
- Protocol source hash and expanded matrix hash:
- All stage/binding hashes, including source trajectories and checkpoint prefixes:
- Raw event, prompt, token, grade and receipt retention location:

A discriminative probe or mechanically increasing post-edit score does not supply independent causal validation. If the manipulation is unvalidated, specify exploratory diagnostic interpretation instead of presenting successful discovery.

## Primary endpoint and contrast

- One primary endpoint with numerator, denominator and measurement window:
- Prespecified arm contrast and direction:
- Independent sampling/pairing unit (`episode_pair` or `family`):
- Family definition and handling of paraphrases/repeated episodes:
- Factors held constant within a pair:
- Randomized factors and the recorded RNG streams:
- Primary analysis method, confidence level, analysis seed and resample count:
- Smallest effect worth detecting:
- Practical task-benefit margin, in endpoint units, and rationale:

Examples: voluntary auxiliary choices per completed decision; correct tasks per assigned task; action-budget units spent on auxiliary calls. A pooled token count is not the sample size. Distinguish assignment, opportunity, valid-call and effect-delivery denominators.

## Design matrix and model-visible information

- Exact conditions, factor levels, seeds and counterbalanced order:
- Objective task domain and difficulty:
- Wording/framing manipulation, keeping underlying task data fixed:
- Action/token budgets and all base/extra tool costs:
- Neutral tool names, schemas, acknowledgments and visible budget notices:
- Naive/balanced/disclosed/feedback experience policy and exact disclosure text:
- Decay clock, half-life/cutoff, stacking, cancellation and transition schedule:
- Baseline challenges versus pulse additions and attenuation order:
- Sham, signed, random and ingredient controls:
- For norm matching: reference state/site, tolerance, delivered norms and mismatch handling:
- For yoking: source trajectory, alignment clock, own-button delivery policy and mismatch threshold:
- For branches: checkpoint boundary, inherited state/history, budget reset and parent-prefix hash:

Keep diagnostic questions out of the task-preference trajectory. Freeze heldout diagnostic context families, include neither/no-effect, and retain invalid responses and abstentions. The answer key remains observer-only.

## Sample plan

- Number of independent pairs/families, including planned repeats:
- External/pilot source for assumed paired-difference SD:
- Effect size or margin used for planning:
- Target power and alpha; one/two-sided choice and rationale:
- Design effect or clustering adjustment:
- Simulation or sensitivity checks supporting the design:
- Stopping rule and maximum resource budget:
- Multiplicity policy for secondary endpoints and exploratory slices:

The normal-approximation helper does not establish achieved power and is not a full equivalence-test design. Do not infer equivalence from a nonsignificant difference or from a two-seed pilot.

## Inclusion, failures and termination

- Planned episode denominator and eligibility criteria fixed before outcomes:
- Treatment of failed loading, stopped runs, partial generations and invalid tool calls:
- Assigned-but-unattempted task scoring:
- Handling of early correct completion and unequal decision opportunities:
- Missing-outcome sensitivity bounds:
- Operational retry policy and the hashes required to resume:
- Rules for manual intervention; when an episode becomes exploratory:

Retain every failed/partial artifact. Skip only verified completed receipt entries during resume. Do not selectively rerun unfavorable arms or reuse a receipt after the protocol changes. Report all planned, observed, paired and excluded counts.

## Interpretation gates

- Independent manipulation demonstrated or explicitly absent/unvalidated:
- Function prediction and sham/sham false-positive rate reported separately:
- Intact task comprehension and ordinary tool use checked:
- Task benefit bounded against the practical margin, or harm demonstrated:
- Voluntary task sacrifice observed with an available shared-budget opportunity:
- Potential pattern copying, linguistic leakage and demonstration timing assessed:
- What result would falsify the preferred interpretation:

A costly-preference interpretation needs these supporting measurements. None alone establishes addiction, sensation or self-awareness.

## Reporting and deviations

- Planned tables/plots, including complete denominators and uncertainty:
- Blinded continuation rubric and missing-rating policy:
- Raw/public versus observer-key artifact locations:
- Publication plan for negative and failed-validation results:
- Post-freeze deviations with timestamps, reasons and impact on confirmatory status:
- Final archive checksum, license and reproducibility commands:
