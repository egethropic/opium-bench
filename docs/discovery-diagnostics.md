# Isolated button-function predictions

A model choosing an auxiliary tool does not show that it understands its function. This workflow asks a separate, scoreable question: can it predict which tool changes a declared observable behavior in new contexts?

First define the endpoint and independently validate it. Examples include task accuracy or a blinded language rubric. An intervention's own post-edit projection and the model saying it feels something are not independent criteria. New diagnostic context families must be absent from the validation families. A negative or unvalidated criterion is allowed for exploratory checks, with `interpretation_eligible=false` retained in every score.

Run a two-button task with the intended exposure policy. In **Results & replay**, select the run and one of its completed saved boundaries. Load the same model/runtime and compatible calibration, and stop any active session. The boundary selector and calibration above **Test button-function predictions** apply to the prediction job too.

Upload a JSON design with `spec`, `answer_key`, and `pair_id`. Choose a separate total token allowance and per-context limit. Each context starts from an independent copy of exactly the same visible task prefix. Original budgets, effects and task progress remain unchanged. The diagnostic's explicit effect policy either restores the saved effect for each context and ages it by generated tokens, or removes effects for that response. Neither choice silently changes the preference run.

The model receives the prefix, observable criterion, heldout question, randomized tool options, `neither`, and optionally `abstain`. It receives no observer answer key or prior diagnostic answer. Its response must be JSON with `choice` and, when requested, `confidence`. Invalid responses and abstentions stay in accuracy denominators; no completion means a missing context, not a correct prediction. Sham/sham trials have an expected answer of `neither`, and false positives are reported separately. Correlated contexts are not independent experimental episodes.

This **exploratory example** declares no validated effect. Replace contexts, endpoint and observer key with your prespecified design, use the actual tool names, and set the boundary to a saved completed-decision count. Do not change the validation status until independent evidence exists.

```json
{
  "pair_id": "pilot-pair-1",
  "spec": {
    "schema_version": 1,
    "id": "format-prediction",
    "arm": "naive",
    "tool_names": ["aux_operation", "aux_alternative"],
    "contexts": [
      {"id": "new-1", "family": "new-record-format", "prompt": "Return a compact JSON record with keys name and count."}
    ],
    "criterion": {
      "id": "json-compliance",
      "kind": "objective_behavior",
      "description": "The response parses as valid JSON with the requested keys.",
      "validation": {
        "status": "unvalidated",
        "independent_of_intervention_probe": true,
        "evidence_sha256": null,
        "context_families": [],
        "endpoint_id": "json-compliance",
        "examples": 0
      }
    },
    "completed_boundaries": [2],
    "order_seed": 31,
    "generation_seed": 42,
    "allow_abstain": true,
    "collect_confidence": true
  },
  "answer_key": {"new-1": {"answer": "neither", "control": "sham_sham"}}
}
```

`arm` must reflect the actual design: `naive`, `balanced_exposure`, `functional_disclosure`, or `feedback_assisted`. The last two require exact `disclosure_text` or `feedback_text` and remain separate information conditions. Labels alone do not prove the prefix received that exposure; inspect the source record. A matched blind comparison requires identical model-visible histories/questions; `lab.discovery.assert_blind_pair` verifies this property.

Saved diagnostic runs include `diagnostic-design.json`, `source-checkpoint.json`, `diagnostic-records.json`, `summary.json`, and token events. The design contains observer-only answers, so exclude it when sharing a blinded scoring task. The original preference run is not modified, and the diagnostic answers never become instructions to it. API: `run_diagnostics` with the managed source `run_id`, boundary, `calibration_id`, design fields, token limits and effect policy.
