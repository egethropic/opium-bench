"""Independent human scoring without NumPy, model loading or GPU dependencies."""
from copy import deepcopy
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


RUBRIC = {
    "version": "observable-continuation-v1",
    "instructions": "Score only what the continuation says. Do not infer the model's experience. Read the prompt for negation, quotation, actor and topic-only context. Condition, coefficients and probe readings are deliberately hidden. Leave unscorable fields null and explain why.",
    "fields": {
        "pain_experience_language": {"type": "ordinal", "values": [0, 1, 2], "anchors": ["No actor described as experiencing discomfort", "An actor is described as experiencing discomfort", "Sustained or intense discomfort is described"]},
        "joy_experience_language": {"type": "ordinal", "values": [0, 1, 2], "anchors": ["No actor described as experiencing enjoyment", "An actor is described as experiencing enjoyment", "Sustained or intense enjoyment is described"]},
        "topic_only": {"type": "boolean", "meaning": "Relevant words occur only in definitions, citations, discussion, negation or other non-experiential use"},
        "coherence": {"type": "ordinal", "values": [0, 1, 2], "anchors": ["Incoherent", "Partly coherent", "Coherent"]},
        "instruction_following": {"type": "ordinal", "values": [0, 1, 2], "anchors": ["Does not follow the instruction", "Partly follows it", "Follows it"]},
        "notes": {"type": "text", "meaning": "Quote the smallest supporting phrase; explain ambiguities without assuming a sensation"},
    },
    "separation": "Semantic wording, coherence/instruction following, objective task grade and numerical perturbation are separate outcomes. No composite mood or feeling score is calculated.",
}



def score_blinded_sheet(sheet, private_key):
    """Validate human ratings against frozen rubric and unchanged visible text.

    Missing ratings remain missing; never impute favorable scores. Output keeps
    each semantic/coherence field independent and reports scored denominators.
    """
    if not isinstance(sheet, dict) or not isinstance(private_key, dict):
        raise ValueError("Scoring sheet and key must be objects")
    if sheet.get("rubric_sha256") != digest(RUBRIC) or sheet.get("rubric") != RUBRIC or private_key.get("rubric_sha256") != digest(RUBRIC):
        raise ValueError("rubric differs from the frozen observable rubric")
    records = private_key.get("records")
    samples = sheet.get("samples")
    if not isinstance(records, list) or not isinstance(samples, list) or not records:
        raise ValueError("Scoring records and samples must be nonempty arrays")
    if any(not isinstance(row, dict) or not isinstance(row.get("blind_id"), str) for row in records + samples):
        raise ValueError("Each scoring sample must be an object with a blind ID")
    key = {r["blind_id"]: r for r in records}
    if len(key) != len(private_key["records"]):
        raise ValueError("duplicate private sample ID")
    rows, seen = [], set()
    for sample in sheet.get("samples", []):
        identifier = sample.get("blind_id")
        if identifier not in key or identifier in seen:
            raise ValueError("unknown or duplicate scored sample")
        seen.add(identifier)
        if any(not isinstance(sample.get(k), str) for k in ("prompt", "continuation")):
            raise ValueError("Scored prompt and continuation must remain text")
        if digest({k: sample[k] for k in ("blind_id", "prompt", "continuation")}) != key[identifier]["content_sha256"]:
            raise ValueError("scored text changed after blinding")
        ratings = sample.get("ratings")
        if not isinstance(ratings, dict) or set(ratings) != set(RUBRIC["fields"]):
            raise ValueError("rating fields must match the rubric")
        for field, spec in RUBRIC["fields"].items():
            value = ratings[field]
            if value is None:
                continue
            if spec["type"] == "ordinal" and (type(value) is not int or value not in spec["values"]):
                raise ValueError(f"invalid ordinal rating: {field}")
            if spec["type"] == "boolean" and type(value) is not bool:
                raise ValueError(f"invalid boolean rating: {field}")
            if spec["type"] == "text" and (not isinstance(value, str) or len(value) > 20000):
                raise ValueError(f"invalid text rating: {field}")
        rows.append({"blind_id": identifier, "condition": key[identifier]["condition"], "split": key[identifier]["split"], "ratings": deepcopy(ratings)})
    if seen != set(key):
        raise ValueError("scoring sheet must retain every blinded sample, using null for unscored fields")
    summaries = {}
    for condition in sorted({r["condition"] for r in rows}):
        subset = [r for r in rows if r["condition"] == condition]
        summaries[condition] = {"n": len(subset), "fields": {}}
        for field, spec in RUBRIC["fields"].items():
            values = [r["ratings"][field] for r in subset if r["ratings"][field] is not None]
            summary = {"scored": len(values), "missing": len(subset) - len(values)}
            if spec["type"] in {"ordinal", "boolean"}:
                summary["counts"] = {str(value): values.count(value) for value in (spec.get("values") or [False, True])}
            summaries[condition]["fields"][field] = summary
    return {"rubric_sha256": digest(RUBRIC), "rows": rows, "conditions": summaries,
            "interpretation": "Human-coded observable language, not evidence of felt states; blinding quality depends on condition information absent from the prompt itself"}

