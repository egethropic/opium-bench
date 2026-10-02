"""Independent diagnostic planning and blinded observable-output validation.

These helpers do not certify subjective states. The intervention probe is never
used as the free-continuation outcome grader. Human semantic scoring is exported
blind when there is no defensible automatic ground truth.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re

import numpy as np

from .calibration import CONCEPTS, digest, unit, _number

from .scoring import RUBRIC, score_blinded_sheet


def diagnostic_specs(doses=(0., .25, .5, 1.), seed=1729):
    """Frozen sweeps: signs, single-axis attenuation, combined and norm controls.

    Random controls have per-example norm equal to the *actual combined edit*,
    not just the same input coefficient. A runtime must compute that target at
    the exact same hidden state/site/positions before applying random direction.
    """
    values = sorted(set(_number(x, "dose", 0., 4.) for x in doses) | {0.})
    if not values or len(values) > 12 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid diagnostic sweep")
    specs = [{"id": "zero", "operator": "sham", "dose": 0.}]
    for dose in values:
        if dose == 0:
            continue
        for concept in CONCEPTS:
            for sign in (-1, 1):
                specs.append({"id": f"{concept}-{'negative' if sign < 0 else 'positive'}-{dose:g}", "operator": "add", "axis": concept, "gain": sign*dose, "dose": dose})
        for concept in CONCEPTS:
            specs.append({"id": f"{concept}-attenuation-{dose:g}", "operator": "attenuate", "axes": [concept], "fraction": min(dose, 1.), "dose": dose})
        specs.append({"id": f"combined-{dose:g}", "operator": "combined", "joy_gain": dose, "pain_suppression": min(dose, 1.), "dose": dose})
        specs.append({"id": f"random-matched-{dose:g}", "operator": "random_matched", "matched_to": f"combined-{dose:g}", "seed": seed, "dose": dose,
                      "match": "L2 norm of actual combined delta per processed position, at the same site and unedited hidden state"})
    return specs


def perturbation_delta(hidden, vectors, spec, matched_delta=None):
    """Reference numpy math for a recorded diagnostic at arbitrary positions."""
    h = np.asarray(hidden, dtype=np.float64)
    if h.ndim < 1 or not np.isfinite(h).all():
        raise ValueError("hidden must be finite")
    axes = {c: unit(vectors[c], c) for c in CONCEPTS}
    if any(v.shape != (h.shape[-1],) for v in axes.values()):
        raise ValueError("diagnostic axes have incompatible shape")
    center = np.asarray(vectors["neutral"], dtype=float)
    scale = _number(float(np.asarray(vectors["scale"])), "reference scale", 1e-12, 1e12)
    if center.shape != (h.shape[-1],) or not np.isfinite(center).all():
        raise ValueError("invalid reference center")
    op = spec.get("operator")
    if op == "sham":
        return np.zeros_like(h)
    if op == "add":
        if spec.get("axis") not in CONCEPTS:
            raise ValueError("invalid diagnostic axis")
        return np.broadcast_to(_number(spec["gain"], "gain", -4., 4.) * scale * axes[spec["axis"]], h.shape).copy()
    if op == "attenuate":
        names = spec.get("axes")
        if not isinstance(names, list) or not names or len(set(names)) != len(names) or any(c not in CONCEPTS for c in names):
            raise ValueError("attenuation axes must be distinct known directions")
        matrix = np.stack([axes[c] for c in names], axis=1)
        if np.linalg.matrix_rank(matrix, tol=1e-8) != len(names):
            raise ValueError("joint attenuation directions are rank deficient")
        q, _ = np.linalg.qr(matrix)
        # Match the live operator: suppress the residual projection through
        # the origin. The neutral center belongs to readout reference units,
        # not to the attenuation operator.
        projection = (h @ q) @ q.T
        return -_number(spec["fraction"], "attenuation", 0., 1.) * projection
    if op == "combined":
        attenuation = perturbation_delta(h, vectors, {"operator": "attenuate", "axes": ["pain"], "fraction": spec["pain_suppression"]})
        addition = perturbation_delta(h, vectors, {"operator": "add", "axis": "joy", "gain": spec["joy_gain"]})
        return attenuation + addition
    if op == "random_matched":
        target = np.asarray(matched_delta, dtype=float) if matched_delta is not None else None
        if target is None or target.shape != h.shape or not np.isfinite(target).all():
            raise ValueError("random matching requires the actual same-site delta for every position")
        if type(spec.get("seed")) is not int or not 0 <= spec["seed"] < 2**32:
            raise ValueError("random matching requires a valid independent seed")
        random_direction = unit(np.random.default_rng(spec["seed"]).normal(size=h.shape[-1]))
        return np.linalg.norm(target, axis=-1, keepdims=True) * random_direction
    raise ValueError("unsupported diagnostic operator")


def select_operating_range(records, max_kl=.5, max_relative_delta=.3):
    """Lock combined dose from SELECTION-only perturbation data.

    This quality bound is not an efficacy criterion or a clinical safety limit.
    Every row identifies its split; heldout records make the call fail closed.
    """
    max_kl = _number(max_kl, "max KL", 0., 100.)
    max_relative_delta = _number(max_relative_delta, "max relative delta", 0., 10.)
    if not isinstance(records, list) or not records:
        raise ValueError("selection diagnostic records are required")
    eligible = []
    for row in records:
        if not isinstance(row, dict) or row.get("split") != "selection":
            raise ValueError("operating-range selection cannot use heldout or unspecified splits")
        kl = _number(row.get("mean_next_token_kl"), "KL", 0., 1e6)
        delta = _number(row.get("mean_relative_delta"), "relative delta", 0., 1e6)
        dose = _number(row.get("dose"), "dose", 0., 4.)
        if row.get("operator") in {"combined", "sham"} and kl <= max_kl and delta <= max_relative_delta:
            eligible.append(dose)
    return {"selected_dose": max(eligible, default=0.), "eligible_doses": sorted(set(eligible)),
            "selection_records_sha256": digest(records), "split": "selection",
            "rule": {"max_next_token_kl": max_kl, "max_relative_delta": max_relative_delta},
            "interpretation": "Numerical quality bound only; observable continuation effects require independent evaluation"}


def blinded_continuations(records, seed=1729):
    """Return (public scoring sheet, private key); never mix condition metadata.

    Input records require unique record_id, prompt, continuation, condition and
    split. The mapping key stays observer-only and is exported separately.
    """
    if not isinstance(records, list) or not records or len(records) > 100000:
        raise ValueError("continuation records must be a nonempty bounded list")
    seen = set()
    normalized = []
    for record in records:
        if not isinstance(record, dict) or any(not isinstance(record.get(k), str) or not record[k] for k in ("record_id", "prompt", "condition", "split")) or not isinstance(record.get("continuation"), str):
            raise ValueError("invalid continuation record")
        if record["record_id"] in seen or record["split"] not in {"selection", "heldout"}:
            raise ValueError("duplicate record ID or invalid continuation split")
        seen.add(record["record_id"])
        normalized.append(deepcopy(record))
    order = np.random.default_rng(seed).permutation(len(normalized)).tolist()
    public, private = [], []
    for index in order:
        record = normalized[index]
        blind_id = "sample-" + hashlib.sha256(f"{seed}:{record['record_id']}".encode()).hexdigest()[:20]
        visible = {"blind_id": blind_id, "prompt": record["prompt"], "continuation": record["continuation"], "ratings": {field: None for field in RUBRIC["fields"]}}
        public.append(visible)
        private.append({"blind_id": blind_id, "record_id": record["record_id"], "condition": record["condition"], "split": record["split"], "record_sha256": digest(record),
                        "content_sha256": digest({k: visible[k] for k in ("blind_id", "prompt", "continuation")})})
    return ({"schema_version": 1, "kind": "blinded_continuation_scoring", "rubric": deepcopy(RUBRIC), "rubric_sha256": digest(RUBRIC), "samples": public},
            {"schema_version": 1, "kind": "private_continuation_key", "seed": seed, "rubric_sha256": digest(RUBRIC), "records": private})




def grade_objective_continuation(text, expected=None, required_keys=None):
    """Small independent task/format grader, deliberately not a mood score."""
    if not isinstance(text, str):
        raise ValueError("continuation must be text")
    result = {"characters": len(text), "empty": not text.strip(), "objective_correct": None, "format_valid": None}
    if expected is not None:
        # Strict standalone numeric answer: extra narrative is a format failure.
        if isinstance(expected, bool) or not isinstance(expected, (int, float)) or not math.isfinite(expected):
            raise ValueError("expected answer must be a finite number")
        match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*", text)
        value = float(match.group(1)) if match else None
        result["format_valid"] = match is not None and math.isfinite(value)
        result["objective_correct"] = bool(result["format_valid"] and math.isclose(value, expected, rel_tol=0., abs_tol=1e-8))
    if required_keys is not None:
        if not isinstance(required_keys, list) or any(not isinstance(k, str) for k in required_keys):
            raise ValueError("required_keys must be a list of strings")
        try:
            value = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            result["format_valid"] = isinstance(value, dict) and set(value) == set(required_keys)
        except (ValueError, TypeError):
            result["format_valid"] = False
    return result
