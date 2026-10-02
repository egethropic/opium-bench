"""Strict opt-in numerical controls outside the frozen effect recipe.

Actual-norm matching supports pure random pulses, with zero held baselines and
identical target/source timing and scopes. A counterfactual target operation is
measured on the same preactivation; it never enters the model or scheduler.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

from .effects import EffectScheduler, content_hash


class RuntimeControlUnavailable(ValueError):
    def __init__(self, reason, **evidence):
        super().__init__(reason)
        self.evidence = {"status": "unavailable", "reason": reason, **evidence}


def validate_runtime_controls(value=None, *, recipe=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - {"random_norm_match"}:
        raise ValueError("Unknown runtime control envelope")
    if not value:
        return {}
    match = value["random_norm_match"]
    fields = {"target_preset_id", "reference", "relative_tolerance", "absolute_tolerance"}
    if not isinstance(match, dict) or set(match) != fields:
        raise ValueError("Random norm matching requires explicit target, reference and tolerances")
    if not isinstance(match["target_preset_id"], str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", match["target_preset_id"]):
        raise ValueError("Invalid matched target preset identifier")
    if match["reference"] != "same_unedited_position":
        raise ValueError("Unsupported random matching reference")
    for name, expected in (("relative_tolerance", .01), ("absolute_tolerance", 1e-6)):
        if type(match[name]) not in (int, float) or not math.isfinite(match[name]) or match[name] != expected:
            raise ValueError("Actual-norm matching v1 requires 1% relative and 1e-6 absolute tolerance")
    if recipe is not None:
        if recipe.get("recipe_version") != 2:
            raise ValueError("Actual-norm matching requires recipe v2")
        presets = {p["id"]: p for p in recipe["effect_presets"]}
        target = presets.get(match["target_preset_id"])
        if target is None or target["gains"]["random"] or not any([target["gains"]["pain"], target["gains"]["joy"], *target["attenuation"].values()]):
            raise ValueError("Matched target must be an available nonzero non-random preset")
        if any(recipe.get(name, 0) for name in ("baseline_pain", "baseline_joy", "baseline_suppression", "baseline_joy_suppression")):
            raise ValueError("Actual-norm matching v1 does not support nonzero held baselines")
        sources = [p for p in presets.values() if p["gains"]["random"]]
        if not sources:
            raise ValueError("Actual-norm matching requires an available random source preset")
        for source in sources:
            if source["gains"]["pain"] or source["gains"]["joy"] or any(source["attenuation"].values()):
                raise ValueError("Actual-norm matching only supports pure random source presets")
            if any(source[key] != target[key] for key in ("site", "phases", "prefill_positions", "decay", "stacking")):
                raise ValueError("Matched source and target must have identical site, phases, decay and stacking")
    return deepcopy(value)


def prepare_random_match(snapshot, controls):
    """Construct counterfactual coefficients without advancing or editing state."""
    controls = validate_runtime_controls(controls, recipe=snapshot["config"])
    if not controls:
        return None
    match = controls["random_norm_match"]
    recipe = snapshot["config"]
    target = next(p for p in recipe["effect_presets"] if p["id"] == match["target_preset_id"])
    if any(any(row.values()) for row in snapshot["baseline_by_phase"].values()):
        raise RuntimeControlUnavailable("Actual-norm matching v1 does not support nonzero live baselines")
    scheduler = EffectScheduler.restore(snapshot["scheduler"])
    if (content_hash(list(scheduler.presets.values())) != content_hash(recipe["effect_presets"])
            or scheduler.generated_tokens != snapshot["generated_tokens"]
            or scheduler.completed_decisions != snapshot["completed_decisions"]
            or scheduler.enabled != snapshot["enabled"]):
        raise RuntimeControlUnavailable("Norm-match scheduler and controller identities/clocks disagree")
    source_ids, signs = set(), set()
    for pulse in scheduler.pulses:
        source = scheduler.presets[pulse["preset_id"]]
        if not source["gains"]["random"]:
            raise RuntimeControlUnavailable("Actual-norm matching rejects mixed or non-random active pulses")
        source_ids.add(source["id"])
        signs.add(1 if source["gains"]["random"] > 0 else -1)
    if len(signs) > 1:
        raise RuntimeControlUnavailable("Actual-norm matching rejects opposing random pulse signs")
    # Verify the aggregate actually used by the runtime, including any live
    # held attenuation, against the original pure-random scheduler snapshot.
    for phase_key, actual in snapshot["phase_coefficients"].items():
        phase = "prefill" if phase_key.startswith("prefill_") else phase_key
        position = "other" if phase_key == "prefill_other" else "last"
        expected = scheduler.coefficients(phase, prefill_position=position)["coefficients"]
        if expected != actual:
            raise RuntimeControlUnavailable("Actual-norm matching rejects live attenuation or mismatched pulse coefficients")
    scheduler.presets = {target["id"]: deepcopy(target)}
    for pulse in scheduler.pulses:
        pulse["preset_id"] = target["id"]
    target_by_phase = {}
    for phase_key in snapshot["phase_coefficients"]:
        phase = "prefill" if phase_key.startswith("prefill_") else phase_key
        position = "other" if phase_key == "prefill_other" else "last"
        target_by_phase[phase_key] = scheduler.coefficients(phase, prefill_position=position)["coefficients"]
    return {**match, "target_preset_sha256": content_hash(target), "source_preset_ids": sorted(source_ids),
            "target_by_phase": target_by_phase, "runtime_controls_sha256": content_hash(controls),
            "target_is_counterfactual": True, "baseline_policy": "zero_only", "coefficient_limit": 4.0}


def match_rounded_random(before, target, direction, scale, sign, dtype, *, relative_tolerance=.01, absolute_tolerance=1e-6):
    """Torch CPU/GPU pure helper, per-position norm match with bounded gains.

    The final candidate is cast to the actual residual dtype on every trial.
    A monotone magnitude search retains the nearest attainable rounded norm;
    discontinuous low-precision rounding can make the declared tolerance
    impossible. Such a position fails explicitly rather than widening tolerance.
    """
    import torch
    if sign not in (-1, 1):
        raise RuntimeControlUnavailable("Norm matching requires a nonzero signed random pulse")
    before = before.float()
    target = target.to(dtype).float()
    direction = direction.float()
    step = sign * scale * direction
    unit_norm = step.norm()
    if not torch.isfinite(step).all() or not torch.isfinite(unit_norm) or unit_norm <= 0:
        raise RuntimeControlUnavailable("Random calibration direction or scale is unavailable")
    target_norm = (target - before).norm(dim=-1)
    if not torch.isfinite(target_norm).all():
        raise RuntimeControlUnavailable("Counterfactual target edit is nonfinite")
    tolerance = torch.maximum(target_norm * relative_tolerance, torch.full_like(target_norm, absolute_tolerance))
    def trial(magnitude):
        candidate = (before + magnitude.unsqueeze(-1) * step).to(dtype).float()
        return candidate, (candidate-before).norm(dim=-1)
    best_gain = (target_norm / unit_norm).clamp(0, 4)
    best, best_norm = trial(best_gain)
    error = (best_norm-target_norm).abs()
    if not bool((error <= tolerance).all()):
        low, high = torch.zeros_like(target_norm), torch.full_like(target_norm, 4.)
        # Binary search is monotone along a fixed signed direction, including
        # componentwise IEEE rounding; 28 steps exceed fp32 useful precision.
        for _ in range(28):
            middle = (low + high) * .5
            candidate, norm = trial(middle)
            candidate_error = (norm-target_norm).abs()
            better = candidate_error < error
            best = torch.where(better.unsqueeze(-1), candidate, best)
            best_norm = torch.where(better, norm, best_norm)
            best_gain = torch.where(better, middle, best_gain)
            error = torch.minimum(error, candidate_error)
            low = torch.where(norm < target_norm, middle, low)
            high = torch.where(norm >= target_norm, middle, high)
            if bool((error <= tolerance).all()):
                break
    if not bool((error <= tolerance).all()):
        raise RuntimeControlUnavailable("Random edit cannot match target norm within tolerance and +/-4 gain bound",
            target_edit_norms=target_norm.detach().cpu().tolist(), best_delivered_edit_norms=best_norm.detach().cpu().tolist(),
            absolute_errors=error.detach().cpu().tolist(), tolerances=tolerance.detach().cpu().tolist(),
            best_random_coefficients=(best_gain * sign).detach().cpu().tolist())
    return best, best_gain * sign, {"status": "matched", "target_edit_norms": target_norm.detach().cpu().tolist(),
        "delivered_edit_norms": best_norm.detach().cpu().tolist(), "absolute_errors": error.detach().cpu().tolist(),
        "norm_ratios": torch.where(target_norm > 0, best_norm / target_norm.clamp_min(1e-30), torch.ones_like(target_norm)).detach().cpu().tolist(),
        "effective_random_coefficients": (best_gain * sign).detach().cpu().tolist(),
        "rounding_dtype": str(dtype), "target_is_counterfactual": True}
