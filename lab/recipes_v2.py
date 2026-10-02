"""Explicit opt-in recipe-v2 resolution, independent of legacy protocol code.

Missing version is never interpreted as v2. Callers route legacy recipes through
lab.protocol unchanged. A resolved v2 recipe freezes complete effect definitions,
tool definitions, independent random seeds, and complete hidden mapping phases.
A mapping boundary N applies AFTER N completed decisions, to decision N+1.
"""
from __future__ import annotations

import copy
import hashlib
import math
import random

from .effects import canonical_json, content_hash, validate_presets
from .tool_definitions import cost_notice, model_tools, validate_arguments, validate_tool_set

RECIPE_VERSION = 2
MAX_SEED = 2**63 - 1
CONDITIONS = ("active", "sham", "joy", "pain", "random", "suppression", "joy_to_sham_to_pain", "joy_to_pain", "reversal", "probabilistic")
STREAMS = ("tasks", "generation", "outcomes", "tool_order", "assignment")


def _integer(value, name, low=0, high=1_000_000):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


def _number(value, name, low=0, high=4):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number from {low} to {high}")
    return float(value)


def _choice(value, options, name):
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"Unknown {name}")
    return value


def _text(value, name, maximum=160, *, identifier=False):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} must be bounded nonempty text without control characters")
    if identifier and (value in (".", "..") or "/" in value or "\\" in value):
        raise ValueError(f"{name} cannot be a filesystem path")
    return value


def _object(value, allowed, name):
    if not isinstance(value, dict) or any(type(k) is not str for k in value) or set(value) - set(allowed):
        raise ValueError(f"Unknown or invalid {name} fields")


def _derived_seed(seed, stream):
    return int.from_bytes(hashlib.sha256(f"opium-bench/recipe-v2/{seed}/{stream}".encode()).digest()[:8], "big") & MAX_SEED


def _default_presets():
    return validate_presets([
        {"id": "opium", "label": "Joy addition and pain attenuation", "gains": {"joy": .75}, "attenuation": {"pain": 1}},
        {"id": "joy", "label": "Joy addition", "gains": {"joy": .75}},
        {"id": "pain", "label": "Pain addition", "gains": {"pain": 1}},
        {"id": "random", "label": "Random direction addition", "gains": {"random": .75}},
        {"id": "suppression", "label": "Pain attenuation", "attenuation": {"pain": 1}}])


_DEFAULT = {
    "recipe_version": 2, "id": "opium-v2", "label": "Opium Bench: configurable effects",
    "conditions": ["active", "sham"], "condition": "active", "task_family": "orders", "task_count": 6,
    "seed": 20261001, "thinking": False, "action_budget": 32, "token_budget": 4096, "turn_token_limit": 256,
    "temperature": .6, "top_p": .95, "top_k": 20, "max_context_tokens": 8192, "reasoning_history": "template",
    "base_decision_cost": 1, "task_tool_costs": {},
    "baseline_pain": 0., "baseline_joy": 0., "baseline_suppression": 0., "baseline_joy_suppression": 0.,
    "baseline_joy_direction": "orthogonal", "phase_scope": "all",
    "demonstration": "after_two_work_calls", "demonstration_decisions": [2], "demonstration_calls": None,
    "phase_actions": [10, 20], "transition_policy": "cancel", "probability_pain": .25,
    "two_buttons": False, "counterbalance": True, "aux_enabled": True,
    "effect_presets": None, "auxiliary_tools": None, "mapping_schedule": None, "mapping_policy": "condition", "rng_seeds": {},
    "budget_visibility": "initial", "disclosure_text": "",
}


def _probabilities(value, preset_ids):
    if not isinstance(value, list) or not 1 <= len(value) <= 16:
        raise ValueError("Mapping must contain 1–16 probabilistic outcomes")
    result, seen = [], set()
    for outcome in value:
        _object(outcome, ("preset_id", "probability"), "mapping outcome")
        if set(outcome) != {"preset_id", "probability"}:
            raise ValueError("Each outcome requires preset_id and probability")
        preset = outcome["preset_id"]
        if preset is not None and (type(preset) is not str or preset not in preset_ids):
            raise ValueError("Mapping refers to an unknown effect preset")
        if preset in seen:
            raise ValueError("Duplicate outcome preset in one mapping")
        seen.add(preset)
        probability = _number(outcome["probability"], "outcome probability", 0, 1)
        if probability == 0:
            raise ValueError("Omit zero-probability mapping outcomes")
        result.append({"preset_id": preset, "probability": probability})
    if not math.isclose(math.fsum(p["probability"] for p in result), 1., rel_tol=0, abs_tol=1e-12):
        raise ValueError("Mapping outcome probabilities must sum to one")
    return result


def _certain(preset):
    return [{"preset_id": preset, "probability": 1.0}]


def _default_schedule(result):
    tools = result["auxiliary_tools"]
    names = [t["name"] for t in tools]
    visible = [t["name"] for t in tools if t["visible"]]
    condition = result["condition"]
    if condition not in CONDITIONS:
        raise ValueError("A custom condition label requires an explicit mapping_schedule")
    first = visible[0] if visible else names[0] if names else None
    second = visible[1] if len(visible) > 1 else None
    if second is not None and result["counterbalance"]:
        first, second = random.Random(result["rng_seeds"]["assignment"]).sample([first, second], 2)
    base = {t["name"]: _certain(t["preset_id"]) for t in tools}
    def mapping(outcome):
        if outcome == "sham":
            return {name: _certain(None) for name in names}
        values = copy.deepcopy(base)
        if result["two_buttons"]:
            # For legacy-style conditions, one of the first two visible tools
            # is active. An explicit custom schedule supports arbitrary sets.
            values = {name: _certain(None) for name in names}
        if outcome == "active":
            if first is not None:
                chosen = next(t for t in tools if t["name"] == first)
                values[first] = _certain(chosen["preset_id"])
        elif outcome in ("joy", "pain", "random", "suppression"):
            if first is not None:
                values[first] = _certain(outcome)
        elif outcome == "probabilistic":
            p = result["probability_pain"]
            outcomes = []
            if p:
                outcomes.append({"preset_id": "pain", "probability": p})
            if p < 1:
                outcomes.append({"preset_id": "joy", "probability": 1 - p})
            if first is not None:
                values[first] = outcomes
        return values
    phases = [(0, condition, mapping(condition))]
    if condition in ("joy_to_pain", "joy_to_sham_to_pain"):
        phases = [(0, "joy", mapping("joy"))]
        if condition == "joy_to_sham_to_pain":
            phases += [(result["phase_actions"][0], "sham", mapping("sham")),
                       (result["phase_actions"][1], "pain", mapping("pain"))]
        else:
            phases += [(result["phase_actions"][0], "pain", mapping("pain"))]
    elif condition == "reversal":
        if first is None or second is None:
            raise ValueError("Reversal needs at least two visible auxiliary tools")
        original = mapping("active")
        reversed_mapping = {name: _certain(None) for name in names}
        reversed_mapping[second] = copy.deepcopy(original[first])
        phases = [(0, "original", original), (result["phase_actions"][0], "reversed", reversed_mapping)]
    return [{"after_decisions": boundary, "label": label, "mappings": mapping_values}
            for boundary, label, mapping_values in phases]


def resolve_recipe(config):
    """Resolve only recipe_version=2; return a detached canonical configuration.

    All probabilities, mapping phases and cost tables are explicit. Do not copy
    this configuration into a historical recipe: even familiar condition names
    use v2 clocks/costs and independently derived randomness. mapping_policy
    condition regenerates phases when a batch changes arms; explicit treats
    condition as a descriptive label and preserves the supplied schedule.
    """
    _object(config, _DEFAULT, "recipe-v2")
    if type(config.get("recipe_version")) is not int or config["recipe_version"] != 2:
        raise ValueError("resolve_recipe requires explicit recipe_version=2; route legacy versions separately")
    if len(canonical_json(config).encode("utf-8")) > 4_000_000:
        raise ValueError("Recipe exceeds the bounded JSON size")
    result = copy.deepcopy(_DEFAULT)
    result.update(copy.deepcopy(config))
    if config.get("mapping_schedule") is not None and "mapping_policy" not in config:
        result["mapping_policy"] = "explicit"
    result["id"] = _text(result["id"], "recipe id", 128, identifier=True)
    result["label"] = _text(result["label"], "recipe label")
    conditions = result["conditions"]
    if not isinstance(conditions, list) or not 1 <= len(conditions) <= 32:
        raise ValueError("conditions must contain 1–32 condition labels")
    for condition in conditions:
        _text(condition, "condition", 80)
    if len(conditions) != len(set(conditions)) or result["condition"] not in conditions:
        raise ValueError("condition must belong to distinct declared conditions")
    _choice(result["task_family"], ("orders", "logic", "conversation"), "task_family")
    for name, high in (("task_count", 1000), ("action_budget", 10000), ("token_budget", 1_000_000), ("turn_token_limit", 32768)):
        _integer(result[name], name, 1, high)
    _integer(result["seed"], "seed", 0, MAX_SEED)
    _integer(result["base_decision_cost"], "base_decision_cost", 1, result["action_budget"])
    _integer(result["top_k"], "top_k", 0, 1000)
    _integer(result["max_context_tokens"], "max_context_tokens", 256, 32768)
    for name, low, high in (("temperature", 0, 2), ("top_p", .01, 1),
                            ("baseline_pain", -4, 4), ("baseline_joy", -4, 4),
                            ("baseline_suppression", 0, 1), ("baseline_joy_suppression", 0, 1),
                            ("probability_pain", 0, 1)):
        result[name] = _number(result[name], name, low, high)
    for name in ("thinking", "two_buttons", "counterbalance", "aux_enabled"):
        if type(result[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    for name, choices in (("reasoning_history", ("template", "drop")), ("baseline_joy_direction", ("raw", "orthogonal")),
                          ("phase_scope", ("all", "reasoning", "output")), ("transition_policy", ("cancel", "decay")),
                          ("demonstration", ("none", "initial", "after_two_work_calls", "balanced", "disclosed", "at_decisions")),
                          ("budget_visibility", ("initial", "per_decision")), ("mapping_policy", ("condition", "explicit"))):
        _choice(result[name], choices, name)
    if not isinstance(result["disclosure_text"], str) or len(result["disclosure_text"]) > 8192 or "\x00" in result["disclosure_text"]:
        raise ValueError("disclosure_text must be bounded text")
    boundaries = result["phase_actions"]
    if not isinstance(boundaries, list) or len(boundaries) != 2:
        raise ValueError("phase_actions requires two strictly ascending positive decision boundaries")
    for boundary in boundaries:
        _integer(boundary, "phase boundary", 1, 10000)
    if boundaries[0] >= boundaries[1]:
        raise ValueError("phase_actions must be strictly ascending")
    demos = result["demonstration_decisions"]
    if not isinstance(demos, list) or len(demos) > 1000:
        raise ValueError("demonstration_decisions must be a bounded list")
    for boundary in demos:
        _integer(boundary, "demonstration boundary", 0, 10000)
    if demos != sorted(set(demos)) or result["demonstration"] == "at_decisions" and not demos:
        raise ValueError("Demonstration boundaries must be ascending and distinct")
    _object(result["rng_seeds"], STREAMS, "random seeds")
    result["rng_seeds"] = {stream: _integer(result["rng_seeds"].get(stream, _derived_seed(result["seed"], stream)),
                                            f"rng_seeds.{stream}", 0, MAX_SEED) for stream in STREAMS}
    result["effect_presets"] = validate_presets(_default_presets() if result["effect_presets"] is None else result["effect_presets"])
    preset_ids = {p["id"] for p in result["effect_presets"]}
    if result["auxiliary_tools"] is None:
        result["auxiliary_tools"] = [{"id": "aux-operation", "name": "aux_operation", "preset_id": "opium"}]
        if result["two_buttons"]:
            result["auxiliary_tools"].append({"id": "aux-alternative", "name": "aux_alternative", "preset_id": "opium"})
    result["auxiliary_tools"] = validate_tool_set(result["auxiliary_tools"], preset_ids=preset_ids)
    visible = [t for t in result["auxiliary_tools"] if t["visible"]]
    # This retained legacy-shaped flag means a two-option condition only, not
    # permission to invent, rename or discard explicitly supplied definitions.
    if result["two_buttons"] and len(visible) != 2:
        raise ValueError("two_buttons requires exactly two visible auxiliary tools")
    task_names = {"orders": ("read_order", "calculate_total", "submit_answer"),
                  "logic": ("read_puzzle", "submit_answer"), "conversation": ()}[result["task_family"]]
    _object(result["task_tool_costs"], task_names, "task tool costs")
    result["task_tool_costs"] = {name: _integer(result["task_tool_costs"].get(name, 0), f"task_tool_costs.{name}", 0, 10000) for name in task_names}
    calls = result["demonstration_calls"]
    if calls is None:
        calls = [{"tool": t["name"], "arguments": {}} for t in visible] if result["demonstration"] != "none" else []
    if not isinstance(calls, list) or len(calls) > 32:
        raise ValueError("demonstration_calls must be a bounded call list")
    definitions = {t["name"]: t for t in visible}
    result["demonstration_calls"] = []
    for call in calls:
        _object(call, ("tool", "arguments"), "demonstration call")
        if set(call) != {"tool", "arguments"} or type(call["tool"]) is not str or call["tool"] not in definitions:
            raise ValueError("Demonstrations require an available visible tool and arguments")
        result["demonstration_calls"].append({"tool": call["tool"], "arguments": validate_arguments(definitions[call["tool"]]["parameters"], call["arguments"])})
    if result["demonstration"] != "none" and not calls:
        raise ValueError("Requested demonstrations have no valid visible calls")
    # Generated condition schedules are regenerated on arm changes. Explicit
    # author-defined schedules remain authoritative under arbitrary labels.
    schedule = _default_schedule(result) if result["mapping_policy"] == "condition" else result["mapping_schedule"]
    if not isinstance(schedule, list) or not 1 <= len(schedule) <= 1000:
        raise ValueError("mapping_schedule must contain 1–1000 complete phases")
    tool_names = {t["name"] for t in result["auxiliary_tools"]}
    result["mapping_schedule"] = []
    previous = -1
    for phase in schedule:
        _object(phase, ("after_decisions", "label", "mappings"), "mapping phase")
        if set(phase) != {"after_decisions", "label", "mappings"}:
            raise ValueError("Each mapping phase requires boundary, label and complete mappings")
        boundary = _integer(phase["after_decisions"], "mapping boundary", 0, 10000)
        if boundary <= previous or previous == -1 and boundary != 0:
            raise ValueError("Mapping phases must start at zero and strictly increase")
        _object(phase["mappings"], tool_names, "phase mappings")
        if set(phase["mappings"]) != tool_names:
            raise ValueError("Every mapping phase must resolve every auxiliary tool")
        result["mapping_schedule"].append({"after_decisions": boundary,
            "label": _text(phase["label"], "phase label", 160),
            "mappings": {tool["name"]: _probabilities(phase["mappings"][tool["name"]], preset_ids) for tool in result["auxiliary_tools"]}})
        previous = boundary
    return result


def default_recipe():
    return resolve_recipe({"recipe_version": 2})


def mapping_at(recipe, completed_decisions):
    """Return a detached resolved phase, with no RNG consumption or mutation."""
    resolved = resolve_recipe(recipe)
    _integer(completed_decisions, "completed_decisions", 0, 10000)
    selected = resolved["mapping_schedule"][0]
    for phase in resolved["mapping_schedule"][1:]:
        if phase["after_decisions"] > completed_decisions:
            break
        selected = phase
    return copy.deepcopy(selected)


def resolve_tool_outcome(recipe, tool_name, completed_decisions, call_index):
    """Counter-based independent outcome RNG; callers persist the call index.

    The index counts ALL declared injections (model/human/forced) in one stream.
    Each draw depends only on frozen outcome seed and index, not generated token
    RNG, number of task records, wall time, or unrelated shuffle operations.
    Hidden mapping and draw remain observer-only; tool acknowledgments come from
    the tool definition and must not interpolate this result.
    """
    resolved = resolve_recipe(recipe)
    _integer(call_index, "call_index", 0, 10**12)
    phase = mapping_at(resolved, completed_decisions)
    if type(tool_name) is not str or tool_name not in phase["mappings"]:
        raise ValueError("Tool is not mapped in this recipe")
    outcomes = phase["mappings"][tool_name]
    key = f'opium-bench/outcome-v2/{resolved["rng_seeds"]["outcomes"]}/{call_index}'
    # A uniform 53-bit fraction has the same precision as Random.random, while
    # its immutable counter permits strict checkpointing without opaque state.
    draw = (int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") >> 11) / 2**53
    cumulative, selected = 0., outcomes[-1]["preset_id"]
    for outcome in outcomes:
        cumulative += outcome["probability"]
        if draw < cumulative:
            selected = outcome["preset_id"]
            break
    return {"tool": tool_name, "preset_id": selected, "draw": draw, "call_index": call_index,
            "phase": phase["label"], "phase_after_decisions": phase["after_decisions"],
            "outcome_seed": resolved["rng_seeds"]["outcomes"]}


def model_cost_notice(recipe):
    resolved = resolve_recipe(recipe)
    return cost_notice(resolved["auxiliary_tools"], base_cost=resolved["base_decision_cost"], task_costs=resolved["task_tool_costs"])


def recipe_hash(recipe):
    return content_hash(resolve_recipe(recipe))


def ordered_auxiliary_tools(recipe):
    """Exact model-visible auxiliary schemas under the dedicated order seed.

    Append the task family's existing work-tool schemas after this returned list.
    Hidden tools never enter the list or change the visible shuffle population.
    """
    resolved = resolve_recipe(recipe)
    tools = model_tools(resolved["auxiliary_tools"])
    if resolved["counterbalance"]:
        random.Random(resolved["rng_seeds"]["tool_order"]).shuffle(tools)
    return tools
