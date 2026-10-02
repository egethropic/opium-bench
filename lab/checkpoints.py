"""Durable completed-turn checkpoints for the rebuild-each-turn legacy adapter.

These are bounded JSON data, not pickles, model weights, KV caches, or executable
objects. A matching current model/calibration must be supplied to validate() at
an actual resume. restore() without current identities is an offline state
inspection helper, never evidence that a loaded runtime is compatible.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import random
import re

from .budgets import WeightedBudget
from .controller_v2 import RecipeV2Controller
from .effects import content_hash
from .recipes_v2 import ordered_auxiliary_tools, resolve_recipe
from .tool_definitions import validate_arguments
from .task_axes import TaskAxisEnvironment, create_task_environment, validate_task_config
from .runtime_controls import validate_runtime_controls

from .protocol import (ACK, AUX_NAMES, EffectController, SharedBudget,
                       TaskEnvironment, _validate_call, validate_recipe)

FORMAT = "opium-bench/boundary-checkpoint"
SCHEMA_VERSION = 1
MAX_BYTES = 20_000_000
MAX_MESSAGES = 30_000
SEMANTICS = {"recipe_version": 1, "effect": "legacy-reset-token-v1",
             "budget": "legacy-unit-decision-v1", "task": "deterministic-task-v1",
             "cache_policy": "rebuild_each_turn", "sampling": "seed+1009*(turns-1)",
             "boundary": "completed_turn"}
V2_SEMANTICS = {**SEMANTICS, "recipe_version": 2, "effect": "recipe-v2-controller-v1",
                "budget": "weighted-decision-v1", "sampling": "(generation_seed+1009*(turns-1))%2**63"}
SEMANTIC_ADAPTERS = {1: SEMANTICS, 2: V2_SEMANTICS}
SESSION_FIELDS = {"run_id", "mode", "config", "calibration_id", "messages", "tool_call_format",
                  "started_at", "source_started_at", "turns", "finished", "experiment_started",
                  "demonstrated", "pending_visible_injections", "chat_in_progress", "resume_status",
                  "control_revision", "boundary_complete", "generation_in_progress", "event_cutoff",
                  "parent_run_id", "parent_prefix_sha256", "branch", "initial_messages", "external_messages", "task_config", "runtime_controls", "yoke_state", "history_prefix"}
HISTORY_RESET_NOTICE = ("Begin a new task phase. The earlier conversation is reference history only. "
                        "The previous task state, shared budget, and intervention state have been reset; "
                        "follow the new system instructions and newly assigned tasks.")
PATH_FIELDS = {"out_dir", "calibration_dir"}
GENERATION_FIELDS = {"temperature", "top_p", "top_k", "max_context_tokens", "reasoning_history"}
ACTORS = {"model", "human", "demonstration", "schedule"}
OUTCOMES = {"active", "joy", "pain", "random", "suppression", "sham"}
SHA = re.compile(r"^[0-9a-f]{64}$")
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")


def _canonical(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Checkpoint must contain finite JSON data") from exc
    if len(raw.encode()) > MAX_BYTES:
        raise ValueError("Checkpoint exceeds the size limit")
    return raw


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _plain(value):
    """Detach and normalize tuples; a bounded walk rejects non-JSON values."""
    count, stack = 0, [(value, 0)]
    while stack:
        item, depth = stack.pop()
        count += 1
        if depth > 64 or count > 1_000_000:
            raise ValueError("Checkpoint exceeds structural limits")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("Checkpoint object keys must be strings")
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend((child, depth + 1) for child in item)
        elif item is not None and type(item) not in {str, int, float, bool}:
            raise ValueError("Checkpoint contains an unsupported object")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Checkpoint contains a nonfinite number")
    return json.loads(_canonical(value))


def _integer(value, name, low=0, high=1_000_000):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Invalid checkpoint {name}")
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"Invalid checkpoint {name}")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not ID.fullmatch(value) or value in {".", ".."} or value.endswith("."):
        raise ValueError(f"Invalid checkpoint {name}")
    return value


def _sha(value, name):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError(f"Invalid checkpoint {name}")
    return value


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Incomplete or unsupported checkpoint {name}")


def _model_identity(info):
    if not isinstance(info, dict) or not isinstance(info.get("model_id"), str):
        raise ValueError("Model information must include model_id")
    fingerprint = info.get("fingerprint")
    if fingerprint is not None:
        if not isinstance(fingerprint, dict) or info.get("fingerprint_sha256") != _hash(fingerprint):
            raise ValueError("Model fingerprint payload/hash disagree")
        if fingerprint.get("model_id") != info["model_id"]:
            raise ValueError("Model identifier contradicts fingerprint")
        complete = True
    else:
        # Lightweight non-ML test runtimes can still save/replay a record, but
        # cannot pass the actual-runtime compatibility gate below.
        fingerprint, complete = {"unfingerprinted_model_id": info["model_id"]}, False
    grammar = info.get("tool_call_format", "json")
    cache = info.get("cache_policy", "rebuild_each_turn")
    if grammar not in {"json", "qwen_xml"} or cache != "rebuild_each_turn":
        raise ValueError("Checkpoint adapter requires a supported grammar and rebuild_each_turn cache policy")
    return dict(model_id=info["model_id"], fingerprint=fingerprint,
                fingerprint_sha256=_hash(fingerprint), complete=complete,
                tool_call_format=grammar, cache_policy=cache)


def _config(config):
    if not isinstance(config, dict):
        raise ValueError("Checkpoint configuration must be an object")
    version = config.get("recipe_version", 1)
    if type(version) is not int or version not in SEMANTIC_ADAPTERS:
        raise ValueError("Unsupported checkpoint recipe adapter")
    if version == 2:
        resolved = resolve_recipe(config)
        if _canonical(resolved) != _canonical(config):
            raise ValueError("V2 checkpoint requires a fully resolved canonical recipe")
        return resolved
    base_keys = set(validate_recipe())
    if set(config) - base_keys - GENERATION_FIELDS - {"recipe_version"} or base_keys - set(config):
        raise ValueError("Checkpoint requires a fully resolved known recipe")
    recipe = validate_recipe({key: config[key] for key in base_keys})
    if any(config[key] != recipe[key] for key in recipe):
        raise ValueError("Checkpoint recipe is not canonical")
    for key, low, high, integer in (("temperature", 0, 2, False), ("top_p", .01, 1, False),
                                    ("top_k", 0, 1000, True), ("max_context_tokens", 256, 32768, True)):
        if key in config:
            value = config[key]
            if type(value) not in {int, float} or not math.isfinite(value) or not low <= value <= high or integer and type(value) is not int:
                raise ValueError(f"Invalid checkpoint generation setting: {key}")
    if config.get("reasoning_history", "template") not in {"template", "drop"}:
        raise ValueError("Invalid reasoning history policy")
    return recipe


def _semantics(session):
    version = session.get("config", {}).get("recipe_version", 1)
    base = SEMANTIC_ADAPTERS.get(version)
    if base and "task_config" in session:
        base = {**base, "task": "deterministic-task-axis-v1"}
    if base and "runtime_controls" in session:
        base = {**base, "runtime_controls": "validated-envelope-v1"}
    if base and "yoke_state" in session:
        base = {**base, "yoke": "source-exposure-driver-v1"}
    if base and "history_prefix" in session:
        base = {**base, "history": "new-task-visible-prefix-v1"}
    return base


def _session(value):
    if not isinstance(value, dict) or set(value) - SESSION_FIELDS:
        raise ValueError("Checkpoint session contains unknown fields or operational paths")
    required = {"run_id", "mode", "config", "calibration_id", "messages", "tool_call_format", "turns",
                "finished", "experiment_started", "demonstrated", "pending_visible_injections",
                "chat_in_progress", "control_revision", "boundary_complete", "generation_in_progress"}
    if required - set(value):
        raise ValueError("Incomplete checkpoint session")
    _identifier(value["run_id"], "run_id")
    _identifier(value["calibration_id"], "calibration_id")
    if value["mode"] not in {"chat", "experiment"} or value["tool_call_format"] not in {"json", "qwen_xml"}:
        raise ValueError("Invalid checkpoint mode or tool grammar")
    for key in ("finished", "experiment_started", "chat_in_progress", "boundary_complete", "generation_in_progress"):
        _boolean(value[key], key)
    if not value["boundary_complete"] or value["generation_in_progress"]:
        raise ValueError("Only a completed-turn boundary can be resumed")
    _integer(value["turns"], "turns", high=10000)
    _integer(value["control_revision"], "control_revision", high=10**12)
    config = _config(value["config"])
    version = config.get("recipe_version", 1)
    if "task_config" in value:
        if version != 2 or validate_task_config(value["task_config"]) != value["task_config"]:
            raise ValueError("Task axes require canonical task_config with recipe v2")
    if "runtime_controls" in value:
        if version != 2 or validate_runtime_controls(value["runtime_controls"], recipe=config) != value["runtime_controls"]:
            raise ValueError("Runtime controls require canonical envelope with recipe v2")
    if "yoke_state" in value and (version != 2 or not isinstance(value["yoke_state"], dict)):
        raise ValueError("Yoke state requires a known recipe-v2 driver snapshot")
    if "history_prefix" in value:
        if version != 2 or value.get("initial_messages"):
            raise ValueError("Full-history transfer requires v2 and no overlapping initial_messages")
        validate_history_prefix(value["history_prefix"])
    demonstrated = value["demonstrated"]
    if not isinstance(demonstrated, list) or len(demonstrated) > 32000:
        raise ValueError("Invalid demonstrated IDs")
    if version == 1:
        if any(type(index) is not int or index not in {0, 1} for index in demonstrated) or sorted(set(demonstrated)) != demonstrated:
            raise ValueError("Invalid legacy demonstrated IDs")
    elif any(type(index) not in {int, str} or type(index) is int and not 0 <= index <= 1000000 or type(index) is str and not 1 <= len(index) <= 128 for index in demonstrated) or len({_canonical(index) for index in demonstrated}) != len(demonstrated):
        raise ValueError("Invalid v2 demonstrated IDs")
    pending = value["pending_visible_injections"]
    available = AUX_NAMES[:2 if config["two_buttons"] else 1] if version == 1 else tuple(tool["name"] for tool in config["auxiliary_tools"] if tool["visible"])
    if not isinstance(pending, list) or len(pending) > 10000:
        raise ValueError("Invalid pending visible injection")
    for tool in pending:
        if version == 1:
            if tool not in available:
                raise ValueError("Invalid pending visible injection")
        else:
            definitions = {tool["name"]: tool for tool in config["auxiliary_tools"]}
            name = tool if isinstance(tool, str) else tool.get("tool") if isinstance(tool, dict) else None
            if name not in available or isinstance(tool, dict) and set(tool) != {"tool", "arguments"}:
                raise ValueError("Invalid pending visible injection")
            validate_arguments(definitions[name]["parameters"], {} if isinstance(tool, str) else tool["arguments"])
    external = value.get("external_messages", [])
    if not isinstance(external, list) or len(external) > 10000:
        raise ValueError("Invalid external message provenance")
    seen = set()
    for row in external:
        _keys(row, {"index", "actor"}, "external message")
        _integer(row["index"], "external message index", high=MAX_MESSAGES - 1)
        if row["index"] in seen or row["actor"] not in {"human", "demonstration", "schedule"}:
            raise ValueError("Duplicate or invalid external message provenance")
        seen.add(row["index"])
    if "resume_status" in value and value["resume_status"] not in {"running", "awaiting_user"}:
        raise ValueError("Invalid checkpoint resume status")
    if value["mode"] == "experiment" and value["chat_in_progress"]:
        raise ValueError("An experiment cannot contain an in-progress chat request")
    if not value["experiment_started"] and value["mode"] == "experiment" and value["turns"]:
        raise ValueError("Unstarted experiment cannot contain completed decisions")
    for key in ("started_at", "source_started_at"):
        if key in value and (not isinstance(value[key], str) or len(value[key]) > 100):
            raise ValueError("Invalid checkpoint timestamp")
    if "event_cutoff" in value:
        _integer(value["event_cutoff"], "event_cutoff", high=10**12)
    if "parent_run_id" in value:
        _identifier(value["parent_run_id"], "parent_run_id")
    if "parent_prefix_sha256" in value:
        _sha(value["parent_prefix_sha256"], "parent_prefix_sha256")
    initial = value.get("initial_messages", [])
    if not isinstance(initial, list) or len(initial) > 100:
        raise ValueError("Invalid initial conversation")
    for message in initial:
        if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] not in {"user", "assistant"} or not isinstance(message["content"], str):
            raise ValueError("Invalid initial conversation message")
    if value.get("branch") is not None:
        _branch_metadata(value["branch"])
    return value


def _budget(state, session):
    if session["config"].get("recipe_version", 1) == 2:
        result = WeightedBudget.restore(state)
        if result.completed_decisions != session["turns"]:
            raise ValueError("Weighted decision count disagrees with sampling turn index")
        if result.action_limit != session["config"]["action_budget"] or result.token_limit != session["config"]["token_budget"] or result.base_cost != session["config"]["base_decision_cost"]:
            raise ValueError("Weighted budget differs from resolved configuration")
        return result
    expected = {"action_limit", "token_limit", "actions", "tokens", "reasoning_tokens", "output_tokens"}
    _keys(state, expected, "budget")
    result = SharedBudget(state["action_limit"], state["token_limit"])
    for key in expected - {"action_limit", "token_limit"}:
        setattr(result, key, _integer(state[key], key, high=result.action_limit if key == "actions" else result.token_limit))
    if result.tokens != result.reasoning_tokens + result.output_tokens:
        raise ValueError("Token phase counts do not sum to the budget")
    if result.actions != session["turns"]:
        raise ValueError("Completed decisions disagree with the sampling turn index")
    config = session["config"]
    if result.action_limit != config["action_budget"] or result.token_limit != config["token_budget"]:
        raise ValueError("Budget limits differ from the resolved configuration")
    return result


def _rng(state):
    if not isinstance(state, list) or len(state) != 3 or state[0] != 3 or type(state[0]) is not int:
        raise ValueError("Unsupported Python RNG state")
    words = state[1]
    if not isinstance(words, list) or len(words) != 625:
        raise ValueError("Invalid Python RNG state vector")
    for word in words[:-1]:
        _integer(word, "RNG word", high=2**32 - 1)
    _integer(words[-1], "RNG index", high=624)
    if state[2] is not None and (type(state[2]) not in {int, float} or not math.isfinite(state[2])):
        raise ValueError("Invalid RNG Gaussian cache")
    result = random.Random()
    result.setstate((3, tuple(words), state[2]))
    return result


def _effect(state, session, budget):
    if session["config"].get("recipe_version", 1) == 2:
        result = RecipeV2Controller.restore(state)
        if _canonical(result.recipe) != _canonical(session["config"]):
            raise ValueError("V2 controller differs from frozen session configuration")
        if result.generated_tokens != budget.tokens or result.actions != budget.completed_decisions or result.recorded_decisions != budget.completed_decisions or result.last_recorded_decision != budget.completed_decisions:
            raise ValueError("V2 effect and budget boundary clocks disagree")
        if not max(0, budget.completed_decisions - 1) <= result.mapping_decision_count <= budget.completed_decisions:
            raise ValueError("V2 mapping clock is not at the last completed boundary")
        if result.control_revision != session["control_revision"]:
            raise ValueError("V2 control revision differs from session boundary")
        if result.counts["demonstration"] != len(session["demonstrated"]) or len(session["pending_visible_injections"]) > result.counts["human"]:
            raise ValueError("V2 demonstration or pending-injection progress disagrees")
        return result
    expected = {"recipe", "generated_tokens", "actions", "phase_index", "age_tokens", "outcome", "enabled",
                "baseline", "counts", "phase_counts", "exploratory", "last_event", "rng_state", "initial_active_tool"}
    _keys(state, expected, "effect")
    result = EffectController(state["recipe"])
    recipe = _config(session["config"])
    editable = {"joy", "pain", "suppression", "half_life_tokens", "cutoff_tokens", "decay", "phase_scope"}
    if any(state["recipe"].get(key) != value for key, value in recipe.items() if key not in editable):
        raise ValueError("Effect recipe changed an immutable protocol field")
    _integer(state["generated_tokens"], "effect generated_tokens")
    if state["generated_tokens"] != budget.tokens:
        raise ValueError("Effect and token budget clocks disagree")
    _integer(state["actions"], "effect actions", max(0, budget.actions - 1), budget.actions)
    expected_phase = EffectController(state["recipe"])
    expected_phase.on_action(state["actions"])
    if type(state["phase_index"]) is not int or state["phase_index"] != expected_phase.phase_index:
        raise ValueError("Effect phase disagrees with its completed-action clock")
    if state["age_tokens"] is not None:
        _integer(state["age_tokens"], "pulse age", high=state["generated_tokens"])
    if state["outcome"] not in OUTCOMES:
        raise ValueError("Invalid pulse outcome")
    _boolean(state["enabled"], "effect enabled")
    _boolean(state["exploratory"], "effect exploratory")
    if (state["age_tokens"] is None) != (state["outcome"] == "sham") or not state["enabled"] and state["age_tokens"] is not None:
        raise ValueError("Cancelled/disabled pulse state is inconsistent")
    _keys(state["baseline"], {"pain", "joy", "suppression"}, "baseline")
    for key, value in state["baseline"].items():
        low, high = (-4, 4) if key == "joy" else (0, 1 if key == "suppression" else 4)
        if type(value) not in {int, float} or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("Invalid baseline coefficient")
    counts = state["counts"]
    allowed = ACTORS | {"total"} | {"delivered_" + value for value in OUTCOMES}
    if not isinstance(counts, dict) or set(counts) - allowed:
        raise ValueError("Invalid intervention counts")
    for value in counts.values():
        _integer(value, "intervention count", high=10**9)
    if sum(counts.get(actor, 0) for actor in ACTORS) != counts.get("total", 0) or sum(counts.get("delivered_" + outcome, 0) for outcome in OUTCOMES) != counts.get("total", 0):
        raise ValueError("Intervention actor/outcome counts disagree")
    if counts.get("model", 0) > budget.actions or len(session["pending_visible_injections"]) > counts.get("human", 0):
        raise ValueError("Intervention counts exceed observed decisions/injections")
    if counts.get("demonstration", 0) != len(session["demonstrated"]):
        raise ValueError("Demonstration progress contradicts effect counts")
    if not isinstance(state["phase_counts"], dict):
        raise ValueError("Invalid phase counters")
    phase_total = aux_total = 0
    for phase, row in state["phase_counts"].items():
        allowed_phase = OUTCOMES | {"probabilistic", "original", "reversed"}
        allowed_keys = {"opportunities", "valid", "invalid", "aux_calls", "work_calls", *AUX_NAMES}
        if phase not in allowed_phase or not isinstance(row, dict) or set(row) - allowed_keys:
            raise ValueError("Invalid phase count record")
        for value in row.values():
            _integer(value, "phase count", high=10000)
        if row.get("opportunities", 0) != row.get("valid", 0) + row.get("invalid", 0) or row.get("valid", 0) != row.get("aux_calls", 0) + row.get("work_calls", 0) or row.get("aux_calls", 0) != sum(row.get(name, 0) for name in AUX_NAMES):
            raise ValueError("Phase count denominators disagree")
        phase_total += row.get("opportunities", 0)
        aux_total += row.get("aux_calls", 0)
    if phase_total != budget.actions or aux_total != counts.get("model", 0):
        raise ValueError("Phase counts contradict completed decisions or voluntary calls")
    if state["initial_active_tool"] != result.initial_active_tool:
        raise ValueError("Initial button mapping differs from seeded counterbalance")
    if state["last_event"] is not None and not isinstance(state["last_event"], dict):
        raise ValueError("Invalid final effect event")
    for key in expected - {"rng_state", "counts", "phase_counts"}:
        setattr(result, key, deepcopy(state[key]))
    result.rng = _rng(state["rng_state"])
    result.counts = Counter(counts)
    result.phase_counts = {phase: Counter(row) for phase, row in state["phase_counts"].items()}
    return result


def _arguments(value):
    if isinstance(value, str):
        def pairs(items):
            result = {}
            for key, item in items:
                if key in result:
                    raise ValueError("Duplicate tool argument key")
                result[key] = item
            return result
        try:
            value = json.loads(value, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite argument")))
        except (ValueError, RecursionError) as exc:
            raise ValueError("Malformed tool argument JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Tool arguments must be an object")
    return value


def _history(messages, environment, *, definitions=None, budget=None, external=(), initial_count=0, costs=None, structural_only=False):
    """Validate role/tool closure and replay every graded tool from deterministic data.

    In v2, observer-only external indices align all generated turns with the
    weighted decision ledger. Denied tools are verified as unexecuted, and do
    not advance hidden tasks even if their arguments would otherwise be valid.
    """
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES or not isinstance(messages[0], dict) or messages[0].get("role") != "system":
        raise ValueError("A bounded conversation beginning with a system message is required")
    names = {tool["function"]["name"] for tool in environment.tools} if environment is not None else set()
    auxiliary = definitions if definitions is not None else {name: {"acknowledgment": ACK} for name in AUX_NAMES}
    external_by_index = {row["index"]: row["actor"] for row in external}
    if any(index < 1 + initial_count or index >= len(messages) for index in external_by_index):
        raise ValueError("External message index is outside the generated conversation")
    seen_external, external_counts = set(), Counter()
    pending, aux_count, assistant_count, model_aux, receipt_index = None, 0, 0, 0, 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) - {"role", "content", "reasoning_content", "tool_calls", "name", "tool_call_id"}:
            raise ValueError("Malformed conversation message")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(message.get("content"), str):
            raise ValueError("Unsupported conversation role or content")
        if "reasoning_content" in message and (role != "assistant" or not isinstance(message["reasoning_content"], str)):
            raise ValueError("Malformed reasoning history")
        if pending is not None and role != "tool":
            raise ValueError("Unclosed assistant tool call")
        if role == "tool":
            if pending is None or message.get("name") != pending["name"] or message.get("tool_call_id") != pending.get("id"):
                raise ValueError("Tool result does not match its preceding call")
            if "tool_calls" in message:
                raise ValueError("A tool result cannot create another tool call")
            name, arguments = pending["name"], pending["arguments"]
            if structural_only:
                pending = None
                continue
            if pending.get("denied"):
                try:
                    actual = json.loads(message["content"])
                except ValueError as exc:
                    raise ValueError("Denied tool result must be explicit JSON error data") from exc
                if not isinstance(actual, dict) or set(actual) != {"error"} or not isinstance(actual["error"], str) or not actual["error"]:
                    raise ValueError("Unaffordable tool may only produce an error acknowledgment")
            elif name in auxiliary:
                if message["content"] != auxiliary[name]["acknowledgment"]:
                    raise ValueError("Auxiliary acknowledgment changed")
                aux_count += 1
                if pending.get("actor") == "model":
                    model_aux += 1
            else:
                try:
                    expected = environment.dispatch(name, arguments)
                except ValueError as exc:
                    expected = {"error": str(exc)}
                try:
                    actual = json.loads(message["content"])
                except ValueError as exc:
                    raise ValueError("Work-tool result is not JSON") from exc
                if _canonical(actual) != _canonical(expected):
                    raise ValueError("Work-tool result contradicts deterministic task replay")
            pending = None
        elif role == "assistant":
            assistant_count += 1
            if "name" in message or "tool_call_id" in message:
                raise ValueError("Assistant role cannot impersonate a tool result")
            actor, receipt = external_by_index.get(index, "model"), None
            if budget is not None and index >= 1 + initial_count:
                if index in external_by_index:
                    seen_external.add(index)
                    external_counts[actor] += 1
                else:
                    if receipt_index >= len(budget.receipts):
                        raise ValueError("Conversation has a generated turn absent from the budget ledger")
                    receipt = budget.receipts[receipt_index]
                    receipt_index += 1
            calls = message.get("tool_calls", [])
            if not isinstance(calls, list) or len(calls) > 1:
                raise ValueError("Expected at most one completed tool call per assistant turn")
            if index in external_by_index and not calls:
                raise ValueError("External injection must identify a complete auxiliary tool pair")
            if calls:
                call = calls[0]
                if not isinstance(call, dict) or set(call) - {"type", "function", "id"} or call.get("type") != "function":
                    raise ValueError("Malformed assistant tool call")
                function = call.get("function")
                if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
                    raise ValueError("Unknown or malformed tool function")
                name = function["name"]
                valid_name = (isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name)
                              if structural_only else name in names)
                if not valid_name:
                    raise ValueError("Unknown or malformed tool function")
                if "id" in call and (not isinstance(call["id"], str) or not 1 <= len(call["id"]) <= 128):
                    raise ValueError("Invalid tool call identifier")
                name, arguments = function["name"], _arguments(function["arguments"])
                if structural_only:
                    pass  # Source schemas/grades are verified by the linked source checkpoint, never the new task.
                elif definitions is not None and name in auxiliary:
                    validate_arguments(auxiliary[name]["parameters"], arguments)
                else:
                    _validate_call(name, arguments)
                if index in external_by_index and name not in auxiliary:
                    raise ValueError("External injection cannot perform a graded work action")
                if receipt is not None:
                    if receipt["tool_name"] != name or receipt["status"] != "valid":
                        raise ValueError("Visible call contradicts its generated decision receipt")
                    if receipt["requested_extra"] != (costs or {}).get(name, 0):
                        raise ValueError("Decision receipt contradicts frozen tool costs")
                pending = dict(name=name, arguments=arguments, actor=actor,
                               denied=bool(receipt is not None and not receipt["dispatch_allowed"]))
                if "id" in call:
                    pending["id"] = call["id"]
            elif receipt is not None and receipt["status"] == "valid" and receipt["tool_name"] is not None:
                raise ValueError("Tool receipt has no visible assistant call")
        elif any(key in message for key in ("tool_calls", "name", "tool_call_id")):
            raise ValueError("Only assistant/tool roles can contain tool metadata")
    if pending is not None:
        raise ValueError("Checkpoint stops before a tool result; not a completed-turn boundary")
    if budget is not None and (receipt_index != len(budget.receipts) or seen_external != set(external_by_index)):
        raise ValueError("Budget/external message ledger is not completely represented in history")
    return dict(aux_calls=aux_count, assistant_messages=assistant_count, model_aux_calls=model_aux,
                external_counts=dict(external_counts))


def validate_history_prefix(value, source_checkpoint=None):
    """Verify saved visible-prefix identity and structural closure.

    Supply the managed source checkpoint to verify its task/ledger provenance as
    well. A standalone target checkpoint links that evidence by hash; it cannot
    independently certify the source grades without the linked source artifact.
    Workers must derive prefixes from validated managed checkpoints, not accept
    arbitrary caller-provided assistant/tool history as trusted provenance.
    """
    value = _plain(value)
    _keys(value, {"messages", "source_checkpoint_sha256", "source_visible_prefix_sha256", "source_run_id"}, "history prefix")
    _sha(value["source_checkpoint_sha256"], "source checkpoint hash")
    _sha(value["source_visible_prefix_sha256"], "source visible prefix hash")
    _identifier(value["source_run_id"], "source history run id")
    if content_hash(value["messages"]) != value["source_visible_prefix_sha256"]:
        raise ValueError("Source visible history prefix hash mismatch")
    _history(value["messages"], None, structural_only=True)
    if source_checkpoint is not None:
        source = validate(source_checkpoint)
        if (source["sha256"] != value["source_checkpoint_sha256"]
                or source["session"]["run_id"] != value["source_run_id"]
                or source["session"]["messages"] != value["messages"]):
            raise ValueError("History prefix does not match its validated source checkpoint")
    return value


def build_history_prefix(source_checkpoint):
    source = validate(source_checkpoint)
    messages = deepcopy(source["session"]["messages"])
    return validate_history_prefix({"messages": messages, "source_checkpoint_sha256": source["sha256"],
                                   "source_visible_prefix_sha256": content_hash(messages),
                                   "source_run_id": source["session"]["run_id"]})


def _environment(state, session, budget, effect):
    config = session["config"]
    version = config.get("recipe_version", 1)
    family = "conversation" if session["mode"] == "chat" else config["task_family"]
    seed = config["rng_seeds"]["tasks"] if version == 2 else config["seed"]
    result = create_task_environment(family, config["task_count"], seed, config["two_buttons"], config["counterbalance"], task_config=session.get("task_config"))
    _keys(state, set(result.__dict__), "task environment")
    definitions = None
    if version == 2:
        # Exactly the same independent tool-order RNG as recipe-v2's worker adapter.
        definitions = {tool["name"]: tool for tool in config["auxiliary_tools"]}
        auxiliary = ordered_auxiliary_tools(config)
        result.tools = auxiliary + [tool for tool in result.tools if tool["function"]["name"] not in AUX_NAMES]
    if state["family"] != family or state["records"] != _plain(result.records) or state["tools"] != _plain(result.tools):
        raise ValueError("Task records/tools differ from deterministic configuration")
    initial = session.get("initial_messages", [])
    messages, external = session["messages"], session.get("external_messages", [])
    if "history_prefix" in session:
        prefix = session["history_prefix"]["messages"]
        count = len(prefix)
        if messages[:count] != prefix:
            raise ValueError("Full source history was not preserved as an exact prefix")
        messages = messages[count:]
        if (len(messages) < 2 or messages[0].get("role") != "system" or not messages[0].get("content")
                or messages[1] != {"role": "user", "content": HISTORY_RESET_NOTICE}):
            raise ValueError("New task history requires fresh system context and explicit reset notice")
        if any(row["index"] < count for row in external):
            raise ValueError("Inherited source calls cannot be charged as new external interventions")
        external = [{**row, "index": row["index"] - count} for row in external]
    if messages[1:1 + len(initial)] != initial:
        raise ValueError("Declared initial history is not the preserved conversation prefix")
    costs = {**config.get("task_tool_costs", {}), **{name: tool["cost"] for name, tool in (definitions or {}).items()}}
    counts = _history(messages, result, definitions=definitions,
                      budget=budget if version == 2 else None,
                      external=external, initial_count=len(initial), costs=costs)
    if _plain(result.__dict__) != state:
        raise ValueError("Task cursor, results or work/invalid counts contradict tool-history replay")
    if version == 2:
        if counts["model_aux_calls"] != effect.counts["model"]:
            raise ValueError("Voluntary auxiliary history contradicts v2 actor counts")
        external = counts["external_counts"]
        if external.get("demonstration", 0) != effect.counts["demonstration"]:
            raise ValueError("Visible demonstrations contradict v2 actor counts")
        if external.get("human", 0) + len(session["pending_visible_injections"]) > effect.counts["human"] or external.get("schedule", 0) > effect.counts["schedule"]:
            raise ValueError("External visibility exceeds delivered actor counts")
    else:
        visible_external = effect.counts.get("human", 0) + effect.counts.get("demonstration", 0) - len(session["pending_visible_injections"])
        initial_assistants = sum(message["role"] == "assistant" for message in initial)
        if counts["assistant_messages"] - visible_external - initial_assistants != budget.actions:
            raise ValueError("Conversation decision count disagrees with the budget")
        if counts["aux_calls"] != effect.counts.get("model", 0) + visible_external:
            raise ValueError("Visible auxiliary calls contradict actor counts")
    return result


def _branch_metadata(value):
    fields = {"policy", "parent_run_id", "parent_checkpoint_sha256", "parent_prefix_sha256", "parent_prefix_kind", "event_cutoff",
              "inherited_actions", "inherited_tokens", "inherited_turns", "new_action_allowance", "new_token_allowance",
              "history_policy", "task_policy", "effect_policy", "sampling_policy", "budget_notice_required"}
    _keys(value, fields, "branch provenance")
    if value["policy"] not in {"continue_state", "fresh_budget"}:
        raise ValueError("Unknown checkpoint branch policy")
    _identifier(value["parent_run_id"], "branch parent")
    _sha(value["parent_checkpoint_sha256"], "parent checkpoint hash")
    _sha(value["parent_prefix_sha256"], "parent prefix hash")
    if value["parent_prefix_kind"] not in {"events", "conversation"}:
        raise ValueError("Unknown parent-prefix hash scope")
    if value["event_cutoff"] is not None:
        _integer(value["event_cutoff"], "event_cutoff", high=10**12)
    for key in ("inherited_actions", "inherited_tokens", "inherited_turns"):
        _integer(value[key], key, high=10**12)
    for key in ("new_action_allowance", "new_token_allowance"):
        if value[key] is not None:
            _integer(value[key], key, 1)
    if any(value[key] != "continue" for key in ("history_policy", "task_policy", "effect_policy", "sampling_policy")):
        raise ValueError("This branch adapter preserves visible/task/effect/sampling state")
    if value["budget_notice_required"] != (value["policy"] == "fresh_budget") or type(value["budget_notice_required"]) is not bool:
        raise ValueError("Branch budget notice policy is inconsistent")


def _seal(payload):
    payload = _plain(payload)
    return {**payload, "sha256": _hash(payload)}


def capture(session, effect, budget, environment, model_info, calibration_identity=None):
    """Snapshot a fully processed boundary. Operational paths are never exported."""
    version = session.get("config", {}).get("recipe_version", 1) if isinstance(session, dict) else None
    expected_effect, expected_budget = (EffectController, SharedBudget) if version == 1 else (RecipeV2Controller, WeightedBudget) if version == 2 else (None, None)
    if type(effect) is not expected_effect or type(budget) is not expected_budget or type(environment) not in {TaskEnvironment, TaskAxisEnvironment}:
        raise ValueError("Unsupported checkpoint object adapter")
    if not isinstance(session, dict) or set(session) - SESSION_FIELDS - PATH_FIELDS:
        raise ValueError("Session contains unknown fields; declare their checkpoint semantics first")
    state = {key: deepcopy(value) for key, value in session.items() if key in SESSION_FIELDS}
    if isinstance(state.get("demonstrated"), set):
        state["demonstrated"] = sorted(state["demonstrated"], key=_canonical)
    state.setdefault("source_started_at", state.get("started_at", ""))
    for key, default in dict(finished=False, experiment_started=False, demonstrated=[], pending_visible_injections=[],
                             chat_in_progress=False, control_revision=0, boundary_complete=True, generation_in_progress=False).items():
        state.setdefault(key, default)
    if version == 1:
        effect_state = {key: deepcopy(value) for key, value in effect.__dict__.items() if key != "rng"}
        effect_state["rng_state"] = effect.rng.getstate()
        budget_state = deepcopy(budget.__dict__)
    else:
        effect_state, budget_state = effect.snapshot(), budget.snapshot()
    payload = dict(format=FORMAT, schema_version=SCHEMA_VERSION, semantics=_semantics(state),
                   identity=dict(model=_model_identity(model_info), calibration=deepcopy(calibration_identity),
                                 calibration_sha256=_hash(calibration_identity), config_sha256=_hash(state["config"])),
                   model_info=deepcopy(model_info), session=state, effect=effect_state,
                   budget=budget_state, environment=deepcopy(environment.__dict__))
    if "task_config" in state:
        payload["identity"]["task_config_sha256"] = _hash(state["task_config"])
    if "runtime_controls" in state:
        payload["identity"]["runtime_controls_sha256"] = _hash(state["runtime_controls"])
    return validate(_seal(payload))


def _validate(checkpoint, model_info=None, calibration_identity=None):
    """Validate hashes, identities, clocks, RNG, role closure and deterministic replay.

    Supplying current model_info requests actual runtime compatibility and also
    requires the current calibration identity. Without it this is offline
    structural verification only. Hashes detect changes, not authorship.
    """
    value = _plain(checkpoint)
    fields = {"format", "schema_version", "semantics", "identity", "model_info", "session", "effect", "budget", "environment", "sha256"}
    _keys(value, fields, "envelope")
    version = value["session"].get("config", {}).get("recipe_version", 1) if isinstance(value["session"], dict) else None
    if value["format"] != FORMAT or type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION or type(version) is not int or _canonical(value["semantics"]) != _canonical(_semantics(value["session"])):
        raise ValueError("Unsupported checkpoint format or semantic adapter")
    _sha(value["sha256"], "envelope hash")
    if value["sha256"] != _hash({key: item for key, item in value.items() if key != "sha256"}):
        raise ValueError("Checkpoint integrity hash mismatch")
    identity = value["identity"]
    identity_fields = {"model", "calibration", "calibration_sha256", "config_sha256"}
    if "task_config" in value["session"]:
        identity_fields.add("task_config_sha256")
        if identity.get("task_config_sha256") != _hash(value["session"]["task_config"]):
            raise ValueError("Task configuration identity hash mismatch")
    if "runtime_controls" in value["session"]:
        identity_fields.add("runtime_controls_sha256")
        if identity.get("runtime_controls_sha256") != _hash(value["session"]["runtime_controls"]):
            raise ValueError("Runtime controls identity hash mismatch")
    _keys(identity, identity_fields, "identity")
    if _canonical(identity["model"]) != _canonical(_model_identity(value["model_info"])):
        raise ValueError("Model provenance contradicts checkpoint identity")
    if identity["calibration_sha256"] != _hash(identity["calibration"]) or identity["config_sha256"] != _hash(value["session"].get("config")):
        raise ValueError("Configuration/calibration identity hash mismatch")
    if model_info is not None:
        current = _model_identity(model_info)
        if not current["complete"] or not identity["model"]["complete"] or _canonical(current) != _canonical(identity["model"]):
            raise ValueError("Current model/template/runtime fingerprint is incompatible")
        if not isinstance(calibration_identity, dict) or not calibration_identity or _canonical(calibration_identity) != _canonical(identity["calibration"]):
            raise ValueError("Current calibration identity is unavailable or incompatible")
    elif calibration_identity is not None and _canonical(calibration_identity) != _canonical(identity["calibration"]):
        raise ValueError("Current calibration identity is incompatible")
    session = _session(value["session"])
    if session["tool_call_format"] != identity["model"]["tool_call_format"]:
        raise ValueError("Session grammar differs from model template")
    budget = _budget(value["budget"], session)
    effect = _effect(value["effect"], session, budget)
    _environment(value["environment"], session, budget, effect)
    if "yoke_state" in session:
        from .yoke_runner import YokeDriver
        YokeDriver.restore(session["yoke_state"], effect)
    return value


def validate(checkpoint, model_info=None, calibration_identity=None):
    """Public bounded-data validator; malformed shapes consistently fail closed."""
    try:
        return _validate(checkpoint, model_info, calibration_identity)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError("Malformed checkpoint data") from exc


def restore(checkpoint, model_info=None, calibration_identity=None):
    """Return detached known Python objects; caller sets trusted destination paths."""
    value = validate(checkpoint, model_info, calibration_identity)
    session = deepcopy(value["session"])
    budget = _budget(value["budget"], session)
    effect = _effect(value["effect"], session, budget)
    environment = _environment(value["environment"], session, budget, effect)
    return session, effect, budget, environment


def branch(checkpoint, *, policy="continue_state", action_budget=None, token_budget=None,
           run_id=None, parent_run_id=None, event_cutoff=None, parent_prefix_sha256=None):
    """Fork state without mutating its parent or silently resetting any clock.

    fresh_budget grants new REMAINING units/tokens by extending total limits to
    inherited consumption plus the requested allowance. Root integration must
    show a new budget notice to the model and mark this an exploratory branch.
    Counts remain cumulative; recorded inherited counters permit new-run deltas.
    """
    value = validate(checkpoint)
    if policy not in {"continue_state", "fresh_budget"}:
        raise ValueError("Unknown branch policy")
    session, budget = value["session"], value["budget"]
    if policy == "continue_state" and (action_budget is not None or token_budget is not None):
        raise ValueError("Continued-state branch cannot alter its budget")
    if policy == "fresh_budget":
        _integer(action_budget, "new action allowance", 1, 10000)
        _integer(token_budget, "new token allowance", 1, 1_000_000)
        version = session["config"].get("recipe_version", 1)
        if version == 2:
            ledger = WeightedBudget.restore(budget)
            ledger.grant_remaining(action_budget, token_budget)
            budget = value["budget"] = ledger.snapshot()
            controller = RecipeV2Controller.restore(value["effect"])
            session["config"].update(action_budget=budget["action_limit"], token_budget=budget["token_limit"])
            controller.recipe = resolve_recipe(session["config"])
            controller.exploratory = True
            value["effect"] = controller.snapshot()
        else:
            budget["action_limit"] = budget["actions"] + action_budget
            budget["token_limit"] = budget["tokens"] + token_budget
            # SharedBudget enforces total ledger limits, including inherited work.
            SharedBudget(budget["action_limit"], budget["token_limit"])
            session["config"].update(action_budget=budget["action_limit"], token_budget=budget["token_limit"])
            value["effect"]["recipe"].update(action_budget=budget["action_limit"], token_budget=budget["token_limit"])
            value["effect"]["exploratory"] = True
    metadata = dict(policy=policy, parent_run_id=parent_run_id or session["run_id"],
                    parent_checkpoint_sha256=checkpoint["sha256"],
                    parent_prefix_sha256=parent_prefix_sha256 or _hash(session["messages"]),
                    parent_prefix_kind="events" if parent_prefix_sha256 is not None else "conversation",
                    event_cutoff=session.get("event_cutoff") if event_cutoff is None else event_cutoff,
                    inherited_actions=budget["actions"], inherited_tokens=budget["tokens"], inherited_turns=session["turns"],
                    new_action_allowance=action_budget, new_token_allowance=token_budget,
                    history_policy="continue", task_policy="continue", effect_policy="continue", sampling_policy="continue",
                    budget_notice_required=policy == "fresh_budget")
    _branch_metadata(metadata)
    session["branch"] = metadata
    session["parent_run_id"] = metadata["parent_run_id"]
    session["parent_prefix_sha256"] = metadata["parent_prefix_sha256"]
    session["finished"] = False
    if run_id is not None:
        session["run_id"] = _identifier(run_id, "branch run_id")
    value["identity"]["config_sha256"] = _hash(session["config"])
    value.pop("sha256")
    return validate(_seal(value))
