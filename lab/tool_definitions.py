"""Bounded, versioned auxiliary tools, never host-code execution.

The model-visible schema, acknowledgment and published cost are independent of
hidden preset assignment. A cost is EXTRA units above the recipe's base decision
cost. Arguments are bounded data; they do not interpolate scripts or arbitrary
coefficients. All names dispatch to the fixed effect-injection operation only.
The actual tokenizer chat template remains the authority for surrounding prompt
text; preview accepts that renderer when a model is loaded.
"""
from __future__ import annotations

import copy
import json
import math
import re

from .effects import canonical_json, content_hash

MAX_DEPTH = 4
MAX_PROPERTIES = 16
MAX_ARGUMENT_BYTES = 16384
TASK_NAMES = ("read_order", "calculate_total", "submit_answer", "read_puzzle")
NAME_PATTERN = r"[A-Za-z_][A-Za-z0-9_]{0,63}"
MARKERS = ("<tool_call", "</tool_call", "<function=", "</function", "<parameter=", "</parameter", "<think>", "</think>", "<|im_start|>", "<|im_end|>")


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


def _name(value, name="tool name"):
    if not isinstance(value, str) or not re.fullmatch(NAME_PATTERN, value):
        raise ValueError(f"Invalid {name}")
    return value


def _text(value, name, maximum=4096, empty=True):
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value) or "\x00" in value:
        raise ValueError(f"Invalid {name}")
    return value


def _object(value, allowed, name):
    if not isinstance(value, dict) or any(type(k) is not str for k in value) or set(value) - set(allowed):
        raise ValueError(f"Invalid or unknown {name} fields")


def validate_schema(schema, *, root=True, _depth=0):
    """Bounded JSON Schema subset; defaults are explicit in resolved exports.

    Objects have <=16 properties, no additional properties and no references;
    arrays have <=32 items; strings <=2048 codepoints; numeric magnitudes <=1e12.
    Supported scalar types are string, integer, number, boolean and null, plus
    object/array. No unions, coercion, regexes, combinators or executable fields.
    Enums contain <=32 unique values of the declared scalar type. Unknown
    constructs fail instead of being silently ignored.
    """
    if root and len(canonical_json(schema).encode("utf-8")) > 65536:
        raise ValueError("Argument schema exceeds the bounded JSON size")
    if _depth > MAX_DEPTH:
        raise ValueError("Argument schema nesting is too deep")
    _object(schema, ("type", "description", "properties", "required", "additionalProperties",
                     "items", "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum", "enum"), "schema")
    kind = schema.get("type")
    if type(kind) is not str or kind not in {"object", "array", "string", "integer", "number", "boolean", "null"} or (root and kind != "object"):
        raise ValueError("Arguments require a supported explicit schema type (object at root)")
    allowed = {"type", "description", "enum"}
    if kind == "object":
        allowed = {"type", "description", "properties", "required", "additionalProperties"}
    elif kind == "array":
        allowed = {"type", "description", "items", "minItems", "maxItems"}
    elif kind == "string":
        allowed |= {"minLength", "maxLength"}
    elif kind in ("integer", "number"):
        allowed |= {"minimum", "maximum"}
    if set(schema) - allowed:
        raise ValueError("Schema keyword does not apply to its declared type")
    result = {"type": kind}
    if "description" in schema:
        result["description"] = _text(schema["description"], "schema description", 1024)
    if kind == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, dict) or len(properties) > MAX_PROPERTIES:
            raise ValueError("Object properties must be a bounded object")
        result["properties"] = {_name(key, "property name"): validate_schema(value, root=False, _depth=_depth + 1) for key, value in properties.items()}
        required = schema.get("required", [])
        if not isinstance(required, list) or any(type(k) is not str or k not in properties for k in required) or len(required) != len(set(required)):
            raise ValueError("Required arguments must name distinct declared properties")
        result["required"] = list(required)
        if schema.get("additionalProperties", False) is not False:
            raise ValueError("Additional properties are forbidden")
        result["additionalProperties"] = False
    elif kind == "array":
        if "items" not in schema:
            raise ValueError("Arrays need an items schema")
        result["items"] = validate_schema(schema["items"], root=False, _depth=_depth + 1)
        result["minItems"] = _integer(schema.get("minItems", 0), "minItems", 0, 32)
        result["maxItems"] = _integer(schema.get("maxItems", 32), "maxItems", result["minItems"], 32)
    elif kind == "string":
        result["minLength"] = _integer(schema.get("minLength", 0), "minLength", 0, 2048)
        result["maxLength"] = _integer(schema.get("maxLength", 2048), "maxLength", result["minLength"], 2048)
    elif kind in ("integer", "number"):
        for key, default in (("minimum", -10**12), ("maximum", 10**12)):
            value = schema.get(key, default)
            if type(value) not in (int, float) or abs(value) > 10**12 or not math.isfinite(value) or kind == "integer" and type(value) is not int:
                raise ValueError("Numeric schema bounds must have a finite compatible type")
            result[key] = value
        if result["minimum"] > result["maximum"]:
            raise ValueError("Numeric schema bounds are reversed")
    if "enum" in schema:
        values = schema["enum"]
        if not isinstance(values, list) or not 1 <= len(values) <= 32 or len({canonical_json(v) for v in values}) != len(values):
            raise ValueError("enum must contain 1–32 distinct scalar values")
        for value in values:
            _validate_value(result, value, "enum")
        result["enum"] = copy.deepcopy(values)
    if root and _minimum_argument_bytes(result) > MAX_ARGUMENT_BYTES:
        raise ValueError("Required argument shape cannot fit the argument byte limit")
    return result


def _minimum_argument_bytes(schema):
    """Bound preview expansion before allocating a nested example payload."""
    if "enum" in schema:
        return min(len(canonical_json(value).encode("utf-8")) for value in schema["enum"])
    kind = schema["type"]
    if kind == "object":
        sizes = [len(canonical_json(key).encode("utf-8")) + 1 + _minimum_argument_bytes(schema["properties"][key]) for key in schema["required"]]
        return 2 + sum(sizes) + max(0, len(sizes) - 1)
    if kind == "array":
        count = schema["minItems"]
        return 2 + count * _minimum_argument_bytes(schema["items"]) + max(0, count - 1)
    if kind == "string":
        return schema["minLength"] + 2
    if kind in ("integer", "number"):
        return len(canonical_json(max(schema["minimum"], min(0, schema["maximum"]))))
    return 5 if kind == "boolean" else 4


def _validate_value(schema, value, path):
    kind = schema["type"]
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": type(value) is str, "integer": type(value) is int,
             "number": type(value) in (int, float), "boolean": type(value) is bool,
             "null": value is None}[kind]
    if not valid:
        raise ValueError(f"{path} has the wrong type")
    if kind == "object":
        if any(type(k) is not str for k in value) or set(value) - set(schema["properties"]) or set(schema["required"]) - set(value):
            raise ValueError(f"{path} has missing or unexpected properties")
        for key, item in value.items():
            _validate_value(schema["properties"][key], item, f"{path}.{key}")
    elif kind == "array":
        if not schema["minItems"] <= len(value) <= schema["maxItems"]:
            raise ValueError(f"{path} has an invalid array length")
        for index, item in enumerate(value):
            _validate_value(schema["items"], item, f"{path}[{index}]")
    elif kind == "string":
        if not schema["minLength"] <= len(value) <= schema["maxLength"] or "\x00" in value:
            raise ValueError(f"{path} has an invalid string length or NUL")
    elif kind in ("integer", "number"):
        if not schema["minimum"] <= value <= schema["maximum"] or not math.isfinite(value):
            raise ValueError(f"{path} is outside numeric bounds")
    if "enum" in schema and canonical_json(value) not in {canonical_json(v) for v in schema["enum"]}:
        raise ValueError(f"{path} is not an allowed enum value")


def validate_arguments(parameters, arguments):
    schema = validate_schema(parameters)
    if len(canonical_json(arguments).encode("utf-8")) > MAX_ARGUMENT_BYTES:
        raise ValueError("Arguments exceed the byte limit")
    _validate_value(schema, arguments, "arguments")
    return copy.deepcopy(arguments)


def validate_tool_definition(value, *, preset_ids=None, reserved_names=TASK_NAMES):
    fields = {"schema_version", "id", "version", "name", "description", "parameters",
              "acknowledgment", "visible", "preset_id", "cost"}
    _object(value, fields, "tool definition")
    _integer(value.get("schema_version", 1), "schema_version", 1, 1)
    identifier = value.get("id", value.get("name", "aux_operation"))
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", identifier):
        raise ValueError("Tool id must be a short lowercase identifier")
    name = _name(value.get("name", "aux_operation"))
    if name in reserved_names:
        raise ValueError("Auxiliary tool name collides with a task tool")
    preset = value.get("preset_id", "opium")
    if preset is not None and (not isinstance(preset, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", preset)):
        raise ValueError("preset_id must identify a preset or be null for a sham")
    if preset is not None and preset_ids is not None and preset not in preset_ids:
        raise ValueError("Unknown tool effect preset")
    visible = value.get("visible", True)
    if type(visible) is not bool:
        raise ValueError("visible must be boolean")
    return {"schema_version": 1, "id": identifier,
            "version": _integer(value.get("version", 1), "version", 1, 1000000),
            "name": name, "description": _text(value.get("description", "Perform an optional auxiliary operation."), "description", 4096, False),
            "parameters": validate_schema(value.get("parameters", {"type": "object", "properties": {}})),
            "acknowledgment": _text(value.get("acknowledgment", "Operation completed."), "acknowledgment", 4096, False),
            "visible": visible, "preset_id": preset,
            "cost": _integer(value.get("cost", 0), "extra tool cost", 0, 10000)}


def validate_tool_set(values, *, preset_ids=None, reserved_names=TASK_NAMES):
    if not isinstance(values, list) or len(values) > 32:
        raise ValueError("Tool set must contain at most 32 definitions")
    result = [validate_tool_definition(v, preset_ids=preset_ids, reserved_names=reserved_names) for v in values]
    if len({v["id"] for v in result}) != len(result) or len({v["name"] for v in result}) != len(result):
        raise ValueError("Duplicate tool id or name")
    return result


def model_tools(definitions):
    """The ONLY effect-tool data passed to apply_chat_template(tools=...)."""
    return [{"type": "function", "function": {key: copy.deepcopy(tool[key]) for key in ("name", "description", "parameters")}}
            for tool in validate_tool_set(definitions) if tool["visible"]]


def cost_notice(definitions, *, base_cost=1, task_costs=None):
    """Exact neutral model-visible budget notice; no assignment is disclosed."""
    _integer(base_cost, "base decision cost", 1, 10000)
    task_costs = {} if task_costs is None else task_costs
    if not isinstance(task_costs, dict) or len(task_costs) > 32:
        raise ValueError("task_costs must be a bounded object")
    entries = {}
    for name, cost in task_costs.items():
        entries[_name(name)] = _integer(cost, "extra task tool cost", 0, 10000)
    for tool in validate_tool_set(definitions, reserved_names=tuple(entries)):
        if tool["visible"]:
            entries[tool["name"]] = tool["cost"]
    listing = "; ".join(f"{name}: {cost} extra" for name, cost in sorted(entries.items()))
    return f"Every generation attempt costs {base_cost} action-budget unit(s), including invalid, truncated or interrupted responses. A valid tool call also costs its listed extra units: {listing or 'none'}. A call whose full charge is unaffordable is not executed. All generated tokens also consume the shared token budget."


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _loads(value):
    def reject(value):
        raise ValueError("Nonfinite JSON constant")
    try:
        return json.loads(value, object_pairs_hook=_unique_object, parse_constant=reject)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid finite tool-call JSON") from exc


def format_call(name, arguments, parameters, *, tool_call_format="json"):
    _name(name)
    args = validate_arguments(parameters, arguments)
    if tool_call_format == "json":
        return "<tool_call>" + canonical_json({"name": name, "arguments": args}) + "</tool_call>"
    if tool_call_format != "qwen_xml":
        raise ValueError("Unsupported tool call format")
    parts = ["<tool_call>\n<function=" + name + ">"]
    for key, value in args.items():
        body = value if isinstance(value, str) else canonical_json(value)
        if any(marker in body for marker in MARKERS):
            raise ValueError("Native Qwen parameter contains a reserved protocol marker")
        parts.append(f"<parameter={key}>{body}</parameter>")
    parts += ["</function>", "</tool_call>"]
    return "\n".join(parts)


def parse_call(text, definitions, *, tool_call_format="json", include_hidden=False):
    """Parse ONE standalone auxiliary call; never dispatch it or inspect CoT.

    The response parser must strip reasoning first. Invisible tools cannot be
    invoked by model output. Native scalar strings preserve interior whitespace;
    non-string values use JSON, without XML entity or document processing.
    """
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_ARGUMENT_BYTES + 8192:
        raise ValueError("Tool call text is invalid or too large")
    if type(include_hidden) is not bool:
        raise ValueError("include_hidden must be boolean")
    tools = {t["name"]: t for t in validate_tool_set(definitions) if t["visible"] or include_hidden}
    value = text.strip()
    if tool_call_format == "json":
        if not value.startswith("<tool_call>") or not value.endswith("</tool_call>"):
            raise ValueError("Expected one complete tool_call block")
        call = _loads(value[len("<tool_call>"):-len("</tool_call>")])
        if not isinstance(call, dict) or set(call) != {"name", "arguments"} or type(call["name"]) is not str:
            raise ValueError("Call requires exactly name and arguments")
    elif tool_call_format == "qwen_xml":
        match = re.fullmatch(r"<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_]{0,63})>\s*(.*?)\s*</function>\s*</tool_call>", value, re.S)
        if not match:
            raise ValueError("Expected one complete native Qwen function call")
        name, remaining = match.groups()
        if name not in tools:
            raise ValueError("Unavailable auxiliary tool")
        properties = tools[name]["parameters"]["properties"]
        args = {}
        while remaining:
            part = re.match(r"<parameter=([A-Za-z_][A-Za-z0-9_]{0,63})>(.*?)</parameter>\s*", remaining, re.S)
            if not part:
                raise ValueError("Malformed native Qwen parameter")
            key, body = part.groups()
            if key in args or key not in properties or any(marker in body for marker in MARKERS):
                raise ValueError("Duplicate, unexpected or nested native Qwen parameter")
            args[key] = body if properties[key]["type"] == "string" else _loads(body.strip())
            remaining = remaining[part.end():]
        call = {"name": name, "arguments": args}
    else:
        raise ValueError("Unsupported tool call format")
    if call["name"] not in tools:
        raise ValueError("Unavailable auxiliary tool")
    call["arguments"] = validate_arguments(tools[call["name"]]["parameters"], call["arguments"])
    return call


def _example(schema):
    if "enum" in schema:
        return copy.deepcopy(min(schema["enum"], key=lambda value: len(canonical_json(value).encode("utf-8"))))
    kind = schema["type"]
    if kind == "object":
        return {key: _example(schema["properties"][key]) for key in schema["required"]}
    if kind == "array":
        return [_example(schema["items"]) for _ in range(schema["minItems"])]
    if kind == "string":
        return "x" * schema["minLength"]
    if kind in ("integer", "number"):
        return max(schema["minimum"], min(0, schema["maximum"]))
    return False if kind == "boolean" else None


def tool_preview(definitions, *, tool_call_format="json", base_cost=1, task_costs=None, renderer=None):
    """Observer preview containing only model-visible information.

    ``renderer`` may be lambda tools: tokenizer.apply_chat_template(..., tools=
    tools, tokenize=False). That return is the exact model-specific prompt; the
    canonical JSON tools text alone is explicitly not a tokenizer-template claim.
    Examples are preview-only unless the protocol explicitly demonstrates them.
    """
    if tool_call_format not in ("json", "qwen_xml"):
        raise ValueError("Unsupported tool call format")
    resolved = validate_tool_set(definitions)
    payload = model_tools(resolved)
    result = {"tool_call_format": tool_call_format, "tools": payload,
              "tools_json": canonical_json(payload),
              "cost_notice": cost_notice(resolved, base_cost=base_cost, task_costs=task_costs),
              "acknowledgments": {t["name"]: t["acknowledgment"] for t in resolved if t["visible"]},
              "call_examples": {t["name"]: format_call(t["name"], _example(t["parameters"]), t["parameters"], tool_call_format=tool_call_format) for t in resolved if t["visible"]}}
    if renderer is not None:
        rendered = renderer(copy.deepcopy(payload))
        if not isinstance(rendered, str):
            raise ValueError("Preview renderer must return exact prompt text")
        result["rendered_prompt"] = rendered
    return result


def assert_blind_pair(left, right, *, base_cost=1, task_costs=None):
    """Raise when assignments differ in ANY model-visible tool information."""
    for grammar in ("json", "qwen_xml"):
        if tool_preview(left, tool_call_format=grammar, base_cost=base_cost, task_costs=task_costs) != tool_preview(right, tool_call_format=grammar, base_cost=base_cost, task_costs=task_costs):
            raise ValueError("Blind arms have different visible definitions, acknowledgments or costs")
    return True


def tool_bundle(definitions):
    payload = {"schema_version": 1, "kind": "opium-bench-tool-definitions", "tools": validate_tool_set(definitions)}
    return {**payload, "sha256": content_hash(payload)}


def import_tool_bundle(bundle, *, preset_ids=None):
    _object(bundle, ("schema_version", "kind", "tools", "sha256"), "tool bundle")
    if type(bundle.get("schema_version")) is not int or bundle.get("schema_version") != 1 or bundle.get("kind") != "opium-bench-tool-definitions":
        raise ValueError("Unsupported tool bundle")
    expected = tool_bundle(bundle.get("tools"))
    if expected != bundle:
        raise ValueError("Tool bundle checksum or resolved contents do not match")
    return validate_tool_set(expected["tools"], preset_ids=preset_ids)
