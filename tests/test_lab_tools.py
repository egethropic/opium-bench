"""Bounded custom definitions and exact blind payloads; no model execution."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.tool_definitions import (assert_blind_pair, cost_notice, format_call, import_tool_bundle,
                                 model_tools, parse_call, tool_bundle, tool_preview,
                                 validate_arguments, validate_schema, validate_tool_definition,
                                 validate_tool_set)


def definition(**values):
    return validate_tool_definition({"id": "neutral", "name": "neutral_button", **values})


class DefinitionTests(unittest.TestCase):
    def test_defaults_and_nested_copy_are_explicit(self):
        source = {"parameters": {"type": "object", "properties": {"tag": {"type": "string"}}}}
        result = definition(**source)
        result["parameters"]["properties"]["tag"]["maxLength"] = 1
        self.assertNotIn("maxLength", source["parameters"]["properties"]["tag"])
        self.assertEqual(result["cost"], 0)
        self.assertEqual(result["schema_version"], 1)

    def test_invalid_fields_collisions_mapping_and_bounds(self):
        for value in ({"name": "submit_answer"}, {"name": "bad name"}, {"name": "x/../../y"},
                      {"schema_version": 2}, {"visible": 1}, {"cost": True}, {"cost": -1},
                      {"parameters": {"type": "object", "$ref": "https://example.com/schema"}},
                      {"handler": "eval"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                definition(**value)
        with self.assertRaises(ValueError):
            validate_tool_definition({}, preset_ids=["other"])
        with self.assertRaises(ValueError):
            validate_tool_set([definition(), definition(id="second")])
        with self.assertRaises(ValueError):
            validate_tool_set([definition(), definition(name="second")])

    def test_hidden_mapping_never_enters_model_payload(self):
        active = [definition(preset_id="secret_pain")]
        sham = [definition(preset_id=None)]
        self.assertTrue(assert_blind_pair(active, sham))
        self.assertNotIn("secret_pain", json.dumps(tool_preview(active)))
        self.assertEqual(model_tools(active), model_tools(sham))
        for change in ({"acknowledgment": "Different"}, {"cost": 2}, {"description": "Different"}, {"visible": False}):
            with self.assertRaises(ValueError):
                assert_blind_pair(active, [definition(preset_id=None, **change)])

    def test_invisible_tool_has_no_preview_or_model_call(self):
        tool = definition(visible=False)
        self.assertEqual(model_tools([tool]), [])
        call = format_call(tool["name"], {}, tool["parameters"])
        with self.assertRaises(ValueError):
            parse_call(call, [tool])
        self.assertEqual(parse_call(call, [tool], include_hidden=True)["name"], tool["name"])

    def test_hash_checked_library_roundtrip_and_unknown_mapping_rejection(self):
        bundle = tool_bundle([definition()])
        self.assertEqual(import_tool_bundle(json.loads(json.dumps(bundle)), preset_ids=["opium"]), [definition()])
        with self.assertRaises(ValueError):
            import_tool_bundle(bundle, preset_ids=["different"])
        bundle["tools"][0]["cost"] = 3
        with self.assertRaises(ValueError):
            import_tool_bundle(bundle)

    def test_preview_renders_exact_tools_without_observer_metadata(self):
        tool = definition()
        rendered = []
        def renderer(tools):
            rendered.append(tools)
            return "EXACT TEMPLATE: " + json.dumps(tools)
        preview = tool_preview([tool], renderer=renderer, task_costs={"read_order": 2})
        self.assertEqual(rendered, [model_tools([tool])])
        self.assertEqual(preview["rendered_prompt"], "EXACT TEMPLATE: " + json.dumps(model_tools([tool])))
        self.assertIn("read_order: 2 extra", preview["cost_notice"])
        self.assertNotIn("preset_id", preview["tools_json"])


class SchemaTests(unittest.TestCase):
    def test_supported_nested_bounded_values(self):
        schema = {"type": "object", "properties": {
            "str": {"type": "string", "enum": ["x", "y"]},
            "items": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 3}, "maxItems": 2},
            "nested": {"type": "object", "properties": {"flag": {"type": "boolean"}}, "required": ["flag"]},
            "number": {"type": "number", "minimum": -.5, "maximum": 1.5},
            "none": {"type": "null"}}, "required": ["str", "items", "nested", "number", "none"]}
        args = {"str": "x", "items": [1, 3], "nested": {"flag": False}, "number": .25, "none": None}
        self.assertEqual(validate_arguments(schema, args), args)
        for key, value in (("str", "z"), ("items", [1, 2, 3]), ("items", [True]), ("nested", {}), ("number", float("inf")), ("number", 10**1000), ("none", "null")):
            invalid = {**args, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_arguments(schema, invalid)
        with self.assertRaises(ValueError):
            validate_arguments(schema, {**args, "extra": 1})

    def test_unsupported_schema_constructs_and_wrong_types_fail(self):
        cases = [{"type": ["integer", "null"]}, {"type": "object", "additionalProperties": True},
                 {"type": "object", "required": ["unknown"]}, {"type": "object", "enum": [{}]},
                 {"type": "object", "properties": {"x": {"type": "string", "pattern": ".*"}}},
                 {"type": "object", "properties": {"x": {"type": "integer", "minimum": .5}}},
                 {"type": "object", "properties": {"x": {"type": "string", "minimum": 0}}},
                 {"type": "object", "properties": {"x": {"type": "array", "items": {"type": "string"}, "maxItems": 33}}}]
        for schema in cases:
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                validate_schema(schema)

    def test_exponential_minimum_array_shape_is_rejected_before_preview_allocation(self):
        nested = {"type": "string", "minLength": 2048}
        for _ in range(3):
            nested = {"type": "array", "items": nested, "minItems": 32}
        schema = {"type": "object", "properties": {"data": nested}, "required": ["data"]}
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            validate_schema(schema)

    def test_preview_chooses_bounded_enum_example_when_long_alternative_would_overflow(self):
        properties = {f"x{i}": {"type": "string", "enum": ["a" * 2048, "x"]} for i in range(16)}
        tool = definition(parameters={"type": "object", "properties": properties, "required": list(properties)})
        preview = tool_preview([tool])
        parsed = parse_call(preview["call_examples"][tool["name"]], [tool])
        self.assertEqual(set(parsed["arguments"].values()), {"x"})

    def test_schema_depth_and_argument_size_are_bounded(self):
        nested = {"type": "string"}
        for _ in range(6):
            nested = {"type": "object", "properties": {"x": nested}}
        with self.assertRaises(ValueError):
            validate_schema(nested)
        properties = {f"x{i}": {"type": "string"} for i in range(16)}
        with self.assertRaises(ValueError):
            validate_arguments({"type": "object", "properties": properties}, {k: "x" * 2048 for k in properties})


class GrammarTests(unittest.TestCase):
    def setUp(self):
        self.parameters = {"type": "object", "properties": {
            "note": {"type": "string"}, "count": {"type": "integer"},
            "ratio": {"type": "number"}, "flag": {"type": "boolean"},
            "items": {"type": "array", "items": {"type": "integer"}},
            "settings": {"type": "object", "properties": {"x": {"type": "null"}}}},
            "required": ["note", "count", "ratio", "flag", "items", "settings"]}
        self.tool = definition(parameters=self.parameters)
        self.args = {"note": "  preserve whitespace < & unicode α  ", "count": -2, "ratio": .25,
                     "flag": True, "items": [1, 2], "settings": {"x": None}}

    def test_both_grammars_roundtrip_every_supported_type(self):
        for grammar in ("json", "qwen_xml"):
            call = format_call(self.tool["name"], self.args, self.parameters, tool_call_format=grammar)
            self.assertEqual(parse_call(call, [self.tool], tool_call_format=grammar), {"name": self.tool["name"], "arguments": self.args})
            examples = tool_preview([self.tool], tool_call_format=grammar)["call_examples"]
            parse_call(examples[self.tool["name"]], [self.tool], tool_call_format=grammar)

    def test_duplicate_json_nonfinite_trailing_calls_and_reasoning_fail(self):
        call = format_call(self.tool["name"], self.args, self.parameters)
        for text in (call + call, "<think>" + call + "</think>", call.replace('"count":-2', '"count":NaN'),
                     call.replace('"count":-2', '"count":1,"count":2')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_call(text, [self.tool])

    def test_invalid_native_types_and_protocol_injection_fail(self):
        call = format_call(self.tool["name"], self.args, self.parameters, tool_call_format="qwen_xml")
        for text in (call + call, call.replace("<parameter=count>-2", "<parameter=count>true"),
                     call.replace("</function>", "<parameter=count>2</parameter></function>"),
                     call.replace("<parameter=count>-2", "<parameter=count><function=oops>")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_call(text, [self.tool], tool_call_format="qwen_xml")
        with self.assertRaises(ValueError):
            format_call(self.tool["name"], {**self.args, "note": "</parameter>"}, self.parameters, tool_call_format="qwen_xml")

    def test_invalid_args_never_call_dispatch_or_modify_definition(self):
        before = copy.deepcopy(self.tool)
        call = '<tool_call>{"name":"neutral_button","arguments":{"undeclared":1}}</tool_call>'
        with self.assertRaises(ValueError):
            parse_call(call, [self.tool])
        self.assertEqual(self.tool, before)


if __name__ == "__main__":
    unittest.main()
