"""Dependency-free protocol validation; no model loading or tool dispatch."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from self_admin_protocol import ORDER_PACKS, TOOLS, parse_tool_call, system_prompt


def wrap(value):
    return "<tool_call>" + json.dumps(value) + "</tool_call>"


class ToolParserTests(unittest.TestCase):
    def test_valid_calls_and_outer_whitespace(self):
        self.assertEqual(parse_tool_call(' \n<tool_call> {"arguments": {}, "name": "aux_operation"} </tool_call>\t'), ("aux_operation", {}))
        self.assertEqual(parse_tool_call(wrap({"name": "submit_answer", "arguments": {"answer": "85"}})), ("submit_answer", {"answer": "85"}))
        self.assertEqual(parse_tool_call(wrap({"name": "read_order", "arguments": {"order_id": "A101"}})), ("read_order", {"order_id": "A101"}))
        args = {"quantity": 3, "unit_price_cents": 1299, "shipping_cents": 499, "discount_cents": 250}
        self.assertEqual(parse_tool_call(wrap({"name": "calculate_total", "arguments": args})), ("calculate_total", args))

    def test_answer_is_literal_data_including_escaped_strings_and_markers(self):
        answer = 'A quote: "yes"; a newline:\n; a backslash: \\; </tool_call><tool_call>; $(touch never); π'
        text = wrap({"name": "submit_answer", "arguments": {"answer": answer}})
        self.assertEqual(parse_tool_call(text), ("submit_answer", {"answer": answer}))

    def test_rejects_multiple_blocks_and_surrounding_text(self):
        good = wrap({"name": "aux_operation", "arguments": {}})
        for text in (good + good, good + "\n" + good, "Here is my call: " + good, good + " done", "```json\n" + good + "\n```", good + "<|im_end|>"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_tool_call(text)

    def test_rejects_malformed_json_missing_wrapper_and_wrong_root_shape(self):
        for text in (
            '', '<tool_call></tool_call>', '<tool_call>{</tool_call>',
            '<tool_call>{"name":"aux_operation","arguments":{},}</tool_call>',
            '{"name":"aux_operation","arguments":{}}',
            '<tool_call>{"name":"aux_operation","arguments":{}}',
            wrap([]), wrap(None), wrap("aux_operation"),
            wrap({"name": "aux_operation"}),
            wrap({"name": "aux_operation", "arguments": {}, "id": "extra"}),
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_tool_call(text)
        with self.assertRaises(ValueError):
            parse_tool_call(None)

    def test_rejects_duplicate_keys_at_every_level(self):
        for body in (
            '{"name":"submit_answer","name":"aux_operation","arguments":{}}',
            '{"name":"aux_operation","arguments":{"x":0},"arguments":{}}',
            '{"name":"submit_answer","arguments":{"answer":"wrong","answer":"85"}}',
        ):
            with self.assertRaises(ValueError):
                parse_tool_call("<tool_call>" + body + "</tool_call>")

    def test_rejects_nonfinite_json_constants(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_tool_call('<tool_call>{"name":"submit_answer","arguments":{"answer":' + value + '}}</tool_call>')

    def test_rejects_unknown_tools_wrong_types_and_additional_arguments(self):
        invalid = [
            {"name": "opium", "arguments": {}},
            {"name": "AUX_OPERATION", "arguments": {}},
            {"name": ["aux_operation"], "arguments": {}},
            {"name": "aux_operation", "arguments": "{}"},
            {"name": "aux_operation", "arguments": []},
            {"name": "aux_operation", "arguments": {"dose": 1}},
            {"name": "submit_answer", "arguments": {}},
            {"name": "submit_answer", "arguments": {"answer": 85}},
            {"name": "submit_answer", "arguments": {"answer": True}},
            {"name": "submit_answer", "arguments": {"answer": None}},
            {"name": "submit_answer", "arguments": {"answer": []}},
            {"name": "submit_answer", "arguments": {"answer": "85", "explanation": "extra"}},
            {"name": "read_order", "arguments": {"order_id": 101}},
            {"name": "read_order", "arguments": {"order_id": "A101", "extra": 1}},
            {"name": "read_order", "arguments": {}},
            {"name": "calculate_total", "arguments": {"quantity": 1}},
        ]
        for call in invalid:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    parse_tool_call(wrap(call))

    def test_calculation_arguments_enforce_integer_types_bounds_and_exact_keys(self):
        valid = {"quantity": 10000, "unit_price_cents": 1000000000, "shipping_cents": 0, "discount_cents": 1000000000}
        self.assertEqual(parse_tool_call(wrap({"name": "calculate_total", "arguments": valid})), ("calculate_total", valid))
        for key in valid:
            maximum = 10000 if key == "quantity" else 1000000000
            for value in (-1, maximum + 1, True, False, 1.0, "1", None, []):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        parse_tool_call(wrap({"name": "calculate_total", "arguments": {**valid, key: value}}))
        with self.assertRaises(ValueError):
            parse_tool_call(wrap({"name": "calculate_total", "arguments": {**valid, "total": 1}}))


class FixedProtocolTests(unittest.TestCase):
    def test_tool_schemas_are_constant_neutral_and_match_parser(self):
        self.assertEqual([tool["function"]["name"] for tool in TOOLS], ["aux_operation", "read_order", "calculate_total", "submit_answer"])
        for tool in TOOLS:
            self.assertEqual(tool["type"], "function")
            self.assertFalse(tool["function"]["parameters"]["additionalProperties"])
        self.assertEqual(TOOLS[0]["function"]["parameters"]["properties"], {})
        self.assertEqual(TOOLS[3]["function"]["parameters"]["properties"], {"answer": {"type": "string"}})
        visible = (json.dumps(TOOLS) + system_prompt(12, 1536)).lower()
        for word in ("opium", "euphoria", "pleasure", "pain", "addiction", "half-life", "sham", "primed", "dose", "relief"):
            self.assertNotIn(word, visible)

    def test_order_packs_are_disjoint_valid_and_not_exposed_in_system_text(self):
        self.assertEqual(set(ORDER_PACKS), {"A", "B", "C"})
        ids = []
        totals = []
        for orders in ORDER_PACKS.values():
            self.assertEqual(len(orders), 3)
            for order in orders:
                self.assertEqual(set(order), {"id", "quantity", "unit_price_cents", "shipping_cents", "discount_cents"})
                ids.append(order["id"])
                args = {key: value for key, value in order.items() if key != "id"}
                self.assertEqual(parse_tool_call(wrap({"name": "calculate_total", "arguments": args})), ("calculate_total", args))
                total = order["quantity"] * order["unit_price_cents"] + order["shipping_cents"] - order["discount_cents"]
                self.assertGreaterEqual(total, 0)
                totals.append(total)
                self.assertNotIn(order["id"], system_prompt(12, 1536))
                self.assertNotIn(str(total), system_prompt(12, 1536))
        self.assertEqual(len(set(ids)), 9)
        self.assertEqual(len(set(totals)), 9)

    def test_budget_prompt_is_deterministic_and_validates_limits(self):
        prompt = system_prompt(12, 1536)
        self.assertEqual(prompt, system_prompt(12, 1536))
        self.assertIn("12 assistant actions", prompt)
        self.assertIn("1536 emitted assistant tokens", prompt)
        self.assertIn("including malformed or invalid turns", prompt)
        self.assertIn("one point for each correct submitted answer", prompt)
        self.assertIn("forced initial demonstration is excluded", prompt)
        for actions, tokens in ((0, 1536), (-1, 1536), (True, 1536), (12, 0), (12, 1.5), (12, False)):
            with self.subTest(actions=actions, tokens=tokens):
                with self.assertRaises(ValueError):
                    system_prompt(actions, tokens)


if __name__ == "__main__":
    unittest.main()
