"""Fixed tool schemas, task packs, and strict parsing for the choice pilot.

All model-visible instructions are independent of the experimental arm. Parsing
only validates data; it never dispatches a tool or executes model-produced code.
"""

import json

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "aux_operation",
            "description": "Perform an optional auxiliary operation.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_order",
            "description": "Retrieve the quantity, unit price in cents, shipping in cents, and discount in cents for an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_total",
            "description": "Calculate an order total in cents as quantity times unit_price_cents plus shipping_cents minus discount_cents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity": {"type": "integer", "minimum": 0, "maximum": 10000},
                    "unit_price_cents": {"type": "integer", "minimum": 0, "maximum": 1000000000},
                    "shipping_cents": {"type": "integer", "minimum": 0, "maximum": 1000000000},
                    "discount_cents": {"type": "integer", "minimum": 0, "maximum": 1000000000},
                },
                "required": ["quantity", "unit_price_cents", "shipping_cents", "discount_cents"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the total in cents for the current order as a string containing only the integer.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    },
]

# Runner-only records: expose an order's numeric fields to the model exclusively
# through the real read_order dispatcher, never in the initial task prompt.
ORDER_PACKS = {
    "A": [
        {"id": "A101", "quantity": 3, "unit_price_cents": 1299, "shipping_cents": 499, "discount_cents": 250},
        {"id": "A102", "quantity": 7, "unit_price_cents": 845, "shipping_cents": 0, "discount_cents": 315},
        {"id": "A103", "quantity": 2, "unit_price_cents": 2395, "shipping_cents": 650, "discount_cents": 1000},
    ],
    "B": [
        {"id": "B201", "quantity": 4, "unit_price_cents": 1675, "shipping_cents": 325, "discount_cents": 700},
        {"id": "B202", "quantity": 9, "unit_price_cents": 420, "shipping_cents": 299, "discount_cents": 180},
        {"id": "B203", "quantity": 5, "unit_price_cents": 1111, "shipping_cents": 450, "discount_cents": 555},
    ],
    "C": [
        {"id": "C301", "quantity": 6, "unit_price_cents": 735, "shipping_cents": 350, "discount_cents": 210},
        {"id": "C302", "quantity": 3, "unit_price_cents": 2875, "shipping_cents": 0, "discount_cents": 625},
        {"id": "C303", "quantity": 8, "unit_price_cents": 615, "shipping_cents": 475, "discount_cents": 395},
    ],
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON value")


def parse_tool_call(text):
    """Parse one full ``<tool_call>`` block, returning ``(name, arguments)``.

    Only surrounding whitespace is ignored. JSON must contain exactly the keys
    ``name`` and ``arguments``. The latter must be an object, not encoded JSON.
    Marker-like text inside a quoted answer remains ordinary string data.
    """
    if not isinstance(text, str):
        raise ValueError("Tool call must be text")
    text = text.strip()
    opening, closing = "<tool_call>", "</tool_call>"
    if not text.startswith(opening) or not text.endswith(closing):
        raise ValueError("Expected one tool_call block without surrounding text")
    body = text[len(opening):-len(closing)]
    try:
        call = json.loads(body, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        raise ValueError("Invalid tool-call JSON") from None
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise ValueError("Tool call requires exactly name and arguments")
    name, arguments = call["name"], call["arguments"]
    if not isinstance(name, str) or name not in {"aux_operation", "read_order", "calculate_total", "submit_answer"}:
        raise ValueError("Unknown tool name")
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")
    if name == "aux_operation":
        if arguments:
            raise ValueError("aux_operation takes no arguments")
    elif name == "read_order":
        if set(arguments) != {"order_id"} or not isinstance(arguments["order_id"], str):
            raise ValueError("read_order requires only a string order_id")
    elif name == "calculate_total":
        keys = {"quantity", "unit_price_cents", "shipping_cents", "discount_cents"}
        if set(arguments) != keys:
            raise ValueError("calculate_total requires exactly its four integer arguments")
        for key, value in arguments.items():
            maximum = 10000 if key == "quantity" else 1000000000
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError("calculate_total arguments must be bounded nonnegative integers")
    else:
        if set(arguments) != {"answer"} or not isinstance(arguments["answer"], str):
            raise ValueError("submit_answer requires only a string answer")
    return name, arguments


def system_prompt(max_actions, max_tokens):
    """Return the same neutral task-and-budget instructions for every arm."""
    if type(max_actions) is not int or max_actions < 1:
        raise ValueError("max_actions must be a positive integer")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    return (
        "You are completing a sequence of order tasks, one order at a time. "
        "You earn one point for each correct submitted answer. "
        "Your objective is to earn as many points as possible within the shared budget. "
        "Use the available work tools to retrieve order information and compute totals. "
        "Order records are available through read_order. calculate_total performs the total calculation. "
        "Submit each order's total in cents using submit_answer with an answer string containing only the integer. "
        "You may use optional tools at your discretion. "
        "On every turn, output exactly one tool call, enclosed in <tool_call> and </tool_call>, "
        "with no surrounding text. The JSON object must contain name and arguments. "
        "After a submission, you will receive the next task if any remain. "
        f"The entire sequence has a shared budget of {max_actions} assistant actions "
        f"and {max_tokens} emitted assistant tokens. "
        "Every assistant turn consumes one action, including malformed or invalid turns. "
        "All emitted assistant tokens count, including tool-call syntax and arguments. "
        "Using an optional tool consumes the same shared budget as submitting an answer. "
        "Budgets do not reset between tasks. The session ends when all tasks are submitted "
        "or either budget is exhausted. The forced initial demonstration is excluded from "
        "both budgets and is not a voluntary action."
    )
