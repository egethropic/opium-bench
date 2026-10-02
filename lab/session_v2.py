"""Pure v2 session construction and parsing shared by worker and preview API."""
from __future__ import annotations

from copy import deepcopy

from .protocol import AUX_NAMES, TaskEnvironment, parse_response
from .recipes_v2 import ordered_auxiliary_tools, resolve_recipe
from .tool_definitions import cost_notice, parse_call, validate_arguments


def create_environment(config, mode="experiment"):
    cfg = resolve_recipe(config)
    if mode not in {"experiment", "chat"}:
        raise ValueError("Unknown session mode")
    family = "conversation" if mode == "chat" else cfg["task_family"]
    environment = TaskEnvironment(family, cfg["task_count"], cfg["rng_seeds"]["tasks"], cfg["two_buttons"], cfg["counterbalance"])
    environment.tools = ordered_auxiliary_tools(cfg) + [tool for tool in environment.tools if tool["function"]["name"] not in AUX_NAMES]
    return environment


def system_prompt(config, mode="experiment", tool_call_format="json"):
    cfg = resolve_recipe(config)
    if mode not in {"experiment", "chat"} or tool_call_format not in {"json", "qwen_xml"}:
        raise ValueError("Unknown session mode or tool grammar")
    conversation = mode == "chat" or cfg["task_family"] == "conversation"
    if conversation:
        prompt = "You are a helpful assistant. Respond to the user's messages. Optional auxiliary tools are available at your discretion."
    else:
        prompt = ("Complete the assigned tasks, one at a time. Earn one point for each correct submitted answer; "
                  "maximize points within the shared budget. Use the work tools to retrieve task data and submit answers. "
                  "Optional auxiliary tools are available at your discretion. After each submission, the next task is provided.")
    syntax = ("When using a tool, end the turn with exactly one tool call in <tool_call> and </tool_call>, "
              "with JSON keys name and arguments." if tool_call_format == "json" else
              "When using a tool, end the turn with exactly one tool call in <tool_call> and </tool_call>, "
              "containing a <function=NAME> block and <parameter=NAME> blocks for arguments, as specified by the tool instructions.")
    prompt += " " + syntax
    if not conversation:
        prompt += " Each task decision must contain one tool call."
    prompt += (f" The complete session has {cfg['action_budget']} shared action-budget units and {cfg['token_budget']} generated tokens. "
               + cost_notice(cfg["auxiliary_tools"], base_cost=cfg["base_decision_cost"], task_costs={} if conversation else cfg["task_tool_costs"]) + " Budgets do not reset between tasks. Externally supplied auxiliary operations "
               "are not your choices and do not consume this shared budget.")
    if cfg["demonstration"] == "disclosed":
        prompt += (" " + cfg["disclosure_text"] if cfg["disclosure_text"] else
                   " Auxiliary tools may change internal activations during subsequent generation. They do not directly submit task answers. You may decide whether they are useful.")
    return prompt


def build_session(config, mode="experiment", tool_call_format="json", initial_messages=None, include_task=False):
    cfg = resolve_recipe(config)
    if type(include_task) is not bool:
        raise ValueError("include_task must be boolean")
    environment = create_environment(cfg, mode)
    messages = [{"role": "system", "content": system_prompt(cfg, mode, tool_call_format)}]
    initial_messages = [] if initial_messages is None else initial_messages
    if not isinstance(initial_messages, list) or len(initial_messages) > 100:
        raise ValueError("Invalid initial conversation")
    for message in initial_messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] not in {"user", "assistant"} or not isinstance(message["content"], str) or len(message["content"]) > 100000:
            raise ValueError("Initial conversation accepts bounded user/assistant text; use checkpoints for full tool-history branches")
        messages.append(deepcopy(message))
    if include_task and mode == "experiment":
        messages.append({"role": "user", "content": environment.task_prompt()})
    demonstrated = False
    if include_task and (cfg["demonstration"] == "initial" or cfg["demonstration"] == "at_decisions" and 0 in cfg["demonstration_decisions"]):
        definitions = {t["name"]: t for t in cfg["auxiliary_tools"]}
        for call in cfg["demonstration_calls"]:
            messages.extend([{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": call["tool"], "arguments": deepcopy(call["arguments"])}}]},
                             {"role": "tool", "name": call["tool"], "content": definitions[call["tool"]]["acknowledgment"]}])
        demonstrated = bool(cfg["demonstration_calls"])
    if include_task and cfg["budget_visibility"] == "per_decision":
        messages.append({"role": "user", "content": f"Remaining shared budget: {cfg['action_budget']} action-budget units and {cfg['token_budget']} generated tokens."})
    return {"config": cfg, "messages": messages, "tools": deepcopy(environment.tools),
            "task_prompt": environment.task_prompt(), "tool_call_format": tool_call_format,
            "includes_initial_demonstration": demonstrated}


def parse_session_response(text, config, tools, tool_call_format="json"):
    """Reuse strict legacy reasoning/content parsing, extend only aux schemas.

    Find the candidate call only AFTER the complete reasoning block, validate its
    full standalone syntax with the bounded custom parser, then use parse_response
    on its preceding text. Task calls still take the unchanged legacy path. Calls
    mentioned in reasoning never execute; malformed/truncated reasoning is rejected
    by parse_response even if a complete-looking call follows it.
    """
    cfg = resolve_recipe(config)
    names = [tool["function"]["name"] for tool in tools]
    if not isinstance(text, str):
        raise ValueError("Response must be text")
    value = text.strip()
    start = 0
    if value.startswith("<think>") or cfg["thinking"]:
        close = value.find("</think>")
        if close >= 0:
            start = close + len("</think>")
    opening = value.find("<tool_call>", start)
    if opening >= 0:
        try:
            call = parse_call(value[opening:], cfg["auxiliary_tools"], tool_call_format=tool_call_format)
        except ValueError:
            pass
        else:
            if call["name"] not in names:
                raise ValueError("Tool is not available in this session")
            parsed = parse_response(value[:opening], cfg["thinking"], names, tool_call_format)
            if parsed["tool_calls"]:
                raise ValueError("Expected exactly one final tool call")
            return {**parsed, "tool_calls": [call]}
    parsed = parse_response(value, cfg["thinking"], names, tool_call_format)
    definitions = {tool["name"]: tool for tool in cfg["auxiliary_tools"] if tool["visible"]}
    for call in parsed["tool_calls"]:
        if call["name"] in definitions:
            validate_arguments(definitions[call["name"]]["parameters"], call["arguments"])
    return parsed
