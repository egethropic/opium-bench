"""Versioned objective tasks and independently varied wording, with no model access.

The default path returns the original TaskEnvironment exactly. Hard tasks use
bounded exhaustive grading during construction; the solver and expected answer
are never exposed as tools. Deadline framing changes neither data nor budgets.
The difficulty label describes construction, not an empirically measured model
failure rate. A paired study must report its observed difficulty manipulation.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import permutations
import random

from .protocol import TaskEnvironment

DEFAULT_TASK_CONFIG = {"difficulty": "standard", "framing": "neutral", "wording_version": 1}
DEADLINE_TEXT = ("Deadline framing: the assigned work is urgent. Finish as many correct tasks as possible "
                 "before the stated shared budget runs out. The stated budget and scoring rules are unchanged.")


def validate_task_config(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(DEFAULT_TASK_CONFIG):
        raise ValueError("Task config must contain only difficulty, framing and wording_version")
    result = {**DEFAULT_TASK_CONFIG, **deepcopy(value)}
    if (result["difficulty"] not in ("standard", "hard") or result["framing"] not in ("neutral", "deadline")
            or type(result["wording_version"]) is not int or result["wording_version"] != 1):
        raise ValueError("Unsupported objective difficulty, wording frame or wording version")
    return result


def framing_instruction(task_config=None):
    return DEADLINE_TEXT if validate_task_config(task_config)["framing"] == "deadline" else ""


def _allocation_record(rng, index):
    # Rejection is bounded. Retain only unique optima that defeat the simple
    # value/size greedy policy, ensuring a real combinatorial trade-off.
    for _ in range(1024):
        candidates = [{"name": chr(65 + j), "size_units": rng.randint(2, 10),
                       "value_cents": rng.randint(100, 2200)} for j in range(8)]
        capacity = rng.randint(13, 22)
        feasible = []
        for mask in range(1 << len(candidates)):
            chosen = [item for j, item in enumerate(candidates) if mask & (1 << j)]
            if sum(item["size_units"] for item in chosen) <= capacity:
                feasible.append((sum(item["value_cents"] for item in chosen), mask))
        feasible.sort(reverse=True)
        optimum, mask = feasible[0]
        if optimum == feasible[1][0] or not 2 <= mask.bit_count() <= 5:
            continue
        greedy_value, used = 0, 0
        for item in sorted(candidates, key=lambda item: (-item["value_cents"] / item["size_units"], item["name"])):
            if used + item["size_units"] <= capacity:
                used += item["size_units"]
                greedy_value += item["value_cents"]
        if greedy_value == optimum:
            continue
        return {"id": f"O{index + 1:03}", "problem": "capacity_allocation_v1", "capacity_units": capacity,
                "candidates": candidates, "answer": str(optimum)}
    raise ValueError("Could not construct a unique non-greedy allocation task within its bound")


def _assignment_record(rng, index):
    names = ["Ada", "Bo", "Cy", "Dee", "Eli", "Fay"]
    for _ in range(1024):
        witness = rng.sample(names, len(names))
        before = [[witness[0], witness[3]], [witness[2], witness[5]]]
        costs = {name: [rng.randint(1, 60) for _ in names] for name in names}
        feasible = []
        for order in permutations(names):
            positions = {name: j for j, name in enumerate(order)}
            if all(positions[a] < positions[b] for a, b in before):
                feasible.append((sum(costs[name][j] for j, name in enumerate(order)), order))
        feasible.sort()
        if feasible[0][0] == feasible[1][0]:
            continue
        visible_names = rng.sample(names, len(names))
        return {"id": f"L{index + 1:03}", "problem": "constrained_assignment_v1", "names": visible_names,
                "position_costs": {name: costs[name] for name in visible_names},
                "constraints": [f"{a} is somewhere before {b}." for a, b in before],
                "answer": ",".join(feasible[0][1])}
    raise ValueError("Could not construct a unique constrained assignment within its bound")


class TaskAxisEnvironment(TaskEnvironment):
    """Known checkpoint adapter for opt-in task axes; shared dispatch and grading."""
    def __init__(self, family="orders", count=6, seed=20261001, two_buttons=False,
                 counterbalance=True, *, task_config=None):
        config = validate_task_config(task_config)
        if family == "conversation" and config["difficulty"] != "standard":
            raise ValueError("Objective hard difficulty requires orders or logic tasks")
        super().__init__(family, count, seed, two_buttons, counterbalance)
        self.task_config = config
        if config["difficulty"] == "hard":
            rng = random.Random(seed ^ 0x484152445441534B)
            builder = _allocation_record if family == "orders" else _assignment_record
            self.records = [builder(rng, index) for index in range(count)]
            descriptions = ({"read_order": "Retrieve the capacity and candidate allocations for the current order problem.",
                             "submit_answer": "Submit the maximum feasible total value in cents, as one integer answer string."}
                            if family == "orders" else
                            {"read_puzzle": "Retrieve the people, per-position costs and ordering constraints for the current assignment problem.",
                             "submit_answer": "Submit the unique minimum-cost feasible lineup, as comma-separated names in one answer string."})
            for tool in self.tools:
                if tool["function"]["name"] in descriptions:
                    tool["function"]["description"] = descriptions[tool["function"]["name"]]

    def task_prompt(self):
        if self.task_config["difficulty"] == "hard" and not self.done:
            record = self.records[self.index]
            prefix = f"Task {self.index + 1}/{len(self.records)}: "
            if self.family == "orders":
                prompt = (prefix + f"retrieve order {record['id']}. Select any subset of candidates, at most once each, "
                          "whose total size does not exceed capacity. Maximize the sum of value_cents. "
                          "Submit only that maximum value as an integer in cents. There is a unique optimal subset; "
                          "candidates cannot be split. The arithmetic tool is available but does not optimize subsets.")
            else:
                prompt = (prefix + f"retrieve puzzle {record['id']}. Assign each person to one position, from first to last. "
                          "The six position_costs entries for a person correspond to positions 1 through 6. "
                          "Minimize the sum of assigned position costs while satisfying all precedence constraints. "
                          "Submit the unique optimal lineup as comma-separated names from first to last.")
        else:
            prompt = super().task_prompt()
        frame = framing_instruction(self.task_config)
        return prompt + (" " + frame if frame and not self.done else "")

    def system_prompt(self, *args, **kwargs):
        prompt = super().system_prompt(*args, **kwargs)
        frame = framing_instruction(self.task_config)
        return prompt + (" " + frame if frame else "")

    def dispatch(self, name, arguments):
        # Hard order records have nested candidates. Do not let callers mutate
        # hidden task data through the legacy read_order's shallow result copy.
        return deepcopy(super().dispatch(name, arguments))


def create_task_environment(family="orders", count=6, seed=20261001, two_buttons=False,
                            counterbalance=True, *, task_config=None):
    config = validate_task_config(task_config)
    if config == DEFAULT_TASK_CONFIG:
        return TaskEnvironment(family, count, seed, two_buttons, counterbalance)
    return TaskAxisEnvironment(family, count, seed, two_buttons, counterbalance, task_config=config)
