"""Opt-in weighted shared budgets with auditable decision receipts.

Budget units, completed decisions and generated tokens are different quantities.
The caller reserves a base charge before generation; validates a returned call;
then completes the decision with the tool's EXTRA charge before dispatching it.
Only a receipt with dispatch_allowed=True authorizes work/effect dispatch. This
module never mutates an environment or intervention, and has no model access.
"""
from __future__ import annotations

import copy
import re

from .effects import canonical_json

STATUSES = ("valid", "invalid", "truncated", "stopped", "error")


def _integer(value, name, low=0, high=1_000_000):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


def _name(value):
    if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", value)):
        raise ValueError("Invalid tool name")
    return value


class WeightedBudget:
    """One in-flight generation at a time; every attempt incurs its base charge.

    Ending a truncated, stopped, invalid or failed generation does not refund the
    base charge. Insufficient extra units deny dispatch without deducting the
    extra cost. The attempt still consumes one completed decision and its base
    charge. Prompts, tool replies and external injections do not consume this
    ledger unless a protocol deliberately represents them as model decisions.
    """
    def __init__(self, actions, tokens, *, base_cost=1):
        self.action_limit = _integer(actions, "action budget", 1, 10000)
        self.token_limit = _integer(tokens, "token budget", 1, 1_000_000)
        self.base_cost = _integer(base_cost, "base decision cost", 1, self.action_limit)
        self.initial_action_limit = self.action_limit
        self.initial_token_limit = self.token_limit
        self.grants = []
        self.action_units = 0
        self.completed_decisions = 0
        self.attempts = 0
        self.tokens = 0
        self.reasoning_tokens = 0
        self.output_tokens = 0
        self.in_flight = None
        self.receipts = []

    @property
    def actions(self):
        """Compatibility display alias; never use this as a decay clock."""
        return self.action_units

    @property
    def exhausted(self):
        return self.tokens >= self.token_limit or self.action_limit - self.action_units < self.base_cost

    @property
    def can_begin(self):
        return self.in_flight is None and not self.exhausted

    def begin_decision(self):
        """Reserve base cost BEFORE calling generation; returns an attempt id."""
        if self.in_flight is not None:
            raise ValueError("A generation attempt is already in flight")
        if self.exhausted:
            raise ValueError("Shared budget cannot afford another generation attempt")
        self.attempts += 1
        self.action_units += self.base_cost
        self.in_flight = {"attempt_id": self.attempts, "base_charge": self.base_cost,
                          "units_before": self.action_units - self.base_cost,
                          "tokens_before": self.tokens, "reasoning_before": self.reasoning_tokens,
                          "output_before": self.output_tokens}
        return self.attempts

    def consume_tokens(self, count=1, phase="output"):
        """Count ALL emitted tokens, including reasoning, syntax and stop tokens."""
        _integer(count, "token increment")
        if not isinstance(phase, str) or phase not in ("reasoning", "output"):
            raise ValueError("Unknown emitted-token phase")
        if self.in_flight is None:
            raise ValueError("Generated tokens require an active generation attempt")
        if self.tokens + count > self.token_limit:
            raise ValueError("Shared token budget would be exceeded")
        before = self.tokens
        self.tokens += count
        if phase == "reasoning":
            self.reasoning_tokens += count
        else:
            self.output_tokens += count
        return {"attempt_id": self.in_flight["attempt_id"], "phase": phase,
                "count": count, "tokens_before": before, "tokens_after": self.tokens}

    def complete_decision(self, attempt_id, *, status="valid", tool_name=None, extra_cost=0):
        """Finish one attempt and atomically admit or deny the extra tool cost.

        Validate syntax AND bounded arguments before passing status='valid'.
        This method records completed decisions even on denial. No invalid or
        partial call executes. A valid text-only chat turn has no tool_name and
        no extra charge. Denied expensive choices are retained separately.
        """
        _integer(attempt_id, "attempt_id", 1, 10000)
        if self.in_flight is None or attempt_id != self.in_flight["attempt_id"]:
            raise ValueError("Attempt is not the active uncompleted generation")
        if not isinstance(status, str) or status not in STATUSES:
            raise ValueError("Unknown generation completion status")
        _name(tool_name)
        _integer(extra_cost, "extra tool cost", 0, 10000)
        if extra_cost and (status != "valid" or tool_name is None):
            raise ValueError("Only a validated tool call can request an extra cost")
        valid_tool = status == "valid" and tool_name is not None
        affordable = extra_cost <= self.action_limit - self.action_units
        allowed = valid_tool and affordable
        extra_charge = extra_cost if allowed else 0
        attempted = self.in_flight
        self.action_units += extra_charge
        self.completed_decisions += 1
        receipt = {"attempt_id": attempt_id, "completed_decision": self.completed_decisions,
                   "status": status, "tool_name": tool_name, "base_charge": self.base_cost,
                   "requested_extra": extra_cost, "extra_charge": extra_charge,
                   "total_charge": self.base_cost + extra_charge,
                   "units_before": attempted["units_before"], "units_after": self.action_units,
                   "tokens_before": attempted["tokens_before"], "tokens_after": self.tokens,
                   "reasoning_tokens": self.reasoning_tokens - attempted["reasoning_before"],
                   "output_tokens": self.output_tokens - attempted["output_before"],
                   "affordable": affordable, "dispatch_allowed": allowed,
                   "denial_reason": "insufficient_action_units" if valid_tool and not affordable else None}
        self.receipts.append(receipt)
        self.in_flight = None
        return copy.deepcopy(receipt)

    def grant_remaining(self, actions, tokens, *, reason="fresh_budget_branch"):
        """Set an explicit fresh remaining allowance at a completed boundary.

        Clocks/counters and historical costs never reset. Final limits become
        spent + requested allowance. A durable grant ledger preserves which
        earlier expensive choices were unaffordable under their original limits.
        The caller must notify the model and record branch provenance separately.
        """
        if self.in_flight is not None:
            raise ValueError("Budget grants require a completed decision boundary")
        _integer(actions, "new remaining action units", self.base_cost, 10000)
        _integer(tokens, "new remaining token allowance", 1, 1_000_000)
        action_limit = _integer(self.action_units + actions, "extended action limit", 1, 10000)
        token_limit = _integer(self.tokens + tokens, "extended token limit", 1, 1_000_000)
        if not isinstance(reason, str) or not 1 <= len(reason) <= 160 or any(ord(c) < 32 for c in reason):
            raise ValueError("Grant reason must be bounded text")
        if len(self.grants) >= 1000:
            raise ValueError("Maximum boundary grant count reached")
        receipt = {"grant_id": len(self.grants) + 1, "after_completed_decisions": self.completed_decisions,
                   "action_units": self.action_units, "tokens": self.tokens,
                   "previous_action_limit": self.action_limit, "previous_token_limit": self.token_limit,
                   "remaining_actions": actions, "remaining_tokens": tokens,
                   "action_limit": action_limit, "token_limit": token_limit, "reason": reason}
        self.action_limit = action_limit
        self.token_limit = token_limit
        self.grants.append(receipt)
        return copy.deepcopy(receipt)

    def extend_remaining(self, actions, tokens, *, reason="fresh_budget_branch"):
        """Alias: allowance is set to these values, not added to unused allowance."""
        return self.grant_remaining(actions, tokens, reason=reason)

    def snapshot(self):
        return {"schema_version": 1, "action_limit": self.action_limit, "token_limit": self.token_limit,
                "initial_action_limit": self.initial_action_limit, "initial_token_limit": self.initial_token_limit,
                "grants": copy.deepcopy(self.grants),
                "base_cost": self.base_cost, "actions": self.action_units, "action_units": self.action_units,
                "completed_decisions": self.completed_decisions, "attempts": self.attempts,
                "tokens": self.tokens, "reasoning_tokens": self.reasoning_tokens, "output_tokens": self.output_tokens,
                "actions_remaining": self.action_limit - self.action_units,
                "tokens_remaining": self.token_limit - self.tokens, "exhausted": self.exhausted,
                "in_flight": copy.deepcopy(self.in_flight), "receipts": copy.deepcopy(self.receipts)}

    @classmethod
    def restore(cls, state, *, allow_in_flight=False):
        """Validate receipt arithmetic; boundary restore is the default.

        allow_in_flight is for forensic/accounting recovery only: it cannot
        reconstruct an interrupted model generation. A caller may close that
        attempt as stopped/error while retaining its already spent budget.
        """
        if type(allow_in_flight) is not bool:
            raise ValueError("allow_in_flight must be boolean")
        if not isinstance(state, dict) or len(canonical_json(state)) > 20_000_000:
            raise ValueError("Budget checkpoint must be bounded finite JSON")
        expected = {"schema_version", "action_limit", "token_limit", "initial_action_limit", "initial_token_limit", "grants", "base_cost", "actions", "action_units",
                    "completed_decisions", "attempts", "tokens", "reasoning_tokens", "output_tokens",
                    "actions_remaining", "tokens_remaining", "exhausted", "in_flight", "receipts"}
        if set(state) != expected or type(state["schema_version"]) is not int or state["schema_version"] != 1:
            raise ValueError("Unsupported or incomplete weighted budget checkpoint")
        result = cls(state["initial_action_limit"], state["initial_token_limit"], base_cost=state["base_cost"])
        if not isinstance(state["grants"], list) or len(state["grants"]) > 1000:
            raise ValueError("Invalid budget grant ledger")
        grant_index = 0
        def apply_grants():
            nonlocal grant_index
            while grant_index < len(state["grants"]):
                grant = state["grants"][grant_index]
                if not isinstance(grant, dict) or not {"after_completed_decisions", "remaining_actions", "remaining_tokens", "reason"} <= set(grant):
                    raise ValueError("Invalid grant receipt")
                boundary = _integer(grant["after_completed_decisions"], "grant boundary", 0, 10000)
                if boundary < result.completed_decisions:
                    raise ValueError("Budget grants are not chronologically ordered")
                if boundary > result.completed_decisions:
                    break
                actual = result.grant_remaining(grant["remaining_actions"], grant["remaining_tokens"], reason=grant["reason"])
                if canonical_json(actual) != canonical_json(grant):
                    raise ValueError("Grant receipt contradicts its historical boundary")
                grant_index += 1
        if not isinstance(state["receipts"], list) or len(state["receipts"]) > 10000:
            raise ValueError("Invalid receipt list")
        # Replay the accounting-only ledger; each receipt must be exactly the
        # consequence of the preceding receipt, not merely plausible totals.
        for receipt in state["receipts"]:
            apply_grants()
            if not isinstance(receipt, dict):
                raise ValueError("Invalid decision receipt")
            for key in ("reasoning_tokens", "output_tokens", "attempt_id", "requested_extra", "status", "tool_name"):
                if key not in receipt:
                    raise ValueError("Incomplete decision receipt")
            attempt = result.begin_decision()
            result.consume_tokens(receipt["reasoning_tokens"], "reasoning")
            result.consume_tokens(receipt["output_tokens"], "output")
            generated = result.complete_decision(attempt, status=receipt["status"],
                                                 tool_name=receipt["tool_name"], extra_cost=receipt["requested_extra"])
            if canonical_json(generated) != canonical_json(receipt):
                raise ValueError("Receipt contradicts weighted budget accounting")
        apply_grants()
        if grant_index != len(state["grants"]):
            raise ValueError("Grant refers to a future uncompleted boundary")
        pending = state["in_flight"]
        if pending is not None:
            if not allow_in_flight:
                raise ValueError("Only completed decision boundaries can be resumed")
            if not isinstance(pending, dict):
                raise ValueError("Malformed in-flight attempt")
            result.begin_decision()
            if result.in_flight != pending:
                raise ValueError("In-flight attempt contradicts its preceding budget")
            reason = _integer(state["reasoning_tokens"], "reasoning_tokens") - result.reasoning_tokens
            output = _integer(state["output_tokens"], "output_tokens") - result.output_tokens
            result.consume_tokens(reason, "reasoning")
            result.consume_tokens(output, "output")
        if canonical_json(result.snapshot()) != canonical_json(state):
            raise ValueError("Weighted budget checkpoint totals or fields do not match receipts")
        return result
