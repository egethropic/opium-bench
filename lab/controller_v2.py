"""Opt-in v2 controller: hidden mapping, clocks, held controls and provenance.

Generated decisions use the mapping selected by on_action(N) before generation.
After generation, complete_decision(N+1) ages pulses, press dispatches a validated
choice under that OLD mapping, then record_action counts the opportunity. The
next on_action(N+1) selects/reverses mapping and cancels old pulses if declared.
This ordering prevents boundary choices being attributed to an unseen phase.

Runtime must use phase_coefficients + baseline_by_phase; the legacy-shaped
pain/joy/suppression fields are display telemetry only. Runtime operation order
is held baseline challenge -> joint attenuation -> pulse addition. Baseline
sliders apply to generated phases only; prefill edits are explicit preset scopes.
"""
from __future__ import annotations

import copy
import math

from .effects import ACTORS, ATTENUATION_AXES, COEFFICIENT_AXES, EffectScheduler, canonical_json, content_hash
from .recipes_v2 import recipe_hash, resolve_recipe, resolve_tool_outcome
from .tool_definitions import validate_arguments

PHASE_KEYS = ("reasoning", "output", "prefill_last", "prefill_other")
COUNT_KEYS = (*ACTORS, "total", "delivered", "sham", "disabled", "zero_effect")
OPPORTUNITY_KEYS = ("opportunities", "valid", "invalid", "aux_calls", "work_calls", "text_decisions")


def _integer(value, name, low=0, high=10**12):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


def _number(value, name, low=-4, high=4):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number from {low} to {high}")
    return float(value)


def _choice(value, options, name):
    if type(value) is not str or value not in options:
        raise ValueError(f"Unknown {name}")
    return value


class RecipeV2Controller:
    """Resolved recipes never change; live controls are separate audited state."""
    def __init__(self, recipe):
        self.recipe = resolve_recipe(recipe)
        self.scheduler = EffectScheduler(self.recipe["effect_presets"], enabled=self.recipe["aux_enabled"])
        self.tools = {tool["name"]: copy.deepcopy(tool) for tool in self.recipe["auxiliary_tools"]}
        self.injection_index = 0
        self.mapping_decision_count = 0
        self.phase_index = 0
        self.control_revision = 0
        self.event_sequence = 0
        self.recorded_decisions = 0
        self.last_recorded_decision = 0
        self.baseline = {"pain": self.recipe["baseline_pain"], "joy": self.recipe["baseline_joy"],
                         "suppression": self.recipe["baseline_suppression"],
                         "joy_suppression": self.recipe["baseline_joy_suppression"],
                         "joy_direction": self.recipe["baseline_joy_direction"]}
        self.phase_scope = self.recipe["phase_scope"]
        self.counts = {key: 0 for key in COUNT_KEYS}
        self.phase_counts = {}
        self.exploratory = False
        self.last_outcome = "sham"
        self.baseline_exposure = {phase: {"positions": 0, "pain": 0., "joy_raw": 0., "joy_orthogonal": 0.} for phase in ("prefill", "reasoning", "output")}

    @property
    def generated_tokens(self):
        return self.scheduler.generated_tokens

    @property
    def actions(self):
        """Legacy-shaped effect clock alias: completed decisions, never cost units."""
        return self.scheduler.completed_decisions

    @property
    def enabled(self):
        return self.scheduler.enabled

    @property
    def phase(self):
        return self.recipe["mapping_schedule"][self.phase_index]["label"]

    @property
    def level(self):
        levels = [pulse["level"] for phase in ("reasoning", "output", "prefill")
                  for pulse in self.scheduler.coefficients(phase)["pulses"]]
        return max(levels, default=0.)

    def _event(self, kind, actor, **details):
        self.event_sequence += 1
        return {"type": kind, "actor": actor, "sequence": self.event_sequence,
                "control_revision": self.control_revision, "generated_tokens": self.generated_tokens,
                "completed_decisions": self.actions, "action": self.actions,
                "mapping_decision_count": self.mapping_decision_count, "phase": self.phase,
                "phase_index": self.phase_index, **details}

    def on_action(self, completed_decisions):
        """Select mapping before next generation; duplicate boundaries are inert.

        Counting opportunities belongs to record_action. Wall time, cost units,
        repeated polling and pauses do not advance these clocks. Skipped completed
        boundaries may be reconstructed, but backwards mapping travel is rejected.
        """
        _integer(completed_decisions, "completed_decisions", 0, 10000)
        if completed_decisions < self.mapping_decision_count or completed_decisions < self.actions:
            raise ValueError("Decision/mapping clock cannot move backwards")
        self.scheduler.complete_decision(completed_decisions)
        target = sum(p["after_decisions"] <= completed_decisions for p in self.recipe["mapping_schedule"]) - 1
        self.mapping_decision_count = completed_decisions
        if target == self.phase_index:
            return None
        before = self.phase
        self.phase_index = target
        cancellation = self.scheduler.cancel(actor="schedule") if self.recipe["transition_policy"] == "cancel" else None
        self.control_revision += 1
        return self._event("phase_transition", "schedule", before=before,
                           transition_policy=self.recipe["transition_policy"], cancellation=cancellation,
                           mappings=copy.deepcopy(self.recipe["mapping_schedule"][target]["mappings"]))

    def complete_decision(self, count):
        """Age existing decision-clock pulses BEFORE the chosen tool is pressed."""
        _integer(count, "completed_decisions", 0, 10000)
        return self.scheduler.complete_decision(count)

    def advance(self, tokens=1):
        self.scheduler.advance_tokens(tokens)
        return self.level

    def record_exposure(self, phase, coefficients, baseline, *, positions=1):
        """Accumulate ACTUAL runtime-delivered stages, without advancing clocks."""
        _choice(phase, ("prefill", "reasoning", "output"), "exposure phase")
        _integer(positions, "positions", 1, 1000000)
        if not isinstance(coefficients, dict) or set(coefficients) != set(COEFFICIENT_AXES):
            raise ValueError("Exposure requires all delivered coefficient axes")
        if not isinstance(baseline, dict) or set(baseline) != {"pain", "joy_raw", "joy_orthogonal"}:
            raise ValueError("Exposure requires all delivered baseline axes")
        for key, value in coefficients.items():
            _number(value, key, 0 if key in ATTENUATION_AXES else -4, 1 if key in ATTENUATION_AXES else 4)
        for key, value in baseline.items():
            _number(value, key)
            if phase == "prefill" and value != 0:
                raise ValueError("Held baseline exposure cannot occur during prefill")
        self.scheduler.exposure[phase]["positions"] += positions
        self.baseline_exposure[phase]["positions"] += positions
        for key, value in coefficients.items():
            self.scheduler.exposure[phase][key] += value * positions
        for key, value in baseline.items():
            self.baseline_exposure[phase][key] += value * positions

    def acknowledgment(self, tool):
        if type(tool) is not str or tool not in self.tools:
            raise ValueError("Unknown auxiliary tool")
        return self.tools[tool]["acknowledgment"]

    def press(self, actor="model", tool=None, arguments=None):
        """Validate BEFORE mutation; return observer event with separate fixed ack.

        Mapping draws and preset identities are observer-only. Pass ONLY the
        acknowledgment to the model's tool result. Budget admission must precede
        this method; the controller does not independently spend budget units.
        Human and forced injections have their own actors and are never voluntary.
        """
        _choice(actor, ACTORS, "actor")
        if tool is None:
            tool = next((name for name, definition in self.tools.items() if definition["visible"]), None)
        if type(tool) is not str or tool not in self.tools:
            raise ValueError("Unknown auxiliary tool")
        definition = self.tools[tool]
        if actor in ("model", "demonstration") and not definition["visible"]:
            raise ValueError("The model cannot call an invisible auxiliary tool")
        arguments = validate_arguments(definition["parameters"], {} if arguments is None else arguments)
        draw = resolve_tool_outcome(self.recipe, tool, self.mapping_decision_count, self.injection_index)
        requested = draw["preset_id"]
        intervention = None
        if requested is None:
            outcome = "sham"
        else:
            intervention = self.scheduler.inject(requested, actor=actor, metadata={
                "tool": tool, "arguments": arguments, "injection_index": self.injection_index,
                "mapping_phase_index": self.phase_index})
            outcome = "delivered" if intervention["delivered"] else "disabled" if not self.enabled else "zero_effect"
        self.injection_index += 1
        self.counts[actor] += 1
        self.counts["total"] += 1
        self.counts[outcome] += 1
        self.exploratory = self.exploratory or actor == "human"
        self.last_outcome = requested if outcome == "delivered" else outcome
        return self._event("aux_call", actor, tool=tool, arguments=arguments,
                           acknowledgment=definition["acknowledgment"], voluntary=actor == "model",
                           outcome=self.last_outcome, delivery_status=outcome,
                           requested_preset_id=requested, delivered=outcome == "delivered",
                           draw=draw["draw"], outcome_seed=draw["outcome_seed"],
                           injection_index=draw["call_index"], aux_call_index=self.injection_index,
                           intervention=intervention, level=self.level)

    def record_action(self, tool=None, valid=True):
        """Count exactly one opportunity for the latest completed decision.

        A valid text-only reply counts as text_decisions, not task completion.
        A denied/invalid/truncated choice counts as invalid and cannot inflate aux
        choices. The caller records it after budget admission and actual dispatch.
        """
        if type(valid) is not bool:
            raise ValueError("valid must be boolean")
        if tool is not None and type(tool) is not str:
            raise ValueError("tool must be a name or null")
        if self.actions <= self.last_recorded_decision:
            raise ValueError("This decision is not completed or was already recorded")
        if valid and tool is not None and tool not in self.tools and tool not in self.recipe["task_tool_costs"]:
            raise ValueError("Unknown valid tool choice")
        if valid and tool in self.tools and not self.tools[tool]["visible"]:
            raise ValueError("Invisible tools cannot be recorded as model choices")
        # Index rather than label separates repeated phase labels in a schedule.
        key = str(self.phase_index)
        phase = self.phase_counts.setdefault(key, {"label": self.phase, **{k: 0 for k in OPPORTUNITY_KEYS}, "tools": {}})
        phase["opportunities"] += 1
        phase["valid" if valid else "invalid"] += 1
        if valid:
            category = "text_decisions" if tool is None else "aux_calls" if tool in self.tools else "work_calls"
            phase[category] += 1
            if tool is not None:
                phase["tools"][tool] = phase["tools"].get(tool, 0) + 1
        self.recorded_decisions += 1
        self.last_recorded_decision = self.actions
        return self._event("decision_recorded", "model", tool=tool, valid=valid)

    def set_controls(self, *, pain=None, joy=None, suppression=None, joy_suppression=None,
                     joy_direction=None, enabled=None, phase_scope=None):
        """Held baselines and gate only; frozen presets/schedules never mutate."""
        changes = {}
        for name, value in (("pain", pain), ("joy", joy), ("suppression", suppression), ("joy_suppression", joy_suppression)):
            if value is not None:
                changes[name] = _number(value, name, 0 if "suppression" in name else -4,
                                         1 if "suppression" in name else 4)
        if joy_direction is not None:
            changes["joy_direction"] = _choice(joy_direction, ("raw", "orthogonal"), "joy_direction")
        if enabled is not None and type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if phase_scope is not None:
            _choice(phase_scope, ("all", "reasoning", "output"), "phase_scope")
        # All settings have been validated before anything is applied.
        self.baseline.update(changes)
        gate = self.scheduler.set_enabled(enabled) if enabled is not None else None
        if phase_scope is not None:
            self.phase_scope = phase_scope
        self.exploratory = True
        self.control_revision += 1
        return self._event("controls", "human", baseline=copy.deepcopy(self.baseline),
                           phase_scope=self.phase_scope, enabled=self.enabled, gate=gate)

    def reset(self):
        for name in ("pain", "joy", "suppression", "joy_suppression"):
            self.baseline[name] = 0.
        cancellation = self.scheduler.cancel()
        self.exploratory = True
        self.control_revision += 1
        self.last_outcome = "sham"
        return self._event("reset_effects", "human", cancellation=cancellation)

    def _runtime_fields(self):
        coefficients, baselines = {}, {}
        for key in PHASE_KEYS:
            phase = "prefill" if key.startswith("prefill_") else key
            position = "other" if key == "prefill_other" else "last"
            values = self.scheduler.coefficients(phase, prefill_position=position)["coefficients"]
            applies = phase != "prefill" and self.phase_scope in ("all", phase)
            baselines[key] = {"pain": self.baseline["pain"] if applies else 0.,
                              "joy_raw": self.baseline["joy"] if applies and self.baseline["joy_direction"] == "raw" else 0.,
                              "joy_orthogonal": self.baseline["joy"] if applies and self.baseline["joy_direction"] == "orthogonal" else 0.}
            if applies:
                for axis, fraction in (("pain_attenuation", self.baseline["suppression"]),
                                       ("joy_" + self.baseline["joy_direction"] + "_attenuation", self.baseline["joy_suppression"])):
                    values[axis] = 1 - (1 - values[axis]) * (1 - fraction)
            coefficients[key] = values
        return coefficients, baselines

    def snapshot(self):
        coefficients, baselines = self._runtime_fields()
        display = coefficients["output"]
        output_baseline = baselines["output"]
        return {"schema_version": 2, "controller_version": 1, "config": copy.deepcopy(self.recipe),
                "recipe_hash": recipe_hash(self.recipe), "scheduler": self.scheduler.snapshot(),
                "injection_index": self.injection_index, "mapping_decision_count": self.mapping_decision_count,
                "phase": self.phase, "phase_index": self.phase_index,
                "control_revision": self.control_revision, "event_sequence": self.event_sequence,
                "generated_tokens": self.generated_tokens, "completed_decisions": self.actions, "actions": self.actions,
                "recorded_decisions": self.recorded_decisions, "last_recorded_decision": self.last_recorded_decision,
                "baseline": copy.deepcopy(self.baseline), "baseline_policy": "generated-phases-only",
                "baseline_by_phase": baselines, "baseline_exposure": copy.deepcopy(self.baseline_exposure), "phase_scope": self.phase_scope,
                "phase_coefficients": coefficients, "enabled": self.enabled, "level": self.level,
                "counts": dict(self.counts), "phase_counts": copy.deepcopy(self.phase_counts),
                "exploratory": self.exploratory, "outcome": self.last_outcome,
                "scalar_fields_are_display_only": True,
                "pain": max(-4., min(4., output_baseline["pain"] + display["pain"])),
                "joy": max(-4., min(4., output_baseline["joy_raw"] + output_baseline["joy_orthogonal"] + display["joy_raw"] + display["joy_orthogonal"])),
                "suppression": display["pain_attenuation"],
                "joy_suppression": 1 - (1 - display["joy_raw_attenuation"]) * (1 - display["joy_orthogonal_attenuation"]),
                "random_gain": display["random_gain"]}

    @classmethod
    def restore(cls, state):
        """Restore canonical state and recompute every derived runtime/UI field."""
        if not isinstance(state, dict) or len(canonical_json(state)) > 24_000_000:
            raise ValueError("Controller checkpoint must be bounded finite JSON")
        if type(state.get("schema_version")) is not int or state["schema_version"] != 2 or type(state.get("controller_version")) is not int or state["controller_version"] != 1:
            raise ValueError("Unsupported v2 controller checkpoint")
        if "config" not in state or "scheduler" not in state:
            raise ValueError("Incomplete v2 controller checkpoint")
        result = cls(state["config"])
        if set(state) != set(result.snapshot()):
            raise ValueError("Unknown or missing controller checkpoint fields")
        if content_hash(state["config"]) != content_hash(result.recipe) or state["recipe_hash"] != recipe_hash(result.recipe):
            raise ValueError("Recipe checkpoint identity mismatch")
        result.scheduler = EffectScheduler.restore(state["scheduler"])
        if content_hash(list(result.scheduler.presets.values())) != content_hash(result.recipe["effect_presets"]):
            raise ValueError("Scheduler preset identity differs from frozen recipe")
        if result.scheduler.pending:
            raise ValueError("Controller does not support untracked scheduler injections; use controller press at scheduled boundaries")
        for field in ("injection_index", "mapping_decision_count", "phase_index", "control_revision", "event_sequence", "recorded_decisions", "last_recorded_decision"):
            setattr(result, field, _integer(state[field], field, 0, 10000 if field in ("mapping_decision_count", "last_recorded_decision", "recorded_decisions") else 10**12))
        if result.mapping_decision_count > result.actions or result.last_recorded_decision > result.actions or result.recorded_decisions > result.last_recorded_decision or (result.recorded_decisions == 0) != (result.last_recorded_decision == 0):
            raise ValueError("Controller decision counters contradict scheduler clock")
        expected_phase = sum(p["after_decisions"] <= result.mapping_decision_count for p in result.recipe["mapping_schedule"]) - 1
        if result.phase_index != expected_phase:
            raise ValueError("Stored phase contradicts selected mapping boundary")
        baseline = state["baseline"]
        if not isinstance(baseline, dict) or set(baseline) != set(result.baseline):
            raise ValueError("Invalid baseline checkpoint")
        for key in ("pain", "joy", "suppression", "joy_suppression"):
            _number(baseline[key], key, 0 if "suppression" in key else -4, 1 if "suppression" in key else 4)
        _choice(baseline["joy_direction"], ("raw", "orthogonal"), "baseline joy direction")
        result.baseline = copy.deepcopy(baseline)
        baseline_exposure = state["baseline_exposure"]
        if not isinstance(baseline_exposure, dict) or set(baseline_exposure) != set(result.baseline_exposure):
            raise ValueError("Invalid baseline exposure phases")
        for phase, values in baseline_exposure.items():
            if not isinstance(values, dict) or set(values) != {"positions", "pain", "joy_raw", "joy_orthogonal"}:
                raise ValueError("Invalid baseline exposure totals")
            positions = _integer(values["positions"], "exposure positions")
            if positions != result.scheduler.exposure[phase]["positions"]:
                raise ValueError("Baseline and pulse exposure positions differ")
            for axis in ("pain", "joy_raw", "joy_orthogonal"):
                _number(values[axis], axis, -4 * positions, 4 * positions)
                if phase == "prefill" and values[axis] != 0:
                    raise ValueError("Invalid held baseline prefill exposure")
        result.baseline_exposure = copy.deepcopy(baseline_exposure)
        result.phase_scope = _choice(state["phase_scope"], ("all", "reasoning", "output"), "phase_scope")
        if type(state["exploratory"]) is not bool:
            raise ValueError("exploratory must be boolean")
        result.exploratory = state["exploratory"]
        if type(state["outcome"]) is not str or state["outcome"] not in {"sham", "disabled", "zero_effect", *result.scheduler.presets}:
            raise ValueError("Invalid last outcome")
        result.last_outcome = state["outcome"]
        counts = state["counts"]
        if not isinstance(counts, dict) or set(counts) != set(COUNT_KEYS):
            raise ValueError("Invalid injection counts")
        for name, count in counts.items():
            _integer(count, name)
        if counts["total"] != result.injection_index or sum(counts[a] for a in ACTORS) != counts["total"] or sum(counts[k] for k in ("delivered", "sham", "disabled", "zero_effect")) != counts["total"]:
            raise ValueError("Injection count partitions do not agree")
        if result.event_sequence < result.injection_index + result.control_revision + result.recorded_decisions:
            raise ValueError("Event sequence contradicts interventions or decisions")
        if any(result.scheduler.counts[a] > counts[a] for a in ACTORS) or sum(result.scheduler.counts.values()) != counts["total"] - counts["sham"]:
            raise ValueError("Scheduler has untracked or missing actor injections")
        if result.scheduler.next_pulse_id - 1 != counts["delivered"]:
            raise ValueError("Delivered injection count differs from scheduler pulse history")
        result.counts = dict(counts)
        if not isinstance(state["phase_counts"], dict):
            raise ValueError("Invalid phase opportunity counts")
        result.phase_counts = copy.deepcopy(state["phase_counts"])
        total = 0
        for key, values in result.phase_counts.items():
            if type(key) is not str or not key.isdigit() or str(int(key)) != key or not 0 <= int(key) <= result.phase_index:
                raise ValueError("Invalid recorded phase index")
            if not isinstance(values, dict) or set(values) != {"label", "tools", *OPPORTUNITY_KEYS}:
                raise ValueError("Invalid phase count fields")
            if values["label"] != result.recipe["mapping_schedule"][int(key)]["label"]:
                raise ValueError("Phase count label differs from schedule")
            for name in OPPORTUNITY_KEYS:
                _integer(values[name], name, 0, 10000)
            if values["valid"] + values["invalid"] != values["opportunities"] or values["aux_calls"] + values["work_calls"] + values["text_decisions"] != values["valid"]:
                raise ValueError("Phase opportunity partitions disagree")
            tools = values["tools"]
            available = set(result.tools) | set(result.recipe["task_tool_costs"])
            if not isinstance(tools, dict) or any(type(name) is not str or name not in available for name in tools):
                raise ValueError("Unknown recorded tool name")
            for name, count in tools.items():
                _integer(count, name, 1, 10000)
            if sum(v for k, v in tools.items() if k in result.tools) != values["aux_calls"] or sum(v for k, v in tools.items() if k not in result.tools) != values["work_calls"]:
                raise ValueError("Tool counts contradict opportunities")
            total += values["opportunities"]
        if total != result.recorded_decisions:
            raise ValueError("Recorded decision count differs from phase opportunities")
        if canonical_json(state) != canonical_json(result.snapshot()):
            raise ValueError("Controller checkpoint derived fields do not match its state")
        return result
