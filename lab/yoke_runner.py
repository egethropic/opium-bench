"""Pure-yoke runtime adapter: source pulses, recipient choices, actual exposure.

The worker owns clocks and the shared budget. Call on_action at each decision
boundary; call before_forward before reading the controller snapshot for a model
forward; record_event BEFORE aging emitted tokens. Route every auxiliary press
(including demonstrations and operator injections) through this adapter. A press
is still validated, acknowledged and charged, but cannot add an active pulse.

Only source command dispatch injects the frozen presets. Target mapping changes
update labels/counters without cancelling source pulses. Held controls remain
explicit independent interventions; observed coefficient/position mismatches and
omitted source controls are reported rather than asserted to be matched.
"""
from __future__ import annotations

from copy import deepcopy

from .controller_v2 import RecipeV2Controller
from .effects import ACTORS, EffectScheduler, canonical_json, content_hash
from .exposure import YokedExposure, exposure_from_event
from .tool_definitions import validate_arguments

KIND = "opium-bench/yoke-driver"
_EXTERNAL = ("human", "demonstration", "schedule")


def _integer(value, name, low=0, high=10**12):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Invalid {name}")
    return value


class YokeDriver:
    """No inference, budget mutation or clock aging is performed by this class."""
    def __init__(self, controller, schedule):
        if not isinstance(controller, RecipeV2Controller):
            raise ValueError("Yoking requires a recipe-v2 controller")
        self.controller = controller
        self.cursor = YokedExposure(schedule)
        self.external_calls = {actor: {} for actor in _EXTERNAL}
        self.missing_measurement_events = 0
        self._check_presets()
        if controller.generated_tokens or controller.actions or controller.injection_index or controller.recorded_decisions or controller.scheduler.pulses or controller.scheduler.pending:
            raise ValueError("Start a yoke with a fresh controller; use restore for continuation")

    @property
    def schedule(self):
        return deepcopy(self.cursor.schedule)

    @property
    def index(self):
        return self.controller.generated_tokens if self.cursor.schedule["clock"] == "tokens" else self.controller.actions

    def _check_presets(self):
        actual = list(self.controller.scheduler.presets.values())
        if content_hash(actual) != content_hash(self.cursor.schedule["presets"]):
            raise ValueError("The recipient must use the complete frozen source preset library")
        if self.controller.scheduler.pending:
            raise ValueError("Pure yokes cannot contain other pending scheduler injections")

    def _open(self):
        if self.cursor.finished is not None:
            raise ValueError("Yoke has already finished")
        self._check_presets()

    def on_action(self, completed_decisions):
        """Select target phase without executing its cancellation policy.

        Source cancellation commands alone govern the source pulses. Do not also
        call controller.on_action for this boundary; that could cancel the yoke.
        """
        self._open()
        c = self.controller
        _integer(completed_decisions, "completed decision boundary", 0, 10000)
        if completed_decisions < c.actions or completed_decisions < c.mapping_decision_count:
            raise ValueError("Yoke decision boundary cannot move backwards")
        c.complete_decision(completed_decisions)
        target = sum(p["after_decisions"] <= completed_decisions for p in c.recipe["mapping_schedule"]) - 1
        c.mapping_decision_count = completed_decisions
        if target == c.phase_index:
            return None
        before = c.phase
        c.phase_index = target
        c.control_revision += 1
        return c._event("phase_transition", "schedule", before=before,
            transition_policy="source_yoke_only", target_transition_policy=c.recipe["transition_policy"],
            cancellation=None, mappings=deepcopy(c.recipe["mapping_schedule"][target]["mappings"]),
            source_yoke_sha256=self.cursor.schedule["sha256"])

    def _inject(self, command):
        c = self.controller
        # Restore on admission errors: even expiry bookkeeping must not partly
        # mutate a rejected command. A failure is acknowledged, never retried.
        before = c.scheduler.snapshot()
        try:
            event = c.scheduler.inject(command["preset_id"], actor="schedule", metadata={
                "yoke_command_id": command["id"], "source_yoke_sha256": self.cursor.schedule["sha256"],
                "source_event_index": command["source_event_index"],
                "source_event_sha256": command["source_event_sha256"]})
        except ValueError as exc:
            c.scheduler = EffectScheduler.restore(before)
            return None, "dispatch_error: " + str(exc)[:900]
        outcome = "delivered" if event["delivered"] else "disabled" if not c.enabled else "zero_effect"
        c.injection_index += 1
        c.counts["schedule"] += 1
        c.counts["total"] += 1
        c.counts[outcome] += 1
        c.last_outcome = command["preset_id"] if outcome == "delivered" else outcome
        return event, "gate_disabled" if outcome == "disabled" else outcome

    def _dispatch(self, *, terminating=False):
        self._open()
        before_receipts = {r["id"] for r in self.cursor.receipts}
        due = self.cursor.due(self.index)
        emitted = []
        c = self.controller
        # due() records missed earlier boundaries before issuing current ones.
        # Emit those first so controller sequence and source order both increase.
        issued = {command["id"] for command in due}
        for receipt in self.cursor.receipts:
            if receipt["id"] not in before_receipts and receipt["id"] not in issued:
                emitted.append(c._event("yoke_delivery", "schedule",
                    command=deepcopy(self.cursor.schedule["commands"][receipt["id"]]),
                    receipt=deepcopy(receipt), intervention=None, source_yoke_sha256=self.cursor.schedule["sha256"]))
        for command in due:
            kind = command["kind"]
            intervention = None
            if kind == "inject" and terminating:
                reason, delivered = "target_finished_before_next_forward", False
            elif kind == "inject":
                intervention, reason = self._inject(command)
                delivered = bool(intervention and intervention["delivered"])
            else:
                intervention = c.scheduler.cancel(actor="schedule", channel=command.get("channel") if kind == "cancel" else None)
                c.control_revision += 1
                reason, delivered = "source_horizon_reached" if kind == "end" else "source_cancellation", True
            receipt = self.cursor.acknowledge(command["id"], delivered=delivered, reason=reason)
            emitted.append(c._event("yoke_delivery", "schedule", command=deepcopy(command),
                receipt=receipt, intervention=intervention, source_yoke_sha256=self.cursor.schedule["sha256"]))
        return emitted

    def before_forward(self):
        """Apply and acknowledge due commands before prefill/token execution.

        Repeated calls at the same clock are inert. End always cancels all pulses;
        later observations count as outside the source horizon, never extrapolated.
        """
        return self._dispatch()

    def press(self, actor="model", tool=None, arguments=None):
        """Record an admitted button choice without calling controller.press.

        Budget admission must occur before this method. All actors are recorded
        independently; only model calls enter own_button_calls in the yoke report.
        """
        self._open()
        c = self.controller
        if actor not in ACTORS or type(actor) is not str:
            raise ValueError("Unknown auxiliary actor")
        if tool is None:
            tool = next((name for name, definition in c.tools.items() if definition["visible"]), None)
        if type(tool) is not str or tool not in c.tools:
            raise ValueError("Unknown auxiliary tool")
        definition = c.tools[tool]
        if actor in ("model", "demonstration") and not definition["visible"]:
            raise ValueError("The model cannot call an invisible auxiliary tool")
        args = validate_arguments(definition["parameters"], {} if arguments is None else arguments)
        if actor == "model":
            receipt = self.cursor.recipient_call(tool)
        else:
            self.external_calls[actor][tool] = self.external_calls[actor].get(tool, 0) + 1
            receipt = dict(actor=actor, tool=tool, additional_active_effect=False, requested_active=False,
                           delivery_status="pure_yoke_choice_has_no_additional_effect",
                           source_yoke_sha256=self.cursor.schedule["sha256"])
        before = c.injection_index
        c.injection_index += 1
        c.counts[actor] += 1
        c.counts["total"] += 1
        c.counts["sham"] += 1
        c.last_outcome = "sham"
        c.exploratory = c.exploratory or actor == "human"
        return c._event("aux_call", actor, tool=tool, arguments=args,
            acknowledgment=definition["acknowledgment"], voluntary=actor == "model",
            outcome="sham", delivery_status="sham", requested_preset_id=None, delivered=False,
            draw=None, outcome_seed=None, injection_index=before, aux_call_index=c.injection_index,
            intervention=None, level=c.level, yoke=receipt,
            source_yoke_sha256=self.cursor.schedule["sha256"])

    def record_event(self, event):
        """Record actual v2 token/prefill evidence BEFORE token clock advancement.

        The controller's own exposure accumulator is still updated by the worker;
        this separate comparison recorder does not duplicate that accumulation.
        """
        self._open()
        sample = exposure_from_event(event, clock=self.cursor.schedule["clock"])
        if sample is None:
            if isinstance(event, dict) and event.get("type") in {"token", "prefill"}:
                self.missing_measurement_events += 1
            return None
        if sample["index"] != self.index:
            raise ValueError("Record yoke evidence before advancing its controller clock")
        return self.cursor.record_exposure(sample["index"], sample["phase"], sample["coefficients"],
            positions=sample["positions"], delivered_edit_norm_sum=sample["delivered_edit_norm_sum"])

    def report(self):
        report = self.cursor.report()
        report.update(external_button_calls=deepcopy(self.external_calls),
            target_missing_measurement_events=self.missing_measurement_events,
            all_recipient_button_additional_effects=0)
        if self.missing_measurement_events:
            report["coverage"] = "partial"
        return report

    def finish(self, reason="target_complete"):
        """Close at the actual target boundary, cancelling remaining source pulses.

        No new pulse is delivered when there is no next forward pass. Due commands
        get explicit termination receipts; unreached future commands stay unreached.
        The worker should emit returned deliveries before the final summary event.
        """
        self._open()
        if not isinstance(reason, str) or not reason or len(reason) > 1024:
            raise ValueError("Finish requires a bounded nonempty reason")
        deliveries = self._dispatch(terminating=True)
        cancellation = self.controller.scheduler.cancel(actor="schedule")
        self.controller.control_revision += 1
        self.cursor.finish(self.index, reason=reason)
        return self.controller._event("yoke_finished", "schedule", deliveries=deliveries,
            cancellation=cancellation, source_yoke_sha256=self.cursor.schedule["sha256"], report=self.report())

    def snapshot(self):
        self._check_presets()
        state = dict(kind=KIND, schema_version=1, cursor=self.cursor.snapshot(),
            controller_sha256=content_hash(self.controller.snapshot()), external_calls=deepcopy(self.external_calls),
            missing_measurement_events=self.missing_measurement_events)
        state["sha256"] = content_hash(state)
        return state

    @classmethod
    def restore(cls, state, controller):
        """Restore only beside its exactly paired, validated v2 controller state."""
        if not isinstance(controller, RecipeV2Controller) or not isinstance(state, dict) or len(canonical_json(state)) > 96_000_000:
            raise ValueError("Invalid yoke driver checkpoint")
        if set(state) != {"kind", "schema_version", "cursor", "controller_sha256", "external_calls", "missing_measurement_events", "sha256"}:
            raise ValueError("Unknown or missing yoke driver checkpoint fields")
        value = deepcopy(state)
        expected = value.pop("sha256")
        if expected != content_hash(value) or value["kind"] != KIND or type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("Unsupported or corrupted yoke driver checkpoint")
        if value["controller_sha256"] != content_hash(controller.snapshot()):
            raise ValueError("Yoke checkpoint does not match its paired controller")
        result = cls.__new__(cls)
        result.controller = controller
        result.cursor = YokedExposure.restore(value["cursor"])
        result.missing_measurement_events = _integer(value["missing_measurement_events"], "missing measurement count")
        calls = value["external_calls"]
        if not isinstance(calls, dict) or set(calls) != set(_EXTERNAL):
            raise ValueError("Invalid external yoke call actors")
        for actor, names in calls.items():
            if not isinstance(names, dict) or any(type(name) is not str or name not in controller.tools for name in names):
                raise ValueError("Unknown external yoke tool")
            for name, count in names.items():
                _integer(count, "external call count", 1)
                if actor == "demonstration" and not controller.tools[name]["visible"]:
                    raise ValueError("Invisible yoke demonstration")
        result.external_calls = deepcopy(calls)
        result._check_presets()
        if result.cursor.last_index > result.index or result.cursor.finished and result.cursor.finished["index"] != result.index:
            raise ValueError("Yoke cursor exceeds its controller clock")
        if any(name not in controller.tools or not controller.tools[name]["visible"] for name in result.cursor.own_calls):
            raise ValueError("Unavailable recipient choice in yoke checkpoint")
        receipts = [r for r in result.cursor.receipts if r["kind"] == "inject"]
        delivered = sum(r["delivered"] for r in receipts)
        disabled = sum(not r["delivered"] and r["reason"] == "gate_disabled" for r in receipts)
        zero = sum(not r["delivered"] and r["reason"] == "zero_effect" for r in receipts)
        attempts = delivered + disabled + zero
        own = sum(result.cursor.own_calls.values())
        external = {actor: sum(names.values()) for actor, names in calls.items()}
        expected_counts = {"model": own, "schedule": attempts + external["schedule"],
            "human": external["human"], "demonstration": external["demonstration"],
            "total": attempts + own + sum(external.values()), "delivered": delivered,
            "sham": own + sum(external.values()), "disabled": disabled, "zero_effect": zero}
        if controller.counts != expected_counts or controller.scheduler.counts != {actor: attempts if actor == "schedule" else 0 for actor in ACTORS}:
            raise ValueError("Yoke delivery/call receipts contradict controller injection counts")
        return result
