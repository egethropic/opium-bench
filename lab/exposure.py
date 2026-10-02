"""Auditable pure-yoked pulse schedules and observed exposure coverage.

Matching injection indices does not guarantee matching numeric/behavioral effects.
Only delivered source pulses with frozen preset identities are scheduled. Recipient
choices are recorded but always request zero additional active effect. Missing
telemetry and early termination remain explicit; no source horizon is extrapolated.
"""
from __future__ import annotations

from copy import deepcopy
from functools import wraps
import hashlib
import json
import math
import re

from .effects import ATTENUATION_AXES, COEFFICIENT_AXES, content_hash, validate_presets

KIND = "opium-bench/yoked-exposure"
PHASES = {"reasoning", "output", "prefill_last", "prefill_other"}
AXES = (*COEFFICIENT_AXES, "baseline_pain", "baseline_joy_raw", "baseline_joy_orthogonal")
SHA = re.compile(r"^[0-9a-f]{64}$")
MAX_BYTES = 64 * 1024**2
MAX_EVENTS = 200000


def _checked(function):
    @wraps(function)
    def validate_shape(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, TypeError, AttributeError, IndexError, OverflowError, UnicodeError) as exc:
            raise ValueError("Malformed research evidence structure") from exc
    return validate_shape


def _canonical(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Exposure evidence must be finite JSON") from exc
    if len(raw.encode()) > MAX_BYTES:
        raise ValueError("Exposure evidence exceeds size limit")
    return raw


def _plain(value):
    count, stack = 0, [(value, 0)]
    while stack:
        item, depth = stack.pop()
        count += 1
        if depth > 64 or count > 3_000_000:
            raise ValueError("Exposure evidence exceeds structural bounds")
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError("Exposure object keys must be strings")
            stack.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            stack.extend((v, depth + 1) for v in item)
        elif item is not None and type(item) not in {str, int, float, bool}:
            raise ValueError("Unsupported exposure evidence value")
    return json.loads(_canonical(value))


def _integer(value, name, low=0, high=1_000_000):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Invalid {name}")
    return value


def _id(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value):
        raise ValueError(f"Invalid {name}")
    return value


def _hash(value, name):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError(f"Invalid {name}")
    return value


def _seal(value):
    value = _plain(value)
    return {**value, "sha256": content_hash(value)}


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"Invalid {name}")


def _source_events(value):
    if isinstance(value, bytes):
        if len(value) > MAX_BYTES:
            raise ValueError("Source event bytes exceed size limit")
        raw, events, lines = value, [], []
        def pairs(items):
            result = {}
            for key, item in items:
                if key in result:
                    raise ValueError("Duplicate source JSON key")
                result[key] = item
            return result
        for line in raw.splitlines(keepends=True):
            if not line.strip():
                raise ValueError("Source JSONL must contain one record per nonempty line")
            try:
                event = json.loads(line, object_pairs_hook=pairs,
                                   parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite source value")))
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise ValueError("Use a complete, valid event prefix for a yoke") from exc
            if not isinstance(event, dict):
                raise ValueError("Source event must be an object")
            events.append(event)
            lines.append(line)
        encoding = "raw-jsonl"
    elif isinstance(value, list):
        events = _plain(value)
        if any(not isinstance(event, dict) for event in events):
            raise ValueError("Source event must be an object")
        raw, lines, encoding = _canonical(events).encode(), None, "canonical-json-array"
    else:
        raise ValueError("Source events must be JSONL bytes or a JSON event array")
    if len(events) > MAX_EVENTS:
        raise ValueError("Source has too many events")
    _plain(events)
    return events, raw, lines, encoding


def _coefficients(value):
    _keys(value, AXES, "complete applied coefficient vector")
    result = {}
    for axis, coefficient in value.items():
        low, high = (0, 1) if axis in ATTENUATION_AXES else (-4, 4)
        if type(coefficient) not in {int, float} or not math.isfinite(coefficient) or not low <= coefficient <= high:
            raise ValueError("Invalid applied exposure coefficient")
        result[axis] = float(coefficient)
    return result


@_checked
def exposure_from_event(event, *, clock):
    """Normalize an actual runtime-v2 hook measurement, without inventing gaps."""
    if clock not in {"tokens", "decisions"}:
        raise ValueError("Yoke clock must be tokens or decisions")
    if not isinstance(event, dict) or event.get("type") not in {"token", "prefill"}:
        return None
    dose = event.get("dose")
    if not isinstance(dose, dict) or dose.get("schema_version") != 2 or type(dose.get("schema_version")) is not int:
        return None
    index = dose.get("generated_token_index" if clock == "tokens" else "completed_decisions")
    _integer(index, "measured global index")
    coefficients = _coefficients(dose.get("effective"))
    if type(dose.get("applied")) is not bool or dose["applied"] != any(coefficients.values()):
        raise ValueError("Hook application flag contradicts actual coefficients")
    phase = dose.get("phase")
    if event["type"] == "prefill":
        if phase != "prefill" or dose.get("prefill_position") not in {"last", "other"}:
            raise ValueError("Invalid measured prefill scope")
        phase = "prefill_" + dose["prefill_position"]
        positions = _integer(event.get("positions"), "measured positions", 1)
        norms = (event.get("measurements") or {}).get("delivered_edit_norms")
        if norms is not None and (not isinstance(norms, list) or len(norms) != positions):
            raise ValueError("Delivered norm count differs from measured positions")
    else:
        if phase not in {"reasoning", "output"}:
            raise ValueError("Invalid measured generated phase")
        positions = 1
        norm = (event.get("measurements") or {}).get("delivered_edit_norm")
        norms = None if norm is None else [norm]
        if clock == "tokens" and "generation_index" in event and event["generation_index"] != index:
            raise ValueError("Token event and hook global indices disagree")
    if norms is not None and any(type(value) not in {int, float} or not math.isfinite(value) or value < 0 for value in norms):
        raise ValueError("Delivered edit norms must be finite nonnegative measurements")
    return dict(index=index, phase=phase, positions=positions, coefficients=coefficients,
                delivered_edit_norm_sum=None if norms is None else math.fsum(norms))


def _add_bin(bins, sample):
    key = (sample["index"], sample["phase"])
    row = bins.setdefault(key, dict(index=key[0], phase=key[1], positions=0,
        cumulative_coefficients={axis: 0. for axis in AXES}, delivered_edit_norm_sum=0., measured_norm_positions=0))
    row["positions"] += sample["positions"]
    for axis, coefficient in sample["coefficients"].items():
        row["cumulative_coefficients"][axis] += coefficient * sample["positions"]
    if sample["delivered_edit_norm_sum"] is not None:
        row["delivered_edit_norm_sum"] += sample["delivered_edit_norm_sum"]
        row["measured_norm_positions"] += sample["positions"]


def _gaps(indices, end):
    gaps, cursor = [], 0
    for index in sorted(set(i for i in indices if 0 <= i < end)):
        if index > cursor:
            gaps.append(dict(start=cursor, end_exclusive=index))
        cursor = index + 1
    if cursor < end:
        gaps.append(dict(start=cursor, end_exclusive=end))
    return gaps


def _clock(node, clock):
    return _integer(node.get("generated_tokens" if clock == "tokens" else "completed_decisions"), "source intervention clock")


def _nodes(event, depth=0):
    if depth > 8:
        raise ValueError("Intervention nesting exceeds bounds")
    yield event
    for key in ("intervention", "cancellation", "gate"):
        child = event.get(key)
        if isinstance(child, dict):
            yield from _nodes(child, depth + 1)


@_checked
def build_yoke(source_events, presets, *, clock, source, pair_id):
    """Freeze source pulse/cancellation timing and measured exposure by clock.

    source.events_sha256 hashes the supplied raw JSONL or canonical JSON array.
    prefix_sha256 uses the same encoding for its first prefix_event_count rows.
    Lists never pretend to carry a raw-file hash. End clocks come from the actual
    source boundary; neither generated opportunities nor cost units are inferred.
    """
    if clock not in {"tokens", "decisions"}:
        raise ValueError("Yoke clock must be tokens or decisions")
    _id(pair_id, "yoke pair id")
    source = _plain(source)
    fields = {"run_id", "events_sha256", "prefix_sha256", "prefix_event_count", "config_sha256", "model_fingerprint_sha256", "end_tokens", "end_decisions"}
    _keys(source, fields, "source provenance")
    _id(source["run_id"], "source run id")
    for key in ("events_sha256", "prefix_sha256", "config_sha256", "model_fingerprint_sha256"):
        _hash(source[key], key)
    _integer(source["end_tokens"], "source token horizon")
    _integer(source["end_decisions"], "source completed-decision horizon", high=10000)
    end = source["end_tokens" if clock == "tokens" else "end_decisions"]
    events, raw, lines, encoding = _source_events(source_events)
    if source["events_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Source event hash mismatch")
    prefix_count = _integer(source["prefix_event_count"], "source prefix count", 0, len(events))
    prefix = events[:prefix_count]
    prefix_raw = b"".join(lines[:prefix_count]) if lines is not None else _canonical(prefix).encode()
    if source["prefix_sha256"] != hashlib.sha256(prefix_raw).hexdigest():
        raise ValueError("Source prefix hash mismatch")
    source["hash_encoding"] = encoding
    resolved = validate_presets(presets)
    by_preset = {preset["id"]: preset for preset in resolved}
    commands, audit, seen, bins, omitted_controls = [], [], {}, {}, []
    audited = set()
    def record_attempt(event_index, event, node, injection=None):
        if injection is not None and injection.get("delivered"):
            identity = ("pulse", injection.get("pulse_id"))
        elif injection is not None:
            identity = ("scheduler_attempt", injection.get("sequence"))
        else:
            identity = ("aux_attempt", node.get("aux_call_index", content_hash(node)))
        if identity in audited:
            return
        audited.add(identity)
        audit.append(dict(source_event_index=event_index, source_event_sha256=content_hash(event),
            delivered=node["delivered"], delivery_status=node.get("delivery_status", "delivered" if node["delivered"] else "undelivered"),
            requested_preset_id=node.get("requested_preset_id", node.get("preset_id")), actor=node.get("actor"),
            scheduled=bool(node["delivered"] and injection is not None)))
    missing_measurements = 0
    last_sequence = -1
    for event_index, event in enumerate(prefix):
        if "run_id" in event and event["run_id"] != source["run_id"]:
            raise ValueError("Source prefix mixes run identities")
        if "seq" in event:
            _integer(event["seq"], "source event sequence", high=10**12)
            if event["seq"] <= last_sequence:
                raise ValueError("Source event sequence must strictly increase")
            last_sequence = event["seq"]
        sample = exposure_from_event(event, clock=clock)
        if sample is not None:
            if sample["index"] >= end:
                raise ValueError("Measured source position lies outside the declared horizon")
            _add_bin(bins, sample)
        elif event.get("type") in {"token", "prefill"}:
            missing_measurements += 1
        nodes = list(_nodes(event))
        calls = [node for node in nodes if node.get("type") == "aux_call"]
        injections = [node for node in nodes if node.get("type") == "effect_injection"]
        if calls:
            call = calls[0]
            if type(call.get("delivered")) is not bool:
                raise ValueError("Source call must record actual delivery")
            if len(calls) != 1 or len(injections) > 1 or call["delivered"] != bool(injections and injections[0].get("delivered")):
                raise ValueError("Source call delivery contradicts its resolved injection")
            record_attempt(event_index, event, call, injections[0] if injections else None)
            if call["delivered"] and not injections:
                raise ValueError("Delivered source call lacks a resolved preset event; use a fully recorded v2 source")
        for node in nodes:
            kind = node.get("type")
            if kind not in {"effect_injection", "effect_cancelled"}:
                continue
            if kind == "effect_injection":
                if type(node.get("delivered")) is not bool:
                    raise ValueError("Source pulse must record delivery")
                if not node["delivered"]:
                    if not calls:
                        record_attempt(event_index, event, node, node)
                    continue
                preset_id = node.get("preset_id")
                if preset_id not in by_preset or node.get("preset_hash") != content_hash(by_preset[preset_id]):
                    raise ValueError("Delivered source preset identity mismatch")
                pulse_id = _integer(node.get("pulse_id"), "source pulse id", 1, 10**12)
                identity = (kind, pulse_id)
                body = dict(kind="inject", preset_id=preset_id, preset_sha256=node["preset_hash"])
                if not calls:
                    record_attempt(event_index, event, node, node)
            else:
                sequence = _integer(node.get("sequence"), "source cancellation sequence", high=10**12)
                channel = node.get("channel")
                if channel is not None and (type(channel) is not str or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", channel)):
                    raise ValueError("Invalid source cancellation channel")
                identity, body = (kind, sequence), dict(kind="cancel", channel=channel)
            if identity in seen:
                if seen[identity] != content_hash(node):
                    raise ValueError("Repeated source intervention identity changed contents")
                continue  # A standalone event may also occur inside its parent tool/control event.
            seen[identity] = content_hash(node)
            at = _clock(node, clock)
            if at > end:
                raise ValueError("Source intervention exceeds its declared horizon")
            commands.append(dict(at=at, source_event_index=event_index, source_event_sha256=content_hash(event), **body))
        if event.get("type") == "control" or any(node.get("type") == "controls" for node in nodes):
            omitted_controls.append(dict(source_event_index=event_index, reason="Held baseline changes are not pulse events; match these explicitly in the target protocol"))
    commands.sort(key=lambda item: (item["at"], item["source_event_index"]))
    commands.append(dict(kind="end", at=end, source_event_index=prefix_count, source_event_sha256=source["prefix_sha256"]))
    for index, command in enumerate(commands):
        command["id"] = index
    return validate_yoke(_seal(dict(kind=KIND, schema_version=1, clock=clock, pair_id=pair_id, source=source,
        presets=resolved, commands=commands, delivery_audit=audit, source_exposure_bins=[bins[key] for key in sorted(bins)],
        source_observation_gaps=_gaps([index for index, _ in bins], end), source_missing_measurement_events=missing_measurements,
        omitted_controls=omitted_controls, recipient_button_effect="none", late_policy="skip", source_horizon_policy="cancel_at_end",
        exposure_units="Applied coefficients times measured positions, with delivered numeric edit norms recorded separately")))


def _validate_bins(rows, end):
    if not isinstance(rows, list) or len(rows) > MAX_EVENTS:
        raise ValueError("Invalid exposure bins")
    previous = None
    for row in rows:
        _keys(row, {"index", "phase", "positions", "cumulative_coefficients", "delivered_edit_norm_sum", "measured_norm_positions"}, "exposure bin")
        _integer(row["index"], "exposure index", 0, max(0, end - 1))
        if not end or row["phase"] not in PHASES:
            raise ValueError("Exposure bin lies outside observed horizon")
        key = (row["index"], row["phase"])
        if previous is not None and key <= previous:
            raise ValueError("Exposure bins must have unique sorted index/phase keys")
        previous = key
        n = _integer(row["positions"], "exposure positions", 1, 10**9)
        _integer(row["measured_norm_positions"], "norm positions", 0, n)
        _keys(row["cumulative_coefficients"], AXES, "cumulative coefficients")
        for axis, total in row["cumulative_coefficients"].items():
            low, high = (0, n) if axis in ATTENUATION_AXES else (-4*n, 4*n)
            if type(total) not in {int, float} or not math.isfinite(total) or not low - 1e-8 <= total <= high + 1e-8:
                raise ValueError("Cumulative coefficient exceeds its measured-position bound")
        norm = row["delivered_edit_norm_sum"]
        if type(norm) not in {int, float} or not math.isfinite(norm) or norm < 0 or not row["measured_norm_positions"] and norm != 0:
            raise ValueError("Invalid cumulative delivered norm")


@_checked
def validate_yoke(schedule, *, source_events=None):
    value = _plain(schedule)
    expected = value.pop("sha256", None) if isinstance(value, dict) else None
    if expected != content_hash(value):
        raise ValueError("Yoke schedule hash mismatch")
    fields = {"kind", "schema_version", "clock", "pair_id", "source", "presets", "commands", "delivery_audit",
              "source_exposure_bins", "source_observation_gaps", "source_missing_measurement_events", "omitted_controls",
              "recipient_button_effect", "late_policy", "source_horizon_policy", "exposure_units"}
    _keys(value, fields, "yoke schedule")
    if value["kind"] != KIND or type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["clock"] not in {"tokens", "decisions"}:
        raise ValueError("Unsupported yoke format or clock")
    if value["recipient_button_effect"] != "none" or value["late_policy"] != "skip" or value["source_horizon_policy"] != "cancel_at_end":
        raise ValueError("Pure yoke must disable additional button effects and stop at source horizon")
    _id(value["pair_id"], "pair id")
    source = value["source"]
    _keys(source, {"run_id", "events_sha256", "prefix_sha256", "prefix_event_count", "config_sha256", "model_fingerprint_sha256", "end_tokens", "end_decisions", "hash_encoding"}, "yoke source")
    for key in ("events_sha256", "prefix_sha256", "config_sha256", "model_fingerprint_sha256"):
        _hash(source[key], key)
    _id(source["run_id"], "source run id")
    _integer(source["prefix_event_count"], "source prefix count", 0, MAX_EVENTS)
    _integer(source["end_tokens"], "token horizon")
    _integer(source["end_decisions"], "decision horizon", 0, 10000)
    if source["hash_encoding"] not in {"raw-jsonl", "canonical-json-array"}:
        raise ValueError("Unknown source hash encoding")
    end = source["end_tokens" if value["clock"] == "tokens" else "end_decisions"]
    presets = validate_presets(value["presets"])
    if _canonical(presets) != _canonical(value["presets"]):
        raise ValueError("Yoke presets must be fully resolved")
    by_id = {preset["id"]: content_hash(preset) for preset in presets}
    commands = value["commands"]
    if not isinstance(commands, list) or not 1 <= len(commands) <= MAX_EVENTS:
        raise ValueError("Invalid yoke command list")
    previous = -1
    for index, command in enumerate(commands):
        base = {"id", "kind", "at", "source_event_index", "source_event_sha256"}
        kind = command.get("kind") if isinstance(command, dict) else None
        extra = {"preset_id", "preset_sha256"} if kind == "inject" else {"channel"} if kind == "cancel" else set()
        _keys(command, base | extra, "yoke command")
        if command["id"] != index or type(command["id"]) is not int or kind not in {"inject", "cancel", "end"}:
            raise ValueError("Invalid yoke command identity")
        _integer(command["at"], "scheduled index", 0, end)
        if command["at"] < previous or kind == "end" and (index != len(commands)-1 or command["at"] != end):
            raise ValueError("Yoke commands are not ordered through the source horizon")
        previous = command["at"]
        _integer(command["source_event_index"], "source event index", 0, source["prefix_event_count"])
        _hash(command["source_event_sha256"], "source event hash")
        if kind != "end" and command["source_event_index"] >= source["prefix_event_count"]:
            raise ValueError("Source intervention index lies outside its event prefix")
        if kind == "end" and (command["source_event_index"] != source["prefix_event_count"] or command["source_event_sha256"] != source["prefix_sha256"]):
            raise ValueError("Source horizon must reference the verified event prefix")
        if kind == "inject" and (type(command["preset_id"]) is not str or command["preset_id"] not in by_id or command["preset_sha256"] != by_id[command["preset_id"]]):
            raise ValueError("Yoke command preset hash mismatch")
        if kind == "cancel" and command["channel"] is not None and (type(command["channel"]) is not str or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", command["channel"])):
            raise ValueError("Invalid cancellation channel")
    if commands[-1]["kind"] != "end":
        raise ValueError("Yoke must cancel at its observed source horizon")
    _validate_bins(value["source_exposure_bins"], end)
    if value["source_observation_gaps"] != _gaps([row["index"] for row in value["source_exposure_bins"]], end):
        raise ValueError("Source observation gap report contradicts its measurements")
    _integer(value["source_missing_measurement_events"], "missing measurement count", 0, MAX_EVENTS)
    if not isinstance(value["delivery_audit"], list) or len(value["delivery_audit"]) > MAX_EVENTS or not isinstance(value["omitted_controls"], list) or len(value["omitted_controls"]) > MAX_EVENTS:
        raise ValueError("Source coverage records must be bounded lists")
    for row in value["delivery_audit"]:
        _keys(row, {"source_event_index", "source_event_sha256", "delivered", "delivery_status", "requested_preset_id", "actor", "scheduled"}, "source delivery audit")
        _integer(row["source_event_index"], "audit source index", 0, source["prefix_event_count"]-1)
        _hash(row["source_event_sha256"], "audit source hash")
        if type(row["delivered"]) is not bool or type(row["scheduled"]) is not bool or row["scheduled"] != row["delivered"]:
            raise ValueError("Delivery audit contradicts scheduling")
        if type(row["delivery_status"]) is not str or row["delivery_status"] not in {"delivered", "sham", "disabled", "zero_effect", "undelivered"} or row["delivered"] != (row["delivery_status"] == "delivered"):
            raise ValueError("Invalid source delivery status")
        if row["requested_preset_id"] is not None and (type(row["requested_preset_id"]) is not str or row["requested_preset_id"] not in by_id):
            raise ValueError("Unknown requested source preset")
        if type(row["actor"]) is not str or row["actor"] not in {"model", "human", "demonstration", "schedule"}:
            raise ValueError("Unknown source delivery actor")
    if sum(row["scheduled"] for row in value["delivery_audit"]) != sum(row["kind"] == "inject" for row in commands):
        raise ValueError("Delivery audit and source pulse schedule disagree")
    for row in value["omitted_controls"]:
        _keys(row, {"source_event_index", "reason"}, "source control omission")
        _integer(row["source_event_index"], "control source index", 0, source["prefix_event_count"]-1)
        if type(row["reason"]) is not str or not row["reason"] or len(row["reason"]) > 1024:
            raise ValueError("Invalid control omission reason")
    value["sha256"] = expected
    if source_events is not None:
        rebuilt_source = {key: item for key, item in source.items() if key != "hash_encoding"}
        rebuilt = build_yoke(source_events, presets, clock=value["clock"], source=rebuilt_source, pair_id=value["pair_id"])
        if _canonical(value) != _canonical(rebuilt):
            raise ValueError("Yoke schedule differs from the verified source prefix")
    return value


class YokedExposure:
    """Index cursor with explicit delivery acknowledgments and no budget mutation.

    due(index) is called BEFORE the forward pass/decision at that global index.
    It emits each command once. Apply commands to the target effect scheduler,
    then acknowledge them before checkpointing. Late pulses are skipped, never
    shifted into a new unobserved effect. Command 'end' cancels all yoked pulses.
    """
    def __init__(self, schedule):
        self.schedule = validate_yoke(schedule)
        self.cursor, self.last_index = 0, -1
        self.pending, self.receipts, self.own_calls, self.bins = {}, [], {}, {}
        self.outside_positions = 0
        self.finished = None

    @property
    def horizon(self):
        return self.schedule["source"]["end_tokens" if self.schedule["clock"] == "tokens" else "end_decisions"]

    def due(self, index):
        _integer(index, "target index")
        if self.finished is not None or index < self.last_index:
            raise ValueError("Yoke is finished or its clock moved backwards")
        if self.pending:
            if index == self.last_index:
                return []
            raise ValueError("Acknowledge pending yoke commands before advancing")
        self.last_index = index
        issued = []
        commands = self.schedule["commands"]
        while self.cursor < len(commands) and commands[self.cursor]["at"] <= index:
            command = deepcopy(commands[self.cursor])
            self.cursor += 1
            command.update(observed_index=index, index_error=index-command["at"], actor="schedule", source_yoke_sha256=self.schedule["sha256"])
            if command["at"] < index and command["kind"] != "end":
                self.receipts.append(self._receipt(command, False, index, "missed_source_index"))
            else:
                self.pending[command["id"]] = command
                issued.append(deepcopy(command))
        return issued

    @staticmethod
    def _receipt(command, delivered, actual_index, reason):
        return dict(id=command["id"], kind=command["kind"], scheduled_index=command["at"], observed_index=command["observed_index"],
                    actual_index=actual_index, index_error=actual_index-command["at"], delivered=delivered, reason=reason)

    def acknowledge(self, identifier, *, delivered, actual_index=None, reason=""):
        if self.finished is not None or type(identifier) is not int or identifier not in self.pending or type(delivered) is not bool:
            raise ValueError("Unknown or invalid pending yoke acknowledgment")
        command = self.pending[identifier]
        actual_index = command["observed_index"] if actual_index is None else _integer(actual_index, "actual delivery index")
        if actual_index != self.last_index or not isinstance(reason, str) or len(reason) > 1024:
            raise ValueError("Invalid yoke delivery timing/reason")
        if delivered and command["kind"] == "inject" and (actual_index != command["at"] or actual_index > self.horizon):
            raise ValueError("A pure yoke cannot deliver a pulse at a shifted or extrapolated index")
        receipt = self._receipt(command, delivered, actual_index, reason)
        self.receipts.append(receipt)
        del self.pending[identifier]
        self.receipts.sort(key=lambda row: row["id"])
        return deepcopy(receipt)

    def recipient_call(self, tool_name):
        """Record preference separately; this authorizes NO additional effect."""
        if self.finished is not None or type(tool_name) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", tool_name):
            raise ValueError("Invalid recipient auxiliary choice")
        self.own_calls[tool_name] = self.own_calls.get(tool_name, 0) + 1
        return dict(actor="model", tool=tool_name, additional_active_effect=False, requested_active=False,
                    delivery_status="pure_yoke_choice_has_no_additional_effect", source_yoke_sha256=self.schedule["sha256"])

    def record_exposure(self, index, phase, coefficients, *, positions=1, delivered_edit_norm_sum=None):
        _integer(index, "target exposure index")
        _integer(positions, "target measured positions", 1)
        if self.finished is not None or self.pending or index != self.last_index or phase not in PHASES:
            raise ValueError("Exposure must follow an acknowledged current yoke boundary")
        coefficients = _coefficients(coefficients)
        if delivered_edit_norm_sum is not None and (type(delivered_edit_norm_sum) not in {int, float} or not math.isfinite(delivered_edit_norm_sum) or delivered_edit_norm_sum < 0):
            raise ValueError("Invalid measured edit norm")
        if index >= self.horizon:
            self.outside_positions += positions
            return dict(covered=False, reason="outside_observed_source_horizon", positions=positions)
        _add_bin(self.bins, dict(index=index, phase=phase, positions=positions, coefficients=coefficients,
                               delivered_edit_norm_sum=delivered_edit_norm_sum))
        return dict(covered=True, index=index, phase=phase, positions=positions)

    def finish(self, index, *, reason="target_complete"):
        _integer(index, "target final index")
        if self.finished is not None or self.pending or index < self.last_index or not isinstance(reason, str) or not reason or len(reason)>1024:
            raise ValueError("Finish requires an acknowledged monotonic boundary and reason")
        self.finished = dict(index=index, reason=reason)
        return self.report()

    def report(self):
        source_bins = {(row["index"],row["phase"]): row for row in self.schedule["source_exposure_bins"]}
        keys = sorted(set(source_bins) | set(self.bins))
        missing, excess, matched, coefficient_l1 = 0, 0, 0, {axis: 0. for axis in AXES}
        by_phase = {}
        for phase in sorted(PHASES):
            original = [row for row in source_bins.values() if row["phase"]==phase]
            actual = [row for row in self.bins.values() if row["phase"]==phase]
            by_phase[phase] = dict(source_positions=sum(row["positions"] for row in original), target_positions=sum(row["positions"] for row in actual),
                source_cumulative={axis:math.fsum(row["cumulative_coefficients"][axis] for row in original) for axis in AXES},
                target_cumulative={axis:math.fsum(row["cumulative_coefficients"][axis] for row in actual) for axis in AXES},
                source_delivered_edit_norm_sum=math.fsum(row["delivered_edit_norm_sum"] for row in original),
                target_delivered_edit_norm_sum=math.fsum(row["delivered_edit_norm_sum"] for row in actual),
                source_norm_positions=sum(row["measured_norm_positions"] for row in original),
                target_norm_positions=sum(row["measured_norm_positions"] for row in actual))
        for key in keys:
            source, target = source_bins.get(key, {}), self.bins.get(key, {})
            a, b = source.get("positions",0), target.get("positions",0)
            matched += min(a,b)
            missing += max(0,a-b)
            excess += max(0,b-a)
            for axis in AXES:
                coefficient_l1[axis] += abs(source.get("cumulative_coefficients",{}).get(axis,0)-target.get("cumulative_coefficients",{}).get(axis,0))
        expected = [row for row in self.schedule["commands"] if row["kind"]=="inject"]
        reached = [row for row in self.receipts if row["kind"]=="inject"]
        target_end = self.finished["index"] if self.finished else max(0,self.last_index)
        return dict(schema_version=1, source_yoke_sha256=self.schedule["sha256"], clock=self.schedule["clock"],
            source_requested_attempts=len(self.schedule["delivery_audit"]), source_delivered_pulses=len(expected),
            source_requested_pulses=len(expected), target_due_pulses=len(reached), delivered_pulses=sum(row["delivered"] for row in reached),
            failed_or_missed_pulses=sum(not row["delivered"] for row in reached), unreached_source_pulses=len(expected)-len(reached),
            source_undelivered_attempts=sum(not row["delivered"] for row in self.schedule["delivery_audit"]),
            absolute_index_error=sum(abs(row["index_error"]) for row in reached), matched_observed_positions=matched,
            uncovered_source_positions=missing, excess_target_positions=excess, coefficient_l1_error=coefficient_l1,
            by_phase=by_phase, source_observation_gaps=deepcopy(self.schedule["source_observation_gaps"]),
            target_observation_gaps=_gaps([index for index,_ in self.bins],min(target_end,self.horizon)),
            uncovered_source_intervals=_gaps([index for index,_ in self.bins],self.horizon),
            outside_source_positions=self.outside_positions, own_button_calls=dict(self.own_calls),
            own_button_additional_effects=0, source_missing_measurement_events=self.schedule["source_missing_measurement_events"],
            source_controls_not_yoked=len(self.schedule["omitted_controls"]), finished=deepcopy(self.finished),
            schedule_complete=self.cursor == len(self.schedule["commands"]) and not self.pending,
            observed_coefficients_equal=not missing and not excess and all(v < 1e-10 for v in coefficient_l1.values()),
            terminated_before_source_horizon=self.finished is not None and self.finished["index"] < self.horizon,
            coverage="partial" if missing or self.schedule["source_observation_gaps"] or self.schedule["source_missing_measurement_events"] or self.schedule["omitted_controls"] or len(expected)!=len(reached) or any(not row["delivered"] for row in self.receipts) or self.cursor != len(self.schedule["commands"]) or self.pending else "observed_source_covered",
            limitation="Equal injection indices do not ensure equal phases, numeric perturbations, token histories, or behavior. Coefficient sums are applied controls; edit norms are measured separately.")

    def snapshot(self):
        if self.pending:
            raise ValueError("Cannot checkpoint an unacknowledged effect dispatch")
        return _seal(dict(kind="opium-bench/yoke-cursor", schema_version=1, schedule=deepcopy(self.schedule), cursor=self.cursor,
            last_index=self.last_index, receipts=deepcopy(self.receipts), own_calls=dict(self.own_calls),
            exposure_bins=[deepcopy(self.bins[key]) for key in sorted(self.bins)], outside_positions=self.outside_positions,
            finished=deepcopy(self.finished)))

    @classmethod
    @_checked
    def restore(cls, state):
        value = _plain(state)
        expected = value.pop("sha256", None) if isinstance(value,dict) else None
        if expected != content_hash(value):
            raise ValueError("Yoke cursor hash mismatch")
        _keys(value,{"kind","schema_version","schedule","cursor","last_index","receipts","own_calls","exposure_bins","outside_positions","finished"},"yoke cursor")
        if value["kind"]!="opium-bench/yoke-cursor" or type(value["schema_version"]) is not int or value["schema_version"]!=1:
            raise ValueError("Unsupported yoke cursor version")
        result=cls(value["schedule"])
        _integer(value["cursor"],"cursor",0,len(result.schedule["commands"]))
        _integer(value["last_index"],"last target index",-1)
        if not isinstance(value["receipts"],list) or len(value["receipts"])!=value["cursor"]:
            raise ValueError("Every issued command needs one delivery receipt")
        for index,receipt in enumerate(value["receipts"]):
            _keys(receipt,{"id","kind","scheduled_index","observed_index","actual_index","index_error","delivered","reason"},"delivery receipt")
            command=result.schedule["commands"][index]
            if type(receipt["id"]) is not int or receipt["id"]!=index or receipt["kind"]!=command["kind"] or receipt["scheduled_index"]!=command["at"] or type(receipt["delivered"]) is not bool:
                raise ValueError("Receipt contradicts source command")
            _integer(receipt["observed_index"],"observed receipt index",command["at"],value["last_index"])
            _integer(receipt["actual_index"],"actual receipt index",receipt["observed_index"],value["last_index"])
            if receipt["index_error"]!=receipt["actual_index"]-command["at"] or not isinstance(receipt["reason"],str) or len(receipt["reason"])>1024:
                raise ValueError("Receipt timing/reason mismatch")
            if receipt["delivered"] and command["kind"]=="inject" and receipt["actual_index"]!=command["at"]:
                raise ValueError("A shifted active pulse is not an exact pure yoke")
        if value["cursor"]<len(result.schedule["commands"]) and result.schedule["commands"][value["cursor"]]["at"]<=value["last_index"]:
            raise ValueError("Cursor skipped an unrecorded due command")
        _validate_bins(value["exposure_bins"],result.horizon)
        if any(row["index"]>value["last_index"] for row in value["exposure_bins"]):
            raise ValueError("Exposure precedes its observed cursor boundary")
        if not isinstance(value["own_calls"],dict) or len(value["own_calls"])>32:
            raise ValueError("Invalid recipient choices")
        for name,count in value["own_calls"].items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}",name):
                raise ValueError("Invalid recipient tool name")
            _integer(count,"recipient choice count",1,10000)
        _integer(value["outside_positions"],"outside positions",0,10**9)
        if value["finished"] is not None:
            _keys(value["finished"],{"index","reason"},"yoke finish")
            _integer(value["finished"]["index"],"finished index",max(0,value["last_index"]))
            if not isinstance(value["finished"]["reason"],str) or not value["finished"]["reason"] or len(value["finished"]["reason"])>1024:
                raise ValueError("Invalid termination reason")
        result.cursor,result.last_index=value["cursor"],value["last_index"]
        result.receipts,result.own_calls=deepcopy(value["receipts"]),dict(value["own_calls"])
        result.bins={(row["index"],row["phase"]):deepcopy(row) for row in value["exposure_bins"]}
        result.outside_positions,result.finished=value["outside_positions"],deepcopy(value["finished"])
        return result
