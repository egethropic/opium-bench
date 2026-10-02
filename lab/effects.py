"""Versioned, model-independent effects for opt-in research recipes.

No historical recipe is resolved here. The old controller remains the authority
for unversioned recipes. Values describe activation interventions, not feelings.

Boundary contract: read coefficients BEFORE a model forward pass, record its
actual exposure, then advance_tokens AFTER sampling every emitted token (including
reasoning, syntax and EOS). Prefill does not advance either clock. At completion
of a decision, call complete_decision once BEFORE dispatching that decision's
new injection. Invalid decisions also complete. Thus an injection first affects
the next decision at age zero. Cost units and wall time never age effects.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re

SCHEMA_VERSION = 1
PHASES = ("prefill", "reasoning", "output")
ACTORS = ("model", "human", "demonstration", "schedule")
OPERATION_ORDER = ["baseline_challenge", "joint_attenuation", "addition"]
GAIN_AXES = ("pain", "joy_raw", "joy_orthogonal", "random_gain")
ATTENUATION_AXES = ("pain_attenuation", "joy_raw_attenuation", "joy_orthogonal_attenuation")
COEFFICIENT_AXES = GAIN_AXES + ATTENUATION_AXES
MAX_CLOCK = 10**12
MAX_PULSES = 10000


def canonical_json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Expected bounded finite JSON data") from exc


def content_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _object(value, allowed, name):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}")


def _integer(value, name, low=0, high=MAX_CLOCK):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


def _number(value, name, low=-4, high=4):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number from {low} to {high}")
    return float(value)


def _choice(value, options, name):
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"Unknown {name}")
    return value


def _identifier(value, name="id"):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value):
        raise ValueError(f"{name} must be a short lowercase identifier")
    return value


def validate_preset(value, *, capabilities=None):
    """Return a detached resolved preset; unknown fields/versions fail.

    ``capabilities`` optionally declares adapter support as {phases: [...],
    prefill_positions: [...], sites: [location], layers: [integer]}. A caller
    MUST check adapter capabilities before admitting a preset into a real run.
    A calibrated layer resolves against the immutable calibration, never a GUI
    selection made after a run starts. Only a single residual-post edit site is
    supported. Attenuation uses joint projection (see joint_attenuation).
    """
    fields = {"schema_version", "id", "version", "label", "gains", "attenuation",
              "joy_direction", "site", "phases", "prefill_positions", "decay",
              "stacking", "operation_order"}
    _object(value, fields, "preset")
    _integer(value.get("schema_version", 1), "schema_version", 1, 1)
    result = {"schema_version": 1, "id": _identifier(value.get("id", "opium")),
              "version": _integer(value.get("version", 1), "version", 1, 1000000)}
    label = value.get("label", "Activation intervention")
    if not isinstance(label, str) or not 1 <= len(label) <= 160:
        raise ValueError("label must contain 1–160 characters")
    result["label"] = label
    for field in ("gains", "attenuation"):
        keys = ("pain", "joy", "random") if field == "gains" else ("pain", "joy")
        supplied = value.get(field, {})
        _object(supplied, keys, field)
        result[field] = {key: _number(supplied.get(key, 0), f"{field}.{key}",
                                     -4 if field == "gains" else 0,
                                     4 if field == "gains" else 1) for key in keys}
    result["joy_direction"] = _choice(value.get("joy_direction", "orthogonal"),
                                      ("raw", "orthogonal"), "joy_direction")
    site = value.get("site", {})
    _object(site, ("location", "layer"), "site")
    layer = site.get("layer", "calibrated")
    if layer != "calibrated":
        _integer(layer, "site.layer", 0, 4095)
    result["site"] = {"location": _choice(site.get("location", "residual_post"),
                                          ("residual_post",), "site.location"), "layer": layer}
    phases = value.get("phases", ["reasoning", "output"])
    if not isinstance(phases, list) or not phases or any(type(p) is not str or p not in PHASES for p in phases) or len(phases) != len(set(phases)):
        raise ValueError("phases must be distinct supported processing phases")
    result["phases"] = [p for p in PHASES if p in phases]
    result["prefill_positions"] = _choice(value.get("prefill_positions", "last"),
                                          ("last", "all"), "prefill_positions")
    decay = value.get("decay", {})
    _object(decay, ("shape", "clock", "half_life", "cutoff"), "decay")
    result["decay"] = {
        "shape": _choice(decay.get("shape", "exponential"), ("constant", "pulse", "exponential", "linear"), "decay.shape"),
        "clock": _choice(decay.get("clock", "tokens"), ("tokens", "decisions"), "decay.clock"),
        "half_life": _number(decay.get("half_life", 128), "decay.half_life", .001, 1000000),
        "cutoff": _integer(decay.get("cutoff", 768), "decay.cutoff", 1, 1000000)}
    stacking = value.get("stacking", {})
    _object(stacking, ("policy", "cap", "channel"), "stacking")
    result["stacking"] = {
        "policy": _choice(stacking.get("policy", "reset"), ("reset", "capped_additive"), "stacking.policy"),
        "cap": _number(stacking.get("cap", 4), "stacking.cap", .001, 4),
        "channel": _identifier(stacking.get("channel", "aux"), "stacking.channel")}
    order = value.get("operation_order", OPERATION_ORDER)
    if order != OPERATION_ORDER:
        raise ValueError("operation_order must be baseline_challenge, joint_attenuation, addition")
    result["operation_order"] = list(OPERATION_ORDER)
    if capabilities is not None:
        _object(capabilities, ("phases", "prefill_positions", "sites", "layers"), "capabilities")
        if any(p not in capabilities.get("phases", []) for p in result["phases"]):
            raise ValueError("Adapter does not implement a requested processing phase")
        if "prefill" in phases and result["prefill_positions"] not in capabilities.get("prefill_positions", []):
            raise ValueError("Adapter does not implement the requested prefill positions")
        if result["site"]["location"] not in capabilities.get("sites", []):
            raise ValueError("Adapter does not implement the requested edit site")
        if layer != "calibrated" and layer not in capabilities.get("layers", []):
            raise ValueError("Adapter does not implement the requested layer")
    return result


def preset_bundle(presets):
    """JSON-only library export with detached resolved definitions and hashes."""
    resolved = validate_presets(presets)
    payload = {"schema_version": 1, "kind": "opium-bench-effect-presets", "presets": resolved}
    return {**payload, "sha256": content_hash(payload)}


def import_preset_bundle(bundle):
    _object(bundle, ("schema_version", "kind", "presets", "sha256"), "preset bundle")
    if bundle.get("schema_version") != 1 or type(bundle.get("schema_version")) is not int or bundle.get("kind") != "opium-bench-effect-presets":
        raise ValueError("Unsupported preset bundle")
    resolved = preset_bundle(bundle.get("presets"))
    if bundle != resolved:
        raise ValueError("Preset bundle checksum or resolved contents do not match")
    return copy.deepcopy(resolved["presets"])


def validate_presets(presets, *, capabilities=None):
    if not isinstance(presets, list) or not 1 <= len(presets) <= 128:
        raise ValueError("presets must contain 1–128 definitions")
    if len(canonical_json(presets)) > 2000000:
        raise ValueError("Preset library exceeds the bounded JSON size")
    result = [validate_preset(p, capabilities=capabilities) for p in presets]
    if len({p["id"] for p in result}) != len(result):
        raise ValueError("Duplicate preset id")
    if len({canonical_json(p["site"]) for p in result}) != 1:
        raise ValueError("A run must use one compatible edit site")
    channels = {}
    for p in result:
        policy = (p["stacking"]["policy"], p["stacking"]["cap"])
        channel = p["stacking"]["channel"]
        if channel in channels and channels[channel] != policy:
            raise ValueError("Presets sharing a stacking channel must agree on policy and cap")
        channels[channel] = policy
    return result


def decay_level(preset, age):
    """Age zero is full dose; finite effects are zero AT their cutoff."""
    _integer(age, "age")
    decay = preset["decay"]
    if decay["shape"] == "constant":
        return 1.0
    if age >= decay["cutoff"]:
        return 0.0
    if decay["shape"] == "exponential":
        return math.exp2(-age / decay["half_life"])
    if decay["shape"] == "linear":
        return 1 - age / decay["cutoff"]
    return 1.0


def joint_attenuation(vector, directions, fractions, *, rank_tolerance=1e-7):
    """Joint attenuation using a symmetric (Löwdin) orthogonalized basis.

    Returns a list and requires only NumPy when actually called. Unit directions
    form D. Q=D(D.T D)^(-1/2), and result=x-Q diag(fractions) Q.T x. This is
    permutation-equivariant, bounded, and removes the entire declared span when
    every fraction is one. Fractions belong to the symmetrically orthogonalized
    axes, NOT independent removals of overlapping raw projections. Dependent or
    nearly dependent directions are rejected instead of silently reordered.
    """
    import numpy as np
    x = np.asarray(vector, dtype=float)
    ds = np.asarray(directions, dtype=float)
    fs = np.asarray(fractions, dtype=float)
    tolerance = _number(rank_tolerance, "rank_tolerance", 1e-12, .1)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("vector must be a finite nonempty vector")
    if ds.size == 0 and fs.size == 0:
        return x.tolist()
    if ds.ndim != 2 or ds.shape[1] != x.size or fs.shape != (ds.shape[0],) or not np.isfinite(ds).all() or not np.isfinite(fs).all() or np.any(fs < 0) or np.any(fs > 1):
        raise ValueError("Directions and attenuation fractions have incompatible finite shapes")
    norms = np.linalg.norm(ds, axis=1)
    if np.any(norms <= tolerance):
        raise ValueError("Attenuation direction has zero norm")
    d = (ds / norms[:, None]).T
    eigenvalues, eigenvectors = np.linalg.eigh(d.T @ d)
    if eigenvalues.min() <= tolerance * eigenvalues.max():
        raise ValueError("Attenuation directions are rank deficient")
    q = d @ ((eigenvectors * (1 / np.sqrt(eigenvalues))) @ eigenvectors.T)
    return (x - q @ (fs * (q.T @ x))).tolist()


class EffectScheduler:
    """Serializable pulses with independent ages and explicit clock boundaries.

    Reset replaces the entire named channel. Capped-additive combines signed
    gains per axis/channel, clips to that channel's cap, then clips aggregate
    gains to ±4. Fractional attenuation composes as 1-product(1-fraction).
    Disabling cancels live AND pending pulses; re-enabling never revives them.
    A transition may cancel or preserve existing pulses; it never rewrites their
    immutable preset. An inactive/sham injection changes no pulse or schedule.
    """
    def __init__(self, presets, *, enabled=True):
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        self.presets = {p["id"]: p for p in validate_presets(presets)}
        self.enabled = enabled
        self.generated_tokens = 0
        self.completed_decisions = 0
        self.sequence = 0
        self.next_pulse_id = 1
        self.pulses = []
        self.pending = []
        self.counts = {actor: 0 for actor in ACTORS}
        self.exposure = {p: {"positions": 0, **{a: 0.0 for a in COEFFICIENT_AXES}} for p in PHASES}

    def _clock(self, clock):
        return self.generated_tokens if clock == "tokens" else self.completed_decisions

    def _preset(self, identifier):
        if not isinstance(identifier, str) or identifier not in self.presets:
            raise ValueError("Unknown effect preset")
        return self.presets[identifier]

    def _event(self, kind, actor, **data):
        self.sequence += 1
        return {"type": kind, "sequence": self.sequence, "actor": actor,
                "generated_tokens": self.generated_tokens,
                "completed_decisions": self.completed_decisions, **data}

    def _expire(self):
        self.pulses = [p for p in self.pulses if decay_level(self._preset(p["preset_id"]),
                         self._clock(p["clock"]) - p["started_at"]) > 0]

    def inject(self, preset_id, *, actor="model", active=True, metadata=None):
        preset = self._preset(preset_id)
        _choice(actor, ACTORS, "actor")
        if type(active) is not bool:
            raise ValueError("active must be boolean")
        metadata = {} if metadata is None else metadata
        if not isinstance(metadata, dict) or len(canonical_json(metadata)) > 8192:
            raise ValueError("Injection metadata must be a bounded JSON object")
        self._expire()
        delivered = active and self.enabled and any(preset[section][axis] for section in ("gains", "attenuation") for axis in preset[section])
        if delivered and preset["stacking"]["policy"] == "capped_additive" and len(self.pulses) >= MAX_PULSES:
            raise ValueError("Maximum concurrent pulse count reached")
        if delivered:
            if preset["stacking"]["policy"] == "reset":
                self.pulses = [p for p in self.pulses if self._preset(p["preset_id"])["stacking"]["channel"] != preset["stacking"]["channel"]]
            pulse = {"id": self.next_pulse_id, "preset_id": preset_id,
                     "clock": preset["decay"]["clock"], "started_at": self._clock(preset["decay"]["clock"]),
                     "actor": actor, "metadata": copy.deepcopy(metadata)}
            self.next_pulse_id += 1
            self.pulses.append(pulse)
        self.counts[actor] += 1
        return self._event("effect_injection", actor, preset_id=preset_id,
                           preset_hash=content_hash(preset), requested_active=active,
                           delivered=delivered, pulse_id=pulse["id"] if delivered else None,
                           metadata=copy.deepcopy(metadata))

    def schedule(self, preset_id, *, clock, at, actor="schedule", active=True, metadata=None):
        self._preset(preset_id)
        _choice(clock, ("tokens", "decisions"), "schedule clock")
        _choice(actor, ACTORS, "actor")
        _integer(at, "schedule boundary")
        if at < self._clock(clock):
            raise ValueError("Cannot schedule an injection in the past")
        if type(active) is not bool:
            raise ValueError("active must be boolean")
        metadata = {} if metadata is None else metadata
        if not isinstance(metadata, dict) or len(canonical_json(metadata)) > 8192:
            raise ValueError("Schedule metadata must be a bounded JSON object")
        if len(self.pending) >= MAX_PULSES:
            raise ValueError("Maximum pending pulse count reached")
        if at == self._clock(clock):
            return self.inject(preset_id, actor=actor, active=active, metadata=metadata)
        event = self._event("effect_scheduled", actor, preset_id=preset_id, clock=clock, at=at,
                            active=active, metadata=copy.deepcopy(metadata))
        self.pending.append({"schedule_id": event["sequence"], "preset_id": preset_id,
                             "clock": clock, "at": at, "actor": actor, "active": active,
                             "metadata": copy.deepcopy(metadata)})
        return event

    def _advance(self, clock, target):
        attribute = "generated_tokens" if clock == "tokens" else "completed_decisions"
        events = []
        due = sorted((p for p in self.pending if p["clock"] == clock and p["at"] <= target),
                     key=lambda p: (p["at"], p["schedule_id"]))
        # Ordinary per-token aging needs no state copy. Only a scheduled batch
        # can fail injection admission; roll the whole boundary back if it does.
        backup = copy.deepcopy(self.__dict__) if due else None
        try:
            for item in due:
                setattr(self, attribute, item["at"])
                self.pending.remove(item)
                events.append(self.inject(item["preset_id"], actor=item["actor"], active=item["active"], metadata=item["metadata"]))
            setattr(self, attribute, target)
            self._expire()
        except ValueError:
            if backup is not None:
                self.__dict__.update(backup)
            raise
        return events

    def advance_tokens(self, count=1):
        _integer(count, "token increment", 0, 1000000)
        target = _integer(self.generated_tokens + count, "generated token clock")
        return self._advance("tokens", target)

    def complete_decision(self, completed_count):
        _integer(completed_count, "completed decision count")
        if completed_count < self.completed_decisions:
            raise ValueError("Decision clock cannot go backwards")
        return self._advance("decisions", completed_count)

    def cancel(self, *, actor="human", channel=None, pending=True):
        _choice(actor, ACTORS, "actor")
        if channel is not None:
            _identifier(channel, "channel")
        if type(pending) is not bool:
            raise ValueError("pending must be boolean")
        def keep(p):
            return channel is not None and self._preset(p["preset_id"])["stacking"]["channel"] != channel
        cancelled = len(self.pulses)
        self.pulses = [p for p in self.pulses if keep(p)]
        cancelled -= len(self.pulses)
        cancelled_pending = len(self.pending)
        if pending:
            self.pending = [p for p in self.pending if keep(p)]
        cancelled_pending -= len(self.pending)
        return self._event("effect_cancelled", actor, channel=channel,
                           cancelled=cancelled, cancelled_pending=cancelled_pending)

    def set_enabled(self, enabled, *, actor="human"):
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        _choice(actor, ACTORS, "actor")
        cancelled = None
        if not enabled:
            cancelled = self.cancel(actor=actor)
        self.enabled = enabled
        return self._event("effect_gate", actor, enabled=enabled, cancellation=cancelled)

    def coefficients(self, phase="output", *, prefill_position="last"):
        _choice(phase, PHASES, "processing phase")
        _choice(prefill_position, ("last", "other"), "prefill position")
        totals = {axis: 0.0 for axis in COEFFICIENT_AXES}
        channels, delivered = {}, []
        for pulse in self.pulses:
            preset = self._preset(pulse["preset_id"])
            age = self._clock(pulse["clock"]) - pulse["started_at"]
            eligible = self.enabled and phase in preset["phases"]
            if phase == "prefill" and prefill_position == "other" and preset["prefill_positions"] == "last":
                eligible = False
            level = decay_level(preset, age) if eligible else 0.0
            if level == 0:
                continue
            channel = preset["stacking"]["channel"]
            values = channels.setdefault(channel, {a: 0.0 for a in COEFFICIENT_AXES})
            joy = "joy_" + preset["joy_direction"]
            gains = {"pain": preset["gains"]["pain"], joy: preset["gains"]["joy"], "random_gain": preset["gains"]["random"]}
            for axis, gain in gains.items():
                values[axis] += gain * level
            for axis, fraction in (("pain_attenuation", preset["attenuation"]["pain"]), (joy + "_attenuation", preset["attenuation"]["joy"])):
                values[axis] = 1 - (1 - values[axis]) * (1 - fraction * level)
            delivered.append({**copy.deepcopy(pulse), "age": age, "level": level,
                              "prefill_positions": preset["prefill_positions"]})
        for channel, values in channels.items():
            cap = next(p["stacking"]["cap"] for p in self.presets.values() if p["stacking"]["channel"] == channel)
            for axis in GAIN_AXES:
                totals[axis] += max(-cap, min(cap, values[axis]))
            for axis in ATTENUATION_AXES:
                totals[axis] = 1 - (1 - totals[axis]) * (1 - values[axis])
        for axis in GAIN_AXES:
            totals[axis] = max(-4.0, min(4.0, totals[axis]))
        return {"schema_version": 1, "phase": phase, "prefill_position": prefill_position if phase == "prefill" else None,
                "site": copy.deepcopy(next(iter(self.presets.values()))["site"]),
                "operation_order": list(OPERATION_ORDER), "coefficients": totals,
                "pulses": delivered, "generated_tokens": self.generated_tokens,
                "completed_decisions": self.completed_decisions}

    def record_exposure(self, phase="output", *, positions=1, prefill_position="last"):
        """Record coefficients actually sent to edited positions, before aging.

        Call only for positions the adapter really processes; this does not
        claim a behavioral effect or replace measurements of activation deltas.
        For prompt processing, record the last position separately from other
        positions using prefill_position="last" or "other"; a last-position
        preset never contributes exposure at non-final prompt positions.
        """
        _integer(positions, "positions", 1, 1000000)
        result = self.coefficients(phase, prefill_position=prefill_position)
        self.exposure[phase]["positions"] += positions
        for axis, coefficient in result["coefficients"].items():
            self.exposure[phase][axis] += coefficient * positions
        return result

    def snapshot(self):
        return {"schema_version": 1, "presets": copy.deepcopy(list(self.presets.values())),
                "presets_hash": content_hash(list(self.presets.values())), "enabled": self.enabled,
                "generated_tokens": self.generated_tokens, "completed_decisions": self.completed_decisions,
                "sequence": self.sequence, "next_pulse_id": self.next_pulse_id,
                "pulses": copy.deepcopy(self.pulses), "pending": copy.deepcopy(self.pending),
                "counts": dict(self.counts), "exposure": copy.deepcopy(self.exposure)}

    @classmethod
    def restore(cls, state):
        """Restore only a fully validated canonical JSON boundary checkpoint."""
        if not isinstance(state, dict) or len(canonical_json(state)) > 20_000_000:
            raise ValueError("Effect state must be bounded JSON")
        expected = {"schema_version", "presets", "presets_hash", "enabled", "generated_tokens",
                    "completed_decisions", "sequence", "next_pulse_id", "pulses", "pending", "counts", "exposure"}
        if set(state) != expected or type(state["schema_version"]) is not int or state["schema_version"] != 1:
            raise ValueError("Unsupported or incomplete effect checkpoint")
        result = cls(state["presets"], enabled=state["enabled"])
        if state["presets"] != list(result.presets.values()) or state["presets_hash"] != content_hash(state["presets"]):
            raise ValueError("Effect preset identity mismatch")
        for key in ("generated_tokens", "completed_decisions", "sequence", "next_pulse_id"):
            setattr(result, key, _integer(state[key], key, 1 if key == "next_pulse_id" else 0))
        for key in ("pulses", "pending"):
            if not isinstance(state[key], list) or len(state[key]) > MAX_PULSES:
                raise ValueError("Too many checkpoint pulses")
        ids = set()
        for pulse in state["pulses"]:
            keys = {"id", "preset_id", "clock", "started_at", "actor", "metadata"}
            if not isinstance(pulse, dict) or set(pulse) != keys:
                raise ValueError("Malformed checkpoint pulse")
            _integer(pulse["id"], "pulse id", 1, result.next_pulse_id - 1)
            if pulse["id"] in ids:
                raise ValueError("Duplicate pulse id")
            ids.add(pulse["id"])
            preset = result._preset(pulse["preset_id"])
            _choice(pulse["actor"], ACTORS, "pulse actor")
            if pulse["clock"] != preset["decay"]["clock"]:
                raise ValueError("Pulse clock differs from frozen preset")
            _integer(pulse["started_at"], "pulse start", 0, result._clock(pulse["clock"]))
            if not result.enabled or decay_level(preset, result._clock(pulse["clock"]) - pulse["started_at"]) == 0:
                raise ValueError("Disabled or expired pulses cannot be restored")
            if not isinstance(pulse["metadata"], dict) or len(canonical_json(pulse["metadata"])) > 8192:
                raise ValueError("Invalid pulse metadata")
        reset_channels = [result._preset(p["preset_id"])["stacking"]["channel"] for p in state["pulses"] if result._preset(p["preset_id"])["stacking"]["policy"] == "reset"]
        if len(reset_channels) != len(set(reset_channels)):
            raise ValueError("Reset channels cannot contain stacked pulses")
        ids = set()
        for pending in state["pending"]:
            keys = {"schedule_id", "preset_id", "clock", "at", "actor", "active", "metadata"}
            if not isinstance(pending, dict) or set(pending) != keys:
                raise ValueError("Malformed pending injection")
            _integer(pending["schedule_id"], "schedule id", 1, result.sequence)
            if pending["schedule_id"] in ids:
                raise ValueError("Duplicate schedule id")
            ids.add(pending["schedule_id"])
            result._preset(pending["preset_id"])
            _choice(pending["clock"], ("tokens", "decisions"), "schedule clock")
            _choice(pending["actor"], ACTORS, "schedule actor")
            _integer(pending["at"], "schedule boundary", result._clock(pending["clock"]) + 1)
            if type(pending["active"]) is not bool or not isinstance(pending["metadata"], dict) or len(canonical_json(pending["metadata"])) > 8192:
                raise ValueError("Invalid pending injection")
        if not isinstance(state["counts"], dict) or set(state["counts"]) != set(ACTORS):
            raise ValueError("Invalid injection counts")
        for actor, count in state["counts"].items():
            _integer(count, f"counts.{actor}")
        if sum(state["counts"].values()) < result.next_pulse_id - 1 or sum(state["counts"].values()) > result.sequence:
            raise ValueError("Injection counts contradict pulse or event count")
        exposure = state["exposure"]
        if not isinstance(exposure, dict) or set(exposure) != set(PHASES):
            raise ValueError("Invalid exposure phases")
        for phase, values in exposure.items():
            if not isinstance(values, dict) or set(values) != {"positions", *COEFFICIENT_AXES}:
                raise ValueError("Invalid exposure totals")
            positions = _integer(values["positions"], "exposure positions")
            for axis in COEFFICIENT_AXES:
                _number(values[axis], axis, 0 if axis in ATTENUATION_AXES else -4 * positions,
                        positions if axis in ATTENUATION_AXES else 4 * positions)
        result.pulses = copy.deepcopy(state["pulses"])
        result.pending = copy.deepcopy(state["pending"])
        result.counts = dict(state["counts"])
        result.exposure = copy.deepcopy(exposure)
        return result
