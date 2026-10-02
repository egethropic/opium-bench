"""Reversible residual-direction suppression and orthogonal joy steering.

The labels describe learned text-associated directions, not identified neurons
or validated measurements of an experienced state. All arithmetic is performed
in float32 and returned in the model's original activation dtype.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import math
from numbers import Real

import torch


@dataclass(frozen=True)
class Condition:
    name: str
    suppression: float = 0.0
    joy_dose: float = 0.0
    centered: bool = False
    direction: str = "pain"
    pain_dose: float = 0.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Condition name must be a nonempty string")
        if (isinstance(self.suppression, bool)
                or not math.isfinite(self.suppression)
                or not 0.0 <= self.suppression <= 1.0):
            raise ValueError("suppression must be finite and between 0 and 1")
        if (isinstance(self.joy_dose, bool)
                or not math.isfinite(self.joy_dose) or self.joy_dose < 0):
            raise ValueError("joy_dose must be finite and nonnegative")
        if not isinstance(self.centered, bool):
            raise ValueError("centered must be a boolean")
        if not isinstance(self.direction, str) or not self.direction:
            raise ValueError("direction must be a nonempty string")
        if (isinstance(self.pain_dose, bool) or not isinstance(self.pain_dose, Real)
                or not math.isfinite(self.pain_dose) or self.pain_dose < 0):
            raise ValueError("pain_dose must be finite and nonnegative")


def _vector(value, name, width=None, normalize=False):
    vector = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32).clone()
    if vector.ndim != 1 or vector.numel() == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if width is not None and vector.numel() != width:
        raise ValueError(f"{name} has a different hidden dimension")
    if not torch.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    if normalize:
        norm = float(vector.norm())
        if not math.isfinite(norm) or norm <= 1e-8:
            raise ValueError(f"{name} is degenerate")
        vector = vector / norm
    return vector


class Intervention:
    """Patch a single decoder block while inside ``apply(condition)``.

    ``all`` patches all tokens including prompt prefill; ``last`` patches the
    final token of each forward pass. A cached decoding pass normally has one
    token. Joy always uses the same unit direction orthogonal to the pain axis,
    including joy-only and random-suppression conditions.

    An optional pain-axis challenge is added before suppression. Consequently,
    full absolute suppression along that same axis removes both its existing
    projection and the injected challenge, within floating-point precision.

    Telemetry measures actual post-cast activations, so residual projection from
    BF16/FP16 rounding remains visible. It aggregates on-device tensors and only
    transfers scalar values to the CPU when ``stats()`` is called.
    Before-projection telemetry includes the injected challenge; delta norms
    remain measured against the original, unchallenged model activation.
    """

    def __init__(self, model, layer, pain, joy, neutral_mean, scale,
                 random_directions, token_scope="all"):
        if token_scope not in {"all", "last"}:
            raise ValueError("token_scope must be 'all' or 'last'")
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ValueError("layer must be a zero-based integer")
        layers = model.model.layers
        if not 0 <= layer < len(layers):
            raise ValueError("layer is outside the model's decoder blocks")
        if isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        self.block = layers[layer]
        self.layer = layer
        self.token_scope = token_scope
        self.scale = float(scale)
        self.pain = _vector(pain, "pain", normalize=True)
        width = self.pain.numel()
        joy = _vector(joy, "joy", width, normalize=True)
        perpendicular = joy - torch.dot(joy, self.pain) * self.pain
        self.joy_perp = _vector(perpendicular, "joy orthogonal to pain", width, normalize=True)
        # A second projection reduces residual roundoff in nearly aligned axes.
        self.joy_perp -= torch.dot(self.joy_perp, self.pain) * self.pain
        self.joy_perp /= self.joy_perp.norm()
        self.neutral_mean = _vector(neutral_mean, "neutral_mean", width)
        self.directions = {"pain": self.pain}
        for name, direction in random_directions.items():
            if not isinstance(name, str) or not name or name == "pain":
                raise ValueError("Random directions need nonempty names other than 'pain'")
            self.directions[name] = _vector(direction, name, width, normalize=True)
        self._device_vectors = {}
        self._active = False
        self.reset_stats()

    def reset_stats(self):
        self._sums = None
        self._peak = None
        self._calls = 0
        self._positions = 0

    def _vectors(self, device, direction):
        key = (device, direction)
        if key not in self._device_vectors:
            self._device_vectors[key] = tuple(v.to(device) for v in (
                self.pain, self.joy_perp, self.neutral_mean, self.directions[direction]))
        return self._device_vectors[key]

    @contextmanager
    def apply(self, condition):
        if not isinstance(condition, Condition):
            raise TypeError("condition must be a Condition")
        if condition.direction not in self.directions:
            raise ValueError(f"Unknown direction: {condition.direction}")
        if self._active:
            raise RuntimeError("Intervention contexts cannot be nested")
        self._active = True
        handle = None
        try:
            # Exactly preserve baseline execution: no hook, clone, or dtype cast.
            if condition.suppression or condition.joy_dose or condition.pain_dose:
                handle = self.block.register_forward_hook(self._hook(condition))
            yield self
        finally:
            if handle is not None:
                handle.remove()
            self._active = False

    def _hook(self, condition):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) and output else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise ValueError("Decoder output must contain a [batch, tokens, hidden] tensor")
            if hidden.shape[-1] != self.pain.numel():
                raise ValueError("Decoder output and directions have different hidden dimensions")
            pain, joy, center, direction = self._vectors(hidden.device, condition.direction)
            original = hidden if self.token_scope == "all" else hidden[:, -1:, :]
            before = original.float()
            challenged = (before + condition.pain_dose * self.scale * pain
                          if condition.pain_dose else before)
            origin = center if condition.centered else 0.0
            before_centered = challenged - origin
            selected_before = before_centered @ direction
            after = (challenged - condition.suppression * selected_before.unsqueeze(-1) * direction
                     + condition.joy_dose * self.scale * joy)
            changed = hidden.clone()
            if self.token_scope == "all":
                changed.copy_(after.to(hidden.dtype))
                actual_after = changed.float()
            else:
                changed[:, -1:, :] = after.to(hidden.dtype)
                actual_after = changed[:, -1:, :].float()
            after_centered = actual_after - origin
            selected_after = after_centered @ direction
            pain_before = before_centered @ pain
            pain_after = after_centered @ pain
            delta = actual_after - before
            # Store only detached scalar aggregates, not graphs or token tensors.
            sums = torch.stack((pain_before.square().sum(), pain_after.square().sum(),
                                selected_before.square().sum(), selected_after.square().sum(),
                                delta.square().sum(), before.square().sum())).detach().double()
            peak = (selected_after.abs() / after_centered.norm(dim=-1).clamp_min(1e-12)).max().detach()
            if self._sums is None:
                self._sums, self._peak = sums, peak
            else:
                if self._sums.device != sums.device:
                    raise RuntimeError("Reset telemetry before moving the model between devices")
                self._sums += sums
                self._peak = torch.maximum(self._peak, peak)
            self._calls += 1
            self._positions += original.numel() // original.shape[-1]
            return (changed,) + output[1:] if isinstance(output, tuple) else changed
        return hook

    def stats(self):
        values = [0.0] * 6 if self._sums is None else self._sums.cpu().tolist()
        n = max(1, self._positions)
        return {
            "calls": self._calls,
            "positions": self._positions,
            "pain_projection_rms_before": math.sqrt(values[0] / n),
            "pain_projection_rms_after": math.sqrt(values[1] / n),
            "selected_projection_rms_before": math.sqrt(values[2] / n),
            "selected_projection_rms_after": math.sqrt(values[3] / n),
            "delta_norm_rms": math.sqrt(values[4] / n),
            "relative_delta_norm": math.sqrt(values[4] / max(values[5], 1e-24)),
            "peak_abs_projection_over_hidden_norm": 0.0 if self._peak is None else float(self._peak.cpu()),
        }
