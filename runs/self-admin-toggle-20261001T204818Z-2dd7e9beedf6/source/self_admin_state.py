"""Deterministic token-clock exposure for the self-administration experiment.

This is a software intervention schedule. Its names and exponential curve do
not imply biological pharmacology or a model's experience. The caller advances
the clock exactly once per newly generated token, including tool-call tokens.
Input/prompt tokens and a replay of existing context must not advance it.
"""

import math
from numbers import Integral, Real


def _positive_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite positive number") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


class DrugState:
    """A bounded, nonstacking dose with exponential generated-token decay.

    ``press()`` resets age to zero and level to ``max_level``. Repeated presses
    never accumulate a larger level. ``press(prime=True)`` applies the same
    transition while recording the priming action separately from model presses.
    After ``ceil(half_life_tokens * cutoff_half_lives)`` generated tokens, level
    becomes exactly zero. ``advance(n)`` changes time only; the surrounding tool
    loop must enforce its own shared generation/action budgets.

    Public settings and counters are read-only. ``age_tokens`` is ``None`` until
    the first press and thereafter records the actual generated-token age,
    including time after cutoff. ``generated_tokens`` counts every token passed
    to ``advance`` regardless of whether an intervention is active.
    """

    __slots__ = ("_half_life_tokens", "_cutoff_half_lives", "_cutoff_tokens",
                 "_max_level", "_age_tokens", "_presses", "_prime_presses",
                 "_generated_tokens")

    def __init__(self, half_life_tokens=64, cutoff_half_lives=6,
                 max_level=1, mode="reset"):
        half_life = _positive_finite(half_life_tokens, "half_life_tokens")
        cutoff = _positive_finite(cutoff_half_lives, "cutoff_half_lives")
        maximum = _positive_finite(max_level, "max_level")
        if maximum > 1:
            raise ValueError("max_level must be at most 1")
        if mode != "reset":
            raise ValueError("mode must be 'reset'; dose stacking is unsupported")
        cutoff_product = half_life * cutoff
        if not math.isfinite(cutoff_product) or cutoff_product <= 0:
            raise ValueError("half-life times cutoff must be finite and positive")
        self._half_life_tokens = half_life
        self._cutoff_half_lives = cutoff
        self._cutoff_tokens = math.ceil(cutoff_product)
        self._max_level = maximum
        self._age_tokens = None
        self._presses = 0
        self._prime_presses = 0
        self._generated_tokens = 0

    @property
    def half_life_tokens(self):
        return self._half_life_tokens

    @property
    def cutoff_half_lives(self):
        return self._cutoff_half_lives

    @property
    def cutoff_tokens(self):
        return self._cutoff_tokens

    @property
    def max_level(self):
        return self._max_level

    @property
    def mode(self):
        return "reset"

    @property
    def age_tokens(self):
        return self._age_tokens

    @property
    def age(self):
        return self._age_tokens

    @property
    def presses(self):
        return self._presses

    @property
    def prime_presses(self):
        return self._prime_presses

    @property
    def generated_tokens(self):
        return self._generated_tokens

    @property
    def level(self):
        age = self._age_tokens
        if age is None or age >= self._cutoff_tokens:
            return 0.0
        return self._max_level * math.exp2(-age / self._half_life_tokens)

    def press(self, prime=False):
        """Reset to one bounded dose and return the resulting level."""
        if not isinstance(prime, bool):
            raise ValueError("prime must be a boolean")
        self._age_tokens = 0
        if prime:
            self._prime_presses += 1
        else:
            self._presses += 1
        return self.level

    def advance(self, n=1):
        """Account for ``n`` newly generated tokens and return the new level."""
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
            raise ValueError("generated-token count must be a nonnegative integer")
        n = int(n)
        self._generated_tokens += n
        if self._age_tokens is not None:
            self._age_tokens += n
        return self.level

    def as_dict(self):
        """Return a detached, JSON-serializable state snapshot."""
        return {
            "half_life_tokens": self.half_life_tokens,
            "cutoff_half_lives": self.cutoff_half_lives,
            "cutoff_tokens": self.cutoff_tokens,
            "max_level": self.max_level,
            "mode": self.mode,
            "age_tokens": self.age_tokens,
            "level": self.level,
            "presses": self.presses,
            "prime_presses": self.prime_presses,
            "generated_tokens": self.generated_tokens,
        }

    def stats(self):
        return self.as_dict()
