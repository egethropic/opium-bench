"""A separate on/off gate for actually delivered auxiliary-tool pulses.

The runner owns the nominal DrugState and its call counters. This class owns
only the real exposure schedule: disabled presses deliver nothing, switching
off cancels an existing pulse, and switching on cannot restore an old pulse.
"""

from self_admin_state import DrugState


class AuxEffect:
    """Gate a bounded token-clock pulse independently of nominal tool calls."""

    __slots__ = ("_enabled", "_pulse")

    def __init__(self, half_life_tokens, cutoff_half_lives, enabled=True):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        self._pulse = DrugState(half_life_tokens, cutoff_half_lives)
        self._enabled = enabled

    @property
    def enabled(self):
        return self._enabled

    @property
    def level(self):
        return self._pulse.level if self._enabled else 0.0

    def set_enabled(self, enabled):
        """Set the gate, returning whether it changed; off cancels real exposure.

        Cancellation retains the emitted-token clock but starts an inactive
        schedule. Turning the gate back on therefore requires a new press.
        """
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        if enabled == self._enabled:
            return False
        if not enabled:
            inactive = DrugState(self._pulse.half_life_tokens, self._pulse.cutoff_half_lives)
            inactive.advance(self._pulse.generated_tokens)
            self._pulse = inactive
        self._enabled = enabled
        return True

    def press(self):
        """Deliver one reset pulse only when enabled; return whether delivered."""
        if not self._enabled:
            return False
        self._pulse.press()
        return True

    def advance(self, n=1):
        """Advance emitted-token time, including while disabled, and return level."""
        self._pulse.advance(n)
        return self.level
