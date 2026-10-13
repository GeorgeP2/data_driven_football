"""The expected-points seam.

The optimiser is only as good as the number it maximises. Finding the best 15 players *given*
expected points is a solved problem that runs in milliseconds; forecasting those expected
points is the actual hard part, and the actual edge.

So expected points is a pluggable callable rather than a column baked into the solver. The
baseline here reads FPL's own `ep_next`, which is a weak signal — roughly recent form with a
fixture nudge. A real model (minutes x per-90 scoring rate x fixture difficulty) can be dropped
in behind the same protocol without the solver knowing anything changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from fpl_optimiser.players import Player


@runtime_checkable
class ExpectedPoints(Protocol):
    """Expected FPL points for a player in the gameweek being optimised."""

    @property
    def name(self) -> str:
        """Which model produced these numbers, for logging and provenance."""
        ...

    def __call__(self, player: Player) -> float: ...


@dataclass(frozen=True)
class EpNextProvider:
    """Baseline: FPL's published `ep_next`, discounted by availability.

    Args:
        apply_availability: Multiply by the player's 0..1 availability, so a 25%-fit player is
            worth a quarter of their points and an injured one is worth nothing. Turning this
            off optimises as though everyone is fully fit, which is useful for asking "who
            would I pick in a healthy world?" but should not be used to pick a real squad.
    """

    apply_availability: bool = True
    name: str = "ep_next"

    def __call__(self, player: Player) -> float:
        points = player.ep_next
        if self.apply_availability:
            points *= player.availability
        return max(0.0, points)


@dataclass(frozen=True)
class StaticProvider:
    """Expected points supplied directly, keyed by player id.

    The escape hatch for hand-tuned projections, a CSV of someone else's numbers, or a model
    that runs elsewhere. Players with no entry are worth `default`.
    """

    points_by_player_id: Mapping[int, float]
    default: float = 0.0
    name: str = "static"

    def __call__(self, player: Player) -> float:
        return max(0.0, float(self.points_by_player_id.get(player.id, self.default)))
