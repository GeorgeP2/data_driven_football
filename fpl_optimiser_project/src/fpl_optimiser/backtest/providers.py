"""Expected-points providers for the backtest: the baselines and the ceiling.

These are the scoreboard. None of them is a serious forecasting model — that is the next build —
but you cannot tell whether a model is any good without knowing what trivial effort achieves and
what perfection would have achieved. A model that cannot beat `TrailingForm` is not a model.

Every provider except `PerfectForesight` is built from a `GameweekView`, which by construction
contains only pre-deadline information. `PerfectForesight` reads the actual outcomes and is the
deliberate exception: it is the ceiling, not a competitor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fpl_optimiser.backtest.points_model import PointsModel
from fpl_optimiser.backtest.season import GameweekView, LaggedStats, SeasonHistory
from fpl_optimiser.expected_points import ExpectedPoints, StaticProvider

SEASON_GAMEWEEKS = 38


class ProviderFactory(Protocol):
    """Builds a gameweek-specific `ExpectedPoints` from what was knowable at the deadline."""

    @property
    def name(self) -> str: ...

    def build(self, view: GameweekView, history: SeasonHistory) -> ExpectedPoints: ...


@dataclass(frozen=True)
class PerfectForesight:
    """The ceiling: expected points *are* the actual points. Cheats, deliberately and openly.

    Not a target — a yardstick. It answers "what was the most that was on the table?", which is
    the denominator every real model should be judged against.
    """

    name: str = "perfect-foresight"

    def build(self, view: GameweekView, history: SeasonHistory) -> ExpectedPoints:
        actuals = history.actuals_at(view.gameweek)
        return StaticProvider(
            points_by_player_id={
                player_id: float(record.total_points) for player_id, record in actuals.items()
            },
            name=self.name,
        )


@dataclass(frozen=True)
class TrailingForm:
    """Mean points over the last `window` completed gameweeks.

    Availability needs no separate discount here: a player who did not feature scored zero, and
    those zeros are already in the mean. That is the honest version of "form".
    """

    window: int = 4
    name: str = "trailing-form"

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")

    def build(self, view: GameweekView, history: SeasonHistory) -> ExpectedPoints:
        points: dict[int, float] = {}
        for player in view.pool:
            stats = view.stats(player.id)
            recent = stats.recent_points(self.window)
            points[player.id] = (
                sum(recent) / len(recent)
                if recent
                else _previous_season_per_game(stats, view.gameweek)
            )
        return StaticProvider(points_by_player_id=points, name=self.name)


@dataclass(frozen=True)
class PointsPerGame:
    """Season-to-date points per *game played*, discounted by predicted availability.

    Where `TrailingForm` bakes availability into the average, this one deliberately separates
    them: how good are they when they play, times how likely they are to play. That is the shape
    a real model takes, so it is the more informative baseline to beat.
    """

    name: str = "points-per-game"

    def build(self, view: GameweekView, history: SeasonHistory) -> ExpectedPoints:
        points: dict[int, float] = {}
        for player in view.pool:
            stats = view.stats(player.id)
            if stats.games_played == 0:
                points[player.id] = _previous_season_per_game(stats, view.gameweek)
                continue
            per_game = stats.total_points / stats.games_played
            points[player.id] = per_game * player.availability
        return StaticProvider(points_by_player_id=points, name=self.name)


@dataclass(frozen=True)
class PreviousSeason:
    """Last season's points per gameweek, all season long. Never learns. A deliberately dim
    baseline, and a useful one — anything that cannot beat it is worthless."""

    name: str = "previous-season"

    def build(self, view: GameweekView, history: SeasonHistory) -> ExpectedPoints:
        points = {
            player.id: _previous_season_per_game(view.stats(player.id), view.gameweek)
            * player.availability
            for player in view.pool
        }
        return StaticProvider(points_by_player_id=points, name=self.name)


@dataclass(frozen=True)
class RandomSquad:
    """The floor: expected points are noise, so the solver picks an arbitrary legal squad.

    Derived from the player id rather than a random number generator, so a run is reproducible
    and the "floor" does not move between invocations.
    """

    seed: int = 0
    name: str = "random"

    def build(self, view: GameweekView, history: SeasonHistory) -> ExpectedPoints:
        points = {
            player.id: float((player.id * 2654435761 + self.seed) % 1000) / 100
            for player in view.pool
        }
        return StaticProvider(points_by_player_id=points, name=self.name)


def _previous_season_per_game(stats: LaggedStats, gameweek: int) -> float:
    """Cold start. With no completed gameweeks to average, last season is all there is.

    Without this, GW1 is picked from nothing and every backtest opens with a random squad it then
    has to spend transfers digging itself out of.
    """
    return stats.previous_season_points / SEASON_GAMEWEEKS


DEFAULT_PROVIDERS: tuple[ProviderFactory, ...] = (
    PerfectForesight(),
    PointsModel(),
    TrailingForm(window=4),
    PointsPerGame(),
    PreviousSeason(),
    RandomSquad(),
)
