"""The goals and assists rates shrink toward the player's own last season, not the position.

Shrinking everyone toward the positional mean treated a 0.9-xG-a-game striker as a 0.3 one until
he had re-proved it, and on 2025-26 that took most of a season. Where a player has last season's
minutes, that record is his prior; where he has none, the positional mean still is.
"""

from __future__ import annotations

from dataclasses import replace

from fpl_helpers import lagged, make_player
from fpl_optimiser.backtest.history import Fixture
from fpl_optimiser.backtest.points_model import build_priors, expected_points
from fpl_optimiser.backtest.season import GameweekView
from fpl_optimiser.backtest.strength import TeamRatings

GAMEWEEK = 9
FWD = 4


def _view(*forwards) -> GameweekView:
    """Forwards who have all played eight full matches this season without an expected goal,
    so the only thing telling them apart is what they did last year. A ninth forward who does
    create chances is in the pool so that the positional prior is a real number, not zero."""
    pool, stats = [], {}
    for player_id, previous_xg, previous_minutes in (*forwards, (99, 0.0, 0)):
        pool.append(make_player(player_id, FWD))
        stats[player_id] = replace(
            lagged(player_id, minutes=(90,) * 8, points=(2,) * 8),
            expected_goals=((0.4,) * 8 if player_id == 99 else (0.0,) * 8),
            previous_season_expected_goals=previous_xg,
            previous_season_minutes=previous_minutes,
        )
    return GameweekView(
        gameweek=GAMEWEEK,
        pool=tuple(pool),
        lagged=stats,
        fixtures={
            p.id: (Fixture(opponent_team_id=2, was_home=True, goals_for=0, goals_against=0),)
            for p in pool
        },
        team_ratings=TeamRatings({}, {}, 1.4),
        teams={p.id: 1 for p in pool},
    )


def _forecast(view: GameweekView) -> dict[int, float]:
    priors = build_priors(view)
    return {p.id: expected_points(p, view, priors) for p in view.pool}


class TestGoalsPrior:
    def test_last_season_lifts_a_proven_scorer_over_an_unknown(self) -> None:
        # 25 xG in 3,000 minutes is a striker; no history is a positional-average forward.
        view = _view((1, 25.0, 3000), (2, 0.0, 0))
        points = _forecast(view)

        assert points[1] > points[2]

    def test_no_history_means_the_positional_prior_not_a_zero(self) -> None:
        # A player with no last season must not be read as one who never threatened.
        view = _view((1, 0.0, 0), (2, 0.0, 2700))
        points = _forecast(view)

        assert points[1] > points[2], "zero minutes is no evidence; zero xG in 2,700 minutes is"

    def test_a_short_last_season_counts_for_less(self) -> None:
        # Same rate (0.75 xG/90), one over a full season and one over three matches.
        view = _view((1, 25.0, 3000), (2, 2.25, 270))
        points = _forecast(view)

        assert points[1] > points[2]
