"""The defensive-contribution term: a threshold, discounted for minutes exactly once.

Scaling the Poisson mean by expected minutes *and* sitting that under a convex threshold punished
rotation risk twice. Over 2025-26 it forecast 4.6% of defender-matches clearing the bar against
14.6% that did. The form here applies the minutes discount once, as the chance of playing an hour.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fpl_helpers import lagged, make_player
from fpl_optimiser.backtest.history import Fixture
from fpl_optimiser.backtest.points_model import (
    DEFENSIVE_CONTRIBUTION_POINTS,
    DEFENSIVE_CONTRIBUTION_THRESHOLD,
    _poisson_at_least,
    _shrunk_rate,
    build_priors,
    expected_points,
    forecast_minutes,
)
from fpl_optimiser.backtest.season import GameweekView
from fpl_optimiser.backtest.strength import TeamRatings

GAMEWEEK = 9


def _view(actions_per_match: int, minutes_per_match: int = 90, matches: int = 8) -> GameweekView:
    """One defender with a steady defensive-contribution record and a single fixture ahead."""
    defender = make_player(1, 2)
    stats = replace(
        lagged(1, minutes=(minutes_per_match,) * matches, points=(2,) * matches),
        defensive_contribution=(actions_per_match,) * matches,
    )
    return GameweekView(
        gameweek=GAMEWEEK,
        pool=(defender,),
        lagged={1: stats},
        fixtures={1: (Fixture(opponent_team_id=2, was_home=True, goals_for=0, goals_against=0),)},
        team_ratings=TeamRatings({}, {}, 1.4),
        teams={1: 1},
    )


def _dc_term(view: GameweekView) -> float:
    player = view.pool[0]
    priors = build_priors(view)
    with_rule = expected_points(player, view, priors, defensive_contribution=True)
    without = expected_points(player, view, priors, defensive_contribution=False)
    return with_rule - without


class TestDefensiveContribution:
    def test_minutes_are_discounted_once_not_through_the_mean(self) -> None:
        view = _view(actions_per_match=12)
        stats = view.stats(1)
        minutes = forecast_minutes(stats, GAMEWEEK)
        rate = _shrunk_rate(
            float(sum(stats.defensive_contribution)),
            stats.nineties,
            build_priors(view).defensive_contribution[2],
        )
        expected = (
            _poisson_at_least(DEFENSIVE_CONTRIBUTION_THRESHOLD[2], rate)
            * minutes.probability_sixty
            * DEFENSIVE_CONTRIBUTION_POINTS
        )

        assert _dc_term(view) == pytest.approx(expected)

    def test_a_regular_who_clears_the_bar_every_week_is_forecast_to_keep_doing_so(self) -> None:
        """Twelve actions a match, eight matches running: more likely than not to clear ten
        again when he plays. The double-discounted form said 51%."""
        view = _view(actions_per_match=12)
        chance_when_playing = _dc_term(view) / (
            DEFENSIVE_CONTRIBUTION_POINTS
            * forecast_minutes(view.stats(1), GAMEWEEK).probability_sixty
        )

        assert chance_when_playing > 0.6

    def test_a_full_back_well_short_of_the_bar_gets_little(self) -> None:
        assert _dc_term(_view(actions_per_match=5)) < 0.2

    def test_keepers_are_not_eligible(self) -> None:
        view = _view(actions_per_match=12)
        keeper = make_player(1, 1)
        view = replace(view, pool=(keeper,))

        assert _dc_term(view) == 0.0
