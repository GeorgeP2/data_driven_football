"""What a start is worth is the player's own habit, not a constant.

Measured over 2025-26, a constant 82 minutes per start missed by 9.9 minutes; the player's own
recent minutes-when-starting missed by 7.9. The same record says whether a start lasts to the
hour, which is what the second appearance point and a clean sheet are paid on.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fpl_helpers import lagged
from fpl_optimiser.backtest.points_model import STARTER_MINUTES, forecast_minutes

GAMEWEEK = 9


class TestMinutesWhenStarting:
    def test_a_player_who_finishes_his_matches_is_forecast_more_minutes(self) -> None:
        finishes = forecast_minutes(lagged(1, minutes=(90,) * 6), GAMEWEEK)
        hooked = forecast_minutes(lagged(2, minutes=(65,) * 6), GAMEWEEK)

        def per_start(forecast):
            return forecast.expected_minutes / forecast.probability_starts

        assert per_start(finishes) > STARTER_MINUTES > per_start(hooked)

    def test_an_early_hook_every_week_costs_the_hour(self) -> None:
        # Both start every week. Only one of them is reliably on the pitch at 60.
        finishes = forecast_minutes(lagged(1, minutes=(90,) * 6), GAMEWEEK)
        hooked = forecast_minutes(replace(lagged(2, minutes=(55,) * 6), starts=(1,) * 6), GAMEWEEK)

        assert hooked.probability_starts > 0.8, "he does start"
        assert hooked.probability_sixty < 0.5
        assert finishes.probability_sixty > 0.85

    def test_a_single_start_barely_moves_the_prior(self) -> None:
        # One 90-minute start against two matches' worth of prior: closer to 82 than to 90.
        one_start = forecast_minutes(lagged(1, minutes=(0, 0, 0, 0, 0, 90)), GAMEWEEK)
        starter_minutes = one_start.expected_minutes / one_start.probability_starts

        assert starter_minutes == pytest.approx((90 + 2 * STARTER_MINUTES) / 3, abs=0.5)
