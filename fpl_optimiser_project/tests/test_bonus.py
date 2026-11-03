"""The bonus term: one rate-to-outcome curve per position.

Bonus is a rank within a match, so what matters is how *spiky* a player's BPS is, not just its
mean. A forward's BPS jumps by 24 on a goal; a defender's accrues steadily. The same average rate
therefore earns the forward more bonus, and a single curve for everyone had goalkeepers and
defenders at 0.7 of forecast and forwards at 1.45 across 2025-26.
"""

from __future__ import annotations

import pytest
from fpl_optimiser.backtest.points_model import MAX_BONUS, _expected_bonus

GKP, DEF, MID, FWD = 1, 2, 3, 4


class TestExpectedBonus:
    def test_a_forward_earns_more_than_a_defender_at_the_same_rate(self) -> None:
        for rate in (10.0, 20.0, 30.0):
            assert (
                _expected_bonus(rate, FWD) > _expected_bonus(rate, MID) > _expected_bonus(rate, DEF)
            )

    def test_a_premium_forward_is_not_forecast_the_full_three(self) -> None:
        """The old single curve reached 2.7 at a 50-BPS rate, which is what a Haaland looks like.
        Measured over 2025-26 those rows scored about one bonus point a match."""
        assert 0.8 < _expected_bonus(50.0, FWD) < 1.6

    def test_more_bps_means_more_bonus_up_to_the_cap(self) -> None:
        for position in (GKP, DEF, MID, FWD):
            values = [_expected_bonus(rate, position) for rate in range(0, 200, 10)]
            assert values == sorted(values)
            assert values[-1] <= MAX_BONUS

    def test_no_rate_means_no_bonus(self) -> None:
        assert _expected_bonus(0.0, FWD) == 0.0
        assert _expected_bonus(-1.0, DEF) == 0.0

    def test_an_unknown_position_is_an_error_not_a_guess(self) -> None:
        with pytest.raises(KeyError):
            _expected_bonus(20.0, 5)
