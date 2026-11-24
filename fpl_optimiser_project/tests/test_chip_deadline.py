"""The live advisor must count chip deadlines against the whole season.

`load_live_history` builds only as far as the planning horizon. Passing its gameweeks to
`choose_chip` made the GW19 expiry look four weeks away at GW12 instead of seven, so every
threshold was scaled by 0.8 and a 15-point bench played a bench boost that needs 18.
"""

from __future__ import annotations

from fpl_optimiser.backtest.chips import Chip, ChipState, choose_chip
from fpl_optimiser.live import season_gameweeks

PAYLOAD = {"events": [{"id": gameweek} for gameweek in range(38, 0, -1)]}


def _decide(remaining: list[int]) -> Chip | None:
    return choose_chip(
        12,
        remaining,
        ChipState(),
        bench_expected_points=15.0,
        captain_expected_points=5.0,
        squad_with_fixtures=15,
        wildcard_gain=0.0,
    ).chip


def test_season_gameweeks_covers_the_whole_season_in_order() -> None:
    assert season_gameweeks(PAYLOAD) == list(range(1, 39))


def test_a_bench_below_the_threshold_is_held_seven_weeks_from_expiry() -> None:
    assert _decide(season_gameweeks(PAYLOAD)) is None


def test_the_horizon_truncated_list_was_what_fired_it_early() -> None:
    horizon_only = list(range(1, 17))  # what `history.gameweeks` held at GW12
    assert _decide(horizon_only) is Chip.BENCH_BOOST
