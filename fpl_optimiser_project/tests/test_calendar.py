"""Tests for the multi-competition fixture calendar."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fpl_optimiser.backtest.calendar import FixtureCalendar, build_calendar, season_start_year


def at(day: int, hour: int = 15) -> datetime:
    """A kickoff in a notional month."""
    return datetime(2025, 3, day, hour, tzinfo=UTC)


class TestFixtureCalendar:
    def test_rest_is_measured_from_the_last_match_in_any_competition(self) -> None:
        # Sunday league game, having played on the Thursday. FPL's own data would show a full
        # week's gap between gameweeks; the truth is three days.
        calendar = FixtureCalendar(
            {"Spurs": [at(2), at(6, hour=20), at(9)]}  # Sun, Thu (Europe), Sun
        )

        assert calendar.rest_before("Spurs", at(9)) == pytest.approx(2.79, abs=0.01)

    def test_a_club_with_no_earlier_match_has_no_rest_figure(self) -> None:
        calendar = FixtureCalendar({"Spurs": [at(9)]})

        assert calendar.rest_before("Spurs", at(9)) is None

    def test_an_unknown_club_has_no_rest_figure(self) -> None:
        calendar = FixtureCalendar({"Spurs": [at(2)]})

        assert calendar.rest_before("Barcelona", at(9)) is None

    def test_matches_within_counts_a_congested_week(self) -> None:
        calendar = FixtureCalendar({"Spurs": [at(2), at(6), at(9), at(12)]})

        assert calendar.matches_within("Spurs", at(12), days=7) == 2  # the 6th and the 9th


class TestBuildCalendar:
    def test_european_fixtures_are_merged_into_the_league_ones(self, monkeypatch) -> None:
        from fpl_optimiser.backtest import calendar as module

        monkeypatch.setattr(
            module,
            "load_european_fixtures",
            lambda season, **kwargs: {
                "Spurs": [at(6, hour=20)],
                "Barcelona": [at(6, hour=20)],  # a foreign club: correctly ignored
            },
        )

        merged = build_calendar({"Spurs": [at(2), at(9)], "Everton": [at(2)]}, "2024-25")

        # Spurs now show three days' rest, not seven — the Thursday is visible.
        spurs_rest = merged.rest_before("Spurs", at(9))
        assert spurs_rest is not None
        assert spurs_rest < 3.0
        # Everton played no European football, so they still show a full week.
        assert merged.rest_before("Everton", at(9)) == pytest.approx(7.0)
        assert "Barcelona" not in merged.teams, "foreign clubs are not Premier League teams"

    def test_a_club_not_in_europe_is_untouched(self, monkeypatch) -> None:
        from fpl_optimiser.backtest import calendar as module

        monkeypatch.setattr(module, "load_european_fixtures", lambda season, **kw: {})

        merged = build_calendar({"Everton": [at(2), at(9)]}, "2024-25")

        assert merged.rest_before("Everton", at(9)) == pytest.approx(7.0)


class TestSeasonNaming:
    def test_the_start_year_is_taken(self) -> None:
        assert season_start_year("2024-25") == 2024

    def test_a_malformed_season_raises(self) -> None:
        with pytest.raises(ValueError, match="YYYY-YY"):
            season_start_year("last year")
