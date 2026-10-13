"""Tests for the forecasting model, the multi-gameweek planner and the chips.

The model's job is to answer "will they play, and how well?", and the planner's is to stop the
squad churning. Each test below pins one of those behaviours to a scenario where the right answer
is obvious by construction — which is the only way to tell a working model from a plausible one.
"""

from __future__ import annotations

import pytest
from fpl_helpers import lagged, make_player, record
from fpl_optimiser.backtest.chips import Chip, ChipState, choose_chip
from fpl_optimiser.backtest.history import Fixture
from fpl_optimiser.backtest.points_model import (
    CEILING_START_PROBABILITY,
    FLOOR_START_PROBABILITY,
    forecast_minutes,
)
from fpl_optimiser.backtest.scoring import score_gameweek
from fpl_optimiser.backtest.strength import TeamRatings, build_ratings


class TestMinutesModel:
    def test_a_nailed_starter_is_expected_to_start(self) -> None:
        nailed = lagged(1, minutes=(90, 90, 90, 90, 90, 90))

        forecast = forecast_minutes(nailed, 7)

        # Not 1.0 — nobody is a certainty. Measured over 27,000 predictions, players the model
        # called nailed-on started 86% of the time; claiming 100% is how a squad fills up with
        # absentees. The ceiling is calibration, not pessimism.
        assert forecast.probability_starts == pytest.approx(CEILING_START_PROBABILITY)
        assert forecast.expected_minutes > 70

    def test_nobody_is_a_certainty(self) -> None:
        """Regression: the uncalibrated model claimed 98%, and only 86% of them started.

        There is deliberately no floor at the other end. A 5% floor is *better calibrated* — a
        player with no recent starts really does start about one game in twenty — but it made the
        points forecast measurably worse, because it hands thousands of non-players a small noisy
        forecast where zero was the right answer. Calibration is a means, not the goal.
        """
        nailed = forecast_minutes(lagged(1, minutes=(90,) * 6), 7)
        never = forecast_minutes(lagged(2, minutes=(0,) * 6), 7)

        assert nailed.probability_starts <= CEILING_START_PROBABILITY
        assert nailed.probability_starts > 0.8, "still clearly a starter"
        assert never.probability_starts == FLOOR_START_PROBABILITY == 0.0

    def test_a_player_who_has_lost_his_place_is_not(self) -> None:
        # Started early, dropped since. The decay must weight the recent benchings most.
        dropped = lagged(1, minutes=(90, 90, 90, 0, 0, 0))

        forecast = forecast_minutes(dropped, 7)

        assert forecast.probability_starts < 0.3, "recent form must dominate"
        assert forecast.expected_minutes < 30

    def test_recency_beats_history(self) -> None:
        # Same six matches, opposite order. The one starting *now* must rate far higher.
        rising = forecast_minutes(lagged(1, minutes=(0, 0, 0, 90, 90, 90)), 7)
        falling = forecast_minutes(lagged(2, minutes=(90, 90, 90, 0, 0, 0)), 7)

        assert rising.probability_starts > falling.probability_starts * 3

    def test_an_injury_flag_scales_everything_down(self) -> None:
        nailed = lagged(1, minutes=(90, 90, 90))

        fit = forecast_minutes(nailed, 4)
        doubtful = forecast_minutes(nailed, 4, injury_availability=0.25)

        assert doubtful.expected_minutes == pytest.approx(fit.expected_minutes * 0.25)
        assert doubtful.probability_starts == pytest.approx(fit.probability_starts * 0.25)

    def test_a_flagged_out_player_is_expected_to_play_nothing(self) -> None:
        nailed = lagged(1, minutes=(90, 90, 90))

        forecast = forecast_minutes(nailed, 4, injury_availability=0.0)

        assert forecast.expected_minutes == 0.0

    def test_gameweek_one_falls_back_on_last_season(self) -> None:
        regular = lagged(1, previous_season_minutes=3000)
        fringe = lagged(2, previous_season_minutes=200)

        assert forecast_minutes(regular, 1).probability_starts > 0.8
        assert forecast_minutes(fringe, 1).probability_starts < 0.15


class TestTeamStrength:
    def _season(self, xg_for: float, xg_against: float, matches: int) -> list:
        """Team 1 playing team 2 `matches` times, each side creating a fixed amount of xG.

        Both sides are required: a team's expected goals *conceded* is now derived from what the
        opposition actually generated, so a fixture with only one half present is skipped.
        """
        records = []
        for gw in range(1, matches + 1):
            records.append(_xg_record(player_id=1, gameweek=gw, team_id=1, opponent=2, xg=xg_for))
            records.append(
                _xg_record(player_id=2, gameweek=gw, team_id=2, opponent=1, xg=xg_against)
            )
        return records

    def test_ratings_are_shrunk_toward_the_mean_early_on(self) -> None:
        # One match, four goals. A model that believes this is a 4-goals-a-game team will chase
        # noise all season, so the rating must stay much closer to average than to 4.
        ratings = build_ratings(self._season(4, 0, matches=1))

        assert ratings.attack[1] < 2.0, "one thrashing must not make a superteam"

    def test_evidence_eventually_outweighs_the_prior(self) -> None:
        few = build_ratings(self._season(4, 0, matches=1))
        many = build_ratings(self._season(4, 0, matches=20))

        assert many.attack[1] > few.attack[1] * 1.5

    def test_a_good_attack_expects_more_against_a_poor_defence(self) -> None:
        ratings = TeamRatings(attack={1: 2.5, 2: 1.0}, defence={1: 0.8, 2: 2.0}, league_mean=1.4)

        strong, _ = ratings.expected_goals(1, 2, at_home=True)
        weak, _ = ratings.expected_goals(2, 1, at_home=True)

        assert strong > weak

    def test_home_advantage_raises_expected_goals(self) -> None:
        ratings = TeamRatings(attack={1: 1.4, 2: 1.4}, defence={1: 1.4, 2: 1.4}, league_mean=1.4)

        home, _ = ratings.expected_goals(1, 2, at_home=True)
        away, _ = ratings.expected_goals(1, 2, at_home=False)

        assert home > away

    def test_a_clean_sheet_is_likelier_against_a_toothless_attack(self) -> None:
        ratings = TeamRatings(attack={1: 0.4, 2: 2.5}, defence={1: 1.4, 2: 1.4}, league_mean=1.4)

        # Team 2 faces the feeble attack of team 1; team 1 faces the strong attack of team 2.
        against_weak = ratings.clean_sheet_probability(2, 1, at_home=True)
        against_strong = ratings.clean_sheet_probability(1, 2, at_home=True)

        assert against_weak > against_strong


class TestChips:
    def _remaining(self, first: int, last: int = 38) -> list[int]:
        return list(range(first, last + 1))

    def test_bench_boost_waits_for_a_big_bench(self) -> None:
        state = ChipState()

        meagre = choose_chip(
            5,
            self._remaining(5),
            state,
            bench_expected_points=6.0,
            captain_expected_points=2.0,
            squad_with_fixtures=15,
            wildcard_gain=0.0,
        )
        bumper = choose_chip(
            5,
            self._remaining(5),
            state,
            bench_expected_points=30.0,
            captain_expected_points=2.0,
            squad_with_fixtures=15,
            wildcard_gain=0.0,
        )

        assert meagre.chip is None
        assert bumper.chip is Chip.BENCH_BOOST

    def test_an_unplayed_chip_is_played_at_the_last_chance(self) -> None:
        # An unplayed chip is worth exactly nothing, so on the final gameweek of the half the
        # threshold must collapse and a mediocre opportunity becomes worth taking.
        state = ChipState()

        held = choose_chip(
            5,
            self._remaining(5, 19),
            state,
            bench_expected_points=6.0,
            captain_expected_points=1.0,
            squad_with_fixtures=15,
            wildcard_gain=0.0,
        )
        forced = choose_chip(
            19,
            [19],
            state,
            bench_expected_points=6.0,
            captain_expected_points=1.0,
            squad_with_fixtures=15,
            wildcard_gain=0.0,
        )

        assert held.chip is None, "early on, hold out for something better"
        assert forced.chip is not None, "at the deadline, take what is on offer"

    def test_a_blank_gameweek_triggers_the_free_hit(self) -> None:
        state = ChipState()

        decision = choose_chip(
            29,
            self._remaining(29),
            state,
            bench_expected_points=2.0,
            captain_expected_points=2.0,
            squad_with_fixtures=6,  # nine of the fifteen have no match
            wildcard_gain=0.0,
        )

        assert decision.chip is Chip.FREE_HIT

    def test_each_chip_can_only_be_played_once_per_half(self) -> None:
        state = ChipState()
        state.play(Chip.BENCH_BOOST, 5)

        assert not state.available(Chip.BENCH_BOOST, 10), "same half — already spent"
        assert state.available(Chip.BENCH_BOOST, 25), "second half — a fresh one"
        with pytest.raises(ValueError, match="already been played"):
            state.play(Chip.BENCH_BOOST, 10)


class TestChipScoring:
    def _squad_and_actuals(self, rules):
        from test_backtest import _legal_xi, _squad_of

        xi = _legal_xi()
        bench = [make_player(2, 1), make_player(40, 3), make_player(41, 3)]
        squad = _squad_of(xi, bench, captain=xi[5], vice=xi[6])
        actuals = {p.id: record(p.id, 1, points=2) for p in xi + bench}
        return squad, actuals

    def test_bench_boost_counts_all_fifteen(self, rules) -> None:
        squad, actuals = self._squad_and_actuals(rules)

        normal = score_gameweek(squad, actuals, rules, 1)
        boosted = score_gameweek(squad, actuals, rules, 1, bench_boost=True)

        # Everyone scores 2. Normally the eleven count, plus the captain again: 24.
        assert normal.points_before_hits == 24
        # Boosted, the bench counts too — every one of them, at 2 points each.
        assert boosted.points_before_hits == normal.points_before_hits + 2 * len(squad.bench)
        assert len(squad.bench) > 0, "the fixture must actually have a bench to boost"

    def test_triple_captain_triples_rather_than_doubles(self, rules) -> None:
        squad, actuals = self._squad_and_actuals(rules)
        actuals[squad.captain.id] = record(squad.captain.id, 1, points=10)

        doubled = score_gameweek(squad, actuals, rules, 1)
        tripled = score_gameweek(squad, actuals, rules, 1, captain_multiplier=3)

        assert tripled.points_before_hits == doubled.points_before_hits + 10


def _xg_record(*, player_id: int, gameweek: int, team_id: int, opponent: int, xg: float):
    """A record carrying only the expected goals the team strength model reads."""
    from dataclasses import replace

    base = record(
        player_id=player_id,
        gameweek=gameweek,
        team_id=team_id,
        opponents=(
            Fixture(
                opponent_team_id=opponent,
                was_home=True,
                goals_for=0,
                goals_against=0,
            ),
        ),
    )
    return replace(base, expected_goals=xg)
