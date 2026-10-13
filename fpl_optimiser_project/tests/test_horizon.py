"""The live planning horizon.

`--horizon 5` was inert: `load_live_history` built the completed gameweeks plus *one* upcoming
one, so the list the planner sliced never had more than a single future week in it. The
five-gameweek planner — the biggest measured lever in the system — ran as a one-week planner
against the live API, with no error and no warning.

The fix has two halves and the second is the one that bites. Building five future gameweeks is
easy; the trap is that `view_at` treats every lower-numbered gameweek as *completed*, so the
unplayed placeholders (schedule filled in, everything else zero) would be read as real matches
in which every player in the game logged zero minutes. The minutes model decays fast by design,
so that forecasts a squad of players who never play — the GW1 cold start again, arriving from a
different direction. Hence `known_through`.
"""

from __future__ import annotations

import pytest
from fpl_helpers import bootstrap_payload
from fpl_optimiser.backtest.history import Fixture, GameweekRecord
from fpl_optimiser.backtest.season import SeasonHistory
from fpl_optimiser.live import load_live_history
from fpl_optimiser.rules import GameRules


def played(player_id: int, gameweek: int, *, minutes: int = 90) -> GameweekRecord:
    """A gameweek that actually happened."""
    return GameweekRecord(
        player_id=player_id,
        gameweek=gameweek,
        name=f"p{player_id}",
        element_type=3,
        team_id=1,
        team_name="ARS",
        price_tenths=50,
        total_points=5,
        minutes=minutes,
        started=1 if minutes >= 60 else 0,
        expected_goals=0.1,
        expected_assists=0.1,
        bonus=0,
        saves=0,
        defensive_contribution=0,
        yellow_cards=0,
        red_cards=0,
        bonus_points_system=10,
        expected_goals_conceded=1.0,
        fpl_expected_points=None,
        opponents=(Fixture(opponent_team_id=2, was_home=True, goals_for=1, goals_against=0),),
    )


def scheduled(player_id: int, gameweek: int) -> GameweekRecord:
    """An upcoming gameweek as `load_live_history` writes it: fixture known, rest zeroed."""
    return GameweekRecord(
        player_id=player_id,
        gameweek=gameweek,
        name=f"p{player_id}",
        element_type=3,
        team_id=1,
        team_name="ARS",
        price_tenths=50,
        total_points=0,
        minutes=0,
        started=0,
        expected_goals=0.0,
        expected_assists=0.0,
        bonus=0,
        saves=0,
        defensive_contribution=0,
        yellow_cards=0,
        red_cards=0,
        bonus_points_system=0,
        expected_goals_conceded=0.0,
        fpl_expected_points=None,
        opponents=(Fixture(opponent_team_id=2, was_home=False, goals_for=0, goals_against=0),),
    )


@pytest.fixture
def live_history(rules: GameRules) -> SeasonHistory:
    """Two played gameweeks and a five-week horizon of scheduled ones, as a live GW3 run has."""
    records: list[GameweekRecord] = []
    for player_id in range(1, 4):
        records.extend(played(player_id, gw) for gw in (1, 2))
        records.extend(scheduled(player_id, gw) for gw in range(3, 8))
    return SeasonHistory(records, rules)


class TestKnownThrough:
    """The half of the fix that stops the placeholders being read as played matches."""

    def test_a_later_view_reads_no_more_history_than_the_next_one(
        self, live_history: SeasonHistory
    ) -> None:
        # Planning GW3..GW7 from GW2, every view knows exactly the same thing.
        for gameweek in range(3, 8):
            view = live_history.view_at(gameweek, known_through=2)

            assert view.stats(1).minutes == (90, 90), (
                f"GW{gameweek} must see the two played gameweeks and nothing else"
            )

    def test_without_it_the_unplayed_weeks_read_as_zero_minute_matches(
        self, live_history: SeasonHistory
    ) -> None:
        # The bug this guards: the default cutoff is right for a backtest and wrong live.
        view = live_history.view_at(7)

        assert view.stats(1).minutes == (90, 90, 0, 0, 0, 0), (
            "the default treats GW3-GW6 as played, which live they are not"
        )

    def test_a_proven_starter_stays_available_across_the_whole_horizon(
        self, live_history: SeasonHistory
    ) -> None:
        # The consequence that matters. Four fake zero-minute weeks would decay a nailed-on
        # starter to unpickable, and the planner would field whoever happened to be least wrong.
        availability = {
            gameweek: {
                player.id: player.availability
                for player in live_history.view_at(gameweek, known_through=2).pool
            }
            for gameweek in range(3, 8)
        }

        assert all(availability[gameweek][1] == 1.0 for gameweek in range(3, 8)), (
            f"a 90-minute starter decayed across the horizon: {availability}"
        )

    def test_the_default_is_the_gameweek_before(self, live_history: SeasonHistory) -> None:
        # A backtest replays a finished season, so standing at GW3 it really does hold GW2.
        assert (
            live_history.view_at(3).stats(1).minutes
            == live_history.view_at(3, known_through=2).stats(1).minutes
        )

    def test_a_cutoff_at_or_after_the_gameweek_is_rejected(
        self, live_history: SeasonHistory
    ) -> None:
        # Reading the gameweek you are forecasting is the definition of lookahead.
        with pytest.raises(ValueError, match="known_through must be before"):
            live_history.view_at(3, known_through=3)


class TestTheHorizonIsBuilt:
    """The other half: there have to be future gameweeks to take a view of."""

    @staticmethod
    def _load(monkeypatch: pytest.MonkeyPatch, horizon: int, *, last: int = 38):
        payload = bootstrap_payload()
        # `load_live_history` derives the season label from the first deadline, to key its cache.
        payload["events"] = [{"id": 1, "deadline_time": "2026-08-21T17:30:00Z", "finished": True}]
        monkeypatch.setattr(
            "fpl_optimiser.live._fetch_histories",
            lambda *args, **kwargs: {},
        )
        monkeypatch.setattr(
            "fpl_optimiser.live._fixtures_for",
            lambda gameweek: {1: [(2, True)], 2: [(1, False)]} if gameweek <= last else {},
        )
        return load_live_history(
            payload,
            GameRules.from_bootstrap(payload),
            gameweek=3,
            horizon=horizon,
        )

    def test_the_requested_horizon_is_actually_built(self, monkeypatch: pytest.MonkeyPatch) -> None:
        history = self._load(monkeypatch, horizon=5)

        assert history.gameweeks == (3, 4, 5, 6, 7)

    def test_a_horizon_of_one_is_the_old_behaviour(self, monkeypatch: pytest.MonkeyPatch) -> None:
        history = self._load(monkeypatch, horizon=1)

        assert history.gameweeks == (3,)

    def test_it_stops_at_the_end_of_the_season_rather_than_inventing_blanks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every club blanking is what the free-hit rule fires on, so emitting fixture-less weeks
        # past GW38 would burn a chip on the end of the season.
        history = self._load(monkeypatch, horizon=5, last=4)

        assert history.gameweeks == (3, 4)

    def test_a_horizon_below_one_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValueError, match="horizon must be at least 1"):
            self._load(monkeypatch, horizon=0)

    def test_every_built_gameweek_carries_its_own_fixture(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The whole point of the horizon: the views differ by fixture and nothing else.
        history = self._load(monkeypatch, horizon=3)

        for gameweek in (3, 4, 5):
            view = history.view_at(gameweek, known_through=2)
            assert all(view.fixtures[player.id] for player in view.pool), (
                f"GW{gameweek} has players with no fixture"
            )


class TestTheBacktestIsUnaffected:
    """`view_at` is shared with the backtest, so the default path must be an exact no-op.

    Checked here on a fixture and, when this landed, against the whole 2025-26 archive: 38
    gameweeks and 841 players, no field differing. End-to-end backtest totals cannot show this —
    CP-SAT runs eight workers and breaks ties differently run to run, so two runs of *identical*
    code differ by more than this change does.
    """

    def test_the_default_equals_the_explicit_previous_gameweek(
        self, live_history: SeasonHistory
    ) -> None:
        for gameweek in (3, 4, 5, 6, 7):
            default = live_history.view_at(gameweek)
            explicit = live_history.view_at(gameweek, known_through=gameweek - 1)

            for field in ("pool", "lagged", "fixtures", "teams", "days_rest", "suspended"):
                assert getattr(default, field) == getattr(explicit, field), (
                    f"GW{gameweek} differs in {field}"
                )
            assert vars(default.team_ratings) == vars(explicit.team_ratings)

    def test_a_ban_is_still_served_down_across_the_horizon(self, rules: GameRules) -> None:
        # Suspensions read the gameweek being planned, not the cutoff, so the unplayed weeks'
        # real fixtures serve the ban. Cutting them off would leave a one-match ban outstanding
        # for the whole horizon and bench a player who is available again.
        records = [played(1, 1), played(1, 2)]
        booked = records[-1]
        records[-1] = GameweekRecord(**{**vars(booked), "red_cards": 1})
        records.extend(scheduled(1, gw) for gw in (3, 4, 5))
        history = SeasonHistory(records, rules)

        banned_now = history.view_at(3, known_through=2).suspended
        two_weeks_later = history.view_at(5, known_through=2).suspended

        assert 1 in banned_now, "a red card in GW2 bans the player in GW3"
        assert 1 not in two_weeks_later, "by GW5 the ban has been served"
