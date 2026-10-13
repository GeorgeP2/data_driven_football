"""Tests for seeding the live minutes model from last season.

The bug these guard against is silent in both directions. If the carry-over is missing, every
player at GW1 has zero previous-season minutes and `forecast_minutes` gives the whole pool a
zero chance of starting — no error, just an arbitrary squad. If the name join breaks, the
result is identical, which is why the match rate is checked rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
from fpl_helpers import bootstrap_payload
from fpl_optimiser.backtest.carryover import carry_over_between_seasons, rekey_teams
from fpl_optimiser.backtest.history import Fixture, GameweekRecord
from fpl_optimiser.backtest.points_model import forecast_minutes
from fpl_optimiser.backtest.season import LaggedStats
from fpl_optimiser.cli_advise import build_parser, forecast_gameweek
from fpl_optimiser.live import (
    MODEL_FROM_GAMEWEEK,
    _fetch_histories,
    early_season_forecast,
    gameweek_is_finished,
    load_live_history,
    load_previous_season,
    normalise_name,
    previous_season_label,
    season_label,
)
from fpl_optimiser.rules import GameRules


def lagged(*, previous_minutes: int) -> LaggedStats:
    """A player with no completed gameweeks this season — i.e. everyone, at GW1."""
    return LaggedStats(
        player_id=1,
        points=(),
        minutes=(),
        starts=(),
        expected_goals=(),
        expected_assists=(),
        bonus=(),
        bps=(),
        saves=(),
        defensive_contribution=(),
        previous_season_points=0,
        previous_season_minutes=previous_minutes,
    )


def archive_record(
    player_id: int,
    name: str,
    *,
    minutes: int = 90,
    team_id: int = 1,
    team_name: str = "Arsenal",
    opponent_id: int = 2,
) -> GameweekRecord:
    return GameweekRecord(
        player_id=player_id,
        gameweek=1,
        name=name,
        element_type=3,
        team_id=team_id,
        team_name=team_name,
        price_tenths=50,
        total_points=5,
        minutes=minutes,
        started=1,
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
        opponents=(
            Fixture(
                opponent_team_id=opponent_id,
                was_home=True,
                goals_for=1,
                goals_against=0,
            ),
        ),
    )


ARSENAL_AND_VILLA = [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Aston Villa"}]


def live_payload(
    elements: list[dict[str, Any]],
    teams: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "events": [{"id": 1, "deadline_time": "2026-08-21T17:30:00Z"}],
        "elements": elements,
        "teams": ARSENAL_AND_VILLA if teams is None else teams,
    }


def element(player_id: int, first: str, second: str) -> dict[str, Any]:
    return {"id": player_id, "first_name": first, "second_name": second}


@dataclass
class FakePlayer:
    id: int


@dataclass
class FakeView:
    gameweek: int
    pool: list[FakePlayer]


class ExplodingModel:
    """Asserts the model is never even consulted before the handover gameweek."""

    def build(self, view: Any, history: Any):
        raise AssertionError("the model must not be built before model_from")


@dataclass
class ConstantModel:
    value: float

    def build(self, view: Any, history: Any):
        return lambda player: self.value


class TestTheColdStartItself:
    """The behaviour the carry-over exists to produce."""

    def test_without_carryover_a_proven_starter_is_forecast_never_to_start(
        self,
    ) -> None:
        # The live path before this change: SeasonHistory built with no previous season, so
        # previous_season_minutes is 0 for everyone.
        forecast = forecast_minutes(lagged(previous_minutes=0), gameweek=1)

        assert forecast.probability_starts == 0.0
        assert forecast.expected_minutes == 0.0

    def test_with_carryover_a_proven_starter_is_forecast_to_start(self) -> None:
        # 3200 minutes last season is a nailed-on starter.
        forecast = forecast_minutes(lagged(previous_minutes=3200), gameweek=1)

        assert forecast.probability_starts > 0.8
        assert forecast.expected_minutes > 60

    def test_a_fringe_player_is_still_rated_below_a_starter(self) -> None:
        # The carry-over must discriminate, not just switch everyone on.
        fringe = forecast_minutes(lagged(previous_minutes=400), gameweek=1)
        starter = forecast_minutes(lagged(previous_minutes=3200), gameweek=1)

        assert fringe.probability_starts < starter.probability_starts


class TestNameNormalisation:
    def test_accents_are_stripped_so_the_two_sources_agree(self) -> None:
        assert normalise_name("João Pedro") == normalise_name("Joao Pedro")

    def test_punctuation_and_case_are_removed(self) -> None:
        assert normalise_name("N'Golo  Kanté") == "n golo kante"

    def test_distinct_players_do_not_collide(self) -> None:
        assert normalise_name("Gabriel Jesus") != normalise_name("Gabriel Martinelli")


class TestEarlySeasonHandover:
    """GW1 falls back on ep_next because the model measurably has no signal there."""

    def test_the_fallback_is_ep_next_discounted_by_availability(self, rules) -> None:
        payload = bootstrap_payload()

        early = early_season_forecast(payload, rules)

        # Raya: 8.8 ep_next, fully available.
        assert early[10] == pytest.approx(8.8)
        # Salah: 6.4 ep_next at a 75% chance of playing.
        assert early[11] == pytest.approx(6.4 * 0.75)
        # Injured: worth nothing regardless of ep_next.
        assert early[12] == 0.0

    def test_the_handover_defaults_to_gameweek_two(self) -> None:
        # GW1 is the one gameweek the evidence is unambiguous about.
        assert MODEL_FROM_GAMEWEEK == 2
        assert build_parser().parse_args(["--team-id", "1"]).model_from == 2

    def test_gw1_uses_ep_next_not_the_model(self) -> None:
        view = FakeView(gameweek=1, pool=[FakePlayer(10), FakePlayer(11)])

        points = forecast_gameweek(view, None, ExplodingModel(), {10: 8.8, 11: 4.8}, model_from=2)

        assert points == {10: 8.8, 11: 4.8}

    def test_gw2_uses_the_model_not_ep_next(self) -> None:
        view = FakeView(gameweek=2, pool=[FakePlayer(10), FakePlayer(11)])

        points = forecast_gameweek(view, None, ConstantModel(3.5), {10: 8.8, 11: 4.8}, model_from=2)

        assert points == {10: 3.5, 11: 3.5}

    def test_a_non_default_handover_moves_the_boundary(self) -> None:
        # --model-from 4: GW3 still falls back, GW4 does not.
        early = {10: 8.8}
        pool = [FakePlayer(10)]

        third = forecast_gameweek(FakeView(3, pool), None, ExplodingModel(), early, model_from=4)
        fourth = forecast_gameweek(FakeView(4, pool), None, ConstantModel(3.5), early, model_from=4)

        assert third == {10: 8.8}
        assert fourth == {10: 3.5}

    def test_model_from_one_uses_the_model_even_at_gw1(self) -> None:
        # The documented escape hatch, so the old behaviour stays reachable.
        points = forecast_gameweek(
            FakeView(1, [FakePlayer(10)]),
            None,
            ConstantModel(2.0),
            {10: 8.8},
            model_from=1,
        )

        assert points == {10: 2.0}

    def test_a_player_missing_from_ep_next_is_worth_nothing_not_a_crash(self) -> None:
        points = forecast_gameweek(
            FakeView(1, [FakePlayer(99)]), None, ExplodingModel(), {}, model_from=2
        )

        assert points == {99: 0.0}

    def test_every_player_in_the_pool_gets_a_forecast(self, rules) -> None:
        payload = bootstrap_payload()

        early = early_season_forecast(payload, rules)

        assert set(early) == {int(e["id"]) for e in payload["elements"]}


class TestSeasonCacheIsolation:
    """FPL renumbers element ids each summer, so a cross-season cache hit is silent corruption."""

    def test_the_history_cache_is_keyed_by_season(self, tmp_path, monkeypatch) -> None:
        # Last season's cache, under the old season-blind filename this replaced.
        (tmp_path / "element_summaries.json").write_text('{"1": [{"round": 1}]}')
        # At GW1 of a new season the API returns nothing for anyone.
        monkeypatch.setattr("fpl_optimiser.live._get", lambda url: {"history": []})

        histories = _fetch_histories(
            [1], tmp_path, refresh=False, season="2026-27", completed_through=0
        )

        # Must refetch rather than read last season's file just because id 1 exists in both.
        assert histories == {1: []}

    def test_two_seasons_do_not_share_a_cache_file(self, tmp_path) -> None:
        (tmp_path / "element_summaries_2025-26_through_gw38.json").write_text(
            '{"1": [{"round": 5}]}'
        )
        (tmp_path / "element_summaries_2026-27_through_gw38.json").write_text(
            '{"1": [{"round": 9}]}'
        )

        old = _fetch_histories([1], tmp_path, refresh=False, season="2025-26", completed_through=38)
        new = _fetch_histories([1], tmp_path, refresh=False, season="2026-27", completed_through=38)

        assert old[1][0]["round"] == 5
        assert new[1][0]["round"] == 9

    def test_a_newly_completed_gameweek_is_a_cache_miss(self, tmp_path, monkeypatch) -> None:
        # The season-keyed cache this replaced: written on the Tuesday after GW2, it served
        # every run until someone remembered --refresh, so GW4 was planned never having seen GW3.
        (tmp_path / "element_summaries_2026-27_through_gw02.json").write_text(
            '{"1": [{"round": 1}, {"round": 2}]}'
        )
        monkeypatch.setattr(
            "fpl_optimiser.live._get",
            lambda url: {"history": [{"round": 1}, {"round": 2}, {"round": 3}]},
        )

        through_two = _fetch_histories(
            [1], tmp_path, refresh=False, season="2026-27", completed_through=2
        )
        through_three = _fetch_histories(
            [1], tmp_path, refresh=False, season="2026-27", completed_through=3
        )

        assert [row["round"] for row in through_two[1]] == [1, 2]
        assert [row["round"] for row in through_three[1]] == [1, 2, 3]
        assert (tmp_path / "element_summaries_2026-27_through_gw03.json").is_file()


class TestUnfinishedGameweekIsNotCached:
    """A cache named `through_gwNN` must not be written while GW NN is still being played.

    On 2026-10-12, the Monday of GW6 with COV-NEW still to play, a GW7 run wrote
    `through_gw06` from provisional rows. The key said complete, so every later GW7 run would
    have read it and never seen Monday night's fixture or the confirmed bonus.
    """

    @staticmethod
    def _payload(*, previous_finished: bool) -> dict[str, Any]:
        payload = bootstrap_payload()
        payload["events"] = [
            {"id": 1, "deadline_time": "2026-08-21T17:30:00Z", "finished": True},
            {"id": 2, "deadline_time": "2026-08-28T17:30:00Z", "finished": previous_finished},
        ]
        return payload

    def _load(self, tmp_path, monkeypatch, *, previous_finished: bool) -> None:
        monkeypatch.setattr("fpl_optimiser.live._get", lambda url: {"history": []})
        monkeypatch.setattr(
            "fpl_optimiser.live._fixtures_for",
            lambda gameweek: {1: [(2, True)], 2: [(1, False)]},
        )
        payload = self._payload(previous_finished=previous_finished)
        load_live_history(
            payload, GameRules.from_bootstrap(payload), gameweek=3, cache_dir=tmp_path
        )

    def test_a_run_during_the_previous_gameweek_writes_no_cache(
        self, tmp_path, monkeypatch
    ) -> None:
        self._load(tmp_path, monkeypatch, previous_finished=False)

        assert list(tmp_path.glob("element_summaries_*")) == []

    def test_a_run_after_the_previous_gameweek_finishes_is_cached(
        self, tmp_path, monkeypatch
    ) -> None:
        self._load(tmp_path, monkeypatch, previous_finished=True)

        assert (tmp_path / "element_summaries_2026-27_through_gw02.json").is_file()

    def test_an_uncacheable_fetch_ignores_a_provisional_file_already_on_disk(
        self, tmp_path, monkeypatch
    ) -> None:
        # The file the bug left behind: GW2 rows from before the gameweek finished.
        stale = tmp_path / "element_summaries_2026-27_through_gw02.json"
        stale.write_text('{"1": [{"round": 2, "total_points": 1}]}')
        monkeypatch.setattr(
            "fpl_optimiser.live._get",
            lambda url: {"history": [{"round": 2, "total_points": 3}]},
        )

        histories = _fetch_histories(
            [1],
            tmp_path,
            refresh=False,
            season="2026-27",
            completed_through=2,
            cacheable=False,
        )

        assert histories[1][0]["total_points"] == 3
        assert stale.read_text() == '{"1": [{"round": 2, "total_points": 1}]}'


class TestGameweekIsFinished:
    def test_true_when_fpl_marks_it_finished(self) -> None:
        payload = {"events": [{"id": 6, "finished": True}]}

        assert gameweek_is_finished(payload, 6)

    def test_false_while_it_is_still_being_played(self) -> None:
        payload = {"events": [{"id": 6, "finished": False}]}

        assert not gameweek_is_finished(payload, 6)

    def test_a_gameweek_missing_from_events_is_unfinished(self) -> None:
        assert not gameweek_is_finished({"events": [{"id": 5, "finished": True}]}, 6)

    def test_gameweek_zero_has_nothing_to_finish(self) -> None:
        assert gameweek_is_finished({"events": []}, 0)


class TestSeasonLabels:
    def test_derives_the_live_season_from_the_first_deadline(self) -> None:
        assert season_label(live_payload([])) == "2026-27"

    def test_derives_the_prior_season_from_the_first_deadline(self) -> None:
        payload = live_payload([])

        assert previous_season_label(payload) == "2025-26"

    def test_uses_the_earliest_deadline_not_the_first_listed(self) -> None:
        payload = {
            "events": [
                {"id": 2, "deadline_time": "2027-01-02T12:00:00Z"},
                {"id": 1, "deadline_time": "2026-08-21T17:30:00Z"},
            ],
            "elements": [],
        }

        assert previous_season_label(payload) == "2025-26"

    def test_a_payload_with_no_events_fails_clearly(self) -> None:
        with pytest.raises(ValueError, match="no event deadlines"):
            previous_season_label({"events": [], "elements": []})


class TestTheJoin:
    def test_records_are_rekeyed_to_this_seasons_element_ids(self, monkeypatch) -> None:
        # Last season Raya was element 7; this season he is element 1.
        monkeypatch.setattr(
            "fpl_optimiser.live.load_season",
            lambda *a, **k: [archive_record(7, "David Raya Martin", minutes=90)],
        )
        payload = live_payload([element(1, "David", "Raya Martín")])

        carried = load_previous_season(payload, min_match_rate=0.5)

        assert [r.player_id for r in carried] == [1]
        assert carried[0].minutes == 90

    def test_unmatched_players_are_dropped_not_guessed(self, monkeypatch) -> None:
        # A promoted-club player with no Premier League history must not inherit anyone else's.
        monkeypatch.setattr(
            "fpl_optimiser.live.load_season",
            lambda *a, **k: [archive_record(7, "David Raya Martin")],
        )
        payload = live_payload(
            [
                element(1, "David", "Raya Martín"),
                element(2, "Jack", "Andrews"),  # newly promoted, unseen
            ]
        )

        carried = load_previous_season(payload, min_match_rate=0.5)

        assert {r.player_id for r in carried} == {1}

    def test_a_name_shared_by_two_players_is_skipped(self, monkeypatch) -> None:
        # Two archive players normalise to the same name: we cannot tell which is which, and a
        # wrong match is worse than no match.
        monkeypatch.setattr(
            "fpl_optimiser.live.load_season",
            lambda *a, **k: [
                archive_record(7, "Danny Ward"),
                archive_record(8, "Danny Ward"),
            ],
        )
        payload = live_payload([element(1, "Danny", "Ward")])

        with pytest.raises(ValueError, match="below the"):
            load_previous_season(payload, min_match_rate=0.5)

    def test_all_gameweeks_for_a_matched_player_are_carried(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "fpl_optimiser.live.load_season",
            lambda *a, **k: [
                archive_record(7, "David Raya Martin", minutes=90),
                archive_record(7, "David Raya Martin", minutes=45),
            ],
        )
        payload = live_payload([element(1, "David", "Raya Martín")])

        carried = load_previous_season(payload, min_match_rate=0.5)

        assert sum(r.minutes for r in carried) == 135


class TestTheMatchRateCheck:
    def test_a_broken_join_raises_rather_than_carrying_zeroes(self, monkeypatch) -> None:
        # What a renamed archive column looks like: nothing matches.
        monkeypatch.setattr(
            "fpl_optimiser.live.load_season",
            lambda *a, **k: [archive_record(7, "")],
        )
        payload = live_payload([element(i, "First", f"Player{chr(96 + i)}") for i in range(1, 11)])

        with pytest.raises(ValueError, match="cold start"):
            load_previous_season(payload, min_match_rate=0.5)

    def test_a_healthy_join_passes_the_floor(self, monkeypatch) -> None:
        # 8 of 10 matched, comfortably inside the measured 75-85% band.
        monkeypatch.setattr(
            "fpl_optimiser.live.load_season",
            lambda *a, **k: [
                archive_record(100 + i, f"First Player{chr(96 + i)}") for i in range(1, 9)
            ],
        )
        payload = live_payload([element(i, "First", f"Player{chr(96 + i)}") for i in range(1, 11)])

        carried = load_previous_season(payload, min_match_rate=0.5)

        assert len({r.player_id for r in carried}) == 8

    def test_the_floor_is_configurable_on_its_non_default_value(self, monkeypatch) -> None:
        # 8 of 10 is fine at the default floor but not at a stricter one.
        monkeypatch.setattr(
            "fpl_optimiser.live.load_season",
            lambda *a, **k: [
                archive_record(100 + i, f"First Player{chr(96 + i)}") for i in range(1, 9)
            ],
        )
        payload = live_payload([element(i, "First", f"Player{chr(96 + i)}") for i in range(1, 11)])

        with pytest.raises(ValueError, match="below the 95% floor"):
            load_previous_season(payload, min_match_rate=0.95)

    @pytest.mark.parametrize("bad_rate", [-0.1, 1.5])
    def test_an_out_of_range_floor_fails_clearly(self, bad_rate: float) -> None:
        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            load_previous_season(live_payload([]), min_match_rate=bad_rate)

    def test_the_failure_names_the_numbers(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "fpl_optimiser.live.load_season", lambda *a, **k: [archive_record(7, "")]
        )
        payload = live_payload([element(1, "First", "Player")])

        with pytest.raises(ValueError) as caught:
            load_previous_season(payload, min_match_rate=0.5)

        assert "0 of 1" in str(caught.value)

    def test_the_match_rate_is_logged_for_a_healthy_join(self, monkeypatch, caplog) -> None:
        # The rate drifts season to season with promotions, so it has to be visible even when
        # it passes — a silent 60% is how you find out too late.
        monkeypatch.setattr(
            "fpl_optimiser.live.load_season",
            lambda *a, **k: [archive_record(7, "David Raya Martin")],
        )
        payload = live_payload([element(1, "David", "Raya Martín")])

        with caplog.at_level("INFO", logger="fpl_optimiser.live"):
            load_previous_season(payload, min_match_rate=0.5)

        assert "matched 1 of 1" in caplog.text


class TestTeamIdsAreRekeyed:
    """Team ids are reissued alphabetically each summer, and the strength prior is keyed by them.

    The bug this guards against: in 2026-27 the carry-over kept 2025-26's ids, so Ipswich got
    Liverpool's record, Man City got Newcastle's and Bournemouth got Burnley's. No error — the
    advisor just rated every fixture against the wrong defence.
    """

    # Last season: Bournemouth 4, Brentford 5, Burnley 3. This season Burnley went down and
    # Brighton's neighbour moved up a slot: Bournemouth 3, Brentford 4.
    LAST_SEASON: ClassVar[list[GameweekRecord]] = [
        archive_record(7, "Antoine Semenyo", team_id=4, team_name="Bournemouth", opponent_id=5),
        archive_record(8, "Bryan Mbeumo", team_id=5, team_name="Brentford", opponent_id=3),
        archive_record(9, "Josh Cullen", team_id=3, team_name="Burnley", opponent_id=5),
    ]
    THIS_SEASON: ClassVar[list[dict[str, Any]]] = [
        {"id": 3, "name": "Bournemouth"},
        {"id": 4, "name": "Brentford"},
    ]

    def test_a_clubs_records_follow_its_name_not_its_old_number(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "fpl_optimiser.live.load_season", lambda *a, **k: list(self.LAST_SEASON)
        )
        payload = live_payload(
            [element(1, "Antoine", "Semenyo"), element(2, "Bryan", "Mbeumo")],
            teams=self.THIS_SEASON,
        )

        carried = load_previous_season(payload, min_match_rate=0.5)

        assert {r.player_id: r.team_id for r in carried} == {1: 3, 2: 4}

    def test_opponents_are_rekeyed_by_the_same_map(self) -> None:
        teams = {t["name"]: t["id"] for t in self.THIS_SEASON}

        carried = rekey_teams(self.LAST_SEASON, teams)

        # Bournemouth's opponent was Brentford (5 last season, 4 now).
        bournemouth = next(r for r in carried if r.team_name == "Bournemouth")
        assert bournemouth.opponents[0].opponent_team_id == 4

    def test_a_relegated_club_is_dropped_and_its_fixtures_keep_a_distinct_id(
        self,
    ) -> None:
        teams = {t["name"]: t["id"] for t in self.THIS_SEASON}

        carried = rekey_teams(self.LAST_SEASON, teams)

        # Burnley's own records go; Brentford's match against them keeps a non-colliding id.
        assert {r.team_name for r in carried} == {"Bournemouth", "Brentford"}
        brentford = next(r for r in carried if r.team_name == "Brentford")
        assert brentford.opponents[0].opponent_team_id < 0
        assert brentford.opponents[0].opponent_team_id not in teams.values()

    def test_more_than_three_missing_clubs_is_a_broken_join_not_relegation(
        self,
    ) -> None:
        # Twenty clubs, only ten of which match: the names have changed shape.
        records = [
            archive_record(i, f"Player {chr(96 + i)}", team_id=i, team_name=f"Club {i}")
            for i in range(1, 21)
        ]
        teams = {f"Club {i}": i for i in range(1, 11)}

        with pytest.raises(ValueError, match="10 of 20 clubs"):
            rekey_teams(records, teams)

    def test_seventeen_of_twenty_is_a_normal_summer(self) -> None:
        records = [
            archive_record(i, f"Player {chr(96 + i)}", team_id=i, team_name=f"Club {i}")
            for i in range(1, 21)
        ]
        teams = {f"Club {i}": i for i in range(1, 18)}

        assert len(rekey_teams(records, teams)) == 17

    def test_the_floor_can_be_set_explicitly(self) -> None:
        teams = {t["name"]: t["id"] for t in self.THIS_SEASON}

        # Two of three match: fine by default (3 - 3 -> floor 1), not at a floor of 3.
        assert len(rekey_teams(self.LAST_SEASON, teams)) == 2
        with pytest.raises(ValueError, match="below the floor of 3"):
            rekey_teams(self.LAST_SEASON, teams, min_matches=3)


class TestBacktestCarryOver:
    """The backtest joins two archive seasons, and archive ids are no more stable than live ones:
    of 784 element ids present in both 2024-25 and 2025-26, one belonged to the same player.
    """

    def test_players_and_teams_are_both_rekeyed_by_name(self) -> None:
        previous = [
            archive_record(351, "Erling Haaland", team_id=13, team_name="Man City", opponent_id=12),
            # Left the league, but his club's name is what tells us id 12 was Liverpool.
            archive_record(200, "Darwin Nunez", team_id=12, team_name="Liverpool"),
        ]
        current = [
            archive_record(430, "Erling Haaland", team_id=14, team_name="Man City"),
            archive_record(431, "Alisson Becker", team_id=13, team_name="Liverpool"),
        ]

        carried = carry_over_between_seasons(previous, current)

        assert [(r.player_id, r.team_id) for r in carried] == [(430, 14)]
        # His opponent, Liverpool, was 12 and is now 13.
        assert carried[0].opponents[0].opponent_team_id == 13

    def test_a_player_who_left_the_league_is_dropped_not_reassigned(self) -> None:
        previous = [archive_record(351, "Kevin De Bruyne", team_id=13, team_name="Man City")]
        current = [archive_record(351, "Rayan Cherki", team_id=14, team_name="Man City")]

        assert carry_over_between_seasons(previous, current) == []
