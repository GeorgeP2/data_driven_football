"""Tests for payload parsing, the availability discount, and rule extraction."""

from __future__ import annotations

import pytest
from fpl_helpers import make_player
from fpl_optimiser.expected_points import EpNextProvider, StaticProvider
from fpl_optimiser.players import availability_from, parse_players
from fpl_optimiser.rules import GameRules, PositionRule


class TestGameRules:
    def test_reads_the_rules_from_the_payload(self, payload) -> None:
        rules = GameRules.from_bootstrap(payload)

        assert rules.squad_size == 15
        assert rules.starting_size == 11
        assert rules.club_limit == 3
        assert rules.budget_tenths == 1000
        assert rules.position(2).short_name == "DEF"
        assert rules.position(2).squad_select == 5
        assert rules.position(2).min_play == 3
        assert rules.position(2).max_play == 5

    def test_missing_game_settings_raises(self, payload) -> None:
        del payload["game_settings"]

        with pytest.raises(ValueError, match="game_settings"):
            GameRules.from_bootstrap(payload)

    def test_position_quotas_that_do_not_sum_to_the_squad_size_raise(self, payload) -> None:
        payload["element_types"][1]["squad_select"] = 4  # 2+4+5+3 = 14, not 15

        with pytest.raises(ValueError, match="sum to 14"):
            GameRules.from_bootstrap(payload)

    def test_unknown_element_type_raises_naming_the_options(self, rules) -> None:
        with pytest.raises(ValueError, match="expected one of: 1, 2, 3, 4"):
            rules.position(9)

    def test_impossible_formation_raises(self) -> None:
        # Eleven starters, but the positions together allow at most nine.
        with pytest.raises(ValueError, match="No legal formation"):
            GameRules(
                squad_size=15,
                starting_size=11,
                club_limit=3,
                budget_tenths=1000,
                positions=(
                    PositionRule(1, "GKP", 2, 1, 1),
                    PositionRule(2, "DEF", 5, 3, 3),
                    PositionRule(3, "MID", 5, 2, 3),
                    PositionRule(4, "FWD", 3, 1, 2),
                ),
            ).validate()


class TestAvailability:
    @pytest.mark.parametrize(
        ("status", "chance", "expected"),
        [
            ("a", None, 1.0),  # fit, no percentage published
            ("a", 100, 1.0),
            ("d", 75, 0.75),  # doubtful, three-quarters likely
            ("d", 25, 0.25),
            ("d", None, 1.0),  # doubtful but no number: assume they play
            ("i", None, 0.0),  # injured
            ("s", None, 0.0),  # suspended
            ("u", None, 0.0),  # unavailable
            ("i", 50, 0.5),  # an explicit percentage overrides the status flag
        ],
    )
    def test_discount(self, status: str, chance: int | None, expected: float) -> None:
        assert availability_from(status, chance) == pytest.approx(expected)


class TestParsePlayers:
    def test_parses_the_payload(self, payload) -> None:
        rules = GameRules.from_bootstrap(payload)

        players = {p.id: p for p in parse_players(payload, rules)}

        assert players[10].name == "Raya"
        assert players[10].position == "GKP"
        assert players[10].team_name == "ARS"
        assert players[10].cost_tenths == 62
        assert players[10].cost_millions == 6.2
        assert players[10].ep_next == 8.8  # parsed from the API's string "8.8"
        assert players[11].availability == 0.75
        assert players[12].availability == 0.0

    def test_player_on_an_unknown_team_raises(self, payload) -> None:
        payload["elements"][0]["team"] = 99
        rules = GameRules.from_bootstrap(payload)

        with pytest.raises(ValueError, match="team 99"):
            parse_players(payload, rules)

    def test_null_ep_next_is_zero_not_an_error(self, payload) -> None:
        # Live data: players who have never featured carry `"ep_next": null`. That is "no
        # forecast", worth zero points — not a corrupt payload.
        payload["elements"][0]["ep_next"] = None
        rules = GameRules.from_bootstrap(payload)

        players = {p.id: p for p in parse_players(payload, rules)}

        assert players[10].ep_next == 0.0

    def test_unparseable_ep_next_raises(self, payload) -> None:
        payload["elements"][0]["ep_next"] = "not-a-number"
        rules = GameRules.from_bootstrap(payload)

        with pytest.raises(ValueError, match="ep_next"):
            parse_players(payload, rules)

    def test_empty_elements_raises(self, payload) -> None:
        rules = GameRules.from_bootstrap(payload)
        payload["elements"] = []

        with pytest.raises(ValueError, match="elements"):
            parse_players(payload, rules)


class TestExpectedPointsProviders:
    def test_ep_next_is_discounted_by_availability(self) -> None:
        doubtful = make_player(1, 3, ep_next=8.0, availability=0.25)

        assert EpNextProvider()(doubtful) == pytest.approx(2.0)

    def test_ignoring_availability_is_the_undiscounted_forecast(self) -> None:
        doubtful = make_player(1, 3, ep_next=8.0, availability=0.25)

        assert EpNextProvider(apply_availability=False)(doubtful) == pytest.approx(8.0)

    def test_injured_players_are_worth_nothing(self) -> None:
        injured = make_player(1, 3, ep_next=9.0, availability=0.0, status="i")

        assert EpNextProvider()(injured) == pytest.approx(0.0)

    def test_negative_forecasts_are_floored_at_zero(self) -> None:
        # FPL can publish a negative ep_next; a player is never worse than not playing.
        assert EpNextProvider()(make_player(1, 2, ep_next=-1.5)) == 0.0

    def test_static_provider_defaults_unknown_players(self) -> None:
        provider = StaticProvider(points_by_player_id={1: 5.0}, default=0.5)

        assert provider(make_player(1, 3)) == pytest.approx(5.0)
        assert provider(make_player(2, 3)) == pytest.approx(0.5)
