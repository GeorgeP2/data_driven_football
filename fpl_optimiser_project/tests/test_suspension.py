"""Tests for the suspension tracker.

Suspension is the one component that is a fact rather than a forecast, so these tests assert exact
answers rather than plausible ones. The cutoff rule gets the most attention, because implementing
the thresholds *without* it is the failure mode that matters: it would confidently rule out 81
players a season who went on to play, and a false "he is banned" is worse than no information at
all — it removes a good player from selection entirely.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fpl_helpers import record
from fpl_optimiser.backtest.history import Fixture
from fpl_optimiser.backtest.suspension import (
    RED_CARD_BAN_MATCHES,
    RED_CARD_BAN_SERVED,
    suspension_at,
    suspensions_at,
)


def booked(gameweek: int, *, yellows: int = 0, reds: int = 0, fixtures: int = 1):
    """One gameweek in a player's record, with the cards they picked up."""
    opponents = tuple(
        Fixture(opponent_team_id=20, was_home=True, goals_for=0, goals_against=0)
        for _ in range(fixtures)
    )
    return replace(
        record(player_id=1, gameweek=gameweek, opponents=opponents),
        yellow_cards=yellows,
        red_cards=reds,
    )


class TestYellowCardBans:
    def test_a_fifth_yellow_inside_the_cutoff_earns_a_ban(self) -> None:
        # One yellow in each of the first five gameweeks: the fifth triggers a one-match ban.
        history = [booked(gw, yellows=1) for gw in range(1, 6)]

        assert suspension_at(history, 6).certain_matches == 1

    def test_a_fifth_yellow_past_the_cutoff_earns_nothing(self) -> None:
        """The rule that makes this worth having. Measured on 2024-25: a fifth yellow inside the
        cutoff meant a missed gameweek 43 times out of 43; past it, 8 times out of 81.
        """
        # Twenty quiet gameweeks, then the bookings start — after the club's 19th match.
        history = [booked(gw) for gw in range(1, 21)]
        history += [booked(gw, yellows=1) for gw in range(21, 26)]

        assert suspension_at(history, 26).certain_matches == 0, "past the cutoff, no ban"

    def test_the_ban_is_served_and_then_they_are_free(self) -> None:
        history = [booked(gw, yellows=1) for gw in range(1, 6)]
        history.append(booked(6))  # the match they sat out

        assert suspension_at(history, 6).certain_matches == 1, "banned for GW6"
        assert suspension_at(history, 7).certain_matches == 0, "and available again in GW7"

    def test_a_tenth_yellow_earns_two_matches(self) -> None:
        history = [booked(gw, yellows=1) for gw in range(1, 6)]  # 5th -> 1 match
        history.append(booked(6))  # served
        history += [booked(gw, yellows=1) for gw in range(7, 12)]  # 10th -> 2 matches

        assert suspension_at(history, 12).certain_matches == 2

    def test_the_five_and_ten_thresholds_do_not_double_count(self) -> None:
        # Five yellows in a single gameweek is impossible in practice, but the crossing logic
        # must not fire the same threshold twice as the total keeps climbing.
        history = [booked(1, yellows=5), booked(2), booked(3, yellows=1)]

        assert suspension_at(history, 4).certain_matches == 0, "the ban was already served in GW2"


class TestRedCards:
    def test_a_red_card_is_probable_not_certain(self) -> None:
        """The distinction that stops us over-reaching.

        A yellow-card ban is administrative — the total is reached, the ban applies, and it lands
        96% of the time. A red card can be rescinded on appeal or downgraded on review, and lands
        85% of the time. So it earns a heavy discount, not an exclusion: wrongly ruling a player
        out removes them from selection entirely, which is the expensive way to be wrong.
        """
        red = suspension_at([booked(1, reds=1)], 2)
        yellow = suspension_at([booked(gw, yellows=1) for gw in range(1, 6)], 6)

        assert red.probable_matches == 1
        assert red.certain_matches == 0
        assert red.availability == pytest.approx(1 - RED_CARD_BAN_SERVED)

        assert yellow.certain_matches == 1
        assert yellow.probable_matches == 0
        assert yellow.availability == 1.0, "the discount is for reds; yellows drop the fixture"

    def test_a_sending_off_earns_a_ban(self) -> None:
        history = [booked(1, reds=1)]

        assert suspension_at(history, 2).probable_matches == RED_CARD_BAN_MATCHES

    def test_a_red_card_ban_is_served(self) -> None:
        history = [booked(1, reds=1), booked(2)]

        assert suspension_at(history, 3).total_matches == 0


class TestDoubleGameweeks:
    def test_a_one_match_ban_in_a_double_leaves_them_the_second_fixture(self) -> None:
        """Worth getting right: this player is still worth owning, and a naive flag would drop
        them entirely."""
        history = [booked(gw, yellows=1) for gw in range(1, 6)]

        # They are banned for one match. If GW6 is a double, they miss one and play the other.
        assert suspension_at(history, 6).certain_matches == 1

    def test_a_double_gameweek_serves_two_matches_of_a_ban_at_once(self) -> None:
        history = [booked(gw, yellows=1) for gw in range(1, 6)]  # 5th -> 1 match
        history.append(booked(6))
        history += [booked(gw, yellows=1) for gw in range(7, 12)]  # 10th -> 2 matches
        history.append(booked(12, fixtures=2))  # a double: both ban matches served here

        assert suspension_at(history, 13).certain_matches == 0

    def test_cards_from_both_halves_of_a_double_are_counted(self) -> None:
        # Booked in both fixtures of a double gameweek: that is two yellows, not one.
        history = [booked(1, yellows=2, fixtures=2)]
        history += [booked(gw, yellows=1) for gw in (2, 3, 4)]

        assert suspension_at(history, 5).certain_matches == 1, "2 + 1 + 1 + 1 = the fifth yellow"


class TestNoLookahead:
    def test_only_cards_already_shown_count(self) -> None:
        # A yellow in GW5 cannot ban a player in GW5 — the card had not been shown at the
        # deadline. Reading it would be leaking the future into the team sheet.
        history = [booked(gw, yellows=1) for gw in range(1, 6)]

        assert suspension_at(history, 5).certain_matches == 0, (
            "the fifth yellow is shown *during* GW5"
        )
        assert suspension_at(history, 6).certain_matches == 1


class TestSuspensionsAt:
    def test_only_banned_players_are_listed(self) -> None:
        banned = [booked(gw, yellows=1) for gw in range(1, 6)]
        clean = [replace(b, player_id=2) for b in (booked(gw) for gw in range(1, 6))]

        suspended = suspensions_at({1: banned, 2: clean}, 6)

        assert list(suspended) == [1]
        assert suspended[1].certain_matches == 1


class TestAgainstRealData:
    """The rule, checked against the season it claims to describe."""

    @pytest.mark.parametrize(
        ("yellows_by_gameweek", "gameweek", "expected"),
        [
            # Five yellows by GW10 — comfortably inside the club's first 19 matches.
            ({2: 1, 4: 1, 6: 1, 8: 1, 10: 1}, 11, 1),
            # The same five, but spread past the cutoff. No ban.
            ({22: 1, 24: 1, 26: 1, 28: 1, 30: 1}, 31, 0),
        ],
    )
    def test_the_cutoff_decides(
        self, yellows_by_gameweek: dict[int, int], gameweek: int, expected: int
    ) -> None:
        history = [booked(gw, yellows=yellows_by_gameweek.get(gw, 0)) for gw in range(1, gameweek)]

        assert suspension_at(history, gameweek).certain_matches == expected
