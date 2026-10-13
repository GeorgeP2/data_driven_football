"""The advice text: who is shown replacing whom, and why.

`format_advice` used to zip two independently id-sorted tuples, so the OUT/IN pairing it printed
was an accident of player id. Live at GW5 2026-27 it showed a forward being replaced by a
midfielder and a midfielder by a forward — impossible under fixed squad quotas, and the only
part of the output a reader would take as reasoning.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from fpl_optimiser.cli_advise import horizon_value, pair_transfers
from fpl_optimiser.players import Player

GKP, DEF, MID, FWD = 1, 2, 3, 4


def make(player_id: int, element_type: int, name: str, cost_tenths: int) -> Player:
    return Player(
        id=player_id,
        name=name,
        element_type=element_type,
        position={1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[element_type],
        team_id=1,
        team_name="TST",
        cost_tenths=cost_tenths,
        status="a",
        availability=1.0,
        ep_next=0.0,
        minutes=0,
        total_points=0,
    )


class TestThePairing:
    def test_the_real_gw5_case_pairs_within_position(self) -> None:
        # The exact ids from the live run: sorted by id, OUT is [Furo, Elanga] and IN is
        # [Groß, Emersonn], so zip paired FWD->MID and MID->FWD. Both are wrong.
        furo = make(107, FWD, "Furo", 45)
        gross = make(124, MID, "Groß", 56)
        emersonn = make(316, FWD, "Emersonn", 55)
        elanga = make(454, MID, "Elanga", 62)

        paired = pair_transfers([furo, elanga], [gross, emersonn])

        assert {(out.name, into.name) for out, into in paired} == {
            ("Elanga", "Groß"),
            ("Furo", "Emersonn"),
        }

    def test_every_pair_shares_a_position(self) -> None:
        # The invariant, stated as itself: squad quotas are fixed per position, so a swap that
        # crosses one cannot happen however the sets are ordered.
        out = [make(1, DEF, "OutDef", 50), make(2, FWD, "OutFwd", 60)]
        into = [make(90, FWD, "InFwd", 65), make(91, DEF, "InDef", 45)]

        paired = pair_transfers(out, into)

        assert all(a.element_type == b.element_type for a, b in paired)

    def test_two_moves_in_one_position_pair_most_expensive_first(self) -> None:
        cheap_out, dear_out = make(1, MID, "CheapOut", 45), make(2, MID, "DearOut", 90)
        cheap_in, dear_in = make(90, MID, "CheapIn", 50), make(91, MID, "DearIn", 100)

        paired = pair_transfers([cheap_out, dear_out], [dear_in, cheap_in])

        assert [(a.name, b.name) for a, b in paired] == [
            ("DearOut", "DearIn"),
            ("CheapOut", "CheapIn"),
        ]

    def test_a_single_swap_is_unchanged(self) -> None:
        out, into = make(500, GKP, "OldKeeper", 45), make(3, GKP, "NewKeeper", 50)

        assert pair_transfers([out], [into]) == [(out, into)]

    def test_no_transfers_pairs_nothing(self) -> None:
        assert pair_transfers([], []) == []


class TestTheHorizonValue:
    POINTS: ClassVar[dict[int, dict[int, float]]] = {
        5: {1: 4.0, 2: 0.0},
        6: {1: 4.0, 2: 0.0},
        7: {1: 4.0, 2: 0.0},
    }

    def test_later_gameweeks_are_discounted(self) -> None:
        # 4 + 4*0.85 + 4*0.7225, the planner's own decay.
        value = horizon_value(self.POINTS, discount=0.85)

        assert value[1] == pytest.approx(4.0 + 3.4 + 2.89)

    def test_the_discount_is_applied_in_gameweek_order_not_dict_order(self) -> None:
        # A dict built back to front must not invert the decay.
        reversed_order = {7: {1: 1.0}, 5: {1: 1.0}, 6: {1: 1.0}}

        assert horizon_value(reversed_order, discount=0.5)[1] == pytest.approx(1.75)

    def test_a_non_default_discount_is_honoured(self) -> None:
        assert horizon_value(self.POINTS, discount=0.5)[1] == pytest.approx(7.0)

    def test_an_undiscounted_horizon_is_a_plain_sum(self) -> None:
        assert horizon_value(self.POINTS, discount=1.0)[1] == pytest.approx(12.0)

    def test_a_player_absent_from_a_gameweek_is_not_invented(self) -> None:
        # Blanks are real: a missing player contributes nothing that week, and must not appear
        # with a plausible-looking default.
        value = horizon_value({5: {1: 4.0}, 6: {}}, discount=0.85)

        assert value[1] == pytest.approx(4.0)

    def test_a_zero_forecast_player_scores_zero(self) -> None:
        assert horizon_value(self.POINTS)[2] == pytest.approx(0.0)
