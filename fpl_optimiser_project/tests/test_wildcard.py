"""What a wildcard is worth.

The advisor passed a hardcoded `wildcard_gain=0.0` into `choose_chip`, whose branch tests
`>= max(25 * urgency, 15)`. The chip was therefore unreachable in every gameweek of every
season — not mistimed, impossible. These pin the replacement to cases where the right answer is
obvious by construction, because a chip valuation is otherwise very hard to tell from a
plausible-looking number.
"""

from __future__ import annotations

import pytest
from fpl_helpers import make_player
from fpl_optimiser.backtest.chips import WILDCARD_FLOOR, Chip, ChipState, choose_chip
from fpl_optimiser.backtest.history import Fixture
from fpl_optimiser.backtest.season import GameweekView
from fpl_optimiser.backtest.strength import TeamRatings
from fpl_optimiser.backtest.wildcard import wildcard_gain
from fpl_optimiser.rules import TransferRules

TRANSFER_RULES = TransferRules(sell_on_fee=0.5, max_free_transfers=5)
SQUAD_SIZE = 15


def _pool():
    """Thirty players: fifteen cheap, fifteen good, no club limit in the way.

    Position quotas are 2/5/5/3, so the two halves are built to mirror each other exactly and a
    rebuild is a straight swap of every slot.
    """
    shape = [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4]
    pool = []
    for index, element_type in enumerate(shape):
        pool.append(make_player(index + 1, element_type, cost_tenths=45, team_id=index % 10 + 1))
        pool.append(make_player(index + 101, element_type, cost_tenths=45, team_id=index % 10 + 11))
    return tuple(pool)


def _views(points_good: float, gameweeks=(1, 2, 3, 4, 5)):
    """Views where every `1xx` player is worth `points_good` and every `x` player nothing."""
    pool = _pool()
    fixtures: dict[int, tuple[Fixture, ...]] = {p.id: () for p in pool}
    views, points = [], {}
    for gameweek in gameweeks:
        views.append(
            GameweekView(
                gameweek=gameweek,
                pool=pool,
                lagged={},
                fixtures=fixtures,
                team_ratings=TeamRatings({}, {}, 1.4),
                teams={p.id: p.team_id for p in pool},
            )
        )
        points[gameweek] = {p.id: (points_good if p.id > 100 else 0.0) for p in pool}
    return views, points


def _gain(rules, points_good: float, *, free_transfers: int, held_are_bad: bool = True):
    views, points = _views(points_good)
    pool = views[0].pool
    held = {p.id for p in pool if (p.id <= 100) == held_are_bad}
    squad = set(sorted(held)[:SQUAD_SIZE])
    return wildcard_gain(
        views,
        points,
        rules,
        TRANSFER_RULES,
        current_squad_ids=squad,
        purchase_prices={i: 45 for i in squad},
        bank_tenths=0,
        free_transfers=free_transfers,
    )


class TestTheGain:
    def test_a_squad_that_cannot_be_improved_is_worth_nothing(self, rules) -> None:
        # Already holding the good fifteen: a wildcard buys exactly nothing, and the floor at
        # zero must hold rather than returning a small negative from solver noise.
        assert _gain(rules, 5.0, free_transfers=1, held_are_bad=False) == pytest.approx(0.0)

    def test_a_squad_needing_a_full_rebuild_is_worth_something(self, rules) -> None:
        # Fifteen worthless players, one transfer a week: the chip has real value here.
        assert _gain(rules, 5.0, free_transfers=1) > WILDCARD_FLOOR

    def test_the_gain_is_never_negative(self, rules) -> None:
        assert _gain(rules, 0.0, free_transfers=1) >= 0.0

    def test_more_free_transfers_make_the_chip_worth_less(self, rules) -> None:
        # The whole point of the rule: the counterfactual is what free transfers reach anyway,
        # so a manager with five banked needs the chip less than one with a single transfer.
        assert _gain(rules, 5.0, free_transfers=5) < _gain(rules, 5.0, free_transfers=1)

    def test_a_bigger_quality_gap_is_worth_more(self, rules) -> None:
        assert _gain(rules, 8.0, free_transfers=1) > _gain(rules, 3.0, free_transfers=1)


class TestItReachesTheChipDecision:
    """The bug was never in `choose_chip` — it was that nothing ever fed it a real number."""

    @staticmethod
    def _decide(gain: float) -> Chip | None:
        return choose_chip(
            5,
            list(range(1, 39)),
            ChipState(),
            bench_expected_points=0.0,
            captain_expected_points=0.0,
            squad_with_fixtures=15,
            wildcard_gain=gain,
        ).chip

    def test_the_old_hardcoded_zero_could_never_fire(self) -> None:
        assert self._decide(0.0) is None

    def test_a_gain_below_the_floor_still_does_not_fire(self) -> None:
        assert self._decide(WILDCARD_FLOOR - 0.1) is None

    def test_a_large_gain_now_fires(self) -> None:
        # The property the fix exists for: a big enough number reaches the branch and wins.
        assert self._decide(60.0) is Chip.WILDCARD


class TestTheChipIsModelledNotApproximated:
    """Regression: a wildcard is not "a lot of free transfers".

    The first implementation passed `free_transfers=squad_size` to mean "unlimited". But the
    allowance carries forward as `min(available - used + 1, cap)`, so a large opening bank is
    one that *refills* — and the solver then prefers to spread its moves across the horizon
    rather than spend them at once. On the rebuild fixture below that proxy valued the chip at
    19 points *less* than not playing it, which the `max(0.0, ...)` floor would have reported
    as a flat "worth nothing".
    """

    @staticmethod
    def _plan(rules, *, free_transfers: int, chip: bool):
        from fpl_optimiser.backtest.planner import plan_transfers

        views, points = _views(8.0)
        pool = views[0].pool
        squad = set(sorted(p.id for p in pool if p.id <= 100)[:SQUAD_SIZE])
        return plan_transfers(
            views,
            points,
            rules,
            TRANSFER_RULES,
            current_squad_ids=squad,
            purchase_prices={i: 45 for i in squad},
            bank_tenths=0,
            free_transfers=free_transfers,
            wildcard=chip,
        )

    def test_a_wildcard_rebuilds_the_whole_squad_at_once(self, rules) -> None:
        plan = self._plan(rules, free_transfers=1, chip=True)

        assert len(plan.transfers_in) == SQUAD_SIZE
        assert plan.hits == 0

    def test_the_large_free_transfer_proxy_does_not_rebuild(self, rules) -> None:
        # The tell that the proxy is wrong: given fifteen "free transfers" it declines to use
        # them, because holding them back refills the bank.
        proxy = self._plan(rules, free_transfers=SQUAD_SIZE, chip=False)

        assert len(proxy.transfers_in) < SQUAD_SIZE

    def test_the_chip_beats_the_proxy_it_replaced(self, rules) -> None:
        assert (
            self._plan(rules, free_transfers=1, chip=True).horizon_value
            > self._plan(rules, free_transfers=SQUAD_SIZE, chip=False).horizon_value
        )

    def test_a_wildcard_never_costs_a_hit(self, rules) -> None:
        assert self._plan(rules, free_transfers=0, chip=True).hits == 0
