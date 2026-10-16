"""What the advisor records each week: the planned path and the raw API payload.

The planner solves the whole horizon and used to keep only the first week. A move that only
makes sense as the first step of a chain — selling cheap now to fund a bigger buy later — then
reached the advice with its reason thrown away. These pin that the later weeks come back, that
they agree with the executed week, and that they survive into the log.

The snapshot exists because FPL overwrites injury flags in place and keeps no history of them.
"""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime

from fpl_helpers import make_player
from fpl_optimiser.backtest.history import Fixture
from fpl_optimiser.backtest.planner import plan_transfers
from fpl_optimiser.backtest.season import GameweekView
from fpl_optimiser.backtest.strength import TeamRatings
from fpl_optimiser.cli_advise import write_bootstrap_snapshot, write_plan_log
from fpl_optimiser.rules import TransferRules

TRANSFER_RULES = TransferRules(sell_on_fee=0.5, max_free_transfers=5)
SQUAD_SIZE = 15
GAMEWEEKS = (1, 2, 3, 4, 5)


def _pool():
    """Fifteen cheap players and fifteen mirrors, so every upgrade is a like-for-like swap."""
    shape = [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4]
    pool = []
    for index, element_type in enumerate(shape):
        pool.append(make_player(index + 1, element_type, cost_tenths=45, team_id=index % 10 + 1))
        pool.append(make_player(index + 101, element_type, cost_tenths=45, team_id=index % 10 + 11))
    return tuple(pool)


def _plan(rules):
    """Hold the fifteen worthless players with one free transfer; the `1xx` mirrors score 5."""
    pool = _pool()
    fixtures: dict[int, tuple[Fixture, ...]] = {p.id: () for p in pool}
    views, points = [], {}
    for gameweek in GAMEWEEKS:
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
        points[gameweek] = {p.id: (5.0 if p.id > 100 else 0.0) for p in pool}
    squad = {p.id for p in pool if p.id <= 100}
    return plan_transfers(
        views,
        points,
        rules,
        TRANSFER_RULES,
        current_squad_ids=squad,
        purchase_prices={i: 45 for i in squad},
        bank_tenths=0,
        free_transfers=1,
    )


class TestThePlanIsReturned:
    def test_every_week_of_the_horizon_comes_back_in_order(self, rules) -> None:
        assert [week.gameweek for week in _plan(rules).plan] == list(GAMEWEEKS)

    def test_the_first_week_is_the_decision_that_is_executed(self, rules) -> None:
        decision = _plan(rules)
        first = decision.plan[0]

        assert {p.id for p in first.transfers_in} == {p.id for p in decision.transfers_in}
        assert {p.id for p in first.transfers_out} == {p.id for p in decision.transfers_out}
        assert first.hits == decision.hits

    def test_the_weeks_chain_into_one_consistent_squad(self, rules) -> None:
        # Nobody is bought twice or sold before they were bought: the weeks are one path, not
        # five independent answers.
        owned = {i for i in range(1, SQUAD_SIZE + 1)}
        for week in _plan(rules).plan:
            sold = {p.id for p in week.transfers_out}
            bought = {p.id for p in week.transfers_in}
            assert sold <= owned
            assert not bought & owned
            owned = (owned - sold) | bought
            assert len(owned) == SQUAD_SIZE


class TestThePlanIsLogged:
    def test_the_log_pairs_each_move_within_its_position(self, rules, tmp_path) -> None:
        decision = _plan(rules)
        written = write_plan_log(
            tmp_path,
            payload={"events": [{"id": 1, "deadline_time": "2026-08-15T10:00:00Z"}]},
            gameweek=1,
            plan=decision.plan,
        )
        logged = json.loads(written.read_text())
        by_id = {p.id: p for p in _pool()}

        assert [week["gameweek"] for week in logged["weeks"]] == list(GAMEWEEKS)
        for week in logged["weeks"]:
            for move in week["transfers"]:
                assert by_id[move["out_id"]].element_type == by_id[move["in_id"]].element_type


PAYLOAD = {
    "events": [{"id": 1, "deadline_time": "2026-08-15T10:00:00Z"}],
    "elements": [{"id": 7, "status": "d", "news": "Knock - 75% chance of playing"}],
}


class TestTheBootstrapSnapshot:
    def test_it_round_trips_the_payload_exactly(self, tmp_path) -> None:
        written = write_bootstrap_snapshot(tmp_path, payload=PAYLOAD, gameweek=7)

        with gzip.open(written, "rt", encoding="utf-8") as handle:
            assert json.load(handle) == PAYLOAD

    def test_two_runs_before_one_deadline_both_survive(self, tmp_path) -> None:
        # Flags change between runs in the same week; overwriting would lose the earlier one.
        for hour in (9, 21):
            write_bootstrap_snapshot(
                tmp_path,
                payload=PAYLOAD,
                gameweek=7,
                now=datetime(2026, 10, 16, hour, tzinfo=UTC),
            )

        assert len(list(tmp_path.glob("bootstrap_gw07_2026-27_*.json.gz"))) == 2
