"""Pick next gameweek's squad when you already own one.

This is the same integer program as `optimise_squad`, with three additions:

- **Prices you actually face.** A player you already own effectively costs their *selling* price,
  because keeping them means declining to bank that money. Selling prices carry the sell-on fee,
  so they depend on what you paid, not just what they are worth.
- **A transfer count.** Since the squad is always 15 and you already hold 15, the number of
  players bought equals the number sold, so counting the incomers is enough.
- **A penalty for churn.** Transfers beyond your free ones cost 4 points each, and the solver
  weighs that against the points they would buy. It will take a hit when, and only when, the
  hit pays for itself *this gameweek* — which is precisely the myopia of a one-week horizon,
  and the main reason a multi-gameweek planner would beat this.

A wildcard is this same solve with `free_transfers` set to the squad size: unlimited changes, no
penalty.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ortools.sat.python import cp_model

from fpl_optimiser.expected_points import ExpectedPoints
from fpl_optimiser.model import (
    DEFAULT_BENCH_WEIGHTS,
    DEFAULT_SOLVER_TIMEOUT_SECONDS,
    POINTS_SCALE,
    BenchWeights,
    Squad,
    add_squad_constraints,
    build_squad,
    create_squad_vars,
    points_objective_terms,
    solve,
    validate_pool,
    validate_solver_arguments,
)
from fpl_optimiser.players import Player
from fpl_optimiser.rules import GameRules, TransferRules

logger = logging.getLogger(__name__)

TRANSFER_TIEBREAK = 1
"""One scaled unit — a hundredth of a point — docked per transfer, purely to break ties toward
keeping the player you have. Small enough that it never overrides a genuine gain."""


@dataclass(frozen=True)
class TransferDecision:
    """The squad to field, and what it cost to get there."""

    squad: Squad
    transfers_in: tuple[Player, ...]
    transfers_out: tuple[Player, ...]
    hits: int
    """Transfers taken beyond the free allowance."""

    bank_tenths: int
    """Money left over after the moves."""

    horizon_value: float = 0.0
    """The solver's objective for the whole plan, in points.

    Not a points forecast — it is bench-weighted, discounted by week and already net of any
    hits, so it is only meaningful *compared with another solve over the same pool and
    horizon*. That comparison is exactly what valuing a wildcard needs, which is why it is
    exposed at all."""

    @property
    def transfer_count(self) -> int:
        return len(self.transfers_in)

    @property
    def hit_points(self) -> int:
        return self.hits * 4


def optimise_transfers(
    players: Sequence[Player],
    rules: GameRules,
    transfer_rules: TransferRules,
    expected_points: ExpectedPoints,
    *,
    current_squad_ids: Iterable[int],
    purchase_prices: dict[int, int],
    bank_tenths: int,
    free_transfers: int,
    bench_weights: BenchWeights = DEFAULT_BENCH_WEIGHTS,
    timeout_seconds: float = DEFAULT_SOLVER_TIMEOUT_SECONDS,
) -> TransferDecision:
    """Choose transfers, the XI and the captain, trading points gained against points docked.

    Args:
        players: This gameweek's pool, priced at this gameweek's prices.
        current_squad_ids: The 15 you own going in.
        purchase_prices: What you paid for each owned player, in tenths. Needed for the sell-on
            fee, which is charged on profit.
        bank_tenths: Cash in hand.
        free_transfers: Transfers available at no cost. Pass `rules.squad_size` for a wildcard.

    Raises:
        ValueError: on bad arguments, or if no legal squad is reachable.
    """
    if free_transfers < 0:
        raise ValueError(f"free_transfers must be >= 0, got {free_transfers}")
    if bank_tenths < 0:
        raise ValueError(f"bank_tenths must be >= 0, got {bank_tenths}")

    current = set(current_squad_ids)
    if len(current) != rules.squad_size:
        raise ValueError(
            f"The current squad must hold {rules.squad_size} players, got {len(current)}"
        )

    pool = list(players)
    by_id = {p.id: p for p in pool}
    missing = current - by_id.keys()
    if missing:
        ids = ", ".join(str(i) for i in sorted(missing))
        raise ValueError(f"Owned players are absent from this gameweek's pool: {ids}")

    missing_prices = current - purchase_prices.keys()
    if missing_prices:
        ids = ", ".join(str(i) for i in sorted(missing_prices))
        raise ValueError(f"No purchase price recorded for owned players: {ids}")

    # What each player costs *you*. Keeping someone you own is economically identical to selling
    # them and buying them back at their selling price, which is what makes this a single clean
    # budget constraint rather than a separate buy/sell ledger.
    sell_prices = {
        player_id: transfer_rules.selling_price(
            purchase_prices[player_id], by_id[player_id].cost_tenths
        )
        for player_id in current
    }
    prices = {p.id: sell_prices[p.id] if p.id in current else p.cost_tenths for p in pool}
    funds = bank_tenths + sum(sell_prices.values())

    bench_weights.validate()
    validate_solver_arguments(funds, timeout_seconds)
    validate_pool(pool, rules, set())

    points = {player.id: float(expected_points(player)) for player in pool}

    model = cp_model.CpModel()
    variables = create_squad_vars(model, pool)
    add_squad_constraints(model, variables, pool, rules, prices=prices, budget_tenths=funds)

    incoming = [variables.in_squad[p.id] for p in pool if p.id not in current]
    transfers = model.NewIntVar(0, rules.squad_size, "transfers")
    model.Add(transfers == sum(incoming))

    # hits = max(0, transfers - free_transfers). The lower bounds alone suffice: the objective
    # penalises hits, so the solver will never inflate it above what the constraints force.
    hits = model.NewIntVar(0, rules.squad_size, "hits")
    model.Add(hits >= transfers - free_transfers)

    objective = points_objective_terms(model, variables, pool, points, bench_weights)
    penalty = transfer_rules.hit_cost_points * POINTS_SCALE * hits
    # A free transfer is free, so with tied expected points the solver would happily churn the
    # squad for no gain — and worse, burn a banked transfer doing it. This nudge, worth a
    # hundredth of a point, breaks those ties toward holding without ever outweighing a real
    # improvement.
    churn = TRANSFER_TIEBREAK * transfers
    model.Maximize(sum(objective) - penalty - churn)

    solver, status = solve(model, timeout_seconds)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ValueError(
            f"No legal squad is reachable from the current one with £{funds / 10:.1f}m of "
            f"funds. This should not happen from a legal starting squad — suspect the "
            f"selling-price or bank arithmetic."
        )

    squad = build_squad(pool, points, solver, variables)
    selected = squad.player_ids

    transfers_in = tuple(sorted((by_id[i] for i in selected - current), key=lambda p: p.id))
    transfers_out = tuple(sorted((by_id[i] for i in current - selected), key=lambda p: p.id))
    spent = sum(prices[i] for i in selected)

    return TransferDecision(
        squad=squad,
        transfers_in=transfers_in,
        transfers_out=transfers_out,
        hits=solver.Value(hits),
        bank_tenths=funds - spent,
    )
