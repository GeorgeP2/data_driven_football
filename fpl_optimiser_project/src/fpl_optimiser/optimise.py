"""Pick a squad from scratch — the wildcard / season-opener case.

The integer program itself lives in `model.py`, shared with the transfer-constrained solve in
`backtest/transfers.py`. This module is the entry point for the case where you own nothing yet
and simply buy the best legal fifteen your budget allows.

The `bench_weight` knob is the one genuinely judgement-based parameter: at zero the solver has
no reason to prefer a good bench over a worthless one, and will fill the four bench slots with
the cheapest bodies it can find to free up money for the XI. A small positive weight buys
insurance against a starter being dropped or injured at the deadline. The backtest is what lets
you tune it, because auto-substitutions are the only mechanism by which a good bench pays off.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from ortools.sat.python import cp_model

from fpl_optimiser.expected_points import ExpectedPoints
from fpl_optimiser.model import (
    DEFAULT_BENCH_WEIGHTS,
    DEFAULT_SOLVER_TIMEOUT_SECONDS,
    BenchWeights,
    Squad,
    add_squad_constraints,
    build_squad,
    create_squad_vars,
    infeasible_message,
    points_objective_terms,
    solve,
    validate_pool,
    validate_solver_arguments,
)
from fpl_optimiser.players import Player
from fpl_optimiser.rules import GameRules

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BENCH_WEIGHTS",
    "DEFAULT_SOLVER_TIMEOUT_SECONDS",
    "Squad",
    "optimise_squad",
]


def optimise_squad(
    players: Sequence[Player],
    rules: GameRules,
    expected_points: ExpectedPoints,
    *,
    budget_tenths: int | None = None,
    bench_weights: BenchWeights = DEFAULT_BENCH_WEIGHTS,
    excluded_player_ids: Iterable[int] = (),
    locked_player_ids: Iterable[int] = (),
    timeout_seconds: float = DEFAULT_SOLVER_TIMEOUT_SECONDS,
) -> Squad:
    """Pick the squad that maximises expected points subject to the FPL rules.

    Args:
        players: The pool to select from.
        rules: Squad-legality rules, normally from `GameRules.from_bootstrap`.
        expected_points: Callable giving each player's expected points for the gameweek.
        budget_tenths: Budget in tenths of a million. Defaults to the game's £100.0m.
        bench_weights: How much each bench slot counts. The first substitute is worth far more
            than the third, which is why this is four numbers rather than one.
        excluded_player_ids: Players to keep out of the squad (e.g. ones you refuse to own).
        locked_player_ids: Players that must be in the 15 (e.g. ones you will not sell).
        timeout_seconds: Give up after this long and return the best squad found so far.

    Raises:
        ValueError: if an argument is out of range, the pool cannot fill the squad, or no
            legal squad exists within the budget.
    """
    budget = rules.budget_tenths if budget_tenths is None else budget_tenths
    bench_weights.validate()
    validate_solver_arguments(budget, timeout_seconds)

    excluded = set(excluded_player_ids)
    locked = set(locked_player_ids)
    overlap = excluded & locked
    if overlap:
        ids = ", ".join(str(i) for i in sorted(overlap))
        raise ValueError(f"Player ids are both locked and excluded: {ids}")

    pool = [player for player in players if player.id not in excluded]
    validate_pool(pool, rules, locked)

    points = {player.id: float(expected_points(player)) for player in pool}
    prices = {player.id: player.cost_tenths for player in pool}

    model = cp_model.CpModel()
    variables = create_squad_vars(model, pool)
    add_squad_constraints(
        model,
        variables,
        pool,
        rules,
        prices=prices,
        budget_tenths=budget,
        locked_player_ids=locked,
    )
    model.Maximize(sum(points_objective_terms(model, variables, pool, points, bench_weights)))

    solver, status = solve(model, timeout_seconds)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ValueError(infeasible_message(pool, rules, budget, locked, status))

    return build_squad(pool, points, solver, variables)
