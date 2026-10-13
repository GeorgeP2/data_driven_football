"""Plan transfers over several gameweeks at once, not just the next one.

The one-week planner has a specific, expensive failure: it panics at blank gameweeks. When half
your squad has no fixture it sees a wall of zeroes, takes four transfers to chase players who do
play, pays a 16-point hit, and churns straight back the following week. Across the run-in of
2025-26 that cost 48 points in hits alone.

The fix is to look further ahead. This plans over a rolling horizon — solve for the next H
gameweeks together, but only *execute* the first one, then re-plan next week with fresh
information. That is a standard receding-horizon controller, and it changes the character of the
decisions: a player who blanks this week but has a double next week is worth keeping, and the
planner can now see that.

Two approximations, both deliberate and both conservative:

- **Prices are held fixed across the horizon.** Predicting price rises is a separate forecasting
  problem, and getting it wrong is worse than ignoring it.
- **Future gameweeks are discounted.** A point three weeks out is worth less than a point now,
  because the forecast decays and the plan will be revised anyway. Without this the planner
  over-commits to distant fixtures it will never actually play.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from ortools.sat.python import cp_model

from fpl_optimiser.backtest.season import GameweekView
from fpl_optimiser.backtest.transfers import TransferDecision
from fpl_optimiser.model import (
    DEFAULT_BENCH_WEIGHTS,
    DEFAULT_SOLVER_TIMEOUT_SECONDS,
    GOALKEEPER,
    POINTS_SCALE,
    BenchWeights,
    SquadVars,
    build_squad,
)
from fpl_optimiser.players import Player
from fpl_optimiser.rules import GameRules, TransferRules

logger = logging.getLogger(__name__)

DEFAULT_HORIZON = 5
"""Gameweeks to plan over. Long enough to see past a blank and to bank a transfer for a double;
short enough that the forecast is still worth something at the far end."""

DEFAULT_DISCOUNT = 0.85
"""How much a point next gameweek is worth relative to this one. Compounds across the horizon,
so week 5 counts for about half of week 1."""

CANDIDATES_PER_POSITION = 40
"""The pool is ~840 players and the model is solving 15 x H binary decisions over it. Restricting
each position to its most promising candidates (by best expected points anywhere in the horizon),
plus everyone currently owned, keeps the solve fast without meaningfully changing the answer —
the 41st-best midfielder was never going into the squad."""

TRANSFER_TIEBREAK = 1
"""A hundredth of a point per transfer, to break ties toward holding rather than churning."""


@dataclass(frozen=True)
class PlannedGameweek:
    """One gameweek of the plan. Only the first is executed; the rest inform it."""

    gameweek: int
    transfers_in: tuple[Player, ...]
    transfers_out: tuple[Player, ...]
    hits: int


def _shortlist(
    views: Sequence[GameweekView],
    points: dict[int, dict[int, float]],
    rules: GameRules,
    owned: set[int],
) -> list[Player]:
    """Everyone worth considering: the best candidates per position, plus the current squad.

    The squad must always be in the pool — you cannot plan around players you are not allowed to
    keep — and dropping the no-hopers is what makes a 5-gameweek solve tractable.
    """
    by_id = {p.id: p for view in views for p in view.pool}
    best: dict[int, float] = {}
    for player_id in by_id:
        best[player_id] = max(points[view.gameweek].get(player_id, 0.0) for view in views)

    shortlisted: set[int] = set(owned)
    for rule in rules.positions:
        candidates = sorted(
            (p for p in by_id.values() if p.element_type == rule.element_type),
            key=lambda p: (-best[p.id], p.id),
        )
        shortlisted.update(p.id for p in candidates[:CANDIDATES_PER_POSITION])

    return [by_id[i] for i in sorted(shortlisted) if i in by_id]


def plan_transfers(
    views: Sequence[GameweekView],
    points_by_gameweek: dict[int, dict[int, float]],
    rules: GameRules,
    transfer_rules: TransferRules,
    *,
    current_squad_ids: set[int],
    purchase_prices: dict[int, int],
    bank_tenths: int,
    free_transfers: int,
    wildcard: bool = False,
    bench_weights: BenchWeights = DEFAULT_BENCH_WEIGHTS,
    discount: float = DEFAULT_DISCOUNT,
    timeout_seconds: float = DEFAULT_SOLVER_TIMEOUT_SECONDS,
) -> TransferDecision:
    """Solve the horizon, return only the first gameweek's decision.

    Args:
        views: This gameweek's view first, then the next few. Prices come from the first.
        points_by_gameweek: Expected points per player, per gameweek in the horizon.
        current_squad_ids: The 15 you own going in.
        free_transfers: Free transfers available *now*.
        wildcard: Play a wildcard in the first gameweek of the horizon — that week's transfers
            are unlimited and free, and the banked allowance is *frozen* across it (FPL neither
            drains it nor adds the usual one). Note this cannot be faked by passing a large
            `free_transfers`: the allowance carries forward as `min(available - used + 1, cap)`,
            so a large opening bank is one that *refills*, and the solver then prefers to spread
            its moves over the horizon rather than spend them at once. Measured on a
            rebuild-from-nothing fixture, that proxy made the "wildcard" worth 19 points *less*
            than not playing one.

    Raises:
        ValueError: if the current squad is not a legal starting point.
    """
    if not views:
        raise ValueError("Cannot plan over an empty horizon")
    if len(current_squad_ids) != rules.squad_size:
        raise ValueError(
            f"The current squad must hold {rules.squad_size} players, got {len(current_squad_ids)}"
        )

    now = views[0]
    pool = _shortlist(views, points_by_gameweek, rules, current_squad_ids)
    by_id = {p.id: p for p in pool}

    missing = current_squad_ids - by_id.keys()
    if missing:
        ids = ", ".join(str(i) for i in sorted(missing))
        raise ValueError(f"Owned players are absent from the pool: {ids}")

    # Prices are this gameweek's, held fixed across the horizon. A player you own effectively
    # costs their selling price: keeping them means declining to bank that money.
    sell_prices = {
        player_id: transfer_rules.selling_price(
            purchase_prices[player_id], by_id[player_id].cost_tenths
        )
        for player_id in current_squad_ids
    }
    prices = {p.id: sell_prices[p.id] if p.id in current_squad_ids else p.cost_tenths for p in pool}
    funds = bank_tenths + sum(sell_prices.values())

    model = cp_model.CpModel()
    horizon = list(range(len(views)))

    squad = {(p.id, t): model.NewBoolVar(f"squad_{p.id}_{t}") for p in pool for t in horizon}
    xi = {(p.id, t): model.NewBoolVar(f"xi_{p.id}_{t}") for p in pool for t in horizon}
    captain = {(p.id, t): model.NewBoolVar(f"capt_{p.id}_{t}") for p in pool for t in horizon}
    bought = {(p.id, t): model.NewBoolVar(f"in_{p.id}_{t}") for p in pool for t in horizon}
    sold = {(p.id, t): model.NewBoolVar(f"out_{p.id}_{t}") for p in pool for t in horizon}

    objective = []

    for t in horizon:
        view = views[t]
        points = points_by_gameweek[view.gameweek]
        weight = discount**t

        for player in pool:
            key = (player.id, t)
            model.Add(xi[key] <= squad[key])
            model.Add(captain[key] <= xi[key])

            # Squad evolution: you hold a player this week if you held them last week and did
            # not sell, or you bought them. At t=0 "last week" is the squad you actually own.
            held_before = (
                (1 if player.id in current_squad_ids else 0)
                if t == 0
                else squad[(player.id, t - 1)]
            )
            model.Add(squad[key] == held_before + bought[key] - sold[key])
            # You cannot buy someone you already have, nor sell someone you do not.
            model.Add(bought[key] + sold[key] <= 1)

        model.Add(sum(squad[(p.id, t)] for p in pool) == rules.squad_size)
        model.Add(sum(xi[(p.id, t)] for p in pool) == rules.starting_size)
        model.Add(sum(captain[(p.id, t)] for p in pool) == 1)

        for rule in rules.positions:
            of_position = [p for p in pool if p.element_type == rule.element_type]
            model.Add(sum(squad[(p.id, t)] for p in of_position) == rule.squad_select)
            starters = sum(xi[(p.id, t)] for p in of_position)
            model.Add(starters >= rule.min_play)
            model.Add(starters <= rule.max_play)

        model.Add(sum(prices[p.id] * squad[(p.id, t)] for p in pool) <= funds)

        for team_id in {p.team_id for p in pool}:
            of_team = [p for p in pool if p.team_id == team_id]
            model.Add(sum(squad[(p.id, t)] for p in of_team) <= rules.club_limit)

        # The bench is not four equal places: the first substitute comes on in about half of all
        # gameweeks, the reserve keeper almost never. Give each outfielder a "am I first on the
        # bench?" binary and let the solver promote its best, which a decreasing weight makes it
        # want to do anyway.
        outfield = [p for p in pool if p.position != GOALKEEPER]
        first_sub = {p.id: model.NewBoolVar(f"first_sub_{p.id}_{t}") for p in outfield}
        for player in outfield:
            model.Add(first_sub[player.id] <= squad[(player.id, t)] - xi[(player.id, t)])
        if outfield:
            model.Add(sum(first_sub.values()) <= 1)

        for player in pool:
            key = (player.id, t)
            scaled = round(points.get(player.id, 0.0) * POINTS_SCALE * weight)
            keeper = player.position == GOALKEEPER
            bench_weight = bench_weights.keeper if keeper else bench_weights.rest
            bench_value = round(scaled * bench_weight)
            objective.append((scaled - bench_value) * xi[key])
            objective.append(scaled * captain[key])
            objective.append(bench_value * squad[key])
            if not keeper:
                promotion = round(scaled * (bench_weights.first - bench_weights.rest))
                objective.append(promotion * first_sub[player.id])

    # Free-transfer accounting. You earn one a week and may bank up to the cap; anything beyond
    # what you have banked costs 4 points. Modelling the *banking* is the whole reason a
    # horizon helps: it lets the planner deliberately skip a week to afford two moves the next.
    cap = transfer_rules.max_free_transfers
    available = model.NewIntVar(free_transfers, free_transfers, "ft_0")
    first_hits: cp_model.IntVar | None = None

    for t in horizon:
        made = model.NewIntVar(0, rules.squad_size, f"transfers_{t}")
        model.Add(made == sum(bought[(p.id, t)] for p in pool))

        hits = model.NewIntVar(0, rules.squad_size, f"hits_{t}")

        if wildcard and t == 0:
            # Unlimited and free: no hit however many moves, and the allowance is frozen — it
            # neither drains nor earns its weekly one, which is the same rule the live
            # free-transfer count is derived under.
            model.Add(hits == 0)
        else:
            model.Add(hits >= made - available)
            # Carry the allowance forward: whatever was not used, plus the one you earn, capped.
            used = model.NewIntVar(0, cap, f"used_{t}")
            model.AddMinEquality(used, [made, available])
            carried = model.NewIntVar(1, cap, f"ft_{t + 1}")
            model.AddMinEquality(carried, [available - used + 1, cap])
            available = carried

        objective.append(-transfer_rules.hit_cost_points * POINTS_SCALE * hits)
        objective.append(-TRANSFER_TIEBREAK * made)

        if t == 0:
            first_hits = hits

    assert first_hits is not None  # the horizon is non-empty, checked above

    model.Maximize(sum(objective))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_seconds
    solver.parameters.num_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ValueError(
            f"No legal plan exists from the current squad with £{funds / 10:.1f}m of funds. "
            f"Suspect the selling-price or bank arithmetic."
        )

    # Read back only the first gameweek — the rest of the plan is thrown away and re-solved
    # next week with better information. That is the point of a rolling horizon.
    first = SquadVars(
        in_squad={p.id: squad[(p.id, 0)] for p in pool},
        in_xi={p.id: xi[(p.id, 0)] for p in pool},
        is_captain={p.id: captain[(p.id, 0)] for p in pool},
    )
    now_points = points_by_gameweek[now.gameweek]
    result = build_squad(pool, {p.id: now_points.get(p.id, 0.0) for p in pool}, solver, first)
    selected = result.player_ids

    spent = sum(prices[i] for i in selected)
    return TransferDecision(
        squad=result,
        transfers_in=tuple(
            sorted((by_id[i] for i in selected - current_squad_ids), key=lambda p: p.id)
        ),
        transfers_out=tuple(
            sorted((by_id[i] for i in current_squad_ids - selected), key=lambda p: p.id)
        ),
        hits=solver.Value(first_hits),
        bank_tenths=funds - spent,
        horizon_value=solver.ObjectiveValue() / POINTS_SCALE,
    )
