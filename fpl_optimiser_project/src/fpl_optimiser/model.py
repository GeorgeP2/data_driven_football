"""The shared CP-SAT squad model.

Picking a squad from scratch and picking one under transfer restrictions are the *same* integer
program — same variables, same legality constraints, same objective — differing only in the
prices you pay, the money you have, and (for transfers) a penalty for churn. So the model lives
here once, and `optimise.py` and `backtest/transfers.py` are thin entry points onto it.

Three binary decisions per player: in the 15, in the starting XI, wearing the armband. The
objective is:

    sum over players of  ep[p] * ( start[p] + capt[p] + bench_weight * (squad[p] - start[p]) )

The `capt` term adds the captain's expected points a *second* time, which is exactly what the
armband does.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ortools.sat.python import cp_model

from fpl_optimiser.players import Player
from fpl_optimiser.rules import GameRules

logger = logging.getLogger(__name__)

POINTS_SCALE = 100
"""CP-SAT is integer-only, so expected points are scaled to hundredths before entering the
model. Two decimal places is finer than the forecast itself is meaningful to."""

DEFAULT_SOLVER_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class BenchWeights:
    """How much each bench slot's expected points count when picking the squad.

    A bench player is worth their points times the chance anyone actually needs them, and that
    chance is wildly different by slot. A starter you would willingly field fails to appear about
    one time in nine, so across eleven of them:

        1st outfield substitute   comes on ~50% of gameweeks
        2nd outfield substitute                ~8%
        3rd outfield substitute                ~0%
        reserve keeper                         ~5%   (only if your own keeper blanks)

    The first substitute is worth **more than ten times** the third. For most of this project a
    single uniform weight was smeared across that spread, which is why the squad kept ending up
    with four mediocre bench players instead of one good one and three cheap bodies — the shape a
    real manager picks by instinct.

    The four bench places are fixed by the rules: fifteen players, eleven starting, two keepers of
    whom one plays. So the bench is always exactly one reserve keeper and three outfielders, and
    the only decision is *which* outfielder is first — which the solver makes for itself, because
    a decreasing weight gives it every reason to promote its best.

    ## And yet the shaped default loses, so the flat one ships

    Three seeds per shape, against a season-score noise band of about 30 points:

        flat 0.25                        2225   <- best
        keeper .05  first .50  rest .08  2154   (-71)
        keeper .02  first .60  rest .05  2207   (-18)
        keeper .05  first .40  rest .15  2118   (-107)

    The reasoning above is not wrong about auto-substitutions. It is just not the whole story, and
    the missing half is this: a bench player is not only insurance against a blank, they are
    **next week's starter**. Over a three-to-five gameweek horizon the planner promotes bench
    players into the eleven as fixtures rotate and blanks arrive, and it does so without spending
    a transfer. Starving the second and third bench places, as the shaped weights do, buys one
    good substitute and leaves the squad with no depth to rotate into. A flat weight quietly pays
    for that depth, and the backtest can see the difference even though the theory could not.

    The mechanism is kept — it costs nothing, it is tested, and it lets a future model express a
    shape if one can be made to pay. The *default* is flat because that is what the evidence says.
    """

    keeper: float = 0.25
    first: float = 0.25
    rest: float = 0.25
    """All three equal, at the value the scalar sweep settled on. See the note above: this is a
    measured default, not a placeholder."""

    @classmethod
    def uniform(cls, weight: float) -> BenchWeights:
        """One weight for every slot — the old behaviour, kept for sweeps and comparisons."""
        return cls(keeper=weight, first=weight, rest=weight)

    def validate(self) -> None:
        for name, value in (
            ("keeper", self.keeper),
            ("first", self.first),
            ("rest", self.rest),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"bench weight '{name}' must be between 0.0 and 1.0, got {value}")


DEFAULT_BENCH_WEIGHTS = BenchWeights()

POSITION_ORDER = ("GKP", "DEF", "MID", "FWD")


@dataclass(frozen=True)
class Squad:
    """A rule-legal 15-man squad."""

    starting_xi: tuple[Player, ...]
    bench: tuple[Player, ...]
    """Bench in the order it should be set: the reserve keeper first (FPL fixes them in the
    keeper slot), then outfielders by descending expected points."""

    captain: Player
    vice_captain: Player
    expected_points: dict[int, float]
    """Expected points per player id, from the provider that was used."""

    @property
    def players(self) -> tuple[Player, ...]:
        return self.starting_xi + self.bench

    @property
    def player_ids(self) -> frozenset[int]:
        return frozenset(player.id for player in self.players)

    @property
    def total_cost_tenths(self) -> int:
        return sum(player.cost_tenths for player in self.players)

    @property
    def total_cost_millions(self) -> float:
        return self.total_cost_tenths / 10

    @property
    def starting_expected_points(self) -> float:
        """Expected points of the XI, counting the captain twice."""
        return (
            sum(self.expected_points[player.id] for player in self.starting_xi)
            + self.expected_points[self.captain.id]
        )

    @property
    def bench_expected_points(self) -> float:
        return sum(self.expected_points[player.id] for player in self.bench)

    @property
    def formation(self) -> str:
        """Outfield shape, e.g. "3-5-2"."""
        counts: dict[str, int] = {}
        for player in self.starting_xi:
            counts[player.position] = counts.get(player.position, 0) + 1
        outfield = [p for p in POSITION_ORDER if p != "GKP" and counts.get(p)]
        return "-".join(str(counts[position]) for position in outfield)


@dataclass(frozen=True)
class SquadVars:
    """The decision variables, one set per player id."""

    in_squad: dict[int, cp_model.IntVar]
    in_xi: dict[int, cp_model.IntVar]
    is_captain: dict[int, cp_model.IntVar]


def create_squad_vars(model: cp_model.CpModel, pool: Sequence[Player]) -> SquadVars:
    return SquadVars(
        in_squad={p.id: model.NewBoolVar(f"squad_{p.id}") for p in pool},
        in_xi={p.id: model.NewBoolVar(f"start_{p.id}") for p in pool},
        is_captain={p.id: model.NewBoolVar(f"capt_{p.id}") for p in pool},
    )


def add_squad_constraints(
    model: cp_model.CpModel,
    variables: SquadVars,
    pool: Sequence[Player],
    rules: GameRules,
    *,
    prices: dict[int, int],
    budget_tenths: int,
    locked_player_ids: Iterable[int] = (),
) -> None:
    """Add every FPL legality rule: squad size, formation, budget, and the per-club cap.

    Args:
        prices: What each player costs *this* buyer, in tenths. For a fresh squad that is the
            market price; under transfer rules a player you already own effectively costs their
            selling price, since keeping them is declining to bank that money.
        budget_tenths: Total funds available, in the same units.
    """
    in_squad, in_xi, is_captain = (
        variables.in_squad,
        variables.in_xi,
        variables.is_captain,
    )

    # You cannot start a player you do not own, nor captain one you do not start. Captaincy
    # implies selection transitively, so it needs no constraint of its own.
    for player in pool:
        model.Add(in_xi[player.id] <= in_squad[player.id])
        model.Add(is_captain[player.id] <= in_xi[player.id])

    model.Add(sum(in_squad.values()) == rules.squad_size)
    model.Add(sum(in_xi.values()) == rules.starting_size)
    model.Add(sum(is_captain.values()) == 1)

    for rule in rules.positions:
        of_position = [p for p in pool if p.element_type == rule.element_type]
        model.Add(sum(in_squad[p.id] for p in of_position) == rule.squad_select)
        starters = sum(in_xi[p.id] for p in of_position)
        model.Add(starters >= rule.min_play)
        model.Add(starters <= rule.max_play)

    model.Add(sum(prices[p.id] * in_squad[p.id] for p in pool) <= budget_tenths)

    for team_id in {p.team_id for p in pool}:
        of_team = [p for p in pool if p.team_id == team_id]
        model.Add(sum(in_squad[p.id] for p in of_team) <= rules.club_limit)

    for player_id in locked_player_ids:
        model.Add(in_squad[player_id] == 1)


GOALKEEPER = "GKP"


def points_objective_terms(
    model: cp_model.CpModel,
    variables: SquadVars,
    pool: Sequence[Player],
    points: dict[int, float],
    bench_weights: BenchWeights,
    *,
    suffix: str = "",
) -> list[cp_model.LinearExpr]:
    """The expected-points half of the objective, as integer-coefficient linear terms.

    The starting eleven and the captain are straightforward: full points, and the captain's
    counted twice. The bench is where the care is needed, because its four places are not worth
    the same. A reserve keeper is nearly dead money; the first outfield substitute comes on in
    about half of all gameweeks.

    So each outfielder gets one extra binary — *are you the first substitute?* — and nothing
    forces the solver to promote its best bench player into that slot. It does so anyway, because
    the slot is worth more than the others and the solver is maximising. The ordering falls out
    of the objective rather than being imposed on it.

    Args:
        suffix: Disambiguates variable names when the same pool is modelled across several
            gameweeks, as the multi-week planner does.
    """
    in_squad, in_xi = variables.in_squad, variables.in_xi
    outfield = [p for p in pool if p.position != GOALKEEPER]

    # "Is this player the first substitute?" — at most one, and only if actually benched.
    first_sub = {p.id: model.NewBoolVar(f"first_sub_{p.id}{suffix}") for p in outfield}
    for player in outfield:
        model.Add(first_sub[player.id] <= in_squad[player.id] - in_xi[player.id])
    if outfield:
        # There are always three outfield substitutes under the FPL rules, so exactly one of them
        # is first. `<=` rather than `==` keeps the model feasible under odd rule variants where
        # the bench could be all-goalkeeper.
        model.Add(sum(first_sub.values()) <= 1)

    terms: list[cp_model.LinearExpr] = []
    for player in pool:
        scaled = round(points[player.id] * POINTS_SCALE)
        keeper = player.position == GOALKEEPER
        bench_weight = bench_weights.keeper if keeper else bench_weights.rest
        bench_value = round(scaled * bench_weight)

        # ep*start + ep*capt + bench*(squad - start)
        #   == (ep - bench)*start + ep*capt + bench*squad
        terms.append((scaled - bench_value) * in_xi[player.id])
        terms.append(scaled * variables.is_captain[player.id])
        terms.append(bench_value * in_squad[player.id])

        if not keeper:
            # Being *first* on the bench is worth the difference between the two weights.
            promotion = round(scaled * (bench_weights.first - bench_weights.rest))
            terms.append(promotion * first_sub[player.id])

    return terms


def solve(model: cp_model.CpModel, timeout_seconds: float) -> tuple[cp_model.CpSolver, int]:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_seconds
    status = solver.Solve(model)
    if status == cp_model.FEASIBLE:
        logger.warning(
            "Solver hit the %.1fs timeout; returning the best squad found, which may not be "
            "optimal.",
            timeout_seconds,
        )
    return solver, status


def build_squad(
    pool: Sequence[Player],
    points: dict[int, float],
    solver: cp_model.CpSolver,
    variables: SquadVars,
) -> Squad:
    """Read the solved variables back into a `Squad`."""
    selected = [p for p in pool if solver.Value(variables.in_squad[p.id])]
    starters = [p for p in selected if solver.Value(variables.in_xi[p.id])]
    benched = [p for p in selected if not solver.Value(variables.in_xi[p.id])]

    def by_points(candidates: Iterable[Player]) -> list[Player]:
        return sorted(candidates, key=lambda p: (-points[p.id], p.id))

    # The `is_captain` variable exists so the solver prices the armband in when choosing the
    # XI, but it is not read back here. Given a fixed XI, the optimal captain is by definition
    # its highest-scoring member, and deriving it breaks ties by player id — where reading the
    # solver's flag would hand back an arbitrary one of several equally optimal choices and
    # make the output wobble between runs.
    ranked_starters = by_points(starters)
    captain, vice_captain = ranked_starters[0], ranked_starters[1]

    # Present the XI the way a team sheet reads: keeper, defenders, midfielders, forwards,
    # each group strongest first.
    ordered_xi = [
        player
        for position in POSITION_ORDER
        for player in by_points(p for p in starters if p.position == position)
    ]
    # FPL fixes the reserve keeper in the first bench slot; the rest is a free ordering.
    bench_keepers = by_points(p for p in benched if p.position == "GKP")
    bench_outfield = by_points(p for p in benched if p.position != "GKP")

    return Squad(
        starting_xi=tuple(ordered_xi),
        bench=tuple(bench_keepers + bench_outfield),
        captain=captain,
        vice_captain=vice_captain,
        expected_points={p.id: points[p.id] for p in selected},
    )


def validate_solver_arguments(budget_tenths: int, timeout_seconds: float) -> None:
    if budget_tenths < 1:
        raise ValueError(
            f"budget_tenths must be >= 1 (tenths of a million, so 1000 = £100.0m), got "
            f"{budget_tenths}"
        )
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")


def validate_pool(pool: Sequence[Player], rules: GameRules, locked: set[int]) -> None:
    """Fail with a specific message when the pool obviously cannot make a squad.

    Without this, "you excluded so many midfielders that only four remain" arrives as a bare
    INFEASIBLE from the solver, which tells you nothing about which rule you broke.
    """
    if not pool:
        raise ValueError("The player pool is empty (did the exclusions remove everyone?)")

    available_ids = {p.id for p in pool}
    missing = locked - available_ids
    if missing:
        ids = ", ".join(str(i) for i in sorted(missing))
        raise ValueError(f"Locked player ids are not in the pool: {ids}")

    for rule in rules.positions:
        count = sum(1 for p in pool if p.element_type == rule.element_type)
        if count < rule.squad_select:
            raise ValueError(
                f"Only {count} {rule.short_name} available but the squad needs {rule.squad_select}"
            )

    locked_by_team: dict[int, int] = {}
    for player in pool:
        if player.id in locked:
            locked_by_team[player.team_id] = locked_by_team.get(player.team_id, 0) + 1
    for team_id, count in locked_by_team.items():
        if count > rules.club_limit:
            raise ValueError(
                f"{count} locked players come from team {team_id}, but at most "
                f"{rules.club_limit} players may come from one club"
            )


def infeasible_message(
    pool: Sequence[Player],
    rules: GameRules,
    budget_tenths: int,
    locked: set[int],
    status: int,
) -> str:
    if status != cp_model.INFEASIBLE:
        return (
            f"The solver returned {cp_model.CpSolver().StatusName(status)} without finding a "
            f"squad. Try a longer timeout."
        )

    # The most common cause by far is a budget that cannot buy a legal squad. Price up the
    # cheapest possible one (ignoring the club cap, so it is a true lower bound) to say so
    # concretely rather than just "infeasible".
    cheapest = 0
    for rule in rules.positions:
        costs = sorted(p.cost_tenths for p in pool if p.element_type == rule.element_type)
        cheapest += sum(costs[: rule.squad_select])

    if cheapest > budget_tenths:
        return (
            f"No legal squad fits a budget of £{budget_tenths / 10:.1f}m: the cheapest "
            f"possible {rules.squad_size}-man squad in this pool costs £{cheapest / 10:.1f}m."
        )
    reason = (
        " The locked players may be the cause — together they may not admit a legal squad."
        if locked
        else ""
    )
    return (
        f"No legal squad exists for this pool within £{budget_tenths / 10:.1f}m, despite the "
        f"cheapest {rules.squad_size} players by position costing only £{cheapest / 10:.1f}m. "
        f"The per-club limit of {rules.club_limit} is the likely binding constraint.{reason}"
    )
