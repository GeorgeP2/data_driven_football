"""Command-line entrypoint: fetch the FPL data, solve, print the squad."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fpl_optimiser.api import fetch_bootstrap
from fpl_optimiser.expected_points import EpNextProvider
from fpl_optimiser.model import DEFAULT_BENCH_WEIGHTS, BenchWeights
from fpl_optimiser.optimise import DEFAULT_SOLVER_TIMEOUT_SECONDS, Squad, optimise_squad
from fpl_optimiser.players import parse_players
from fpl_optimiser.rules import GameRules

logger = logging.getLogger("fpl_optimiser")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fpl_optimiser.cli",
        description=(
            "Pick the highest expected-points Fantasy Premier League squad that satisfies "
            "the budget, position quotas and per-club limit."
        ),
    )
    parser.add_argument(
        "--budget",
        type=_budget_millions,
        default=None,
        metavar="MILLIONS",
        help="Budget in millions, e.g. 100.0 (default: the game's own £100.0m).",
    )
    parser.add_argument(
        "--bench-weight",
        type=_unit_interval,
        default=None,
        metavar="0..1",
        help=(
            "How much bench players' expected points count towards the objective "
            "Flatten the per-slot bench weights to one number. 0 lets the solver field a worthless "
            "to spend more on the XI."
        ),
    )
    parser.add_argument(
        "--exclude",
        type=int,
        nargs="*",
        default=[],
        metavar="PLAYER_ID",
        help="Player ids to keep out of the squad.",
    )
    parser.add_argument(
        "--lock",
        type=int,
        nargs="*",
        default=[],
        metavar="PLAYER_ID",
        help="Player ids that must be in the squad.",
    )
    parser.add_argument(
        "--ignore-availability",
        action="store_true",
        help=(
            "Do not discount expected points by injury/doubt. Answers 'who would I pick in a "
            "healthy world?' — do not use it to pick a real squad."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch bootstrap-static instead of using the cached copy.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Where to cache the API payload (default: ~/.cache/fpl_optimiser/).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_SOLVER_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=f"Solver time limit (default: {DEFAULT_SOLVER_TIMEOUT_SECONDS}s).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log debug detail.",
    )
    return parser


def _budget_millions(raw: str) -> int:
    """Parse a budget in millions into the API's integer tenths."""
    try:
        millions = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"budget must be a number of millions, e.g. 100.0 — got {raw!r}"
        ) from None
    if millions <= 0:
        raise argparse.ArgumentTypeError(f"budget must be positive, got {millions}")
    tenths = round(millions * 10)
    if abs(millions * 10 - tenths) > 1e-6:
        raise argparse.ArgumentTypeError(f"budget is only meaningful to 0.1m, got {millions}")
    return tenths


def _unit_interval(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {raw!r}") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0.0 and 1.0, got {value}")
    return value


def format_squad(squad: Squad) -> str:
    """Render the squad as a team sheet."""
    points = squad.expected_points
    lines: list[str] = []

    def row(player, marker: str = "") -> str:
        return (
            f"  {player.position:<4} {player.name:<20} {player.team_name:<4} "
            f"£{player.cost_millions:>4.1f}m  {points[player.id]:>5.2f} xP {marker}".rstrip()
        )

    lines.append(f"Starting XI ({squad.formation})")
    for player in squad.starting_xi:
        marker = ""
        if player.id == squad.captain.id:
            marker = "(C)"
        elif player.id == squad.vice_captain.id:
            marker = "(V)"
        lines.append(row(player, marker))

    lines.append("")
    lines.append("Bench (in order)")
    for index, player in enumerate(squad.bench, start=1):
        lines.append(row(player, f"[{index}]"))

    lines.append("")
    lines.append(
        f"Cost £{squad.total_cost_millions:.1f}m  |  "
        f"XI expected points (captain doubled): {squad.starting_expected_points:.2f}  |  "
        f"bench: {squad.bench_expected_points:.2f}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        payload = fetch_bootstrap(cache_path=args.cache_path, refresh=args.refresh)
        rules = GameRules.from_bootstrap(payload)
        players = parse_players(payload, rules)
        squad = optimise_squad(
            players,
            rules,
            EpNextProvider(apply_availability=not args.ignore_availability),
            budget_tenths=args.budget,
            bench_weights=(
                BenchWeights.uniform(args.bench_weight)
                if args.bench_weight is not None
                else DEFAULT_BENCH_WEIGHTS
            ),
            excluded_player_ids=args.exclude,
            locked_player_ids=args.lock,
            timeout_seconds=args.timeout,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"fpl-optimise: {exc}") from exc

    logger.info(
        "Optimised %d players from %d clubs using the '%s' expected-points model",
        len(players),
        len({p.team_id for p in players}),
        EpNextProvider().name,
    )
    sys.stdout.write(format_squad(squad) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
