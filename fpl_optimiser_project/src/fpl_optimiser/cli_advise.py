"""`cli_advise`: what to do before this week's deadline.

Pulls your actual squad from the FPL API, forecasts the upcoming gameweek with the model the
backtest validated, plans transfers over a rolling horizon, and tells you what to do — the
transfer to make, the eleven to field, who to captain, and whether to play a chip.

It recommends; it does not act. You keep the veto, which matters because the model has no idea
that a manager said something ominous in a press conference an hour ago.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from fpl_optimiser.api import fetch_bootstrap
from fpl_optimiser.backtest.chips import Chip, ChipState, choose_chip
from fpl_optimiser.backtest.history import GameweekRecord
from fpl_optimiser.backtest.planner import DEFAULT_DISCOUNT, DEFAULT_HORIZON, plan_transfers
from fpl_optimiser.backtest.points_model import PointsModel
from fpl_optimiser.backtest.wildcard import wildcard_gain
from fpl_optimiser.live import (
    MODEL_FROM_GAMEWEEK,
    early_season_forecast,
    fetch_squad,
    free_transfers_for,
    injury_availability,
    load_live_history,
    load_previous_season,
    next_gameweek,
    season_label,
)
from fpl_optimiser.model import DEFAULT_BENCH_WEIGHTS, BenchWeights, Squad
from fpl_optimiser.players import Player
from fpl_optimiser.rules import GameRules, TransferRules

logger = logging.getLogger("fpl_optimiser")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fpl_optimiser.cli_advise",
        description="Recommend transfers, the XI, the captain and any chip for the next deadline.",
    )
    parser.add_argument("--team-id", type=int, required=True, help="Your FPL entry id.")
    parser.add_argument(
        "--gameweek",
        type=int,
        default=None,
        help="Gameweek to advise on (default: the next one).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Gameweeks to plan ahead over (default: {DEFAULT_HORIZON}).",
    )
    parser.add_argument("--bench-weight", type=float, default=None, metavar="0..1")
    parser.add_argument(
        "--free-transfers",
        type=int,
        default=None,
        help=(
            "Free transfers you have banked. Derived from your transfer history by default; "
            "pass a number only to override it."
        ),
    )
    parser.add_argument("--no-chips", action="store_true", help="Do not suggest playing a chip.")
    parser.add_argument(
        "--used-chips",
        nargs="*",
        default=[],
        metavar="CHIP",
        help="Chips already used this half: wildcard, freehit, bboost, 3xc.",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Re-fetch player history from the API."
    )
    parser.add_argument(
        "--previous-season",
        default=None,
        metavar="YYYY-YY",
        help=(
            "Season to carry minutes over from when a player has not yet played this one. "
            "Defaults to the season before the live one."
        ),
    )
    parser.add_argument(
        "--model-from",
        type=int,
        default=MODEL_FROM_GAMEWEEK,
        metavar="GW",
        help=(
            "First gameweek to use the model's own forecast. Earlier gameweeks fall back on "
            f"FPL's ep_next (default: {MODEL_FROM_GAMEWEEK}). Set to 1 to use the model "
            "throughout, which measurement says is a bad idea at GW1."
        ),
    )
    parser.add_argument(
        "--no-carryover",
        action="store_true",
        help=(
            "Do not seed the minutes model from last season. Early in a season this leaves it "
            "with nothing to go on and the forecast collapses toward zero."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--log-forecast",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Write this run's expected points for every player, for every gameweek in the "
            "horizon, to DIR/pool_gw<NN>_<season>.json. Scored later by "
            "fpl_optimiser.forecast.calibrate. Logging the whole pool rather than the chosen XI is "
            "deliberate: ~650 predictions a week instead of 11."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def pair_transfers(
    transfers_out: Sequence[Player], transfers_in: Sequence[Player]
) -> list[tuple[Player, Player]]:
    """Match each outgoing player to the incoming one that actually replaced them.

    The planner returns two independent tuples, each sorted by player id, so zipping them pairs
    by nothing at all — it happily showed a forward being replaced by a midfielder, which the
    fixed squad quotas make impossible. Every swap is within a position, so grouping by
    `element_type` recovers the real pairing.

    Where one position has more than one move the pairing *inside* it is presentational: the
    solver swapped a set for a set and has no opinion on which replaced which. Those are matched
    most-expensive-first so the output is at least stable between runs.
    """
    paired: list[tuple[Player, Player]] = []
    by_position: dict[int, list[Player]] = {}
    for player in transfers_in:
        by_position.setdefault(player.element_type, []).append(player)
    for group in by_position.values():
        group.sort(key=lambda p: (-p.cost_tenths, p.id))

    for outgoing in sorted(transfers_out, key=lambda p: (p.element_type, -p.cost_tenths, p.id)):
        replacements = by_position.get(outgoing.element_type)
        if replacements:
            paired.append((outgoing, replacements.pop(0)))
    return paired


def horizon_value(
    points_by_gameweek: Mapping[int, Mapping[int, float]],
    discount: float = DEFAULT_DISCOUNT,
) -> dict[int, float]:
    """Each player's expected points over the whole plan, discounted as the planner discounts.

    This is the quantity the transfer decision is actually made on — not the upcoming
    gameweek's xP, which is all the output used to show. A move can be flat this week and still
    be right, and without this the recommendation looks arbitrary in exactly that case.
    """
    weeks = sorted(points_by_gameweek)
    totals: dict[int, float] = {}
    for index, gameweek in enumerate(weeks):
        weight = discount**index
        for player_id, points in points_by_gameweek[gameweek].items():
            totals[player_id] = totals.get(player_id, 0.0) + points * weight
    return totals


def format_advice(
    squad: Squad,
    transfers_in,
    transfers_out,
    hits: int,
    gameweek: int,
    chip,
    chip_reason: str,
    gameweek_points: Mapping[int, float] | None = None,
    horizon_points: Mapping[int, float] | None = None,
    horizon: Sequence[int] = (),
) -> str:
    """Render the advice.

    `gameweek_points` and `horizon_points` cover the whole pool, not just the chosen squad, so
    the players being sold can be priced too — `squad.expected_points` is keyed on the squad the
    planner picked and by definition does not contain them.
    """
    gameweek_points = gameweek_points or {}
    horizon_points = horizon_points or {}
    lines: list[str] = [f"Gameweek {gameweek}", ""]

    if transfers_in:
        span = f"GW{horizon[0]}-{horizon[-1]}" if len(horizon) > 1 else f"GW{gameweek}"
        lines.append("TRANSFERS")
        total_gain = 0.0
        for out, into in pair_transfers(transfers_out, transfers_in):
            gain = horizon_points.get(into.id, 0.0) - horizon_points.get(out.id, 0.0)
            total_gain += gain
            lines.append(
                f"  OUT  {out.name:<18} {out.team_name:<4} £{out.cost_millions:>4.1f}m"
                f"   {gameweek_points.get(out.id, 0.0):>5.2f} xP"
            )
            lines.append(
                f"  IN   {into.name:<18} {into.team_name:<4} £{into.cost_millions:>4.1f}m"
                f"   {gameweek_points.get(into.id, 0.0):>5.2f} xP"
                f"   {gain:+.1f} over {span}"
            )
        if hits:
            lines.append(
                f"  (a hit of -{hits * 4} points against {total_gain:+.1f} discounted over "
                f"{span} — the model thinks it pays)"
            )
        lines.append("")
    else:
        lines.append("TRANSFERS\n  None — hold and bank the free transfer.\n")

    if chip is not None:
        lines.append(f"CHIP\n  Play the {chip.value} — {chip_reason}.\n")

    lines.append(f"STARTING XI ({squad.formation})")
    for player in squad.starting_xi:
        mark = ""
        if player.id == squad.captain.id:
            mark = "(C)"
        elif player.id == squad.vice_captain.id:
            mark = "(V)"
        lines.append(
            f"  {player.position:<4} {player.name:<18} {player.team_name:<4} "
            f"{squad.expected_points[player.id]:>5.2f} xP {mark}".rstrip()
        )

    lines.append("")
    lines.append("BENCH (in order)")
    for index, player in enumerate(squad.bench, start=1):
        lines.append(
            f"  {index}. {player.position:<4} {player.name:<18} {player.team_name:<4} "
            f"{squad.expected_points[player.id]:>5.2f} xP"
        )

    lines.append("")
    lines.append(
        f"Forecast for the XI (captain doubled): {squad.starting_expected_points:.1f} points"
    )
    return "\n".join(lines)


def forecast_gameweek(
    view: Any,
    history: Any,
    model: PointsModel,
    early: dict[int, float],
    *,
    model_from: int,
) -> dict[int, float]:
    """Expected points for every player in one gameweek, from whichever forecast is better here.

    Before `model_from` the model has no current-season minutes and its ranking is measurably
    worthless (0.009 and 0.027 rank correlation at GW1 across two seasons, against `ep_next`'s
    0.468 and 0.443), so FPL's own forecast stands in. From `model_from` onward a single round
    of real minutes restores it and the model takes over.
    """
    if view.gameweek < model_from:
        logger.info(
            "GW%d: using FPL's ep_next — the model has no current-season data yet",
            view.gameweek,
        )
        return {player.id: early.get(player.id, 0.0) for player in view.pool}

    forecast = model.build(view, history)
    return {player.id: float(forecast(player)) for player in view.pool}


def write_forecast_log(
    directory: Path,
    *,
    payload: Mapping[str, Any],
    gameweek: int,
    views: Sequence[Any],
    points_by_gameweek: Mapping[int, Mapping[int, float]],
    early: Mapping[int, float],
    model_from: int,
) -> Path:
    """Record expected points for every player in the horizon, for later calibration.

    One row per (player, gameweek), carrying the horizon depth so the calibration can ask
    whether a five-weeks-out forecast decays — the planner leans on those weeks as hard as it
    leans on the imminent one. `ep_next` rides along as the free benchmark to beat.

    `ep_next` comes from `early`, not `player.ep_next`: the Player objects on a view's pool are
    rebuilt from history and carry 0.0 there, which would log a benchmark of all zeros. It is
    FPL's forecast for the *next* gameweek only, so it is meaningful at horizon depth 0 and is
    the same number at every other depth — the head-to-head belongs on depth 0 alone.
    """
    season = season_label(payload)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"pool_gw{gameweek:02d}_{season}.json"

    rows: list[dict[str, Any]] = []
    for view in views:
        forecast = points_by_gameweek[view.gameweek]
        for player in view.pool:
            rows.append(
                {
                    "id": player.id,
                    "name": player.name,
                    "position": player.position,
                    "team": player.team_name,
                    "gameweek": view.gameweek,
                    "horizon_depth": view.gameweek - gameweek,
                    "xp": round(float(forecast.get(player.id, 0.0)), 4),
                    "ep_next": round(float(early.get(player.id, 0.0)), 4),
                    "cost_tenths": player.cost_tenths,
                    "availability": round(float(player.availability), 3),
                    "fixtures": view.fixture_count(player.id),
                }
            )

    destination.write_text(
        json.dumps(
            {
                "season": season,
                "forecast_for_gameweek": gameweek,
                "known_through_gameweek": gameweek - 1,
                "model_from": model_from,
                "source": "cli_advise --log-forecast",
                "rows": rows,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        payload = fetch_bootstrap(refresh=args.refresh)
        rules = GameRules.from_bootstrap(payload)
        transfer_rules = TransferRules.from_bootstrap(payload)

        gameweek = args.gameweek or next_gameweek(payload)
        if gameweek is None:
            raise SystemExit(
                "fpl-advise: there is no upcoming gameweek — the season is over. The advisor "
                "will work again when the next season's fixtures are published. To see it run "
                "against a past gameweek, pass --gameweek."
            )

        # Seed the minutes model from last season. Without this, a player with no completed
        # gameweeks forecasts a zero chance of starting — which at GW1 is every player alive.
        previous: list[GameweekRecord] = []
        if not args.no_carryover:
            previous = load_previous_season(
                payload, season=args.previous_season, refresh=args.refresh
            )

        history = load_live_history(
            payload,
            rules,
            gameweek=gameweek,
            horizon=args.horizon,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
            previous_season=previous,
        )
        squad_ids, bank, purchase_prices = fetch_squad(args.team_id, gameweek - 1, payload)

        # The planner prices every hit against this, so a hand-typed default is a silently
        # wrong answer, not a missing one. It is derivable from public history — so derive it.
        if args.free_transfers is None:
            free_transfers = free_transfers_for(args.team_id, gameweek, transfer_rules)
            logger.info(
                "You have %d free transfer(s) banked for GW%d.",
                free_transfers,
                gameweek,
            )
        elif args.free_transfers < 0:
            raise ValueError(
                f"--free-transfers must be >= 0, got {args.free_transfers}. Leave it off to "
                f"derive the real count from your transfer history."
            )
        else:
            free_transfers = args.free_transfers

        # Injury flags — the one thing the advisor knows that its own backtest never could.
        injuries = injury_availability(payload)
        flagged = [p for p in squad_ids if injuries.get(p, 1.0) < 1.0]

        weeks = [gw for gw in history.gameweeks if gw >= gameweek][: args.horizon]
        # Every view in a live horizon knows the same thing: everything through the last played
        # gameweek. Only the fixtures differ. Defaulting `known_through` here would read the
        # unplayed weeks as completed ones in which nobody played a minute.
        views = [
            replace(
                history.view_at(gw, known_through=gameweek - 1),
                injury_availability=injuries,
            )
            for gw in weeks
        ]
        if len(weeks) < args.horizon:
            logger.warning(
                "Planning over %d gameweek(s), not the %d requested — the season ends first.",
                len(weeks),
                args.horizon,
            )

        model = PointsModel()
        early = early_season_forecast(payload, rules)
        points_by_gameweek = {
            view.gameweek: forecast_gameweek(
                view, history, model, early, model_from=args.model_from
            )
            for view in views
        }

        if args.log_forecast is not None:
            written = write_forecast_log(
                args.log_forecast,
                payload=payload,
                gameweek=gameweek,
                views=views,
                points_by_gameweek=points_by_gameweek,
                early=early,
                model_from=args.model_from,
            )
            logger.info("Forecast for %d players logged to %s", len(views[0].pool), written)

        decision = plan_transfers(
            views,
            points_by_gameweek,
            rules,
            transfer_rules,
            current_squad_ids=set(squad_ids),
            purchase_prices=purchase_prices,
            bank_tenths=bank,
            free_transfers=free_transfers,
            bench_weights=(
                BenchWeights.uniform(args.bench_weight)
                if args.bench_weight is not None
                else DEFAULT_BENCH_WEIGHTS
            ),
        )

        chip = None
        chip_reason = ""
        if not args.no_chips:
            state = ChipState()
            for name in args.used_chips:
                state.play(Chip(name), gameweek)
            chip_decision = choose_chip(
                gameweek,
                list(history.gameweeks),
                state,
                bench_expected_points=decision.squad.bench_expected_points,
                captain_expected_points=decision.squad.expected_points[decision.squad.captain.id],
                squad_with_fixtures=sum(
                    1 for p in decision.squad.players if views[0].fixture_count(p.id) > 0
                ),
                wildcard_gain=wildcard_gain(
                    views,
                    points_by_gameweek,
                    rules,
                    transfer_rules,
                    current_squad_ids=set(squad_ids),
                    purchase_prices=purchase_prices,
                    bank_tenths=bank,
                    free_transfers=free_transfers,
                    bench_weights=(
                        BenchWeights.uniform(args.bench_weight)
                        if args.bench_weight is not None
                        else DEFAULT_BENCH_WEIGHTS
                    ),
                ),
            )
            chip, chip_reason = chip_decision.chip, chip_decision.reason

    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"fpl-advise: {exc}") from exc

    if flagged:
        logger.warning(
            "%d of your squad carry an injury flag; the model has discounted them.",
            len(flagged),
        )

    sys.stdout.write(
        format_advice(
            decision.squad,
            decision.transfers_in,
            decision.transfers_out,
            decision.hits,
            gameweek,
            chip,
            chip_reason,
            gameweek_points=points_by_gameweek[gameweek],
            horizon_points=horizon_value(points_by_gameweek),
            horizon=weeks,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
