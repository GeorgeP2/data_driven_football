"""`cli_backtest`: play a completed season with each baseline and report the scores."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fpl_optimiser.api import fetch_bootstrap
from fpl_optimiser.backtest.carryover import carry_over_between_seasons
from fpl_optimiser.backtest.engine import run_season
from fpl_optimiser.backtest.history import load_season
from fpl_optimiser.backtest.providers import DEFAULT_PROVIDERS
from fpl_optimiser.backtest.report import format_results
from fpl_optimiser.backtest.season import SeasonHistory
from fpl_optimiser.model import DEFAULT_BENCH_WEIGHTS, BenchWeights
from fpl_optimiser.rules import GameRules, TransferRules

logger = logging.getLogger("fpl_optimiser")

_PREVIOUS_SEASON = {"2025-26": "2024-25", "2024-25": "2023-24"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fpl_optimiser.cli_backtest",
        description=(
            "Simulate a completed FPL season under the real transfer rules, comparing "
            "baseline expected-points models against a perfect-foresight ceiling."
        ),
    )
    parser.add_argument(
        "--season",
        default="2025-26",
        help="Season to backtest, e.g. 2025-26 (default: %(default)s).",
    )
    parser.add_argument(
        "--bench-weight",
        type=_unit_interval,
        default=None,
        metavar="0..1",
        help=(
            "Flatten the per-slot bench weights to a single number (default: use the tuned shape). "
            "Auto-subs are simulated, so this is the knob the backtest exists to tune."
        ),
    )
    parser.add_argument(
        "--gameweeks",
        type=int,
        default=None,
        metavar="N",
        help="Simulate only the first N gameweeks. Useful for a quick check.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download the season archive instead of using the cached copy.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Where season archives are cached (default: ~/.cache/fpl_optimiser/seasons).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Log debug detail.")
    return parser


def _unit_interval(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {raw!r}") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0.0 and 1.0, got {value}")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        # The squad and transfer rules still come from the live API: they are properties of the
        # game, not of the season, and are not carried in the historical archive.
        payload = fetch_bootstrap()
        rules = GameRules.from_bootstrap(payload)
        transfer_rules = TransferRules.from_bootstrap(payload)

        records = load_season(args.season, cache_dir=args.cache_dir, refresh=args.refresh)
        previous: list = []
        previous_name = _PREVIOUS_SEASON.get(args.season)
        if previous_name is not None:
            try:
                previous = carry_over_between_seasons(
                    load_season(previous_name, cache_dir=args.cache_dir), records
                )
            except (RuntimeError, ValueError) as exc:
                # Without it, GW1 is picked blind. Worth saying so out loud.
                logger.warning(
                    "No %s data for the GW1 cold start (%s); the opening squad will be "
                    "picked with no prior information.",
                    previous_name,
                    exc,
                )

        history = SeasonHistory(records, rules, previous_season=previous)
        weeks = history.gameweeks if args.gameweeks is None else history.gameweeks[: args.gameweeks]

        results = []
        for provider in DEFAULT_PROVIDERS:
            logger.info("Simulating %s over %d gameweeks...", provider.name, len(weeks))
            results.append(
                run_season(
                    history,
                    rules,
                    transfer_rules,
                    provider,
                    bench_weights=(
                        BenchWeights.uniform(args.bench_weight)
                        if args.bench_weight is not None
                        else DEFAULT_BENCH_WEIGHTS
                    ),
                    gameweeks=weeks,
                )
            )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"fpl-backtest: {exc}") from exc

    sys.stdout.write(format_results(results, args.season) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
