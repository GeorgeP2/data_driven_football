"""`cli_evaluate`: score the forecast itself on ~30,000 predictions, not one season total."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fpl_optimiser.api import fetch_bootstrap
from fpl_optimiser.backtest.carryover import carry_over_between_seasons
from fpl_optimiser.backtest.evaluate import evaluate_season, format_evaluation
from fpl_optimiser.backtest.history import load_season
from fpl_optimiser.backtest.points_model import PointsModel
from fpl_optimiser.backtest.season import SeasonHistory
from fpl_optimiser.rules import GameRules, rules_for_season

logger = logging.getLogger("fpl_optimiser")

PREVIOUS_SEASON = {
    "2025-26": "2024-25",
    "2024-25": "2023-24",
    "2023-24": "2022-23",
    "2022-23": "2021-22",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fpl_optimiser.cli_evaluate",
        description=(
            "Measure forecast quality directly: rank correlation against actual points, "
            "minutes-model calibration, and a head-to-head against FPL's own ep_next."
        ),
    )
    parser.add_argument("--season", default="2024-25")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        # The squad shape (15 players, 2/5/5/3, £100m) has not changed in years, so the live
        # payload is a safe source for it. The rules that *did* change come from rules_for_season.
        payload = fetch_bootstrap()
        game_rules = GameRules.from_bootstrap(payload)
        season_rules = rules_for_season(args.season)

        records = load_season(args.season, cache_dir=args.cache_dir)
        previous: list = []
        earlier = PREVIOUS_SEASON.get(args.season)
        if earlier is not None:
            try:
                previous = carry_over_between_seasons(
                    load_season(earlier, cache_dir=args.cache_dir), records
                )
            except (RuntimeError, ValueError) as exc:
                logger.warning("No %s data for the cold start: %s", earlier, exc)

        history = SeasonHistory(records, game_rules, previous_season=previous)
        evaluation = evaluate_season(
            history,
            args.season,
            PointsModel(defensive_contribution=season_rules.defensive_contribution),
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"fpl-evaluate: {exc}") from exc

    sys.stdout.write(format_evaluation(evaluation) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
