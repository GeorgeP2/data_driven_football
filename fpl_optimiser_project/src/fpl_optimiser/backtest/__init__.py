"""Backtest a squad-selection strategy over a completed FPL season."""

from fpl_optimiser.backtest.engine import SeasonResult, run_season
from fpl_optimiser.backtest.history import GameweekRecord, load_season
from fpl_optimiser.backtest.providers import (
    DEFAULT_PROVIDERS,
    PerfectForesight,
    PointsPerGame,
    PreviousSeason,
    ProviderFactory,
    RandomSquad,
    TrailingForm,
)
from fpl_optimiser.backtest.report import format_results
from fpl_optimiser.backtest.scoring import GameweekScore, score_gameweek
from fpl_optimiser.backtest.season import GameweekView, SeasonHistory
from fpl_optimiser.backtest.transfers import TransferDecision, optimise_transfers

__all__ = [
    "DEFAULT_PROVIDERS",
    "GameweekRecord",
    "GameweekScore",
    "GameweekView",
    "PerfectForesight",
    "PointsPerGame",
    "PreviousSeason",
    "ProviderFactory",
    "RandomSquad",
    "SeasonHistory",
    "SeasonResult",
    "TrailingForm",
    "TransferDecision",
    "format_results",
    "load_season",
    "optimise_transfers",
    "run_season",
    "score_gameweek",
]
