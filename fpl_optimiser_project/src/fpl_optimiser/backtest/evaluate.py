"""Score the *forecast*, not just the season.

A season simulation produces one number. Ten hyperparameters were tuned against that one number,
which is not model selection — it is curve-fitting, and it is why the current system's 20-point
margin over a decent human cannot be trusted.

But the forecast underneath makes a prediction for every player in every gameweek: roughly thirty
thousand of them, each with a known answer. That is the signal we should have been steering by all
along. It is three orders of magnitude more evidence, it takes seconds rather than ten minutes to
compute, and — unlike a season total — an improvement in it cannot be an accident of solver
tie-breaks or a lucky captain pick.

Three things are measured here:

- **Rank correlation** between forecast and actual points. This is the metric that matters, more
  than absolute error: the optimiser never needs to know that Haaland will score 7.3, only that he
  will out-score Wissa. A model with a large bias but the right ordering picks the right squad.
- **Calibration of the minutes model** — the component the backtest identified as the biggest
  lever, and one that has never been checked. When it says "70% chance of starting", do 70% of
  them start? An over-confident minutes model is how a squad ends up full of players who never
  take the pitch.
- **FPL's own forecast**, as an external benchmark. The archive snapshots `ep_next` weekly. If our
  model cannot out-forecast the one FPL ships for free, we have built nothing.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from fpl_optimiser.backtest.history import GameweekRecord
from fpl_optimiser.backtest.points_model import PointsModel, forecast_minutes
from fpl_optimiser.backtest.season import SeasonHistory

logger = logging.getLogger(__name__)

POSITION_NAMES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

MISSING_FORECAST_TOLERANCE = 0.99
"""If this fraction of a gameweek's FPL forecasts are exactly zero, treat the whole gameweek as a
failed snapshot rather than a forecast. The 2025-26 archive has 27 such gameweeks; scoring them as
though FPL predicted nothing would flatter us enormously."""


@dataclass
class ForecastScore:
    """How good a forecast was, across every player and gameweek it covered."""

    name: str
    observations: int = 0
    absolute_error: float = 0.0
    _pairs: list[tuple[float, float]] = field(default_factory=list)

    def add(self, forecast: float, actual: float) -> None:
        self.observations += 1
        self.absolute_error += abs(forecast - actual)
        self._pairs.append((forecast, actual))

    @property
    def mean_absolute_error(self) -> float:
        return self.absolute_error / self.observations if self.observations else 0.0

    @property
    def rank_correlation(self) -> float:
        """Spearman: does the forecast get the *ordering* right?"""
        if len(self._pairs) < 2:
            return 0.0
        forecasts = _ranks([f for f, _ in self._pairs])
        actuals = _ranks([a for _, a in self._pairs])
        return _pearson(forecasts, actuals)

    @property
    def correlation(self) -> float:
        if len(self._pairs) < 2:
            return 0.0
        return _pearson([f for f, _ in self._pairs], [a for _, a in self._pairs])


@dataclass
class Calibration:
    """Is a probability honest? Bucketed predicted-vs-observed, plus a Brier score."""

    buckets: int = 10
    _sum_predicted: list[float] = field(default_factory=list)
    _sum_observed: list[float] = field(default_factory=list)
    _counts: list[int] = field(default_factory=list)
    squared_error: float = 0.0
    observations: int = 0

    def __post_init__(self) -> None:
        self._sum_predicted = [0.0] * self.buckets
        self._sum_observed = [0.0] * self.buckets
        self._counts = [0] * self.buckets

    def add(self, predicted: float, happened: bool) -> None:
        index = min(self.buckets - 1, int(predicted * self.buckets))
        self._sum_predicted[index] += predicted
        self._sum_observed[index] += 1.0 if happened else 0.0
        self._counts[index] += 1
        self.squared_error += (predicted - (1.0 if happened else 0.0)) ** 2
        self.observations += 1

    @property
    def brier_score(self) -> float:
        """Mean squared error of the probability. Lower is better; 0.25 is a coin flip."""
        return self.squared_error / self.observations if self.observations else 0.0

    def reliability(self) -> list[tuple[float, float, int]]:
        """(mean predicted, observed rate, count) per bucket. Perfect calibration is y = x."""
        rows = []
        for index in range(self.buckets):
            count = self._counts[index]
            if not count:
                continue
            rows.append(
                (
                    self._sum_predicted[index] / count,
                    self._sum_observed[index] / count,
                    count,
                )
            )
        return rows


@dataclass
class Evaluation:
    """Everything measured in one pass over a season."""

    season: str
    overall: ForecastScore
    by_position: dict[int, ForecastScore]
    minutes: Calibration
    fpl_benchmark: ForecastScore | None
    """FPL's own forecast, scored on exactly the same rows as ours — or None if the archive never
    captured it for this season."""

    ours_on_benchmark_rows: ForecastScore | None
    """Our forecast, restricted to the rows where FPL made one. The only fair comparison."""


def evaluate_season(
    history: SeasonHistory,
    season: str,
    model: PointsModel,
    *,
    gameweeks: Sequence[int] | None = None,
) -> Evaluation:
    """Forecast every player in every gameweek, then compare against what happened.

    Only players with a fixture are scored. A blank gameweek is a guaranteed zero for both the
    model and reality, and including those rows would inflate every correlation with thousands of
    free correct answers.
    """
    weeks = list(history.gameweeks if gameweeks is None else gameweeks)

    overall = ForecastScore("points-model")
    by_position = {t: ForecastScore(POSITION_NAMES[t]) for t in POSITION_NAMES}
    minutes = Calibration()
    fpl = ForecastScore("fpl-ep_next")
    ours_paired = ForecastScore("points-model")

    for gameweek in weeks:
        view = history.view_at(gameweek)
        actuals = history.actuals_at(gameweek)
        forecast = model.build(view, history)

        fpl_usable = _fpl_forecast_usable(actuals)

        for player in view.pool:
            if view.fixture_count(player.id) == 0:
                continue
            record = actuals[player.id]
            predicted = float(forecast(player))
            actual = float(record.total_points)

            overall.add(predicted, actual)
            by_position[player.element_type].add(predicted, actual)

            # Did they start? Scored per fixture, since a double gives two chances to.
            starts = forecast_minutes(view.stats(player.id), gameweek)
            for _ in range(record.fixture_count):
                minutes.add(starts.probability_starts, record.started > 0)

            if fpl_usable and record.fpl_expected_points is not None:
                fpl.add(record.fpl_expected_points, actual)
                ours_paired.add(predicted, actual)

    has_benchmark = fpl.observations > 0
    return Evaluation(
        season=season,
        overall=overall,
        by_position=by_position,
        minutes=minutes,
        fpl_benchmark=fpl if has_benchmark else None,
        ours_on_benchmark_rows=ours_paired if has_benchmark else None,
    )


def _fpl_forecast_usable(actuals: dict[int, GameweekRecord]) -> bool:
    """Did the archive actually capture FPL's forecast this gameweek?

    It writes a literal 0.0 for every player when the weekly scrape failed, which is
    indistinguishable from "FPL predicted nothing" unless you look at the whole gameweek at once.
    Scoring those weeks would hand us a fraudulent win over the benchmark.
    """
    values = [record.fpl_expected_points for record in actuals.values()]
    known = [v for v in values if v is not None]
    if not known:
        return False
    zeroes = sum(1 for v in known if v == 0.0)
    return zeroes / len(known) < MISSING_FORECAST_TOLERANCE


def _ranks(values: list[float]) -> list[float]:
    """Fractional ranks, averaging ties — which matter here, since most players score 0 or 1."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        average = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[order[position]] = average
        index = stop + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = math.sqrt(variance_x * variance_y)
    return covariance / denominator if denominator else 0.0


def format_evaluation(evaluation: Evaluation) -> str:
    lines = [f"Forecast quality — {evaluation.season}", ""]

    overall = evaluation.overall
    lines.append(
        f"  points forecast   n={overall.observations:>6}  "
        f"rank-corr {overall.rank_correlation:>5.3f}  "
        f"corr {overall.correlation:>5.3f}  MAE {overall.mean_absolute_error:>5.2f}"
    )
    for _element_type, score in sorted(evaluation.by_position.items()):
        if not score.observations:
            continue
        lines.append(
            f"    {score.name:<14} n={score.observations:>6}  "
            f"rank-corr {score.rank_correlation:>5.3f}  "
            f"corr {score.correlation:>5.3f}  MAE {score.mean_absolute_error:>5.2f}"
        )

    lines.append("")
    lines.append(
        f"  minutes model     n={evaluation.minutes.observations:>6}  "
        f"Brier {evaluation.minutes.brier_score:.4f}  (0.25 = a coin flip; lower is better)"
    )
    lines.append("    P(start) says ... they actually started")
    for predicted, observed, count in evaluation.minutes.reliability():
        drift = observed - predicted
        flag = "  <-- over-confident" if drift < -0.08 else ""
        lines.append(f"      {predicted:>5.0%}  ->  {observed:>5.0%}   (n={count:>5}){flag}")

    lines.append("")
    if evaluation.fpl_benchmark is None or evaluation.ours_on_benchmark_rows is None:
        lines.append(
            "  FPL benchmark     unavailable — the archive did not capture ep_next this season."
        )
    else:
        theirs = evaluation.fpl_benchmark
        ours = evaluation.ours_on_benchmark_rows
        lines.append(f"  Head to head, on the {ours.observations} rows FPL also forecast:")
        lines.append(
            f"    ours            rank-corr {ours.rank_correlation:>5.3f}   "
            f"MAE {ours.mean_absolute_error:>5.2f}"
        )
        lines.append(
            f"    FPL's ep_next   rank-corr {theirs.rank_correlation:>5.3f}   "
            f"MAE {theirs.mean_absolute_error:>5.2f}"
        )
        delta = ours.rank_correlation - theirs.rank_correlation
        verdict = "we beat it" if delta > 0 else "WE LOSE TO IT"
        lines.append(f"    -> {verdict} by {delta:+.3f} on rank correlation")

    return "\n".join(lines)
