"""Probability of winning the title, finishing in the top four and being relegated, for every
team in the table.

Each outcome has its own XGBoost model on six inputs: points, gameweek, league position, goal
difference, last season's finish and the distance to the line that matters (gap to 17th for
relegation, margin to the title or the top four otherwise). A table's probabilities are rescaled
to add up to the number of teams the outcome happens to. The models are fitted on every completed
20-team Premier League season (1995-96 onwards) and applied to the current table. A two-input
logistic regression answers the simpler question of what a points total at a given gameweek has
meant for relegation. All are checked by leaving one season out at a time against a baseline
that gives every team the base rate.

Needs the ``ml`` extra. Run from the project folder:
    PYTHONPATH=src python -m football_stats.relegation
    PYTHONPATH=src python -m football_stats.relegation --points 8 --gameweek 11
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from xgboost import XGBClassifier

from football_stats.provisional import fetch_provisional_results
from football_stats.standings import OUTPUTS, load_results, standings, table_now

TEAMS = 20
RELEGATED = 3
BASE_RATE = RELEGATED / TEAMS

# Inputs of the relegation model, each with the direction it is allowed to move the chance of
# relegation when the others are held fixed: -1 lowers it, +1 raises it, 0 is left free. With the
# gap to 17th known, more points or more matches played no longer push in one direction.
FEATURES = {
    "points": 0,
    "gameweek": 0,
    "position": 1,
    "goal_difference": -1,
    "gap_to_17th": -1,
    "last_season_finish": 1,
}
PROMOTED = 21  # last season's finish given to a team that was not in the division


@dataclass(frozen=True)
class Outcome:
    """A way a season can end: the final positions that count, and the model inputs for it."""

    name: str  # prefix of the probability column, e.g. "relegation"
    label: str  # true/false column in the training data, e.g. "relegated"
    first: int
    last: int
    features: dict[str, int]

    @property
    def places(self) -> int:
        return self.last - self.first + 1

    @property
    def base_rate(self) -> float:
        return self.places / TEAMS

    @property
    def column(self) -> str:
        return f"{self.name}_probability"


def _top(margin: str) -> dict[str, int]:
    return {
        "points": 0,
        "gameweek": 0,
        "position": -1,
        "goal_difference": 1,
        margin: 1,
        "last_season_finish": -1,
    }


TITLE = Outcome("title", "champions", 1, 1, _top("title_margin"))
TOP4 = Outcome("top4", "top4", 1, 4, _top("top4_margin"))
RELEGATION = Outcome("relegation", "relegated", TEAMS - RELEGATED + 1, TEAMS, FEATURES)
OUTCOMES = [TITLE, TOP4, RELEGATION]


def training_data(standings: pd.DataFrame) -> pd.DataFrame:
    """Rows from completed 20-team seasons, with a true/false column per outcome: ``champions``
    (finished 1st), ``top4`` and ``relegated`` (bottom three). The two 22-team seasons are left
    out: they ran to 42 gameweeks."""
    n_teams = standings.groupby("season")["team"].transform("nunique")
    df = standings[(n_teams == TEAMS) & standings["final_position"].notna()].copy()
    for outcome in OUTCOMES:
        df[outcome.label] = df["final_position"].between(outcome.first, outcome.last)
    return fill_promoted(df)


def fill_promoted(table: pd.DataFrame) -> pd.DataFrame:
    """Give teams with no finish last season (promoted teams) a place below everyone else."""
    return table.assign(last_season_finish=table["last_season_finish"].fillna(PROMOTED))


def rescale(probabilities, places: int) -> np.ndarray:
    """Shift the probabilities of one table, on the log-odds scale, until they add up to
    ``places``: the number of teams that outcome happens to. Order is kept and every value stays
    between 0 and 1."""
    p = np.clip(np.asarray(probabilities, float), 1e-9, 1 - 1e-9)
    if not 0 < places < len(p):
        return p
    log_odds = np.log(p / (1 - p))
    shift = brentq(lambda c: expit(log_odds + c).sum() - places, -50, 50)
    return expit(log_odds + shift)


def rescale_by_table(data: pd.DataFrame, probabilities: pd.Series, places: int) -> pd.Series:
    """``rescale`` within each season and gameweek of ``data``."""
    tables = [data["season"], data["gameweek"]]
    return probabilities.groupby(tables).transform(lambda p: rescale(p, places))


@dataclass(frozen=True)
class RelegationModel:
    """P(relegated) = expit(intercept + points * per_point + gameweek * per_gameweek)."""

    intercept: float
    per_point: float
    per_gameweek: float

    @classmethod
    def fit(cls, points, gameweek, relegated) -> RelegationModel:
        """Maximum-likelihood fit by Newton's method."""
        y = np.asarray(relegated, dtype=float)
        x = np.column_stack(
            [np.ones(len(y)), np.asarray(points, float), np.asarray(gameweek, float)]
        )
        w = np.zeros(3)
        for _ in range(100):
            p = expit(x @ w)
            step = np.linalg.solve((x * (p * (1 - p))[:, None]).T @ x, x.T @ (p - y))
            w -= step
            if np.abs(step).max() < 1e-10:
                break
        return cls(*map(float, w))

    def predict(self, points, gameweek):
        """Probability of relegation; takes scalars or arrays."""
        z = (
            self.intercept
            + self.per_point * np.asarray(points, float)
            + self.per_gameweek * np.asarray(gameweek, float)
        )
        return expit(z)


def log_loss(y, p) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def xgboost_classifier(monotone=None, **extra) -> XGBClassifier:
    """XGBoost with the project's fixed settings; ``monotone`` is one direction per feature."""
    if monotone is not None:
        extra["monotone_constraints"] = tuple(monotone)
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=20,
        subsample=0.8,
        random_state=0,
        **extra,
    )


def fit_xgboost(data: pd.DataFrame, outcome: Outcome = RELEGATION) -> XGBClassifier:
    """The main model for ``outcome``, fitted on ``data`` (see ``training_data``)."""
    model = xgboost_classifier(list(outcome.features.values()))
    x = data[list(outcome.features)].to_numpy(float)
    return model.fit(x, data[outcome.label].to_numpy(int))


def predict_xgboost(
    model: XGBClassifier, rows: pd.DataFrame, outcome: Outcome = RELEGATION
) -> np.ndarray:
    """Probability of ``outcome`` for each row of a standings table."""
    return model.predict_proba(rows[list(outcome.features)].to_numpy(float))[:, 1]


def xgboost_fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, outcome: Outcome = RELEGATION
) -> np.ndarray:
    return predict_xgboost(fit_xgboost(train, outcome), test, outcome)


def logistic_fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = RelegationModel.fit(train["points"], train["gameweek"], train["relegated"])
    return model.predict(test["points"], test["gameweek"])


def cross_validate(
    data: pd.DataFrame, fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]
) -> pd.Series:
    """Leave one season out: fit on the rest, predict the season held out. Returns those
    out-of-sample probabilities, indexed like ``data``."""
    predicted = pd.Series(np.nan, index=data.index)
    for season in data["season"].unique():
        held_out = data["season"] == season
        predicted[held_out] = fit_predict(data[~held_out], data[held_out])
    return predicted


def scores(
    data: pd.DataFrame, predictions: dict[str, pd.Series], outcome: Outcome = RELEGATION
) -> pd.DataFrame:
    """Log loss and Brier score of each set of held-out predictions and of the base rate."""
    y = data[outcome.label]
    base = {f"base rate ({outcome.base_rate:.0%})": np.full(len(data), outcome.base_rate)}
    candidates = {**base, **predictions}
    rows = {
        name: {"log_loss": log_loss(y, p), "brier": brier(y, p)} for name, p in candidates.items()
    }
    return pd.DataFrame(rows).T


def probability_table(model: RelegationModel, gameweeks, points) -> pd.DataFrame:
    """Relegation probability in percent: one row per points total, one column per gameweek."""
    grid = model.predict(np.asarray(points)[:, None], np.asarray(gameweeks)[None, :])
    table = pd.DataFrame(100 * grid, index=list(points), columns=list(gameweeks)).round(0)
    return table.astype(int).rename_axis(index="points", columns="gameweek")


def predict_table(data: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Fit a model per outcome on ``data`` (see ``training_data``) and score every team in
    ``table`` (see ``standings.table_now``): one probability column per outcome, rescaled so
    each adds up to the number of teams it happens to."""
    table = fill_promoted(table)
    out = table.drop(columns="final_position")
    for outcome in OUTCOMES:
        raw = predict_xgboost(fit_xgboost(data, outcome), table, outcome)
        out[outcome.column] = rescale(raw, outcome.places)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--points", type=int, help="points total to look up (logistic model)")
    ap.add_argument("--gameweek", type=int, help="matches played, 1 to 38")
    ap.add_argument(
        "--provisional",
        action="store_true",
        help="add finished matches football-data.co.uk has not published yet (from the FPL API)",
    )
    ap.add_argument("--out", type=Path, default=OUTPUTS / "table_probabilities.csv")
    args = ap.parse_args()
    if (args.points is None) != (args.gameweek is None):
        ap.error("--points and --gameweek go together")

    results = load_results()
    data = training_data(standings(results))

    if args.points is not None:
        model = RelegationModel.fit(data["points"], data["gameweek"], data["relegated"])
        p = float(model.predict(args.points, args.gameweek))
        print(f"{args.points} points after {args.gameweek} matches: {p:.0%} chance of relegation")
        return

    print(f"Fitted on {data['season'].nunique()} seasons, {len(data):,} rows")
    print("\nHeld-out seasons (lower is better):")
    for outcome in OUTCOMES:
        raw = cross_validate(data, partial(xgboost_fit_predict, outcome=outcome))
        held_out = {
            "xgboost": raw,
            "xgboost, rescaled": rescale_by_table(data, raw, outcome.places),
        }
        if outcome is RELEGATION:
            held_out["logistic: points + gameweek"] = cross_validate(data, logistic_fit_predict)
        print(f"\n{outcome.name}")
        print(scores(data, held_out, outcome).round(4))

    season = results["season"].max()
    latest = results[results["season"] == season]
    if args.provisional:
        extra = fetch_provisional_results(latest)
        print(f"\nProvisional results added ({len(extra)}):")
        for match in extra.itertuples():
            print(f"  {match.date:%a %d %b}  {match.home} {match.hg}-{match.ag} {match.away}")
        latest = pd.concat([latest, extra], ignore_index=True)
    table = table_now(pd.concat([results[results["season"] != season], latest]))
    table = predict_table(data, table).rename(columns={"gameweek": "played"})
    print(f"\nChance of each finish, {table['season'].iloc[0]} table as it stands:")
    columns = [o.column for o in OUTCOMES]
    shown = table[["position", "team", "played", "points", "goal_difference", *columns]].copy()
    shown.insert(
        5, "last_season", table["last_season_finish"].astype(object).replace(PROMOTED, "promoted")
    )
    for outcome in OUTCOMES:
        shown[outcome.column] = shown[outcome.column].map("{:.0%}".format)
    print(shown.rename(columns={o.column: o.name for o in OUTCOMES}).to_string(index=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
