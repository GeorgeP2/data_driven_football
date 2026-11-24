"""Simulate the rest of the season to get every team's chance of every finishing position.

Each team carries an Elo rating, updated after every match since 1993-94. The rating gap between
two teams sets the expected goals for each side, goals are drawn from Poisson distributions, and
the remaining fixtures are played out thousands of times. Ratings keep updating inside each
simulated season, so a team that starts a run gets stronger in that run. Counting where each
team finishes gives one consistent set of probabilities: title, top four, relegation and every
position in between.

Run from the project folder:
    PYTHONPATH=src python -m football_stats.simulation --provisional
    PYTHONPATH=src python -m football_stats.simulation --backtest
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from football_stats.provisional import fetch_provisional_results
from football_stats.relegation import (
    OUTCOMES,
    brier,
    fill_promoted,
    fit_xgboost,
    log_loss,
    predict_xgboost,
    rescale,
    training_data,
)
from football_stats.standings import OUTPUTS, load_results, standings, table_now

BASE = 1500.0  # rating of an average team
BURN_IN = 2  # seasons at the start of the data used only to settle the ratings
BACKTEST_FROM = "2010-11"  # first season scored in the backtest; parameters come from before it
SNAPSHOT_EVERY = 10  # matches between backtest snapshots, about one gameweek


@dataclass(frozen=True)
class Parameters:
    """Settings of the rating system and of the goals model."""

    k: float  # how far one result moves a rating
    home: float  # home advantage, in rating points
    carry: float  # share of a team's gap to average kept from one season to the next
    promoted: float  # rating given to a team that was not in the division last season
    home_goals: tuple[float, float] = (0.0, 0.0)  # log expected goals: intercept, slope
    away_goals: tuple[float, float] = (0.0, 0.0)

    def expected_goals(self, rating_gap):
        """Expected home and away goals for a home-minus-away rating gap."""
        x = np.asarray(rating_gap, float) / 400
        home = np.exp(self.home_goals[0] + self.home_goals[1] * x)
        away = np.exp(self.away_goals[0] - self.away_goals[1] * x)
        return home, away

    def rating_change(self, rating_gap, goal_difference):
        """Rating points the home team gains (and the away team loses) for a result."""
        expected = 1 / (1 + 10 ** (-(np.asarray(rating_gap, float) + self.home) / 400))
        score = (np.sign(goal_difference) + 1) / 2
        margin = np.abs(goal_difference)
        weight = np.where(margin <= 1, 1.0, np.where(margin == 2, 1.5, 1.75 + (margin - 3) / 8))
        return self.k * weight * (score - expected)


def rate(results: pd.DataFrame, params: Parameters) -> pd.DataFrame:
    """Run the ratings through every match in date order. Returns ``results`` sorted that way
    with each side's rating before and after the match."""
    matches = results.sort_values(["season", "date"], kind="stable").reset_index(drop=True)
    ratings: dict[str, float] = {}
    last_teams: set[str] = set()
    columns: dict[str, list[float]] = {
        "home_before": [],
        "away_before": [],
        "home_after": [],
        "away_after": [],
    }
    for season, group in matches.groupby("season", sort=True):
        teams = set(group["home"]) | set(group["away"])
        ratings = {
            team: BASE + params.carry * (ratings[team] - BASE)
            if team in last_teams
            else (params.promoted if last_teams else BASE)
            for team in teams
        }
        last_teams = teams
        for home, away, hg, ag in zip(
            group["home"], group["away"], group["hg"], group["ag"], strict=True
        ):
            before_home, before_away = ratings[home], ratings[away]
            change = float(params.rating_change(before_home - before_away, hg - ag))
            ratings[home] += change
            ratings[away] -= change
            columns["home_before"].append(before_home)
            columns["away_before"].append(before_away)
            columns["home_after"].append(ratings[home])
            columns["away_after"].append(ratings[away])
        del season
    return matches.assign(**columns)


def _poisson_fit(x: np.ndarray, goals: np.ndarray) -> tuple[tuple[float, float], float]:
    """Poisson regression of goals on x. Returns (intercept, slope) and the log-likelihood,
    leaving out the constant term."""
    design = np.column_stack([np.ones(len(x)), x])
    w = np.array([np.log(goals.mean()), 0.0])
    for _ in range(50):
        rate_ = np.exp(design @ w)
        step = np.linalg.solve((design * rate_[:, None]).T @ design, design.T @ (goals - rate_))
        w += step
        if np.abs(step).max() < 1e-10:
            break
    rate_ = np.exp(design @ w)
    return (float(w[0]), float(w[1])), float(np.sum(goals * np.log(rate_) - rate_))


def fit_goals(rated: pd.DataFrame, params: Parameters) -> tuple[Parameters, float]:
    """Fit the goals model on rated matches (skipping the burn-in seasons). Returns the
    parameters with the goals model filled in, and its log-likelihood per match."""
    seasons = sorted(rated["season"].unique())
    used = rated[rated["season"].isin(seasons[BURN_IN:])]
    x = ((used["home_before"] - used["away_before"]) / 400).to_numpy(float)
    home, home_ll = _poisson_fit(x, used["hg"].to_numpy(float))
    away, away_ll = _poisson_fit(-x, used["ag"].to_numpy(float))
    fitted = Parameters(params.k, params.home, params.carry, params.promoted, home, away)
    return fitted, (home_ll + away_ll) / len(used)


def fit_parameters(results: pd.DataFrame) -> Parameters:
    """Choose the rating settings that make the goals model fit ``results`` best."""

    def build(v) -> Parameters:
        return Parameters(*(float(x) for x in v))

    def loss(v) -> float:
        if not (1 <= v[0] <= 80 and 0 <= v[2] <= 1):
            return 1e9
        params = build(v)
        return -fit_goals(rate(results, params), params)[1]

    best = minimize(
        loss, [20.0, 60.0, 0.8, 1400.0], method="Nelder-Mead", options={"xatol": 0.5, "fatol": 1e-6}
    )
    params = build(best.x)
    return fit_goals(rate(results, params), params)[0]


def latest_ratings(rated: pd.DataFrame) -> pd.Series:
    """Each team's rating after its most recent match in ``rated`` (one season's matches)."""
    home = rated[["home", "home_after"]].set_axis(["team", "rating"], axis=1)
    away = rated[["away", "away_after"]].set_axis(["team", "rating"], axis=1)
    both = pd.concat([home, away]).sort_index(kind="stable")
    return both.groupby("team")["rating"].last()


def remaining_fixtures(played: pd.DataFrame, seed: int = 0) -> list[tuple[str, str]]:
    """Every home and away pairing not in ``played`` yet, in a fixed random order."""
    teams = sorted(set(played["home"]) | set(played["away"]))
    done = set(zip(played["home"], played["away"], strict=True))
    fixtures = [(h, a) for h in teams for a in teams if h != a and (h, a) not in done]
    order = np.random.default_rng(seed).permutation(len(fixtures))
    return [fixtures[i] for i in order]


def simulate(
    table: pd.DataFrame,
    ratings: pd.Series,
    fixtures: list[tuple[str, str]],
    params: Parameters,
    sims: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Play ``fixtures`` out ``sims`` times from ``table`` (team, points, goal_difference,
    scored). Returns one row per team: ``expected_points`` and the probability of finishing in
    each position, in columns 1 to the number of teams."""
    rng = np.random.default_rng(seed)
    teams = list(table["team"])
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    def start(column) -> np.ndarray:
        return np.tile(table[column].to_numpy(float), (sims, 1))

    points, difference, scored = start("points"), start("goal_difference"), start("scored")
    rating = np.tile(ratings.reindex(teams).to_numpy(float), (sims, 1))
    for home, away in fixtures:
        h, a = index[home], index[away]
        gap = rating[:, h] - rating[:, a]
        home_rate, away_rate = params.expected_goals(gap)
        hg, ag = rng.poisson(home_rate), rng.poisson(away_rate)
        margin = hg - ag
        points[:, h] += np.where(margin > 0, 3, margin == 0)
        points[:, a] += np.where(margin < 0, 3, margin == 0)
        difference[:, h] += margin
        difference[:, a] -= margin
        scored[:, h] += hg
        scored[:, a] += ag
        change = params.rating_change(gap, margin)
        rating[:, h] += change
        rating[:, a] -= change

    # rank on points, then goal difference, then goals scored, then a coin toss
    key = points * 1e7 + (difference + 1000) * 1e3 + scored + rng.random(points.shape) * 0.5
    order = np.argsort(-key, axis=1)
    position = np.empty_like(order)
    np.put_along_axis(position, order, np.arange(n)[None, :].repeat(sims, axis=0), axis=1)
    finish = np.stack([np.bincount(position[:, i], minlength=n) for i in range(n)]) / sims
    out = pd.DataFrame(finish, columns=range(1, n + 1))
    out.insert(0, "expected_points", points.mean(axis=0))
    out.insert(0, "team", teams)
    return out


def summarise(simulated: pd.DataFrame) -> pd.DataFrame:
    """Title, top-four and relegation probabilities and the average finishing position."""
    positions = [c for c in simulated.columns if isinstance(c, int)]
    out = simulated[["team", "expected_points"]].copy()
    for outcome in OUTCOMES:
        out[outcome.column] = simulated[list(range(outcome.first, outcome.last + 1))].sum(axis=1)
    out["expected_position"] = simulated[positions].to_numpy() @ np.array(positions)
    return out


def goals_scored(played: pd.DataFrame) -> pd.Series:
    home, away = played.groupby("home")["hg"].sum(), played.groupby("away")["ag"].sum()
    return home.add(away, fill_value=0)


def forecast(
    history: pd.DataFrame, played: pd.DataFrame, params: Parameters, sims: int, seed: int = 0
) -> pd.DataFrame:
    """Simulate the rest of the season in ``played`` (its matches so far). ``history`` holds the
    seasons before it. Returns the table as it stands with the simulated finish of each team."""
    everything = pd.concat([history, played], ignore_index=True)
    season = played["season"].iloc[0]
    rated = rate(everything, params)
    table = table_now(everything)
    table["scored"] = table["team"].map(goals_scored(played))
    simulated = simulate(
        table,
        latest_ratings(rated[rated["season"] == season]),
        remaining_fixtures(played, seed),
        params,
        sims,
        seed,
    )
    ratings = latest_ratings(rated[rated["season"] == season]).rename("rating")
    return table.merge(ratings, left_on="team", right_index=True).merge(simulated, on="team")


def snapshots(season_results: pd.DataFrame) -> list[pd.DataFrame]:
    """The season's matches so far at the end of each day that takes the number played past
    another ``SNAPSHOT_EVERY``, once every team has played and while matches remain."""
    matches = season_results.sort_values("date", kind="stable")
    teams = set(matches["home"]) | set(matches["away"])
    played_by_day = matches.groupby("date").size().cumsum()
    out, target = [], SNAPSHOT_EVERY
    for day, count in played_by_day.items():
        if count < target or count == len(matches):
            continue
        target = (count // SNAPSHOT_EVERY + 1) * SNAPSHOT_EVERY
        played = matches[matches["date"] <= day]
        if set(played["home"]) | set(played["away"]) == teams:
            out.append(played)
    return out


def backtest(results: pd.DataFrame, sims: int = 2000, first: str = BACKTEST_FROM) -> pd.DataFrame:
    """Score the simulation and the XGBoost models on the same as-it-stands tables.

    The simulation's parameters are fitted on the seasons before ``first``; XGBoost is fitted on
    every other season than the one being predicted. Returns one row per team and snapshot with
    the truth and both sets of probabilities.
    """
    params = fit_parameters(results[results["season"] < first])
    full = standings(results)
    data = training_data(full)
    final = full.drop_duplicates(["season", "team"]).set_index(["season", "team"])
    seasons = [s for s in sorted(data["season"].unique()) if s >= first]
    rows = []
    for season in seasons:
        history = results[results["season"] < season]
        train = data[data["season"] != season]
        models = {o.name: fit_xgboost(train, o) for o in OUTCOMES}
        for number, played in enumerate(snapshots(results[results["season"] == season])):
            table = forecast(history, played, params, sims, seed=number)
            summary = summarise(table[["team", "expected_points", *range(1, len(table) + 1)]])
            row = table[["season", "team"]].assign(snapshot=number, played=len(played))
            finished = final.loc[list(zip(row["season"], row["team"], strict=True))]
            features = fill_promoted(table)
            for outcome in OUTCOMES:
                place = finished["final_position"].to_numpy()
                row[outcome.label] = (place >= outcome.first) & (place <= outcome.last)
                row[f"simulation_{outcome.name}"] = summary[outcome.column].to_numpy()
                raw = predict_xgboost(models[outcome.name], features, outcome)
                row[f"xgboost_{outcome.name}"] = rescale(raw, outcome.places)
            rows.append(row)
    return pd.concat(rows, ignore_index=True)


def backtest_scores(rows: pd.DataFrame, sims: int) -> pd.DataFrame:
    """Log loss and Brier score per outcome and model. Simulated probabilities are kept away
    from exactly 0 and 1 by half a simulation."""
    edge = 0.5 / sims
    table = {}
    for outcome in OUTCOMES:
        y = rows[outcome.label]
        for model in ("xgboost", "simulation"):
            p = rows[f"{model}_{outcome.name}"].clip(edge, 1 - edge)
            table[(outcome.name, model)] = {"log_loss": log_loss(y, p), "brier": brier(y, p)}
    return pd.DataFrame(table).T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sims", type=int, help="simulated seasons (default 10,000; backtest 2,000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--provisional",
        action="store_true",
        help="add finished matches football-data.co.uk has not published yet (from the FPL API)",
    )
    ap.add_argument("--backtest", action="store_true", help="score against XGBoost on past seasons")
    ap.add_argument("--out", type=Path, default=OUTPUTS / "simulation.csv")
    args = ap.parse_args()

    results = load_results()
    if args.backtest:
        sims = args.sims or 2000
        rows = backtest(results, sims)
        print(
            f"{rows['season'].nunique()} seasons from {BACKTEST_FROM}, "
            f"{rows.groupby(['season', 'snapshot']).ngroups} tables, {len(rows):,} rows "
            "(lower is better):\n"
        )
        print(backtest_scores(rows, sims).round(4))
        return

    season = results["season"].max()
    played = results[results["season"] == season]
    if args.provisional:
        extra = fetch_provisional_results(played)
        print(f"Provisional results added ({len(extra)}):")
        for match in extra.itertuples():
            print(f"  {match.date:%a %d %b}  {match.home} {match.hg}-{match.ag} {match.away}")
        played = pd.concat([played, extra], ignore_index=True)

    params = fit_parameters(results[results["season"] != season])
    print(f"\n{params}")
    table = forecast(results[results["season"] != season], played, params, args.sims or 10_000)
    summary = summarise(table[["team", "expected_points", *range(1, len(table) + 1)]])
    shown = table[["position", "team", "gameweek", "points", "rating"]].merge(summary, on="team")
    shown = shown.rename(columns={"gameweek": "played", **{o.column: o.name for o in OUTCOMES}})
    for outcome in OUTCOMES:
        shown[outcome.name] = shown[outcome.name].map("{:.0%}".format)
    print(f"\nSimulated finish, {season} from the table as it stands:")
    print(
        shown.round({"rating": 0, "expected_points": 1, "expected_position": 1}).to_string(
            index=False
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.drop(columns=["final_position", "scored"]).to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
