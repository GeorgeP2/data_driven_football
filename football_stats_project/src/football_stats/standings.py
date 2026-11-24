"""League standings after every gameweek, for every Premier League season.

Builds one row per season, team and gameweek with the team's points and league position at that
point, plus where it finished that season.

Run from the project folder: ``PYTHONPATH=src python -m football_stats.standings``
"""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from football.football_data import canon, read_fd_csv

PROJECT = Path(__file__).resolve().parents[2]
CACHE = PROJECT / "data" / "fd_cache"
OUTPUTS = PROJECT / "outputs"

SEASON_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
FIRST_SEASON = 1993  # first season football-data.co.uk has, as its starting year
COLUMNS = [
    "season",
    "team",
    "gameweek",
    "points",
    "goal_difference",
    "position",
    "gap_to_17th",
    "title_margin",
    "top4_margin",
    "last_season_finish",
    "final_position",
]
SAFE_POSITION = 17  # the lowest place that stays up in a 20-team season
MARGINS = {"title_margin": 1, "top4_margin": 4}  # column -> number of places that count

# Points adjustments made by the league: (season, team, date imposed, points). An adjustment
# counts from the team's first match on or after that date. Everton's ten-point deduction was cut
# to six on appeal, hence the +4. Spurs' 1994-95 deduction is left out: it was quashed mid-season.
DEDUCTIONS = [
    ("1996-97", "Middlesbrough", "1997-01-14", -3),
    ("2009-10", "Portsmouth", "2010-03-17", -9),
    ("2023-24", "Everton", "2023-11-17", -10),
    ("2023-24", "Everton", "2024-02-26", 4),
    ("2023-24", "Everton", "2024-04-08", -2),
    ("2023-24", "Nott'm Forest", "2024-03-18", -4),
]


def season_code(start_year: int) -> str:
    """1999 -> '9900', the season code football-data.co.uk uses in its URLs."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """1999 -> '1999-00'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def latest_season(today: pd.Timestamp | None = None) -> int:
    """Starting year of the season in progress (seasons start in August)."""
    today = today or pd.Timestamp.today()
    return today.year if today.month >= 8 else today.year - 1


def clean_results(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """Played matches from a football-data file as season, date, home, away, hg, ag."""
    df = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    date = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    out = pd.DataFrame(
        {
            "season": season,
            "date": date,
            "home": df["HomeTeam"].map(canon),
            "away": df["AwayTeam"].map(canon),
            "hg": df["FTHG"].astype(int),
            "ag": df["FTAG"].astype(int),
        }
    )
    return out.dropna(subset=["date"]).reset_index(drop=True)


def is_complete(results: pd.DataFrame) -> bool:
    """True when every team has played every other team home and away."""
    n_teams = pd.concat([results["home"], results["away"]]).nunique()
    return n_teams > 1 and len(results) == n_teams * (n_teams - 1)


def load_season(start_year: int, cache_dir: Path = CACHE) -> pd.DataFrame:
    """One season's results, downloaded once and cached. A cached season that is still in
    progress is downloaded again so new results are picked up."""
    path = cache_dir / f"E0_{season_code(start_year)}.csv"
    label = season_label(start_year)
    if path.exists():
        results = clean_results(read_fd_csv(path), label)
        if is_complete(results):
            return results
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = read_fd_csv(SEASON_URL.format(code=season_code(start_year)))
    raw.to_csv(path, index=False)
    return clean_results(raw, label)


def points_at(table: pd.DataFrame, position: int, keys: list[str]) -> np.ndarray:
    """Points of the team in ``position`` within each group of ``keys``, one value per row of
    ``table`` (NaN where the group has no team in that position)."""
    there = table.loc[table["position"].eq(position).fillna(False), [*keys, "points"]]
    there = there.rename(columns={"points": "there"})
    return table[keys].merge(there, on=keys, how="left")["there"].to_numpy(float)


def add_gaps(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Add ``gap_to_17th`` and the margin columns to a ranked table; ``keys`` identify one table
    (a season and gameweek, or just a season)."""
    points = table["points"].to_numpy(float)
    gap = points - points_at(table, SAFE_POSITION, keys)
    table["gap_to_17th"] = pd.array(gap, dtype="Int64")
    for column, places in MARGINS.items():
        inside = table["position"].le(places).fillna(False).to_numpy(bool)
        line = np.where(inside, points_at(table, places + 1, keys), points_at(table, places, keys))
        table[column] = pd.array(points - line, dtype="Int64")
    return table


def standings(results: pd.DataFrame, deductions=DEDUCTIONS) -> pd.DataFrame:
    """Points and league position of every team after each of its matches.

    ``results`` has one row per played match: season, date, home, away, hg, ag.

    A team's gameweek is the number of league matches it has played, so the position at gameweek
    n compares every team after n matches each. Points include ``deductions`` from the gameweek
    they were imposed. Ties on points are split by goal difference, then goals scored, then team
    name. ``gap_to_17th`` is the team's points minus those of the team in 17th at that gameweek:
    negative for the bottom three, zero for 17th itself. ``title_margin`` and ``top4_margin``
    measure the distance to the line below 1st and below 4th: for a team inside those places, its
    lead over the best team outside them; for a team outside, its (negative) gap to the last team
    inside. ``last_season_finish`` is the team's final position the season before; it is empty
    for promoted teams and when the season before is not in ``results``. A gameweek that not
    every team has reached yet has no position, gap or margin, and a season that is still in
    progress has no final position.
    """
    home = results.rename(columns={"home": "team", "hg": "gf", "ag": "ga"})
    away = results.rename(columns={"away": "team", "ag": "gf", "hg": "ga"})
    cols = ["season", "date", "team", "gf", "ga"]
    played = pd.concat([home[cols], away[cols]], ignore_index=True)
    played = played.sort_values(["season", "date"], kind="stable").reset_index(drop=True)

    played["won"] = np.select(
        [played["gf"] > played["ga"], played["gf"] == played["ga"]], [3, 1], default=0
    )
    played["diff"] = played["gf"] - played["ga"]
    by_team = played.groupby(["season", "team"])
    played["gameweek"] = by_team.cumcount() + 1
    played["points"] = by_team["won"].cumsum()
    played["goal_difference"] = by_team["diff"].cumsum()
    played["scored"] = by_team["gf"].cumsum()
    for season, team, imposed, points in deductions:
        affected = (
            (played["season"] == season)
            & (played["team"] == team)
            & (played["date"] >= pd.Timestamp(imposed))
        )
        played.loc[affected, "points"] += points

    played = played.sort_values(
        ["season", "gameweek", "points", "goal_difference", "scored", "team"],
        ascending=[True, True, False, False, False, True],
    ).reset_index(drop=True)
    by_gameweek = played.groupby(["season", "gameweek"])
    n_teams = played.groupby("season")["team"].transform("nunique")
    everyone_played = by_gameweek["team"].transform("size") == n_teams
    played["position"] = (by_gameweek.cumcount() + 1).where(everyone_played).astype("Int64")
    played = add_gaps(played, ["season", "gameweek"])

    last_gameweek = played["gameweek"] == 2 * (n_teams - 1)
    final = played[last_gameweek & everyone_played].set_index(["season", "team"])["position"]
    final = final.rename("final_position")
    out = played.join(final, on=["season", "team"])

    seasons = sorted(out["season"].unique())
    following = dict(pairwise(seasons))
    last = final.rename("last_season_finish").reset_index()
    last["season"] = last["season"].map(following)
    out = out.merge(last.dropna(subset="season"), on=["season", "team"], how="left")
    return out.sort_values(["season", "gameweek", "position", "team"])[COLUMNS].reset_index(
        drop=True
    )


def table_now(results: pd.DataFrame, deductions=DEDUCTIONS) -> pd.DataFrame:
    """The latest season's table as it stands: every team at its most recent match, ranked the
    usual way even when teams have played different numbers of matches. ``gameweek`` is each
    team's own matches played. Same columns as ``standings``; pass the season before as well to
    get ``last_season_finish``."""
    seasons = sorted(results["season"].unique())
    season = results[results["season"] == seasons[-1]]
    rows = standings(results[results["season"].isin(seasons[-2:])], deductions)
    rows = rows[rows["season"] == seasons[-1]]
    table = rows.loc[rows.groupby("team")["gameweek"].idxmax()].set_index("team")
    table["scored"] = (
        season.groupby("home")["hg"].sum().add(season.groupby("away")["ag"].sum(), fill_value=0)
    )
    table = table.reset_index().sort_values(
        ["points", "goal_difference", "scored", "team"], ascending=[False, False, False, True]
    )
    table["position"] = pd.array(range(1, len(table) + 1), dtype="Int64")
    return add_gaps(table.reset_index(drop=True), ["season"])[COLUMNS]


def load_results(first: int = FIRST_SEASON, last: int | None = None, cache_dir: Path = CACHE):
    """Match results for every season from ``first`` to ``last`` (starting years, inclusive)."""
    last = latest_season() if last is None else last
    results = [load_season(year, cache_dir) for year in range(first, last + 1)]
    return pd.concat(results, ignore_index=True)


def build(first: int = FIRST_SEASON, last: int | None = None, cache_dir: Path = CACHE):
    """Standings for every season from ``first`` to ``last`` (starting years, inclusive)."""
    return standings(load_results(first, last, cache_dir))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first", type=int, default=FIRST_SEASON, help="first season, e.g. 1993")
    ap.add_argument("--last", type=int, default=None, help="last season (default: current)")
    ap.add_argument("--out", type=Path, default=OUTPUTS / "standings.csv")
    args = ap.parse_args()

    df = build(args.first, args.last)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df)
    print(f"\n{df['season'].nunique()} seasons, {len(df):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
