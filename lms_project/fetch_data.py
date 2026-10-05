#!/usr/bin/env python3
"""
Pull EPL data from football-data.co.uk and write results.csv / fixtures.csv
in the format lms_optimiser.py expects.

  results.csv   : date,home,away,hg,ag   (last N seasons of completed PL matches)
  fixtures.csv  : gw,home,away,odds_h,odds_d,odds_a

Where things come from
  - Results + odds history:  https://football-data.co.uk/mmz4281/<SSYY>/E0.csv
  - Upcoming fixtures + odds: https://football-data.co.uk/fixtures.csv  (next round or so only, Div == E0)
  - Full-season schedule (gameweek numbers for later weeks): NOT on football-data.
    Supply it with --schedule schedule.csv (columns: gw,home,away), e.g. exported from the
    official FPL API or fixturedownload.com. Odds from football-data are merged onto it.

Usage
  python fetch_data.py --seasons 2526 2627 --schedule schedule.csv
  python fetch_data.py --seasons 2526 2627              # odds-only fixtures; gw derived from dates

Check the printed warnings: team names must match across all sources.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from fd_common import canon, read_fd_csv

BASE = "https://football-data.co.uk"
SEASON_URL = BASE + "/mmz4281/{s}/E0.csv"
FIXTURES_URL = BASE + "/fixtures.csv"

# Odds columns in order of preference: market average, Pinnacle, Bet365.
ODDS_SETS = [("AvgH", "AvgD", "AvgA"), ("PSH", "PSD", "PSA"), ("B365H", "B365D", "B365A")]

def read_fd(src: str) -> pd.DataFrame:
    df = read_fd_csv(src)
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    if "Div" in df.columns:
        df = df[df["Div"] == "E0"]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    return df.dropna(subset=["Date"])


def best_odds(df: pd.DataFrame) -> pd.DataFrame:
    """First available odds set per row, falling through Avg -> Pinnacle -> Bet365."""
    out = pd.DataFrame(index=df.index, columns=["odds_h", "odds_d", "odds_a"], dtype=float)
    for h, d, a in ODDS_SETS:
        if not all(c in df.columns for c in (h, d, a)):
            continue
        ok = out["odds_h"].isna() & df[[h, d, a]].notna().all(axis=1)
        out.loc[ok, ["odds_h", "odds_d", "odds_a"]] = df.loc[ok, [h, d, a]].values
    return out


def fetch_results(seasons):
    frames = []
    for s in seasons:
        try:
            frames.append(read_fd(SEASON_URL.format(s=s)))
        except Exception as e:  # network or 404
            print(f"! could not load season {s}: {e}", file=sys.stderr)
    if not frames:
        sys.exit("No results loaded.")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["FTHG", "FTAG"])
    res = pd.DataFrame({
        "date": df["Date"].dt.strftime("%Y-%m-%d"),
        "home": df["HomeTeam"].map(canon),
        "away": df["AwayTeam"].map(canon),
        "hg": df["FTHG"].astype(int),
        "ag": df["FTAG"].astype(int),
    })
    return res.sort_values("date").reset_index(drop=True)


def derive_gw(df: pd.DataFrame, start_gw: int, gap_days: int = 3):
    """Cluster dates into gameweeks: a new gameweek starts after a >gap_days break."""
    d = df["Date"].sort_values()
    new = (d.diff().dt.days > gap_days).fillna(False)
    gw = start_gw + new.cumsum()
    return gw.reindex(df.index)


def fetch_fixtures(schedule_path=None, start_gw=1):
    fx = read_fd(FIXTURES_URL)
    fx = fx.join(best_odds(fx))
    fx["home"] = fx["HomeTeam"].map(canon)
    fx["away"] = fx["AwayTeam"].map(canon)

    if schedule_path:
        sched = pd.read_csv(schedule_path)
        sched["home"] = sched["home"].map(canon)
        sched["away"] = sched["away"].map(canon)
        # the optimiser treats the first gw in fixtures.csv as "now": drop gameweeks already played
        sched = sched[sched["gw"] >= start_gw]
        merged = sched.merge(fx[["home", "away", "odds_h", "odds_d", "odds_a"]],
                             on=["home", "away"], how="left")
        unmatched = set(fx["home"]) - set(sched["home"]) - set(sched["away"])
        if unmatched:
            print(f"! football-data teams not found in schedule: {sorted(unmatched)}", file=sys.stderr)
        return merged[["gw", "home", "away", "odds_h", "odds_d", "odds_a"]]

    # No schedule: gameweeks derived from dates, only covers what fixtures.csv lists.
    fx["gw"] = derive_gw(fx, start_gw)
    print("! no --schedule given: fixtures only cover the next round(s); "
          "later gameweeks will be missing, so the planner horizon will be short.", file=sys.stderr)
    return fx[["gw", "home", "away", "odds_h", "odds_d", "odds_a"]]


def season_start(today: pd.Timestamp) -> str:
    """1 July of the year the current season began (Jan-Jun belongs to last year's season)."""
    year = today.year if today.month >= 7 else today.year - 1
    return f"{year}-07-01"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", default=["2526", "2627"])
    ap.add_argument("--schedule", help="CSV with gw,home,away for the rest of the season")
    ap.add_argument("--start-gw", type=int, help="gameweek number of the first fixture in fixtures.csv")
    ap.add_argument("--out-results", default="results.csv")
    ap.add_argument("--out-fixtures", default="fixtures.csv")
    args = ap.parse_args()

    res = fetch_results(args.seasons)
    res.to_csv(args.out_results, index=False)
    print(f"results: {len(res)} matches, {res.home.nunique()} teams -> {args.out_results}")

    # current-season completed matches / 10 + 1 = next gameweek, used if --start-gw is absent
    cur = res[res["date"] >= season_start(pd.Timestamp.today())]
    start_gw = args.start_gw or (len(cur) // 10 + 1)
    fx = fetch_fixtures(args.schedule, start_gw)
    fx.to_csv(args.out_fixtures, index=False)
    n_odds = fx["odds_h"].notna().sum()
    print(f"fixtures: {len(fx)} rows ({n_odds} with odds) -> {args.out_fixtures}")

    known = set(res.home) | set(res.away)
    bad = (set(fx.home) | set(fx.away)) - known
    if bad:
        print(f"! teams in fixtures but not in results (promoted? name mismatch?): {sorted(bad)}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
