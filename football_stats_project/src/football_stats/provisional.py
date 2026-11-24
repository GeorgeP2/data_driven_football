"""Provisional results for matches football-data.co.uk has not published yet.

The Fantasy Premier League API marks a match ``finished_provisional`` at the final whistle, a day
or more before the result reaches football-data.co.uk. Matches still in play are left out.
"""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

import pandas as pd

FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def _get(url: str):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def match_names(fpl_names, known_names) -> dict[str, str]:
    """Map FPL team names onto the names already used in the results, where the FPL name is the
    same or a longer form ("Coventry City" -> "Coventry")."""
    mapping = {}
    for name in fpl_names:
        matches = [k for k in known_names if name == k] or [
            k for k in known_names if name.startswith(k + " ")
        ]
        if len(matches) != 1:
            raise ValueError(f"cannot match FPL team {name!r} to a team in the results")
        mapping[name] = matches[0]
    return mapping


def provisional_results(fixtures, teams: dict[int, str], results: pd.DataFrame) -> pd.DataFrame:
    """Finished matches from the FPL fixture list that are not in ``results`` yet.

    ``fixtures`` is the FPL fixtures payload, ``teams`` maps FPL team ids to FPL names, and
    ``results`` is the current season so far (season, date, home, away, hg, ag). Returns rows in
    the same shape.
    """
    known = set(results["home"]) | set(results["away"])
    names = match_names(teams.values(), known)
    played = set(zip(results["home"], results["away"], strict=True))
    rows = []
    for fixture in fixtures:
        if not (fixture["finished"] or fixture["finished_provisional"]):
            continue
        home, away = names[teams[fixture["team_h"]]], names[teams[fixture["team_a"]]]
        if (home, away) in played:
            continue
        kickoff = pd.Timestamp(fixture["kickoff_time"]).tz_localize(None).normalize()
        rows.append(
            {
                "season": results["season"].iloc[0],
                "date": kickoff,
                "home": home,
                "away": away,
                "hg": int(fixture["team_h_score"]),
                "ag": int(fixture["team_a_score"]),
            }
        )
    return pd.DataFrame(rows, columns=["season", "date", "home", "away", "hg", "ag"])


def fetch_provisional_results(results: pd.DataFrame) -> pd.DataFrame:
    """Download the FPL fixture list and return the finished matches missing from ``results``."""
    teams = {team["id"]: team["name"] for team in _get(BOOTSTRAP_URL)["teams"]}
    return provisional_results(_get(FIXTURES_URL), teams, results)
