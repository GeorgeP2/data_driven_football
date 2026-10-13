"""Run FPL.ipynb's pipeline for each mini-league and write the chart data to JSON.

Usage, from the project folder:
    PYTHONPATH=src python -m fpl_optimiser.leagues.export           # pull histories, simulate
    PYTHONPATH=src python -m fpl_optimiser.leagues.export --cached  # reuse the CSVs in data/
"""

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from fpl_optimiser.leagues import insights
from fpl_optimiser.leagues.analysis import (
    add_features,
    add_required_run_rate,
    load_histories,
    simulate_season_finish,
    simulate_win_prob_by_gw,
    win_prob_deltas,
)
from fpl_optimiser.leagues.gameweeks import (
    GameweekSource,
    get_json,
    league_gameweeks,
    league_season_transfers,
)

HISTORY_DIR = Path("data")
OUT_DIR = Path("outputs/league_pages")

LEAGUES_PATH = Path("leagues.json")
"""Which leagues to export and who is in them. Git-ignored, because it names real people: copy
leagues.example.json to leagues.json and fill it in."""


def load_leagues(path: Path = LEAGUES_PATH) -> dict[str, dict[str, Any]]:
    """Read the league config, with entry ids as ints.

    Entry ids are reissued every season. Keys in "entries" are the short names the notebook
    uses, and their order fixes each manager's colour. Keep the names free of '_':
    FPL.ipynb splits win-prob column names on it, hence "labels" for display names.
    """
    if not path.is_file():
        raise SystemExit(f"No {path}; copy leagues.example.json to {path} and fill it in.")
    leagues: dict[str, dict[str, Any]] = json.loads(path.read_text())
    for cfg in leagues.values():
        cfg["entries"] = {int(entry): name for entry, name in cfg["entries"].items()}
        cfg.setdefault("labels", {})
    return leagues


# Same settings as the notebook's Parameters cell.
TOTAL_GWS = 38
OUTLOOK_GWS = 5  # how many upcoming gameweeks the fixture outlook covers
N_ITER = 10000
SEED = 42
EWM_SPAN = 5
RECENT_WINDOW = 5


def r(x: Any, n: int = 3) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def season_meta(source: GameweekSource) -> dict[str, Any]:
    events = sorted(source.events.values(), key=lambda e: e["id"])
    start_year = int(events[0]["deadline_time"][:4])
    current = next((e for e in events if e["is_current"]), None)
    provisional = ""
    if current and not current["finished"]:
        provisional = (
            f"GW{current['id']} is not yet marked finished by FPL, so bonus points and these "
            "figures may move slightly when it is."
        )
    return {"season": f"{start_year}-{(start_year + 1) % 100:02d}", "provisional": provisional}


def league_meta(league_id: int, entries: dict[int, str]) -> dict[str, Any]:
    standings = get_json(f"leagues-classic/{league_id}/standings/")
    rows = {row["entry"]: row for row in standings["standings"]["results"]}
    missing = sorted(set(entries) - set(rows))
    if missing:
        raise SystemExit(
            f"Entries {missing} are not in league {league_id}. Ids are reissued each season; "
            f"current entries: {sorted(rows)}"
        )
    return {
        "leagueName": standings["league"]["name"],
        "teams": {name: rows[entry]["entry_name"] for entry, name in entries.items()},
    }


def league_insights(
    source: GameweekSource, team_map: dict[int, str], gws: list[int], leader: str
) -> dict[str, Any]:
    """Captaincy, chips, gap to the leader, fixture outlook, bench luck and head-to-head."""
    picks = {gw: {name: source.picks(entry, gw) for entry, name in team_map.items()} for gw in gws}
    live = {gw: source.live_points(gw) for gw in gws}
    name_of = source.name
    next_gw = gws[-1] + 1
    upcoming = list(range(next_gw, min(next_gw + OUTLOOK_GWS, TOTAL_GWS + 1)))

    captaincy = insights.captaincy(picks, live)
    for row in captaincy.values():
        for week in row["weeks"]:
            week["captain"] = name_of(week["captain"]) if week["captain"] else None
            week["best"] = name_of(week["best"])
    gap = insights.gap_to_leader(picks, live, leader)
    for row in gap.values():
        if row:
            row["only_mine"] = [name_of(e) for e in row["only_mine"]]
            row["only_theirs"] = [name_of(e) for e in row["only_theirs"]]
    player_team = {element: info["team_id"] for element, info in source.players.items()}
    return {
        "leader": leader,
        "chips": insights.chips_left(picks, source.chips, next_gw),
        "captaincy": captaincy,
        "most_captained": {
            str(gw): {"player": name_of(element), "managers": count}
            for gw, (element, count) in insights.most_captained(picks).items()
        },
        "gap": gap,
        "outlook": insights.fixture_outlook(
            picks[gws[-1]], source.fixtures(), player_team, upcoming
        ),
        "bench": insights.bench_luck(picks, live),
        "h2h": insights.head_to_head(picks),
    }


def export_league(
    key: str, cfg: dict[str, Any], refresh: bool, common: dict[str, Any], source: GameweekSource
) -> None:
    team_map: dict[int, str] = cfg["entries"]
    history = load_histories(team_map, output_dir=str(HISTORY_DIR), refresh=refresh)
    features = add_features(history, ewma_span=EWM_SPAN)
    names = list(team_map.values())

    season = simulate_season_finish(
        features,
        team_map,
        total_gws=TOTAL_GWS,
        n_iter=N_ITER,
        seed=SEED,
        recent_window=RECENT_WINDOW,
    )
    win = simulate_win_prob_by_gw(features, team_map, total_gws=TOTAL_GWS, n_iter=N_ITER, seed=SEED)
    win_form = simulate_win_prob_by_gw(
        features,
        team_map,
        total_gws=TOTAL_GWS,
        n_iter=N_ITER,
        seed=SEED,
        rate_column="ewma_points",
        rate_games=EWM_SPAN,
    )
    deltas = win_prob_deltas(win, n_iter=N_ITER, alpha=0.05)
    run_rate = add_required_run_rate(features, total_gws=TOTAL_GWS)

    managers = {}
    for name in names:
        person = features[features["name"] == name].sort_values("event")
        rates = run_rate[run_rate["name"] == name].sort_values("event")
        finishes = season[season["name"] == name]
        moves = deltas[deltas["name"] == name].sort_values("game_week")
        managers[name] = {
            "points": [int(x) for x in person["points"]],
            "total": [int(x) for x in person["total_points"]],
            "bench": [int(x) for x in person["points_on_bench"]],
            "behind": [int(x) for x in person["points_from_leader"]],
            "ewma": [r(x, 1) for x in person["ewma_points"]],
            "mean": [r(x, 1) for x in rates["running_mean_points"]],
            "transfers": [int(x) for x in person["event_transfers"]],
            "hits": [int(x) for x in person["event_transfers_cost"]],
            "overall_rank": [int(x) for x in person["overall_rank"]],
            "win": [r(x) for x in win[f"{name}_win_prob"]],
            "win_form": [r(x) for x in win_form[f"{name}_win_prob"]],
            "delta": [r(x) for x in moves["delta_significant"]],
            "delta_gws": [int(x) for x in moves["game_week"]],
            "rank_prob": {
                int(k): r(v)
                for k, v in finishes["final_rank"].value_counts(normalize=True).sort_index().items()
            },
        }

    gameweeks = league_gameweeks(
        source, team_map, sorted(int(g) for g in features["event"].unique())
    )
    payload = {
        "data": {
            "league": key,
            "league_id": cfg["league_id"],
            "gws": sorted(int(g) for g in features["event"].unique()),
            "win_gws": [int(x) for x in win["game_week"]],
            "names": names,
            "n_iter": N_ITER,
            "managers": managers,
            "gameweeks": gameweeks,
            "season_transfers": league_season_transfers(source, team_map, gameweeks),
            "insights": league_insights(
                source,
                team_map,
                sorted(int(g) for g in gameweeks),
                max(names, key=lambda n: managers[n]["total"][-1]),
            ),
        },
        "meta": {**common, **league_meta(cfg["league_id"], team_map), "labels": cfg["labels"]},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{key}.json").write_text(json.dumps(payload))
    standings = sorted(names, key=lambda n: -managers[n]["total"][-1])
    print(
        f"{key}: GW{payload['data']['gws'][-1]}, "
        + ", ".join(
            f"{n} {managers[n]['total'][-1]} ({managers[n]['win'][-1]:.0%})" for n in standings
        )
    )


def run(cached: bool = False) -> None:
    """Export every league in leagues.json. FPL.ipynb calls this directly."""
    source = GameweekSource()
    common = {**season_meta(source), "pulled": dt.date.today().strftime("%-d %b %Y")}
    for key, cfg in load_leagues().items():
        export_league(key, cfg, refresh=not cached, common=common, source=source)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cached", action="store_true", help="reuse the history CSVs instead of pulling"
    )
    run(cached=parser.parse_args().cached)


if __name__ == "__main__":
    main()
