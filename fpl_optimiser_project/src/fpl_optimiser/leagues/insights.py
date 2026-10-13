"""Season-level insights for a league, computed from FPL payloads already fetched.

Every function takes plain data, so the tests can feed them small hand-built payloads:
- picks: {gw: {name: picks payload}} as returned by entry/<id>/event/<gw>/picks/
- live:  {gw: {element id: points}}
"""

from collections import Counter
from typing import Any

Picks = dict[int, dict[str, dict[str, Any]]]
Live = dict[int, dict[int, int]]

FIRST_HALF_LAST_GW = 19  # bootstrap chips: one set for GW1/2-19, a second for GW20-38
CAPTAIN_DEFAULT_MULTIPLIER = 2


def _xi(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Players whose points counted that week (post auto-sub; all 15 on a bench boost)."""
    return [p for p in payload["picks"] if p["multiplier"] > 0]


def _armband(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The pick that actually carried the armband: the vice if auto-subs moved it."""
    carried = max(payload["picks"], key=lambda p: p["multiplier"])
    return carried if carried["multiplier"] > 1 else None


def net_points(payload: dict[str, Any]) -> int:
    history = payload["entry_history"]
    return int(history["points"] - history["event_transfers_cost"])


def chips_left(
    picks: Picks, chips: list[dict[str, Any]], next_gw: int
) -> dict[str, list[dict[str, Any]]]:
    """Every chip slot in the season for each manager: used (and when), available, or expired."""
    names = {name for rows in picks.values() for name in rows}
    played = {
        name: [
            (gw, rows[name]["active_chip"])
            for gw, rows in sorted(picks.items())
            if rows[name]["active_chip"]
        ]
        for name in names
    }
    result = {}
    for name in names:
        slots = []
        for chip in sorted(chips, key=lambda c: (c["start_event"], c["name"])):
            used = next(
                (
                    gw
                    for gw, played_chip in played[name]
                    if played_chip == chip["name"]
                    and chip["start_event"] <= gw <= chip["stop_event"]
                ),
                None,
            )
            status = (
                "used" if used else ("expired" if next_gw > chip["stop_event"] else "available")
            )
            slots.append(
                {
                    "chip": chip["name"],
                    "start": chip["start_event"],
                    "stop": chip["stop_event"],
                    "status": status,
                    "used_gw": used,
                }
            )
        result[name] = slots
    return result


def captaincy(picks: Picks, live: Live) -> dict[str, dict[str, Any]]:
    """Captain points against the best captain available from the same week's XI."""
    result: dict[str, dict[str, Any]] = {}
    for gw, rows in sorted(picks.items()):
        for name, payload in rows.items():
            carried = _armband(payload)
            multiplier = carried["multiplier"] if carried else CAPTAIN_DEFAULT_MULTIPLIER
            points = live[gw].get(carried["element"], 0) * multiplier if carried else 0
            best = max(_xi(payload), key=lambda p: live[gw].get(p["element"], 0))
            best_points = live[gw].get(best["element"], 0) * multiplier
            row = result.setdefault(name, {"points": 0, "best": 0, "lost": 0, "weeks": []})
            row["points"] += points
            row["best"] += best_points
            row["lost"] += best_points - points
            row["weeks"].append(
                {
                    "gw": gw,
                    "captain": carried["element"] if carried else None,
                    "points": points,
                    "best": best["element"],
                    "best_points": best_points,
                }
            )
    return result


def gap_to_leader(
    picks: Picks, live: Live, leader: str, recent: int = 5
) -> dict[str, dict[str, Any] | None]:
    """Where each manager's gap to the leader came from, split into three terms.

    Each week: players in both XIs cancel, so "unique" is your other players' base points minus
    the leader's other players' base points. "captaincy" is your armband bonus minus theirs, and
    "hits" is their hit minus yours. The three sum exactly to the week's points difference, and
    a week where they don't raises.
    """
    gws = sorted(picks)
    recent_gws = set(gws[-recent:])
    result: dict[str, dict[str, Any] | None] = {}
    names = picks[gws[-1]].keys()
    for name in names:
        if name == leader:
            result[name] = None
            continue
        season = {"unique": 0, "captaincy": 0, "hits": 0, "gap": 0}
        last = dict(season)
        for gw in gws:
            mine, theirs = picks[gw][name], picks[gw][leader]
            pts = live[gw]
            my_xi = {p["element"]: p["multiplier"] for p in _xi(mine)}
            their_xi = {p["element"]: p["multiplier"] for p in _xi(theirs)}
            unique = sum(pts.get(e, 0) for e in my_xi if e not in their_xi) - sum(
                pts.get(e, 0) for e in their_xi if e not in my_xi
            )
            captain = sum(pts.get(e, 0) * (m - 1) for e, m in my_xi.items()) - sum(
                pts.get(e, 0) * (m - 1) for e, m in their_xi.items()
            )
            hits = (
                theirs["entry_history"]["event_transfers_cost"]
                - mine["entry_history"]["event_transfers_cost"]
            )
            gap = net_points(mine) - net_points(theirs)
            if unique + captain + hits != gap:
                raise ValueError(
                    f"{name} v {leader} GW{gw}: terms sum to {unique + captain + hits}, "
                    f"gap is {gap}"
                )
            for bucket in (season, last) if gw in recent_gws else (season,):
                bucket["unique"] += unique
                bucket["captaincy"] += captain
                bucket["hits"] += hits
                bucket["gap"] += gap
        latest_mine = {p["element"] for p in picks[gws[-1]][name]["picks"]}
        latest_theirs = {p["element"] for p in picks[gws[-1]][leader]["picks"]}
        result[name] = {
            "season": season,
            "recent": last,
            "recent_gws": len(recent_gws),
            "only_mine": sorted(latest_mine - latest_theirs),
            "only_theirs": sorted(latest_theirs - latest_mine),
        }
    return result


def fixture_outlook(
    latest: dict[str, dict[str, Any]],
    fixtures: list[dict[str, Any]],
    player_team: dict[int, int],
    gws: list[int],
) -> dict[str, list[dict[str, Any]]]:
    """Average FPL difficulty (1 easy .. 5 hard) facing each 15-man squad, per upcoming gameweek."""
    by_gw_team: dict[int, dict[int, list[int]]] = {}
    for f in fixtures:
        if f["event"] in gws:
            teams = by_gw_team.setdefault(f["event"], {})
            teams.setdefault(f["team_h"], []).append(f["team_h_difficulty"])
            teams.setdefault(f["team_a"], []).append(f["team_a_difficulty"])
    result = {}
    for name, payload in latest.items():
        squad_teams = [player_team[p["element"]] for p in payload["picks"]]
        weeks = []
        for gw in gws:
            fixtures_for = [by_gw_team.get(gw, {}).get(team, []) for team in squad_teams]
            difficulties = [d for team_fixtures in fixtures_for for d in team_fixtures]
            weeks.append(
                {
                    "gw": gw,
                    "difficulty": round(sum(difficulties) / len(difficulties), 2)
                    if difficulties
                    else None,
                    "blanks": sum(1 for t in fixtures_for if not t),
                    "doubles": sum(1 for t in fixtures_for if len(t) > 1),
                }
            )
        result[name] = weeks
    return result


def bench_luck(picks: Picks, live: Live) -> dict[str, dict[str, int]]:
    """Bench points left unused (bench boost weeks excluded) and points gained from auto-subs."""
    result: dict[str, dict[str, int]] = {}
    for gw, rows in sorted(picks.items()):
        for name, payload in rows.items():
            row = result.setdefault(name, {"bench_unused": 0, "autosub_points": 0, "autosubs": 0})
            if payload["active_chip"] != "bboost":
                row["bench_unused"] += payload["entry_history"]["points_on_bench"]
            for sub in payload["automatic_subs"]:
                row["autosub_points"] += live[gw].get(sub["element_in"], 0)
                row["autosubs"] += 1
    return result


def head_to_head(picks: Picks) -> dict[str, dict[str, list[int]]]:
    """{a: {b: [wins, draws, losses]}} on weekly score after hits."""
    names = sorted({name for rows in picks.values() for name in rows})
    grid = {a: {b: [0, 0, 0] for b in names if b != a} for a in names}
    for rows in picks.values():
        for a in names:
            for b in names:
                if a == b:
                    continue
                diff = net_points(rows[a]) - net_points(rows[b])
                grid[a][b][0 if diff > 0 else 1 if diff == 0 else 2] += 1
    return grid


def most_captained(picks: Picks) -> dict[int, tuple[int, int]]:
    """{gw: (element, managers)} for the armband the league backed most each week."""
    result = {}
    for gw, rows in picks.items():
        counts = Counter(c["element"] for c in (_armband(p) for p in rows.values()) if c)
        if counts:
            result[gw] = counts.most_common(1)[0]
    return result
