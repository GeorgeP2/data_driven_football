"""Per-gameweek detail for each manager: star player, captain, transfers, chips, bench misses.

Picks and transfers are public once a gameweek's deadline has passed. Responses for *finished*
gameweeks are cached under cache/; an unfinished gameweek (bonus still provisional) is always
fetched fresh, so a mid-week run never freezes provisional points.
"""

import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

API = "https://fantasy.premierleague.com/api"
HTTP_TIMEOUT_S = 20
HTTP_ATTEMPTS = 3
CACHE_DIR = Path("data/league_cache")

CHIP_LABELS = {
    "wildcard": "Wildcard",
    "freehit": "Free hit",
    "bboost": "Bench boost",
    "3xc": "Triple captain",
}
GOALKEEPER = 1
# A benched player only counts as "missed" if they beat the weakest starter in the same role by
# this much; at 1 point every 0-point bench player shows up whenever a starter got -1.
MISSED_MARGIN = 2


def get_json(path: str) -> Any:
    last_error: Exception | None = None
    for attempt in range(HTTP_ATTEMPTS):
        try:
            with urllib.request.urlopen(f"{API}/{path}", timeout=HTTP_TIMEOUT_S) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise SystemExit(
        f"FPL API request failed after {HTTP_ATTEMPTS} attempts: {path} ({last_error})"
    )


def cached_json(path: str, cache_key: str, cacheable: bool) -> Any:
    target = CACHE_DIR / f"{cache_key}.json"
    if cacheable and target.exists():
        return json.loads(target.read_text())
    payload = get_json(path)
    if cacheable:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload))
    return payload


class GameweekSource:
    """Fetches and memoises the FPL payloads one export run needs."""

    def __init__(self) -> None:
        bootstrap = get_json("bootstrap-static/")
        self.events = {e["id"]: e for e in bootstrap["events"]}
        self.chips: list[dict[str, Any]] = bootstrap["chips"]
        self.teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
        self.players = {
            p["id"]: {
                "name": p["web_name"],
                "team": self.teams[p["team"]],
                "team_id": p["team"],
                "type": p["element_type"],
            }
            for p in bootstrap["elements"]
        }
        self._live: dict[int, dict[int, int]] = {}
        self._picks: dict[tuple[int, int], dict[str, Any]] = {}
        self._transfers: dict[int, list[dict[str, Any]]] = {}
        self._fixtures: list[dict[str, Any]] | None = None

    def finished(self, gw: int) -> bool:
        return bool(self.events[gw]["finished"])

    def live_points(self, gw: int) -> dict[int, int]:
        if gw not in self._live:
            payload = cached_json(f"event/{gw}/live/", f"live_gw{gw:02d}", self.finished(gw))
            self._live[gw] = {e["id"]: e["stats"]["total_points"] for e in payload["elements"]}
        return self._live[gw]

    def picks(self, entry: int, gw: int) -> dict[str, Any]:
        if (entry, gw) not in self._picks:
            self._picks[(entry, gw)] = cached_json(
                f"entry/{entry}/event/{gw}/picks/", f"picks_{entry}_gw{gw:02d}", self.finished(gw)
            )
        return self._picks[(entry, gw)]

    def fixtures(self) -> list[dict[str, Any]]:
        if self._fixtures is None:
            self._fixtures = get_json("fixtures/")
        return self._fixtures

    def transfers(self, entry: int) -> list[dict[str, Any]]:
        if entry not in self._transfers:
            self._transfers[entry] = get_json(f"entry/{entry}/transfers/")
        return self._transfers[entry]

    def name(self, element: int) -> str:
        return self.players.get(element, {}).get("name", f"#{element}")


def net_transfers(
    source: GameweekSource, entry: int, gw: int, live: dict[int, int]
) -> list[dict[str, Any]]:
    """The week's transfers as a net squad change, paired out->in by position.

    The endpoint logs every swap, so a wildcard or free hit reads A->B, B->C, C->A; only who
    actually left and arrived is a change. Position counts are fixed (2/5/5/3), so the net
    outs and ins of each position are equal in number and pair off.
    """
    moves = [t for t in source.transfers(entry) if t["event"] == gw]
    ins = Counter(t["element_in"] for t in moves)
    outs = Counter(t["element_out"] for t in moves)
    arrived, left = ins - outs, outs - ins

    def by_position(players: Counter[int]) -> dict[int, list[int]]:
        grouped: dict[int, list[int]] = {}
        for element in sorted(players.elements(), key=lambda e: -live.get(e, 0)):
            grouped.setdefault(source.players.get(element, {}).get("type", 0), []).append(element)
        return grouped

    arrived_by, left_by = by_position(arrived), by_position(left)
    pairs = []
    for position in sorted(set(arrived_by) | set(left_by)):
        for element_out, element_in in zip(
            left_by.get(position, []), arrived_by.get(position, []), strict=False
        ):
            pairs.append(
                {
                    "in": source.name(element_in),
                    "in_points": live.get(element_in, 0),
                    "out": source.name(element_out),
                    "out_points": live.get(element_out, 0),
                    "in_id": element_in,
                    "out_id": element_out,
                }
            )
    return pairs


def manager_gameweek(source: GameweekSource, entry: int, gw: int) -> dict[str, Any]:
    """One manager's gameweek, reconciled against FPL's own score."""
    picks = source.picks(entry, gw)
    live = source.live_points(gw)
    history = picks["entry_history"]

    scored = [(p, live.get(p["element"], 0)) for p in picks["picks"]]
    xi = [(p, pts) for p, pts in scored if p["multiplier"] > 0]
    # FPL's "points" is the gross score; hits come off total_points, not off this.
    total = sum(pts * p["multiplier"] for p, pts in xi)
    if total != history["points"]:
        raise ValueError(
            f"entry {entry} GW{gw}: picks add up to {total}, FPL says {history['points']}"
        )

    star_pick, star_pts = max(xi, key=lambda item: item[1] * item[0]["multiplier"])
    captain_pick, captain_pts = next(((p, pts) for p, pts in scored if p["is_captain"]), (None, 0))
    # Auto-subs move the armband to the vice; the multiplier says who actually carried it.
    carried = max(xi, key=lambda item: item[0]["multiplier"])
    armband = carried if carried[0]["multiplier"] > 1 else (captain_pick, captain_pts)

    missed = []
    if picks["active_chip"] != "bboost":
        for p, pts in scored:
            if p["multiplier"] > 0 or p["element"] in {
                s["element_in"] for s in picks["automatic_subs"]
            }:
                continue
            same_role = [
                q_pts
                for q, q_pts in xi
                if (q["element_type"] == GOALKEEPER) == (p["element_type"] == GOALKEEPER)
            ]
            if same_role and pts - min(same_role) >= MISSED_MARGIN:
                missed.append({"player": source.name(p["element"]), "points": pts})

    transfers = net_transfers(source, entry, gw, live)
    net = sum(t["in_points"] - t["out_points"] for t in transfers) - history["event_transfers_cost"]

    return {
        "points": history["points"],
        "net_points": history["points"] - history["event_transfers_cost"],
        "star": {
            "player": source.name(star_pick["element"]),
            "points": star_pts * star_pick["multiplier"],
        },
        "captain": {
            "player": source.name(armband[0]["element"]) if armband[0] else None,
            "points": armband[1]
            * (armband[0]["multiplier"] if armband[0] and armband[0]["multiplier"] > 1 else 2),
            "vice_stood_in": bool(
                captain_pick and armband[0] and armband[0]["element"] != captain_pick["element"]
            ),
        },
        "chip": CHIP_LABELS.get(picks["active_chip"] or "", picks["active_chip"]),
        "transfers": transfers,
        "hit": history["event_transfers_cost"],
        "transfer_net": net if transfers else None,
        "bench_points": history["points_on_bench"],
        "missed": sorted(missed, key=lambda m: -m["points"])[:2],
        "auto_subs": [
            {"in": source.name(s["element_in"]), "out": source.name(s["element_out"])}
            for s in picks["automatic_subs"]
        ],
    }


def league_gameweeks(
    source: GameweekSource, entries: dict[int, str], gws: list[int]
) -> dict[str, dict[str, Any]]:
    """{"<gw>": {name: manager_gameweek}} for every gameweek played so far."""
    return {
        str(gw): {name: manager_gameweek(source, entry, gw) for entry, name in entries.items()}
        for gw in gws
    }


def season_transfers(
    source: GameweekSource, entry: int, weeks: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Season-to-date transfer record for one manager, from their manager_gameweek rows.

    Each signing is scored over the weeks it was kept: from the week it arrived until the week
    it was sold again (or the latest gameweek), against what the player sold scored over the
    same weeks. A free hit squad reverts after one week, so its signings count for that week
    only. Points are raw player points: no captaincy, and bench weeks count the same as starts.
    """
    played = sorted(weeks)
    latest = played[-1]
    sold_at: dict[int, list[int]] = {}
    for gw in played:
        if weeks[gw]["chip"] != CHIP_LABELS["freehit"]:
            for t in weeks[gw]["transfers"]:
                sold_at.setdefault(t["out_id"], []).append(gw)

    moves = []
    for gw in played:
        row = weeks[gw]
        free_hit = row["chip"] == CHIP_LABELS["freehit"]
        for t in row["transfers"]:
            last = (
                gw
                if free_hit
                else min((s - 1 for s in sold_at.get(t["in_id"], []) if s > gw), default=latest)
            )
            span = range(gw, last + 1)
            gained_in = sum(source.live_points(w).get(t["in_id"], 0) for w in span)
            gained_out = sum(source.live_points(w).get(t["out_id"], 0) for w in span)
            moves.append(
                {
                    "gw": gw,
                    "in": t["in"],
                    "out": t["out"],
                    "weeks": len(span),
                    "in_points": gained_in,
                    "out_points": gained_out,
                    "gain": gained_in - gained_out,
                    "chip": row["chip"]
                    if row["chip"] in (CHIP_LABELS["wildcard"], CHIP_LABELS["freehit"])
                    else None,
                }
            )

    hits = sum(weeks[gw]["hit"] for gw in played)
    regular = [m for m in moves if m["chip"] is None]
    # Wildcard/free hit pairings are matched by position, not chosen, so best/worst use real
    # swaps only.
    ranked = sorted(regular, key=lambda m: m["gain"])
    return {
        "transfers": len(regular),
        "hits": hits,
        "chips": [{"gw": gw, "chip": weeks[gw]["chip"]} for gw in played if weeks[gw]["chip"]],
        "gain": sum(m["gain"] for m in moves) - hits,
        "gain_regular": sum(m["gain"] for m in regular) - hits,
        "gain_chips": sum(m["gain"] for m in moves if m["chip"]),
        "best": ranked[-1] if ranked and ranked[-1]["gain"] > 0 else None,
        "worst": ranked[0] if ranked and ranked[0]["gain"] < 0 else None,
        "moves": moves,
    }


def league_season_transfers(
    source: GameweekSource, entries: dict[int, str], gameweeks: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {
        name: season_transfers(
            source, entry, {int(gw): rows[name] for gw, rows in gameweeks.items()}
        )
        for entry, name in entries.items()
    }
