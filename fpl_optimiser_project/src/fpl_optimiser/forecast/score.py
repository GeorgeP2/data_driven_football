"""Score a recorded gameweek forecast against what actually happened.

    PYTHONPATH=src python -m fpl_optimiser.forecast.score 5

Reads records/gw<NN>_<season>.json, pulls the real points from the FPL API and
prints forecast-vs-actual per player. Also scores the squad the manager *actually* fielded,
which may differ from the recommendation the file records.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

API = "https://fantasy.premierleague.com/api"
RECORDS_DIR = Path("records")


def _get(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _load_record(gameweek: int) -> dict[str, Any]:
    matches = sorted(RECORDS_DIR.glob(f"gw{gameweek:02d}_*.json"))
    if not matches:
        available = sorted(p.name for p in RECORDS_DIR.glob("gw*.json"))
        raise SystemExit(f"No forecast recorded for GW{gameweek}. Available: {available or 'none'}")
    return json.loads(matches[0].read_text())


def _actual_points(gameweek: int) -> dict[int, int]:
    live = _get(f"{API}/event/{gameweek}/live/")
    return {e["id"]: e["stats"]["total_points"] for e in live["elements"]}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not argv[1].isdigit():
        raise SystemExit("usage: python -m fpl_optimiser.forecast.score <gameweek>   e.g. 5")
    gameweek = int(argv[1])

    record = _load_record(gameweek)
    entry_id = record["entry_id"]
    points = _actual_points(gameweek)

    print(f"=== GW{gameweek} {record['season']} — forecast vs actual (entry {entry_id}) ===\n")
    print(f"{'':4} {'player':<16} {'xP':>6} {'actual':>7} {'diff':>7}")

    forecast_total = 0.0
    actual_total = 0
    for row in record["xi"]:
        got = points.get(row["id"], 0)
        multiplier = 2 if row.get("captain") else 1
        xp = row["xp"] * multiplier
        got_mult = got * multiplier
        forecast_total += xp
        actual_total += got_mult
        flag = " (C)" if row.get("captain") else ""
        name = row["name"] + flag
        print(f"{row['pos']:<4} {name:<16} {xp:>6.2f} {got_mult:>7} {got_mult - xp:>+7.2f}")

    print(
        f"\n{'':4} {'RECOMMENDED XI':<16} {forecast_total:>6.2f} {actual_total:>7} "
        f"{actual_total - forecast_total:>+7.2f}"
    )

    print("\nBench (not counted, for auto-sub context):")
    for row in record["bench"]:
        print(
            f"  {row['order']}. {row['pos']:<4} {row['name']:<16} "
            f"xP {row['xp']:>5.2f}  actual {points.get(row['id'], 0):>3}"
        )

    # What was actually fielded — public once the deadline has passed.
    try:
        picks = _get(f"{API}/entry/{entry_id}/event/{gameweek}/picks/")
    except Exception as exc:
        print(f"\nCould not fetch the real picks ({exc}); recommendation scored only.")
        return 0

    fielded = sum(points.get(p["element"], 0) * p["multiplier"] for p in picks["picks"])
    history = picks["entry_history"]
    print("\n=== What was actually fielded ===")
    print(f"  squad points          {fielded}")
    print(
        f"  FPL-reported points   {history['points']} "
        f"(after a {history['event_transfers_cost']}-point hit)"
    )
    print(f"  recommendation        {actual_total} from the XI above")
    events = _get(f"{API}/bootstrap-static/")["events"]
    print(f"  gameweek average      {events[gameweek - 1]['average_entry_score']}")

    recommended_ids = {r["id"] for r in record["xi"]}
    actual_ids = {p["element"] for p in picks["picks"] if p["multiplier"] > 0}
    if recommended_ids != actual_ids:
        print(
            "\n  NOTE: the fielded XI differs from the recommendation, so the two "
            "totals above are not a like-for-like model score."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
