"""Score every logged forecast against what actually happened.

    PYTHONPATH=src python -m fpl_optimiser.forecast.calibrate

Reads every pool_gw<NN>_<season>.json written by `cli_advise --log-forecast`, joins each row
to the real points from the FPL API, and reports the three things that say whether the live
forecast is any good:

  * calibration  — mean actual / mean xP, overall and by position. 1.0 is unbiased. This is the
    metric that caught the defender bias in September 2026.
  * ranking      — Spearman rank correlation between xP and actual, against FPL's own `ep_next`
    over the same rows. Beating `ep_next` is the bar; it is free.
  * horizon      — both of the above split by how many weeks ahead the forecast was made. The
    planner trusts week+4 as much as week+0, so it matters whether that trust is earned.

Only gameweeks that have finished are scored; the rest are listed as pending.

A caution that belongs on every number below: one gameweek is ~650 correlated predictions, not
650 independent ones, and docs/project-notes.md puts the season noise band at +-30 points.
Early in a season these ratios move a long way on very little. Read the trend across weeks,
not one week.
"""

from __future__ import annotations

import json
import statistics
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

API = "https://fantasy.premierleague.com/api"
LOG_DIR = Path("outputs/forecast_log")
MIN_ROWS_TO_REPORT = 30
"""Below this a per-cell ratio is noise dressed as a measurement, so it is withheld."""


def _get(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _finished_gameweeks() -> set[int]:
    return {e["id"] for e in _get(f"{API}/bootstrap-static/")["events"] if e["finished"]}


def _actuals(gameweek: int) -> dict[int, int]:
    live = _get(f"{API}/event/{gameweek}/live/")
    return {e["id"]: e["stats"]["total_points"] for e in live["elements"]}


def _spearman(pairs: list[tuple[float, float]]) -> float | None:
    """Rank correlation, ties averaged. None when there is nothing to correlate."""
    if len(pairs) < MIN_ROWS_TO_REPORT:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    xs, ys = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return statistics.correlation(xs, ys)


def _ratio(rows: list[dict[str, Any]]) -> str:
    if len(rows) < MIN_ROWS_TO_REPORT:
        return f"n={len(rows)}"
    predicted = sum(r["xp"] for r in rows)
    if predicted <= 0:
        return "xP~0"
    return f"{sum(r['actual'] for r in rows) / predicted:.2f}"


def main() -> int:
    logs = sorted(LOG_DIR.glob("pool_gw*.json"))
    if not logs:
        raise SystemExit(
            "No pool logs found. Generate one with:\n"
            "  python -m fpl_optimiser.cli_advise --team-id <id> "
            "--log-forecast outputs/forecast_log"
        )

    finished = _finished_gameweeks()
    actuals_cache: dict[int, dict[int, int]] = {}
    scored: list[dict[str, Any]] = []
    pending: list[str] = []

    for log in logs:
        payload = json.loads(log.read_text())
        for row in payload["rows"]:
            gameweek = row["gameweek"]
            if gameweek not in finished:
                pending.append(f"GW{gameweek} (from {log.name})")
                continue
            if gameweek not in actuals_cache:
                actuals_cache[gameweek] = _actuals(gameweek)
            actual = actuals_cache[gameweek].get(row["id"])
            if actual is None:
                continue
            scored.append({**row, "actual": actual, "made_at": payload["forecast_for_gameweek"]})

    if not scored:
        print("Nothing scorable yet — every logged gameweek is still to be played.")
        print("Pending:", ", ".join(sorted(set(pending))) or "none")
        return 0

    print(
        f"=== Live forecast calibration — {len(scored)} predictions "
        f"over {len({r['gameweek'] for r in scored})} played gameweek(s) ===\n"
    )

    print(f"Overall calibration (actual/xP): {_ratio(scored)}   [1.00 = unbiased]")
    model_rho = _spearman([(r["xp"], r["actual"]) for r in scored])
    print(
        f"Rank correlation (all depths): {f'{model_rho:.4f}' if model_rho is not None else 'n/a'}"
    )

    # FPL publishes ep_next for the NEXT gameweek only, so the head-to-head is fair at depth 0
    # and meaningless beyond it — the same stale number would be scored against five results.
    imminent = [r for r in scored if r["horizon_depth"] == 0]
    head_to_head = _spearman([(r["xp"], r["actual"]) for r in imminent])
    ep_rho = _spearman([(r["ep_next"], r["actual"]) for r in imminent])
    if head_to_head is not None and ep_rho is not None:
        verdict = "model ahead" if head_to_head > ep_rho else "ep_next ahead"
        print(
            f"Next-gameweek head-to-head (depth 0, n={len(imminent)}): "
            f"model {head_to_head:.4f} vs ep_next {ep_rho:.4f}  ({verdict})"
        )
    elif imminent and all(r["ep_next"] == 0.0 for r in imminent):
        print(
            "Next-gameweek head-to-head: skipped — ep_next logged as all zeros "
            "(logs written before the 2026-09-18 fix)."
        )

    print("\nBy position:")
    by_position: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_position[row["position"]].append(row)
    for position in sorted(by_position):
        rows = by_position[position]
        print(
            f"  {position:<4} n={len(rows):>5}  actual/xP {_ratio(rows):<6} "
            f"mean xP {statistics.fmean(r['xp'] for r in rows):.2f}  "
            f"mean actual {statistics.fmean(r['actual'] for r in rows):.2f}"
        )

    print("\nBy horizon depth (weeks ahead when forecast):")
    by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_depth[row["horizon_depth"]].append(row)
    for depth in sorted(by_depth):
        rows = by_depth[depth]
        rho = _spearman([(r["xp"], r["actual"]) for r in rows])
        print(
            f"  +{depth}  n={len(rows):>5}  actual/xP {_ratio(rows):<6} "
            f"rank corr {f'{rho:.4f}' if rho is not None else 'n/a'}"
        )

    print("\nBy gameweek:")
    by_gameweek: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_gameweek[row["gameweek"]].append(row)
    for gameweek in sorted(by_gameweek):
        rows = by_gameweek[gameweek]
        print(f"  GW{gameweek:<3} n={len(rows):>5}  actual/xP {_ratio(rows)}")

    if pending:
        print("\nNot yet played:", ", ".join(sorted(set(pending))))

    print(
        "\nOne gameweek is ~650 correlated predictions, not 650 independent ones. Read the "
        "trend across weeks; a single week's ratio moves a long way on very little."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
