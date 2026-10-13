"""Build the model's view of the world from the *live* FPL API.

The backtest reads a finished season from an archive. The advisor has to assemble the same thing
from what the API exposes mid-season: each player's completed gameweeks (`element-summary`), the
fixtures still to come (`fixtures`), and today's prices and injury flags (`bootstrap-static`).

It deliberately reuses the backtest's `SeasonHistory` rather than growing a parallel code path.
The forecast the advisor gives you is then, by construction, produced by exactly the code the
backtest validated — which is the only reason to trust it.

One genuine advantage over the backtest: **injury flags**. `status` and
`chance_of_playing_next_round` are not retained historically, which is why the simulated manager
fields players who were never going to start. Live, they are right there, and that whole class of
blunder disappears.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fpl_optimiser.backtest.carryover import (
    normalise_name,  # noqa: F401  (re-exported)
    rekey_players,
    rekey_teams,
    unique_ids_by_name,
)
from fpl_optimiser.backtest.history import Fixture, GameweekRecord, load_season
from fpl_optimiser.backtest.season import SeasonHistory
from fpl_optimiser.expected_points import EpNextProvider
from fpl_optimiser.players import parse_players
from fpl_optimiser.rules import GameRules, TransferRules

logger = logging.getLogger(__name__)

ELEMENT_SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{id}/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
ENTRY_PICKS_URL = "https://fantasy.premierleague.com/api/entry/{team_id}/event/{event}/picks/"
ENTRY_HISTORY_URL = "https://fantasy.premierleague.com/api/entry/{team_id}/history/"

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fpl_optimiser" / "live"

_USER_AGENT = "fpl-optimiser/0.1"
_UNAVAILABLE_STATUSES = frozenset({"i", "s", "u"})

_UNLIMITED_TRANSFER_CHIPS = frozenset({"wildcard", "freehit"})
"""Chips that freeze the allowance: the week's transfers are free, and the count neither drains
nor accrues. Measured, not assumed — see `free_transfers_for`."""

OPENING_FREE_TRANSFERS = 1
"""What you carry into the gameweek *after* the one you entered on.

Your first gameweek is unlimited squad creation, and an unused one does **not** bank: enter at
GW1 making no transfers and you have one free transfer in GW2, not two. Getting this backwards
inflates the count by one for the whole season.
"""

MIN_CARRYOVER_MATCH_RATE = 0.5
"""How much of the live pool must join to last season's archive before we trust the carry-over.

Measured joining the 2025-26 archive onto the 2026-27 pool: **81% matched uniquely, none
ambiguously**, and 80 of the 105 misses were players at newly-promoted clubs who have no
Premier League history to carry. So the healthy band is roughly 75-85%, and it moves with how
many players the promoted clubs bring up.

The floor is set well below that band because it is guarding against a *different* failure: the
archive changing a column, or FPL changing its name format, which would send the rate to near
zero. That failure is silent and catastrophic — every player would carry zero minutes, which is
precisely the cold start this carry-over exists to fix — so it must raise rather than warn.
"""


def _get(url: str, timeout: float = 30.0) -> Any:
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"FPL API returned HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach the FPL API at {url}") from exc


def next_gameweek(payload: dict[str, Any]) -> int | None:
    """The gameweek the advisor should be advising on, or None between seasons."""
    for event in payload.get("events", []):
        if event.get("is_next"):
            return int(event["id"])
    return None


def injury_availability(payload: dict[str, Any]) -> dict[int, float]:
    """Each player's chance of featuring, from FPL's own injury flags.

    An explicit percentage wins; otherwise a hard-unavailable status is zero and everyone else
    is assumed fit. This is the information the backtest cannot have.
    """
    availability: dict[int, float] = {}
    for element in payload.get("elements", []):
        chance = element.get("chance_of_playing_next_round")
        status = str(element.get("status", "a"))
        if chance is not None:
            availability[int(element["id"])] = min(1.0, max(0.0, float(chance) / 100))
        elif status in _UNAVAILABLE_STATUSES:
            availability[int(element["id"])] = 0.0
    return availability


def _fetch_histories(
    player_ids: list[int],
    cache_dir: Path,
    refresh: bool,
    season: str,
    completed_through: int,
    *,
    cacheable: bool = True,
) -> dict[int, list[dict[str, Any]]]:
    """Every player's completed gameweeks. Cached, because this is ~840 requests.

    The cache is **per season and per completed gameweek**, and neither is a tidiness point.
    FPL renumbers element ids every summer, so a cache written last season and read this one
    hands each player the history of whoever happened to hold their id before — and because
    those players then look like they have current-season minutes, the previous-season
    carry-over never fires and the forecast runs on the wrong footballers entirely.

    And a cache keyed by season alone outlives the gameweek it was written in: one fetched on a
    Tuesday served every run for the following fortnight, so the advisor planned GW4 having
    never seen GW3. Keying it by `completed_through` makes a newly finished gameweek a cache
    miss, which is what `--refresh` was being relied on to do by hand.

    `cacheable=False` bypasses the cache in both directions. Pass it while `completed_through`
    is still being played: a file named `through_gw06` written on the Monday morning of GW6
    is missing Monday night's fixture and the provisional bonus, and every run until GW7 is
    finished would have reused it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"element_summaries_{season}_through_gw{completed_through:02d}.json"

    if cacheable and path.is_file() and not refresh:
        with path.open(encoding="utf-8") as handle:
            cached = json.load(handle)
        return {int(k): v for k, v in cached.items()}

    if not cacheable:
        logger.warning(
            "GW%d is not finished yet, so its rows are provisional and will not be cached. "
            "Re-run once FPL marks it finished.",
            completed_through,
        )
    logger.info("Fetching gameweek history for %d players...", len(player_ids))

    def fetch_one(player_id: int) -> tuple[int, list[dict[str, Any]]]:
        payload = _get(ELEMENT_SUMMARY_URL.format(id=player_id))
        return player_id, payload.get("history", [])

    # Eight at a time: enough to be quick, few enough to be a polite consumer of a free API.
    with ThreadPoolExecutor(max_workers=8) as pool:
        histories = dict(pool.map(fetch_one, player_ids))

    if cacheable:
        with path.open("w", encoding="utf-8") as handle:
            json.dump({str(k): v for k, v in histories.items()}, handle)
    return histories


def gameweek_is_finished(payload: dict[str, Any], gameweek: int) -> bool:
    """Whether FPL has marked `gameweek` finished — every fixture played, bonus confirmed.

    Gameweek 0 has nothing to finish, so it counts as finished. A gameweek missing from
    `events` counts as unfinished: the safe answer, since it only costs a re-fetch.
    """
    if gameweek == 0:
        return True
    return any(
        int(event["id"]) == gameweek and bool(event.get("finished"))
        for event in payload.get("events", [])
    )


def season_label(payload: dict[str, Any]) -> str:
    """The live season, as the archive names it, e.g. `"2026-27"`.

    Derived from the first gameweek deadline rather than hardcoded, so it stays correct without
    an annual edit: a season whose opening deadline falls in August 2026 is `"2026-27"`.

    Raises:
        ValueError: if the payload has no dated events to derive a season from.
    """
    deadlines = [
        str(event["deadline_time"])
        for event in payload.get("events", [])
        if event.get("deadline_time")
    ]
    if not deadlines:
        raise ValueError(
            "bootstrap-static has no event deadlines, so the season cannot be determined. "
            "Pass an explicit season instead."
        )
    start_year = int(min(deadlines)[:4])
    return f"{start_year}-{str(start_year + 1)[2:]}"


def previous_season_label(payload: dict[str, Any]) -> str:
    """The season before the live one, e.g. `"2025-26"` while 2026-27 is being played.

    Raises:
        ValueError: if the payload has no dated events to derive a season from.
    """
    start_year = int(season_label(payload).split("-")[0])
    return f"{start_year - 1}-{str(start_year)[2:]}"


def load_previous_season(
    payload: dict[str, Any],
    *,
    season: str | None = None,
    cache_dir: Path | None = None,
    refresh: bool = False,
    min_match_rate: float = MIN_CARRYOVER_MATCH_RATE,
) -> list[GameweekRecord]:
    """Last season's records, re-keyed to *this* season's element ids.

    This is what stops the advisor opening the season blind. `forecast_minutes` falls back on
    `previous_season_minutes` when a player has no completed gameweeks, and at GW1 nobody has
    any — so without this the whole pool forecasts a zero chance of starting and the squad the
    advisor returns is arbitrary.

    FPL renumbers element ids every season, and the archive does not carry the stable `code`, so
    the join has to go through names. It is deliberately an *exact* match on the normalised full
    name: a wrong match hands a player someone else's minutes, which is worse than the missing
    data it would be papering over. Players with no match — those at newly-promoted clubs, and
    signings from abroad — are simply left out, and the model treats them as it always has.

    Raises:
        ValueError: if `min_match_rate` is not a probability, or if the join matched a smaller
            share of the pool than that, which means the two sides no longer agree on names.
    """
    if not 0.0 <= min_match_rate <= 1.0:
        raise ValueError(f"min_match_rate must be between 0.0 and 1.0, got {min_match_rate}")

    label = previous_season_label(payload) if season is None else season
    records = load_season(label, cache_dir=cache_dir, refresh=refresh)

    # Team ids are reissued alphabetically each summer, and the strength prior is keyed by
    # them. Left as the archive's, every club below a promoted side inherits its neighbour's
    # record — see `backtest/carryover.py`. Teams first: the opponent table needs every club.
    live_teams = {str(team["name"]): int(team["id"]) for team in payload["teams"]}
    records = rekey_teams(records, live_teams)

    live_ids = unique_ids_by_name(
        (f"{element['first_name']} {element['second_name']}", int(element["id"]))
        for element in payload["elements"]
    )
    rekeyed, matched = rekey_players(records, live_ids)

    pool_size = len(payload["elements"])
    rate = matched / pool_size if pool_size else 0.0
    logger.info(
        "Carried over %s: matched %d of %d live players (%.1f%%) to last season's archive",
        label,
        matched,
        pool_size,
        rate * 100,
    )
    if rate < min_match_rate:
        raise ValueError(
            f"Only {matched} of {pool_size} live players ({rate:.1%}) joined to the {label} "
            f"archive, below the {min_match_rate:.0%} floor. The archive's name column or the "
            f"API's name fields have probably changed shape. Carrying on would silently give "
            f"every unmatched player zero previous-season minutes, which is the cold start "
            f"this carry-over exists to prevent."
        )
    return rekeyed


MODEL_FROM_GAMEWEEK = 2
"""The first gameweek the model's own forecast is used. Before it, FPL's `ep_next` stands in.

This is not a hedge, it is a measured result. Scored gameweek by gameweek on the rows FPL also
forecast, over two independently cold-started seasons:

```
             ours   ep_next
  GW1       0.009     0.468      2025-26  (n=690)
  GW1       0.027     0.443      2024-25  (n=616)
  GW2+      ~0.73      ~0.79     both
```

At GW1 the model has nothing but last season's per-90 rates, every one of them shrunk hard
toward its positional mean — so the pool collapses into near-ties and the ranking carries **no
signal at all**. 0.009 is not a weak forecast, it is a coin. From GW2 a single round of real
minutes restores it to ~0.73, and the remaining gap to `ep_next` is small enough that the live
injury flags — which the backtest cannot see, and which the availability-oracle experiment
valued far above it — plausibly close it.

So the handover is at GW2, and deliberately no later: there is no crossover after which the
model pulls ahead, and pretending otherwise would be reading noise. This guards the one
gameweek where the evidence is overwhelming.
"""


def early_season_forecast(payload: dict[str, Any], rules: GameRules) -> dict[int, float]:
    """FPL's own `ep_next`, availability-discounted, keyed by player id.

    What the advisor falls back on before `MODEL_FROM_GAMEWEEK`. It is a weak forecast — coarse
    and clustered — but at GW1 it is the only one of the two carrying any signal.
    """
    provider = EpNextProvider()
    return {player.id: provider(player) for player in parse_players(payload, rules)}


def load_live_history(
    payload: dict[str, Any],
    rules: GameRules,
    *,
    gameweek: int,
    horizon: int = 1,
    cache_dir: Path | None = None,
    refresh: bool = False,
    previous_season: Sequence[GameweekRecord] = (),
) -> SeasonHistory:
    """Assemble a `SeasonHistory` covering the gameweek being planned and the horizon after it.

    Each upcoming gameweek gets a record with the *schedule* filled in (opponent, venue, how many
    matches) and zeroes for everything that has not happened yet. Nothing reads those zeroes: the
    model only ever sees completed gameweeks plus the fixture list.

    Args:
        horizon: How many gameweeks to build forward, starting at `gameweek`. One means the next
            deadline only. The planner cannot look further ahead than this — it can only slice
            the gameweeks it is given, so a horizon of one silently reduces a five-week planner
            to a one-week one. Callers reading `view_at` on these must pass `known_through`;
            see that method.
        previous_season: Last season's records, already re-keyed to this season's element ids by
            `load_previous_season`. Empty means no carry-over, which is only sensible well into
            a season — at GW1 it leaves the minutes model with nothing to go on.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    directory = DEFAULT_CACHE_DIR if cache_dir is None else cache_dir

    elements = {int(e["id"]): e for e in payload["elements"]}
    teams = {int(t["id"]): t for t in payload["teams"]}
    histories = _fetch_histories(
        sorted(elements),
        directory,
        refresh,
        season_label(payload),
        completed_through=gameweek - 1,
        cacheable=gameweek_is_finished(payload, gameweek - 1),
    )

    records: list[GameweekRecord] = []
    for player_id, element in elements.items():
        team_id = int(element["team"])
        for row in histories.get(player_id, []):
            round_number = int(row["round"])
            if round_number >= gameweek:
                continue  # not yet played, as far as this plan is concerned
            was_home = bool(row["was_home"])
            home, away = row.get("team_h_score") or 0, row.get("team_a_score") or 0
            records.append(
                GameweekRecord(
                    player_id=player_id,
                    gameweek=round_number,
                    name=str(element["web_name"]),
                    element_type=int(element["element_type"]),
                    team_id=team_id,
                    team_name=str(teams[team_id]["short_name"]),
                    price_tenths=int(row["value"]),
                    total_points=int(row["total_points"]),
                    minutes=int(row["minutes"]),
                    started=int(row.get("starts") or 0),
                    expected_goals=float(row.get("expected_goals") or 0.0),
                    expected_assists=float(row.get("expected_assists") or 0.0),
                    bonus=int(row.get("bonus") or 0),
                    saves=int(row.get("saves") or 0),
                    defensive_contribution=int(row.get("defensive_contribution") or 0),
                    yellow_cards=int(row.get("yellow_cards") or 0),
                    red_cards=int(row.get("red_cards") or 0),
                    bonus_points_system=int(row.get("bps") or 0),
                    expected_goals_conceded=float(row.get("expected_goals_conceded") or 0.0),
                    # Live, we have no need of a benchmark — and no archive to read one from.
                    fpl_expected_points=None,
                    opponents=(
                        Fixture(
                            opponent_team_id=int(row["opponent_team"]),
                            was_home=was_home,
                            goals_for=int(home if was_home else away),
                            goals_against=int(away if was_home else home),
                        ),
                    ),
                )
            )

    # The upcoming gameweeks: schedule known, outcome unknown. Prices are today's throughout —
    # the planner already holds them fixed across the horizon rather than forecasting rises.
    for upcoming_gameweek in range(gameweek, gameweek + horizon):
        upcoming = _fixtures_for(upcoming_gameweek)
        if not upcoming:
            # Past the last scheduled gameweek of the season. Stop rather than emitting weeks in
            # which every club blanks, which the free-hit rule would read as a mass blank.
            logger.info(
                "GW%d has no fixtures; planning over GW%d-%d.",
                upcoming_gameweek,
                gameweek,
                upcoming_gameweek - 1,
            )
            break
        for player_id, element in elements.items():
            team_id = int(element["team"])
            opponents = tuple(
                Fixture(
                    opponent_team_id=opponent,
                    was_home=at_home,
                    goals_for=0,
                    goals_against=0,
                )
                for opponent, at_home in upcoming.get(team_id, [])
            )
            records.append(
                GameweekRecord(
                    player_id=player_id,
                    gameweek=upcoming_gameweek,
                    name=str(element["web_name"]),
                    element_type=int(element["element_type"]),
                    team_id=team_id,
                    team_name=str(teams[team_id]["short_name"]),
                    price_tenths=int(element["now_cost"]),
                    total_points=0,
                    minutes=0,
                    started=0,
                    expected_goals=0.0,
                    expected_assists=0.0,
                    bonus=0,
                    saves=0,
                    defensive_contribution=0,
                    yellow_cards=0,
                    red_cards=0,
                    bonus_points_system=0,
                    expected_goals_conceded=0.0,
                    fpl_expected_points=None,
                    opponents=opponents,
                )
            )

    return SeasonHistory(records, rules, previous_season=previous_season)


def _fixtures_for(gameweek: int) -> dict[int, list[tuple[int, bool]]]:
    """Each team's opponents in a gameweek: (opponent id, at home). Two entries in a double."""
    fixtures = _get(f"{FIXTURES_URL}?event={gameweek}")
    schedule: dict[int, list[tuple[int, bool]]] = {}
    for fixture in fixtures:
        home, away = int(fixture["team_h"]), int(fixture["team_a"])
        schedule.setdefault(home, []).append((away, True))
        schedule.setdefault(away, []).append((home, False))
    return schedule


def free_transfers_for(team_id: int, gameweek: int, transfer_rules: TransferRules) -> int:
    """How many free transfers you actually have banked going into `gameweek`.

    Replays the allowance forward from your first gameweek: you hold some number, spend
    `event_transfers` of them, and earn one more each week up to the cap. Every input is public,
    so this needs no authentication — which matters, because the alternative was typing it in by
    hand and the default was wrong more often than not.

    Getting this wrong is not a cosmetic error. The planner prices a hit at -4 against whatever
    it is told, so an under-count makes it decline moves that are in fact free, and an
    over-count makes it take hits it thinks are free. Under-counting is the failure that was
    actually happening: the advisor assumed 1 all season.

    Three rules, all of them checked against real entries rather than taken from the rulebook.
    Replaying 375 gameweeks of eight finished seasons and predicting each week's hit from the
    count reproduces `event_transfers_cost` exactly for every gameweek up to GW10:

    - **Your first gameweek does not bank.** It is unlimited squad creation, so entering at GW1
      and making no transfers leaves you **one** free transfer in GW2, not two. Three entries
      settle it: 2 transfers in GW2 cost -4 and 4 transfers cost -12, both of which need a
      count of one.
    - **A wildcard or free hit freezes the allowance.** It spends nothing *and earns nothing* —
      the "+1 anyway" reading is wrong, and costs a point of accuracy per chip played. Entry
      825485 wildcarded GW4 and free-hit GW5, then paid -8 for 3 transfers in GW6, which only
      works if it went into GW6 on one.
    - **The count floors at zero, it does not go negative.** Taking a -12 leaves you on zero and
      back to one next week, not on minus two.

    Known limitation: 9 of those 375 gameweeks still mispredict, all at GW16-19, where several
    unrelated entries made five transfers for nothing. That is the shape of a one-off allowance
    FPL granted mid-season, not a rule, so it is deliberately not modelled.

    Args:
        team_id: The FPL entry id.
        gameweek: The gameweek being planned — the count is as it stands *before* it.
        transfer_rules: Supplies the cap, which FPL has changed between seasons (2 before
            2024-25, 5 since), so it is read from the API rather than assumed.

    Returns:
        Free transfers banked, in `1..transfer_rules.max_free_transfers`.

    Raises:
        ValueError: if `gameweek` is not a gameweek number.
        RuntimeError: if the entry's history is not public.
    """
    if gameweek < 1:
        raise ValueError(f"gameweek must be >= 1, got {gameweek}")

    payload = _get(ENTRY_HISTORY_URL.format(team_id=team_id))
    seasons = payload.get("current")
    if seasons is None:
        raise RuntimeError(
            f"No history found for team {team_id}. Check the team id — the history endpoint is "
            f"public, so a 'not found' here means the entry does not exist."
        )

    transfers_by_gameweek = {
        int(entry["event"]): int(entry.get("event_transfers", 0)) for entry in seasons
    }
    chip_gameweeks = {
        int(chip["event"])
        for chip in payload.get("chips", [])
        if str(chip.get("name", "")) in _UNLIMITED_TRANSFER_CHIPS
    }

    played = [gw for gw in transfers_by_gameweek if gw < gameweek]
    if not played:
        # Nothing completed yet: either GW1, or an entry that starts this week. Both are
        # unlimited in practice; one is the conservative thing to tell the planner.
        return OPENING_FREE_TRANSFERS

    banked = OPENING_FREE_TRANSFERS
    for gw in range(min(played) + 1, gameweek):
        if gw in chip_gameweeks:
            continue
        spent = transfers_by_gameweek.get(gw, 0)
        banked = min(transfer_rules.max_free_transfers, max(0, banked - spent) + 1)
    return banked


def fetch_squad(
    team_id: int, gameweek: int, bootstrap: dict[str, Any]
) -> tuple[list[int], int, dict[int, int]]:
    """Your current squad: player ids, bank, and what you paid for each.

    **The purchase prices are estimates.** FPL publishes them only through the authenticated
    `/my-team/` endpoint; the public picks endpoint gives the players and nothing about what
    they cost you. So today's market price stands in, which means the sell-on fee is not
    applied and the planner will believe it has slightly more money than it really does — by
    half of whatever your squad has risen in value.

    In practice that is a few tenths of a million, and the recommendation is unlikely to turn
    on it. If it matters to you, pass the real figures in.

    Returns:
        (player ids, bank in tenths, purchase price per player in tenths).

    Raises:
        RuntimeError: if the team or gameweek is not public.
    """
    payload = _get(ENTRY_PICKS_URL.format(team_id=team_id, event=gameweek))
    picks = payload.get("picks")
    if not picks:
        raise RuntimeError(
            f"No squad found for team {team_id} in GW{gameweek}. Check the team id, and note "
            f"that a squad is only public after that gameweek's deadline."
        )

    prices = {int(e["id"]): int(e["now_cost"]) for e in bootstrap["elements"]}
    player_ids = [int(p["element"]) for p in picks]
    missing = [p for p in player_ids if p not in prices]
    if missing:
        raise RuntimeError(
            f"Players in your squad are not in the current player list: {missing}. The season "
            f"may have rolled over."
        )

    logger.warning(
        "Purchase prices are not public, so today's market prices are being used instead. The "
        "sell-on fee is therefore not applied and the budget is a slight over-estimate."
    )
    purchase = {player_id: prices[player_id] for player_id in player_ids}
    bank = int(payload.get("entry_history", {}).get("bank", 0))
    return player_ids, bank, purchase
