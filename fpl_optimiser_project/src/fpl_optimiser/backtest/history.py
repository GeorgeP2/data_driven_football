"""Load a completed season's gameweek-by-gameweek player data.

Source is the community archive (vaastav/Fantasy-Premier-League), which snapshots the FPL API
weekly and is the only place a *finished* season survives — the live API's `element-summary`
history is wiped when the next season loads.

The file is one row per player **per fixture**, which is the subtlety this module exists to
handle. In a double gameweek a player appears twice; in a blank they are absent entirely. Left
raw, that silently corrupts a dozen gameweeks of any backtest built on top.

Note what a fixture row carries: the *schedule* (who, where, how many games) is known before the
deadline and is legitimate model input, while the *result* (goals, points, minutes) is not. Both
are kept here, and `season.py` is responsible for handing out only the former.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

ARCHIVE_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/"
    "data/{season}/gws/merged_gw.csv"
)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fpl_optimiser" / "seasons"

_USER_AGENT = "fpl-optimiser/0.1"

_ELEMENT_TYPE_BY_POSITION = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
"""The archive writes `GK`; the live API's `element_types` call it `GKP`. Map both."""

_NOT_FOOTBALLERS = frozenset({"AM"})
"""Assistant Managers — the 2024-25 experiment where you could field Pep Guardiola. Scrapped
for 2025-26, and never a position in the squad rules we optimise against, so they are dropped.
They are also the reason a naive parser blows up on the 2024-25 archive."""


@dataclass(frozen=True)
class Fixture:
    """One match a player's team had in a gameweek."""

    opponent_team_id: int
    was_home: bool
    goals_for: int
    """Goals the player's team scored. An *outcome* — never a model input."""

    goals_against: int

    kickoff: datetime | None = None
    """When it kicked off. Needed to work out how much rest a club actually had — which, once
    European fixtures are merged in, is not what the gameweek number implies."""


@dataclass(frozen=True)
class GameweekRecord:
    """One player's complete record for one gameweek, with any double already aggregated."""

    player_id: int
    gameweek: int
    name: str
    element_type: int
    team_id: int
    team_name: str
    price_tenths: int
    """The player's price during this gameweek — what you would have paid at the deadline."""

    total_points: int
    minutes: int
    started: int
    """How many of their fixtures they started."""

    expected_goals: float
    expected_assists: float
    bonus: int
    """Bonus points actually awarded (0-3)."""

    saves: int
    defensive_contribution: int
    """Raw CBIT/CBIRT count, *not* points. Worth 2 points at 10 for a defender, 12 for a
    midfielder or forward — a 2025-26 addition, and one the old models all miss."""

    yellow_cards: int
    red_cards: int
    """Cards shown. Not for scoring — for suspensions, which are the one thing in this system
    that is a fact rather than a forecast."""

    bonus_points_system: int
    """BPS: the underlying score FPL ranks bonus on. The top three in a match take 3, 2 and 1
    bonus points — so bonus is a *rank within a fixture*, not a rate."""

    expected_goals_conceded: float
    """The team's expected goals against in this match, attributed to the player. A far steadier
    measure of defensive strength than the goals actually conceded."""

    fpl_expected_points: float | None
    """FPL's own pre-deadline forecast (`ep_next`), as snapshotted by the archive.

    **A benchmark, never a model input.** It is the yardstick our forecast has to beat, and the
    moment it leaks into a feature the comparison becomes meaningless. `None` where the archive
    did not capture it — which is most of 2025-26."""

    opponents: tuple[Fixture, ...]
    """One entry per match: empty in a blank gameweek, two in a double."""

    @property
    def fixture_count(self) -> int:
        return len(self.opponents)

    @property
    def played(self) -> bool:
        return self.minutes > 0

    @property
    def goals_conceded(self) -> int:
        return sum(f.goals_against for f in self.opponents)


def load_season(
    season: str,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
    timeout_seconds: float = 120.0,
) -> list[GameweekRecord]:
    """Return every (player, gameweek) record for a finished season, e.g. `"2025-26"`.

    Players absent from a gameweek (a blank) get an explicit zero-scoring record with no
    fixtures, so downstream code never has to distinguish "missing" from "played and scored
    nothing" — the FPL scoring rules treat them identically, and a `None` here would be a
    landmine.

    Raises:
        RuntimeError: if the archive cannot be fetched and nothing is cached.
        ValueError: if the CSV is missing columns the backtest needs.
    """
    raw = _read_csv(season, cache_dir=cache_dir, refresh=refresh, timeout=timeout_seconds)
    team_ids = _derive_team_ids(raw)
    records = _aggregate_fixtures(raw, season, team_ids)
    return _fill_blank_gameweeks(records)


def _read_csv(
    season: str, *, cache_dir: Path | None, refresh: bool, timeout: float
) -> list[dict[str, str]]:
    directory = DEFAULT_CACHE_DIR if cache_dir is None else cache_dir
    path = directory / f"{season}_merged_gw.csv"

    if refresh or not path.is_file():
        _download(season, path, timeout)

    with path.open(encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"The {season} archive is empty at {path}")

    required = {
        "element",
        "GW",
        "name",
        "position",
        "team",
        "value",
        "total_points",
        "minutes",
        "opponent_team",
        "was_home",
        "fixture",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"The {season} archive is missing required columns: {', '.join(sorted(missing))}"
        )
    return rows


def _download(season: str, path: Path, timeout: float) -> None:
    url = ARCHIVE_URL.format(season=season)
    logger.info("Downloading %s gameweek history from %s", season, url)
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(
                f"No archive for season {season!r} (HTTP 404). Seasons are named like '2025-26'."
            ) from exc
        raise RuntimeError(f"Archive returned HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach the season archive at {url}") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def _derive_team_ids(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    """Map team name to FPL team id, using the archive alone.

    The archive names a player's own team but gives the *opponent* as an id, so the two are
    never stated together. They can be recovered anyway: in a given fixture, the id one side
    calls its opponent is the other side's own id. Deriving it this way — rather than reading
    ids from the live `bootstrap-static` — means the mapping still works after the API rolls
    over to a new season and renumbers everything.
    """
    sides: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        if row["position"].strip().upper() in _NOT_FOOTBALLERS:
            continue
        sides.setdefault(row["fixture"], set()).add((row["team"], row["opponent_team"]))

    candidates: dict[str, set[str]] = {}
    for entries in sides.values():
        if len(entries) != 2:
            continue
        (name_a, opponent_a), (name_b, opponent_b) = sorted(entries)
        # Whatever B calls its opponent *is* A's id, and vice versa.
        candidates.setdefault(name_a, set()).add(opponent_b)
        candidates.setdefault(name_b, set()).add(opponent_a)

    team_ids: dict[str, int] = {}
    for name, ids in candidates.items():
        if len(ids) != 1:
            raise ValueError(
                f"Could not pin down a team id for {name!r}: the archive implies {sorted(ids)}"
            )
        team_ids[name] = int(next(iter(ids)))
    return team_ids


def _aggregate_fixtures(
    rows: Iterable[dict[str, str]], season: str, team_ids: dict[str, int]
) -> dict[tuple[int, int], GameweekRecord]:
    """Collapse one-row-per-fixture into one record per (player, gameweek).

    In a double gameweek a player has two rows; their points, minutes and underlying stats for
    the gameweek are the sum of both, and both opponents are kept. Price is a property of the
    player, not the fixture, so it is taken once.
    """
    records: dict[tuple[int, int], GameweekRecord] = {}
    dropped = 0

    for row in rows:
        if row["position"].strip().upper() in _NOT_FOOTBALLERS:
            dropped += 1
            continue

        try:
            player_id = int(row["element"])
            gameweek = int(row["GW"])
            points = int(row["total_points"])
            minutes = int(row["minutes"])
            price = int(row["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"The {season} archive has an unparseable row for element "
                f"{row.get('element')!r} in GW {row.get('GW')!r}: {exc}"
            ) from exc

        position = row["position"].strip().upper()
        if position not in _ELEMENT_TYPE_BY_POSITION:
            raise ValueError(
                f"Unknown position {position!r} for element {player_id} in GW {gameweek}; "
                f"expected one of: {', '.join(sorted(_ELEMENT_TYPE_BY_POSITION))}"
            )

        team_name = row["team"]
        if team_name not in team_ids:
            raise ValueError(f"No team id could be derived for {team_name!r}")

        was_home = row["was_home"].strip().lower() == "true"
        home_goals = _optional_int(row.get("team_h_score"))
        away_goals = _optional_int(row.get("team_a_score"))
        fixture = Fixture(
            opponent_team_id=int(row["opponent_team"]),
            was_home=was_home,
            kickoff=_parse_kickoff(row.get("kickoff_time")),
            goals_for=(home_goals if was_home else away_goals),
            goals_against=(away_goals if was_home else home_goals),
        )

        key = (player_id, gameweek)
        existing = records.get(key)
        if existing is None:
            records[key] = GameweekRecord(
                player_id=player_id,
                gameweek=gameweek,
                name=row["name"],
                element_type=_ELEMENT_TYPE_BY_POSITION[position],
                team_id=team_ids[team_name],
                team_name=team_name,
                price_tenths=price,
                total_points=points,
                minutes=minutes,
                started=_optional_int(row.get("starts")),
                expected_goals=_optional_float(row.get("expected_goals")),
                expected_assists=_optional_float(row.get("expected_assists")),
                bonus=_optional_int(row.get("bonus")),
                saves=_optional_int(row.get("saves")),
                defensive_contribution=_optional_int(row.get("defensive_contribution")),
                yellow_cards=_optional_int(row.get("yellow_cards")),
                red_cards=_optional_int(row.get("red_cards")),
                bonus_points_system=_optional_int(row.get("bps")),
                expected_goals_conceded=_optional_float(row.get("expected_goals_conceded")),
                fpl_expected_points=_benchmark_forecast(row.get("xP")),
                opponents=(fixture,),
            )
        else:
            # A double gameweek: same player, second fixture. Everything accumulates.
            records[key] = GameweekRecord(
                player_id=existing.player_id,
                gameweek=existing.gameweek,
                name=existing.name,
                element_type=existing.element_type,
                team_id=existing.team_id,
                team_name=existing.team_name,
                price_tenths=existing.price_tenths,
                total_points=existing.total_points + points,
                minutes=existing.minutes + minutes,
                started=existing.started + _optional_int(row.get("starts")),
                expected_goals=existing.expected_goals + _optional_float(row.get("expected_goals")),
                expected_assists=existing.expected_assists
                + _optional_float(row.get("expected_assists")),
                bonus=existing.bonus + _optional_int(row.get("bonus")),
                saves=existing.saves + _optional_int(row.get("saves")),
                defensive_contribution=existing.defensive_contribution
                + _optional_int(row.get("defensive_contribution")),
                yellow_cards=existing.yellow_cards + _optional_int(row.get("yellow_cards")),
                red_cards=existing.red_cards + _optional_int(row.get("red_cards")),
                bonus_points_system=existing.bonus_points_system + _optional_int(row.get("bps")),
                expected_goals_conceded=existing.expected_goals_conceded
                + _optional_float(row.get("expected_goals_conceded")),
                fpl_expected_points=_add_forecasts(
                    existing.fpl_expected_points, _benchmark_forecast(row.get("xP"))
                ),
                opponents=(*existing.opponents, fixture),
            )

    if dropped:
        logger.info(
            "Dropped %d non-footballer rows from %s (Assistant Managers)",
            dropped,
            season,
        )
    return records


def _fill_blank_gameweeks(
    records: dict[tuple[int, int], GameweekRecord],
) -> list[GameweekRecord]:
    """Give every player a record in every gameweek, scoring zero where they had no fixture.

    A blank gameweek is not missing data — it is a real, known outcome worth exactly zero
    points, and the model must be able to see that a player has no fixture at all.
    """
    if not records:
        return []

    gameweeks = sorted({gw for _, gw in records})
    by_player: dict[int, list[GameweekRecord]] = {}
    for record in records.values():
        by_player.setdefault(record.player_id, []).append(record)

    filled: list[GameweekRecord] = []
    for player_id, player_records in by_player.items():
        seen = {r.gameweek: r for r in player_records}
        # Carry the most recent known price forward into blanks, so a player who blanks in GW31
        # still has a sane price if you want to own them through it.
        last_known = min(player_records, key=lambda r: r.gameweek)
        for gameweek in gameweeks:
            played_this_week = seen.get(gameweek)
            if played_this_week is not None:
                last_known = played_this_week
                filled.append(played_this_week)
            else:
                filled.append(
                    GameweekRecord(
                        player_id=player_id,
                        gameweek=gameweek,
                        name=last_known.name,
                        element_type=last_known.element_type,
                        team_id=last_known.team_id,
                        team_name=last_known.team_name,
                        price_tenths=last_known.price_tenths,
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
                        # No fixture, so FPL forecast nothing. That is a real zero, not a gap.
                        fpl_expected_points=0.0,
                        opponents=(),
                    )
                )

    filled.sort(key=lambda r: (r.gameweek, r.player_id))
    return filled


def _benchmark_forecast(value: str | None) -> float | None:
    """FPL's own `ep_next`, or None where the archive did not capture it.

    The distinction matters. A missing snapshot is *unknown*, and must be excluded from the
    benchmark; a zero would silently drag FPL's apparent accuracy down and flatter us. The
    2025-26 archive is missing this for 27 of 38 gameweeks, so this is not a rare edge case.
    """
    if value is None or value == "":
        return None
    try:
        forecast = float(value)
    except ValueError:
        return None
    # The archive writes a literal 0.0 for gameweeks it failed to snapshot, so a whole gameweek
    # of zeroes is a missing capture rather than a forecast that nobody would score. Callers
    # detect that at the gameweek level; here we keep the value and let them decide.
    return forecast


def _add_forecasts(left: float | None, right: float | None) -> float | None:
    """Sum two forecasts across a double gameweek, preserving 'unknown'."""
    if left is None or right is None:
        return None
    return left + right


def _parse_kickoff(value: str | None) -> datetime | None:
    """The archive writes ISO-8601 with a trailing Z, which `fromisoformat` will not take."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _optional_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def _optional_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)
