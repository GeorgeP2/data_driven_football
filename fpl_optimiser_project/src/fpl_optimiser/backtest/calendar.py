"""Every match a club plays — not just the ones FPL tells you about.

The FPL archive contains Premier League fixtures and nothing else. That is a bigger blind spot
than it sounds: a club that played in Bucharest on Thursday night and kicks off in the league on
Sunday looks, in FPL's data, like a club that has had a perfectly restful week. Congestion — the
thing that actually gets players rested — is *invisible*.

The evidence that this matters, measured on 2024-25 regular starters:

    no European football   start rate  82.1%
    Conference League                  79.8%
    Champions League                   77.8%
    Europa League                      74.1%   <- Thursday night, then Sunday

An 8-point swing for Europa League clubs, and not a trace of it in any FPL feed. This module goes
and gets the missing fixtures so the minutes model can see the week a player has actually had.

Source is `fixturedownload.com`, which publishes plain CSVs for every competition with no API key
and no registration.

## Read this before spending more time on congestion

**As a club-wide discount, this data is worth almost nothing, and that is measured, not guessed.**
Wiring true all-competition rest into the minutes model moved rank correlation by +0.0001 on
2024-25 — despite touching 24% of all player-gameweeks, so it is not a coverage problem.

The reason is worth understanding, because it says what data would *actually* help. A club-wide
discount shifts every Spurs player down together, which **preserves the ordering within the club**
— and within-club ordering is most of what the optimiser reads. Knowing Spurs are tired does not
tell you *which* Spurs player gets rested, and that is the only question that matters.

Rotation is a per-player decision. To exploit it we would need to know that *Bruno Fernandes*
played 90 minutes on Thursday, not merely that Manchester United did — and no fixture list can
tell you that. That needs player-level European match logs (FBref, Understat), which is a
different and considerably harder scrape.

The calendar is kept because it is correct, cached, and the prerequisite for that next step. The
small congestion discount it feeds is kept because it is principled and marginally improves mean
absolute error. Neither should be mistaken for a win.
"""

from __future__ import annotations

import csv
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

FIXTURE_URL = "https://fixturedownload.com/download/{competition}-{year}-UTC.csv"

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fpl_optimiser" / "calendar"

_USER_AGENT = "fpl-optimiser/0.1"

EUROPEAN_COMPETITIONS = ("champions-league", "europa-league", "conference-league")

_FIXTURE_NAME_TO_FPL = {
    # fixturedownload and FPL disagree on a handful of clubs. Everything else matches.
    "Tottenham": "Spurs",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton": "Wolves",
    "Brighton and Hove Albion": "Brighton",
    "Sheffield United": "Sheffield Utd",
    "Manchester United": "Man Utd",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "West Ham United": "West Ham",
    "Leicester City": "Leicester",
}


class FixtureCalendar:
    """Every match each club plays, across every competition, as kickoff times."""

    def __init__(self, kickoffs_by_team: dict[str, list[datetime]]) -> None:
        self._kickoffs = {team: sorted(times) for team, times in kickoffs_by_team.items()}

    def rest_before(self, team: str, kickoff: datetime) -> float | None:
        """Days since this club last played *anything*, or None if we have no earlier match."""
        earlier = [k for k in self._kickoffs.get(team, ()) if k < kickoff]
        if not earlier:
            return None
        return (kickoff - max(earlier)).total_seconds() / 86400

    def matches_within(self, team: str, kickoff: datetime, days: float) -> int:
        """How many matches this club played in the `days` before a kickoff."""
        window = kickoff - timedelta(days=days)
        return sum(1 for k in self._kickoffs.get(team, ()) if window <= k < kickoff)

    @property
    def teams(self) -> tuple[str, ...]:
        return tuple(sorted(self._kickoffs))


def season_start_year(season: str) -> int:
    """`"2024-25"` -> 2024, which is how fixturedownload names its files."""
    try:
        return int(season.split("-")[0])
    except (ValueError, IndexError):
        raise ValueError(f"Season {season!r} is not in the expected 'YYYY-YY' form") from None


def _download(competition: str, year: int, path: Path, timeout: float) -> None:
    url = FIXTURE_URL.format(competition=competition, year=year)
    logger.info("Downloading %s %d fixtures from %s", competition, year, url)
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"No fixture list for {competition} {year} (HTTP {exc.code})") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach the fixture source at {url}") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def _parse_kickoff(raw: str) -> datetime | None:
    """fixturedownload writes `17/09/2024 16:45`, in UTC."""
    try:
        naive = datetime.strptime(raw.strip(), "%d/%m/%Y %H:%M")
    except ValueError:
        return None
    return naive.replace(tzinfo=UTC)


def load_european_fixtures(
    season: str,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
    timeout_seconds: float = 60.0,
) -> dict[str, list[datetime]]:
    """Kickoff times for every European match played by an English club that season.

    A competition we cannot fetch is logged and skipped rather than raising: a calendar missing
    the Conference League is far better than no calendar at all, and silently *pretending* a club
    had a free midweek is the only outcome worth avoiding.
    """
    directory = DEFAULT_CACHE_DIR if cache_dir is None else cache_dir
    year = season_start_year(season)

    kickoffs: dict[str, list[datetime]] = {}
    for competition in EUROPEAN_COMPETITIONS:
        path = directory / f"{competition}-{year}.csv"
        if refresh or not path.is_file():
            try:
                _download(competition, year, path, timeout_seconds)
            except RuntimeError as exc:
                logger.warning(
                    "Skipping %s for %s: %s. Clubs in that competition will look better "
                    "rested than they were.",
                    competition,
                    season,
                    exc,
                )
                continue

        with path.open(encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                kickoff = _parse_kickoff(row.get("Date", ""))
                if kickoff is None:
                    continue
                for side in ("Home Team", "Away Team"):
                    name = _FIXTURE_NAME_TO_FPL.get(
                        row.get(side, "").strip(), row.get(side, "").strip()
                    )
                    kickoffs.setdefault(name, []).append(kickoff)

    return kickoffs


def build_calendar(
    league_kickoffs: dict[str, list[datetime]],
    season: str,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> FixtureCalendar:
    """Merge the league fixtures we already have with the European ones we do not.

    Args:
        league_kickoffs: Premier League kickoff times per club, from the FPL archive.
    """
    european = load_european_fixtures(season, cache_dir=cache_dir, refresh=refresh)

    merged: dict[str, list[datetime]] = {
        team: list(times) for team, times in league_kickoffs.items()
    }
    matched = 0
    for team, times in european.items():
        if team not in merged:
            # A foreign club. Expected — the European lists are full of them.
            continue
        merged[team].extend(times)
        matched += 1

    logger.info("Merged European fixtures for %d of %d clubs", matched, len(league_kickoffs))
    return FixtureCalendar(merged)
