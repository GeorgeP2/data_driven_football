"""Re-key last season's records to this season's ids.

FPL renumbers *both* players and teams every summer: element ids are reissued, and team ids are
handed out alphabetically over whichever twenty clubs are in the league, so promotion and
relegation shift every club below the change. Neither id is stable, and the archive does not
carry a stable `code`. So a previous season can only be joined to the current one by name — and
must be, on both axes, before it is used as a prior.

Getting the *team* half wrong is silent and expensive. The strength prior in `strength.py` is
keyed by team id, so an un-remapped carry-over hands each club last season's record of whoever
held its id: in 2026-27 that gave Ipswich Liverpool's attack, Man City Newcastle's, Sunderland
Wolves' and Bournemouth Burnley's. The model then rated every fixture against the wrong defence,
and the live advisor spent the opening weeks stacking Brentford at Bournemouth.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import replace

from fpl_optimiser.backtest.history import GameweekRecord

logger = logging.getLogger(__name__)

CLUBS_RELEGATED_PER_SEASON = 3
"""How many of last season's clubs can legitimately be missing this season. Any more unmatched
means the two sides no longer agree on names, not that the league changed shape."""


def normalise_name(name: str) -> str:
    """A name reduced to a cross-season join key: unaccented, lowercase, alphabetic.

    `"João Pedro"` and `"Joao Pedro"` are the same footballer, and the archive and the API do
    not agree on which spelling to use.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", stripped.lower())).strip()


def unique_ids_by_name(names_to_ids: Iterable[tuple[str, int]]) -> dict[str, int]:
    """Normalised name -> id, keeping only names that identify exactly one id.

    A name shared by two players is dropped rather than guessed: handing a player someone
    else's minutes is worse than the missing data it would paper over.
    """
    candidates: dict[str, set[int]] = {}
    for name, identifier in names_to_ids:
        candidates.setdefault(normalise_name(name), set()).add(identifier)
    return {name: next(iter(ids)) for name, ids in candidates.items() if len(ids) == 1}


def rekey_players(
    records: Sequence[GameweekRecord], current_ids_by_name: dict[str, int]
) -> tuple[list[GameweekRecord], int]:
    """Re-key player ids by name. Returns the records kept and how many players matched.

    `current_ids_by_name` maps *normalised* names to this season's ids (see
    `unique_ids_by_name`). Archive players whose name is absent or ambiguous on either side are
    dropped — newly promoted clubs and signings from abroad have no history to carry anyway.
    """
    archive_ids_by_name = unique_ids_by_name((record.name, record.player_id) for record in records)
    remap = {
        archive_id: current_ids_by_name[name]
        for name, archive_id in archive_ids_by_name.items()
        if name in current_ids_by_name
    }
    kept = [
        replace(record, player_id=remap[record.player_id])
        for record in records
        if record.player_id in remap
    ]
    return kept, len(remap)


def rekey_teams(
    records: Sequence[GameweekRecord],
    current_team_ids_by_name: dict[str, int],
    *,
    min_matches: int | None = None,
) -> list[GameweekRecord]:
    """Re-key team ids — a record's own club and each fixture's opponent — by team name.

    Records for clubs no longer in the league are dropped: there is no current id to give them,
    and the strength prior falls back on the league mean for a club it has no history of, which
    is the right answer for a promoted side. An opponent that has gone down keeps a distinct
    negative id, so two such fixtures in one gameweek do not collapse into one.

    Raises:
        ValueError: if fewer than `min_matches` of the archive's clubs are found this season —
            by default all but the relegated three — which means the names have changed shape
            rather than the league.
    """
    if not records:
        return []
    # The archive names a record's own club but gives opponents only as ids, so this table is
    # only complete over a whole season: re-key teams *before* filtering players, or a club whose
    # players all matched nothing vanishes from it and its fixtures lose their opponent.
    archive_names = {record.team_id: record.team_name for record in records}
    if min_matches is None:
        min_matches = max(1, len(archive_names) - CLUBS_RELEGATED_PER_SEASON)
    remap = {
        archive_id: current_team_ids_by_name[name]
        for archive_id, name in archive_names.items()
        if name in current_team_ids_by_name
    }
    if len(remap) < min_matches:
        raise ValueError(
            f"Only {len(remap)} of {len(archive_names)} clubs in the previous season match a "
            f"current team name, below the floor of {min_matches}. Relegation removes "
            f"{CLUBS_RELEGATED_PER_SEASON}; anything more means the archive's team names and "
            f"the current names disagree, "
            f"and carrying on would hand clubs each other's strength prior."
        )
    missing = sorted(name for tid, name in archive_names.items() if tid not in remap)
    if missing:
        logger.info("Not carrying over %s: no longer in the league", ", ".join(missing))

    def opponent(archive_id: int) -> int:
        return remap.get(archive_id, -archive_id)

    return [
        replace(
            record,
            team_id=remap[record.team_id],
            opponents=tuple(
                replace(fixture, opponent_team_id=opponent(fixture.opponent_team_id))
                for fixture in record.opponents
            ),
        )
        for record in records
        if record.team_id in remap
    ]


def carry_over_between_seasons(
    previous: Sequence[GameweekRecord], current: Sequence[GameweekRecord]
) -> list[GameweekRecord]:
    """Re-key an archive season to the ids of the archive season that followed it.

    For the backtest, where both seasons come from the archive and the current one is complete.
    The live path does the same join against `bootstrap-static` in `live.load_previous_season`.
    """
    teams = {record.team_name: record.team_id for record in current}
    players = unique_ids_by_name((record.name, record.player_id) for record in current)
    rekeyed, matched = rekey_players(rekey_teams(previous, teams), players)
    logger.info(
        "Carried over %d of %d previous-season players to the current season's ids",
        matched,
        len({record.player_id for record in previous}),
    )
    return rekeyed
