"""Who is banned, and for how many matches.

This is the only part of the system that is not a forecast. A suspended player has a zero percent
chance of playing, and that is arithmetic rather than prediction — so unlike everything else here,
it does not need a backtest to justify it.

It matters because suspensions are one of the ways a "nailed on" starter silently fails to appear.
Across 2024-25 the model was at least 80% sure about 4,900 players who then did not start; 65 of
those were serving bans it could have known about with certainty.

**The cutoff is the whole rule, and getting it wrong is worse than not having it.** A fifth yellow
card earns a one-match ban *only if it arrives inside the club's first 19 matches*. Measured
against the data:

    5th yellow within the cutoff   ->  missed the next gameweek  43/43   (100%)
    5th yellow past the cutoff     ->  missed the next gameweek   8/81   ( 10%)

Implement the thresholds without the cutoffs and you would confidently bench 81 players who went
on to play. That is not a small bug — a false "he is banned" removes a good player from selection
entirely, which is exactly why bad availability data is worse than none.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from fpl_optimiser.backtest.history import GameweekRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YellowCardThreshold:
    """A yellow-card total that earns a ban — but only if reached in time."""

    cards: int
    matches_banned: int
    cutoff_team_match: int
    """The ban applies only if the card is picked up within this many of the club's matches.
    Reach the total afterwards and nothing happens."""


YELLOW_THRESHOLDS: tuple[YellowCardThreshold, ...] = (
    YellowCardThreshold(cards=5, matches_banned=1, cutoff_team_match=19),
    YellowCardThreshold(cards=10, matches_banned=2, cutoff_team_match=32),
    YellowCardThreshold(cards=15, matches_banned=3, cutoff_team_match=38),
)

RED_CARD_BAN_MATCHES = 1
"""A sending-off costs at least one match. Straight reds for violent conduct carry three, but the
data records only that a red was shown, never why — so one is the floor."""

RED_CARD_BAN_SERVED = 0.85
"""How often a red-card ban actually costs the player the next gameweek. Measured: 40 times out of
47 in 2024-25.

Not 100%, and the difference matters. A yellow-card ban is administrative — the total is reached,
the ban is applied, nobody debates it — and lands 96% of the time. A red card can be rescinded on
appeal, downgraded on review, or served in a cup fixture, so it is a strong signal rather than a
fact.

That is why the two are kept apart below. Ruling a player out is an expensive mistake: it removes
them from selection entirely, and a wrong exclusion is worse than no information at all. Certainty
earns a hard exclusion; 85% earns a heavy discount."""


@dataclass(frozen=True)
class Suspension:
    """A ban, split by how sure we are of it."""

    certain_matches: int
    """Matches missed through yellow-card accumulation. Administrative and automatic."""

    probable_matches: int
    """Matches missed through a sending-off — likely, but subject to appeal."""

    @property
    def total_matches(self) -> int:
        return self.certain_matches + self.probable_matches

    @property
    def availability(self) -> float:
        """The discount for the *probable* part. The certain part drops fixtures outright."""
        return 1.0 - RED_CARD_BAN_SERVED if self.probable_matches else 1.0


def suspension_at(records: Sequence[GameweekRecord], gameweek: int) -> Suspension:
    """What ban, if any, this player is serving in `gameweek`.

    Args:
        records: The player's completed gameweeks, in any order. Only those *before* `gameweek`
            are read — a ban is earned by cards already shown, so this needs no lookahead. A card
            picked up *during* a gameweek cannot bench the player in that same gameweek, and
            reading it would be leaking the result into the team sheet.
    """
    completed = sorted((r for r in records if r.gameweek < gameweek), key=lambda r: r.gameweek)

    yellows = 0
    team_matches = 0
    certain = 0
    probable = 0

    for record in completed:
        fixtures = record.fixture_count

        # Serve any outstanding ban against this gameweek's fixtures. A player banned for two
        # matches who then has a double gameweek serves both at once. Yellow bans are served
        # first, being the ones actually enforced.
        served = min(certain, fixtures)
        certain -= served
        served_probable = min(probable, fixtures - served)
        probable -= served_probable

        team_matches += fixtures

        before = yellows
        yellows += record.yellow_cards
        for threshold in YELLOW_THRESHOLDS:
            crossed = before < threshold.cards <= yellows
            if crossed and team_matches <= threshold.cutoff_team_match:
                certain += threshold.matches_banned

        probable += record.red_cards * RED_CARD_BAN_MATCHES

    return Suspension(certain_matches=certain, probable_matches=probable)


def suspensions_at(
    records_by_player: dict[int, list[GameweekRecord]], gameweek: int
) -> dict[int, Suspension]:
    """Every player serving a ban this gameweek."""
    suspended = {}
    for player_id, records in records_by_player.items():
        ban = suspension_at(records, gameweek)
        if ban.total_matches > 0:
            suspended[player_id] = ban
    return suspended
