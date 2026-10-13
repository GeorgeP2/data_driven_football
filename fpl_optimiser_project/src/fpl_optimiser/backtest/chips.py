"""When to play the four chips.

You get eight in a season — a wildcard, free hit, bench boost and triple captain in each half —
and they expire. George's four quantifiable chips last season were worth about 45 points between
them, which is roughly what a *badly timed* bench boost is worth. Played well they are worth
several times that.

Chip timing is not an optimisation problem so much as a patience problem. The value of playing a
chip this week is easy to compute; the hard part is knowing whether to wait for something better.
So each chip has a threshold it must clear, and that threshold collapses to zero as the deadline
approaches — because an unplayed chip is worth exactly nothing, and at the last opportunity you
take whatever is on offer.

Deliberately kept as explicit, readable rules rather than folded into the integer program. The
IP would need a binary for every chip in every gameweek and a linearised product term for each
one, which triples the model for a decision that is made eight times a season and benefits far
more from being auditable than from being provably optimal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Chip(Enum):
    WILDCARD = "wildcard"
    FREE_HIT = "freehit"
    BENCH_BOOST = "bboost"
    TRIPLE_CAPTAIN = "3xc"


FIRST_HALF = range(1, 20)
SECOND_HALF = range(20, 39)

BENCH_BOOST_THRESHOLD = 18.0
"""Expected points from the bench needed to justify a bench boost. A typical bench is worth
about 8; this asks for roughly double, which in practice means a double gameweek."""

TRIPLE_CAPTAIN_THRESHOLD = 11.0
"""Expected points from the captain. A premium in a good single fixture forecasts about 6, so
this demands a double gameweek or something exceptional. It is set deliberately high: the first
version of this fired in GW2 and threw the chip away on a 49-point week."""

EARLIEST_CHIP_GAMEWEEK = 4
"""No chip before this. Early in a season the model is running almost entirely on priors and
last year's data — its forecasts are at their least trustworthy exactly when it is most tempted
to act on them. Backtesting caught this: unguarded, it burned the triple captain in GW2."""

FREE_HIT_MAX_PLAYERS = 9
"""Play the free hit when this many or fewer of your fifteen have a fixture — the classic use,
rescuing a blank gameweek without wrecking the squad to do it.

Crucially this is measured on the squad you hold *before* transfers. Measured afterwards it
never fires at all, because the planner has already quietly transferred its way around the
blank — which is a worse outcome bought with real transfers."""

WILDCARD_THRESHOLD = 25.0
"""Expected-points gain, over the planning horizon, of an unrestricted squad over the current
one. A wildcard is worth playing when the squad has drifted badly out of shape, not to chase a
single good fixture."""

MINIMUM_GAMEWEEKS_TO_EXPLOIT = 4
"""A wildcard rebuilds the squad for the *rest of the season*, so it needs a rest of the season
to pay off. Played in GW38 it is worth nothing at all — which is precisely what the first
version did."""

WILDCARD_FLOOR = 15.0
"""A wildcard is never worth playing for less than this, however close the deadline. Unlike the
other three chips, a wildcard is not free: it tears up a squad you have spent transfers
assembling. Letting the urgency decay collapse its threshold to nothing fired one for a 5.6-point
gain, and the rebuilt squad was worse."""


@dataclass
class ChipState:
    """Which chips are still in hand. Each is available once per half of the season."""

    used: dict[tuple[Chip, bool], int] = field(default_factory=dict)
    """Maps (chip, is_first_half) to the gameweek it was played."""

    def available(self, chip: Chip, gameweek: int) -> bool:
        return (chip, _is_first_half(gameweek)) not in self.used

    def play(self, chip: Chip, gameweek: int) -> None:
        if not self.available(chip, gameweek):
            raise ValueError(f"{chip.value} has already been played in this half of the season")
        self.used[(chip, _is_first_half(gameweek))] = gameweek

    @property
    def played(self) -> dict[int, Chip]:
        """Gameweek to chip, for reporting."""
        return {gameweek: chip for (chip, _), gameweek in self.used.items()}


def _is_first_half(gameweek: int) -> bool:
    return gameweek in FIRST_HALF


def last_chance(gameweek: int, remaining_gameweeks: list[int]) -> bool:
    """Is this the final gameweek in which the chip can still be played?

    An unplayed chip scores nothing, so on the last opportunity every threshold drops to zero.
    """
    half = FIRST_HALF if _is_first_half(gameweek) else SECOND_HALF
    later = [gw for gw in remaining_gameweeks if gw > gameweek and gw in half]
    return not later


def _urgency(gameweek: int, remaining_gameweeks: list[int]) -> float:
    """Scales a threshold down as the deadline nears: 1.0 early, 0.0 at the last chance.

    This is what stops the model hoarding chips into oblivion, which is the commonest way real
    managers waste them.
    """
    half = FIRST_HALF if _is_first_half(gameweek) else SECOND_HALF
    later = len([gw for gw in remaining_gameweeks if gw > gameweek and gw in half])
    if later <= 0:
        return 0.0
    # Ease off over the last five gameweeks of the half rather than falling off a cliff.
    return min(1.0, later / 5)


@dataclass(frozen=True)
class ChipDecision:
    chip: Chip | None
    reason: str


def choose_chip(
    gameweek: int,
    remaining_gameweeks: list[int],
    state: ChipState,
    *,
    bench_expected_points: float,
    captain_expected_points: float,
    squad_with_fixtures: int,
    wildcard_gain: float,
) -> ChipDecision:
    """Pick at most one chip to play this gameweek.

    Only one chip may be active at a time, so where several qualify the most valuable wins.

    Args:
        squad_with_fixtures: How many of the fifteen you *currently hold* have a match. Measured
            before transfers, deliberately — see `FREE_HIT_MAX_PLAYERS`.
    """
    if gameweek < EARLIEST_CHIP_GAMEWEEK:
        return ChipDecision(chip=None, reason="")

    urgency = _urgency(gameweek, remaining_gameweeks)
    candidates: list[tuple[float, Chip, str]] = []

    if state.available(Chip.FREE_HIT, gameweek) and squad_with_fixtures <= FREE_HIT_MAX_PLAYERS:
        # Roughly the points rescued: the players who would otherwise sit out entirely.
        gain = float(15 - squad_with_fixtures) * 2
        candidates.append(
            (
                gain,
                Chip.FREE_HIT,
                f"only {squad_with_fixtures} of your squad have a fixture",
            )
        )

    if (
        state.available(Chip.BENCH_BOOST, gameweek)
        and bench_expected_points >= BENCH_BOOST_THRESHOLD * urgency
    ):
        candidates.append(
            (
                bench_expected_points,
                Chip.BENCH_BOOST,
                f"the bench forecasts {bench_expected_points:.1f} pts",
            )
        )

    if (
        state.available(Chip.TRIPLE_CAPTAIN, gameweek)
        and captain_expected_points >= TRIPLE_CAPTAIN_THRESHOLD * urgency
    ):
        # The chip's value is the *extra* captain score, not the whole thing — it upgrades a
        # double to a treble. Scoring it as the full amount made it look better than a bench
        # boost when it is not.
        candidates.append(
            (
                captain_expected_points,
                Chip.TRIPLE_CAPTAIN,
                f"the captain forecasts {captain_expected_points:.1f} pts",
            )
        )

    weeks_left = len([gw for gw in remaining_gameweeks if gw > gameweek])
    if (
        state.available(Chip.WILDCARD, gameweek)
        and weeks_left >= MINIMUM_GAMEWEEKS_TO_EXPLOIT
        and wildcard_gain >= max(WILDCARD_THRESHOLD * urgency, WILDCARD_FLOOR)
    ):
        candidates.append(
            (
                wildcard_gain,
                Chip.WILDCARD,
                f"an unrestricted squad is worth {wildcard_gain:.1f} pts more",
            )
        )

    if not candidates:
        return ChipDecision(chip=None, reason="")

    _, chip, reason = max(candidates, key=lambda c: c[0])
    return ChipDecision(chip=chip, reason=reason)
