"""Score a squad against what actually happened, including automatic substitutions.

Auto-subs are not a detail. They are the *only* mechanism by which a good bench pays off, and
therefore the only way `bench_weight` — the one judgement-based knob in the optimiser — can be
tuned by evidence rather than taste. Without them, a worthless bench would look strictly optimal
and the backtest would confidently teach the wrong lesson.

The FPL rules, as implemented here:

- A starter who played zero minutes is replaced by the first player in bench order who played,
  provided the resulting XI is still a legal formation.
- Goalkeepers only ever replace goalkeepers.
- If the captain played zero minutes, the armband passes to the vice-captain. If neither played,
  nobody is doubled.
"""

from __future__ import annotations

from dataclasses import dataclass

from fpl_optimiser.backtest.history import GameweekRecord
from fpl_optimiser.model import Squad
from fpl_optimiser.players import Player
from fpl_optimiser.rules import GameRules


@dataclass(frozen=True)
class GameweekScore:
    """What a squad actually scored in one gameweek."""

    gameweek: int
    fielded_xi: tuple[Player, ...]
    """The XI after auto-subs — who actually earned you points."""

    substitutions: tuple[tuple[Player, Player], ...]
    """(came off, came on) pairs."""

    captain: Player | None
    """Whoever ended up doubled, or None if neither captain nor vice played."""

    points_before_hits: int
    transfer_hit_points: int = 0

    @property
    def points(self) -> int:
        return self.points_before_hits - self.transfer_hit_points


def score_gameweek(
    squad: Squad,
    actuals: dict[int, GameweekRecord],
    rules: GameRules,
    gameweek: int,
    *,
    transfer_hit_points: int = 0,
    bench_boost: bool = False,
    captain_multiplier: int = 2,
) -> GameweekScore:
    """Apply auto-subs and captaincy, then total the XI's actual points.

    Args:
        squad: The squad as it was set before the deadline.
        actuals: Outcomes for this gameweek, keyed by player id.
        rules: Used for the formation legality check when substituting.
        transfer_hit_points: Points already docked for transfers made this gameweek.
        bench_boost: All fifteen score, not just the eleven. With every player counting there is
            nothing to substitute *to*, so auto-subs do not apply.
        captain_multiplier: 2 normally, 3 with the triple captain chip.
    """
    played = {player_id: record.minutes > 0 for player_id, record in actuals.items()}

    def did_play(player: Player) -> bool:
        return played.get(player.id, False)

    def points_of(player: Player) -> int:
        record = actuals.get(player.id)
        # A player absent from the gameweek's records had no fixture: a blank, worth zero.
        return record.total_points if record is not None else 0

    substitutions: list[tuple[Player, Player]] = []

    if bench_boost:
        # Every player counts, so there is no bench to substitute from and nothing to gain by
        # trying. A blanking starter simply scores nothing, as does a blanking substitute.
        fielded = list(squad.players)
    else:
        fielded = list(squad.starting_xi)
        bench = list(squad.bench)
        for index, starter in enumerate(fielded):
            if did_play(starter):
                continue
            replacement = _find_replacement(starter, fielded, bench, rules, did_play)
            if replacement is None:
                continue
            fielded[index] = replacement
            bench.remove(replacement)
            substitutions.append((starter, replacement))

    captain: Player | None = None
    if did_play(squad.captain):
        captain = squad.captain
    elif did_play(squad.vice_captain):
        captain = squad.vice_captain

    total = sum(points_of(player) for player in fielded)
    if captain is not None:
        # The captain is already counted once as a member of the side; the armband adds their
        # score again (twice more, with the triple captain chip).
        total += points_of(captain) * (captain_multiplier - 1)

    return GameweekScore(
        gameweek=gameweek,
        fielded_xi=tuple(fielded),
        substitutions=tuple(substitutions),
        captain=captain,
        points_before_hits=total,
        transfer_hit_points=transfer_hit_points,
    )


def _find_replacement(
    starter: Player,
    fielded: list[Player],
    bench: list[Player],
    rules: GameRules,
    did_play,
) -> Player | None:
    """First bench player, in order, who played and whose arrival keeps the XI legal."""
    for candidate in bench:
        if not did_play(candidate):
            continue
        # Keepers are a closed shop: only a keeper can replace a keeper, and never otherwise.
        if (candidate.position == "GKP") != (starter.position == "GKP"):
            continue
        if _is_legal_formation([candidate if p is starter else p for p in fielded], rules):
            return candidate
    return None


def _is_legal_formation(xi: list[Player], rules: GameRules) -> bool:
    for rule in rules.positions:
        count = sum(1 for p in xi if p.element_type == rule.element_type)
        if not rule.min_play <= count <= rule.max_play:
            return False
    return True
