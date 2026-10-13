"""The leakage chokepoint.

Every piece of information the backtest hands to a forecasting model passes through here, so
this is the one place lookahead can enter — and therefore the one place to guard it.

The line is not "everything about this gameweek is forbidden". The fixture *schedule* — who you
play, home or away, and how many matches you have — is published weeks ahead and is exactly what
a real manager plans around; withholding it would cripple the model for no honest reason. The
fixture *result* is what must never be seen. So `view_at(gw)` serves the schedule and all
completed-gameweek statistics, while `actuals_at(gw)` serves the outcomes and is reserved for
the scorer and the perfect-foresight ceiling.

The sharpest remaining leak is availability. FPL's `status` and `chance_of_playing_next_round`
are live-only fields; no historical record of who was injury-flagged before a past deadline
exists. The tempting substitute — "skip players who ended up with zero minutes" — *is reading
the answer*. So availability is predicted from lagged minutes alone, which is both leak-free and
exactly what a real minutes model must do anyway.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from fpl_optimiser.backtest.calendar import FixtureCalendar
from fpl_optimiser.backtest.history import Fixture, GameweekRecord
from fpl_optimiser.backtest.strength import TeamRatings, build_ratings
from fpl_optimiser.backtest.suspension import Suspension, suspensions_at
from fpl_optimiser.players import Player
from fpl_optimiser.rules import GameRules

AVAILABILITY_WINDOW = 3
"""How many completed gameweeks the availability proxy looks back over. Short enough to react
to a player losing their place, long enough not to panic at one rested match."""

FULL_APPEARANCE_MINUTES = 60
"""Minutes that count as a full appearance. 60 is the FPL threshold for the 2-point appearance
bonus, which makes it the natural cut between "played" and "cameo"."""


@dataclass(frozen=True)
class LaggedStats:
    """Everything known about a player *before* the gameweek in question.

    Each tuple is one entry per completed gameweek, oldest first, and they are all the same
    length — so `points[-1]` and `minutes[-1]` describe the same match.
    """

    player_id: int
    points: tuple[int, ...]
    minutes: tuple[int, ...]
    starts: tuple[int, ...]
    expected_goals: tuple[float, ...]
    expected_assists: tuple[float, ...]
    bonus: tuple[int, ...]
    bps: tuple[int, ...]
    saves: tuple[int, ...]
    defensive_contribution: tuple[int, ...]
    previous_season_points: int
    previous_season_minutes: int

    @property
    def nineties(self) -> float:
        """Ninety-minute blocks played this season — the denominator for every per-90 rate."""
        return self.total_minutes / 90

    @property
    def games_played(self) -> int:
        return sum(1 for m in self.minutes if m > 0)

    @property
    def total_points(self) -> int:
        return sum(self.points)

    @property
    def total_minutes(self) -> int:
        return sum(self.minutes)

    def recent_points(self, window: int) -> tuple[int, ...]:
        return self.points[-window:] if window > 0 else ()

    def recent_minutes(self, window: int) -> tuple[int, ...]:
        return self.minutes[-window:] if window > 0 else ()


@dataclass(frozen=True)
class GameweekView:
    """What a manager could legitimately have known at the gameweek `gameweek` deadline."""

    gameweek: int
    pool: tuple[Player, ...]
    lagged: dict[int, LaggedStats]
    fixtures: dict[int, tuple[Fixture, ...]]
    """This gameweek's schedule per player: empty in a blank, two entries in a double. The
    goals on these fixtures are the *result* and must not be read — only the opponent, the
    venue and the count."""

    team_ratings: TeamRatings
    teams: dict[int, int]
    """Player id to team id."""

    days_rest: dict[int, float] = field(default_factory=dict)
    """Days since the player's club last played *anything*, European fixtures included.

    Not the same as the gap between gameweeks: a club that played in Bucharest on Thursday and
    kicks off on Sunday has had two days' rest, not seven, and FPL's own data cannot see that."""

    suspended: dict[int, Suspension] = field(default_factory=dict)
    """Players serving a ban, split into the part that is certain (yellow-card accumulation) and
    the part that is merely likely (a sending-off, which can be appealed).

    In a double gameweek a one-match ban leaves them available for the other fixture, so this is a
    count of matches rather than a flag."""

    injury_availability: dict[int, float] = field(default_factory=dict)
    """Live `chance_of_playing` flags, 0..1, where they exist.

    Always empty in a backtest: FPL does not retain historical injury flags, which is exactly
    why the simulated manager fields players who were never going to start. In a *live* season
    this is populated straight from the API, and that whole class of blunder disappears. It is
    the one advantage the real advisor has over its own backtest."""

    def stats(self, player_id: int) -> LaggedStats:
        try:
            return self.lagged[player_id]
        except KeyError:
            raise ValueError(
                f"No lagged stats for player {player_id} at GW{self.gameweek}; they are not "
                f"in this gameweek's pool"
            ) from None

    def fixture_count(self, player_id: int) -> int:
        return len(self.fixtures.get(player_id, ()))


def availability_proxy(stats: LaggedStats, gameweek: int) -> float:
    """Probability the player features, predicted from lagged minutes only.

    A player who has completed each of the last few matches scores 1.0; one who has not appeared
    scores 0.0 and will not be picked. Cameos count partially. At GW1 there is nothing to go on
    but last season, so their minutes there stand in.
    """
    recent = stats.recent_minutes(AVAILABILITY_WINDOW)
    if not recent:
        if gameweek > 1:
            # In the pool but with no completed gameweeks behind them: a new signing. Nothing
            # to go on, so assume nothing.
            return 0.0
        # GW1: fall back on last season's workload. A full-time starter played ~38 x 90 minutes.
        expected = 38 * FULL_APPEARANCE_MINUTES
        return _clamp_unit(stats.previous_season_minutes / expected)
    return _clamp_unit(sum(min(m / FULL_APPEARANCE_MINUTES, 1.0) for m in recent) / len(recent))


class SeasonHistory:
    """A finished season, queryable as of any gameweek without leaking the future."""

    def __init__(
        self,
        records: Sequence[GameweekRecord],
        rules: GameRules,
        *,
        previous_season: Sequence[GameweekRecord] = (),
        calendar: FixtureCalendar | None = None,
    ) -> None:
        if not records:
            raise ValueError("Cannot build a SeasonHistory from an empty record list")

        self._rules = rules
        self._previous_records = list(previous_season)
        self._by_gameweek: dict[int, dict[int, GameweekRecord]] = {}
        for record in records:
            self._by_gameweek.setdefault(record.gameweek, {})[record.player_id] = record

        self.gameweeks: tuple[int, ...] = tuple(sorted(self._by_gameweek))

        self._previous_points: dict[int, int] = {}
        self._previous_minutes: dict[int, int] = {}
        for record in previous_season:
            self._previous_points[record.player_id] = (
                self._previous_points.get(record.player_id, 0) + record.total_points
            )
            self._previous_minutes[record.player_id] = (
                self._previous_minutes.get(record.player_id, 0) + record.minutes
            )

        self._calendar = calendar
        self._ratings_cache: dict[int, TeamRatings] = {}

        self._records_by_player: dict[int, list[GameweekRecord]] = {}
        for record in records:
            self._records_by_player.setdefault(record.player_id, []).append(record)

    def _require_gameweek(self, gameweek: int) -> dict[int, GameweekRecord]:
        try:
            return self._by_gameweek[gameweek]
        except KeyError:
            available = f"{self.gameweeks[0]}..{self.gameweeks[-1]}"
            raise ValueError(f"GW{gameweek} is not in this season; it covers {available}") from None

    def actuals_at(self, gameweek: int) -> dict[int, GameweekRecord]:
        """What actually happened. **Scoring and the perfect-foresight ceiling only.**

        Anything else calling this is leaking the future into a forecast.
        """
        return dict(self._require_gameweek(gameweek))

    def ratings_before(self, gameweek: int) -> TeamRatings:
        """Team ratings from completed gameweeks only. Cached: the same for every player."""
        cached = self._ratings_cache.get(gameweek)
        if cached is not None:
            return cached

        completed = [
            record
            for gw in self.gameweeks
            if gw < gameweek
            for record in self._by_gameweek[gw].values()
        ]
        ratings = build_ratings(completed, previous_season=self._previous_records)
        self._ratings_cache[gameweek] = ratings
        return ratings

    def view_at(self, gameweek: int, *, known_through: int | None = None) -> GameweekView:
        """The pool, schedule and lagged stats as they stood before the deadline.

        Prices are that gameweek's prices — a player's price is set before their match, so it is
        pre-deadline information. Points and minutes are strictly from *completed* gameweeks.

        Args:
            known_through: The last gameweek whose results are actually known. Defaults to
                `gameweek - 1`, which is what a backtest wants: it replays a finished season, so
                standing at GW5 it really does have GW4 in hand.

                A *live* horizon does not. Planning GW3 through GW7 from GW2, every one of those
                views knows the same thing — everything through GW2 — because GW3 to GW6 have
                not been played. Passing `gameweek - 1` there would read the unplayed gameweeks'
                placeholder records as completed matches in which every player logged zero
                minutes, and the minutes model would forecast a squad of players who never play.
        """
        current = self._require_gameweek(gameweek)
        cutoff = gameweek - 1 if known_through is None else known_through
        if cutoff >= gameweek:
            raise ValueError(
                f"known_through must be before the gameweek being planned; got "
                f"known_through={cutoff} for GW{gameweek}"
            )
        completed = [gw for gw in self.gameweeks if gw <= cutoff]

        lagged: dict[int, LaggedStats] = {}
        fixtures: dict[int, tuple[Fixture, ...]] = {}
        teams: dict[int, int] = {}
        days_rest: dict[int, float] = {}
        pool: list[Player] = []

        for player_id, record in current.items():
            history = [
                self._by_gameweek[gw][player_id]
                for gw in completed
                if player_id in self._by_gameweek[gw]
            ]
            stats = LaggedStats(
                player_id=player_id,
                points=tuple(r.total_points for r in history),
                minutes=tuple(r.minutes for r in history),
                starts=tuple(r.started for r in history),
                expected_goals=tuple(r.expected_goals for r in history),
                expected_assists=tuple(r.expected_assists for r in history),
                bonus=tuple(r.bonus for r in history),
                bps=tuple(r.bonus_points_system for r in history),
                saves=tuple(r.saves for r in history),
                defensive_contribution=tuple(r.defensive_contribution for r in history),
                previous_season_points=self._previous_points.get(player_id, 0),
                previous_season_minutes=self._previous_minutes.get(player_id, 0),
            )
            lagged[player_id] = stats
            # The schedule is pre-deadline information; the goals attached to it are not, and
            # nothing downstream is permitted to read them.
            fixtures[player_id] = record.opponents
            teams[player_id] = record.team_id

            # How much rest has this club actually had? The gameweek number will not tell you:
            # a Thursday night in Europe is invisible to FPL, and it is exactly what gets a
            # starter rested on the Sunday.
            if self._calendar is not None and record.opponents:
                kickoff = record.opponents[0].kickoff
                if kickoff is not None:
                    rest = self._calendar.rest_before(record.team_name, kickoff)
                    if rest is not None:
                        days_rest[player_id] = rest

            pool.append(
                Player(
                    id=player_id,
                    name=record.name,
                    element_type=record.element_type,
                    position=self._rules.position(record.element_type).short_name,
                    team_id=record.team_id,
                    team_name=record.team_name,
                    cost_tenths=record.price_tenths,
                    # FPL's injury flags are live-only, so there is no historical status to
                    # report. "?" says so, rather than quietly implying everyone was fit.
                    status="?",
                    availability=availability_proxy(stats, cutoff + 1),
                    # FPL's own forecast is not retained for past gameweeks. Any backtest
                    # provider must build its own; none may read this.
                    ep_next=0.0,
                    minutes=stats.total_minutes,
                    total_points=stats.total_points,
                )
            )

        pool.sort(key=lambda p: p.id)
        return GameweekView(
            gameweek=gameweek,
            pool=tuple(pool),
            lagged=lagged,
            fixtures=fixtures,
            team_ratings=self.ratings_before(cutoff + 1),
            teams=teams,
            days_rest=days_rest,
            # Suspensions read `gameweek`, not the cutoff, deliberately. A ban is served against
            # fixtures, and the unplayed gameweeks in between carry a real schedule with zero
            # cards — so counting them serves the ban down correctly and invents no new ones.
            suspended=suspensions_at(self._records_by_player, gameweek),
        )


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, value))
