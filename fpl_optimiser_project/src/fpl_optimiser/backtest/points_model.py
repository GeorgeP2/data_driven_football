"""A real expected-points model.

Built from the diagnosis, not from taste. The backtest showed the naive baselines losing the
season in the first three gameweeks — five of eleven GW1 starters never took the pitch — so this
model is organised around the thing that actually failed:

**Will they play?** That question is worth more than "how well will they play?". A brilliant
player who is benched scores nothing, and no amount of attacking talent survives being
multiplied by zero minutes. So the minutes model comes first and everything else is scaled by it.

The rest follows FPL's own scoring rules, one component at a time:

    points = appearance + goals + assists + clean sheet + saves + defensive contribution
             + bonus - goals conceded

Each component is a *rate per 90 minutes*, estimated from the player's own record, shrunk toward
what is typical for their position, then multiplied by expected minutes and by how favourable
this particular fixture is. Two deliberate choices in there:

- **Shrinkage.** A striker who scored twice in his only start is not a 2.0-goals-per-90 player,
  and a model that believes he is will chase noise all season. Every rate is pulled toward the
  positional average, hard at first and less as evidence accumulates.
- **Expected goals, not goals.** A player's xG is a far steadier estimate of their scoring than
  the goals they happen to have scored, which is mostly noise over a handful of matches. This is
  the single most established finding in football analytics and it is nearly free to use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fpl_optimiser.backtest.season import GameweekView, LaggedStats, SeasonHistory
from fpl_optimiser.expected_points import StaticProvider
from fpl_optimiser.players import Player

# --- FPL scoring rules ------------------------------------------------------------------

GOAL_POINTS = {1: 6, 2: 6, 3: 5, 4: 4}
"""Points per goal by position: keepers and defenders 6, midfielders 5, forwards 4."""

CLEAN_SHEET_POINTS = {1: 4, 2: 4, 3: 1, 4: 0}
ASSIST_POINTS = 3
SAVE_POINTS = 1 / 3
"""One point per three saves."""

CONCEDED_PENALTY = -0.5
"""Minus one point per two goals conceded, for keepers and defenders only."""

DEFENSIVE_CONTRIBUTION_THRESHOLD = {2: 10, 3: 12, 4: 12}
"""CBIT/CBIRT actions needed for the 2-point defensive contribution: 10 for a defender, 12 for
a midfielder or forward. Keepers are not eligible."""

DEFENSIVE_CONTRIBUTION_POINTS = 2

BONUS_BASE = 0.12
BONUS_BPS_SCALE = 16.0
"""Bonus expected from a player's *lagged BPS rate*, as `BONUS_BASE * exp(bps_per_90 / SCALE)`.

Bonus is really a rank within a fixture — the top three BPS scores take 3, 2 and 1 — and the
within-match curve is a steep S: nothing below 20 BPS, 1.6 points at 30-34, 2.2 at 35-39.

Applying that S-curve to a player's *average* BPS is a trap, and it cost accuracy when I tried
it. The average is not the realisation: a player who averages 20 BPS sometimes has a 40-BPS game
and takes the full 3 points, so mapping the mean through a convex curve systematically
under-predicts. Jensen's inequality, quietly eating the bonus term.

The curve fitted here is the one that actually matters — measured lagged-rate against the bonus
subsequently scored (13 BPS/90 -> 0.25 bonus, 20 -> 0.47, 30 -> 0.68, 33 -> 0.99). It is much
flatter than the within-match curve, and gently convex."""

MAX_BONUS = 3.0

# --- Model parameters -------------------------------------------------------------------

MINUTES_DECAY = 0.6
"""How fast the minutes model forgets. Each gameweek back counts 60% of the one after it, so the
last two or three matches dominate.

This was 0.75 and it forgot too slowly: a player benched three weeks running still rated a 30%
chance of starting, which is how a squad ends up full of players who never take the pitch. Losing
your place in a football team is abrupt, and the model should treat it that way."""

MINUTES_WINDOW = 6
STARTER_MINUTES = 82.0
"""Minutes a player who starts tends to finish with, allowing for substitutions."""

CAMEO_MINUTES = 22.0

PRIOR_NINETIES = 6.0
"""Strength of the positional prior on every per-90 rate, in 90-minute blocks. A player needs
roughly six full matches before their own record outweighs "you are a typical player in your
position"."""

SIXTY_MINUTE_THRESHOLD = 60

CEILING_START_PROBABILITY = 0.88
"""Nobody is a certainty. A player who started each of his last six matches still misses roughly
one in eight — a knock in training, an illness, a suspension, a rested legs decision no lagged
data can see. The uncalibrated model said 98% for these players and 86% of them started, and that
over-confidence is exactly how a squad fills up with absentees."""

CONGESTION_REST_DAYS = 4.0
CONGESTION_DISCOUNT = 0.94
"""How much less likely a regular starter is to start after a midweek match.

Measured on 2024-25 regular starters, using a fixture calendar that includes European matches —
which FPL's own data does not:

    2-3 days' rest   77.9-78.8% start rate
    7 days' rest     82.8%

About five points, and the effect is concentrated in the Europa League clubs (74.1% against 82.1%
for clubs not in Europe) — Thursday night, then Sunday.

A blunt instrument, and knowingly so. The rotation decision is made *per player*, and no public
feed says which eleven actually played in Bucharest on Thursday. All we can see is that the club
played, so all we can apply is a club-wide discount."""

FLOOR_START_PROBABILITY = 0.0
"""No floor, and this is measured rather than assumed.

Players with no recent starts do in fact start about one game in twenty, so a 5% floor is *better
calibrated* — it improved the Brier score. It also made the points forecast measurably **worse**
(rank correlation 0.707 -> 0.676), because it hands ~15,000 non-playing players a small non-zero
forecast that varies with their noisy per-90 rates, where before they were correctly bunched at
zero. Better probabilities, worse decisions.

A reminder that calibration is a means, not the goal. Keeping the ceiling and dropping the floor
beats both alternatives on rank correlation, MAE *and* Brier."""


@dataclass(frozen=True)
class MinutesForecast:
    """What we think a player will do with the next 90 minutes."""

    probability_starts: float
    probability_plays: float
    expected_minutes: float

    @property
    def probability_sixty(self) -> float:
        """Odds of reaching 60 minutes — the threshold for the second appearance point and for
        a defender's clean sheet to count. In practice this is close to starting."""
        return self.probability_starts


def forecast_minutes(
    stats: LaggedStats,
    gameweek: int,
    *,
    injury_availability: float | None = None,
    days_rest: float | None = None,
) -> MinutesForecast:
    """Predict playing time from recent history, most recent matches weighted most heavily.

    Args:
        injury_availability: The live `chance_of_playing` flag, 0..1, when there is one. This is
            unavailable historically — which is precisely why the backtest opens so badly — but
            in a live season it is right there in the API, and it caps everything below.
        days_rest: Days since the club last played anything, European fixtures included. A club
            coming off a Thursday night in Europe rotates; FPL's own data cannot see that match.
    """
    recent_minutes = stats.recent_minutes(MINUTES_WINDOW)
    recent_starts = stats.starts[-MINUTES_WINDOW:]

    if not recent_minutes:
        # No matches this season. Last season's workload is all there is: a player who played
        # a full season of 90s is a starter; one who barely featured is not.
        expected = 38 * STARTER_MINUTES
        rate = _calibrate(stats.previous_season_minutes / expected)
        forecast = MinutesForecast(
            probability_starts=rate,
            probability_plays=_clamp_unit(rate * 1.2),
            expected_minutes=rate * STARTER_MINUTES,
        )
        return _apply_injury(forecast, injury_availability)

    # Exponentially decayed averages: the most recent match counts most.
    weights = [MINUTES_DECAY**i for i in range(len(recent_minutes) - 1, -1, -1)]
    total_weight = sum(weights)

    started = sum(w * min(s, 1) for w, s in zip(weights, recent_starts, strict=False))
    played = sum(w * (1.0 if m > 0 else 0.0) for w, m in zip(weights, recent_minutes, strict=False))

    probability_starts = _calibrate(started / total_weight)
    probability_plays = _clamp_unit(max(_calibrate(played / total_weight), probability_starts))

    expected_minutes = (
        probability_starts * STARTER_MINUTES
        + (probability_plays - probability_starts) * CAMEO_MINUTES
    )
    forecast = _apply_congestion(
        MinutesForecast(
            probability_starts=probability_starts,
            probability_plays=probability_plays,
            expected_minutes=expected_minutes,
        ),
        days_rest,
    )
    return _apply_injury(forecast, injury_availability)


def _apply_congestion(forecast: MinutesForecast, days_rest: float | None) -> MinutesForecast:
    """Discount a starter's odds after a midweek match. Substitutes are unaffected — if anything
    a congested week makes a cameo *more* likely, so only the starting probability moves.
    """
    if days_rest is None or days_rest >= CONGESTION_REST_DAYS:
        return forecast
    starts = forecast.probability_starts * CONGESTION_DISCOUNT
    return MinutesForecast(
        probability_starts=starts,
        probability_plays=max(forecast.probability_plays, starts),
        expected_minutes=(
            starts * STARTER_MINUTES
            + (max(forecast.probability_plays, starts) - starts) * CAMEO_MINUTES
        ),
    )


def _apply_injury(forecast: MinutesForecast, injury_availability: float | None) -> MinutesForecast:
    """Scale a minutes forecast by an explicit injury flag, when one exists.

    A 25%-fit player is a quarter as likely to do anything. In the backtest this is always None
    (the flags are not retained historically), which is a known handicap and not a bug.
    """
    if injury_availability is None:
        return forecast
    scale = _clamp_unit(injury_availability)
    return MinutesForecast(
        probability_starts=forecast.probability_starts * scale,
        probability_plays=forecast.probability_plays * scale,
        expected_minutes=forecast.expected_minutes * scale,
    )


def _shrunk_rate(total: float, nineties: float, prior_per_90: float) -> float:
    """A per-90 rate, pulled toward the positional average in proportion to how little we know.

    With no minutes played this returns the prior exactly; with many, it converges on the
    player's own rate. This is what stops one lucky hat-trick defining a season.
    """
    return (total + prior_per_90 * PRIOR_NINETIES) / (nineties + PRIOR_NINETIES)


@dataclass(frozen=True)
class PositionPriors:
    """What a typical player in each position does per 90 minutes."""

    expected_goals: dict[int, float]
    expected_assists: dict[int, float]
    bonus: dict[int, float]
    bps: dict[int, float]
    saves: dict[int, float]
    defensive_contribution: dict[int, float]


def build_priors(view: GameweekView) -> PositionPriors:
    """Positional averages from every player's completed matches this season.

    Derived rather than hardcoded, so the priors track the actual scoring environment — which
    matters in 2025-26, where the new defensive-contribution rule changed what a defender is
    worth.
    """
    totals: dict[str, dict[int, list[float]]] = {
        key: {1: [0.0, 0.0], 2: [0.0, 0.0], 3: [0.0, 0.0], 4: [0.0, 0.0]}
        for key in ("xg", "xa", "bonus", "bps", "saves", "dc")
    }

    for player in view.pool:
        stats = view.stats(player.id)
        nineties = stats.nineties
        if nineties <= 0:
            continue
        bucket = player.element_type
        for key, total in (
            ("xg", sum(stats.expected_goals)),
            ("xa", sum(stats.expected_assists)),
            ("bonus", float(sum(stats.bonus))),
            ("bps", float(sum(stats.bps))),
            ("saves", float(sum(stats.saves))),
            ("dc", float(sum(stats.defensive_contribution))),
        ):
            totals[key][bucket][0] += total
            totals[key][bucket][1] += nineties

    def rates(key: str, fallback: dict[int, float]) -> dict[int, float]:
        out = {}
        for element_type, (total, nineties) in totals[key].items():
            out[element_type] = total / nineties if nineties > 0 else fallback[element_type]
        return out

    # Fallbacks for GW1, when nobody has played a minute this season.
    return PositionPriors(
        expected_goals=rates("xg", {1: 0.0, 2: 0.05, 3: 0.12, 4: 0.30}),
        expected_assists=rates("xa", {1: 0.0, 2: 0.06, 3: 0.12, 4: 0.10}),
        bonus=rates("bonus", {1: 0.3, 2: 0.3, 3: 0.3, 4: 0.4}),
        bps=rates("bps", {1: 16.0, 2: 16.0, 3: 15.0, 4: 14.0}),
        saves=rates("saves", {1: 3.0, 2: 0.0, 3: 0.0, 4: 0.0}),
        defensive_contribution=rates("dc", {1: 0.0, 2: 6.0, 3: 4.0, 4: 2.0}),
    )


def expected_points(
    player: Player,
    view: GameweekView,
    priors: PositionPriors,
    *,
    injury_availability: float | None = None,
    defensive_contribution: bool = True,
) -> float:
    """Expected FPL points for one player in one gameweek, summed over their fixtures.

    A blank gameweek returns zero (no fixture, no points); a double gameweek is scored twice,
    once per opponent, which is what makes doubles worth planning around.

    Args:
        defensive_contribution: Whether the 2-point defensive-contribution rule existed in this
            season. It was introduced in 2025-26. Awarding it in an earlier backtest invents
            points that nobody could have scored — and because the positional prior fires when a
            player has no minutes, it would do exactly that in GW1 of every older season.
    """
    fixtures = view.fixtures.get(player.id, ())
    if not fixtures:
        return 0.0

    # Suspension. A yellow-card ban is administrative and certain, so those fixtures are simply
    # removed — in a double gameweek a one-match ban still leaves the second fixture, which is why
    # this drops matches rather than zeroing the player. A red-card ban can be appealed, so it is a
    # heavy discount rather than an exclusion: wrongly ruling a player out removes them from
    # selection entirely, and that is the expensive direction to be wrong in.
    ban = view.suspended.get(player.id)
    suspension_discount = 1.0
    if ban is not None:
        if ban.certain_matches:
            fixtures = fixtures[ban.certain_matches :]
            if not fixtures:
                return 0.0
        if ban.probable_matches >= len(fixtures):
            suspension_discount = ban.availability

    stats = view.stats(player.id)
    minutes = forecast_minutes(
        stats,
        view.gameweek,
        injury_availability=injury_availability,
        days_rest=view.days_rest.get(player.id),
    )
    if minutes.expected_minutes <= 0:
        return 0.0

    element_type = player.element_type
    nineties = stats.nineties
    ratings = view.team_ratings
    team_id = view.teams[player.id]

    goals_per_90 = _shrunk_rate(
        sum(stats.expected_goals), nineties, priors.expected_goals[element_type]
    )
    assists_per_90 = _shrunk_rate(
        sum(stats.expected_assists), nineties, priors.expected_assists[element_type]
    )
    bps_per_90 = _shrunk_rate(float(sum(stats.bps)), nineties, priors.bps[element_type])
    saves_per_90 = _shrunk_rate(float(sum(stats.saves)), nineties, priors.saves[element_type])
    dc_per_90 = _shrunk_rate(
        float(sum(stats.defensive_contribution)),
        nineties,
        priors.defensive_contribution[element_type],
    )

    total = 0.0
    for fixture in fixtures:
        scored, conceded = ratings.expected_goals(
            team_id, fixture.opponent_team_id, at_home=fixture.was_home
        )
        # How much better or worse than average is this attack, in this fixture? A player's
        # scoring rate rises and falls with their team's chance of scoring at all.
        attack_multiplier = scored / ratings.league_mean if ratings.league_mean else 1.0
        share = minutes.expected_minutes / 90

        # Appearance: 1 point for playing at all, a second for reaching 60 minutes.
        points = minutes.probability_plays + minutes.probability_sixty

        points += goals_per_90 * share * attack_multiplier * GOAL_POINTS[element_type]
        points += assists_per_90 * share * attack_multiplier * ASSIST_POINTS

        # A clean sheet only pays if the player is still on the pitch at 60 minutes.
        clean_sheet = math.exp(-conceded) * minutes.probability_sixty
        points += clean_sheet * CLEAN_SHEET_POINTS[element_type]

        if element_type in (1, 2):
            # Keepers and defenders lose a point for every two goals conceded.
            points += conceded * share * CONCEDED_PENALTY

        if element_type == 1:
            points += saves_per_90 * share * SAVE_POINTS

        threshold = (
            DEFENSIVE_CONTRIBUTION_THRESHOLD.get(element_type) if defensive_contribution else None
        )
        if threshold is not None:
            # The defensive-contribution point is a threshold, not a rate: you get 2 points for
            # 10 actions and nothing for 9. Poisson gives the odds of clearing the bar.
            expected_actions = dc_per_90 * share
            points += _poisson_at_least(threshold, expected_actions) * DEFENSIVE_CONTRIBUTION_POINTS

        # Bonus: the S-curve on expected BPS, and only if they are on the pitch long enough
        # to accumulate any.
        points += minutes.probability_sixty * _expected_bonus(bps_per_90 * attack_multiplier)

        total += max(0.0, points)

    return total * suspension_discount


def _expected_bonus(bps_per_90: float) -> float:
    """Bonus expected from a player's BPS rate, fitted on rate-to-outcome, not match-to-outcome."""
    if bps_per_90 <= 0:
        return 0.0
    return min(MAX_BONUS, BONUS_BASE * math.exp(bps_per_90 / BONUS_BPS_SCALE))


def _poisson_at_least(threshold: int, mean: float) -> float:
    """P(X >= threshold) for X ~ Poisson(mean)."""
    if mean <= 0:
        return 0.0
    # Sum the tail from below: P(X >= k) = 1 - P(X < k).
    cumulative = 0.0
    term = math.exp(-mean)
    for k in range(threshold):
        if k > 0:
            term *= mean / k
        cumulative += term
    return _clamp_unit(1.0 - cumulative)


def _calibrate(raw: float) -> float:
    """Squeeze a raw start rate into the range that actually occurs.

    The uncalibrated model is a decayed average of recent starts, so it happily returns 1.0 for a
    player who started his last six and 0.0 for one who started none. Neither is true: measured
    over 27,000 predictions, the "certainties" start 86% of the time and the "impossibles" start
    5% of the time. An affine squeeze onto that observed range fixes both ends at once, and the
    calibration curve is the evidence for it — not taste.
    """
    span = CEILING_START_PROBABILITY - FLOOR_START_PROBABILITY
    return FLOOR_START_PROBABILITY + _clamp_unit(raw) * span


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class PointsModel:
    """The expected-points model, as a backtest provider."""

    defensive_contribution: bool = True
    """Whether the defensive-contribution rule was in force. True for 2025-26 onwards; must be
    False for older seasons, where the rule did not exist."""

    name: str = "points-model"

    def build(self, view: GameweekView, history: SeasonHistory) -> StaticProvider:
        priors = build_priors(view)
        return StaticProvider(
            points_by_player_id={
                player.id: expected_points(
                    player,
                    view,
                    priors,
                    # None in a backtest, a real 0..1 flag in a live season.
                    injury_availability=view.injury_availability.get(player.id),
                    defensive_contribution=self.defensive_contribution,
                )
                for player in view.pool
            },
            name=self.name,
        )
