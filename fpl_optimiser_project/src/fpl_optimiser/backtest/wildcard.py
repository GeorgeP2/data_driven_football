"""What a wildcard is actually worth, measured rather than assumed.

The live advisor passed `wildcard_gain=0.0` into `choose_chip`, so the branch tested `0.0 >= 15`
and the chip could never be suggested in any gameweek of any season. This computes the number.

**The rule is not "does a rebuild beat the squad I hold".** That question always answers yes and
would fire the chip every week. It is "does a rebuild beat *what free transfers reach anyway* in
the next few weeks" — a far smaller number, because the free-transfer path is greedy: it makes
the best move first, so most of the gain arrives without spending a chip.

`docs/wildcard-timing.md` derives a closed form for that difference — a transition drag `D` plus an
annuitised reachability term `rho` — under the constraint that the live horizon was one gameweek
and the multi-week path could not be observed. That constraint was lifted in September 2026, and
with a real horizon the difference does not need approximating: solve the horizon both ways and
subtract. Doing it this way drops the two weakest parts of the closed form — the `c(n)` recovery
curve, measured on a single gameweek, and the linear approximation the note itself measures as a
55% overstatement — and keeps the planner as the single definition of what a squad is worth.

It also gets the competitor right for free. The note's §4 warns that the real alternative to a
wildcard is not patience but *a hit* — taking a -4 to compress the queue — and that a rule
ignoring hits fires too often. The free-transfer solve prices hits itself, so if a hit is the
better answer the comparison already reflects it.

**Known conservative bias.** The gap is measured over the planner's horizon, not to the end of
the half, so a *permanent* advantage (the note's `rho`: squads a one-a-week path cannot walk to
on its own budget) is counted for five weeks rather than annuitised over the remaining season.
Where `rho` is genuinely positive this understates the chip. Understating is the right direction
to be wrong in — the floor exists because an earlier version fired a wildcard for a 5.6-point
gain and rebuilt into a worse squad — but it is a bias, not a neutrality.
"""

from __future__ import annotations

from collections.abc import Sequence

from fpl_optimiser.backtest.planner import plan_transfers
from fpl_optimiser.backtest.season import GameweekView
from fpl_optimiser.model import DEFAULT_BENCH_WEIGHTS, BenchWeights
from fpl_optimiser.rules import GameRules, TransferRules


def wildcard_gain(
    views: Sequence[GameweekView],
    points_by_gameweek: dict[int, dict[int, float]],
    rules: GameRules,
    transfer_rules: TransferRules,
    *,
    current_squad_ids: set[int],
    purchase_prices: dict[int, int],
    bank_tenths: int,
    free_transfers: int,
    bench_weights: BenchWeights = DEFAULT_BENCH_WEIGHTS,
) -> float:
    """Points the horizon is worth with a wildcard, over what it is worth without one.

    Two solves over the same pool, horizon and discount, differing only in how many transfers
    this week is allowed: the real free-transfer count, versus the whole squad. Every shared
    approximation — bench weighting, the discount, the shortlist — cancels in the subtraction,
    which is the reason to measure it this way rather than model the difference directly.

    Args:
        free_transfers: The real banked count. Passing a wrong one here does not merely shift
            the answer, it changes its sign of bias: understate it and the free-transfer path
            looks slower than it is, which makes the chip look better than it is.

    Returns:
        The gain in points, floored at zero. Never negative: a wildcard cannot leave you worse
        off than not playing it, since the unrestricted solve may always choose to stand still.

    Raises:
        ValueError: if the horizon is empty or the squad is not a legal starting point (raised
            by `plan_transfers`).
    """

    def _value(*, chip: bool) -> float:
        return plan_transfers(
            views,
            points_by_gameweek,
            rules,
            transfer_rules,
            current_squad_ids=current_squad_ids,
            purchase_prices=purchase_prices,
            bank_tenths=bank_tenths,
            free_transfers=free_transfers,
            wildcard=chip,
            bench_weights=bench_weights,
        ).horizon_value

    return max(0.0, _value(chip=True) - _value(chip=False))
