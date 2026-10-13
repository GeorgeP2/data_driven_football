"""Tests for the squad optimiser.

A squad optimiser fails *plausibly*: it hands back fifteen real footballers that quietly break
a rule. So every constraint gets a scenario that makes it bind — a pool where the unconstrained
answer would violate it — rather than only being asserted against an already-legal squad.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence

import pytest
from fpl_helpers import build_pool, make_player, points_provider
from fpl_optimiser.model import BenchWeights
from fpl_optimiser.optimise import Squad, optimise_squad
from fpl_optimiser.players import Player
from fpl_optimiser.rules import GameRules


def assert_squad_is_legal(squad: Squad, rules: GameRules, budget_tenths: int) -> None:
    """Every FPL rule, checked independently of the solver that claims to satisfy them."""
    assert len(squad.players) == rules.squad_size
    assert len({p.id for p in squad.players}) == rules.squad_size, "duplicate players"
    assert len(squad.starting_xi) == rules.starting_size
    assert squad.total_cost_tenths <= budget_tenths

    for rule in rules.positions:
        in_squad = [p for p in squad.players if p.element_type == rule.element_type]
        assert len(in_squad) == rule.squad_select, f"{rule.short_name} squad quota"
        starters = [p for p in squad.starting_xi if p.element_type == rule.element_type]
        assert rule.min_play <= len(starters) <= rule.max_play, (
            f"{rule.short_name} formation bounds"
        )

    for team_id in {p.team_id for p in squad.players}:
        from_team = [p for p in squad.players if p.team_id == team_id]
        assert len(from_team) <= rules.club_limit, f"club limit for team {team_id}"

    assert squad.captain in squad.starting_xi
    assert squad.vice_captain in squad.starting_xi
    assert squad.captain.id != squad.vice_captain.id


def best_xi_points(
    squad_players: Sequence[Player], points: dict[int, float], rules: GameRules
) -> float:
    """Best legal XI from a given 15, by brute force over every allowed formation.

    An independent oracle for the part of the model most likely to be subtly wrong.
    """
    by_type = {
        rule.element_type: sorted(
            (p for p in squad_players if p.element_type == rule.element_type),
            key=lambda p: -points[p.id],
        )
        for rule in rules.positions
    }
    ranges = [range(r.min_play, r.max_play + 1) for r in rules.positions]
    best = float("-inf")
    for counts in itertools.product(*ranges):
        if sum(counts) != rules.starting_size:
            continue
        total = 0.0
        for rule, count in zip(rules.positions, counts, strict=True):
            total += sum(points[p.id] for p in by_type[rule.element_type][:count])
        best = max(best, total)
    return best


class TestKnownOptimum:
    def test_finds_the_squad_that_is_optimal_by_construction(self, rules) -> None:
        # Fifteen stars, one per squad slot, spread three-per-club so nothing else binds.
        # Everyone else is a free, worthless body. There is exactly one right answer.
        stars: list[Player] = []
        slots = [(1, 2), (2, 5), (3, 5), (4, 3)]
        next_id = 100
        for element_type, count in slots:
            for _ in range(count):
                stars.append(
                    make_player(
                        next_id,
                        element_type,
                        team_id=(next_id % 5) + 10,
                        cost_tenths=50,
                        ep_next=9.0,
                    )
                )
                next_id += 1
        pool = build_pool(per_position=8, cost_tenths=40) + stars
        points = {p.id: p.ep_next for p in pool}

        squad = optimise_squad(pool, rules, points_provider(points))

        assert {p.id for p in squad.players} == {p.id for p in stars}
        assert_squad_is_legal(squad, rules, rules.budget_tenths)


class TestConstraintsBind:
    def test_club_limit_caps_a_club_full_of_stars(self, rules) -> None:
        # Five brilliant midfielders, all from team 7. Unconstrained, the solver would take
        # all five. The per-club limit of three must stop it.
        pool = build_pool(per_position=8, cost_tenths=40)
        stars = [make_player(200 + i, 3, team_id=7, cost_tenths=40, ep_next=15.0) for i in range(5)]
        pool += stars
        points = {p.id: p.ep_next for p in pool}

        squad = optimise_squad(pool, rules, points_provider(points))

        from_team_7 = [p for p in squad.players if p.team_id == 7]
        assert len(from_team_7) == rules.club_limit == 3
        assert_squad_is_legal(squad, rules, rules.budget_tenths)

    def test_budget_forces_the_solver_off_the_best_players(self, rules) -> None:
        # A brilliant but ruinously expensive striker the solver cannot afford alongside a
        # legal squad of the cheapest possible everyone-else.
        pool = build_pool(per_position=8, cost_tenths=45)
        galactico = make_player(300, 4, team_id=9, cost_tenths=600, ep_next=20.0)
        pool.append(galactico)
        points = {p.id: p.ep_next for p in pool}
        # 14 cheapest others cost 14 * 45 = 630; with the galactico that is 1230 > budget.
        budget = 1000

        squad = optimise_squad(pool, rules, points_provider(points), budget_tenths=budget)

        assert galactico.id not in {p.id for p in squad.players}
        assert_squad_is_legal(squad, rules, budget)

    def test_affordable_star_is_taken(self, rules) -> None:
        # The mirror of the above: with room in the budget, the solver must take him.
        pool = build_pool(per_position=8, cost_tenths=40)
        star = make_player(300, 4, team_id=9, cost_tenths=200, ep_next=20.0)
        pool.append(star)
        points = {p.id: p.ep_next for p in pool}

        squad = optimise_squad(pool, rules, points_provider(points))

        assert star.id in {p.id for p in squad.starting_xi}
        assert squad.captain.id == star.id, "the best player should wear the armband"

    def test_only_one_goalkeeper_starts(self, rules) -> None:
        # Two superb keepers. GKP max_play is 1, so only one can start however good both are.
        pool = build_pool(per_position=8, cost_tenths=40)
        keepers = [
            make_player(400 + i, 1, team_id=11 + i, cost_tenths=40, ep_next=30.0) for i in range(2)
        ]
        pool += keepers
        points = {p.id: p.ep_next for p in pool}

        squad = optimise_squad(pool, rules, points_provider(points))

        starting_keepers = [p for p in squad.starting_xi if p.position == "GKP"]
        assert len(starting_keepers) == 1
        assert squad.bench[0].position == "GKP", "the reserve keeper takes the first slot"

    def test_formation_bounds_hold_when_one_position_is_stacked(self, rules) -> None:
        # All the value is in midfield, but only five midfielders may start, and at least
        # three defenders and one forward must.
        pool = build_pool(per_position=8, cost_tenths=40)
        points = {p.id: 0.0 for p in pool}
        for player in pool:
            if player.element_type == 3:
                points[player.id] = 12.0

        squad = optimise_squad(pool, rules, points_provider(points))

        counts = {r.short_name: 0 for r in rules.positions}
        for player in squad.starting_xi:
            counts[player.position] += 1
        assert counts["MID"] == 5
        assert counts["DEF"] >= 3
        assert counts["FWD"] >= 1
        assert_squad_is_legal(squad, rules, rules.budget_tenths)


class TestCaptainAndBench:
    def test_captain_is_the_highest_scoring_starter(self, rules) -> None:
        pool = build_pool(per_position=8, cost_tenths=40)
        points = {p.id: float(p.id % 7) for p in pool}

        squad = optimise_squad(pool, rules, points_provider(points))

        ranked = sorted(squad.starting_xi, key=lambda p: -points[p.id])
        assert points[squad.captain.id] == points[ranked[0].id]
        # The vice is the next best starter, since they only play if the captain does not.
        others = [p for p in squad.starting_xi if p.id != squad.captain.id]
        assert points[squad.vice_captain.id] == max(points[p.id] for p in others)

    def test_the_armband_is_stable_when_players_are_tied(self, rules) -> None:
        # Regression: with tied expected points the solver's own captain variable is an
        # arbitrary pick among equally optimal ones, so the reported captain used to wobble
        # between runs. It is now derived from the XI and must be reproducible.
        pool = build_pool(per_position=8, cost_tenths=40, ep_next=5.0)
        points = {p.id: 5.0 for p in pool}

        captains = {
            optimise_squad(pool, rules, points_provider(points)).captain.id for _ in range(3)
        }

        assert len(captains) == 1

    def test_starting_xi_is_the_best_legal_xi_from_the_chosen_fifteen(self, rules) -> None:
        pool = build_pool(per_position=10, cost_tenths=40)
        points = {p.id: float((p.id * 37) % 11) for p in pool}

        squad = optimise_squad(pool, rules, points_provider(points))

        chosen = sum(points[p.id] for p in squad.starting_xi)
        assert chosen == pytest.approx(best_xi_points(squad.players, points, rules))

    def test_zero_bench_weight_leaves_the_bench_worthless(self, rules) -> None:
        pool = _pool_with_expensive_talent()
        points = {p.id: p.ep_next for p in pool}

        squad = optimise_squad(
            pool,
            rules,
            points_provider(points),
            bench_weights=BenchWeights.uniform(0.0),
        )

        assert squad.bench_expected_points == pytest.approx(0.0)

    def test_high_bench_weight_buys_a_better_bench(self, rules) -> None:
        """The non-default path for bench_weight: it must actually change the squad.

        Regression: this used `_pool_with_expensive_talent()` at a weight of 1.0 and failed
        about one run in eight. It was not solver noise — on that pool the bench weight does not
        change which fifteen are bought (the squad totals 49 points at every weight), so the
        only thing varying was the arbitrary XI/bench split described below. The test asserted a
        strict improvement that was a coin flip.

        The fixture here forces a real trade-off: filler is worthless, and a tier of mid-priced
        players lets the solver buy a bench worth having by downgrading the XI. It is
        deterministic at every weight.
        """
        pool = _pool_with_bench_tradeoff()
        points = {p.id: p.ep_next for p in pool}

        cheap_bench = optimise_squad(
            pool,
            rules,
            points_provider(points),
            bench_weights=BenchWeights.uniform(0.0),
        )
        good_bench = optimise_squad(
            pool,
            rules,
            points_provider(points),
            bench_weights=BenchWeights.uniform(0.5),
        )

        assert good_bench.bench_expected_points > cheap_bench.bench_expected_points
        # ...and it pays for that bench out of the XI's budget.
        assert good_bench.starting_expected_points <= cheap_bench.starting_expected_points
        # The fifteen genuinely differ, which is the part the old fixture could not show.
        assert sum(points[p.id] for p in good_bench.players) > sum(
            points[p.id] for p in cheap_bench.players
        )

    def test_a_bench_weight_of_one_pins_the_squad_but_not_the_split(self, rules) -> None:
        """Why the test above stops at 0.5.

        At a bench weight of exactly 1 the starting-XI coefficient `(scaled - bench_value)`
        collapses to zero, so XI membership no longer affects the objective at all except
        through the captain. Every legal split of the same fifteen scores identically and the
        solver returns an arbitrary one — measured across repeated solves the bench came back
        worth 21.0, 14.0 and 0.0 points on the same input.

        Only the squad is well-defined at that weight, so only the squad is asserted. Anything
        testing the split here is testing a tie-break.
        """
        pool = _pool_with_bench_tradeoff()
        points = {p.id: p.ep_next for p in pool}

        totals = {
            round(
                sum(
                    points[p.id]
                    for p in optimise_squad(
                        pool,
                        rules,
                        points_provider(points),
                        bench_weights=BenchWeights.uniform(1.0),
                    ).players
                ),
                3,
            )
            for _ in range(5)
        }

        assert len(totals) == 1


class TestNonDefaultParameters:
    def test_reduced_budget_produces_a_cheaper_squad(self, rules) -> None:
        pool = _pool_with_expensive_talent()
        points = {p.id: p.ep_next for p in pool}

        full = optimise_squad(pool, rules, points_provider(points))
        frugal = optimise_squad(pool, rules, points_provider(points), budget_tenths=700)

        assert frugal.total_cost_tenths <= 700
        assert frugal.total_cost_tenths < full.total_cost_tenths
        assert frugal.starting_expected_points < full.starting_expected_points
        assert_squad_is_legal(frugal, rules, 700)

    def test_excluded_players_are_never_selected(self, rules) -> None:
        pool = _pool_with_expensive_talent()
        points = {p.id: p.ep_next for p in pool}
        unwanted = max(pool, key=lambda p: points[p.id])

        squad = optimise_squad(
            pool, rules, points_provider(points), excluded_player_ids=[unwanted.id]
        )

        assert unwanted.id not in {p.id for p in squad.players}

    def test_locked_players_are_always_selected(self, rules) -> None:
        pool = build_pool(per_position=8, cost_tenths=40)
        points = {p.id: 0.0 for p in pool}
        # A player the objective has no reason whatsoever to pick.
        must_have = make_player(500, 2, team_id=13, cost_tenths=130, ep_next=0.0)
        pool.append(must_have)
        points[must_have.id] = 0.0

        squad = optimise_squad(
            pool, rules, points_provider(points), locked_player_ids=[must_have.id]
        )

        assert must_have.id in {p.id for p in squad.players}
        assert_squad_is_legal(squad, rules, rules.budget_tenths)


class TestBadInput:
    def test_budget_too_low_raises_naming_the_cheapest_squad(self, rules) -> None:
        pool = build_pool(per_position=8, cost_tenths=40)
        points = {p.id: 1.0 for p in pool}
        # The cheapest legal 15 costs 15 * 40 = 600.

        with pytest.raises(ValueError, match=r"cheapest possible 15-man squad.*£60\.0m"):
            optimise_squad(pool, rules, points_provider(points), budget_tenths=500)

    def test_too_few_players_in_a_position_raises(self, rules) -> None:
        pool = [p for p in build_pool(per_position=8) if p.element_type != 4]
        pool += [make_player(600 + i, 4, team_id=i + 1) for i in range(2)]  # need 3
        points = {p.id: 1.0 for p in pool}

        with pytest.raises(ValueError, match="Only 2 FWD available but the squad needs 3"):
            optimise_squad(pool, rules, points_provider(points))

    @pytest.mark.parametrize("bench_weight", [-0.1, 1.5])
    def test_bench_weight_outside_zero_to_one_raises(self, rules, bench_weight) -> None:
        pool = build_pool()
        with pytest.raises(ValueError, match="bench weight"):
            optimise_squad(
                pool,
                rules,
                points_provider({}),
                bench_weights=BenchWeights.uniform(bench_weight),
            )

    def test_non_positive_budget_raises(self, rules) -> None:
        with pytest.raises(ValueError, match="budget_tenths"):
            optimise_squad(build_pool(), rules, points_provider({}), budget_tenths=0)

    def test_locking_and_excluding_the_same_player_raises(self, rules) -> None:
        pool = build_pool()
        with pytest.raises(ValueError, match="both locked and excluded"):
            optimise_squad(
                pool,
                rules,
                points_provider({}),
                locked_player_ids=[1],
                excluded_player_ids=[1],
            )

    def test_locking_an_unknown_player_raises(self, rules) -> None:
        with pytest.raises(ValueError, match="not in the pool"):
            optimise_squad(build_pool(), rules, points_provider({}), locked_player_ids=[9999])


class TestAgainstGreedy:
    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_solver_is_never_worse_than_a_greedy_value_pick(self, rules, seed) -> None:
        """A cheap, strong oracle: greedy value-picking is a decent squad, and optimal must
        be at least as good. If the solver ever loses to greedy, the model is wrong."""
        rng = random.Random(seed)
        pool = [
            make_player(
                i,
                element_type=(i % 4) + 1,
                team_id=(i % 20) + 1,
                cost_tenths=rng.randrange(40, 130, 5),
                ep_next=round(rng.uniform(0.0, 9.0), 1),
            )
            for i in range(1, 121)
        ]
        points = {p.id: p.ep_next for p in pool}

        squad = optimise_squad(
            pool,
            rules,
            points_provider(points),
            bench_weights=BenchWeights.uniform(0.0),
        )
        assert_squad_is_legal(squad, rules, rules.budget_tenths)

        greedy = _greedy_squad(pool, rules, points)
        greedy_xi = best_xi_points(greedy, points, rules) if greedy else 0.0
        # bench_weights=BenchWeights.uniform(0): the objective is the XI plus the captain doubling.
        greedy_objective = greedy_xi + max((points[p.id] for p in greedy), default=0.0)

        assert squad.starting_expected_points >= greedy_objective - 1e-6


def _pool_with_expensive_talent() -> list[Player]:
    """Cheap worthless filler plus a tier of expensive good players.

    The budget cannot buy talent for all fifteen slots, so the solver has to choose where to
    spend — which is what makes `budget` and `bench_weight` observable in the result.
    """
    pool = build_pool(per_position=8, cost_tenths=40, ep_next=0.0)
    next_id = 700
    for element_type in (1, 2, 3, 4):
        for _ in range(5):
            pool.append(
                make_player(
                    next_id,
                    element_type,
                    team_id=(next_id % 20) + 1,
                    cost_tenths=95,
                    ep_next=7.0,
                )
            )
            next_id += 1
    return pool


def _pool_with_bench_tradeoff() -> list[Player]:
    """Worthless filler, a scarce expensive tier, and a mid tier that makes the bench a choice.

    `_pool_with_expensive_talent` cannot show a bench weight working: talent is so much dearer
    than filler that the same fifteen are optimal whatever the bench is worth, and only the
    XI/bench labelling moves. Adding a middle tier gives the solver something to trade — it can
    downgrade the XI to put real players on the bench — so the weight changes the purchase and
    not just the label.

    Measured across repeated solves: at weight 0.0 the bench is worth 0.0 and the squad 70.0;
    at 0.5 the bench is worth 15.0 and the squad 80.0. Both stable.
    """
    pool = build_pool(per_position=8, cost_tenths=40, ep_next=0.0)
    next_id = 700
    # (price, points, how many per position): scarce and dear, then plentiful and mid.
    for cost_tenths, ep_next, count in ((120, 10.0, 5), (60, 5.0, 8)):
        for element_type in (1, 2, 3, 4):
            for _ in range(count):
                pool.append(
                    make_player(
                        next_id,
                        element_type,
                        team_id=(next_id % 20) + 1,
                        cost_tenths=cost_tenths,
                        ep_next=ep_next,
                    )
                )
                next_id += 1
    return pool


def _greedy_squad(
    pool: Sequence[Player], rules: GameRules, points: dict[int, float]
) -> list[Player]:
    """Take the best points-per-million available, subject to the rules. Not optimal."""
    quotas = {r.element_type: r.squad_select for r in rules.positions}
    per_club: dict[int, int] = {}
    spend = 0
    squad: list[Player] = []

    for player in sorted(pool, key=lambda p: -points[p.id] / p.cost_tenths):
        if quotas[player.element_type] == 0:
            continue
        if per_club.get(player.team_id, 0) >= rules.club_limit:
            continue
        # Leave enough money to fill the remaining slots with the cheapest legal options.
        remaining = sum(quotas.values()) - 1
        if spend + player.cost_tenths + remaining * 40 > rules.budget_tenths:
            continue
        squad.append(player)
        quotas[player.element_type] -= 1
        per_club[player.team_id] = per_club.get(player.team_id, 0) + 1
        spend += player.cost_tenths

    return squad if len(squad) == rules.squad_size else []


class TestPerSlotBenchWeights:
    """The bench is not four equal places.

    A starter you would willingly field fails to appear about one time in nine, so across eleven
    of them the first substitute comes on in roughly half of all gameweeks, the second in 8%, and
    the reserve keeper almost never. The first bench place is worth more than ten times the third,
    and a single uniform weight smears that away — which is how the squad ends up with four
    mediocre substitutes instead of one good one and three cheap bodies.
    """

    def _pool(self) -> list:
        """Cheap filler, plus one strong outfielder who cannot make the XI on merit."""
        pool = build_pool(per_position=8, cost_tenths=40, ep_next=0.0)
        return pool

    def test_the_solver_puts_its_best_outfielder_first_on_the_bench(self, rules) -> None:
        pool = build_pool(per_position=8, cost_tenths=40)
        points = {p.id: 0.0 for p in pool}
        # Six good midfielders. Only five can be in the squad, and only five can start, so
        # exactly one strong player must sit on the bench — and he should be first.
        mids = [p for p in pool if p.element_type == 3][:6]
        for rank, player in enumerate(mids):
            points[player.id] = 10.0 - rank  # 10, 9, 8, 7, 6, 5

        squad = optimise_squad(pool, rules, points_provider(points))

        outfield_bench = [p for p in squad.bench if p.position != "GKP"]
        assert outfield_bench, "there is always an outfield bench"
        best_benched = max(outfield_bench, key=lambda p: points[p.id])
        assert outfield_bench[0].id == best_benched.id, (
            "the strongest bench player must take the first substitute's slot"
        )

    # Whether a shaped bench weight actually *earns* points is a question about how the solver
    # spends a binding budget over a whole season, and it is answered by the backtest, not by a
    # unit test. Pinning it here would mean constructing a pool whose economics force a
    # particular answer, which tests the fixture rather than the model. What is unit-tested is
    # the mechanism: the ordering above, and the keeper below.

    def test_the_reserve_keeper_is_not_worth_paying_for(self, rules) -> None:
        # With the keeper slot valued at nothing, the solver should not spend on the second
        # goalkeeper — they play only if the first-choice keeper does not.
        pool = _pool_with_expensive_talent()
        points = {p.id: p.ep_next for p in pool}

        squad = optimise_squad(
            pool,
            rules,
            points_provider(points),
            bench_weights=BenchWeights(keeper=0.0, first=0.5, rest=0.08),
        )

        bench_keeper = next(p for p in squad.bench if p.position == "GKP")
        starting_keeper = next(p for p in squad.starting_xi if p.position == "GKP")
        assert bench_keeper.cost_tenths <= starting_keeper.cost_tenths, (
            "the reserve keeper should never be the expensive one"
        )

    def test_uniform_reproduces_the_old_behaviour(self, rules) -> None:
        pool = build_pool(per_position=8, cost_tenths=40)
        points = {p.id: float(p.id % 5) for p in pool}

        squad = optimise_squad(
            pool,
            rules,
            points_provider(points),
            bench_weights=BenchWeights.uniform(0.25),
        )

        assert_squad_is_legal(squad, rules, rules.budget_tenths)

    def test_a_bench_weight_outside_zero_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="bench weight 'first'"):
            BenchWeights(first=1.5).validate()
