import pytest
from fpl_optimiser.leagues import insights

GK, DEF, MID, FWD = 1, 2, 3, 4


def payload(xi, bench=(), cost=0, chip=None, autosubs=(), live=None):
    """Build a picks payload. xi/bench are (element, multiplier, element_type) tuples."""
    picks = [
        {"element": e, "multiplier": m, "element_type": t, "is_captain": m > 1}
        for e, m, t in [*xi, *bench]
    ]
    live = live or {}
    points = sum(live.get(p["element"], 0) * p["multiplier"] for p in picks)
    on_bench = sum(live.get(e, 0) for e, m, _ in bench if m == 0)
    return {
        "picks": picks,
        "active_chip": chip,
        "automatic_subs": [{"element_in": i, "element_out": o} for i, o in autosubs],
        "entry_history": {
            "points": points,
            "event_transfers_cost": cost,
            "points_on_bench": on_bench,
        },
    }


LIVE = {1: 2, 2: 6, 3: 10, 4: 1, 5: 8, 6: 3, 7: 12}


def test_captaincy_loss_against_best_in_xi():
    week = payload([(1, 1, GK), (2, 2, MID), (3, 1, FWD)], live=LIVE)
    result = insights.captaincy({1: {"a": week}}, {1: LIVE})["a"]
    assert result["points"] == 12  # player 2: 6 x 2
    assert result["best"] == 20  # player 3 would have been 10 x 2
    assert result["lost"] == 8


def test_captaincy_uses_triple_multiplier_for_best():
    week = payload([(1, 1, GK), (2, 3, MID), (3, 1, FWD)], chip="3xc", live=LIVE)
    result = insights.captaincy({1: {"a": week}}, {1: LIVE})["a"]
    assert (result["points"], result["best"]) == (18, 30)


def test_gap_terms_sum_to_points_difference():
    mine = payload([(1, 1, GK), (2, 2, MID), (5, 1, FWD)], cost=4, live=LIVE)
    theirs = payload([(1, 1, GK), (3, 2, MID), (6, 1, FWD)], live=LIVE)
    gap = insights.gap_to_leader({1: {"me": mine, "lead": theirs}}, {1: LIVE}, "lead")
    season = gap["me"]["season"]
    assert season == {
        "unique": (6 + 8) - (10 + 3),
        "captaincy": 6 - 10,
        "hits": -4,
        "gap": (2 + 12 + 8 - 4) - (2 + 20 + 3),
    }
    assert season["unique"] + season["captaincy"] + season["hits"] == season["gap"]
    assert gap["lead"] is None


def test_gap_counts_shared_captain_with_different_multiplier_as_captaincy():
    mine = payload([(1, 1, GK), (3, 3, FWD)], chip="3xc", live=LIVE)
    theirs = payload([(1, 1, GK), (3, 2, FWD)], live=LIVE)
    season = insights.gap_to_leader({1: {"me": mine, "lead": theirs}}, {1: LIVE}, "lead")["me"][
        "season"
    ]
    assert (season["unique"], season["captaincy"], season["gap"]) == (0, 10, 10)


def test_gap_raises_when_terms_do_not_reconcile():
    mine = payload([(1, 1, GK), (2, 2, MID)], live=LIVE)
    mine["entry_history"]["points"] += 1  # FPL disagrees with the picks
    theirs = payload([(1, 1, GK), (3, 2, MID)], live=LIVE)
    with pytest.raises(ValueError, match="terms sum"):
        insights.gap_to_leader({1: {"me": mine, "lead": theirs}}, {1: LIVE}, "lead")


CHIPS = [
    {"name": "wildcard", "start_event": 2, "stop_event": 19},
    {"name": "3xc", "start_event": 1, "stop_event": 19},
    {"name": "wildcard", "start_event": 20, "stop_event": 38},
    {"name": "3xc", "start_event": 20, "stop_event": 38},
]


def test_chips_left_marks_used_available_and_expired():
    picks = {4: {"a": payload([(1, 1, GK)], chip="wildcard")}, 5: {"a": payload([(1, 1, GK)])}}
    slots = insights.chips_left(picks, CHIPS, next_gw=6)["a"]
    status = {(s["chip"], s["start"]): (s["status"], s["used_gw"]) for s in slots}
    assert status[("wildcard", 2)] == ("used", 4)
    assert status[("3xc", 1)] == ("available", None)
    assert status[("wildcard", 20)] == ("available", None)

    after_deadline = insights.chips_left(picks, CHIPS, next_gw=20)["a"]
    assert {(s["chip"], s["start"]): s["status"] for s in after_deadline}[("3xc", 1)] == "expired"


def test_chips_left_does_not_spend_second_half_chip_in_first_half():
    picks = {25: {"a": payload([(1, 1, GK)], chip="3xc")}}
    slots = insights.chips_left(picks, CHIPS, next_gw=26)["a"]
    status = {(s["chip"], s["start"]): s["status"] for s in slots}
    assert status[("3xc", 20)] == "used"
    assert status[("3xc", 1)] == "expired"


def test_fixture_outlook_averages_and_flags_blank_and_double():
    fixtures = [
        {"event": 7, "team_h": 10, "team_a": 20, "team_h_difficulty": 2, "team_a_difficulty": 4},
        {"event": 7, "team_h": 30, "team_a": 10, "team_h_difficulty": 3, "team_a_difficulty": 5},
    ]
    squad = {"a": payload([(1, 1, GK), (2, 1, MID), (3, 1, FWD)])}
    player_team = {1: 10, 2: 20, 3: 40}  # team 10 plays twice, team 40 not at all
    week = insights.fixture_outlook(squad, fixtures, player_team, [7, 8])["a"]
    assert week[0] == {"gw": 7, "difficulty": round((2 + 5 + 4) / 3, 2), "blanks": 1, "doubles": 1}
    assert week[1] == {"gw": 8, "difficulty": None, "blanks": 3, "doubles": 0}


def test_bench_luck_skips_bench_boost_and_counts_autosubs():
    normal = payload([(1, 1, GK), (2, 1, MID)], bench=[(5, 0, FWD)], autosubs=[(7, 4)], live=LIVE)
    boosted = payload([(1, 1, GK), (2, 1, MID), (5, 1, FWD)], chip="bboost", live=LIVE)
    result = insights.bench_luck({1: {"a": normal}, 2: {"a": boosted}}, {1: LIVE, 2: LIVE})["a"]
    assert result == {"bench_unused": 8, "autosub_points": 12, "autosubs": 1}


def test_head_to_head_counts_draws_and_uses_points_after_hits():
    week1 = {"a": payload([(3, 1, FWD)], live=LIVE), "b": payload([(3, 1, FWD)], live=LIVE)}
    week2 = {"a": payload([(3, 1, FWD)], cost=4, live=LIVE), "b": payload([(5, 1, FWD)], live=LIVE)}
    grid = insights.head_to_head({1: week1, 2: week2})
    assert grid["a"]["b"] == [0, 1, 1]  # draw, then 10 - 4 = 6 loses to 8
    assert grid["b"]["a"] == [1, 1, 0]
