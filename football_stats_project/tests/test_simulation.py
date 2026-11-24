import numpy as np
import pandas as pd
import pytest
from football_stats.simulation import (
    BASE,
    Parameters,
    _poisson_fit,
    error_by_gameweek,
    fit_goals,
    latest_ratings,
    rate,
    remaining_fixtures,
    simulate,
    snapshots,
    summarise,
)

PARAMS = Parameters(
    k=20, home=50, carry=0.5, promoted=1400, home_goals=(0.4, 0.8), away_goals=(0.1, 0.8)
)


def season(name, teams, start, seed=0):
    """A double round robin, one match a day, with random scores."""
    rng = np.random.default_rng(seed)
    pairs = [(h, a) for h in teams for a in teams if h != a]
    return pd.DataFrame(
        {
            "season": name,
            "date": pd.date_range(start, periods=len(pairs)),
            "home": [h for h, _ in pairs],
            "away": [a for _, a in pairs],
            "hg": rng.poisson(1.5, len(pairs)),
            "ag": rng.poisson(1.1, len(pairs)),
        }
    )


def test_rating_change_is_bigger_for_upsets_and_wide_margins():
    level = float(PARAMS.rating_change(0, 1))
    assert level > 0
    assert float(PARAMS.rating_change(0, -1)) < 0
    assert float(PARAMS.rating_change(-200, 1)) > level > float(PARAMS.rating_change(200, 1))
    assert float(PARAMS.rating_change(0, 3)) == pytest.approx(1.75 * level)
    # a draw at home costs the home side a little
    assert float(PARAMS.rating_change(0, 0)) < 0


def test_expected_goals_follow_the_rating_gap():
    home, away = PARAMS.expected_goals([0, 400])
    assert home[0] == pytest.approx(np.exp(0.4))
    assert home[1] > home[0]
    assert away[1] < away[0]


def test_rate_is_zero_sum_and_carries_ratings_between_seasons():
    first = season("2020-21", ["A", "B", "C"], "2020-08-01")
    second = season("2021-22", ["A", "B", "D"], "2021-08-01")
    rated = rate(pd.concat([second, first]), PARAMS)
    assert list(rated["season"].unique()) == ["2020-21", "2021-22"]
    change = rated["home_after"] - rated["home_before"]
    assert (change == -(rated["away_after"] - rated["away_before"])).all()
    assert rated.loc[0, ["home_before", "away_before"]].tolist() == [BASE, BASE]

    end = latest_ratings(rated[rated["season"] == "2020-21"])
    opener = rated[rated["season"] == "2021-22"]
    a_start = opener.loc[opener["home"] == "A", "home_before"].iloc[0]
    assert a_start == pytest.approx(BASE + 0.5 * (end["A"] - BASE))
    d_first = opener[(opener["home"] == "D") | (opener["away"] == "D")].iloc[0]
    assert PARAMS.promoted in (d_first["home_before"], d_first["away_before"])


def test_poisson_fit_recovers_the_rates():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20_000)
    (intercept, slope), _ = _poisson_fit(x, rng.poisson(np.exp(0.3 + 0.5 * x)).astype(float))
    assert intercept == pytest.approx(0.3, abs=0.03)
    assert slope == pytest.approx(0.5, abs=0.03)


def test_fit_goals_fills_in_the_goals_model():
    teams = ["A", "B", "C", "D"]
    seasons = [season(f"20{y}-{y + 1}", teams, f"20{y}-08-01", seed=y) for y in range(10, 14)]
    rated = rate(pd.concat(seasons), PARAMS)
    fitted, per_match = fit_goals(rated, PARAMS)
    assert fitted.k == PARAMS.k
    assert fitted.home_goals[0] == pytest.approx(np.log(1.5), abs=0.3)
    assert np.isfinite(per_match)


def test_remaining_fixtures_are_the_pairings_not_played():
    played = season("2020-21", ["A", "B", "C"], "2020-08-01").head(2)
    left = remaining_fixtures(played)
    assert len(left) == 4
    assert set(left) == {("B", "A"), ("B", "C"), ("C", "A"), ("C", "B")}
    assert left == remaining_fixtures(played)


def table(teams, points):
    return pd.DataFrame({"team": teams, "points": points, "goal_difference": 0, "scored": 0})


def test_simulate_with_nothing_left_returns_the_table():
    teams = ["A", "B", "C"]
    out = simulate(table(teams, [9, 6, 3]), pd.Series(BASE, index=teams), [], PARAMS, sims=50)
    assert out[[1, 2, 3]].to_numpy().tolist() == np.eye(3).tolist()
    assert list(out["expected_points"]) == [9, 6, 3]


def test_simulate_gives_a_full_distribution_and_favours_the_stronger_team():
    teams = [f"T{i:02d}" for i in range(20)]
    ratings = pd.Series(np.linspace(1800, 1300, 20), index=teams)
    fixtures = [(h, a) for h in teams for a in teams if h != a]
    out = simulate(table(teams, [0] * 20), ratings, fixtures, PARAMS, sims=300, seed=1)
    places = out[list(range(1, 21))].to_numpy()
    assert places.sum(axis=1) == pytest.approx(np.ones(20))
    assert places.sum(axis=0) == pytest.approx(np.ones(20))
    assert out["expected_points"].iloc[0] > out["expected_points"].iloc[-1]

    summary = summarise(out)
    assert summary["title_probability"].sum() == pytest.approx(1)
    assert summary["top4_probability"].sum() == pytest.approx(4)
    assert summary["relegation_probability"].sum() == pytest.approx(3)
    assert summary["title_probability"].iloc[0] > 0.3
    assert summary["relegation_probability"].iloc[-1] > 0.5
    assert summary["expected_position"].iloc[0] < summary["expected_position"].iloc[-1]
    # same seed, same answer
    again = simulate(table(teams, [0] * 20), ratings, fixtures, PARAMS, sims=300, seed=1)
    pd.testing.assert_frame_equal(out, again)


def test_snapshots_come_every_ten_matches_and_stop_before_the_end():
    matches = season("2020-21", [f"T{i}" for i in range(6)], "2020-08-01")  # 30 matches
    cuts = snapshots(matches)
    assert [len(c) for c in cuts] == [10, 20]
    assert all(len(set(c["home"]) | set(c["away"])) == 6 for c in cuts)


def test_snapshots_wait_until_every_team_has_played():
    # 12 teams: after 10 matches only T0 has been at home, and T11 has not played at all
    matches = season("2020-21", [f"T{i:02d}" for i in range(12)], "2020-08-01")
    assert len(snapshots(matches)[0]) == 20


def test_error_by_gameweek_scores_each_table_then_averages():
    def table(season, snapshot, gameweek, good):
        """Four teams; a good forecast has the order right, a bad one has it reversed."""
        final = [1, 2, 3, 4]
        order = final if good else final[::-1]
        return pd.DataFrame(
            {
                "season": season,
                "snapshot": snapshot,
                "gameweek": gameweek,
                "team": list("ABCD"),
                "final_position": final,
                "final_points": [80, 60, 40, 20],
                "expected_position": order,
                "expected_points": [100 - 20 * p + (0 if good else 5) for p in order],
                "champions": [p == 1 for p in final],
                "top4": True,
                "relegated": False,
                "simulation_title": [1 / p for p in order],
                "simulation_top4": 1.0,
                "simulation_relegation": 0.0,
            }
        )

    rows = pd.concat(
        [table("a", 0, 1, good=False), table("b", 0, 1, good=True), table("a", 1, 2, good=True)]
    )
    errors = error_by_gameweek(rows)
    assert list(errors.index) == [1, 2]
    assert list(errors["tables"]) == [2, 1]
    assert errors.loc[2, "points_error"] == 0
    assert errors.loc[2, "position_error"] == 0
    assert errors.loc[2, "favourite_wins"] == 1
    assert errors.loc[1, "favourite_wins"] == 0.5
    assert errors.loc[1, "position_error"] == pytest.approx((3 + 1 + 1 + 3) / 4 / 2)
    assert errors.loc[1, "top4_right"] == 4
    assert errors.loc[1, "relegation_right"] == 0
