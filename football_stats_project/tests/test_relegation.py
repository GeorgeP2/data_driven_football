from functools import partial

import numpy as np
import pandas as pd
import pytest
from football_stats.relegation import (
    FEATURES,
    OUTCOMES,
    PROMOTED,
    TITLE,
    TOP4,
    RelegationModel,
    brier,
    cross_validate,
    fill_promoted,
    fit_xgboost,
    log_loss,
    logistic_fit_predict,
    predict_table,
    predict_xgboost,
    probability_table,
    rescale,
    rescale_by_table,
    scores,
    training_data,
    xgboost_fit_predict,
)


def synthetic(seed=0, seasons=6, teams=20):
    """Seasons where a team's chance of going down falls with its points and with time."""
    rng = np.random.default_rng(seed)
    truth = RelegationModel(intercept=1.0, per_point=-0.3, per_gameweek=0.25)
    rows = []
    for s in range(seasons):
        for gw in range(1, 39):
            points = rng.integers(0, 3 * gw + 1, size=teams)
            rows.append(
                pd.DataFrame(
                    {
                        "season": f"S{s}",
                        "gameweek": gw,
                        "points": points,
                        "relegated": rng.random(teams) < truth.predict(points, gw),
                    }
                )
            )
    return truth, pd.concat(rows, ignore_index=True)


def test_fit_recovers_the_coefficients():
    truth, data = synthetic(seasons=40)
    model = RelegationModel.fit(data["points"], data["gameweek"], data["relegated"])
    assert model.per_point == pytest.approx(truth.per_point, rel=0.1)
    assert model.per_gameweek == pytest.approx(truth.per_gameweek, rel=0.1)
    assert model.intercept == pytest.approx(truth.intercept, abs=0.3)


def test_predict_takes_scalars_and_arrays():
    model = RelegationModel(intercept=0.0, per_point=-0.5, per_gameweek=0.25)
    assert float(model.predict(10, 20)) == pytest.approx(0.5)
    p = model.predict([0, 10, 40], [20, 20, 20])
    assert p.shape == (3,)
    assert p[0] > p[1] > p[2]


def with_table_features(data, seed=1):
    rng = np.random.default_rng(seed)
    return data.assign(
        position=rng.integers(1, 21, len(data)),
        goal_difference=rng.integers(-30, 31, len(data)),
        gap_to_17th=data["points"] - data["gameweek"],
        last_season_finish=rng.integers(1, 22, len(data)),
        title_margin=data["points"] - 2 * data["gameweek"],
        top4_margin=data["points"] - 2 * data["gameweek"] + 5,
        champions=data["points"] > 2.2 * data["gameweek"],
        top4=data["points"] > 1.8 * data["gameweek"],
    )


def test_cross_validation_beats_the_base_rate():
    _, data = synthetic()
    data = with_table_features(data)
    held_out = {
        "logistic": cross_validate(data, logistic_fit_predict),
        "xgboost": cross_validate(data, xgboost_fit_predict),
    }
    table = scores(data, held_out)
    assert list(table.index) == ["base rate (15%)", "logistic", "xgboost"]
    for name, predicted in held_out.items():
        assert predicted.between(0, 1).all()
        assert table.loc[name, "log_loss"] < table.loc["base rate (15%)", "log_loss"]


def test_xgboost_respects_the_feature_directions():
    _, data = synthetic()
    model = fit_xgboost(with_table_features(data))
    row = {
        "points": 10,
        "gameweek": 12,
        "position": 15,
        "goal_difference": -5,
        "gap_to_17th": 0,
        "last_season_finish": 10,
    }
    for feature, direction in FEATURES.items():
        if direction == 0:
            continue
        low, high = pd.DataFrame([row, {**row, feature: row[feature] + 5}]).pipe(
            lambda rows: predict_xgboost(model, rows)
        )
        assert (high - low) * direction >= 0


def test_predict_table_scores_every_team():
    _, data = synthetic()
    table = with_table_features(data).head(20).assign(team="T", final_position=pd.NA)
    scored = predict_table(with_table_features(data), table)
    assert len(scored) == 20
    assert "final_position" not in scored.columns
    for outcome in OUTCOMES:
        assert scored[outcome.column].between(0, 1).all()
        assert scored[outcome.column].sum() == pytest.approx(outcome.places)


def test_rescale_hits_the_total_and_keeps_the_order():
    p = np.array([0.39, 0.14, 0.05, 0.02, 0.0])
    scaled = rescale(p, 1)
    assert scaled.sum() == pytest.approx(1)
    assert (np.diff(scaled) <= 0).all()
    assert ((scaled >= 0) & (scaled <= 1)).all()
    # too high is pulled down just the same
    assert rescale([0.9, 0.9, 0.9, 0.9, 0.9], 3).sum() == pytest.approx(3)


def test_rescale_by_table_works_within_each_season_and_gameweek():
    data = pd.DataFrame({"season": ["a"] * 3 + ["b"] * 3, "gameweek": 1})
    p = pd.Series([0.5, 0.2, 0.1, 0.9, 0.8, 0.7])
    scaled = rescale_by_table(data, p, 1)
    assert scaled.groupby(data["season"]).sum().tolist() == pytest.approx([1, 1])


def test_promoted_teams_are_placed_below_everyone():
    table = pd.DataFrame({"last_season_finish": pd.array([3, pd.NA], dtype="Int64")})
    assert list(fill_promoted(table)["last_season_finish"]) == [3, PROMOTED]


def test_scoring_rules():
    assert log_loss([1, 0], [0.5, 0.5]) == pytest.approx(np.log(2))
    assert brier([1, 0], [0.9, 0.2]) == pytest.approx((0.01 + 0.04) / 2)


def test_training_data_keeps_completed_20_team_seasons():
    def season(name, teams, finished=True):
        return pd.DataFrame(
            {
                "season": name,
                "team": [f"T{i}" for i in range(teams)],
                "gameweek": 1,
                "points": 0,
                "position": range(1, teams + 1),
                "final_position": range(1, teams + 1) if finished else pd.NA,
                "last_season_finish": pd.NA,
            }
        )

    frames = [season("a", 20), season("b", 22), season("c", 20, finished=False)]
    data = training_data(pd.concat(frames, ignore_index=True))
    assert set(data["season"]) == {"a"}
    assert list(data.loc[data["relegated"], "final_position"]) == [18, 19, 20]
    assert list(data.loc[data["top4"], "final_position"]) == [1, 2, 3, 4]
    assert list(data.loc[data["champions"], "final_position"]) == [1]


def test_other_outcomes_validate_against_their_own_base_rate():
    _, data = synthetic()
    data = with_table_features(data)
    for outcome in (TITLE, TOP4):
        held_out = cross_validate(data, partial(xgboost_fit_predict, outcome=outcome))
        table = scores(data, {"xgboost": held_out}, outcome)
        assert table.index[0] == f"base rate ({outcome.base_rate:.0%})"
        assert table.loc["xgboost", "log_loss"] < table.iloc[0]["log_loss"]


def test_probability_table_shape():
    model = RelegationModel(intercept=0.0, per_point=-0.5, per_gameweek=0.25)
    table = probability_table(model, [10, 20], [0, 5, 10])
    assert table.shape == (3, 2)
    assert table.loc[10, 20] == 50
