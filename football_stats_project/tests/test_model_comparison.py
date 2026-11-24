import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("xgboost")

from football_stats.model_comparison import (
    FEATURE_SETS,
    held_out_predictions,
    models,
    score_table,
)


def synthetic(seasons=3, teams=20):
    """Seasons where the chance of relegation falls with points and rises with gameweek."""
    rng = np.random.default_rng(0)
    gameweek = np.tile(np.repeat(np.arange(1, 39), teams), seasons)
    points = rng.integers(0, 3 * gameweek + 1)
    chance = 1 / (1 + np.exp(-(1.0 - 0.3 * points + 0.25 * gameweek)))
    return pd.DataFrame(
        {
            "season": np.repeat([f"S{s}" for s in range(seasons)], 38 * teams),
            "gameweek": gameweek,
            "points": points,
            "position": rng.integers(1, teams + 1, len(points)),
            "goal_difference": rng.integers(-30, 31, len(points)),
            "gap_to_17th": points - gameweek,
            "last_season_finish": rng.integers(1, 22, len(points)),
            "relegated": rng.random(len(points)) < chance,
        }
    )


@pytest.mark.parametrize("features", FEATURE_SETS.values(), ids=list(FEATURE_SETS))
def test_every_model_gives_probabilities_and_beats_the_base_rate(features):
    data = synthetic()
    names = list(models(list(features.values())))
    predictions = held_out_predictions(data, features)
    assert list(predictions.columns) == names
    assert predictions.notna().all().all()
    assert ((predictions >= 0) & (predictions <= 1)).all().all()

    table = score_table(data, predictions)
    assert set(table.index) == {*names, "base rate (15%)"}
    assert table["log_loss"].idxmax() == "base rate (15%)"
    assert list(table.columns[:2]) == ["log_loss", "brier"]
