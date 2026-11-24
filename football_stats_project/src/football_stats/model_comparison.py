"""Compare scikit-learn and XGBoost classifiers with the two-coefficient relegation model.

Every model is scored the same way: leave one season out, fit on the rest, predict the season
held out. It is run three times: on points and gameweek alone, with league position, goal
difference and gap to 17th added, and then with last season's finish as well. Settings are
fixed, not tuned.

Needs the ``ml`` extra (``pip install -e ".[ml]"``). Run from the project folder:
    PYTHONPATH=src python -m football_stats.model_comparison
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from xgboost import XGBClassifier

from football_stats.relegation import (
    BASE_RATE,
    FEATURES,
    brier,
    log_loss,
    training_data,
    xgboost_classifier,
)
from football_stats.standings import build

# Feature sets, each mapping a feature to its direction (see ``relegation.FEATURES``).
FEATURE_SETS = {
    "points + gameweek": {"points": -1, "gameweek": 1},
    "+ position, goal difference, gap to 17th": {
        k: v for k, v in FEATURES.items() if k != "last_season_finish"
    },
    "+ last season's finish": FEATURES,
}
PHASES = [0, 10, 19, 29, 38]


def models(monotone: list[int]) -> dict[str, Callable[[], object]]:
    """Model factories by name; ``monotone`` is the direction of each feature, in order."""

    def xgb(**extra) -> XGBClassifier:
        return xgboost_classifier(n_jobs=1, **extra)

    def boosting(**extra) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=50,
            random_state=0,
            **extra,
        )

    return {
        "logistic": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=1e6, max_iter=2000)
        ),
        "logistic, quadratic terms": lambda: make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            LogisticRegression(C=1e6, max_iter=5000),
        ),
        "logistic, splines": lambda: make_pipeline(
            SplineTransformer(n_knots=6, degree=3),
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            LogisticRegression(C=1.0, max_iter=5000),
        ),
        "k nearest neighbours (200)": lambda: make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=200)
        ),
        "random forest": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=50, random_state=0, n_jobs=-1
        ),
        "sklearn gradient boosting": boosting,
        "sklearn gradient boosting, monotone": lambda: boosting(monotonic_cst=monotone),
        "xgboost": xgb,
        "xgboost, monotone": lambda: xgb(monotone_constraints=tuple(monotone)),
    }


def held_out_predictions(data: pd.DataFrame, features: dict[str, int]) -> pd.DataFrame:
    """One column of leave-one-season-out probabilities per model, indexed like ``data``.
    ``features`` maps each feature to use to its direction (see ``FEATURE_SETS``)."""
    candidates = models(list(features.values()))
    x = data[list(features)].to_numpy(float)
    y = data["relegated"].to_numpy(int)
    out = pd.DataFrame(np.nan, index=data.index, columns=list(candidates))
    for season in data["season"].unique():
        test = (data["season"] == season).to_numpy()
        for name, make in candidates.items():
            model = make().fit(x[~test], y[~test])
            out.loc[test, name] = model.predict_proba(x[test])[:, 1]
    return out


def score_table(data: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Log loss and Brier score per model, overall and log loss by phase of the season."""
    y = data["relegated"]
    phase = pd.cut(data["gameweek"], PHASES)
    labels = {p: f"GW{p.left + 1}-{p.right}" for p in phase.cat.categories}
    candidates = predictions.assign(**{"base rate (15%)": BASE_RATE})
    rows = {}
    for name in candidates.columns:
        p = candidates[name]
        row = {"log_loss": log_loss(y, p), "brier": brier(y, p)}
        for interval, label in labels.items():
            in_phase = phase == interval
            row[label] = log_loss(y[in_phase], p[in_phase])
        rows[name] = row
    return pd.DataFrame(rows).T.sort_values("log_loss")


def main() -> None:
    data = training_data(build())
    print(f"{data['season'].nunique()} seasons held out one at a time (lower is better)")
    for name, features in FEATURE_SETS.items():
        print(f"\nFeatures: {name}")
        print(score_table(data, held_out_predictions(data, features)).round(4).to_string())


if __name__ == "__main__":
    main()
