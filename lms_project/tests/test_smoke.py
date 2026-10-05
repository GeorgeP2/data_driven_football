import numpy as np
import pandas as pd
from lms import optimiser as L


def test_package_imports():
    from lms import backtest, fetch_data, tournament  # noqa: F401


def test_shin_devig_removes_margin():
    p = L.shin_devig([1.8, 3.6, 4.5])
    assert np.isclose(p.sum(), 1.0)
    assert p[0] > p[2] > 0


def test_demo_recommends_an_unused_team():
    results, fixtures = L.demo_data()
    pool = L.Pool(n_players=14, me_used=["Team00"], n_alive=10)
    model = L.DixonColes().fit(results)
    out, _ = L.recommend(model, fixtures, pool, horizon=4, n_sims=200)
    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0
    assert "Team00" not in set(out["pick"])
