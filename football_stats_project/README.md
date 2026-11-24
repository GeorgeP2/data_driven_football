# Football Stats

> **Category:** Analytics & Visualisation · **Status:** 🚧 In progress

## Problem

_What question is being answered, and why does it matter? Who would use the result?_

## Data

| Source | What | Licence | Use |
|--------|------|-------|-----|
| [football-data.co.uk](https://www.football-data.co.uk/englandm.php) | Premier League results, 1993-94 onwards | Free for personal use | Match results that the standings are built from |

Raw data lives in `data/` and is never committed. `football_stats.standings` downloads each season
into `data/fd_cache/` the first time it runs.

## Approach

1. **EDA**: `notebooks/01_eda.ipynb`
2. **Standings**: `src/football_stats/standings.py` builds one row per season, team and gameweek
   with points, goal difference, league position, gap to 17th, margins to the title and the top
   four, last season's finish and final league position, written to `outputs/standings.csv`
3. **Finish probabilities**: `src/football_stats/relegation.py` fits one XGBoost model per outcome
   (title, top four, relegation) on points, gameweek, league position, goal difference, last
   season's finish and the distance to the line that matters, and scores every team in the
   current table. Each table's probabilities are rescaled to add up to the right number of teams. Each is scored
   by leaving one season out at a time. A two-input logistic regression answers what a points
   total at a given gameweek has meant for relegation
4. **Season simulation**: `src/football_stats/simulation.py` rates every team with Elo, turns rating
   gaps into expected goals, and plays the remaining fixtures out 10,000 times to get each team's
   chance of every finishing position. It beats the XGBoost models on all three outcomes
5. **Model comparison**: `src/football_stats/model_comparison.py` runs the scikit-learn and XGBoost
   candidates on both sets of inputs

## Results

Relegation probability on held-out seasons (31 seasons, 1995-96 to 2025-26; lower is better):

| Method | Log loss | Brier score |
|--------|----------|-------------|
| Baseline: every team at the 15% base rate | 0.423 | 0.128 |
| Logistic regression on points and gameweek | 0.228 | 0.072 |
| Best scikit-learn or XGBoost alternative on the same two inputs (`model_comparison.py`) | 0.228 | 0.072 |
| XGBoost with league position, goal difference and gap to 17th added | 0.216 | 0.069 |
| **Main model:** the same plus last season's finish, rescaled so each table sums to three | 0.199 | 0.063 |

Title and top four, same validation:

| Outcome | Baseline log loss (base rate) | XGBoost log loss |
|---------|-------------------------------|------------------|
| Title (finish 1st) | 0.199 | 0.069 |
| Top four | 0.500 | 0.147 |

Simulation against XGBoost on the same 590 as-it-stands tables from 2010-11 to 2025-26 (log
loss; simulation settings fitted on the seasons before 2010-11):

| Outcome | XGBoost | Simulation |
|---------|---------|------------|
| Title | 0.072 | 0.068 |
| Top four | 0.176 | 0.145 |
| Relegation | 0.196 | 0.185 |

How the simulation's error falls through a season, from the same backtest (`--backtest` prints
every gameweek and writes `outputs/backtest_by_gameweek.csv`):

| After gameweek | Points error per team | Position error | Title favourite wins | Top-four picks right (of 4) | Relegation picks right (of 3) |
|----------------|-----------------------|----------------|----------------------|-----------------------------|-------------------------------|
| 1 | 8.4 | 2.9 | 36% | 2.6 | 1.4 |
| 10 | 6.6 | 2.3 | 56% | 3.1 | 1.9 |
| 20 | 4.8 | 1.7 | 84% | 3.4 | 1.9 |
| 30 | 3.3 | 1.3 | 83% | 3.4 | 2.3 |
| 37 | 1.0 | 0.5 | 100% | 3.9 | 2.9 |

_Key figure(s):_

<!-- ![](reports/figures/example.png) -->

## Key takeaways

- …

## Reproduce

```bash
# from the repo root, with the venv active (make setup)
cd football_stats_project
PYTHONPATH=src python -m football_stats.standings
PYTHONPATH=src python -m football_stats.relegation      # needs the ml extra; on macOS also `brew install libomp`
PYTHONPATH=src python -m football_stats.relegation --provisional   # add results not yet on football-data.co.uk
PYTHONPATH=src python -m football_stats.relegation --points 8 --gameweek 11
PYTHONPATH=src python -m football_stats.model_comparison
PYTHONPATH=src python -m football_stats.simulation --provisional   # simulate the rest of the season
PYTHONPATH=src python -m football_stats.simulation --backtest      # score it against XGBoost
pytest
```

## Skills demonstrated

- …
