# Last Man Standing optimiser

> **Category:** Optimisation & simulation · **Status:** 🚧 In progress

## Problem

> In a Premier League Last Man Standing pool, which team should I pick each gameweek to maximise
> my expected share of the pot, rather than just to survive the longest?

Rules: each gameweek every player still in picks one team, which must **win** (a draw counts as a
loss), and no team can be used twice. The last player left wins the pot. If everyone still in goes
out in the same week, the pool resets: everyone rebuys, the pot rolls over and used-team lists clear.

## Data

| Source | What | Licence | Use |
|--------|------|---------|-----|
| [football-data.co.uk](https://football-data.co.uk) | EPL results and pre-match 1X2 odds, 2000/01 onward; upcoming fixtures with odds | See the site's terms: fetched by script, not redistributed | Model fitting, backtests, weekly picks |

Raw data lives in `data/` and is never committed. `lms.backtest` and `lms.tournament` download and
cache seasons in `data/fd_cache/`; `lms.fetch_data` writes `data/results.csv` and `data/fixtures.csv`.

## Approach

1. **Probabilities.** De-vig bookmaker odds with Shin's method for the current week, and fill in later
   gameweeks with a Dixon-Coles model fitted on recent results (`lms.optimiser`).
2. **Planning.** A rolling-horizon assignment (Hungarian algorithm) gives the best season plan for
   each candidate pick this week.
3. **Pot equity.** Monte Carlo over the next few gameweeks, including rivals' likely picks, scores each
   candidate by P(sole survivor) × pot + P(all out) × value of a reset + P(shared) × pot share.
4. **Evaluation.** Replay strategies on real seasons with no look-ahead, solo and against bots
   (`lms.backtest`), and play strategies against each other in the same pool (`lms.tournament`).

## Results

| Strategy | Net profit per stake (office-pool format) |
|----------|-------------------------------------------|
| Always back the favourite (`greedy`) | −0.11 to −0.18 |
| Monte Carlo pot equity (`mc_h4_k5`) | **+0.73 to +0.98** |

Full write-up in [docs/research.md](docs/research.md); next steps in
[docs/further-research.md](docs/further-research.md); plan in [docs/roadmap.md](docs/roadmap.md).

## Key takeaways

- Surviving longest is not the same as winning the pot: always backing the favourite shares its fate
  with everyone else who backs it.
- Planning ahead on its own helps little, because even the best simple strategy survives only about
  2.6 weeks on average.
- Being different pays only while you still pick a strong team.
- Use bookmaker odds for the current week; the model can be badly wrong early in a season.

## Reproduce

```bash
# from the repo root, with the venv active (make setup)
cd lms_project
PYTHONPATH=src python -m lms.optimiser --demo                  # synthetic data
PYTHONPATH=src python -m lms.fetch_data --seasons 2526 2627 --schedule schedule.csv
cp config.example.json config.json                             # then edit with your pool's state
PYTHONPATH=src python -m lms.optimiser --results data/results.csv --fixtures data/fixtures.csv \
    --config config.json --horizon 4
PYTHONPATH=src python -m lms.backtest --first 1415 --last 2425 --jobs 4
PYTHONPATH=src python -m lms.tournament
pytest
```

Run results are written to `outputs/` (git-ignored). Your `config.json` is git-ignored too.

## Skills demonstrated

- Probability calibration from betting markets (Shin de-vigging)
- Statistical modelling (Dixon-Coles Poisson model with time decay)
- Combinatorial optimisation (assignment problem, rolling horizon)
- Monte Carlo simulation with common random numbers and opponent modelling
- Leak-free backtesting on real historical data
