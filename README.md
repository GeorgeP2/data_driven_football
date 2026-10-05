# data_driven_football

a home for data analytics and ML related thoughts about the worlds best sport

Data-driven tools for football prediction games.

## lms_project: Last Man Standing (EPL) optimiser

Picks a team each gameweek for a Last Man Standing pool (pick a team to win, draw = out,
no reusing teams, pool resets with a rebuy if everyone is eliminated).

- `fetch_data.py`: downloads results and odds from [football-data.co.uk](https://football-data.co.uk) into `results.csv` / `fixtures.csv`
- `lms_optimiser.py`: de-vigs odds (Shin), fills later gameweeks with a Dixon-Coles model, plans a rolling horizon (Hungarian assignment) and scores picks by Monte Carlo pot equity
- `backtest.py`: replays strategies against real EPL seasons since 2000/01
- `teams.py`: team name helpers

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r lms_project/requirements.txt
```

### Usage

```bash
cd lms_project
python lms_optimiser.py --demo                       # synthetic data
python fetch_data.py --seasons 2526 2627 --schedule schedule.csv
cp config.example.json config.json                   # then edit with your pool's state
python lms_optimiser.py --results results.csv --fixtures fixtures.csv --config config.json
python backtest.py --first 1415 --last 2425 --jobs 4
```

Downloaded CSVs, `fd_cache/` and your `config.json` are git-ignored.
