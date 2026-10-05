# data_driven_football

[![CI](https://github.com/GeorgeP2/data_driven_football/actions/workflows/ci.yml/badge.svg)](https://github.com/GeorgeP2/data_driven_football/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A home for data analytics and ML related thoughts about the world's best sport.
Each idea lives in its own top-level folder, is reproducible and tested, and follows the same structure.

## Projects

| Project | Area | Status | Highlights |
|---------|------|--------|------------|
| [Last Man Standing optimiser](lms_project) | Optimisation & simulation | 🚧 | Monte Carlo pot-equity picks return +0.73 to +0.98 per stake against an office-pool field, against −0.11 to −0.18 for always backing the favourite |

## Repository layout

```
.
├── <idea>_project/           # one self-contained folder per football idea
│   ├── README.md             # problem, data, approach, results
│   ├── src/<package>/        # pipeline code, run with `python -m <package>.<module>`
│   ├── tests/
│   ├── docs/                 # research notes
│   ├── reports/figures/      # committed figures used in the README
│   ├── data/                 # git-ignored
│   └── outputs/              # git-ignored (run results)
├── src/football/             # shared utilities (football-data.co.uk loading, team names)
├── tests/                    # tests for the shared package
└── docs/                     # conventions
```

## Getting started

```bash
git clone git@github.com:GeorgeP2/data_driven_football.git
cd data_driven_football
make setup            # venv + core, dev and notebook deps + pre-commit hooks
source .venv/bin/activate
make check            # lint, type-check and test everything
```

### Adding a project

Create a new top-level folder named `<idea>_project/` with the same layout as
[`lms_project/`](lms_project), give its package under `src/` a unique name, and add a row to the
table above. See [docs/conventions.md](docs/conventions.md) for the house rules.

## How this was built

I build these projects with an AI coding assistant ([Claude Code](https://claude.com/claude-code));
some commits list it as a co-author. I choose the problems, set the scope and make the design
calls. The assistant speeds up implementation, and I review, test and can explain every change.
