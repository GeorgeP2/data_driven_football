# Project conventions

The goal: any project can be understood in 2 minutes from its README and reproduced with one command.

## Structure

- One top-level folder per football idea: `<idea>_project/`. Create it with `make new-project`.
- Pipeline code lives in `src/<package>/` as importable modules, run with
  `PYTHONPATH=src python -m <package>.<module>`. Notebooks call into it rather than duplicating logic.
- Package names must be unique across projects (the root `conftest.py` puts every `*/src` on one path).
- Notebooks are numbered in reading order: `01_eda.ipynb`, `02_modelling.ipynb`, …
- Anything shared by two or more projects moves into `src/football/`.

## Data

- Never commit raw data. `data/` is git-ignored; the project README says exactly how to get it.
- Prefer public data with clear terms; fetch by script rather than redistributing.
- Run outputs go in `outputs/` (git-ignored).
- Personal state (pool members, picks) stays in git-ignored files; commit an `*.example.*` template.

## Reproducibility

- Fix random seeds so results can be reproduced exactly.
- Default paths are relative to the project folder; no hard-coded absolute paths.

## Quality

- `make check` must pass before pushing (CI runs the same thing).
- Each project has at least a smoke test; test data transforms and model logic properly.
- Notebook outputs are stripped on commit by `nbstripout`. Save figures you want to show into
  `reports/figures/`, and embed them in the README.

## README checklist

- [ ] Problem statement and why it matters
- [ ] Data source and terms
- [ ] Approach, including the baseline
- [ ] Results table and at least one figure
- [ ] Takeaways and limitations
- [ ] Reproduce section
- [ ] Skills demonstrated
