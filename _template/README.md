# {{title}}

> **Category:** {{category}} · **Status:** 🚧 In progress

## Problem

_What question is being answered, and why does it matter? Who would use the result?_

## Data

| Source | What | Terms | Use |
|--------|------|-------|-----|
| _link_ | | | |

Raw data lives in `data/` and is never committed. Describe how to obtain it here so the project is
reproducible.

## Approach

1. **EDA**: `notebooks/01_eda.ipynb`
2. **Pipeline**: `src/{{package}}/run.py`
3. **Evaluation**: results written to `outputs/`, figures to `reports/figures/`

## Results

| Method | Metric | Score |
|--------|--------|-------|
| Baseline | | |

_Key figure(s):_

<!-- ![](reports/figures/example.png) -->

## Key takeaways

- …

## Reproduce

```bash
# from the repo root, with the venv active (make setup)
cd {{slug}}
PYTHONPATH=src python -m {{package}}.run
pytest
```

## Skills demonstrated

- …
