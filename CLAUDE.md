# CLAUDE.md

Data-driven football ideas. Monorepo: one self-contained top-level folder per idea (`<idea>_project/`),
shared utilities in `src/football/`.

## Scope

- Keep changes to what was asked. Scaffolding and examples beyond the request aren't wanted.
- Committed docs must not name people in the pool or other private details; `lms_project/config.json`
  and `lms_project/conversation.md` are git-ignored for that reason.

## Commands

```bash
make setup          # create .venv, install core+dev+notebooks, install pre-commit hooks
make check          # lint + typecheck + tests (same as CI)
make format         # ruff fix + format
```

Use `.venv/bin/...` for tools.

## Non-obvious details

- Root `conftest.py` adds every `*/src` to `sys.path` for tests, so each project's package name must
  be unique (`lms` for `lms_project`).
- Run project code from its folder: `PYTHONPATH=src python -m <package>.<module>`
  (e.g. `cd lms_project && PYTHONPATH=src python -m lms.optimiser --demo`).
- `data/` and `outputs/` inside projects are git-ignored; `reports/figures/` is committed. Default
  CLI paths point there (`data/fd_cache/`, `outputs/...`).
- mypy checks only the shared package (`src/`).
- Notebook outputs are stripped by nbstripout on commit.

## Conventions

Follow `docs/conventions.md`. Anything used by two or more projects belongs in `src/football/`.
