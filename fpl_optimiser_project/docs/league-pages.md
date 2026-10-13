# League pages

One web page per mini-league (The Boys FPL, THG Ing League), built with the same simulation as
`notebooks/01_league_simulation.ipynb` (`fpl_optimiser.leagues.analysis`, 10,000 runs, seed 42).

**Stats only.** These pages go to the leagues, so they never show anything from the advisor
(forecast points, transfer, captain or chip recommendations, wildcard numbers) or anything that tells
a rival how to catch up (required run rate, "needs X a week"). That applies to the data embedded
in the page source as well as what's visible. Commentary describes what happened and never advises.

The leagues and their members live in `leagues.json`, and the write-ups in `commentary/`. Both are
git-ignored because they name real people: copy `leagues.example.json` to `leagues.json` to start.

Weekly refresh, once the gameweek is finished:

```
# from fpl_optimiser_project/
PYTHONPATH=src python -m fpl_optimiser.leagues.export   # pull, simulate, write outputs/league_pages/*.json
PYTHONPATH=src python -m fpl_optimiser.leagues.build    # render outputs/league_pages/<league>_fpl.html
```

Before `build.py`, add `commentary/<league>/gwNN.md` for the new week (plain paragraphs, `**bold**`
only; `build.py` names any week without one). The Boys write-ups are cheeky, THG's neutral. Write
from the numbers in `outputs/league_pages/<league>.json` → `gameweeks`: the "transfers" net is points in minus points
out for that week across all 15 players, so describe it as that, not as "beat his own squad".

`gameweeks.py` fetches picks, transfers and live points. It caches only *finished* gameweeks in
`data/league_cache/`, and it checks every manager's picks against FPL's own score. Note that FPL's gameweek
`points` is before hits.

The "Transfers so far" section comes from `season_transfers` in `gameweeks.py` and needs no extra
step. It scores each signing from the week it arrived until it was sold again (or the latest
gameweek), against what the player sold scored over the same weeks, then takes off hits. It uses
raw player points, so captaincy and bench weeks are not separated. Free-hit signings count for
their one week. Wildcard and free-hit squads have their own column, and best/worst move use
normal transfers only, because inside a chip week the code matches who replaced whom by position.

The five insight sections (captaincy and chips, gap to the leader, next five gameweeks, bench and
auto-subs, head to head) come from `insights.py`, which works on the picks and live points already
fetched, plus FPL's fixtures list. The gap to the leader splits each week's points difference into
players only one manager started, the captaincy difference and hits, and the export fails if those
three don't add up to the real gap. Chips come twice a season (GW1/2-19 and GW20-38), and a
first-half chip not played by GW19 shows as expired. Tests: `make check` from the repo root.

`export --cached` skips the pull and reuses the `fpl_entry_*_history.csv` files in `data/`.

Then publish. The pages are hosted on GitHub Pages from a separate repo, GeorgeP2/fpl-league-pages,
so the public site never links back to this code. With it cloned next to this repo:

```
cp outputs/league_pages/*_fpl.html ../../fpl-league-pages/
cd ../../fpl-league-pages && git add -A && git commit -m "League pages after GWNN" && git push
```

`commentary/` is not in git, but every write-up is embedded in the published pages.

They are also Claude artifacts. To keep those current, ask Claude to republish the two HTML files to
the existing artifacts so the links stay the same:

- The Boys FPL — https://claude.ai/artifact/RxGhGZqzjEyhp4dAsh5xmN
- THG Ing League — https://claude.ai/artifact/T5C8hPe8qUqpbkwfNBZgDD

Entry ids are reissued every season. Update them in `leagues.json`, which the notebook reads too.
The export fails, listing the current ids, if an entry is no longer in its league.
