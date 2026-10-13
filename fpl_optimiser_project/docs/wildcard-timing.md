# When to play the wildcard

Design note, 1 September 2026. Proposes a formula for `wildcard_gain` — the one input to
`choose_chip` that the live advisor currently does not compute.

For the running engineering log see `project-notes.md`; for how to run the tools see the project `README.md`.
This is a design note: it states a rule, derives it, works one example, and says what would have
to be true for it to be worth shipping. Nothing here is implemented yet.

---

## TL;DR

- **The live advisor can never suggest a wildcard.** `cli_advise.py` passes `wildcard_gain=0.0`
  into `choose_chip`, and the branch tests against a floor of 15. It is false in every gameweek
  of every season.
- **The rule is not "does the rebuild beat the current squad".** It is "does the rebuild beat
  what free transfers reach in the next few weeks" — a much smaller number.
- The gain is a **transition drag**: the points lost while a one-a-week transfer queue catches
  up. Closed form below.
- The free-transfer path is **greedy and therefore concave** — the first transfer alone recovered
  45% of a full rebuild in the worked example. Any formula that multiplies "transfers needed" by
  an average value per transfer overrates the chip. The linear approximation overstated the drag
  by 55% here.
- ~~**This is blocked on the horizon defect**~~ — **fixed 1 September 2026** (§1.2). The
  inputs can now be measured over a real multi-week horizon. The worked example in §4 predates
  the fix and its numbers are one-week measurements; re-derive before implementing.

---

## 1. Two defects this sits on top of

Both are the same species as the two in `gw1-2026-27-findings.md`: silent, flattering, invisible
unless you go looking.

### 1.1 `wildcard_gain` is hardcoded to zero

`cli_advise.py`, in the `choose_chip` call:

```python
wildcard_gain = (0.0,)
```

The wildcard branch in `chips.py` tests

```python
wildcard_gain >= max(WILDCARD_THRESHOLD * urgency, WILDCARD_FLOOR)
```

which for a constant zero is `0.0 >= 15.0`. Always false. The bench boost, triple captain and
free hit all receive real inputs; the wildcard receives a placeholder. The chip logic is
exercised by the backtest and dead in the live path, which is why it has never been noticed.

### 1.2 The live horizon was always one gameweek — fixed

**Fixed 1 September 2026.** Recorded here because the formula below was designed around the
defect. What follows describes the code as it was.

`load_live_history` built records for completed gameweeks (`round < gameweek`) plus **exactly
one** upcoming gameweek. So `history.gameweeks` never extends past the next deadline, and

```python
weeks = [gw for gw in history.gameweeks if gw >= gameweek][: args.horizon]
```

can only ever return a single element. **`--horizon` is inert live.** The five-gameweek planner
— the biggest measured lever in the system at +105 points a season — runs as a one-week planner
against the live API.

This mattered here more than anywhere else, because a wildcard is *defined* as a multi-week
gain, and a one-week horizon cannot see one.

**The fix.** Two halves. `load_live_history` takes a `horizon` and builds fixture-only records
for that many upcoming gameweeks instead of one. And `view_at` takes a `known_through` cutoff,
defaulting to `gameweek - 1` so the backtest is untouched, because the obvious half alone would
have made things worse: `view_at` treats every lower-numbered gameweek as completed, so five
unplayed placeholders would have read as five matches in which every player logged zero minutes,
and the minutes model — which decays fast by design — would have forecast a squad of players who
never play. That is the GW1 cold start arriving from a different direction. Live, all five views
now share one cutoff (the last played gameweek) and differ only in the fixture, which is exactly
what the horizon is for.

---

## 2. The rule

> Play the wildcard when the rebuilt squad beats the best squad free transfers can reach in the
> next few weeks by enough to cover both the chip's remaining option value and the cost of the
> churn — and when enough of the season remains to bank the gain.

Three corrections to the intuitive version ("play it when I need more transfers than I have"):

**The counterfactual is the free-transfer path, not paying hits.** Nobody rebuilds by taking four
hits at once. The alternative to a wildcard is spending the next few weeks of free transfers, so
that is what it must beat.

**The cost of not playing it is delay, not hits.** You hold a squad you already know is wrong
while the queue drains. At one transfer a week, a four-player rebuild is four weeks of drag.

**The free-transfer path is greedy.** You make the *best* move first, so the value recovers fast
and the wildcard's remaining edge decays. This is the term the intuitive version misses entirely.

---

## 3. The formula

Play the wildcard iff

```
    D  +  ρ · A(R)   >   max( θ · u(R),  φ )        and       R ≥ 4
```

where the gain term is the transition drag

```
              k-F-1
    D    =     Σ      δ^t · Δ · ( 1 - c(F + t) )
              t=0
```

| symbol | meaning | source |
|---|---|---|
| `k`  | changes in the unrestricted rebuild, \|W \ S\| | two solves |
| `Δ`  | per-week xP of the rebuilt squad over the one you hold | `V(W) − V(S)` |
| `c(n)` | fraction of `Δ` recovered by the best `n` transfers | the recovery curve, §4 |
| `F`  | free transfers banked now | one a week, cap five |
| `δ`  | weekly discount | planner's `DEFAULT_DISCOUNT` |
| `ρ`  | per-week gap the free-transfer path can *never* close | `V(W) − V(FT after k)` |
| `A(R)` | annuity `Σ δ^t` over the `R` weeks left in the half | — |
| `u(R)` | urgency, `min(1, R/5)` | `chips._urgency` |
| `θ`, `φ` | option value 25, churn floor 15 | `WILDCARD_THRESHOLD`, `WILDCARD_FLOOR` |

`ρ` is the reachability term: sequential transfers are path-constrained, because each
intermediate squad must be legal *and* affordable, so there exist squads a one-a-week path cannot
walk to on its own budget. A wildcard solves the whole thing in one shot. `ρ` captures whatever
of that survives after the queue has spent its `k` transfers.

### The closed form worth carrying in your head

Assume each transfer captures an equal share, `c(n) = n/k`:

```
    D  =  Δ · (k - F)(k - F + 1) / (2k)
```

This is the pessimistic-about-free-transfers case and it **overstates** the drag, because the
real `c` is concave. Use it as an upper bound, not an estimate.

### What the two thresholds mean

Keeping them separate is the point, and the existing constants already do:

- `θ · u(R)` — **option value**, decaying to zero as the half runs out. A chip you might use
  better later is worth holding; on the last possible week it is worth nothing.
- `φ` — **churn cost**, which never decays. Unlike the other three chips a wildcard is not free:
  it tears up a squad you spent transfers assembling. Without this floor an earlier version fired
  a wildcard for a 5.6-point gain and the rebuilt squad was worse.

---

## 4. Worked example — GW3, 2026-27

Entry 552800, measured 1 September 2026 against the live API. Best squad reachable with `n`
transfers, forecast for GW3:

| transfers | weekly xP | gain vs no-move | `c(n)` |
|---:|---:|---:|---:|
| 0 | 45.9 | — | 0% |
| 1 | 50.9 | 5.0 | 45% |
| 2 | 52.7 | 6.8 | 61% |
| 3 | 54.4 | 8.5 | 76% |
| 4 | 55.5 | 9.6 | 86% |
| 5 | 57.1 | 11.2 | **100%** |

Five transfers is the whole rebuild — the sixth is worth nothing. So `k = 5`, `Δ = 11.2`,
`F = 2`, `R = 17` (GW3 → GW19).

```
  week 0:  2 done, 61% captured  →  0.39 × 11.2 = 4.4
  week 1:  3 done, 76%           →  0.24 × 11.2 = 2.6
  week 2:  4 done, 86%           →  0.14 × 11.2 = 1.6
  week 3:  5 done, converged     →  0
                                     D ≈ 8.6
```

`ρ ≈ 0` — the free-transfer path converged on the identical 57.07, so the reachability advantage
is worth nothing in this instance. Threshold is `max(25 × 1, 15) = 25`.

**8.6 against 25. Do not play it.** The linear closed form gives 13.4 for the same inputs — a
55% overstatement, and a direct demonstration that the concavity correction earns its place.

### The competitor is a hit, not patience

Week 0's shortfall is 4.4, marginally above a `−4` hit. So the rational impatient play is to take
one hit and compress the queue — and every hit taken shrinks `D` further. **A wildcard must beat
the hit-accelerated path, not the patient one.** A version of this rule that ignores hits will
fire too often.

---

## 5. What this depends on, and what would falsify it

Every input above was a **one-week measurement**, because of §1.2. That defect is now fixed, so
these can be re-derived properly — but the numbers in §4 have not been, and until they are:

- `Δ` is one gameweek's forecast, and the formula assumes it persists across the transition. It
  will not: form moves and fixtures rotate. Expect the true `D` to be lower than a constant-`Δ`
  model says, because the squad you are chasing changes while you chase it.
- `c` is measured on one gameweek. The shape (concave, front-loaded) should be robust — it comes
  from greediness, not from the forecast — but the exact fractions are not.
- **`ρ` cannot be measured at all at horizon 1.** It is the term that needs several weeks to
  appear, and it is the only term that argues *for* the chip on structural rather than temporal
  grounds. The worked example's `ρ ≈ 0` is therefore weak evidence, not a finding.

The horizon fix has landed, so the blocker is gone. **Re-measure §4 over a real five-week
horizon before implementing**; the formula's structure does not change, its inputs do.

One thing the fix exposed that bears on this directly: three gameweeks into a season the model's
per-90 rates are still shrunk hard toward positional means, and over GW3-GW7 it rates Haaland
(19.95) *below* Saka (20.57) and both below a Brentford midfielder on a good run of fixtures.
A longer horizon amplifies whatever the forecast believes. Right now it mostly believes fixture
difficulty, so `Δ` measured today is closer to a fixture score than a squad-quality gap. Wait for
current-season minutes to accumulate before trusting any `Δ` this rule is built on.

### How to validate it

Replay 2025-26 in the season simulator with this rule against the current threshold rule, and —
per the standing bar — treat anything under ~60 points as indistinguishable from luck.

Expect it to be inside the noise. Chip timing as a whole was measured at **+12 points a season**
against a ±30-point band, so the honest prior is that no wildcard rule is separable from any
other on one season of evidence. That is an argument for making the rule *auditable and hard to
fire spuriously*, not for tuning it. It is also an argument against spending much more time on
this than the note itself: the weekly transfer decision is worth an order of magnitude more.

The failure mode to test for explicitly is the one the floor exists to prevent — a wildcard fired
late in a half for a marginal gain, leaving a worse squad. A regression test should assert the
rule declines a rebuild whose `D` is small but whose `u(R)` has decayed.

---

## 6. Implementation sketch

Not built. When it is:

- `chips.py` gains a `wildcard_gain(...)` function taking `Δ`, `c`, `k`, `F`, `δ` and returning
  `D`. Pure, no solver, directly unit-testable — including the closed form as the `c(n) = n/k`
  case.
- The caller computes the recovery curve by solving `plan_transfers` at `free_transfers = 0..k`
  and reading back the value each time. That is `k+1` solves a week, which at current speeds is
  seconds, but it is the part to watch if the horizon fix makes each solve dearer.
- `cli_advise.py` replaces the `wildcard_gain=0.0` literal with that call.
- `ρ` comes free from the same sweep: it is `V(W) − V(FT after k)`, both of which the sweep
  already computes.

Keep it as explicit readable arithmetic rather than folding it into the integer program, for the
reason `chips.py` already gives: this decision is made eight times a season and benefits far more
from being auditable than from being provably optimal.

---

## 7. Open questions

- **Should `θ = 25` survive?** It was tuned as a threshold on a quantity that was never computed.
  Once `wildcard_gain` is real, the constant is measuring something different from what it was
  fitted against, and it should be re-derived rather than inherited.
- **Is `ρ` worth the complexity?** If it stays near zero once the horizon is fixed, drop it and
  keep `D` alone. Measure before deciding.
- **Does the same decomposition give the free hit a better rule?** A free hit is a one-week
  wildcard that reverts, so its gain is `Δ` for a single week with no churn cost and no drag term
  at all. The current `FREE_HIT_MAX_PLAYERS` heuristic — fire when nine or fewer have a fixture —
  is a proxy for exactly that, and could probably be replaced by the measured quantity.
