# Further research

*Companion to [research.md](research.md). 5 October 2026.*

Ordered roughly by expected payoff. Items 1–3 decide whether the headline result (Monte Carlo
pot-equity beats always backing the favourite) holds for our actual pool. Everything later is about
pushing past it.

## Benchmark: how every new strategy is assessed

**Metric: chance of winning the pot as a multiple of fair share,** where fair share = 1 ÷ players
alive. A raw percentage isn't comparable across situations (12% with 11 players alive is good; 12%
with 3 alive is poor). The multiple is.

**Current baseline:**

| Measure | Value | Source |
|---|---|---|
| Tournament, office-pool format | **≈ 1.6×** fair share (11.6% sole-win rate against 7.1% fair share, 14 seats) | `mc_h4_k5`, [research.md](research.md) §4.2 |
| Live, our pool before GW6 | **≈ 1.3×** fair share (about 12% against 9% fair share, 11 alive) | simulation below |

Live estimate, simulated from the pool's state before GW6: 11 alive, 40,000 runs of the rest of the
season, GW6 odds and the model for later weeks.

| Rivals pick like... | Pick | Win outright | Everyone out → reset |
|---|---|---|---|
| moderate favourite-chasers (k=3) | Arsenal | 14% | 33% |
| moderate favourite-chasers (k=3) | Chelsea | 12% | 33% |
| hard favourite-chasers (k=5) | Arsenal | 6% | 38% |
| hard favourite-chasers (k=5) | Chelsea | 7% | 38% |

A reset (about 1 in 3) adds roughly 3 points: about 1/18 in the restarted pool, times the strategy's
edge. Total: about 9–17% to win this pot, most likely around 12%. This understates slightly, because
in the simulation you follow a fixed plan rather than re-optimising each week.

**To be adopted, a new strategy must:**
1. Beat `mc_h4_k5` in a **paired** tournament (same lineups, seeds and, if used, the same shuffled
   orderings and result draws). Report the difference in fair-share multiple with its standard
   error, bootstrapped over **real seasons**.
2. Hold that edge on **held-out real seasons** (real results, not re-drawn ones).
3. Clear about **1.5× fair share** in the office-pool format, and not fall below 1× when the
   opponents differ from what it assumes (k = 1.5 to 20, and the fitted bots from item 1).

## Roadmap at a glance

| # | Avenue | Effort | Why |
|---|---|---|---|
| 1 | Fit bots to the pool's real picks | low (needs data from the sheet) | removes the weakest assumption in every result so far |
| 2 | Tune the MC for our pool | low | k > 5 untested; our pool has 18 seats, the tests had 14 |
| 3 | Headroom benchmark | low–medium | tells us whether anything fancier can win much more |
| 4 | Data augmentation (shuffle rounds, re-draw results) | low | cuts noise ~10×; required before tuning anything with more parameters |
| 5 | Hybrid strategy that changes with the game state | medium | the best play plausibly differs early, mid-game and in the endgame |
| 6 | Exact endgame solver | medium | the pot is decided with 2–4 players left |
| 7 | Learned value function | medium | fixes the MC's crude valuation at the end of its horizon |
| 8 | Deep RL with self-play | high | only if 3–7 show real headroom |
| 9 | Better future-week probabilities | medium | the planner relies on a model that missed Coventry badly |
| 10 | Robustness and live tracking | low, ongoing | an honest test of the whole system |

## 1. Opponents fitted to the pool's real picks

Every opponent so far is a bot that picks with probability ∝ P(win)^k, the same family the MC
assumes. That makes the strongest result partly circular.

- Log every player's pick each week from the shared sheet (the pool's first round already shows
  heavy herding: 8 of the 10 survivors picked Man City).
- Fit a choice model, e.g. a multinomial logit on: P(win), home/away, big-club indicator, popularity
  of the team last week, whether the team just lost, and the player's own past behaviour.
- Rerun the tournament with fitted bots in the colleague seats. **If `mc_h4_k5` still leads, the
  result is real for our pool.**

## 2. Tune the MC strategy for our pool

- Sweep assumed rival sharpness k = 3, 5, 8, 12, 20 and horizon H = 1, 2, 3, 4, 6.
- Use **18 seats** to match our pool. Larger pools probably reward being different more.
- Report paired differences against `mc_h4_k5` on the same lineups and seeds, not unpaired ROIs.

## 3. Headroom benchmark

Before building anything complex, estimate the ceiling. Run an "oracle" MC that knows the
colleagues' exact policy (the bots' true k) and searches deeper (H = 6, more candidates, more
simulations). If it beats `mc_h4_k5` by only a little, no method can win much more. If the gap is
large, items 5–8 are worth the effort.

## 4. Data augmentation

We have **24 seasons = 912 real rounds (~9,100 matches)**. That's too few to tune a hybrid or train a
learned model without overfitting. Two augmentations stretch it, and they combine.

### 4.1 Shuffle gameweek order within a season

Each gameweek is self-contained: every team plays once, and the round's odds and results belong to
it. So any order of the rounds is a legitimate season (1-2-3, 1-3-2, 2-1-3, 2-3-1, 3-1-2, 3-2-1, ...).
Results stay 100% real.

- **How many:** 38! ≈ 5.2 × 10⁴⁴ orders per season, so **24 × 38! ≈ 1.26 × 10⁴⁶ possible seasons**.
  (Exactly, the sum over seasons of (rounds in that season)!, since rounds are 37–39 after
  dropping catch-up fixtures. Same order of magnitude.)
- **They are not independent.** Every order is a rearrangement of the same 912 rounds and the same
  results, like the 52! orders of a deck that still holds only 52 cards. Shuffling removes luck of
  the **order** (which week the big upset lands in); it does not create new rounds. Error bars must
  still come from resampling **real seasons** (as `lms.tournament`'s season bootstrap does).
  Treating 10,000 shuffles as 10,000 samples would make them far too narrow.
- **Diminishing returns:** pools last about 3–5 weeks, so what matters is mainly which rounds come
  first. There are about 1.8 million distinct 4-round openings per season, and a few hundred
  shuffles per season probably capture nearly all the benefit.
- **Avoid look-ahead:** the planner's Dixon-Coles forecasts for future weeks must not be fitted on
  rounds that come later in the shuffled order. Simplest fix: forecast future weeks from their own
  odds plus noise sized to the real gap between model and odds. Clean odds would give the planner
  better information than it has in reality.
- **Unrealistic fixture patterns** (A v B next to B v A, long runs of home games) are probably
  harmless for LMS.

**Invalid shuffles:**
- moving single matches between rounds (teams would play twice or not at all)
- swapping results without regard to the odds (breaks the odds–result link)
- mixing rounds across seasons (team sets differ, so "used Arsenal" loses its meaning)

### 4.2 Re-draw results from the odds (K draws per ordering)

For each shuffled season, draw K sets of results from the odds with the margin removed. The agent
then never sees the same season twice, can't memorise real results, and meets combinations of upsets
that never happened.

**The key assumption, that the odds are calibrated, holds.** Shin-de-vigged odds against results,
9,120 matches, 2002/03–2025/26:

| Odds said (P win) | Matches | Odds said | Actually won | Gap |
|---|---|---|---|---|
| ≤ 30% | 7,089 | 18.3% | 18.4% | +0.1 |
| 30–40% | 3,500 | 35.0% | 33.6% | −1.4 |
| 40–50% | 2,839 | 44.6% | 43.9% | −0.7 |
| 50–55% | 1,106 | 52.4% | 54.7% | +2.3 |
| 55–60% | 998 | 57.3% | 59.1% | +1.8 |
| 60–65% | 748 | 62.2% | 63.9% | +1.7 |
| 65–70% | 585 | 67.3% | 69.1% | +1.7 |
| 70–75% | 548 | 72.3% | 73.4% | +1.0 |
| 75–80% | 452 | 77.4% | 75.2% | −2.1 |
| 80–85% | 273 | 82.2% | 85.7% | +3.6 |
| > 85% | 102 | 87.5% | 88.2% | +0.7 |

Every band is within about 1.5 standard errors. One faint pattern is worth tracking: favourites
priced at 50–75% won 1–2 points more often than implied. Simulating from the odds erases it, which is
one more reason to keep the final verdict on real results.

**Variant without the calibration assumption:** group matches by favourite probability (e.g. 70–75%)
and swap real results within each group. This keeps the empirical calibration exactly.

### 4.3 How to use it

| Purpose | Data |
|---|---|
| Training (RL, value function) | fresh ordering **and** fresh result draw every episode; reusing an ordering K times doesn't help |
| Comparing and tuning policies | K = 10–20 draws per ordering, **every policy on the same orderings and draws** (paired comparisons remove most of the noise) |
| Final verdict | **real results only**, on held-out seasons |

Proposed split: train and tune on shuffles of 18 seasons, test on the real results of the other 6.
Implementation: `--shuffle-rounds N` and `--redraw-results K` flags in `lms.tournament`.

## 5. Hybrid strategy that changes with the game state

The right amount of differentiation plausibly depends on the state:
- **many alive:** differ from the crowd (someone else will survive whatever you do)
- **mid-game:** saving teams matters (players' remaining options diverge)
- **2–4 alive:** solve exactly (item 6); matching a rival can be a hedge, because going out together
  is a reset, not a loss

The MC already reacts to state implicitly. An explicit hybrid makes its parameters functions of the
state: k(n_alive), H(week), and an exact solver below a threshold. **Keep it to 2–3 parameters**,
choose them on augmented training seasons and confirm on held-out real seasons. Add as a
`hybrid_...` entrant in `lms.tournament`.

## 6. Exact endgame solver

With 2–4 players left the game is small: up to 20 picks each and three results per match. Use
expectimax over picks and results with the opponent model, solved exactly for 1–3 weeks ahead,
with a learned or MC value at the leaves. This is where tree search is justified.

## 7. Learned value function

The MC's weakest point is the end of its horizon. If the game is still alive after H weeks, it values
the position as pot ÷ survivors, ignoring that you may hold stronger remaining teams than your rivals.

- Train a model (gradient boosting or a small network) on augmented simulated pools to predict pot
  share from: week, players alive, my remaining team strength vs rivals', and the pot.
- Plug it into the MC as the leaf value (AlphaZero-style, but plain supervised learning: no RL
  instability).
- Test in the tournament against `mc_h4_k5`.

## 8. Deep RL

**Data volume is not the problem** (item 4 gives effectively unlimited episodes). The risks are:

1. **Memorisation.** Give the agent odds-based features only: my P(win) per available team, my
   remaining strength vs rivals', players alive, how many rivals can still pick this week's
   favourite, the pot. **No team names or round IDs.** Validate on held-out seasons: if held-out
   performance falls below training, it's memorising.
2. **Opponent model.** It learns to beat whatever it trains against. Mitigations: train against
   fitted bots (item 1), a mixture of bot types, and **self-play** (populations of agents), which
   finds strategies that hold up when everyone is smart.
3. **Sparse, noisy reward.** It is paid only as the sole survivor (~10% of pools). Use shaped
   rewards (expected pot share from the MC or the value function) and variance reduction.
4. **Limited headroom.** If item 3 shows the MC is near the ceiling, RL can't gain much.

Small networks are adequate: the decision has at most 20 actions and a compact state.

## 9. Better future-week probabilities

The planner and MC use Dixon-Coles for weeks without odds. In gameweek 6 it rated Newcastle away at
Coventry 73% where the market said 45%.

- Fit team ratings to **past odds** rather than past scores (the market is a better signal).
- Use season-long markets (title and relegation odds) to set early-season ratings, especially for
  promoted teams.
- Measure how much future-week error costs: rerun the backtest with future weeks forecast from their
  real odds (an upper bound) vs the model.

## 10. Robustness and live tracking

- **Backtest robustness:**
  - closing vs earlier pre-match odds
  - sensitivity to how catch-up fixtures are dropped
  - paired `mc_h4_k5` vs greedy comparison with matched lineups, to firm up the ~4 standard-error
    result
- **Information and timing:** if picks become visible before the deadline, quantify the value of
  picking last. Model correlated picks (colleagues who talk pick alike, which strengthens herding).
- **Rule variants:** draws not eliminating, used lists resetting after 20 teams, buy-backs,
  double/blank gameweeks.
- **Live log:** record the optimiser's recommendation, the pick actually made, the pool's picks and
  the result every week. One season is a small sample, but it is the only fully out-of-sample test.
