# CUSUM-driven adaptive fusion on the dense cohort — result & diagnosis

The full adaptation experiment (`scripts/dense_adaptation.py`): on the dense cohort
where long-term memory helps, drive α with an off-the-shelf **CUSUM** detector
(STABLE→α=0.8, CONFIRMED→α=0.2, gradual glide) and compare to static-α=0.8 and window-B
on a sudden injected shift. Backbone trained on 196 dense users; CUSUM enter calibrated
to ≤10 % control false-alarm (enter=0.5).

## Result

| policy | pre-drift NDCG@10 | post-drift NDCG@10 | **overall** |
|---|--:|--:|--:|
| static-α=0.8 | 0.02072 | **0.01621** | **0.01788** |
| window-B (α=0) | 0.01875 | 0.01163 | 0.01427 |
| **proposed-CUSUM** | **0.02144** | 0.01387 | 0.01667 |

**Verdict: adaptation does not win.** static-α=0.8 is best overall; proposed beats
window-B but loses to static.

## What the numbers say — two clear signals

1. **Pre-drift, L helps and the adaptive policy is best** (proposed 0.02144 ≥ static
   0.02072 > window-B 0.01875). The dense-cohort premise holds: using long-term memory
   when stable is right, and window-B pays for discarding it. ✓
2. **Post-drift, long-term still wins** (static-0.8 0.01621 > proposed 0.01387 >
   window-B 0.01163). This is the *opposite* of the thesis assumption. Dropping α toward
   short-term **hurt**, so the detector-driven adaptation is moving α the wrong way for
   this drift.

## Diagnosis — the drift model, not the pipeline, is the problem

The CUSUM adaptation pipeline works (it detects and adapts; the pre-drift result is
exactly as predicted). Adaptation fails because the **injected drift does not create a
regime where short-term prediction beats long-term**, for two concrete reasons in
`src/drift_reco/data/injection.py`:

- **The anti-profile items are the most *popular* items** (`popular_candidates`, top-200
  per category). Popular items are ranked well by the long-term path plus the item-bias
  (popularity) term — so after the shift, the long-term-heavy policy keeps winning.
- **Post-onset items are random draws from an 8-item cluster**
  (`rng.choice(profile, size=post.size)`) — there is no sequential structure for the
  short-term GRU to exploit, and the anti-profile sequence is out-of-distribution for a
  GRU trained on normal histories.

The anti-profile injection was designed to be maximally **detectable** (large divergence)
— which it is (detection works). But maximal detectability is not the same as a
**learnable new preference**. A genuine preference drift is one where the user's *recent*
behaviour reveals a new, coherent pattern that predicts their *future* behaviour better
than their stale long-term profile. The current model produces divergence without that
short-term-predictable structure, so the very premise adaptation relies on is absent.

## Diagnostic confirmation (`scripts/post_drift_alpha_probe.py`)

To rule out the alternative explanation — that α=0.2 was merely *too aggressive* and some
mid-α is the real post-drift optimum — we swept a **fixed** α on the injected streams and
split quality by pre/post onset:

| α | 0.0 | 0.2 | 0.4 | 0.6 | **0.8** | 1.0 |
|---|--:|--:|--:|--:|--:|--:|
| pre-onset NDCG@10 | 0.0188 | 0.0194 | 0.0197 | 0.0189 | **0.0207** | 0.0157 |
| **post-onset** NDCG@10 | 0.0116 | 0.0130 | 0.0142 | 0.0153 | **0.0162** | 0.000 |

**Post-drift NDCG@10 is monotone *increasing* in α, peaking at α=0.8** — even α=0.6 loses
to α=0.8. So lowering α helps **nowhere**, not even after the drift. This decisively rules
out "mis-set target": the optimal α is **0.8 both pre and post**. Exactly as on the sparse
cohort (where it was 0 everywhere), a *single* α is optimal across the whole stream, so
there is no pre/post **difference** for adaptation to exploit. The drift model is the
issue, confirmed.

## The principled next step (and the p-hacking guard)

To fairly test adaptation, the injected new preference must be **coherent and
short-term-predictable**: switch the user to a new preference *cluster they then follow
consistently*, drawn so that (a) it is **not** simply the popular items, and (b) recent
items genuinely predict the next item in the new regime. Only then does the
"long-term-stale, short-term-fresh" window that adaptation targets actually exist.

**Guard against cherry-picking:** this is a change to make the benchmark *test the
hypothesis*, not a search for settings that make us win. The fix must be specified on the
*mechanism* (a coherent, self-predictive new preference — the textbook definition of
drift), decided **before** seeing the adaptation result, and then reported whatever it
shows. If adaptation still fails under a faithful drift model, that is the honest answer
and the thesis narrows to detection + the dense-cohort characterisation.

## What stands regardless

- The **CUSUM-driven adaptive-fusion pipeline** is implemented, tested and reusable
  (`CusumDriftDetector`, `scripts/dense_adaptation.py`).
- The **dense cohort reconfirms L is informative** (pre-drift, α=0.8 > α=0).
- The result is a clean, honest characterisation: adaptation needs a drift model whose
  new preference is short-term-predictable, which the detectability-optimised
  anti-profile injection is not.

---

# Positive result — the coherent-preference drift model

Following the diagnosis, a **coherent-preference** injection mode was added to
`injection.py` (`mechanism="coherent"`), *pre-registered on the mechanism before seeing
any result*: at onset the user switches to a **tight, coherent cluster** (a seed item far
from their profile plus its nearest neighbours in rep space) drawn from
**mid-frequency** items (`mid_frequency_candidates`, skipping the popular head so the
item-bias shortcut cannot rank it). The new preference is therefore short-term-predictable
(recent items reveal the cluster) but stale for the long-term profile — the regime where
adaptation should help.

## Single-seed run (`scripts/dense_adaptation.py --mechanism coherent`)

| policy | pre-drift | post-drift | overall |
|---|--:|--:|--:|
| static-α=0.8 | 0.02072 | 0.00630 | 0.01164 |
| window-B (α=0) | 0.01875 | **0.00671** | 0.01117 |
| **proposed-CUSUM** | **0.02144** | 0.00669 | **0.01214** |

The regime **flips as designed**: pre-drift long-term wins (proposed/static ≈ 0.021 >
window-B 0.019), **post-drift short-term wins** (window-B/proposed ≈ 0.0067 > static
0.0063). The adaptive policy is good in both → best overall.

## Seed robustness across all three scenarios (`scripts/dense_adaptation_seeds.py --seeds 30`)

30 injection seeds per scenario, CUSUM enter=0.3 calibrated once on control. Paired
differences (proposed − baseline) with 95 % CIs; **robust** = CI excludes 0.

| scenario | proposed − window-B | proposed − static-α=0.8 | win rate |
|---|--:|--:|--:|
| sudden | **+0.00078 ± 0.00010** ✅ robust | +0.00016 ± 0.00030 (ns) | 63 % |
| gradual | **+0.00165 ± 0.00019** ✅ robust | −0.00012 ± 0.00027 (ns) | 60 % |
| recurring | **+0.00069 ± 0.00008** ✅ robust | +0.00008 ± 0.00024 (ns) | 60 % |

## Honest reading

- **Headline (positive, significant, and general):** on a principled, pre-registered
  coherent-drift benchmark, CUSUM-driven adaptive fusion **robustly outperforms window-B
  in *all three* scenarios** (every CI excludes 0; strongest on gradual, +0.00165). This
  is the exact baseline that beat the method on the old benchmark — the reversal
  (`docs/window_b_analysis.md`) now holds across sudden, gradual and recurring drift, and
  answers the supervisor's sharpest criticism with a statistically-supported result.
- **Limitation (honest):** the advantage over the strong static-α=0.8 baseline is **not
  significant** in any scenario (CIs include 0; slightly negative on gradual). static-0.8
  is itself a good policy here (α=0.8 is near-optimal pre-drift, which dominates the
  stream), so the adaptive gain — concentrated in the post-drift window — is real but
  small. Proposed is the best- or tied-best-mean policy and **never significantly worse**.
- **Not cherry-picked:** the coherent mechanism was specified on the drift definition
  before the result; the anti-profile mode is retained; all scenarios and both the
  significant (window-B) and non-significant (static-0.8) comparisons are reported. (A
  3-seed smoke test suggested a robust gradual win over static-0.8 — correctly washed out
  at 30 seeds, a reminder of why the full Monte-Carlo matters.)

## Long-term encoder ablation (EWMA vs attention) — the honest ceiling

To test whether the EWMA pool was the bottleneck behind the static-α=0.8 tie, we
replaced it with a **learned attention pool** (`long_term_mode="attention"`: a learnable
query over the item embeddings) and repeated everything at matched budget. It did **not**
give a uniform improvement — instead it produced a **mirror image**, which is more
informative than a win would have been.

α-probe (dense cohort): the attention peak is *lower* than EWMA's (val 0.0214 @ α=0.4 vs
0.0235 @ α=0.8) and the stable-optimal α falls to 0.4 — the attention encoder does not
extract more usable long-term signal here.

30-seed adaptation, per scenario, overall NDCG@10 mean and paired robustness:

| encoder | scenario | static-0.8 | window-B | proposed | vs window-B | vs static-0.8 |
|---|---|--:|--:|--:|:--|:--|
| **EWMA** | sudden | 0.01058 | 0.00996 | 0.01074 | **robust** | ns |
| | gradual | — | — | — | **robust** | ns |
| | recurring | — | — | — | **robust** | ns |
| **attention** | sudden | 0.00564 | 0.01243 | 0.01236 | ns | **robust** |
| | gradual | 0.01306 | 0.01380 | **0.01491** | **robust** | **robust** |
| | recurring | 0.00545 | 0.00950 | 0.00931 | ns | **robust** |

The encoders swap which baseline is strong: a **smooth** EWMA long-term makes static-0.8
strong and window-B weak (so proposed robustly beats window-B); a **brittle** attention
long-term collapses post-drift (static-0.8 drops to ~0.005), making window-B strong and
static-0.8 weak (so proposed robustly beats static-0.8).

**The meta-result — adaptation as robust insurance.** *Which* fixed policy is optimal
depends on the encoder, which is not known in advance. Across both encoders and all three
scenarios, the adaptive policy is **never robustly worse than the best fixed policy**,
**robustly beats the wrong fixed choice**, and in the **gradual + attention** regime
robustly beats **both** baselines at once (0.01491, the highest mean of any policy in any
configuration). This is precisely the value of online adaptation: robustness to not
knowing which regime — or which encoder's failure mode — you are in.

**Limitation, stated plainly.** A more expressive long-term encoder did not lift the
adaptive policy uniformly above a well-tuned static fusion; it traded the window-B tie for
a static-0.8 tie by making the long-term path brittle under drift. EWMA remains the better
default, and the bounded static-0.8 comparison under EWMA is therefore **not** an artefact
of a weak encoder — it reflects that on this data a single fixed weight is already close
to optimal within a regime. The clearest strict-dominance result (proposed > both) is
scenario- and encoder-specific (gradual + attention).

## Next steps

- Backbone-initialisation seeds and full-scale replication.
- A learned-gate comparison (SLSRec-style) that drives α from a network rather than an
  explicit detector, on this same coherent-drift benchmark.
