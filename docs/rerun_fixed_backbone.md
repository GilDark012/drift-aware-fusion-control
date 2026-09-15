# Phases 4–8 re-run on the fixed (k-core) backbone

After the k-core fix ([ADR-0006](decisions/ADR-0006-kcore-vocabulary-filtering.md)) the
whole pipeline was re-run. The subsample builder now applies **global k-core first**,
then selects the cohort on the dense test, so cohort users keep usable stream lengths.

## Backbone & data

| | value |
|---|---|
| subsample vocab | 14,988 (0% OOV) |
| cohort | 719 dense-active users, median 13 test interactions |
| model test NDCG@10 | **0.0171** vs popularity **0.0116** (beats floor; MRR 0.0173 vs 0.0104) |
| model across α (0→0.8) | 0.015–0.017 (pure-S strongest — a proper sequential recommender) |

## Phase 4 — drift signal (and the P0.3 answer)

Tested both signals on the *functional* backbone:

| signal | stable-mean D | category-change lift |
|---|--:|--:|
| **short_term_drift** (primary) | **0.258** | **1.60** |
| ls_cosine (true L–S, ablation) | 0.835 | 1.04 |

**P0.3 resolved with evidence:** even with both representations now predictive, L
(pooled embedding) and S (GRU output) sit in different geometric spaces, so true L–S
cosine stays high and insensitive. `short_term_drift` (S vs its own slow average — a
valid long/short divergence) is empirically far better and is kept as primary; the
abstract/Chapter-1 claim is rewritten to match it. Category-change lift rose from ~1.05
on the broken backbone to **1.60** — the fixed backbone made the signal meaningful.

## Phase 7 — detection under controlled injection (n=83 injected users)

| scenario | detection rate | median latency | false-alarm |
|---|--:|--:|--:|
| sudden | **0.39** | 3 | 0.012 |
| recurring | **0.39** | 3 | 0.012 |
| gradual | 0.18 | 7 | 0.012 |

Up from ~0.04–0.05 on the broken backbone.

## Phase 8 — detector operating characteristic (sudden)

| z-enter | detection | false-alarm |
|--:|--:|--:|
| 1.50 | **0.88** | 0.048 |
| 1.75 | **0.65** | 0.024 |
| 2.00 | 0.33 | 0.000 |

The headline result: **65% detection at 2.4% false-alarm** (was 17% @ 10% on the broken
backbone) — a genuine operating characteristic well above the chance diagonal.

## Phase 8 — recovery (detected subset, sudden, n=32) — honest limitation stands

pre-drift ~0.024–0.030, post-drift min ~0.0001, recovery latency ≈ 6 for **all**
policies (static-A, window-B, continuous-D, proposed). The drift-aware policy does
**not** yet beat the trivial sliding window (window-B) on recovery — exactly the
supervisor's point. window-B recovers fast *because it always ignores long-term memory*;
its cost is stability, which the recovery metric does not capture. Confronting this
directly (stability/adaptation trade-off, not raw recovery) is the open analysis task.

## Net

The backbone fix converts the project from "no positive result" to **a real, strong
detection result** (65%/2.4%) on a recommender that beats popularity. The recovery/α
advantage remains unproven and is the honest next question, alongside the supervisor's
baselines/seeds/write-up items.
