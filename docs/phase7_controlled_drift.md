# Phase 7 — Controlled Drift Injection (and a key negative finding)

**Code:** `src/drift_reco/data/injection.py`, `src/drift_reco/detectors/signal.py`,
`src/drift_reco/evaluation/{latency,plots}.py`. **Script:** `scripts/inject_drift.py`.
**Artifacts:** `data/interim/subsample/injected_{sudden,gradual,recurring}.parquet` +
`onsets_*.json`. **Decision:** [ADR-0005](decisions/ADR-0005-ls-alignment-training.md).

## What this delivers

1. A **reproducible controlled-drift framework** that replaces the broken
   `test_drifted.parquet` (Phase-2 Risk C): sudden / gradual / recurring shifts that
   change the user's **preference sequence** (real, in-vocabulary, index-consistent
   items) with a **known onset per user**.
2. A **re-posed drift signal** that actually works: `short_term_drift` =
   `1 − cos(S_t, slow-EWMA(S))` (stable D≈0.4–0.57 vs ≈1.06 for the old L-S cosine).
3. The **latency framework** (`evaluation/latency.py`) measuring detection latency
   against the known onset, with premature/false alarms counted separately.

## The injection

For each user, after an onset at 40% of their test stream, post-onset items are drawn
from the cluster whose **short-term representation is most opposed to the user's own
established profile** `S̄` (mean recent-context short-term rep). This is a deliberately
**strong, clean** controlled shift so that latency is measurable; real drift is weaker
and messier (a stated limitation). Metadata (`item_id/item_idx/category`) is rewritten
consistently with the new `s_item`.

## Honest results (200 injected users/scenario, short-window detector, α=0.8)

| scenario | detection rate | median latency | false-alarm rate |
|---|--:|--:|--:|
| sudden | 0.045 | 5 | 0.01 |
| gradual | 0.040 | 12 | 0.01 |
| recurring | 0.050 | 8 | 0.01 |

At a more lenient threshold (`enter=1.75`) detection rises to ~0.17 but false alarms
rise to ~0.10 — a hard trade-off.

## The key finding (negative, and important)

**Representation-divergence drift detection is only weakly effective on this data**,
and it is *not* a bug or a capacity problem. Established through systematic diagnosis:

- The original `D=1-cos(L,S)` is unusable — `L` (pooled embedding) and `S` (GRU
  output) are near-orthogonal for everyone; injected shifts can't raise it.
- The re-posed `short_term_drift` signal fixes the *stable* baseline (D≈0.4) and is
  the right formulation, but detection still tops out ~5% (1% FA) / ~17% (10% FA).
- Brute-forcing the maximum achievable signal per user, training a **stronger backbone
  (dim 64→128)**, and S-space-targeted injections all failed to lift it materially.
- **Root cause: Amazon purchase sequences are intrinsically noisy at short horizons**
  — users buy diverse items, so a genuine preference shift barely stands out from
  normal variety. The signal-to-noise for short-window divergence is inherently low
  on sparse e-commerce data.

This is a legitimate **threat to validity** for the methodology on this dataset,
reported openly rather than hidden behind a tuned number.

## Consequence for Phase 8

- Demonstrate the drift-aware advantage on the **subset of clearly-detected shifts**
  (where the mechanism provably reduces recovery latency), alongside **honest
  full-cohort** metrics and the detection recall as a stated limit.
- Report the detection/false-alarm trade-off as a curve, not a single number.
- Full-scale training (2.78M interactions) and denser-domain datasets (MovieLens,
  Taobao) are the identified routes to stronger detection — future work.

## Correctness

- Injection is index-consistent and reproducible (seeded); onsets are known ground
  truth. Signal, injection, latency all unit-tested (12 new tests). ruff + mypy clean.
