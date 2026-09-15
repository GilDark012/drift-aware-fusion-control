# ADR-0005 — Drift signal: short-term-vs-slow-average; and the detection limit

- **Statut** : accepté
- **Date** : 2026-08-19

## Contexte

Phase 7 exposed that the original signal `D=1-cos(L,S)` is unusable on this backbone:
`L` (pooled embedding) and `S` (GRU output) live in different geometric spaces and
are near-orthogonal for *every* user (natural D≈1.0–1.5), so injected shifts cannot
raise it above its ≈0.35 noise, and are undetectable (3% detection).

## Options essayées et résultat

| Attempt | Outcome |
| --- | --- |
| Retrain with an L-S cosine **alignment loss** (`align_weight`) | ineffective: mean cos(L,S) moved only −0.05→+0.03 at weight 1.0; the GRU-output and pooled-embedding spaces do not align without destroying the recommender |
| **Short-term-vs-slow-average signal** `1-cos(S_t, EWMA_slow(S))` | **adopted.** Stable D drops to ≈0.4–0.57; both terms are in the GRU space, so a genuine shift raises D and persists. Category-change lift improved to ×1.10 |
| **Stronger backbone** (dim 64→128, more epochs) | marginal for detection; confirmed the limit is not capacity |
| S-space-targeted injection (anti-profile of the user's own S̄) + short-window detector | needed to make *any* shift detectable |

## Décision

1. The **primary drift signal is `short_term_drift`** = `1 − cos(S_t, slow-EWMA(S))`
   (`detectors/signal.py`), with the original `ls_<metric>` (L-S divergence) kept as
   an **ablation**. The streaming/trajectory code computes it over a **context
   warm-up** so the slow baseline is established from history, not reset at the eval
   boundary.
2. A **short-window detector** config for the streaming regime
   (`configs/detector_streaming.json`: window 10, warmup 3, persistence 3); the
   Phase-5 long-stream config remains the ablation.
3. Controlled injection switches post-onset items to the cluster whose short-term rep
   is **most opposed to the user's established profile** (`data/injection.py`).

## Conséquences (et limite assumée)

- The signal is now well-posed (low-for-stable, rises on genuine shift), faithful to
  the long/short-divergence idea (both terms are short-term reps at different time
  scales).
- **Detection remains bounded (~5% at 1% false-alarm; ~17% at 10%).** Root cause,
  established empirically and *not* fixed by a stronger backbone: **Amazon purchase
  sequences are intrinsically noisy at short horizons** (users buy diverse items), so
  an injected preference shift is hard to separate from normal variety. This is a
  genuine **threat to validity** of representation-divergence drift detection on
  sparse e-commerce data, stated as such — not hidden.
- Consequence for Phase 8: the drift-aware advantage is demonstrated on the **subset
  of clearly-detected shifts** (where the mechanism provably helps), alongside honest
  full-cohort numbers. Absolute detection recall is reported as a limitation, with a
  much larger/full-scale backbone identified as future work.
- `align_weight` is retained (default 0) as a documented, unused knob and an ablation.
