# Phase 4 — Long/Short Divergence Signal `D_u(t)`

**Code:** `src/drift_reco/detectors/` (divergence, rolling, trajectory, plots) +
`src/drift_reco/models/checkpoint.py`. **Script:** `scripts/analyze_divergence.py`.
**Run:** `artifacts/runs/*phase4-divergence-signal`.

## What this delivers

The first drift signal: `D_u(t) = dist(L_u(t), S_u(t))`, the distance between a
user's persistent and recent representations **in the shared embedding space of the
Phase-3 backbone**. This is computed over the *same* `L`, `S` the recommender fuses,
so the signal and the recommendation are consistent by construction.

## Divergence metrics (`detectors/divergence.py`)

An abstract `DivergenceMetric` family, selected by name (no scattered dispatch):

- **`cosine`** (primary) — `1 − cos(L,S)`, scale-free, bounded [0,2].
- **`euclidean`** — `‖L − S‖`.
- **`sym_kl`** — symmetric KL of softmaxed activations (distributional alternative).

The alternatives exist for the §14 ablation (which divergence measure works best).

## Causal rolling statistics (`detectors/rolling.py`)

`RollingStatistics` turns a raw `D_u(t)` series into a trailing mean/std baseline,
an EWMA level, and a **trailing z-score** — all **causal** (computed from values at
or before `t`). This is what makes "persistent *increase* above the user's own
baseline" detectable in Phase 5 without leaking the future.

## Trajectories (`detectors/trajectory.py`)

`DivergenceTracker` produces, per user, the `D_u(t)` series annotated with timestamp,
item, category and a category-change mask. Divergence is only scored where **both**
representations exist (position beyond the short window), so early positions with an
empty long history no longer contaminate the signal.

## Validation on the subsample cohort (800 users, 27,842 positions)

| quantity | value |
|---|--:|
| divergence mean ± std (cosine) | 1.061 ± 0.376 |
| mean at a **category change** | 1.102 |
| mean at a **stable** position | 1.047 |
| **change/stable lift** | **×1.05** |
| category-change rate | 25.6% |

**Reading it honestly.** Divergence is measurably higher when the item category
changes (the interpretable indicator) — direction correct, and clear given n≈28k —
but the *natural* effect is modest (~5%). Two reasons this is expected and not a
problem:

1. **Category change is a coarse proxy** for a true preference shift (the brief
   forbids equating the two); many category switches are incidental, not drift.
2. **Absolute population lift is the wrong lens.** The detector (Phase 5) fires on a
   *per-user persistent rise above that user's own trailing baseline* (the z-score),
   not on the population mean. The decisive, ground-truthed test is **controlled
   injection** (Phase 7), where shift onset is known.

The representative trajectories make the phenomenon vivid: e.g. user 1091 shows a
stable 2016 followed by a **sustained rise in `D_u(t)` through 2017–2018** — a
long-term profile becoming progressively inconsistent with recent behaviour, exactly
the situation where the fixed-α fusion is becoming inappropriate.
(`figures/trajectory-user-*.png`, `population-category-effect.png`.)

## Correctness

- All rolling statistics are causal; trajectory scoring uses only preceding history.
- Metrics, rolling causality, trajectory alignment and checkpoint round-trip are unit
  tested (15 new tests); ruff + mypy clean.

## Next

Phase 5 converts a persistent rise in `D_u(t)` into a **state-machine detector**
(STABLE → EMERGING → CONFIRMED, with TEMPORARY-DEVIATION handling), tuned on
validation, with false-alarm control.
