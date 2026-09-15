# Phase 3 — Long/Short-Term Fusion Baseline

**Code:** `src/drift_reco/models/` (config, sequences, fusion, train, evaluate) +
`src/drift_reco/data/subsample.py`. **Scripts:** `scripts/build_subsample.py`,
`scripts/train_baseline.py`. **Runs:** `artifacts/runs/*phase3-*`.

## What this establishes

The shared recommender backbone and **Baseline A** (fixed-α, no drift awareness).
Every later baseline (B–D) and the proposed drift-aware system reuse *this same
trained backbone* and differ only in the α policy — so any measured difference is
attributable to α-control, not to representation quality (ADR-0002).

## Architecture (`h = α·L + (1−α)·S`)

- **Item embedding space** shared by both signals, so `L`, `S` and the drift signal
  `D_u(t)=dist(L,S)` all live in one space.
- **`S_u(t)` short-term** — GRU over the last `short_window` items (default 10),
  read at its last real step (padding masked via lengths).
- **`L_u(t)` long-term** — recency-weighted pool (EWMA, half-life 20) of prior
  history, capped to `max_history` (default 50) most recent items.
- **Scoring** — `h @ E.ᵀ + item_bias`, PAD and OOV columns masked out.
- **Training** — sampled-softmax with 100 uniform negatives; **α ~ U(0,1) per
  example** (ADR-0002) so both paths stay independently predictive.

## Development subsample (`data/interim/subsample/`)

Reproducible (seed 42), built by `SubsampleBuilder`:

| Property | Value |
|---|---|
| eval cohort (≥20 test interactions) | 3,000 users (from a pool of 7,918) |
| + extra train-only users (vocab enrichment) | 20,000 → 23,000 train users |
| item vocabulary | 220,736 |
| rows (train / val / test) | 478,789 / 11,693 / 81,724 |
| unseen-item rate (val / test) | 48.3% / 53.8% — matches full-scale ~49% |

The extra train-only users are **never evaluated**; they exist so the item
embeddings get realistic signal and the OOV rate mirrors full scale (Phase 2 Risk B),
making the subsample a faithful tuning harness rather than a degenerate one.

## Baseline results (subsample, 5 epochs, fixed α=0.5)

Temporal, full-vocabulary ranking over all 220,736 items.

| split | NDCG@10 | HR@10 | MRR | NDCG@10 (seen) | HR@10 (seen) | scored / seen-target |
|---|--:|--:|--:|--:|--:|--:|
| val | 0.0038 | 0.0079 | 0.0032 | 0.0073 | 0.0151 | 11,198 / 5,821 |
| test | 0.0034 | 0.0070 | 0.0029 | 0.0074 | 0.0150 | 80,290 / 37,230 |

Train loss 4.62 → 3.50 over 5 epochs on 434,898 examples.

### Reading these numbers honestly

- Absolute values are **low but far above chance**: random HR@10 over a 220k-item
  catalog ≈ 4.5e-5, so seen-HR@10 = 0.015 is ~**330×** random. The signal is real.
- The **seen** columns condition on rankable targets; ~half of test targets are OOV
  (unseen items, Phase 2 Risk B) and are unrankable by construction — counted as
  misses in the full metric, excluded from the seen metric.
- Per the brief, **NDCG is not the objective**; this backbone exists to study
  *relative* behaviour of α-policies around drift (detection/adaptation/recovery
  latency). Absolute quality can be raised later (epochs, dim, popularity warm-up)
  but is deliberately not over-optimised now.

## Correctness / anti-leakage properties

- Evaluation is strictly chronological; each position is scored from prior history
  only (earlier splits as context + already-seen eval prefix). Same example
  primitive builds train and eval — they cannot diverge.
- Thresholds/tuning touch **val only**; test is read once for reporting.
- 48 unit tests (padding, weighting, masking, convex fusion, end-to-end train/eval,
  subsample OOV mapping); mypy strict + ruff clean.

## Next

Phase 4 computes `D_u(t)=dist(L,S)` over this backbone and validates it on
representative users — the first drift signal.
