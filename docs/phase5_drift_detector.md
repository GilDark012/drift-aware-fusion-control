# Phase 5 — Persistence-Aware Drift Detector

**Code:** `src/drift_reco/detectors/` (threshold, detector, analysis; plots extended).
**Script:** `scripts/detect_drift.py`. **Run:** `artifacts/runs/*phase5-drift-detector`.
**Chosen config:** `configs/detector.json`. **Decision:** [ADR-0003](decisions/ADR-0003-persistence-aware-detector.md).

## The detector

A state machine over the causal divergence score:

```
STABLE ──(elevated)──▶ EMERGING ──(persists ≥ P)──▶ CONFIRMED ──(relaxes)──▶ STABLE
                          │
                          └──(fades early)──▶ TEMPORARY  (counted, never confirmed)
```

- **Score** — `adaptive` strategy: the causal **z-score** of `D_u(t)` against the
  user's own trailing baseline (so the threshold means the same for calm and volatile
  users). `global` strategy (raw `D`) is retained for the H6 ablation.
- **Persistence `P`** — a rise must stay elevated for `P` steps (with a small gap
  tolerance) before it is CONFIRMED; this is the false-alarm control.
- **TEMPORARY** — a rise that fades before persisting is recorded as a temporary
  deviation and never confirmed — directly implementing §4/§5's noise robustness.
- **Recovery** — CONFIRMED is held until the score relaxes for `recovery_persistence`
  steps, then returns to STABLE ("new stable preference").

Everything is causal; thresholds are tuned on **validation only**.

## Validation results (val cohort, 849 users, train+val divergence)

### H5 — persistence suppresses false alarms ✅

| persistence | 1 | 3 | 5 | 8 | 12 |
|---|--:|--:|--:|--:|--:|
| confirmed / 1000 steps | 2.43 | 0.73 | **0.07** | 0.00 | 0.00 |

Monotone, order-of-magnitude reduction — a single elevated step confirms 35× more
often than a 5-step persistence requirement.

### H6 — adaptive threshold is uniform across volatility ✅

confirmed / 1000 steps by volatility (divergence-std) stratum:

| stratum (low→high volatility) | 0 | 1 | 2 |
|---|--:|--:|--:|
| **adaptive** (z-score) | 0.00 | 0.00 | 0.15 |
| **global** (raw D, enter=q90) | 2.49 | 3.57 | 4.56 |

A global threshold fires ~2–4× more on volatile users; the per-user adaptive
threshold is essentially flat — exactly H6.

### Chosen operating point

`adaptive, enter=2.5, exit=1.0, persistence=5` → **0.07 confirmed / 1000 steps** on
quiet validation data (deliberately conservative false-alarm control), while
temporary deviations (6.2 / 1000 steps) are caught and correctly *not* confirmed.

## Qualitative validation

The detector overlays (`figures/detection-user-*.png`) show the mechanism working
end to end: e.g. **user 21128** is stable (~0.4) through early 2017, the divergence
climbs mid-2017, the detector marks **onset**, **confirms** after persistence,
**holds** through the shift, and **returns to STABLE** once a new plateau settles —
the full STABLE→EMERGING→CONFIRMED→new-stable cycle.

## Correctness

- Causal throughout; state-machine transitions, persistence, gap-tolerance, recovery,
  warmup and both threshold strategies are unit-tested (18 new tests, incl. adaptive
  integration). ruff + mypy clean.

## Caveats & next

- Alarm rate on natural validation data is a **false-alarm proxy** (no ground truth
  yet). Detection **power/latency** is measured against known onsets in **Phase 7**;
  `persistence` is swept as the latency↔false-alarm knob in **Phase 8**.
- **Phase 6** drives `α(t)` from the detector state — the adaptive fusion.
