# Phase 8 — Full Evaluation (v1)

**Code:** `src/drift_reco/evaluation/recovery.py` (+ `plots.py`, `latency.py`).
**Script:** `scripts/full_evaluation.py`. **Run:** `artifacts/runs/*phase8-full-evaluation`.

This is the **first stable version** of the end-to-end evaluation, built on the agreed
framing: show the drift-aware advantage **where drift is detected**, alongside honest
full-cohort numbers, plus the detector's operating characteristic. Absolute numbers are
subsample-limited by design; strengthening them (full-scale backbone) is the planned
next step.

## What it produces

For each scenario (sudden / gradual / recurring) it streams every injected user under
four α-policies over one shared backbone — **static-A** (α=0.8, no adaptation),
**window-B** (α=0, pure short-term), **continuous-D** (adaptive, no detector),
**proposed** (detector-driven, gradual) — and:

1. aligns per-step **reciprocal rank (MRR)** to the known onset → **recovery curves**
   (`figures/recovery_full_*`, `recovery_detected_*`);
2. summarises pre-drift quality, post-drift minimum, degradation and **recovery
   latency**, on the **full cohort** and the **detected subset**
   (`data/policy-scenario-summary.csv`);
3. sweeps the detector threshold → the **detection/false-alarm trade-off curve**
   (`figures/detection-tradeoff-sudden.png`).

## Headline result — detector operating characteristic (clean)

Sweeping the adaptive z-threshold on the sudden scenario (injected vs control streams):

| z-enter | detection rate | false-alarm rate |
|---:|---:|---:|
| 1.50 | 0.457 | 0.180 |
| 1.75 | 0.177 | 0.047 |
| 2.00 | 0.037 | 0.003 |
| 2.50 | 0.000 | 0.000 |

The curve sits **above the diagonal** at every point — the signal genuinely
discriminates injected shifts from control variety — but recall is capped (the Phase-7
finding). This is the cleanest, most defensible quantitative result of the project: a
real operating characteristic with an honest recall ceiling.

## Recovery quality — honest, and currently weak

On the subsample backbone, absolute quality is near the floor (MRR ≈ 4e-4; hit@10 ≈
3e-3 over a 220k-item catalog), and the **detected subset is small (~11 users/scenario)**,
so the recovery curves do **not** yet separate the policies meaningfully. This is
expected: the quality dynamics need a stronger recommender to become visible. The
harness is complete and correct; the number is a limitation, not a design flaw.

## Baselines covered (§13)

- **A static** (no drift awareness), **B window** (pure short-term), **D continuous**
  (adaptive without detection), **proposed** (drift-aware). This isolates: long/short
  fusion (A vs B), adaptive α (A vs D), and **drift-awareness** (D vs proposed).
- Baseline C (periodic update) is deferred to v2 (needs a scheduled-α policy).

## Ablations (status)

Available via config today (signal `short_term_drift` vs `ls_<metric>`; α transition
gradual vs abrupt; detector threshold/persistence). The threshold sweep above is the
first ablation. A systematic ablation table (signal, gradual-vs-abrupt, persistence,
window size) is the v2 task.

## Correctness

Recovery alignment, curve/summary and trade-off are unit-tested (9 new tests). All
policies run through one `StreamingEvaluator`; ruff + mypy clean; 105 tests;
compliance 16/0/0.

## Next (dig deeper)

1. **Stronger / full-scale backbone** → raise absolute quality and detection recall so
   the recovery curves separate.
2. **Baseline C** + a systematic **ablation table**.
3. Re-run Phases 5–6 numbers under the new primary signal for a consistent narrative.
