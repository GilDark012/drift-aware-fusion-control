# Phase 6 — Adaptive Fusion: α(t) from Detector State

**Code:** `src/drift_reco/adaptation/` (policy, plots), `src/drift_reco/evaluation/`
(metrics, streaming). **Script:** `scripts/adaptive_fusion.py`.
**Run:** `artifacts/runs/*phase6-adaptive-fusion`. **Config:** `configs/adaptation.json`.
**Decision:** [ADR-0004](decisions/ADR-0004-alpha-policy-and-streaming.md).

## The mechanism (the project's contribution)

α is controlled **online, without retraining**, over the frozen backbone. An
`AlphaPolicy` maps the causal drift signals to a per-step fusion weight:

| policy | α rule | role |
|---|---|---|
| `static` | constant | Baseline A (no drift awareness) |
| `continuous` | α = high − k·max(0, z) every step | Baseline D (adaptive α **without** detection) |
| `state` (abrupt) | α = level(state), stepped | proposed, abrupt (H4) |
| `state` (gradual) | α → level(state), rate-limited | **proposed** (default) |

State levels: STABLE 0.8 → EMERGING 0.5 → CONFIRMED 0.2; TEMPORARY maps to STABLE so
noise never moves α.

## The streaming evaluator

`StreamingEvaluator` replays each user's test stream: at every position it forms
`L`,`S` from prior history (**once**), measures divergence, updates the detector,
lets the policy pick α, and scores the next item under **that** α. Every baseline and
the proposed system run through this one loop — differences isolate α-control.
Ranking metric maths live in one shared `evaluation/metrics.py` (used by the Phase-3
evaluator too), so the two evaluators cannot disagree.

## Natural-data comparison (800 test-cohort users, eval=test)

| policy | NDCG@10 | HR@10 | mean α | α-changes/1k | confirmed_frac |
|---|--:|--:|--:|--:|--:|
| static-0.5 | 0.00264 | 0.00549 | 0.50 | 0 | 0 |
| continuous | 0.00271 | 0.00593 | 0.75 | 557 | 0 |
| state-abrupt | 0.00272 | 0.00598 | 0.79 | 22 | 0 |
| state-gradual | 0.00271 | 0.00598 | 0.80 | 50 | 0 |

**Honest reading.** `confirmed_frac ≈ 0`: the natural test windows (median ~20–80
interactions, per-window z-baseline) almost never sustain a 5-step confirmed rise, so
the state policy rests near α_stable=0.8. The small NDCG gap over static therefore
reflects **α level** (0.5 vs 0.8), not adaptation. `continuous` changes α ~557×/1k
(reacting to noise) vs the drift-aware `state` policies ~22–50×/1k — the detector-free
baseline adapts far more indiscriminately. **The benefit of drift-awareness cannot
show on natural data here; the controlled experiment (Phase 7) is the decisive test.**

## The control law works (synthetic illustration)

Feeding a low→ramp→plateau divergence through the **real** detector and policies
(`figures/alpha-policy-comparison-synthetic.png`) shows the intended behaviour
unambiguously:

- **gradual** glides α 0.8→0.2 as STABLE→EMERGING→CONFIRMED, holds, then ramps back;
- **abrupt** steps α between the three levels and snaps back (H4 contrast);
- **continuous** jitters with z-noise even in stable periods and never cleanly
  reaches the short-term regime — adapting *without* drift-awareness.

## Correctness

- Causal per-step α; `L`/`S` computed once per user; policies, streaming alignment,
  shared metrics all unit-tested (13 new tests). ruff + mypy clean.

## Next

Phase 7 injects sudden/gradual/recurring **preference-sequence** shifts with **known
onsets**, so detection/adaptation/recovery latency and the drift-aware advantage can
finally be measured against ground truth.
