# ADR-0004 — α as an online control; policies unify the baselines

- **Statut** : accepté
- **Date** : 2026-08-18

## Contexte

The project's contribution is *controlling* the long/short fusion weight α online,
not retraining. Baselines A–D and the proposed system must therefore be **the same
backbone under different α policies**, evaluated by the **same streaming loop**, so
any difference isolates the α-control mechanism. We also must decide how α moves once
a shift is confirmed — abruptly or gradually (H4).

## Options envisagées

| Option | Avantages | Inconvénients |
| --- | --- | --- |
| A separate evaluator/model per baseline | conceptually simple | comparisons no longer isolate α-control; duplicated code |
| **One `StreamingEvaluator` + an `AlphaPolicy` family** | static (A), continuous (D), state-abrupt/gradual (proposed) all share one backbone & loop; `L`/`S` computed once and re-fused under each α | policies must agree on a common interface |
| α jumps to target on confirmation (abrupt only) | reactive | can destabilise recommendations; no stability/adaptation trade-off to study |
| α rate-limited toward target (gradual) | smooth, tunable | slightly slower to reach the short-term-dominant regime |

## Décision

Introduce an `AlphaPolicy` family — `static`, `continuous`, `state` — behind one
`StreamingEvaluator` that replays each user's test stream, computes `L`/`S` once, and
re-fuses under the policy's per-step α. The proposed system is the **state** policy
mapping detector states to α (STABLE 0.8 → EMERGING 0.5 → CONFIRMED 0.2), moved
**gradually** by default (`alpha_step`), with **abrupt** retained as the H4 ablation.
TEMPORARY deviations map to the STABLE α, so noise never moves α. Chosen config:
`configs/adaptation.json`.

## Conséquences

- Baselines A (static), D (continuous, detector-free) and the proposed drift-aware
  policy are one-line configuration changes over an identical evaluation — the
  cleanest possible isolation of the contribution.
- Because divergence, detector and policy are causal, the batched streaming passes
  equal a true online loop; α at step *t* never depends on the future.
- **Natural-data finding:** on the short test windows the detector confirms almost
  never (`confirmed_frac≈0`), so the state policy rests near α_stable and the
  policy differences on natural data mostly reflect α *level*, not adaptation. This
  makes the **controlled-injection experiment (Phase 7) the decisive test**, and is
  stated as such rather than hidden.
- The synthetic control-law figure shows the intended behaviour unambiguously:
  gradual glides α down/up, abrupt steps it, continuous jitters with noise (adapting
  without drift-awareness) — motivating the persistence-confirmed design.
