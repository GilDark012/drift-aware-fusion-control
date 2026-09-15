# Confronting window-B — the stability/adaptation trade-off (P1.4)

The supervisor's sharpest point: a **trivial sliding window** (window-B: `α=0`, pure
short-term) recovers as fast as — or faster than — the proposed drift-aware policy, so
what does the contribution buy? This document confronts that head-on with the measured
numbers, and does **not** deflect to the detected subset.

## The measured table (Phase-8, subsample backbone)

Per-policy quality on the injected cohorts. `pre` = mean hit@10 in the **pre-onset**
(stable) window; `recov` = recovery latency in steps; detected subset, sudden n=32.

| scenario | policy | α when stable | **pre-drift hit@10** | recovery latency |
|---|---|--:|--:|--:|
| sudden | **window-B** | **0.0** | **0.0298** | 6 |
| sudden | static-A | 0.8 | 0.0249 | 6 |
| sudden | continuous-D | ≤0.8 | 0.0240 | 6 |
| sudden | proposed | 0.8 | 0.0242 | 6 |
| gradual | **window-B** | **0.0** | **0.0448** | 9 |
| gradual | proposed | 0.8 | 0.0359 | 2 |
| gradual | static-A | 0.8 | 0.0359 | 2 |

(Full table: `artifacts/runs/.../policy-scenario-summary.csv`.)

## What this actually shows — an honest negative result

The expected story was "window-B recovers fast **but pays a pre-drift quality cost** by
throwing away long-term memory." **The data refutes that story on this backbone:**
window-B has the **highest** pre-drift quality of every policy
(0.0298 vs the proposed policy's 0.0242 — window-B is **+23 % better before any drift**),
*and* it recovers at least as fast. It pays no cost we can measure.

The reason is upstream of the detector: on data this sparse, the **long-term
EWMA-pooled representation `L` carries essentially no useful ranking signal** — the
short-term GRU `S` carries all of it. This is visible directly in the backbone sweep
(`docs/rerun_fixed_backbone.md`): quality is **monotone in "how much short-term"**, with
pure-S (`α=0`) the strongest configuration everywhere, drift or no drift.

Consequently there is **no operating regime in this dataset where mixing in `L`
helps**, so a policy whose whole job is to *modulate the `L/S` mixture* cannot beat the
policy that simply sets `α=0` and stays there. The proposed policy moves α in the right
*direction* on confirmed drift (toward short-term, 0.8→0.2), but since `α=0` is uniformly
optimal, sitting at 0.8 while "stable" is a self-inflicted penalty, not a stability
benefit.

## Does a "stability" reframing rescue it? No.

One might argue window-B is *unstable* — reacting to every blip — and that stability is a
value the recovery metric misses. But window-B's α is **constant** (0 α-changes per 1000
steps), so it is not unstable in the control sense, and it is empirically **more
accurate** in the stable window, not less. The stability/adaptation trade-off we hoped to
exhibit **does not exist on this backbone**, because the long-term anchor that stability
would protect is not worth protecting here.

## What survives, and what this means for the thesis

This is a clean, defensible outcome — a **characterised negative result** alongside a
**positive one**, which is exactly the intellectual honesty the review asked us to keep:

1. **Stands:** a validated sequential recommender (beats popularity, +47 % NDCG) and a
   **strong, seed-robust drift-detection result** (0.644 ± 0.014 detection @ 2.4 %
   false-alarm, `docs/seed_robustness.md`). Detecting persistent short-term shifts works.
2. **Not demonstrated:** that drift-aware **fusion control** improves recommendation
   quality. It cannot be, on this data, because the thing it controls (the `L/S` balance)
   has no useful `L` end. window-B is therefore **not a baseline we beat — it is the
   empirical ceiling**, and we state that plainly.

## The concrete next step this dictates

The adaptation half of the thesis needs a setting where **long-term memory demonstrably
helps** — otherwise there is no fusion to adapt. Options, in order of effort:

- **Denser users / longer histories** (higher k-core, users with 100+ interactions) so
  the long-term profile becomes informative and `α>0` can win pre-drift;
- a **stronger long-term encoder** than EWMA-pooling (e.g. an attention pool or a second
  GRU), so `L` is not near-useless;
- failing both, **report the negative result as the finding** and narrow the thesis claim
  to detection, dropping the recommendation-quality-improvement claim entirely.

Either way, the current draft must not claim an adaptation advantage it does not have.
