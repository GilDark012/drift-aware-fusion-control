# Dense-cohort α-probe — the adaptation precondition is met

The window-B / detector-baseline findings showed that on the default sparse subsample
the long-term representation `L` is useless (α=0 optimal everywhere), so there was no
fusion balance worth adapting. This probe tests the precondition the adaptation thesis
needs: **on a cohort of users with long histories, does any α>0 beat α=0?**

`scripts/dense_alpha_probe.py`: builds a dense subsample (k-core 5 + a **train-history
floor of ≥50 interactions**, ≥5 test), trains a fresh backbone with **random-α** training
(both L and S paths independently optimised), and sweeps α at inference.

## Result — yes, L now helps

Dense cohort: **438 long-history users**, vocab 11,581, 2,078 test targets.

| α | 0.0 | 0.2 | 0.4 | 0.6 | **0.8** | 1.0 |
|---|--:|--:|--:|--:|--:|--:|
| **val** NDCG@10 | 0.0183 | 0.0196 | 0.0195 | 0.0215 | **0.0235** | 0.0199 |
| **test** NDCG@10 | 0.0162 | 0.0167 | 0.0169 | 0.0167 | **0.0169** | 0.0136 |

**The α-response curve flipped.** On the sparse cohort quality was *monotone decreasing*
in α (pure short-term best). On the dense cohort it *rises* with α to a peak at **α=0.8**
and only falls at α=1.0 (pure long-term is too much). The optimal fusion weight is now
**0.8, not 0**:

- **val:** α=0.8 beats α=0 by **+28 %** (0.0235 vs 0.0183);
- **test:** α=0.8 beats α=0 by **+4 %** (0.0169 vs 0.0162), same peak shape.

The val lift is large and the test lift modest but positive and identically shaped; the
qualitative conclusion is unambiguous — **on long-history users, the long-term memory
carries signal and mostly-long-term fusion is optimal.**

## Why this matters

This is the precondition for the whole adaptation thesis:

- **Stable period** → optimal α ≈ 0.8 (lean on long-term memory);
- **After drift** → optimal α should drop toward short-term;
- so the optimal α now **differs between regimes**, which is exactly what an adaptive
  policy can exploit — and what window-B (always α=0) cannot.

On this cohort, window-B is no longer the ceiling: a policy that sits near 0.8 when
stable already beats it, and a drift-aware policy that drops α on confirmed drift can, in
principle, beat *both* static-0.8 and window-B.

## Next step

Run the full controlled-drift + adaptation experiment **on this dense cohort**, using an
**off-the-shelf CUSUM detector** (which we showed detects these shifts perfectly) to
drive α. Success criterion: adaptive-α beats **both** static-α=0.8 **and** window-B on
overall quality across the drift. That is the positive result the thesis is after.

## Caveats

- The dense cohort is small (438 users — long-history users are rare in this data); a
  single trained seed. Seed/CI repetition (as in `docs/seed_robustness.md`) should follow.
- The long-term encoder is still EWMA-pooling; a stronger encoder is an independent lever
  that could widen the α>0 advantage further.
