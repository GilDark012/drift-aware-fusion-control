# Second-dataset validation — Electronics alone (generalisation)

The main adaptation results use a subsample mixing three product categories. To test
whether they generalise to a different data distribution, the coherent-drift adaptation
experiment was repeated on a single, distinct domain. `scripts/second_dataset.py` with a
category filter added to the subsample builder.

## Setup

The Electronics category alone was used (Books, after per-category k-core, left too few
long-history users: 21; Electronics leaves 241 at k-core 5, comparable to the mixed
cohort's 196). A fresh backbone was trained on the Electronics-only dense cohort, and the
20-seed per-scenario comparison of the CUSUM-driven policy against static-alpha=0.8 and
window-B was repeated. Thresholds were relaxed slightly for the single domain
(min-test 8, min-train 30) to reach a comparable cohort size; k-core and everything else
match the main pipeline.

## Result (20 seeds, overall NDCG@10)

| scenario | static-0.8 | window-B | proposed | proposed vs window-B / vs static-0.8 |
|---|--:|--:|--:|---|
| sudden | 0.01072 | **0.01656** | 0.01500 | robustly worse / **robustly better** |
| gradual | **0.01566** | 0.01379 | 0.01477 | **robustly better** / tie |
| recurring | 0.01227 | **0.01761** | 0.01643 | robustly worse / **robustly better** |

## Reading — an honest, partial replication

The result does **not** cleanly reproduce the mixed-cohort headline, and it is more
informative for that reason.

- **The specific window-B win is partly domain-specific.** It replicates on gradual drift
  (proposed robustly beats window-B, +0.00098) but not on sudden or recurring, where on
  Electronics window-B is instead the *strong* baseline and robustly beats the proposed
  policy.
- **The deeper structure holds, and matches the encoder ablation.** In every Electronics
  scenario the proposed policy is the **middle** policy: never the worst, and it
  **robustly beats whichever fixed policy is weak** - static-0.8 on sudden and recurring
  (by a large margin, since static-0.8 collapses post-drift on this domain), window-B on
  gradual. Which fixed policy is strong flips with the domain exactly as it flipped with
  the encoder in Section 3.11, and again it is not knowable in advance.
- **But the strong-form claim is weakened.** On the mixed cohort the proposed policy was
  best-or-tied in every scenario; on Electronics it is robustly beaten by window-B on
  sudden and recurring. So the adaptive policy is a robust hedge - it avoids the failure
  mode of the wrong fixed choice and beats it - but it is **not uniformly the best**, and
  on some domain and scenario combinations a fixed policy beats it.

## Conclusion for the thesis

The generalisation test qualifies the headline honestly. The claim that survives across
both the encoder ablation and this second domain is the **robust-hedge** claim: the
adaptive policy is never the worst policy and robustly beats the fixed policy that happens
to be wrong for the current domain, encoder and scenario - a real value when that choice
cannot be made ahead of time. The stronger claim, that it beats the trivial sliding window
specifically, is shown to be partly a property of the mixed corpus rather than a universal
result. Reporting both, rather than only the favourable mixed-cohort numbers, is the
honest position; a fuller generalisation study across more domains and backbone seeds is
the natural next step.
