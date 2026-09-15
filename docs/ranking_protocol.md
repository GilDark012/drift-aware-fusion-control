# Ranking & evaluation protocol

Explicit statement of how every reported ranking number is produced, so the results
are reproducible and comparable to the literature. Drawn directly from
`src/drift_reco/models/evaluate.py` and `src/drift_reco/evaluation/`.

## Task

**Next-item prediction.** At each evaluated position `t` in a user's chronological
stream, the model ranks the *entire* item vocabulary and we record the rank of the
single held-out true next item `x_t`.

## Ranking is full-catalogue and unsampled

The target competes against **every** item in the vocabulary (14 988 items in the
development subsample), not against a sampled subset of negatives. Sampled ranking
(e.g. 1 positive vs 100 negatives) inflates metrics and is known to distort model
ordering (Krichene & Rendle, 2020); we avoid it. The score of item `i` for the fused
query `h_u(t)` is

```
score_u(t, i) = h_u(t) · e_i + b_i
```

(inner product with the item embedding plus the item bias), and the target's 1-based
rank is `rank = #{ i : score(i) > score(x_t) } + 1`.

## One positive per query ⇒ Recall@k = HitRate@k

Each query has exactly one held-out positive, so `Recall@k` and `HitRate@k` coincide;
both names are reported only for continuity with the sequential-recommendation
literature. Reported metrics, all at **k = 10**:

- **NDCG@10** — position-discounted, the headline ranking metric;
- **Recall@10 / HitRate@10** — is the target in the top 10;
- **MRR** — mean reciprocal rank, used for the recovery curves (reciprocal rank is
  defined at every step even when the target is outside the top-k).

## No temporal leakage

- **Global chronological split** at fixed timestamps `T1`, `T2` (see the data card):
  train `t<T1`, validation `T1≤t<T2`, test `t≥T2`. No future interaction can inform a
  past prediction.
- **Causal context per query.** Position `t` is scored from the preceding history
  only — earlier splits as context plus the already-seen prefix of the eval split
  (`build_eval_examples`). The long/short representations at `t` are built from items
  strictly before `t`.
- **Thresholds tuned on validation only.** The detector's enter/exit/persistence are
  set on the validation split; the test split is never used for tuning.

## Evaluable cohort

Per-user drift latency is only measurable for users with enough test-period history,
so the evaluable cohort is users with **≥ 20 test interactions** (ADR-0001). In the
development subsample this is **719 dense-active users** (median 13 test interactions
after k-core). Absolute NDCG is low by construction on data this sparse (density
≈ 4×10⁻⁵); the analysis is on **relative** movement vs the pre-drift baseline and on
detection/recovery latency, not on absolute NDCG.

## Feasible-subset (seen-only) reporting

The item vocabulary is built from **training items only**, so a test target absent from
training is out-of-vocabulary (OOV) and structurally unrankable — exactly the
"unseen-item" phenomenon of a live catalogue. We report metrics **two ways**:

- **full** — over all evaluable queries (OOV targets count as misses; this caps the
  achievable Recall@10 at `1 − OOV_rate`);
- **seen-only** (`ndcg_at_k_seen`, `hit_rate_at_k_seen`) — over queries whose target is
  in-vocabulary, isolating model quality from catalogue coverage.

Reporting both prevents the OOV rate from being mistaken for a modelling failure.

## Floors (sanity baselines)

Every ranking table is anchored by two non-learned floors, computed under the identical
protocol:

- **Popularity** — always rank items by training frequency;
- **Random** — uniform random ranking.

A trained model that does not clearly beat the popularity floor has learned nothing;
enforcing this caught the original broken backbone (ADR-0006). On the fixed backbone the
model reaches **NDCG@10 = 0.0171 vs popularity 0.0116** (+47 %).
