# ADR-0006 — k-core vocabulary filtering (the backbone-collapse fix)

- **Statut** : accepté
- **Date** : 2026-08-25

## Contexte

The supervisor review flagged that the reported NDCG@10 = 0.00224 is 17–30× below a
competent SASRec baseline and demanded popularity/random floors. Investigation
confirmed the backbone was non-functional and traced the root cause.

## Sources used to find the problem (as requested, logged explicitly)

1. **Supervisor review (2026), point #2** — the trigger. It stated that SASRec reports
   NDCG@10 ≈ 0.039–0.045 on Gowalla (173k items) and ≈ 0.071 on Amazon Kindle, i.e. our
   model sits far below a standard baseline on a catalogue of the same order; and it
   asked for the missing popularity and random floors.
2. **Own diagnostic runs, this session** (direct empirical evidence):
   - **Popularity/random floors:** model NDCG@10 **0.00224** vs **popularity 0.00270**
     vs random 2.1e-5 → *the trained model ranks below "recommend the 10 most popular
     items."*
   - **Layered scorer probe** (same test positions, scored 5 ways): pure-S (GRU) was the
     *worst* scorer; the model was below popularity at every α; `corr(item_bias,
     log-popularity)=0.47`.
   - **GRU probe:** `S·E[target]=2.05` vs `S·E[random]=−0.55` (the GRU *does* learn
     above random) and a recently-repeated item ranked at median 99/220,736 — real but
     far too weak a signal.
   - **Item-support distribution (the smoking gun):** subsample train = 478,789
     interactions over **220,736 items = 2.17 interactions/item**; **67% of items appear
     exactly once**; only 7.3% have ≥5 support; **53.8% of test targets had zero train
     support (unrankable).**
3. **Literature convention** — k-core filtering is the standard preprocessing step in
   sequential recommendation and is applied by the exact baselines the supervisor cited:
   **GRU4Rec** (Hidasi et al., 2016) and **SASRec** (Kang & McAuley, 2018) both filter to
   ≥5-core dense vocabularies. Our subsample pipeline omitted this step; the 20k
   "vocabulary-enrichment" users (added in Phase 3) inflated the vocabulary with
   singletons rather than adding support.

## Décision

Add **k-core item filtering** to `SubsampleBuilder` (`data/subsample.py`,
`SubsampleConfig.k_core`, default 5): keep only items with ≥ `k_core` training
interactions across all splits, so the vocabulary is learnable and every test target is
rankable. This is the missing standard step, not a new technique.

## Conséquences (measured)

Rebuild at 5-core:

| | before | after (5-core) |
|---|--:|--:|
| vocabulary | 220,736 | **16,026** |
| interactions / item | 2.17 | **11.9** |
| singleton items | 147,851 | **0** |
| test-target OOV (unrankable) rate | 53.8% | **0%** |

- The "unseen-item phenomenon" (Phase-2 Risk B) is intentionally removed for the
  learnable backbone; it can be studied separately as a robustness question.
- Cohort test-lengths shrink (dense items only); `min_test_interactions` may need
  re-tuning for the latency cohort.
- **Validation:** a backbone retrained on the dense subsample is checked against the
  (recomputed) popularity floor; result recorded in the run journal and appended below.
- All Chapter-3 numbers must be re-run on this backbone; the previous ones stand on the
  collapsed vocabulary and are void.

## Validation result — the fix works

Backbone retrained on the dense 5-core subsample (dim 128, fixed α=0.5, uniform
negatives, 15 epochs), evaluated with full-catalogue ranking; popularity floor
recomputed on the same 5-core test set.

| test @10 | Recall | NDCG | MRR |
|---|--:|--:|--:|
| **Model** | **0.0325** | **0.0156** | **0.0144** |
| Popularity floor | 0.0294 | 0.0128 | 0.0105 |
| Model (val) | 0.0409 | 0.0201 | 0.0176 |

The model now **beats the popularity floor on every metric** (NDCG +22%, Recall +11%,
MRR +37%), and absolute NDCG@10 rose **0.00224 → 0.0156 (~7×)** into the right order of
magnitude for a small-subsample GRU fusion. **Root cause confirmed and fixed:** the
collapse was vocabulary sparsity (missing k-core step), not the architecture, the GRU,
or the evaluation — all of which the diagnostics had already shown to be functioning.

**Downstream:** all later phases (divergence signal, detector, adaptation, injection,
Phase-8 evaluation) must be re-run on this backbone; their prior numbers are void.
