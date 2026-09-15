# Run `20260825-163328__phase3-baseline-fixed-alpha`

- **Task**: phase3-baseline-fixed-alpha
- **Status**: completed
- **Started**: 2026-08-25T16:33:28.278569+00:00
- **Duration**: 92.01 s
- **Git commit**: 7e600cc39b00d3921248d4443656d9381154576b
- **Seed**: 42
- **Dataset**: amazon-subsample

## Objective

Establish the fixed-α long/short fusion baseline (Baseline A).

## Method

GRU short-term + EWMA long-term, α-random training, evaluated at fixed α=0.5 with full-vocabulary temporal ranking.

## Configuration

```json
{
  "model": {
    "embedding_dim": 128,
    "short_window": 10,
    "gru_layers": 1,
    "dropout": 0.1,
    "long_term_mode": "ewma",
    "ewma_halflife": 20.0,
    "max_history": 50,
    "min_context": 2
  },
  "train": {
    "alpha_train_mode": "fixed",
    "alpha_fixed": 0.5,
    "align_weight": 0.0,
    "n_negatives": 100,
    "negatives": "uniform",
    "lr": 0.001,
    "weight_decay": 0.0,
    "batch_size": 512,
    "epochs": 15,
    "device": "auto",
    "seed": 42
  },
  "eval": {
    "top_k": 10,
    "alpha": 0.5,
    "score_chunk": 1024,
    "seed": 42
  },
  "n_items": 14988
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| train_loss | train | 0 | 2.887575230775056 |  |  |
| train_examples | train | 0 | 137765.0 | ex |  |
| ndcg_at_k | val | 0 | 0.01939492285907377 |  |  |
| recall_at_k | val | 0 | 0.040372670807453416 |  |  |
| hit_rate_at_k | val | 0 | 0.040372670807453416 |  |  |
| mrr | val | 0 | 0.018219772139262045 |  |  |
| ndcg_at_k_seen | val | 0 | 0.01939492285907377 |  |  |
| hit_rate_at_k_seen | val | 0 | 0.040372670807453416 |  |  |
| n_scored | val | 0 | 644.0 | positions |  |
| n_seen_target | val | 0 | 644.0 | positions |  |
| ndcg_at_k | test | 0 | 0.017113906296127843 |  |  |
| recall_at_k | test | 0 | 0.03180778032036613 |  |  |
| hit_rate_at_k | test | 0 | 0.03180778032036613 |  |  |
| mrr | test | 0 | 0.017326849431287008 |  |  |
| ndcg_at_k_seen | test | 0 | 0.017113906296127843 |  |  |
| hit_rate_at_k_seen | test | 0 | 0.03180778032036613 |  |  |
| n_scored | test | 0 | 8740.0 | positions |  |
| n_seen_target | test | 0 | 8740.0 | positions |  |

## Findings

- train loss 2.8876, 137,765 examples
- VAL  NDCG@10=0.0194 HR@10=0.0404 MRR=0.0182 (seen NDCG=0.0194)
- TEST NDCG@10=0.0171 HR@10=0.0318 MRR=0.0173 (seen NDCG=0.0171)
- scored: val=644 test=8,740

## Limitations

- Subsample scale; absolute NDCG is low on sparse Amazon data.
- Unseen test targets (OOV) are unrankable and counted as misses in the full metric; the 'seen' metric conditions on rankable targets.

## Next steps

- Add divergence signal D_u(t)=dist(L,S) (Phase 4) over this backbone.
