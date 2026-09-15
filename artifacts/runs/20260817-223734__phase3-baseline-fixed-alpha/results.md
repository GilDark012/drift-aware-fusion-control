# Run `20260817-223734__phase3-baseline-fixed-alpha`

- **Task**: phase3-baseline-fixed-alpha
- **Status**: completed
- **Started**: 2026-08-17T22:37:34.187006+00:00
- **Duration**: 168.55 s
- **Git commit**: f56e064f2f225097174346c4b1e36bd7f71d37a6
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
    "embedding_dim": 64,
    "short_window": 10,
    "gru_layers": 1,
    "dropout": 0.1,
    "long_term_mode": "ewma",
    "ewma_halflife": 20.0,
    "max_history": 50,
    "min_context": 2
  },
  "train": {
    "alpha_train_mode": "random",
    "alpha_fixed": 0.5,
    "n_negatives": 100,
    "lr": 0.001,
    "weight_decay": 0.0,
    "batch_size": 512,
    "epochs": 5,
    "device": "auto",
    "seed": 42
  },
  "eval": {
    "top_k": 10,
    "alpha": 0.5,
    "score_chunk": 1024,
    "seed": 42
  },
  "n_items": 220736
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| train_loss | train | 0 | 3.4974044132232667 |  |  |
| train_examples | train | 0 | 434898.0 | ex |  |
| ndcg_at_k | val | 0 | 0.00379912368953228 |  |  |
| recall_at_k | val | 0 | 0.007858546450734138 |  |  |
| hit_rate_at_k | val | 0 | 0.007858546450734138 |  |  |
| mrr | val | 0 | 0.003221374936401844 |  |  |
| ndcg_at_k_seen | val | 0 | 0.0073084671500212054 |  |  |
| hit_rate_at_k_seen | val | 0 | 0.015117677375021473 |  |  |
| n_scored | val | 0 | 11198.0 | positions |  |
| n_seen_target | val | 0 | 5821.0 | positions |  |
| ndcg_at_k | test | 0 | 0.003425751579925418 |  |  |
| recall_at_k | test | 0 | 0.006962261628359556 |  |  |
| hit_rate_at_k | test | 0 | 0.006962261628359556 |  |  |
| mrr | test | 0 | 0.002946379128843546 |  |  |
| ndcg_at_k_seen | test | 0 | 0.007387955650475088 |  |  |
| hit_rate_at_k_seen | test | 0 | 0.015014773032500671 |  |  |
| n_scored | test | 0 | 80290.0 | positions |  |
| n_seen_target | test | 0 | 37230.0 | positions |  |

## Findings

- train loss: 3.4974 on 434,898 examples
- VAL  NDCG@10=0.0038 HR@10=0.0079 MRR=0.0032 (seen NDCG=0.0073)
- TEST NDCG@10=0.0034 HR@10=0.0070 MRR=0.0029 (seen NDCG=0.0074)
- scored positions: val=11,198 test=80,290

## Limitations

- Subsample scale; absolute NDCG is low on sparse Amazon data.
- Unseen test targets (OOV) are unrankable and counted as misses in the full metric; the 'seen' metric conditions on rankable targets.

## Next steps

- Add divergence signal D_u(t)=dist(L,S) (Phase 4) over this backbone.
