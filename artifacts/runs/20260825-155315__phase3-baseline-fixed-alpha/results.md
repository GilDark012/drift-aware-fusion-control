# Run `20260825-155315__phase3-baseline-fixed-alpha`

- **Task**: phase3-baseline-fixed-alpha
- **Status**: completed
- **Started**: 2026-08-25T15:53:15.550804+00:00
- **Duration**: 102.18 s
- **Git commit**: 96e44d1239be63bc788d97a182da8693c788e8ed
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
  "n_items": 16026
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| train_loss | train | 0 | 2.8352275189665175 |  |  |
| train_examples | train | 0 | 148963.0 | ex |  |
| ndcg_at_k | val | 0 | 0.020052513432295825 |  |  |
| recall_at_k | val | 0 | 0.0408695652173913 |  |  |
| hit_rate_at_k | val | 0 | 0.0408695652173913 |  |  |
| mrr | val | 0 | 0.017601955149655932 |  |  |
| ndcg_at_k_seen | val | 0 | 0.020052513432295825 |  |  |
| hit_rate_at_k_seen | val | 0 | 0.0408695652173913 |  |  |
| n_scored | val | 0 | 2300.0 | positions |  |
| n_seen_target | val | 0 | 2300.0 | positions |  |
| ndcg_at_k | test | 0 | 0.015586924353047059 |  |  |
| recall_at_k | test | 0 | 0.0325029972026109 |  |  |
| hit_rate_at_k | test | 0 | 0.0325029972026109 |  |  |
| mrr | test | 0 | 0.014386669718015941 |  |  |
| ndcg_at_k_seen | test | 0 | 0.015586924353047059 |  |  |
| hit_rate_at_k_seen | test | 0 | 0.0325029972026109 |  |  |
| n_scored | test | 0 | 15014.0 | positions |  |
| n_seen_target | test | 0 | 15014.0 | positions |  |

## Findings

- train loss 2.8352, 148,963 examples
- VAL  NDCG@10=0.0201 HR@10=0.0409 MRR=0.0176 (seen NDCG=0.0201)
- TEST NDCG@10=0.0156 HR@10=0.0325 MRR=0.0144 (seen NDCG=0.0156)
- scored: val=2,300 test=15,014

## Limitations

- Subsample scale; absolute NDCG is low on sparse Amazon data.
- Unseen test targets (OOV) are unrankable and counted as misses in the full metric; the 'seen' metric conditions on rankable targets.

## Next steps

- Add divergence signal D_u(t)=dist(L,S) (Phase 4) over this backbone.
