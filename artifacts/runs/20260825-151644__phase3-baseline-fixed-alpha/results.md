# Run `20260825-151644__phase3-baseline-fixed-alpha`

- **Task**: phase3-baseline-fixed-alpha
- **Status**: completed
- **Started**: 2026-08-25T15:16:44.114934+00:00
- **Duration**: 432.95 s
- **Git commit**: 1496f665dc05bc0a480694f50ccbf29b1e4e51d7
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
    "negatives": "popularity",
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
  "n_items": 220736
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| train_loss | train | 0 | 0.913915396157433 |  |  |
| train_examples | train | 0 | 434898.0 | ex |  |
| ndcg_at_k | val | 0 | 0.00047626785475903986 |  |  |
| recall_at_k | val | 0 | 0.0009823182711198428 |  |  |
| hit_rate_at_k | val | 0 | 0.0009823182711198428 |  |  |
| mrr | val | 0 | 0.0006003478158187071 |  |  |
| ndcg_at_k_seen | val | 0 | 0.0009162081150303605 |  |  |
| hit_rate_at_k_seen | val | 0 | 0.0018897096718776842 |  |  |
| n_scored | val | 0 | 11198.0 | positions |  |
| n_seen_target | val | 0 | 5821.0 | positions |  |
| ndcg_at_k | test | 0 | 0.0004175273712934204 |  |  |
| recall_at_k | test | 0 | 0.0008718395815170009 |  |  |
| hit_rate_at_k | test | 0 | 0.0008718395815170009 |  |  |
| mrr | test | 0 | 0.0004916118717064835 |  |  |
| ndcg_at_k_seen | test | 0 | 0.0009004370841028397 |  |  |
| hit_rate_at_k_seen | test | 0 | 0.0018802041364491002 |  |  |
| n_scored | test | 0 | 80290.0 | positions |  |
| n_seen_target | test | 0 | 37230.0 | positions |  |

## Findings

- train loss 0.9139, 434,898 examples
- VAL  NDCG@10=0.0005 HR@10=0.0010 MRR=0.0006 (seen NDCG=0.0009)
- TEST NDCG@10=0.0004 HR@10=0.0009 MRR=0.0005 (seen NDCG=0.0009)
- scored: val=11,198 test=80,290

## Limitations

- Subsample scale; absolute NDCG is low on sparse Amazon data.
- Unseen test targets (OOV) are unrankable and counted as misses in the full metric; the 'seen' metric conditions on rankable targets.

## Next steps

- Add divergence signal D_u(t)=dist(L,S) (Phase 4) over this backbone.
