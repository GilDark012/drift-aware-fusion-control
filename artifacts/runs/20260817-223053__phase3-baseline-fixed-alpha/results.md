# Run `20260817-223053__phase3-baseline-fixed-alpha`

- **Task**: phase3-baseline-fixed-alpha
- **Status**: completed
- **Started**: 2026-08-17T22:30:53.055823+00:00
- **Duration**: 9.86 s
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
    "epochs": 1,
    "device": "auto",
    "seed": 42
  },
  "eval": {
    "top_k": 10,
    "alpha": 0.5,
    "score_chunk": 1024,
    "seed": 42
  },
  "n_items": 32158
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| train_loss | train | 0 | 4.620534775962292 |  |  |
| train_examples | train | 0 | 35872.0 | ex |  |
| ndcg_at_k | val | 0 | 0.003263015765696764 |  |  |
| recall_at_k | val | 0 | 0.007233434356749058 |  |  |
| hit_rate_at_k | val | 0 | 0.007233434356749058 |  |  |
| mrr | val | 0 | 0.002530797617509961 |  |  |
| ndcg_at_k_seen | val | 0 | 0.01446526105311402 |  |  |
| hit_rate_at_k_seen | val | 0 | 0.032066508313539195 |  |  |
| n_scored | val | 0 | 11198.0 | positions |  |
| n_seen_target | val | 0 | 2526.0 | positions |  |
| ndcg_at_k | test | 0 | 0.003014845075085759 |  |  |
| recall_at_k | test | 0 | 0.006053057499229908 |  |  |
| hit_rate_at_k | test | 0 | 0.006053057499229908 |  |  |
| mrr | test | 0 | 0.002509509911760688 |  |  |
| ndcg_at_k_seen | test | 0 | 0.01551579515197844 |  |  |
| hit_rate_at_k_seen | test | 0 | 0.031151849240433304 |  |  |
| n_scored | test | 0 | 80290.0 | positions |  |
| n_seen_target | test | 0 | 15601.0 | positions |  |

## Findings

- train loss: 4.6205 on 35,872 examples
- VAL  NDCG@10=0.0033 HR@10=0.0072 MRR=0.0025 (seen NDCG=0.0145)
- TEST NDCG@10=0.0030 HR@10=0.0061 MRR=0.0025 (seen NDCG=0.0155)
- scored positions: val=11,198 test=80,290

## Limitations

- Subsample scale; absolute NDCG is low on sparse Amazon data.
- Unseen test targets (OOV) are unrankable and counted as misses in the full metric; the 'seen' metric conditions on rankable targets.

## Next steps

- Add divergence signal D_u(t)=dist(L,S) (Phase 4) over this backbone.
