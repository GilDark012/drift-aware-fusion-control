# Run `20260819-062856__phase3-baseline-fixed-alpha`

- **Task**: phase3-baseline-fixed-alpha
- **Status**: completed
- **Started**: 2026-08-19T06:28:56.376582+00:00
- **Duration**: 371.57 s
- **Git commit**: 652dfa377d6de1a43f19a24d48af46cb180b0be4
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
    "alpha_train_mode": "random",
    "alpha_fixed": 0.5,
    "align_weight": 0.0,
    "n_negatives": 100,
    "lr": 0.001,
    "weight_decay": 0.0,
    "batch_size": 512,
    "epochs": 10,
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
| train_loss | train | 0 | 1.7450036660362693 |  |  |
| train_examples | train | 0 | 434898.0 | ex |  |
| ndcg_at_k | val | 0 | 0.00279426285556422 |  |  |
| recall_at_k | val | 0 | 0.005893909626719057 |  |  |
| hit_rate_at_k | val | 0 | 0.005893909626719057 |  |  |
| mrr | val | 0 | 0.0023031531864452395 |  |  |
| ndcg_at_k_seen | val | 0 | 0.005375391763718973 |  |  |
| hit_rate_at_k_seen | val | 0 | 0.011338258031266105 |  |  |
| n_scored | val | 0 | 11198.0 | positions |  |
| n_seen_target | val | 0 | 5821.0 | positions |  |
| ndcg_at_k | test | 0 | 0.002243147971139916 |  |  |
| recall_at_k | test | 0 | 0.004471291568065762 |  |  |
| hit_rate_at_k | test | 0 | 0.004471291568065762 |  |  |
| mrr | test | 0 | 0.0020093101028661113 |  |  |
| ndcg_at_k_seen | test | 0 | 0.004837559779823364 |  |  |
| hit_rate_at_k_seen | test | 0 | 0.009642761214074671 |  |  |
| n_scored | test | 0 | 80290.0 | positions |  |
| n_seen_target | test | 0 | 37230.0 | positions |  |

## Findings

- train loss 1.7450, 434,898 examples
- VAL  NDCG@10=0.0028 HR@10=0.0059 MRR=0.0023 (seen NDCG=0.0054)
- TEST NDCG@10=0.0022 HR@10=0.0045 MRR=0.0020 (seen NDCG=0.0048)
- scored: val=11,198 test=80,290

## Limitations

- Subsample scale; absolute NDCG is low on sparse Amazon data.
- Unseen test targets (OOV) are unrankable and counted as misses in the full metric; the 'seen' metric conditions on rankable targets.

## Next steps

- Add divergence signal D_u(t)=dist(L,S) (Phase 4) over this backbone.
