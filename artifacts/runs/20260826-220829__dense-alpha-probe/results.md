# Run `20260826-220829__dense-alpha-probe`

- **Task**: dense-alpha-probe
- **Status**: completed
- **Started**: 2026-08-26T22:08:29.901325+00:00
- **Duration**: 74.31 s
- **Git commit**: eef23efcbb2299047fdb33847543cc795cca5a2d
- **Seed**: 42
- **Dataset**: amazon-subsample-dense

## Objective

Test the precondition for the adaptation thesis: on a dense long-history cohort, does any α>0 (using long-term memory) beat α=0?

## Method

Build a dense subsample (train-history floor); train with random-α so both paths are optimised; sweep α at inference on val and test.

## Configuration

```json
{
  "dense": {
    "k_core": 5,
    "min_test": 5,
    "min_train": 50
  },
  "n_items": 11581,
  "alphas": [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
  ]
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| train_loss | train | 0 | 3.193748353731514 |  |  |
| ndcg_at_10 | test | 0 | 0.01328 |  |  |
| ndcg_at_10 | test | 20 | 0.0149 |  |  |
| ndcg_at_10 | test | 40 | 0.01559 |  |  |
| ndcg_at_10 | test | 60 | 0.01475 |  |  |
| ndcg_at_10 | test | 80 | 0.01304 |  |  |
| ndcg_at_10 | test | 100 | 0.00932 |  |  |

## Findings

- L HELPS: best alpha=0.4 beats alpha=0 by +17.4% -> adaptation has room.
- test NDCG@10: a=0 -> 0.01328; best a=0.4 -> 0.01559
- val sweep: a0.0=0.01828, a0.2=0.02000, a0.4=0.02143, a0.6=0.02084, a0.8=0.01893, a1.0=0.01208
- test sweep: a0.0=0.01328, a0.2=0.01490, a0.4=0.01559, a0.6=0.01475, a0.8=0.01304, a1.0=0.00932

## Limitations

- Dense cohort is small (long-history users are rare); one trained seed.
- EWMA-pool long-term encoder; a stronger L encoder is a separate lever.

## Next steps

- If L helps: re-run injection + adaptation (CUSUM) on this cohort.
- If not: report the negative result and narrow the thesis to detection.
