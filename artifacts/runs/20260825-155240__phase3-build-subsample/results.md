# Run `20260825-155240__phase3-build-subsample`

- **Task**: phase3-build-subsample
- **Status**: completed
- **Started**: 2026-08-25T15:52:40.916248+00:00
- **Duration**: 3.15 s
- **Git commit**: 96e44d1239be63bc788d97a182da8693c788e8ed
- **Seed**: 42
- **Dataset**: amazon-processed

## Objective

Provide a small, faithful subsample for pipeline iteration.

## Method

Select test-active cohort users (stratified by activity), keep full cross-split history, re-index densely with train-only item vocab.

## Configuration

```json
{
  "seed": 42,
  "target_users": 3000,
  "min_test_interactions": 20,
  "activity_strata": 3,
  "extra_train_users": 20000,
  "k_core": 5,
  "processed_dir": "data\\processed",
  "out_dir": "data\\interim\\subsample"
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| subsample_rows | train | 0 | 189942.0 | rows |  |
| unseen_item_rate | train | 0 | 0.0 |  |  |
| subsample_rows | val | 0 | 2690.0 | rows |  |
| unseen_item_rate | val | 0 | 0.0 |  |  |
| subsample_rows | test | 0 | 17237.0 | rows |  |
| unseen_item_rate | test | 0 | 0.0 |  |  |
| selected_users | all | 0 | 3000.0 | users |  |
| train_users_total | all | 0 | 23000.0 | users |  |
| item_vocab | all | 0 | 16026.0 | items |  |

## Findings

- eval cohort: 3000 users (pool 7918)
- train users incl. extra: 23000
- item vocab: 16026
- rows: {'train': 189942, 'val': 2690, 'test': 17237}
- unseen-item rate: {'train': 0.0, 'val': 0.0, 'test': 0.0}

## Next steps

- Train the fixed-α baseline on this subsample.

## Notes

```text
[2026-08-25T15:52:44.066879+00:00] cohort pool (>= 20 test interactions): 7918 users; cohort 3000; train users 23000
```
