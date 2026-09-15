# Run `20260825-163228__phase3-build-subsample`

- **Task**: phase3-build-subsample
- **Status**: completed
- **Started**: 2026-08-25T16:32:28.794224+00:00
- **Duration**: 3.26 s
- **Git commit**: 7e600cc39b00d3921248d4443656d9381154576b
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
| subsample_rows | train | 0 | 176282.0 | rows |  |
| unseen_item_rate | train | 0 | 0.0 |  |  |
| subsample_rows | val | 0 | 741.0 | rows |  |
| unseen_item_rate | val | 0 | 0.0 |  |  |
| subsample_rows | test | 0 | 9361.0 | rows |  |
| unseen_item_rate | test | 0 | 0.0 |  |  |
| selected_users | all | 0 | 719.0 | users |  |
| train_users_total | all | 0 | 20719.0 | users |  |
| item_vocab | all | 0 | 14988.0 | items |  |

## Findings

- eval cohort: 719 users (pool 719)
- train users incl. extra: 20719
- item vocab: 14988
- rows: {'train': 176282, 'val': 741, 'test': 9361}
- unseen-item rate: {'train': 0.0, 'val': 0.0, 'test': 0.0}

## Next steps

- Train the fixed-α baseline on this subsample.

## Notes

```text
[2026-08-25T16:32:32.055946+00:00] cohort pool (>= 20 test interactions): 719 users; cohort 719; train users 20719
```
