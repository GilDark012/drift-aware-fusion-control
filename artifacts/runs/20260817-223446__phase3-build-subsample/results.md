# Run `20260817-223446__phase3-build-subsample`

- **Task**: phase3-build-subsample
- **Status**: completed
- **Started**: 2026-08-17T22:34:46.454256+00:00
- **Duration**: 4.32 s
- **Git commit**: f56e064f2f225097174346c4b1e36bd7f71d37a6
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
  "processed_dir": "data\\processed",
  "out_dir": "data\\interim\\subsample"
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| subsample_rows | train | 0 | 478789.0 | rows |  |
| unseen_item_rate | train | 0 | 0.0 |  |  |
| subsample_rows | val | 0 | 11693.0 | rows |  |
| unseen_item_rate | val | 0 | 0.4826819464636962 |  |  |
| subsample_rows | test | 0 | 81724.0 | rows |  |
| unseen_item_rate | test | 0 | 0.5377122999363712 |  |  |
| selected_users | all | 0 | 3000.0 | users |  |
| train_users_total | all | 0 | 23000.0 | users |  |
| item_vocab | all | 0 | 220736.0 | items |  |

## Findings

- eval cohort: 3000 users (pool 7918)
- train users incl. extra: 23000
- item vocab: 220736
- rows: {'train': 478789, 'val': 11693, 'test': 81724}
- unseen-item rate: {'train': 0.0, 'val': 0.4826819464636962, 'test': 0.5377122999363712}

## Next steps

- Train the fixed-α baseline on this subsample.

## Notes

```text
[2026-08-17T22:34:50.776422+00:00] cohort pool (>= 20 test interactions): 7918 users; eval cohort 3000; train users incl. extra 23000
```
