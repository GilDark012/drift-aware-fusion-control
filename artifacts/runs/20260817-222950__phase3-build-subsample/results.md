# Run `20260817-222950__phase3-build-subsample`

- **Task**: phase3-build-subsample
- **Status**: completed
- **Started**: 2026-08-17T22:29:50.850360+00:00
- **Duration**: 2.89 s
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
  "processed_dir": "data\\processed",
  "out_dir": "data\\interim\\subsample"
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| subsample_rows | train | 0 | 39943.0 | rows |  |
| unseen_item_rate | train | 0 | 0.0 |  |  |
| subsample_rows | val | 0 | 11693.0 | rows |  |
| unseen_item_rate | val | 0 | 0.7763619259385958 |  |  |
| subsample_rows | test | 0 | 81724.0 | rows |  |
| unseen_item_rate | test | 0 | 0.8064460868288386 |  |  |
| selected_users | all | 0 | 3000.0 | users |  |
| item_vocab | all | 0 | 32158.0 | items |  |

## Findings

- selected users: 3000 (pool 7918)
- item vocab: 32158
- rows: {'train': 39943, 'val': 11693, 'test': 81724}
- unseen-item rate: {'train': 0.0, 'val': 0.7763619259385958, 'test': 0.8064460868288386}

## Next steps

- Train the fixed-α baseline on this subsample.

## Notes

```text
[2026-08-17T22:29:53.742716+00:00] cohort pool (>= 20 test interactions): 7918 users; selected 3000
```
