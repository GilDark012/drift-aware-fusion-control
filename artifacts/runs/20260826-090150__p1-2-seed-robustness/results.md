# Run `20260826-090150__p1-2-seed-robustness`

- **Task**: p1-2-seed-robustness
- **Status**: completed
- **Started**: 2026-08-26T09:01:50.472165+00:00
- **Duration**: 2.60 s
- **Git commit**: 5c0e5199df8bde5eadd3be8e477faac6bd4b39ee
- **Seed**: 42
- **Dataset**: amazon-subsample

## Objective

Replace the single-seed detection headline with a Monte-Carlo mean and 95% confidence interval over controlled-drift injection seeds.

## Method

Re-inject the sudden scenario under N seeds on the frozen backbone; at each detector operating point compute detection rate, false-alarm rate and median latency per seed, then aggregate to mean ± 95% CI.

## Configuration

```json
{
  "seeds": 2,
  "n_users": 60,
  "operating_points": [
    1.75,
    2.0
  ]
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| detection_rate_mean | all | 175 | 0.675 |  | z=1.75 |
| detection_rate_ci95 | all | 175 | 0.049 |  | z=1.75 |
| detection_rate_mean | all | 200 | 0.3333 |  | z=2.0 |
| detection_rate_ci95 | all | 200 | 0.0327 |  | z=2.0 |

## Findings

- z=1.75: detection 0.68 ±0.05, false-alarm 0.017 ±0.033 (n=2 seeds)
- z=2.0: detection 0.33 ±0.03, false-alarm 0.000 ±0.000 (n=2 seeds)

## Limitations

- Randomness axis is the injection sampling only; backbone-initialisation variance (retraining the GRU) is a separate, heavier axis, not run here.
- Sudden scenario; gradual/recurring can be added with --scenario later.

## Next steps

- Confront window-B pre-drift stability cost (P1.4).
