# Run `20260826-090217__p1-2-seed-robustness`

- **Task**: p1-2-seed-robustness
- **Status**: completed
- **Started**: 2026-08-26T09:02:17.726949+00:00
- **Duration**: 43.22 s
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
  "seeds": 20,
  "n_users": 300,
  "operating_points": [
    1.75,
    2.0
  ]
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| detection_rate_mean | all | 175 | 0.644 |  | z=1.75 |
| detection_rate_ci95 | all | 175 | 0.0139 |  | z=1.75 |
| detection_rate_mean | all | 200 | 0.2789 |  | z=2.0 |
| detection_rate_ci95 | all | 200 | 0.0177 |  | z=2.0 |

## Findings

- z=1.75: detection 0.64 ±0.01, false-alarm 0.024 ±0.000 (n=20 seeds)
- z=2.0: detection 0.28 ±0.02, false-alarm 0.000 ±0.000 (n=20 seeds)

## Limitations

- Randomness axis is the injection sampling only; backbone-initialisation variance (retraining the GRU) is a separate, heavier axis, not run here.
- Sudden scenario; gradual/recurring can be added with --scenario later.

## Next steps

- Confront window-B pre-drift stability cost (P1.4).
