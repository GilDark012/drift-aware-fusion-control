# Run `20260826-154954__dense-adaptation-seeds`

- **Task**: dense-adaptation-seeds
- **Status**: completed
- **Started**: 2026-08-26T15:49:54.553348+00:00
- **Duration**: 570.88 s
- **Git commit**: a6eb1f47b44b0c1d648ac054600b55c75d83ca72
- **Seed**: 42
- **Dataset**: amazon-subsample-dense

## Objective

Confirm the CUSUM adaptive-fusion win generalises across drift scenarios (sudden/gradual/recurring), robust across injection seeds.

## Method

CUSUM enter=0.3 (calibrated once on control); coherent drift; paired proposed-baseline differences with 95% CIs, per scenario.

## Configuration

```json
{
  "seeds": 30,
  "mechanism": "coherent",
  "scenarios": [
    "sudden",
    "gradual",
    "recurring"
  ]
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| proposed_minus_window_robust | all | 0 | 1.0 |  | sudden |
| proposed_minus_static_robust | all | 0 | 0.0 |  | sudden |
| proposed_minus_window_robust | all | 0 | 1.0 |  | gradual |
| proposed_minus_static_robust | all | 0 | 0.0 |  | gradual |
| proposed_minus_window_robust | all | 0 | 1.0 |  | recurring |
| proposed_minus_static_robust | all | 0 | 0.0 |  | recurring |

## Findings

- [sudden] vs window-B +0.00078+/-0.00010 (ROBUST); vs static-0.8 +0.00016+/-0.00030 (ns); win 63%
- [gradual] vs window-B +0.00165+/-0.00019 (ROBUST); vs static-0.8 -0.00012+/-0.00027 (ns); win 60%
- [recurring] vs window-B +0.00069+/-0.00008 (ROBUST); vs static-0.8 +0.00008+/-0.00024 (ns); win 60%

## Limitations

- Dense cohort; one trained backbone seed.

## Next steps

- Backbone-seed variance; stronger long-term encoder; full scale.
