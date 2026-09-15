# Run `20260826-221355__dense-adaptation-seeds`

- **Task**: dense-adaptation-seeds
- **Status**: completed
- **Started**: 2026-08-26T22:13:55.867879+00:00
- **Duration**: 566.30 s
- **Git commit**: eef23efcbb2299047fdb33847543cc795cca5a2d
- **Seed**: 42
- **Dataset**: amazon-subsample-dense

## Objective

Confirm the CUSUM adaptive-fusion win generalises across drift scenarios (sudden/gradual/recurring), robust across injection seeds.

## Method

CUSUM enter=0.5 (calibrated once on control); coherent drift; paired proposed-baseline differences with 95% CIs, per scenario.

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
| proposed_minus_window_robust | all | 0 | 0.0 |  | sudden |
| proposed_minus_static_robust | all | 0 | 1.0 |  | sudden |
| proposed_minus_window_robust | all | 0 | 1.0 |  | gradual |
| proposed_minus_static_robust | all | 0 | 1.0 |  | gradual |
| proposed_minus_window_robust | all | 0 | 0.0 |  | recurring |
| proposed_minus_static_robust | all | 0 | 1.0 |  | recurring |

## Findings

- [sudden] vs window-B -0.00007+/-0.00025 (ns); vs static-0.8 +0.00672+/-0.00058 (ROBUST); win 43%
- [gradual] vs window-B +0.00111+/-0.00026 (ROBUST); vs static-0.8 +0.00185+/-0.00036 (ROBUST); win 87%
- [recurring] vs window-B -0.00019+/-0.00022 (ns); vs static-0.8 +0.00385+/-0.00042 (ROBUST); win 40%

## Limitations

- Dense cohort; one trained backbone seed.

## Next steps

- Backbone-seed variance; stronger long-term encoder; full scale.
