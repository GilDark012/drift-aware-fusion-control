# Run `20260826-140811__dense-adaptation-seeds`

- **Task**: dense-adaptation-seeds
- **Status**: completed
- **Started**: 2026-08-26T14:08:11.427075+00:00
- **Duration**: 185.94 s
- **Git commit**: 18bb5cd9262f9e8ab6f9bdff72339f6d141b9f04
- **Seed**: 42
- **Dataset**: amazon-subsample-dense

## Objective

Confirm the CUSUM adaptive-fusion win over static-0.8 and window-B is robust across injection seeds, not a single-seed artefact.

## Method

CUSUM enter=0.3 (calibrated once on control); 30 coherent-drift seeds; paired proposed-baseline diffs + CIs.

## Configuration

```json
{
  "seeds": 30,
  "mechanism": "coherent"
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| proposed_minus_static_mean | all | 0 | 0.0001616068347270505 |  |  |
| proposed_minus_static_ci95 | all | 0 | 0.00029721585653985394 |  |  |
| win_rate | all | 0 | 0.6333333333333333 |  |  |

## Findings

- NOT ROBUST: proposed - static-0.8 = +0.00016 +/- 0.00030 (95% CI); win rate 63% over 30 seeds.
- proposed - window-B = +0.00078 +/- 0.00010
- static-0.8: 0.01058 +/- 0.00038
- window-B: 0.00996 +/- 0.00034
- proposed-CUSUM: 0.01074 +/- 0.00033

## Limitations

- Dense cohort; one trained backbone seed; sudden scenario.

## Next steps

- Gradual/recurring scenarios; backbone-seed variance; full scale.
