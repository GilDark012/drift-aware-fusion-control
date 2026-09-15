# Run `20260826-140600__dense-adaptation-seeds`

- **Task**: dense-adaptation-seeds
- **Status**: completed
- **Started**: 2026-08-26T14:06:00.311554+00:00
- **Duration**: 71.43 s
- **Git commit**: 18bb5cd9262f9e8ab6f9bdff72339f6d141b9f04
- **Seed**: 42
- **Dataset**: amazon-subsample-dense

## Objective

Confirm the CUSUM adaptive-fusion win over static-0.8 and window-B is robust across injection seeds, not a single-seed artefact.

## Method

CUSUM enter=0.3 (calibrated once on control); 12 coherent-drift seeds; paired proposed-baseline diffs + CIs.

## Configuration

```json
{
  "seeds": 12,
  "mechanism": "coherent"
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| proposed_minus_static_mean | all | 0 | 0.00030880260306061946 |  |  |
| proposed_minus_static_ci95 | all | 0 | 0.0004501511604072786 |  |  |
| win_rate | all | 0 | 0.6666666666666666 |  |  |

## Findings

- NOT ROBUST: proposed - static-0.8 = +0.00031 +/- 0.00045 (95% CI); win rate 67% over 12 seeds.
- proposed - window-B = +0.00078 +/- 0.00019
- static-0.8: 0.01076 +/- 0.00067
- window-B: 0.01029 +/- 0.00052
- proposed-CUSUM: 0.01107 +/- 0.00058

## Limitations

- Dense cohort; one trained backbone seed; sudden scenario.

## Next steps

- Gradual/recurring scenarios; backbone-seed variance; full scale.
