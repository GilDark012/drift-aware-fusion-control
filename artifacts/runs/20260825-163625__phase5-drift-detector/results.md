# Run `20260825-163625__phase5-drift-detector`

- **Task**: phase5-drift-detector
- **Status**: completed
- **Started**: 2026-08-25T16:36:25.510076+00:00
- **Duration**: 4.01 s
- **Git commit**: 7e600cc39b00d3921248d4443656d9381154576b
- **Seed**: 42
- **Dataset**: amazon-subsample

## Objective

Turn a persistent divergence rise into a confirmed-shift detector with false-alarm control, tuned on validation.

## Method

State machine STABLE→EMERGING→CONFIRMED with TEMPORARY suppression; persistence sweep (H5) and adaptive-vs-global by volatility (H6).

## Configuration

```json
{
  "signal": "short_term_drift",
  "persistences": [
    1,
    3,
    5,
    8,
    12
  ]
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| users_tuned | val | 0 | 86.0 | users |  |
| confirmed_per_1k | val | 1 | 0.49875311720698257 |  | persistence=1 |
| confirmed_per_1k | val | 3 | 0.0 |  | persistence=3 |
| confirmed_per_1k | val | 5 | 0.0 |  | persistence=5 |
| confirmed_per_1k | val | 8 | 0.0 |  | persistence=8 |
| confirmed_per_1k | val | 12 | 0.0 |  | persistence=12 |
| chosen_confirmed_per_1k | val | 0 | 0.0 |  |  |
| chosen_temporary_per_1k | val | 0 | 4.488778054862843 |  |  |
| chosen_users_with_confirm | val | 0 | 0.0 |  |  |

## Findings

- persistence 1→5 cuts confirmed alarms 0.5→0.0 per 1k steps (H5)
- adaptive confirmed/1k spread across strata [0.0,0.0] vs global [0.0,5.0] (H6)
- chosen (adaptive, enter=2.5, persistence=5): 0.0 confirmed/1k, 0.0% of users; global enter=0.509
- detector overlays saved for 0 clearest users

## Limitations

- No injected ground truth yet; alarm rate is a false-alarm proxy on natural validation data (Phase 7 adds known onsets).

## Next steps

- Drive α(t) from detector state — adaptive fusion (Phase 6).

## Figures

![persistence-sweep.png](figures/persistence-sweep.png)
![strategy-by-stratum.png](figures/strategy-by-stratum.png)
