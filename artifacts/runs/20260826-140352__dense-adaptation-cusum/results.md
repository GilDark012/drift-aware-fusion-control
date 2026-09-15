# Run `20260826-140352__dense-adaptation-cusum`

- **Task**: dense-adaptation-cusum
- **Status**: completed
- **Started**: 2026-08-26T14:03:52.419294+00:00
- **Duration**: 5.78 s
- **Git commit**: 18bb5cd9262f9e8ab6f9bdff72339f6d141b9f04
- **Seed**: 42
- **Dataset**: amazon-subsample-dense

## Objective

Test whether CUSUM-driven adaptive fusion beats static-α=0.8 and window-B on the dense cohort where long-term memory helps.

## Method

Off-the-shelf CUSUM (enter=0.5, calibrated to ≤0.1 control false-alarm) drives α (STABLE 0.8 → CONFIRMED 0.2, gradual); compare pre/post/overall NDCG@10 against static-0.8 and window-B on a sudden shift.

## Configuration

```json
{
  "top_k": 10,
  "stable_alpha": 0.8,
  "mechanism": "coherent",
  "n_injected": 102
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| cusum_enter | all | 0 | 0.5 |  | calibrated |
| recovery_latency | all | 0 | 8.0 |  | static-0.8 |
| recovery_latency | all | 0 | 8.0 |  | window-B |
| recovery_latency | all | 0 | 8.0 |  | proposed-CUSUM |

## Findings

- ADAPTATION WINS: proposed 0.01214 > static-0.8 0.01164 and > window-B 0.01117 on overall NDCG@10.
- static-0.8: pre=0.02072 post=0.00630 overall=0.01164
- window-B: pre=0.01875 post=0.00671 overall=0.01117
- proposed-CUSUM: pre=0.02144 post=0.00669 overall=0.01214

## Limitations

- Dense cohort is small; single seed; sudden scenario only.
- Short test streams limit post-onset resolution; overall pooled NDCG is the robust headline.

## Next steps

- Seed/CI repetition; gradual & recurring scenarios; full-scale backbone.

## Figures

![dense-recovery-curves.png](figures/dense-recovery-curves.png)
