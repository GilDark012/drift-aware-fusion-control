# Run `20260826-123828__dense-adaptation-cusum`

- **Task**: dense-adaptation-cusum
- **Status**: completed
- **Started**: 2026-08-26T12:38:28.521822+00:00
- **Duration**: 9.61 s
- **Git commit**: fd3f673dd006c70ae9081d9f648132869248dbb2
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
  "n_injected": 102
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| cusum_enter | all | 0 | 0.5 |  | calibrated |
| recovery_latency | all | 0 | 7.0 |  | static-0.8 |
| recovery_latency | all | 0 | 7.0 |  | window-B |
| recovery_latency | all | 0 | 7.0 |  | proposed-CUSUM |

## Findings

- NO WIN: proposed 0.01667 vs static-0.8 0.01788, window-B 0.01427 — adaptation does not dominate.
- static-0.8: pre=0.02072 post=0.01621 overall=0.01788
- window-B: pre=0.01875 post=0.01163 overall=0.01427
- proposed-CUSUM: pre=0.02144 post=0.01387 overall=0.01667

## Limitations

- Dense cohort is small; single seed; sudden scenario only.
- Short test streams limit post-onset resolution; overall pooled NDCG is the robust headline.

## Next steps

- Seed/CI repetition; gradual & recurring scenarios; full-scale backbone.

## Figures

![dense-recovery-curves.png](figures/dense-recovery-curves.png)
