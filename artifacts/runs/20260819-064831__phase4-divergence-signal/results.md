# Run `20260819-064831__phase4-divergence-signal`

- **Task**: phase4-divergence-signal
- **Status**: completed
- **Started**: 2026-08-19T06:48:31.970548+00:00
- **Duration**: 7.18 s
- **Git commit**: 652dfa377d6de1a43f19a24d48af46cb180b0be4
- **Seed**: 42
- **Dataset**: amazon-subsample

## Objective

Compute D_u(t)=dist(L,S) and validate it responds to shifts.

## Method

short_term_drift drift signal over the trained backbone for 600 cohort users; compare divergence at category changes vs stable positions; causal rolling summaries.

## Configuration

```json
{
  "signal": "short_term_drift",
  "max_users": 600,
  "checkpoint": "artifacts/models/baseline-subsample"
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| users_analyzed | test | 0 | 600.0 | users |  |
| divergence_mean | test | 0 | 0.5718359036195443 |  |  |
| divergence_std | test | 0 | 0.2614859682841883 |  |  |
| divergence_mean_category_changed | test | 0 | 0.614020846794843 |  |  |
| divergence_mean_category_stable | test | 0 | 0.5568773660898458 |  |  |
| positions_total | test | 0 | 20197.0 | positions |  |
| category_change_rate | test | 0 | 0.2617715502302322 |  |  |
| category_change_divergence_lift | test | 0 | 1.1026141197050874 |  |  |

## Findings

- divergence mean=0.5718 std=0.2615 over 20,197 positions
- at category change=0.6140 vs stable=0.5569 (lift x1.10)
- category change is an interpretable indicator, not the definition

## Limitations

- Subsample backbone; absolute divergence scale is model-specific.
- Category-change is a coarse proxy for true preference shift.

## Next steps

- Turn persistent rises in D_u(t) into a detector with state model and false-alarm control (Phase 5).

## Figures

![population-category-effect.png](figures/population-category-effect.png)
![divergence-distribution.png](figures/divergence-distribution.png)
![trajectory-user-948.png](figures/trajectory-user-948.png)
![trajectory-user-268.png](figures/trajectory-user-268.png)
![trajectory-user-8678.png](figures/trajectory-user-8678.png)
![trajectory-user-9962.png](figures/trajectory-user-9962.png)
![trajectory-user-3553.png](figures/trajectory-user-3553.png)
![trajectory-user-615.png](figures/trajectory-user-615.png)
