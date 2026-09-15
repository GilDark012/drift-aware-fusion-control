# Run `20260819-054933__phase7-inject-drift`

- **Task**: phase7-inject-drift
- **Status**: completed
- **Started**: 2026-08-19T05:49:33.577201+00:00
- **Duration**: 4.90 s
- **Git commit**: 652dfa377d6de1a43f19a24d48af46cb180b0be4
- **Seed**: 42
- **Dataset**: amazon-subsample

## Objective

Create reproducible preference-sequence shifts with known onsets and confirm they are detectable (replacing the broken drifted parquet).

## Method

Replace post-onset items with a different category's popular seen items (index-consistent); sudden/gradual/recurring; stream a sample and measure detection latency vs the known onset.

## Configuration

```json
{
  "n_users": 300,
  "types": [
    "sudden",
    "gradual",
    "recurring"
  ]
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| injected_users | all | 0 | 300.0 | users | sudden |
| mean_replaced | all | 0 | 24.236666666666668 |  | sudden |
| detection_rate | all | 0 | 0.0 |  | sudden |
| premature_rate | all | 0 | 0.0 |  | sudden |
| injected_users | all | 0 | 300.0 | users | gradual |
| mean_replaced | all | 0 | 18.756666666666668 |  | gradual |
| detection_rate | all | 0 | 0.0 |  | gradual |
| premature_rate | all | 0 | 0.0 |  | gradual |
| injected_users | all | 0 | 300.0 | users | recurring |
| mean_replaced | all | 0 | 14.47 |  | recurring |
| detection_rate | all | 0 | 0.0 |  | recurring |
| premature_rate | all | 0 | 0.0 |  | recurring |

## Findings

- sudden: detection_rate=0.00, median_latency=n/a interactions, premature=0.00
- gradual: detection_rate=0.00, median_latency=n/a interactions, premature=0.00
- recurring: detection_rate=0.00, median_latency=n/a interactions, premature=0.00
- injected splits + onset ground truth saved under data/interim/subsample/

## Limitations

- Injection targets a coherent category profile; real shifts are messier.
- Sanity uses static α; full latency/recovery comparison is Phase 8.

## Next steps

- Phase 8: control vs injected, all policies, detection/adaptation/recovery.
