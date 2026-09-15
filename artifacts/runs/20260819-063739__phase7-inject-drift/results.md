# Run `20260819-063739__phase7-inject-drift`

- **Task**: phase7-inject-drift
- **Status**: completed
- **Started**: 2026-08-19T06:37:39.375251+00:00
- **Duration**: 17.71 s
- **Git commit**: 652dfa377d6de1a43f19a24d48af46cb180b0be4
- **Seed**: 42
- **Dataset**: amazon-subsample

## Objective

Create reproducible preference-sequence shifts with known onsets and confirm they are detectable, replacing the broken drifted parquet.

## Method

Switch post-onset items to a cluster whose short-term rep is most opposed to the user's established profile; sudden/gradual/recurring; stream injected vs control and measure detection latency and false alarms.

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
| detection_rate | all | 0 | 0.035 |  | sudden |
| false_alarm_rate | all | 0 | 0.0 |  | sudden |
| median_detection_latency | all | 0 | 10.0 | interactions | sudden |
| injected_users | all | 0 | 300.0 | users | gradual |
| detection_rate | all | 0 | 0.05 |  | gradual |
| false_alarm_rate | all | 0 | 0.0 |  | gradual |
| median_detection_latency | all | 0 | 13.0 | interactions | gradual |
| injected_users | all | 0 | 300.0 | users | recurring |
| detection_rate | all | 0 | 0.05 |  | recurring |
| false_alarm_rate | all | 0 | 0.0 |  | recurring |
| median_detection_latency | all | 0 | 10.0 | interactions | recurring |

## Findings

- sudden: detection_rate=0.04, median_latency=10, false_alarm=0/200
- gradual: detection_rate=0.05, median_latency=13, false_alarm=0/200
- recurring: detection_rate=0.05, median_latency=10, false_alarm=0/200
- injected splits + onset ground truth saved under data/interim/subsample/

## Limitations

- Strong, clean synthetic shifts (anti-profile cluster); real drift is weaker/messier. Detection is bounded by the subsample backbone.
- Sanity uses static α; full latency/recovery comparison is Phase 8.

## Next steps

- Phase 8: control vs injected, all policies, detection/adaptation/recovery.

## Figures

![injection-sudden.png](figures/injection-sudden.png)
![injection-gradual.png](figures/injection-gradual.png)
![injection-recurring.png](figures/injection-recurring.png)
