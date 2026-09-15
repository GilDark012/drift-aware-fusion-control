# Run `20260819-054905__phase7-inject-drift`

- **Task**: phase7-inject-drift
- **Status**: failed
- **Started**: 2026-08-19T05:49:05.947912+00:00
- **Duration**: 0.04 s
- **Git commit**: 652dfa377d6de1a43f19a24d48af46cb180b0be4
- **Seed**: 42
- **Dataset**: amazon-subsample

## Objective

Controlled sudden/gradual/recurring preference shifts + sanity

## Method

See `config.json` for the exact parameters.

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

_No metric recorded._

## Error

```text
Traceback (most recent call last):
  File "C:\Users\gilal\Downloads\PFE_rec\scripts\inject_drift.py", line 132, in main
    injected, onsets = DriftInjector(inj_cfg, lookups).inject(frames[Split.TEST])
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\gilal\Downloads\PFE_rec\src\drift_reco\data\injection.py", line 174, in inject
    for rows in self._eligible_users(work)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\gilal\Downloads\PFE_rec\src\drift_reco\data\injection.py", line 156, in _eligible_users
    groups = {
             ^
  File "C:\Users\gilal\Downloads\PFE_rec\src\drift_reco\data\injection.py", line 157, in <dictcomp>
    int(u): g.to_numpy()
            ^^^^^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'to_numpy'
```
