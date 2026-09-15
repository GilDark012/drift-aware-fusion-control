# Run `20260825-162831__phase4-divergence-signal`

- **Task**: phase4-divergence-signal
- **Status**: failed
- **Started**: 2026-08-25T16:28:31.013814+00:00
- **Duration**: 1.42 s
- **Git commit**: 7e600cc39b00d3921248d4443656d9381154576b
- **Seed**: 42
- **Dataset**: amazon-subsample

## Objective

Drift signal D_u(t) via short_term_drift over the backbone

## Method

See `config.json` for the exact parameters.

## Configuration

```json
{
  "signal": "short_term_drift",
  "max_users": 600,
  "checkpoint": "artifacts/models/baseline-subsample"
}
```

## Results

_No metric recorded._

## Error

```text
Traceback (most recent call last):
  File "C:\Users\gilal\Downloads\PFE_rec\scripts\analyze_divergence.py", line 121, in main
    trajectories = [
                   ^
  File "C:\Users\gilal\Downloads\PFE_rec\scripts\analyze_divergence.py", line 122, in <listcomp>
    tracker.user_trajectory(s) for s in streams.values() if len(s) > 2
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\gilal\Downloads\PFE_rec\src\drift_reco\detectors\trajectory.py", line 136, in user_trajectory
    divergence = self._divergence(examples)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\gilal\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\gilal\Downloads\PFE_rec\src\drift_reco\detectors\trajectory.py", line 122, in _divergence
    return self.signal.compute(torch.cat(longs), torch.cat(shorts))
                               ^^^^^^^^^^^^^^^^
RuntimeError: torch.cat(): expected a non-empty list of Tensors
```
