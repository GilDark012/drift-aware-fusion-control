# Run `20260827-205021__second-dataset`

- **Task**: second-dataset
- **Status**: completed
- **Started**: 2026-08-27T20:50:21.004508+00:00
- **Duration**: 413.35 s
- **Git commit**: 2644698ded246e217bd62300ce9edebb234b7686
- **Seed**: 42
- **Dataset**: amazon-electronics

## Objective

Test whether the adaptation win replicates on Electronics alone, a single distinct product domain rather than the mixed three-category corpus.

## Method

Fresh backbone on the Electronics-only dense cohort; CUSUM enter=0.5; per-scenario paired differences with 95% CIs over seeds.

## Configuration

```json
{
  "category": "Electronics",
  "seeds": 20
}
```

## Results

_No metric recorded._

## Findings

- [sudden] vs window-B -0.00156+/-0.00047 (ns); vs static-0.8 +0.00428+/-0.00049 (ROBUST)
- [gradual] vs window-B +0.00098+/-0.00034 (ROBUST); vs static-0.8 -0.00089+/-0.00043 (ns)
- [recurring] vs window-B -0.00118+/-0.00034 (ns); vs static-0.8 +0.00416+/-0.00044 (ROBUST)

## Limitations

- Single category; one backbone seed; dense cohort.

## Next steps

- Additional categories; full-scale replication.
