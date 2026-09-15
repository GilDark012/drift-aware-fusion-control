# Run `20260827-204813__second-dataset`

- **Task**: second-dataset
- **Status**: completed
- **Started**: 2026-08-27T20:48:13.588416+00:00
- **Duration**: 3.55 s
- **Git commit**: fdb9ab4056ccaa4b38aa3b0f6af9601514dafc30
- **Seed**: 42
- **Dataset**: amazon-books

## Objective

Test whether the adaptation win replicates on Books alone, a single distinct product domain rather than the mixed three-category corpus.

## Method

Fresh backbone on the Books-only dense cohort; CUSUM enter=0.1; per-scenario paired differences with 95% CIs over seeds.

## Configuration

```json
{
  "category": "Books",
  "seeds": 2
}
```

## Results

_No metric recorded._

## Findings

- [sudden] vs window-B +0.00000+/-0.00000 (ns); vs static-0.8 -0.00227+/-0.00444 (ns)
- [gradual] vs window-B -0.00019+/-0.00000 (ns); vs static-0.8 +0.00000+/-0.00000 (ns)
- [recurring] vs window-B +0.00000+/-0.00000 (ns); vs static-0.8 +0.00000+/-0.00000 (ns)

## Limitations

- Single category; one backbone seed; dense cohort.

## Next steps

- Additional categories; full-scale replication.
