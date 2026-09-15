# Run `20260827-203338__p1-1-learned-gate`

- **Task**: p1-1-learned-gate
- **Status**: completed
- **Started**: 2026-08-27T20:33:38.501274+00:00
- **Duration**: 69.64 s
- **Git commit**: f7f9a3349696eb907db6e95b5ba1c576086ec882
- **Seed**: 42
- **Dataset**: amazon-subsample-dense

## Objective

Compare an explicit CUSUM-driven fusion policy against a learned fusion gate (SLSRec-style) on the coherent-drift benchmark.

## Method

Train the gate on the frozen dense backbone; CUSUM enter=0.3; compare overall NDCG@10 per scenario with 95% CIs over injection seeds.

## Configuration

```json
{
  "seeds": 3,
  "gate_epochs": 3,
  "gate_hidden": 0,
  "scenarios": [
    "sudden",
    "gradual",
    "recurring"
  ]
}
```

## Results

_No metric recorded._

## Findings

- [sudden] proposed 0.01068+/-0.00117 vs learned-gate 0.01002+/-0.00099; diff +0.00067+/-0.00019 (robust)
- [gradual] proposed 0.01583+/-0.00125 vs learned-gate 0.01407+/-0.00147; diff +0.00176+/-0.00024 (robust)
- [recurring] proposed 0.00913+/-0.00042 vs learned-gate 0.00830+/-0.00048; diff +0.00083+/-0.00016 (robust)

## Limitations

- Dense cohort; one backbone seed; gate is a linear/MLP fusion.

## Next steps

- Second-dataset validation.
