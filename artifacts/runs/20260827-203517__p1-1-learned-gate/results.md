# Run `20260827-203517__p1-1-learned-gate`

- **Task**: p1-1-learned-gate
- **Status**: completed
- **Started**: 2026-08-27T20:35:17.092596+00:00
- **Duration**: 460.65 s
- **Git commit**: d7c774ed2e50f8ec66d414e29bdea8212be6fbc9
- **Seed**: 42
- **Dataset**: amazon-subsample-dense

## Objective

Compare an explicit CUSUM-driven fusion policy against a learned fusion gate (SLSRec-style) on the coherent-drift benchmark.

## Method

Train the gate on the frozen dense backbone; CUSUM enter=0.3; compare overall NDCG@10 per scenario with 95% CIs over injection seeds.

## Configuration

```json
{
  "seeds": 20,
  "gate_epochs": 15,
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

- [sudden] proposed 0.01082+/-0.00041 vs learned-gate 0.01008+/-0.00038; diff +0.00074+/-0.00013 (robust)
- [gradual] proposed 0.01534+/-0.00045 vs learned-gate 0.01373+/-0.00041; diff +0.00161+/-0.00025 (robust)
- [recurring] proposed 0.00923+/-0.00023 vs learned-gate 0.00853+/-0.00025; diff +0.00069+/-0.00009 (robust)

## Limitations

- Dense cohort; one backbone seed; gate is a linear/MLP fusion.

## Next steps

- Second-dataset validation.
