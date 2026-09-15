# Run `20260825-163656__phase6-adaptive-fusion`

- **Task**: phase6-adaptive-fusion
- **Status**: completed
- **Started**: 2026-08-25T16:36:56.803357+00:00
- **Duration**: 14.52 s
- **Git commit**: 7e600cc39b00d3921248d4443656d9381154576b
- **Seed**: 42
- **Dataset**: amazon-subsample

## Objective

Drive α(t) from detector state and compare against static and detector-free adaptation (mechanism + no regression on natural data).

## Method

Streaming per-user eval over one backbone; α = policy(state, z). Policies: static (A), continuous (D), state abrupt/gradual (proposed).

## Configuration

```json
{
  "top_k": 10,
  "policies": [
    "static-0.5",
    "continuous",
    "state-abrupt",
    "state-gradual"
  ]
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| users | test | 0 | 441.0 | users |  |
| ndcg_at_k | test | 0 | 0.015454352121862651 |  | static-0.5 |
| hit_rate_at_k | test | 0 | 0.028805496828752643 |  | static-0.5 |
| mrr | test | 0 | 0.016593819721820374 |  | static-0.5 |
| ndcg_at_k_seen | test | 0 | 0.015454352121862651 |  | static-0.5 |
| mean_alpha | test | 0 | 0.5 |  | static-0.5 |
| alpha_changes_per_1k | test | 0 | 0.0 |  | static-0.5 |
| confirmed_frac | test | 0 | 0.0 |  | static-0.5 |
| ndcg_at_k | test | 0 | 0.014554755713284276 |  | continuous |
| hit_rate_at_k | test | 0 | 0.029069767441860465 |  | continuous |
| mrr | test | 0 | 0.015090192052850011 |  | continuous |
| ndcg_at_k_seen | test | 0 | 0.014554755713284276 |  | continuous |
| mean_alpha | test | 0 | 0.7670669070237298 |  | continuous |
| alpha_changes_per_1k | test | 0 | 526.6913319238901 |  | continuous |
| confirmed_frac | test | 0 | 0.0 |  | continuous |
| ndcg_at_k | test | 0 | 0.0144046079033518 |  | state-abrupt |
| hit_rate_at_k | test | 0 | 0.029069767441860465 |  | state-abrupt |
| mrr | test | 0 | 0.014874205508507187 |  | state-abrupt |
| ndcg_at_k_seen | test | 0 | 0.0144046079033518 |  | state-abrupt |
| mean_alpha | test | 0 | 0.7971458773784357 |  | state-abrupt |
| alpha_changes_per_1k | test | 0 | 8.192389006342495 |  | state-abrupt |
| confirmed_frac | test | 0 | 0.0 |  | state-abrupt |
| ndcg_at_k | test | 0 | 0.0144046079033518 |  | state-gradual |
| hit_rate_at_k | test | 0 | 0.029069767441860465 |  | state-gradual |
| mrr | test | 0 | 0.014889741158032108 |  | state-gradual |
| ndcg_at_k_seen | test | 0 | 0.0144046079033518 |  | state-gradual |
| mean_alpha | test | 0 | 0.7988239957716702 |  | state-gradual |
| alpha_changes_per_1k | test | 0 | 17.441860465116278 |  | state-gradual |
| confirmed_frac | test | 0 | 0.0 |  | state-gradual |

## Findings

- NDCG@10 test: static=0.0155 continuous=0.0146 abrupt=0.0144 gradual=0.0144
- α-changes/1k: static=0.0 continuous=526.7 gradual=17.4
- mean α: static=0.50 gradual=0.80
- confirmed_frac on test windows=0.0000 (near-zero: short natural test streams rarely confirm)
- synthetic control-law α(t) figures saved (gradual/abrupt/continuous)

## Limitations

- Natural test data has few confirmed shifts, so headline metrics are close by design; the benefit is measured under injected drift (Phase 7).
- Static uses α=0.5 while state policies rest near α_stable=0.8, so the natural-data gap reflects α level more than adaptation.

## Next steps

- Inject controlled shifts with known onsets (Phase 7).

## Figures

![alpha-trajectory-gradual-synthetic.png](figures/alpha-trajectory-gradual-synthetic.png)
![alpha-policy-comparison-synthetic.png](figures/alpha-policy-comparison-synthetic.png)
