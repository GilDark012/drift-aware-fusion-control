# Seed robustness — detection with confidence intervals (P1.2)

The supervisor asked for Monte-Carlo repetition with confidence intervals instead of a
single-seed headline. `scripts/seed_robustness.py` provides it for the detector's
operating characteristic on the **sudden** scenario.

## What is varied

The backbone is a **frozen** checkpoint and the streaming pass is deterministic given
the injected data, so the meaningful source of randomness is the **controlled-drift
injection**: which users are shifted and which anti-profile items they are shifted onto
(`InjectionConfig.seed`). We re-inject under **20 seeds** (42–61) at the full cohort
(`--n-users 300`) and aggregate to mean ± 95 % CI (normal approximation,
`1.96·s/√n`).

## Result (20 seeds, sudden)

| operating point | detection rate | false-alarm rate | median latency |
|---|--:|--:|--:|
| z-enter = 1.75 | **0.644 ± 0.014** | 0.024 ± 0.000 | 3.0 |
| z-enter = 2.00 | 0.279 ± 0.018 | 0.000 ± 0.000 | 3.0 |

**Reading.** The Phase-8 headline — ~65 % detection at ~2.4 % false-alarm — is
**robust**: 0.644 ± 0.014 across 20 injection seeds, a tight interval. Detection latency
is a stable 3 steps.

**Honesty note.** The single-seed Phase-7 figure of 0.39 at z-enter = 2.0 was a
favourable draw: the 20-seed mean at that stricter threshold is **0.279 ± 0.018**. The
operating point matters, and the z = 1.75 setting is the one that delivers the strong
result; we report both and no longer lean on the lucky single seed.

**Why the false-alarm CI is ~0.** The false-alarm rate is measured on the *control*
(un-injected) test streams. The eligible cohort is (near) exhaustive at `n-users 300`,
so essentially the same control users appear every seed and the false-alarm rate does
not vary with the injection seed — only detection does. This is expected, not a bug.

## Scope / limitation

The randomness axis here is the **injection sampling** only. **Backbone-initialisation
variance** — retraining the GRU under different seeds — is a separate and much heavier
axis (each seed is a full 15-epoch training run); it is noted as future work, not run
here. Reproduce with:

```bash
python scripts/seed_robustness.py --seeds 20 --n-users 300
```
