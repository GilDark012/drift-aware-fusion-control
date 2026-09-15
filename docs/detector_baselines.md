# Our detector vs classical drift detectors (P1.1, detection half)

The review asked us to compare against the concept-drift literature, not only against
static-α ablations. `scripts/baseline_detectors.py` runs our persistence-aware detector
and three canonical detectors — **Page-Hinkley** (Page 1954), **CUSUM** (Page 1954) and
**ADWIN** (Bifet & Gavaldà 2007) — on the **same** per-user divergence signal from the
injected *sudden* scenario, sweeping each one's sensitivity knob. Implementations:
`src/drift_reco/detectors/baselines.py` (dependency-free, unit-tested).

## Operating points (sudden, n=83 injected / 83 control)

| detector | best detection @ ~0 % false-alarm | detection at ~1.0 |
|---|--:|--:|
| **CUSUM** | **1.00 @ 0.0 %** (h=1.0) | 1.00 @ 0.0 % |
| **Page-Hinkley** | 0.98 @ 0.0 % (λ=2.0) | 1.00 @ 6.0 % |
| **ours** (z-score + persistence) | 0.33 @ 0.0 % (z=2.0) | 1.00 @ 10.8 % |
| **ADWIN** | 0.02 (fails) | never |

Full sweep: `artifacts/runs/…__p1-1-baseline-detectors/data/detector-operating-points.csv`;
figure `detector-comparison-sudden.png`.

## The honest reading

**On this benchmark, two trivial detectors strictly dominate ours.** CUSUM on the raw
divergence reaches **100 % detection at 0 % false-alarm**; Page-Hinkley is nearly as
good. Our detector needs to accept ~5–11 % false alarms to reach the same detection. Our
own prior numbers are reproduced exactly (0.33 @ z=2.0, 0.65 @ z=1.75), so this is not a
regression — the simple baselines are genuinely better **here**.

**Why.** Our detector normalises the signal by each user's *own* recent variance (a
causal z-score) and then demands persistence. The injected anti-profile shift produces a
huge, clean, sustained jump in the raw divergence; CUSUM/Page-Hinkley accumulate that
jump directly and fire, while the z-score **dilutes** it (a naturally volatile user's
baseline variance shrinks the normalised spike). The added machinery *costs* detection
power on strong, clean shifts.

**ADWIN fails** for a different reason: it needs to accumulate many post-change samples
before its variance bound is exceeded, and per-user test streams are short (~tens of
steps), so it is far too sluggish — max 0.02 detection.

## What this means for the thesis

Together with the window-B finding (`docs/window_b_analysis.md`), the external baselines
— exactly as the supervisor intended — show that **on the current clean synthetic
benchmark, neither the sophisticated detector nor the long/short fusion control is
empirically justified**: a global-threshold CUSUM detects the shifts perfectly, and the
long-term representation adds nothing to recommendation quality.

This is not a failure of the evaluation; it is the evaluation doing its job. The
uncomfortable but defensible conclusion is that the **benchmark is too easy in one
dimension and too sparse in another**:

- **Too easy for detection.** The anti-profile injection is a strong, unambiguous shift.
  The design rationale for a per-user adaptive z-score + persistence — avoiding false
  alarms on *heterogeneously volatile* users while still catching *weak* drift — is never
  stressed, because the synthetic shift is neither weak nor ambiguous. A global CUSUM
  wins precisely because the benchmark removes the difficulty our detector targets.
- **Too sparse for fusion.** The long-term representation carries no signal, so there is
  nothing for α-control to exploit.

## The regime where our detector *should* win (the real next experiment)

To justify the adaptive detector one must build the harder benchmark it was designed for:

1. **Weak / gradual shifts** near the noise floor, where a single global threshold must
   trade misses against false alarms — and a per-user z-score can separate them.
2. **Heterogeneous user volatility**, so that a global raw-divergence threshold
   false-alarms on volatile users at any sensitivity high enough to catch calm users'
   drift. This is the exact failure mode the z-score normalisation addresses.
3. Report all four detectors on *that* benchmark. If ours still loses, the adaptive
   machinery should be dropped in favour of CUSUM — which would itself be a clean,
   honest result.

Until that experiment exists, the draft must **not** claim the detector is superior to
classical methods; the current evidence says the opposite on this data.
