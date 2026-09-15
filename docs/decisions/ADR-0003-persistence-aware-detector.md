# ADR-0003 — Persistence-aware detector with an adaptive per-user threshold

- **Statut** : accepté
- **Date** : 2026-08-18

## Contexte

The drift signal `D_u(t)` is noisy: single high readings occur constantly and must
not trigger adaptation (brief §4: "do not declare drift from a single unusual
interaction"). We need a detector that (a) fires only on a *persistent* rise, (b)
treats short-lived spikes as temporary deviations, and (c) means the same thing for
calm and volatile users. Thresholds must be tuned on validation only.

## Options envisagées

| Option | Avantages | Inconvénients |
| --- | --- | --- |
| Single-step threshold on raw `D` | trivial | fires on every spike; no false-alarm control; forbidden by §4 |
| Global threshold + persistence | some robustness | a global level over-fires on naturally volatile users (H6) |
| **Adaptive (per-user z-score) threshold + persistence + state machine** | per-user calibrated; persistence kills blips; explicit STABLE/EMERGING/CONFIRMED/TEMPORARY states drive α later | two knobs to tune; conservative by design |

## Décision

Adopt a **state machine** (STABLE → EMERGING → CONFIRMED → recover → STABLE) driven
by a **causal per-user z-score** of `D_u(t)` (`adaptive` strategy), with a
**persistence** requirement to confirm and a **TEMPORARY** path for rises that fade
before persisting. Default tuned on validation: `enter=2.5`, `exit=1.0`,
`persistence=5`, `recovery_persistence=5`, `warmup=8` (`configs/detector.json`).
The `global` raw-divergence threshold is retained as the H6 ablation baseline.

## Conséquences

- The four states are the interface to Phase 6: α is a function of detector state
  (STABLE→high, EMERGING→moderate, CONFIRMED→low), so the detector and the adaptive
  fusion are cleanly decoupled.
- **Validated on the val cohort:** persistence 1→5 cuts confirmed alarms
  2.43→0.07 per 1000 steps (H5); the adaptive threshold's confirmed rate is nearly
  flat across volatility strata `[0.0, 0.0, 0.15]` where the global threshold rises
  `[2.49, 3.57, 4.56]` (H6).
- The default is deliberately **false-alarm-conservative**. `persistence` is the
  latency ↔ false-alarm knob and is **swept as an ablation in Phase 8**; detection
  *power* is re-checked against known onsets in Phase 7 and the operating point may
  be adjusted there — always on validation/controlled data, never on test.
- Explicitly **excludes** using recommendation-quality degradation as the primary
  trigger (that would be a delayed detector, §6); performance degradation may enter
  later only as a secondary confirmatory signal.
