#!/usr/bin/env python3
"""P1.1 (detection half) — our detector vs classical drift detectors.

Runs our persistence-aware state machine and three canonical concept-drift detectors
(Page-Hinkley, CUSUM, ADWIN) on the **same** per-user divergence signal from the
injected *sudden* scenario, sweeping each detector's sensitivity knob to trace its
detection-vs-false-alarm operating curve. This answers the review's call to compare
against the drift-detection literature rather than only against static-alpha ablations.

Usage:
    python scripts/baseline_detectors.py --n-users 300
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from inject_drift import InjectionConfig  # noqa: E402
from seed_robustness import _build, _prepare, _stream_seed  # noqa: E402

from drift_reco.data.injection import DriftInjector  # noqa: E402
from drift_reco.data.schema import Split  # noqa: E402
from drift_reco.detectors.baselines import build_baseline  # noqa: E402
from drift_reco.detectors.detector import (  # noqa: E402
    DetectorConfig,
    DriftState,
    PersistenceDriftDetector,
)
from drift_reco.evaluation.latency import detection_latency  # noqa: E402
from drift_reco.evaluation.plots import detector_comparison  # noqa: E402
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402

StatesFn = Callable[[np.ndarray], np.ndarray]

# Sweep of each detector's primary sensitivity knob (least -> most sensitive).
OURS_ENTER = (3.0, 2.5, 2.0, 1.75, 1.5, 1.25)
PH_THRESH = (3.0, 2.0, 1.0, 0.5, 0.25, 0.1)
CUSUM_THRESH = (3.0, 2.0, 1.0, 0.5, 0.25, 0.1)
ADWIN_DELTA = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
FA_TARGET = 0.05


def _ours_states_fn(base_cfg: DetectorConfig, enter: float) -> StatesFn:
    """States function for our detector at a given z-enter level."""
    cfg = base_cfg.model_copy(update={"enter": enter})

    def run(divergence: np.ndarray) -> np.ndarray:
        return PersistenceDriftDetector(cfg).run(divergence).states

    return run


def _baseline_states_fn(name: str, threshold: float) -> StatesFn:
    """States function for a classical baseline at a given knob value."""
    detector = build_baseline(name, threshold)
    return detector.run


def _operating_point(
    states_fn: StatesFn,
    inj_series: list[tuple[np.ndarray, int]],
    ctl_series: list[np.ndarray],
) -> tuple[float, float]:
    """Return (false_alarm_rate, detection_rate) for one detector configuration."""
    detected = [
        detection_latency(0, states_fn(div), onset).detected
        for div, onset in inj_series
    ]
    false_alarms = [
        bool((states_fn(div) == int(DriftState.CONFIRMED)).any()) for div in ctl_series
    ]
    det = float(np.mean(detected)) if detected else 0.0
    fa = float(np.mean(false_alarms)) if false_alarms else 0.0
    return fa, det


def _sweep(
    knobs: tuple[float, ...],
    make_fn: Callable[[float], StatesFn],
    inj_series: list[tuple[np.ndarray, int]],
    ctl_series: list[np.ndarray],
) -> list[tuple[float, float, float]]:
    """Trace one detector's operating curve; return (knob, fa, det) points."""
    points = []
    for knob in knobs:
        fa, det = _operating_point(make_fn(knob), inj_series, ctl_series)
        points.append((knob, fa, det))
    return points


def _best_at_fa(points: list[tuple[float, float, float]], fa_max: float) -> tuple:
    """Highest detection among points with false-alarm <= ``fa_max`` (else min FA)."""
    feasible = [p for p in points if p[1] <= fa_max]
    if feasible:
        return max(feasible, key=lambda p: p[2])
    return min(points, key=lambda p: p[1])


def _collect(
    base_cfg: DetectorConfig,
    inj_series: list[tuple[np.ndarray, int]],
    ctl_series: list[np.ndarray],
) -> dict[str, list[tuple[float, float, float]]]:
    """Run every detector's sweep and return their (knob, fa, det) curves."""
    return {
        "ours": _sweep(
            OURS_ENTER,
            lambda z: _ours_states_fn(base_cfg, z), inj_series, ctl_series,
        ),
        "page-hinkley": _sweep(
            PH_THRESH,
            lambda t: _baseline_states_fn("page-hinkley", t), inj_series, ctl_series,
        ),
        "cusum": _sweep(
            CUSUM_THRESH,
            lambda t: _baseline_states_fn("cusum", t), inj_series, ctl_series,
        ),
        "adwin": _sweep(
            ADWIN_DELTA,
            lambda d: _baseline_states_fn("adwin", d), inj_series, ctl_series,
        ),
    }


def _rows(curves: dict[str, list[tuple[float, float, float]]]) -> list[dict]:
    """Flatten every operating point into table rows."""
    rows = []
    for name, points in curves.items():
        for knob, fa, det in points:
            rows.append({
                "detector": name,
                "knob": round(knob, 4),
                "false_alarm": round(fa, 4),
                "detection": round(det, 4),
            })
    return rows


def main() -> int:
    """Compare our detector to Page-Hinkley/CUSUM/ADWIN on the same signal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-users", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames, model, meta, detector, evaluator, device = _build()
    ctx_by, control_by, lookups, anchors = _prepare(frames, model, meta, device)
    base_cfg = detector.config

    inj_cfg = InjectionConfig(drift_type="sudden", n_users=args.n_users, seed=args.seed)
    injected, onsets = DriftInjector(inj_cfg, lookups).inject(
        frames[Split.TEST], anchors
    )
    onset_by_user = {int(o.user): int(o.onset_index) for o in onsets}
    inj_series, ctl_series = _stream_seed(
        evaluator, detector, ctx_by, control_by, injected, onset_by_user
    )

    config = RunConfig(
        task="p1-1-baseline-detectors",
        description="Our detector vs Page-Hinkley/CUSUM/ADWIN on the same divergence",
        params={"n_users": args.n_users, "detectors":
                ["ours", "page-hinkley", "cusum", "adwin"]},
        dataset="amazon-subsample",
        tags=["p1-1", "baselines", "drift-detection"],
        seed=args.seed,
    )

    with LabJournal(config) as journal:
        curves = _collect(base_cfg, inj_series, ctl_series)
        journal.table(_rows(curves), "detector_operating_points")
        plot_curves = {
            n: [(fa, det) for _, fa, det in pts] for n, pts in curves.items()
        }
        journal.figure(detector_comparison(plot_curves), "detector_comparison_sudden")
        _summarize(journal, curves, len(inj_series), len(ctl_series))
        _print(curves)
    return 0


def _summarize(
    journal: LabJournal,
    curves: dict[str, list[tuple[float, float, float]]],
    n_inj: int,
    n_ctl: int,
) -> None:
    """Write the run's narrative results."""
    findings = []
    for name, points in curves.items():
        knob, fa, det = _best_at_fa(points, FA_TARGET)
        findings.append(
            f"{name}: best detection {det:.2f} @ false-alarm {fa:.3f} "
            f"(knob={knob:g}, target FA<={FA_TARGET})"
        )
    journal.summary(
        objective="Compare our persistence-aware detector against the classical "
        "concept-drift literature (Page-Hinkley, CUSUM, ADWIN) on the same signal.",
        method=f"Sweep each detector's sensitivity knob over {n_inj} injected and "
        f"{n_ctl} control divergence series; trace detection vs false-alarm.",
        findings=findings,
        limitations=[
            "Single injection seed; sudden scenario. DDM/EDDM excluded (they monitor "
            "classifier error, not a continuous divergence).",
            "Baselines consume the raw divergence; ours adds a causal per-user z-score "
            "and persistence — that pipeline difference is exactly what is compared.",
        ],
        next_steps=["Sequential-recommender baselines (RecBole GRU4Rec/SASRec)."],
    )


def _print(curves: dict[str, list[tuple[float, float, float]]]) -> None:
    """Echo each detector's best operating point to stdout."""
    print("\nDrift-detector comparison (sudden), best detection at FA <= 0.05:")
    for name, points in curves.items():
        knob, fa, det = _best_at_fa(points, FA_TARGET)
        print(
            f"  {name:>13}: detection={det:.3f} @ false-alarm={fa:.3f} (knob={knob:g})"
        )


if __name__ == "__main__":
    raise SystemExit(main())
