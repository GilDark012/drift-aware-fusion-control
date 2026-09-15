#!/usr/bin/env python3
"""Phase 5 — tune and validate the persistence-aware drift detector.

On the validation cohort (no injected ground truth yet), characterise the detector's
alarm rate: show that persistence suppresses false alarms (H5) and that an adaptive
per-user threshold behaves more uniformly across volatility than a global one (H6).
Thresholds are tuned on validation only; the chosen config is saved for later phases.

Usage:
    python scripts/detect_drift.py --max-users 1500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift_reco.config.constants import BASELINE_CHECKPOINT, S_USER_COLUMN  # noqa: E402
from drift_reco.data.schema import Split  # noqa: E402
from drift_reco.data.subsample import read_subsample  # noqa: E402
from drift_reco.detectors.analysis import (  # noqa: E402
    persistence_sweep,
    run_detector_over,
    summarize,
    volatility_strata,
)
from drift_reco.detectors.detector import DetectorConfig  # noqa: E402
from drift_reco.detectors.plots import (  # noqa: E402
    DivergenceFigures,
    persistence_sweep_figure,
    strategy_stratum_figure,
)
from drift_reco.detectors.signal import get_signal  # noqa: E402
from drift_reco.detectors.trajectory import (  # noqa: E402
    DivergenceTracker,
    user_streams,
)
from drift_reco.models.checkpoint import load_backbone  # noqa: E402
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402

PERSISTENCES = [1, 3, 5, 8, 12]


def _series(
    frames: dict[Split, pd.DataFrame],
    splits: list[Split],
    tracker: DivergenceTracker,
    users: np.ndarray,
) -> tuple[list[np.ndarray], list]:
    """Compute divergence series and trajectories for the given users."""
    streams = user_streams([frames[s] for s in splits])
    trajectories = [
        tracker.user_trajectory(streams[int(u)]) for u in users if int(u) in streams
    ]
    trajectories = [t for t in trajectories if len(t) > 2]
    return [t.divergence for t in trajectories], trajectories


def _h6_table(series: list[np.ndarray], enter_global: float) -> pd.DataFrame:
    """Confirmed-rate table by volatility stratum for adaptive vs global (H6)."""
    strata = volatility_strata(series, 3)
    configs = {
        "adaptive": DetectorConfig(strategy="adaptive", enter=2.5, persistence=5),
        "global": DetectorConfig(strategy="global", enter=enter_global, persistence=5),
    }
    rows = []
    for name, cfg in configs.items():
        results = run_detector_over(series, cfg)
        for stratum in np.unique(strata):
            subset = [r for r, s in zip(results, strata, strict=True) if s == stratum]
            summary = summarize(subset)
            rows.append(
                {
                    "strategy": name,
                    "stratum": int(stratum),
                    "confirmed_per_1k": summary.confirmed_per_1k,
                    "users": summary.n_users,
                }
            )
    return pd.DataFrame(rows)


def _save_representatives(
    journal: LabJournal,
    chosen: DetectorConfig,
    trajectories: list,
) -> int:
    """Overlay the chosen detector on the users with the most confirmed drift."""
    from drift_reco.detectors.detector import PersistenceDriftDetector

    detector = PersistenceDriftDetector(chosen)
    figures = DivergenceFigures()
    scored = [(t, detector.run(t.divergence)) for t in trajectories]
    scored.sort(key=lambda tr: tr[1].n_confirmed, reverse=True)
    n_shown = 0
    for traj, result in scored[:5]:
        if result.n_confirmed == 0:
            break
        journal.figure(
            figures.detection_overlay(traj, result), f"detection_user_{traj.user}"
        )
        n_shown += 1
    return n_shown


def main() -> int:
    """Run the Phase 5 detector tuning and validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-users", type=int, default=1500)
    parser.add_argument("--signal", default="short_term_drift")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_backbone(BASELINE_CHECKPOINT, device)
    frames, _ = read_subsample()
    tracker = DivergenceTracker(
        model, meta.architecture, get_signal(args.signal), device
    )

    config = RunConfig(
        task="phase5-drift-detector",
        description="Persistence-aware detector: tune alarm rate on validation (H5/H6)",
        params={"signal": args.signal, "persistences": PERSISTENCES},
        dataset="amazon-subsample",
        tags=["phase5", "detector"],
        seed=args.seed,
    )

    with LabJournal(config) as journal:
        rng = np.random.default_rng(args.seed)
        val_users = frames[Split.VAL][S_USER_COLUMN].unique()
        if val_users.size > args.max_users:
            val_users = rng.choice(val_users, size=args.max_users, replace=False)
        journal.event("drift", "tuning_start", detail=f"{val_users.size} val users")

        # Tuning signal: divergence over train+val only (validation regime).
        series, trajectories = _series(
            frames, [Split.TRAIN, Split.VAL], tracker, val_users
        )
        journal.metric("users_tuned", len(series), split="val", unit="users")

        base = DetectorConfig(strategy="adaptive", enter=2.5, persistence=5)
        sweep = persistence_sweep(series, base, PERSISTENCES)
        for persistence, summary in sweep:
            journal.metric(
                "confirmed_per_1k", summary.confirmed_per_1k, split="val",
                step=persistence, context=f"persistence={persistence}",
            )
        journal.figure(persistence_sweep_figure(sweep), "persistence_sweep")

        pooled = np.concatenate([s[np.isfinite(s)] for s in series])
        enter_global = float(np.quantile(pooled, 0.90))
        h6 = _h6_table(series, enter_global)
        journal.table(h6.to_dict("records"), "h6_strategy_by_stratum")
        journal.figure(strategy_stratum_figure(h6), "strategy_by_stratum")

        chosen = DetectorConfig(strategy="adaptive", enter=2.5, persistence=5)
        Path("configs").mkdir(exist_ok=True)
        Path("configs/detector.json").write_text(
            chosen.model_dump_json(indent=2), encoding="utf-8"
        )
        chosen_summary = summarize(run_detector_over(series, chosen))
        journal.metric(
            "chosen_confirmed_per_1k", chosen_summary.confirmed_per_1k, split="val"
        )
        journal.metric(
            "chosen_temporary_per_1k", chosen_summary.temporary_per_1k, split="val"
        )
        journal.metric(
            "chosen_users_with_confirm",
            chosen_summary.users_with_confirm_frac,
            split="val",
        )

        # Illustrate on full streams (train+val+test) for the clearest cases.
        _, full_traj = _series(
            frames, [Split.TRAIN, Split.VAL, Split.TEST], tracker, val_users
        )
        n_shown = _save_representatives(journal, chosen, full_traj)

        _summarize(journal, sweep, h6, chosen_summary, enter_global, n_shown)
    return 0


def _summarize(journal, sweep, h6, chosen_summary, enter_global, n_shown) -> None:
    """Write the run's narrative results."""
    p1 = next(s for p, s in sweep if p == 1)
    p5 = next(s for p, s in sweep if p == 5)
    adaptive_spread = h6[h6.strategy == "adaptive"]["confirmed_per_1k"]
    global_spread = h6[h6.strategy == "global"]["confirmed_per_1k"]
    journal.summary(
        objective="Turn a persistent divergence rise into a confirmed-shift detector "
        "with false-alarm control, tuned on validation.",
        method="State machine STABLE→EMERGING→CONFIRMED with TEMPORARY suppression; "
        "persistence sweep (H5) and adaptive-vs-global by volatility (H6).",
        findings=[
            f"persistence 1→5 cuts confirmed alarms {p1.confirmed_per_1k:.1f}→"
            f"{p5.confirmed_per_1k:.1f} per 1k steps (H5)",
            f"adaptive confirmed/1k spread across strata "
            f"[{adaptive_spread.min():.1f},{adaptive_spread.max():.1f}] vs global "
            f"[{global_spread.min():.1f},{global_spread.max():.1f}] (H6)",
            f"chosen (adaptive, enter=2.5, persistence=5): "
            f"{chosen_summary.confirmed_per_1k:.1f} confirmed/1k, "
            f"{chosen_summary.users_with_confirm_frac:.1%} of users; "
            f"global enter={enter_global:.3f}",
            f"detector overlays saved for {n_shown} clearest users",
        ],
        limitations=[
            "No injected ground truth yet; alarm rate is a false-alarm proxy on "
            "natural validation data (Phase 7 adds known onsets).",
        ],
        next_steps=["Drive α(t) from detector state — adaptive fusion (Phase 6)."],
    )


if __name__ == "__main__":
    raise SystemExit(main())
