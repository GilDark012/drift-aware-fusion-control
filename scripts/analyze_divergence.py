#!/usr/bin/env python3
"""Phase 4 — compute and validate the long/short divergence signal D_u(t).

Loads the trained baseline backbone, computes per-user divergence trajectories on
the subsample cohort, checks that divergence responds to the interpretable
category-change indicator, and saves representative trajectories + population
figures. Everything is journaled.

Usage:
    python scripts/analyze_divergence.py --signal short_term_drift --representatives 6
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
from drift_reco.detectors.plots import DivergenceFigures  # noqa: E402
from drift_reco.detectors.rolling import RollingStatistics  # noqa: E402
from drift_reco.detectors.signal import get_signal  # noqa: E402
from drift_reco.detectors.trajectory import (  # noqa: E402
    DivergenceTracker,
    UserDivergence,
    user_streams,
)
from drift_reco.models.checkpoint import load_backbone  # noqa: E402
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402


def _cohort_streams(
    frames: dict[Split, pd.DataFrame], max_users: int, seed: int
) -> dict[int, pd.DataFrame]:
    """Build full per-user streams for a sample of test-active cohort users."""
    streams = user_streams([frames[s] for s in Split])
    cohort = frames[Split.TEST][S_USER_COLUMN].unique()
    rng = np.random.default_rng(seed)
    if cohort.size > max_users:
        cohort = rng.choice(cohort, size=max_users, replace=False)
    return {int(u): streams[int(u)] for u in cohort if int(u) in streams}


def _population_frame(trajectories: list[UserDivergence]) -> pd.DataFrame:
    """Concatenate trajectories into one tidy population table."""
    return pd.concat([t.to_frame() for t in trajectories], ignore_index=True)


def _log_population(journal: LabJournal, pop: pd.DataFrame) -> dict[str, float]:
    """Log population-level divergence statistics and return the change effect."""
    changed = pop.loc[pop["category_changed"], "divergence"]
    stable = pop.loc[~pop["category_changed"], "divergence"]
    stats = {
        "divergence_mean": float(pop["divergence"].mean()),
        "divergence_std": float(pop["divergence"].std()),
        "divergence_mean_category_changed": float(changed.mean()),
        "divergence_mean_category_stable": float(stable.mean()),
    }
    for name, value in stats.items():
        journal.metric(name, value, split="test")
    journal.metric("positions_total", len(pop), split="test", unit="positions")
    journal.metric(
        "category_change_rate", float(pop["category_changed"].mean()), split="test"
    )
    return stats


def _save_representatives(
    journal: LabJournal,
    figures: DivergenceFigures,
    trajectories: list[UserDivergence],
    n: int,
) -> None:
    """Save trajectory figures for the users with the most category changes."""
    ranked = sorted(
        trajectories, key=lambda t: int(t.category_changed().sum()), reverse=True
    )
    for traj in ranked[:n]:
        journal.figure(figures.trajectory(traj), f"trajectory_user_{traj.user}")


def main() -> int:
    """Run the Phase 4 divergence analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", default="short_term_drift")
    parser.add_argument("--max-users", type=int, default=800)
    parser.add_argument("--representatives", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_backbone(BASELINE_CHECKPOINT, device)
    frames, _ = read_subsample()

    config = RunConfig(
        task="phase4-divergence-signal",
        description=f"Drift signal D_u(t) via {args.signal} over the backbone",
        params={
            "signal": args.signal,
            "max_users": args.max_users,
            "checkpoint": BASELINE_CHECKPOINT,
        },
        dataset="amazon-subsample",
        tags=["phase4", "divergence"],
        seed=args.seed,
    )

    with LabJournal(config) as journal:
        signal = get_signal(args.signal)
        tracker = DivergenceTracker(model, meta.architecture, signal, device)
        journal.event("drift", "trajectories_start", detail=f"signal={args.signal}")
        streams = _cohort_streams(frames, args.max_users, args.seed)
        trajectories = [
            tracker.user_trajectory(s) for s in streams.values() if len(s) > 2
        ]
        trajectories = [t for t in trajectories if len(t) > 0]
        journal.metric("users_analyzed", len(trajectories), split="test", unit="users")

        pop = _population_frame(trajectories)
        stats = _log_population(journal, pop)
        figures = DivergenceFigures(RollingStatistics())
        journal.figure(
            figures.population_category_effect(pop), "population_category_effect"
        )
        journal.figure(figures.divergence_distribution(pop), "divergence_distribution")
        _save_representatives(journal, figures, trajectories, args.representatives)

        pop.sample(min(len(pop), 50000), random_state=args.seed).to_parquet(
            journal.run_dir / "data" / "divergence_population.parquet"
        )
        lift = (
            stats["divergence_mean_category_changed"]
            / max(stats["divergence_mean_category_stable"], 1e-9)
        )
        journal.metric("category_change_divergence_lift", lift, split="test")
        journal.summary(
            objective="Compute D_u(t)=dist(L,S) and validate it responds to shifts.",
            method=f"{args.signal} drift signal over the trained backbone for "
            f"{len(trajectories)} cohort users; compare divergence at category "
            f"changes vs stable positions; causal rolling summaries.",
            findings=[
                f"divergence mean={stats['divergence_mean']:.4f} "
                f"std={stats['divergence_std']:.4f} over {len(pop):,} positions",
                f"at category change={stats['divergence_mean_category_changed']:.4f} "
                f"vs stable={stats['divergence_mean_category_stable']:.4f} "
                f"(lift x{lift:.2f})",
                "category change is an interpretable indicator, not the definition",
            ],
            limitations=[
                "Subsample backbone; absolute divergence scale is model-specific.",
                "Category-change is a coarse proxy for true preference shift.",
            ],
            next_steps=[
                "Turn persistent rises in D_u(t) into a detector with state model "
                "and false-alarm control (Phase 5).",
            ],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
