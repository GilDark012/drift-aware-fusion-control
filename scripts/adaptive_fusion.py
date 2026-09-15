#!/usr/bin/env python3
"""Phase 6 — adaptive fusion: drive α(t) from the detector state.

Compares α-policies over one shared backbone on the test cohort: static (Baseline A),
continuous/detector-free (Baseline D), and the proposed detector-driven state policy
in abrupt and gradual variants (H4). On natural data the headline metrics are
expected to be close — the mechanism and its lack of regression are the point; the
win under a known shift is measured in Phase 7. Everything is journaled.

Usage:
    python scripts/adaptive_fusion.py --max-users 800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift_reco.adaptation.plots import (  # noqa: E402
    alpha_policy_comparison,
    alpha_trajectory,
)
from drift_reco.adaptation.policy import AdaptationConfig, build_policy  # noqa: E402
from drift_reco.config.constants import BASELINE_CHECKPOINT, S_USER_COLUMN  # noqa: E402
from drift_reco.data.schema import Split  # noqa: E402
from drift_reco.data.subsample import read_subsample  # noqa: E402
from drift_reco.detectors.detector import (  # noqa: E402
    DetectorConfig,
    PersistenceDriftDetector,
)
from drift_reco.detectors.rolling import RollingStatistics  # noqa: E402
from drift_reco.detectors.signal import get_signal  # noqa: E402
from drift_reco.evaluation.metrics import compute_ranking_metrics  # noqa: E402
from drift_reco.evaluation.streaming import (  # noqa: E402
    StreamingEvaluator,
    UserRunResult,
    pool_metrics,
)
from drift_reco.models.checkpoint import load_backbone  # noqa: E402
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402

POLICIES = {
    "static-0.5": AdaptationConfig(policy="static", alpha_static=0.5),
    "continuous": AdaptationConfig(policy="continuous"),
    "state-abrupt": AdaptationConfig(policy="state", mode="abrupt"),
    "state-gradual": AdaptationConfig(policy="state", mode="gradual"),
}


def _user_frames(
    frames: dict[Split, pd.DataFrame], users: np.ndarray
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Build (context=train+val, eval=test) frames per user."""
    context = pd.concat([frames[Split.TRAIN], frames[Split.VAL]], ignore_index=True)
    ctx_by_user = dict(iter(context.groupby(S_USER_COLUMN)))
    test_by_user = dict(iter(frames[Split.TEST].groupby(S_USER_COLUMN)))
    out: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for user in users:
        u = int(user)
        if u in ctx_by_user and u in test_by_user:
            out[u] = (ctx_by_user[u], test_by_user[u])
    return out


def _run_policy(
    evaluator: StreamingEvaluator,
    user_frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    detector: PersistenceDriftDetector,
    config: AdaptationConfig,
) -> list[UserRunResult]:
    """Run one α-policy over every user."""
    policy = build_policy(config)
    results = []
    for context, evaluation in user_frames.values():
        result = evaluator.run_user(context, evaluation, detector, policy)
        if result is not None:
            results.append(result)
    return results


def _activity(results: list[UserRunResult]) -> dict[str, float]:
    """Adaptation-activity summary: mean α, α-changes and confirmed fraction."""
    total_steps = sum(r.alpha.size for r in results)
    changes = sum(int((np.abs(np.diff(r.alpha)) > 1e-6).sum()) for r in results)
    confirmed = sum(int((r.state == 2).sum()) for r in results)
    mean_alpha = float(np.concatenate([r.alpha for r in results]).mean())
    scale = 1000.0 / max(total_steps, 1)
    return {
        "mean_alpha": mean_alpha,
        "alpha_changes_per_1k": changes * scale,
        "confirmed_frac": confirmed / max(total_steps, 1),
    }


def _log_policy(journal: LabJournal, name: str, results: list[UserRunResult]) -> dict:
    """Log a policy's pooled metrics and adaptation activity; return a row."""
    metrics = pool_metrics(results, journal.config.params["top_k"])
    activity = _activity(results)
    for field in ("ndcg_at_k", "hit_rate_at_k", "mrr", "ndcg_at_k_seen"):
        journal.metric(field, getattr(metrics, field), split="test", context=name)
    for field, value in activity.items():
        journal.metric(field, value, split="test", context=name)
    return {
        "policy": name,
        "ndcg_at_k": metrics.ndcg_at_k,
        "hit_rate_at_k": metrics.hit_rate_at_k,
        "mrr": metrics.mrr,
        "ndcg_at_k_seen": metrics.ndcg_at_k_seen,
        **activity,
    }


def _synthetic_divergence() -> tuple[NDArray[np.float64], NDArray[np.datetime64]]:
    """A canonical low→ramp→plateau divergence for illustrating the control law."""
    rng = np.random.default_rng(0)
    low = 0.4 + rng.normal(0, 0.03, 30)
    ramp = np.linspace(0.4, 1.4, 15)
    plateau = 1.4 + rng.normal(0, 0.03, 35)
    divergence = np.concatenate([low, ramp, plateau]).astype(np.float64)
    day = np.timedelta64(1, "D")
    timestamp = np.datetime64("2018-01-01") + np.arange(divergence.size) * day
    return divergence, timestamp


def _illustration_result(
    divergence: NDArray[np.float64],
    timestamp: NDArray[np.datetime64],
    detector: PersistenceDriftDetector,
    policy_name: str,
) -> UserRunResult:
    """Build a UserRunResult from a synthetic divergence for plotting (no ranking)."""
    states = detector.run(divergence).states
    zscore = RollingStatistics(detector.config.rolling).transform(divergence).zscore
    alpha = build_policy(POLICIES[policy_name]).alpha_series(states, zscore)
    n = divergence.size
    dummy = np.ones(n, dtype=np.int64)
    return UserRunResult(
        user=-1,
        timestamp=timestamp,
        divergence=divergence,
        state=states,
        alpha=alpha,
        rank=dummy,
        target=dummy,
        is_seen=np.zeros(n, dtype=bool),
        metrics=compute_ranking_metrics(dummy, np.zeros(n, dtype=bool), 10),
    )


def _illustrate(journal: LabJournal, detector: PersistenceDriftDetector) -> None:
    """Save α(t) control-law figures on a synthetic divergence ramp (H4)."""
    divergence, timestamp = _synthetic_divergence()
    picked = {
        name: _illustration_result(divergence, timestamp, detector, name)
        for name in ("continuous", "state-abrupt", "state-gradual")
    }
    journal.figure(
        alpha_trajectory(picked["state-gradual"]), "alpha_trajectory_gradual_synthetic"
    )
    journal.figure(
        alpha_policy_comparison(picked), "alpha_policy_comparison_synthetic"
    )


def main() -> int:
    """Run the Phase 6 adaptive-fusion comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-users", type=int, default=800)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_backbone(BASELINE_CHECKPOINT, device)
    frames, _ = read_subsample()
    detector = PersistenceDriftDetector(
        DetectorConfig.model_validate_json(Path("configs/detector.json").read_text())
    )
    evaluator = StreamingEvaluator(
        model,
        meta.architecture,
        get_signal("short_term_drift"),
        device,
        top_k=args.top_k,
    )

    config = RunConfig(
        task="phase6-adaptive-fusion",
        description="Compare α-policies (static/continuous/state abrupt|gradual)",
        params={"top_k": args.top_k, "policies": list(POLICIES)},
        dataset="amazon-subsample",
        tags=["phase6", "adaptive-fusion"],
        seed=args.seed,
    )

    with LabJournal(config) as journal:
        rng = np.random.default_rng(args.seed)
        users = frames[Split.TEST][S_USER_COLUMN].unique()
        if users.size > args.max_users:
            users = rng.choice(users, size=args.max_users, replace=False)
        user_frames = _user_frames(frames, users)
        journal.metric("users", len(user_frames), split="test", unit="users")

        rows = []
        for name, adapt_cfg in POLICIES.items():
            journal.event("adaptation", "policy_start", detail=name)
            results = _run_policy(evaluator, user_frames, detector, adapt_cfg)
            rows.append(_log_policy(journal, name, results))
        journal.table(rows, "policy_comparison")

        journal.event("adaptation", "illustrate", detail="synthetic control-law ramp")
        _illustrate(journal, detector)

        Path("configs/adaptation.json").write_text(
            POLICIES["state-gradual"].model_dump_json(indent=2), encoding="utf-8"
        )
        _summarize(journal, rows)
    return 0


def _summarize(journal: LabJournal, rows: list[dict]) -> None:
    """Write the run's narrative results."""
    by_name = {r["policy"]: r for r in rows}
    static = by_name["static-0.5"]
    gradual = by_name["state-gradual"]
    abrupt = by_name["state-abrupt"]
    cont = by_name["continuous"]
    journal.summary(
        objective="Drive α(t) from detector state and compare against static and "
        "detector-free adaptation (mechanism + no regression on natural data).",
        method="Streaming per-user eval over one backbone; α = policy(state, z). "
        "Policies: static (A), continuous (D), state abrupt/gradual (proposed).",
        findings=[
            f"NDCG@10 test: static={static['ndcg_at_k']:.4f} "
            f"continuous={cont['ndcg_at_k']:.4f} abrupt={abrupt['ndcg_at_k']:.4f} "
            f"gradual={gradual['ndcg_at_k']:.4f}",
            f"α-changes/1k: static={static['alpha_changes_per_1k']:.1f} "
            f"continuous={cont['alpha_changes_per_1k']:.1f} "
            f"gradual={gradual['alpha_changes_per_1k']:.1f}",
            f"mean α: static={static['mean_alpha']:.2f} "
            f"gradual={gradual['mean_alpha']:.2f}",
            f"confirmed_frac on test windows={gradual['confirmed_frac']:.4f} "
            "(near-zero: short natural test streams rarely confirm)",
            "synthetic control-law α(t) figures saved (gradual/abrupt/continuous)",
        ],
        limitations=[
            "Natural test data has few confirmed shifts, so headline metrics are "
            "close by design; the benefit is measured under injected drift (Phase 7).",
            "Static uses α=0.5 while state policies rest near α_stable=0.8, so the "
            "natural-data gap reflects α level more than adaptation.",
        ],
        next_steps=["Inject controlled shifts with known onsets (Phase 7)."],
    )


if __name__ == "__main__":
    raise SystemExit(main())
