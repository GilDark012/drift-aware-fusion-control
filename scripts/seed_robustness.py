#!/usr/bin/env python3
"""P1.2 — seed robustness: detection/false-alarm with confidence intervals.

The supervisor's review asks for Monte-Carlo repetition with confidence intervals
rather than a single-seed headline. The backbone is a *frozen* checkpoint and the
streaming pass is deterministic given the injected data, so the meaningful source of
randomness is the **controlled-drift injection**: which users are shifted and which
anti-profile items they are shifted onto (``InjectionConfig.seed``). This script
re-injects the *sudden* scenario under many seeds and reports, for each detector
operating point, the mean and 95 % confidence interval of the detection rate, the
false-alarm rate and the median detection latency.

Backbone-initialisation variance (retraining the GRU under different seeds) is a
separate, much heavier axis — noted as a limitation, not run here.

Usage:
    python scripts/seed_robustness.py --seeds 10 --n-users 300
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Reuse Phase-7's anchor/candidate machinery so the injection matches exactly.
from inject_drift import _anchors, _candidate_reps  # noqa: E402

from drift_reco.adaptation.policy import AdaptationConfig, build_policy  # noqa: E402
from drift_reco.config.constants import (  # noqa: E402
    BASELINE_CHECKPOINT,
    S_USER_COLUMN,
)
from drift_reco.data.injection import (  # noqa: E402
    DriftInjector,
    InjectionConfig,
    ItemLookups,
    item_meta,
    popular_candidates,
)
from drift_reco.data.schema import Split  # noqa: E402
from drift_reco.data.subsample import read_subsample  # noqa: E402
from drift_reco.detectors.detector import (  # noqa: E402
    DetectorConfig,
    DriftState,
    PersistenceDriftDetector,
)
from drift_reco.detectors.signal import get_signal  # noqa: E402
from drift_reco.evaluation.streaming import (  # noqa: E402
    StreamingEvaluator,
)
from drift_reco.models.checkpoint import load_backbone  # noqa: E402
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402

SCENARIO = "sudden"
Z_OPERATING_POINTS = (1.75, 2.0)
_ANCHOR_POOL = 200


def _confirmed_after(
    divergence: np.ndarray, onset: int, cfg: DetectorConfig
) -> tuple[bool, int | None]:
    """Whether drift is confirmed at/after ``onset`` and its detection latency."""
    states = PersistenceDriftDetector(cfg).run(divergence).states
    confirmed = np.where(states == int(DriftState.CONFIRMED))[0]
    hits = confirmed[confirmed >= onset]
    if hits.size == 0:
        return False, None
    return True, int(hits[0] - onset)


def _stream_seed(
    evaluator: StreamingEvaluator,
    detector: PersistenceDriftDetector,
    ctx_by: dict[int, pd.DataFrame],
    control_by: dict[int, pd.DataFrame],
    injected: pd.DataFrame,
    onset_by_user: dict[int, int],
) -> tuple[list[tuple[np.ndarray, int]], list[np.ndarray]]:
    """Return (injected divergence, onset) pairs and control divergence series."""
    policy = build_policy(AdaptationConfig(policy="static", alpha_static=0.8))
    inj_by = dict(iter(injected.groupby(S_USER_COLUMN)))
    inj_series: list[tuple[np.ndarray, int]] = []
    ctl_series: list[np.ndarray] = []
    for user, onset in onset_by_user.items():
        ctx = ctx_by.get(user)
        if ctx is None or user not in inj_by:
            continue
        result = evaluator.run_user(ctx, inj_by[user], detector, policy)
        if result is not None:
            inj_series.append((result.divergence, onset))
        control = control_by.get(user)
        if control is not None:
            ctl_result = evaluator.run_user(ctx, control, detector, policy)
            if ctl_result is not None:
                ctl_series.append(ctl_result.divergence)
    return inj_series, ctl_series


def _seed_metrics(
    inj_series: list[tuple[np.ndarray, int]],
    ctl_series: list[np.ndarray],
    base_cfg: DetectorConfig,
    enter: float,
) -> dict[str, float]:
    """Detection rate, false-alarm rate and median latency at one operating point."""
    cfg = base_cfg.model_copy(update={"enter": enter})
    detections, latencies = [], []
    for divergence, onset in inj_series:
        hit, lat = _confirmed_after(divergence, onset, cfg)
        detections.append(hit)
        if lat is not None:
            latencies.append(lat)
    false_alarms = [_confirmed_after(d, 0, cfg)[0] for d in ctl_series]
    return {
        "detection_rate": float(np.mean(detections)) if detections else 0.0,
        "false_alarm_rate": float(np.mean(false_alarms)) if false_alarms else 0.0,
        "median_latency": float(np.median(latencies)) if latencies else float("nan"),
        "n_injected": float(len(inj_series)),
    }


def _ci95(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, half-width of the 95 % CI, std) for a sample."""
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, float("nan"), 0.0
    std = float(arr.std(ddof=1))
    half = 1.96 * std / math.sqrt(arr.size)
    return mean, half, std


def _build() -> tuple:
    """Load the frozen backbone, subsample, detector, evaluator and lookups."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_backbone(BASELINE_CHECKPOINT, device)
    frames, _ = read_subsample()
    detector = PersistenceDriftDetector(
        DetectorConfig.model_validate_json(
            Path("configs/detector_streaming.json").read_text()
        )
    )
    evaluator = StreamingEvaluator(
        model, meta.architecture, get_signal("short_term_drift"), device, top_k=10
    )
    return frames, model, meta, detector, evaluator, device


def _prepare(frames: dict, model, meta, device: str) -> tuple:
    """Build the per-user context, anti-profile candidates and anchors (seed-free)."""
    context = pd.concat([frames[Split.TRAIN], frames[Split.VAL]], ignore_index=True)
    ctx_by = dict(iter(context.groupby(S_USER_COLUMN)))
    control_by = dict(iter(frames[Split.TEST].groupby(S_USER_COLUMN)))
    candidates = popular_candidates(frames[Split.TRAIN], pool_size=_ANCHOR_POOL)
    lookups = ItemLookups(
        meta=item_meta(frames[Split.TRAIN]),
        candidates=candidates,
        candidate_reps=_candidate_reps(
            model, candidates, meta.architecture.short_window, device
        ),
    )
    test_users = {int(u) for u in frames[Split.TEST][S_USER_COLUMN].unique()}
    anchors = _anchors(model, meta.architecture, ctx_by, test_users, device)
    return ctx_by, control_by, lookups, anchors


def _run_seeds(args: argparse.Namespace, ctx: tuple, base_cfg: DetectorConfig) -> dict:
    """Collect per-seed metrics at every operating point."""
    frames, detector, evaluator, ctx_by, control_by, lookups, anchors = ctx
    per_point: dict[float, list[dict[str, float]]] = {z: [] for z in Z_OPERATING_POINTS}
    for seed in range(args.seed, args.seed + args.seeds):
        inj_cfg = InjectionConfig(
            drift_type=SCENARIO, n_users=args.n_users, seed=seed
        )
        injected, onsets = DriftInjector(inj_cfg, lookups).inject(
            frames[Split.TEST], anchors
        )
        onset_by_user = {int(o.user): int(o.onset_index) for o in onsets}
        inj_series, ctl_series = _stream_seed(
            evaluator, detector, ctx_by, control_by, injected, onset_by_user
        )
        for z in Z_OPERATING_POINTS:
            per_point[z].append(_seed_metrics(inj_series, ctl_series, base_cfg, z))
    return per_point


def _rows(per_point: dict, n_seeds: int) -> list[dict]:
    """Aggregate per-seed metrics into a mean ± CI table."""
    rows = []
    for z, seed_metrics in per_point.items():
        det = [m["detection_rate"] for m in seed_metrics]
        fa = [m["false_alarm_rate"] for m in seed_metrics]
        lat = [m["median_latency"] for m in seed_metrics]
        det_m, det_h, det_s = _ci95(det)
        fa_m, fa_h, _ = _ci95(fa)
        lat_m, lat_h, _ = _ci95(lat)
        rows.append({
            "z_enter": z,
            "n_seeds": n_seeds,
            "detection_mean": round(det_m, 4),
            "detection_ci95": round(det_h, 4),
            "detection_std": round(det_s, 4),
            "false_alarm_mean": round(fa_m, 4),
            "false_alarm_ci95": round(fa_h, 4),
            "median_latency_mean": round(lat_m, 2),
            "median_latency_ci95": round(lat_h, 2),
        })
    return rows


def main() -> int:
    """Run the seed-robustness sweep and log a mean ± CI table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-users", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames, model, meta, detector, evaluator, device = _build()
    ctx_by, control_by, lookups, anchors = _prepare(frames, model, meta, device)
    base_cfg = detector.config
    bundle = (frames, detector, evaluator, ctx_by, control_by, lookups, anchors)

    config = RunConfig(
        task="p1-2-seed-robustness",
        description="Detection/false-alarm CIs over injection seeds (sudden)",
        params={"seeds": args.seeds, "n_users": args.n_users,
                "operating_points": list(Z_OPERATING_POINTS)},
        dataset="amazon-subsample",
        tags=["p1-2", "seeds", "confidence-interval"],
        seed=args.seed,
    )

    with LabJournal(config) as journal:
        per_point = _run_seeds(args, bundle, base_cfg)
        rows = _rows(per_point, args.seeds)
        journal.table(rows, "seed_robustness")
        for row in rows:
            journal.metric("detection_rate_mean", row["detection_mean"],
                           step=int(row["z_enter"] * 100),
                           context=f"z={row['z_enter']}")
            journal.metric("detection_rate_ci95", row["detection_ci95"],
                           step=int(row["z_enter"] * 100),
                           context=f"z={row['z_enter']}")
        _summarize(journal, rows)
        _print(rows)
    return 0


def _summarize(journal: LabJournal, rows: list[dict]) -> None:
    """Write the run's narrative results."""
    findings = [
        f"z={r['z_enter']}: detection {r['detection_mean']:.2f} "
        f"±{r['detection_ci95']:.2f}, false-alarm {r['false_alarm_mean']:.3f} "
        f"±{r['false_alarm_ci95']:.3f} (n={r['n_seeds']} seeds)"
        for r in rows
    ]
    journal.summary(
        objective="Replace the single-seed detection headline with a Monte-Carlo "
        "mean and 95% confidence interval over controlled-drift injection seeds.",
        method="Re-inject the sudden scenario under N seeds on the frozen backbone; "
        "at each detector operating point compute detection rate, false-alarm rate "
        "and median latency per seed, then aggregate to mean ± 95% CI.",
        findings=findings,
        limitations=[
            "Randomness axis is the injection sampling only; backbone-initialisation "
            "variance (retraining the GRU) is a separate, heavier axis, not run here.",
            "Sudden scenario; gradual/recurring can be added with --scenario later.",
        ],
        next_steps=["Confront window-B pre-drift stability cost (P1.4)."],
    )


def _print(rows: list[dict]) -> None:
    """Echo the table to stdout for quick inspection."""
    print("\nSeed robustness (sudden scenario):")
    for r in rows:
        print(
            f"  z={r['z_enter']}: detection={r['detection_mean']:.3f}"
            f" ±{r['detection_ci95']:.3f} | "
            f"false-alarm={r['false_alarm_mean']:.3f} ±{r['false_alarm_ci95']:.3f}"
            f" | median-latency={r['median_latency_mean']:.1f}"
            f" (n={r['n_seeds']})"
        )


if __name__ == "__main__":
    raise SystemExit(main())
