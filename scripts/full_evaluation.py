#!/usr/bin/env python3
"""Phase 8 — full evaluation: control vs injected, baselines vs proposed.

For each controlled scenario, replays every injected user's stream under each
α-policy and aligns per-step hit@k to the known onset. Reports the recovery curve
and its summary (pre-drift quality, degradation, recovery latency) on both the full
injected cohort and the *detected* subset (users the detector confirmed), plus
detection latency and control-stream stability. First-version framing: the drift-aware
advantage is shown where drift is detected, alongside honest full-cohort numbers.

Usage:
    python scripts/full_evaluation.py --max-users 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift_reco.adaptation.policy import AdaptationConfig, build_policy  # noqa: E402
from drift_reco.config.constants import (  # noqa: E402
    BASELINE_CHECKPOINT,
    S_USER_COLUMN,
    SUBSAMPLE_DIR,
)
from drift_reco.data.schema import Split  # noqa: E402
from drift_reco.data.subsample import read_subsample  # noqa: E402
from drift_reco.detectors.detector import (  # noqa: E402
    DetectorConfig,
    DriftState,  # noqa: E402
    PersistenceDriftDetector,
)
from drift_reco.detectors.signal import get_signal  # noqa: E402
from drift_reco.evaluation.latency import (  # noqa: E402
    detection_latency,
    summarize_detection,
)
from drift_reco.evaluation.plots import (  # noqa: E402
    detection_tradeoff,
    recovery_curves,
)
from drift_reco.evaluation.recovery import (  # noqa: E402
    build_recovery_curve,
    summarize_recovery,
)
from drift_reco.evaluation.streaming import (  # noqa: E402
    StreamingEvaluator,
    UserRunResult,
)
from drift_reco.models.checkpoint import load_backbone  # noqa: E402
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402

SCENARIOS = ("sudden", "gradual", "recurring")
POLICIES = {
    "static-A": AdaptationConfig(policy="static", alpha_static=0.8),
    "window-B": AdaptationConfig(policy="static", alpha_static=0.0),
    "continuous-D": AdaptationConfig(policy="continuous"),
    "proposed": AdaptationConfig(policy="state", mode="gradual"),
}
TOP_K = 10


def _run_policy(
    evaluator: StreamingEvaluator,
    detector: PersistenceDriftDetector,
    ctx_by: dict[int, pd.DataFrame],
    inj_by: dict[int, pd.DataFrame],
    onset_by_user: dict[int, int],
    policy_cfg: AdaptationConfig,
) -> list[UserRunResult]:
    """Stream every injected user under one α-policy."""
    policy = build_policy(policy_cfg)
    out = []
    for user in onset_by_user:
        if user not in ctx_by or user not in inj_by:
            continue
        result = evaluator.run_user(ctx_by[user], inj_by[user], detector, policy)
        if result is not None:
            out.append(result)
    return out


def _detected_users(
    results: list[UserRunResult], onset_by_user: dict[int, int]
) -> set[int]:
    """Users whose proposed run confirmed drift at or after the onset."""
    detected = set()
    for result in results:
        onset = onset_by_user.get(result.user)
        if onset is None:
            continue
        confirmed = np.where(result.state == int(DriftState.CONFIRMED))[0]
        if confirmed.size and confirmed.max() >= onset:
            detected.add(result.user)
    return detected


def _curve_and_summary(results, onset_by_user, users=None):
    """Build a recovery curve + summary, optionally restricted to ``users``."""
    subset = [r for r in results if users is None or r.user in users]
    curve = build_recovery_curve(subset, onset_by_user, TOP_K)
    return curve, summarize_recovery(curve, n_users=len(subset))


def _load_scenario(scenario: str) -> tuple[dict[int, pd.DataFrame], dict[int, int]]:
    """Load an injected split and its onset map."""
    root = Path(SUBSAMPLE_DIR)
    injected = pd.read_parquet(root / f"injected_{scenario}.parquet")
    onsets = json.loads((root / f"onsets_{scenario}.json").read_text())
    inj_by = dict(iter(injected.groupby(S_USER_COLUMN)))
    onset_by_user = {int(o["user"]): int(o["onset_index"]) for o in onsets}
    return inj_by, onset_by_user


def _evaluate_scenario(
    journal: LabJournal,
    evaluator: StreamingEvaluator,
    detector: PersistenceDriftDetector,
    ctx_by: dict[int, pd.DataFrame],
    scenario: str,
) -> list[dict]:
    """Run all policies on one scenario and log curves, summaries, detection."""
    inj_by, onset_by_user = _load_scenario(scenario)
    per_policy = {
        name: _run_policy(evaluator, detector, ctx_by, inj_by, onset_by_user, cfg)
        for name, cfg in POLICIES.items()
    }
    detected = _detected_users(per_policy["proposed"], onset_by_user)

    curves_full, curves_det, rows = {}, {}, []
    for name, results in per_policy.items():
        curve_f, sum_f = _curve_and_summary(results, onset_by_user)
        curve_d, sum_d = _curve_and_summary(results, onset_by_user, detected)
        curves_full[name], curves_det[name] = curve_f, curve_d
        journal.metric("recovery_latency_full", _lat(sum_f.recovery_latency),
                       context=f"{scenario}/{name}")
        journal.metric(
            "degradation_detected", sum_d.degradation,
            context=f"{scenario}/{name}",
        )
        rows.append({
            "scenario": scenario, "policy": name,
            "pre_full": round(sum_f.pre_quality, 4),
            "min_full": round(sum_f.min_quality, 4),
            "recov_full": _lat(sum_f.recovery_latency),
            "pre_det": round(sum_d.pre_quality, 4),
            "min_det": round(sum_d.min_quality, 4),
            "degr_det": round(sum_d.degradation, 4),
            "recov_det": _lat(sum_d.recovery_latency),
            "n_detected": len(detected),
        })

    journal.figure(recovery_curves(curves_full), f"recovery_full_{scenario}")
    if detected:
        journal.figure(recovery_curves(curves_det), f"recovery_detected_{scenario}")

    det_records = [
        detection_latency(r.user, r.state, onset_by_user[r.user])
        for r in per_policy["proposed"] if r.user in onset_by_user
    ]
    det = summarize_detection(det_records)
    journal.metric("detection_rate", det.detection_rate, context=scenario)
    journal.metric("detected_users", len(detected), context=scenario, unit="users")
    return rows


def _lat(value: int | None) -> float:
    """Encode a recovery/detection latency (None -> NaN) for logging."""
    return float(value) if value is not None else float("nan")


ENTER_SWEEP = (1.5, 1.75, 2.0, 2.5, 3.0)


def _confirmed_after(divergence: np.ndarray, onset: int, cfg: DetectorConfig) -> bool:
    """Whether the detector confirms drift at or after ``onset`` on a series."""
    states = PersistenceDriftDetector(cfg).run(divergence).states
    confirmed = np.where(states == int(DriftState.CONFIRMED))[0]
    return bool(confirmed.size and confirmed.max() >= onset)


def _tradeoff(
    journal: LabJournal,
    base_cfg: DetectorConfig,
    proposed: list[UserRunResult],
    control: list[UserRunResult],
    onset_by_user: dict[int, int],
) -> None:
    """Sweep the detector threshold: detection (injected) vs false alarm (control)."""
    inj = [
        (r.divergence, onset_by_user[r.user])
        for r in proposed
        if r.user in onset_by_user
    ]
    ctl = [r.divergence for r in control]
    points = []
    for enter in ENTER_SWEEP:
        cfg = base_cfg.model_copy(update={"enter": enter})
        det_rate = (
            np.mean([_confirmed_after(d, o, cfg) for d, o in inj]) if inj else 0.0
        )
        fa_rate = np.mean([_confirmed_after(d, 0, cfg) for d in ctl]) if ctl else 0.0
        points.append((enter, float(det_rate), float(fa_rate)))
        journal.metric("detection_rate", float(det_rate), step=int(enter * 100),
                       context=f"tradeoff/enter={enter}")
        journal.metric("false_alarm_rate", float(fa_rate), step=int(enter * 100),
                       context=f"tradeoff/enter={enter}")
    journal.figure(detection_tradeoff(points), "detection_tradeoff_sudden")


def main() -> int:
    """Run the Phase 8 evaluation across scenarios and policies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_backbone(BASELINE_CHECKPOINT, device)
    frames, _ = read_subsample()
    context = pd.concat([frames[Split.TRAIN], frames[Split.VAL]], ignore_index=True)
    ctx_by = dict(iter(context.groupby(S_USER_COLUMN)))
    detector = PersistenceDriftDetector(
        DetectorConfig.model_validate_json(
            Path("configs/detector_streaming.json").read_text()
        )
    )
    evaluator = StreamingEvaluator(
        model, meta.architecture, get_signal("short_term_drift"), device, top_k=TOP_K
    )

    config = RunConfig(
        task="phase8-full-evaluation",
        description="Control vs injected; baselines A/B/D vs proposed; recovery curves",
        params={
            "policies": list(POLICIES),
            "scenarios": list(SCENARIOS),
            "top_k": TOP_K,
        },
        dataset="amazon-subsample",
        tags=["phase8", "evaluation"],
        seed=args.seed,
    )

    base_cfg = detector.config
    control_by = dict(iter(frames[Split.TEST].groupby(S_USER_COLUMN)))
    with LabJournal(config) as journal:
        rows: list[dict] = []
        for scenario in SCENARIOS:
            journal.event("evaluation", "scenario_start", detail=scenario)
            rows.extend(
                _evaluate_scenario(journal, evaluator, detector, ctx_by, scenario)
            )
        journal.table(rows, "policy_scenario_summary")

        # Detector operating characteristic on the sudden scenario.
        inj_by, onset_by_user = _load_scenario("sudden")
        proposed = _run_policy(
            evaluator, detector, ctx_by, inj_by, onset_by_user, POLICIES["proposed"]
        )
        control = _run_policy(
            evaluator, detector, ctx_by, control_by, onset_by_user, POLICIES["static-A"]
        )
        _tradeoff(journal, base_cfg, proposed, control, onset_by_user)
        _summarize(journal, rows)
    return 0


def _summarize(journal: LabJournal, rows: list[dict]) -> None:
    """Write the run's narrative results."""
    sudden = {r["policy"]: r for r in rows if r["scenario"] == "sudden"}
    findings = [
        "[sudden, detected subset] degradation: "
        + ", ".join(f"{p}={sudden[p]['degr_det']:.3f}" for p in POLICIES),
        "[sudden, detected subset] recovery latency: "
        + ", ".join(f"{p}={sudden[p]['recov_det']}" for p in POLICIES),
        f"detected users (sudden): {sudden['proposed']['n_detected']}",
    ]
    journal.summary(
        objective="Quantify whether drift-aware fusion recovers quality faster than "
        "static/window/continuous baselines after a known preference shift.",
        method="Stream injected users under each α-policy; align hit@10 to the onset; "
        "recovery curves + summaries on full cohort and detected subset; 3 scenarios.",
        findings=findings,
        limitations=[
            "Detection recall is low on this backbone (Phase-7 finding), so the "
            "full-cohort effect is diluted; the detected subset shows the mechanism.",
            "Subsample scale; single seed for the streaming pass.",
        ],
        next_steps=[
            "Trade-off curve (detection vs false alarm); ablations (signal, gradual "
            "vs abrupt, persistence); stronger/full-scale backbone.",
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
