#!/usr/bin/env python3
"""Phase 7 — inject controlled preference shifts with known onsets.

For each scenario (sudden/gradual/recurring) this writes an injected copy of the
subsample test split plus a ground-truth onset file, then sanity-checks that the
injected shift is detectable: it streams the injected users and reports detection
latency against the known onset, and streams the same users' *control* (un-injected)
stream to report the false-alarm rate. Replaces the broken pre-generated
test_drifted.parquet (Phase-2 Risk C).

Usage:
    python scripts/inject_drift.py --n-users 300 --sanity-users 150
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
    S_ITEM_COLUMN,
    S_USER_COLUMN,
    SUBSAMPLE_DIR,
    TIMESTAMP_COLUMN,
)
from drift_reco.data.injection import (  # noqa: E402
    DriftInjector,
    InjectedOnset,
    InjectionConfig,
    ItemLookups,
    item_meta,
    popular_candidates,
)
from drift_reco.data.schema import Split  # noqa: E402
from drift_reco.data.subsample import read_subsample  # noqa: E402
from drift_reco.detectors.detector import (  # noqa: E402
    DetectorConfig,
    PersistenceDriftDetector,
)
from drift_reco.detectors.signal import get_signal  # noqa: E402
from drift_reco.evaluation.latency import (  # noqa: E402
    detection_latency,
    summarize_detection,
)
from drift_reco.evaluation.plots import injection_overlay  # noqa: E402
from drift_reco.evaluation.streaming import StreamingEvaluator  # noqa: E402
from drift_reco.models.checkpoint import load_backbone  # noqa: E402
from drift_reco.models.config import ModelConfig  # noqa: E402
from drift_reco.models.fusion import LongShortFusion  # noqa: E402
from drift_reco.models.sequences import build_examples  # noqa: E402
from drift_reco.models.tensors import to_device  # noqa: E402
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402

DRIFT_TYPES = ("sudden", "gradual", "recurring")
_ANCHOR_RECENT = 30


def _candidate_reps(
    model: LongShortFusion, candidates: np.ndarray, k: int, device: str
) -> np.ndarray:
    """Short-term representation of each candidate as a repeated-item window."""
    windows = to_device(np.repeat(candidates[:, None], k, axis=1), torch.device(device))
    lengths = torch.full((candidates.size,), k)
    with torch.no_grad():
        return model.short_term(windows, lengths).cpu().numpy().astype(np.float64)


def _anchors(
    model: LongShortFusion,
    cfg: ModelConfig,
    context_by_user: dict[int, pd.DataFrame],
    target_users: set[int],
    device: str,
) -> dict[int, np.ndarray]:
    """Per-user established short-term profile: mean S over recent context windows."""
    first = max(cfg.min_context, cfg.short_window + 1)
    anchors: dict[int, np.ndarray] = {}
    for user in target_users:
        frame = context_by_user.get(user)
        if frame is None:
            continue
        seq = frame.sort_values(TIMESTAMP_COLUMN)[S_ITEM_COLUMN].to_numpy(np.int64)
        if seq.size <= first:
            continue
        positions = np.arange(max(first, seq.size - _ANCHOR_RECENT), seq.size)
        examples = build_examples(seq, positions, cfg)
        with torch.no_grad():
            reps = model.short_term(
                to_device(examples.short_items, torch.device(device)),
                to_device(examples.short_len, torch.device(device)),
            )
        anchors[int(user)] = reps.mean(dim=0).cpu().numpy().astype(np.float64)
    return anchors


def _save(injected: pd.DataFrame, onsets: list[InjectedOnset], drift_type: str) -> Path:
    """Persist an injected test split and its ground-truth onsets."""
    out = Path(SUBSAMPLE_DIR)
    injected.to_parquet(out / f"injected_{drift_type}.parquet", index=False)
    (out / f"onsets_{drift_type}.json").write_text(
        json.dumps([o.model_dump() for o in onsets], indent=2), encoding="utf-8"
    )
    return out / f"injected_{drift_type}.parquet"


def _sanity(
    evaluator: StreamingEvaluator,
    detector: PersistenceDriftDetector,
    context_by_user: dict[int, pd.DataFrame],
    injected: pd.DataFrame,
    control: pd.DataFrame,
    onsets: list[InjectedOnset],
    limit: int,
) -> tuple[list, tuple | None, int]:
    """Detection latency on injected users and false-alarm count on their control."""
    policy = build_policy(AdaptationConfig(policy="static", alpha_static=0.8))
    inj_by = dict(iter(injected.groupby(S_USER_COLUMN)))
    ctl_by = dict(iter(control.groupby(S_USER_COLUMN)))
    records, false_alarms, seen, best = [], 0, 0, None
    for onset in onsets[:limit]:
        context = context_by_user.get(onset.user)
        if context is None or onset.user not in inj_by:
            continue
        result = evaluator.run_user(context, inj_by[onset.user], detector, policy)
        if result is None:
            continue
        record = detection_latency(onset.user, result.state, onset.onset_index)
        records.append(record)
        seen += 1
        control_result = evaluator.run_user(
            context, ctl_by[onset.user], detector, policy
        )
        if control_result is not None:
            false_alarms += int((control_result.state == 2).any())
        if best is None and record.detected:
            best = (result, onset.onset_index)
    return records, best, (false_alarms, seen)


def _build(args: argparse.Namespace) -> tuple:
    """Load data, backbone, detector and evaluator."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_backbone(BASELINE_CHECKPOINT, device)
    frames, _ = read_subsample()
    detector = PersistenceDriftDetector(
        DetectorConfig.model_validate_json(
            Path("configs/detector_streaming.json").read_text()
        )
    )
    evaluator = StreamingEvaluator(
        model,
        meta.architecture,
        get_signal("short_term_drift"),
        device,
        top_k=args.top_k,
    )
    return frames, model, meta, detector, evaluator, device


def main() -> int:
    """Inject all three scenarios, persist them, and sanity-check detectability."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-users", type=int, default=300)
    parser.add_argument("--sanity-users", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames, model, meta, detector, evaluator, device = _build(args)
    k = meta.architecture.short_window
    context = pd.concat([frames[Split.TRAIN], frames[Split.VAL]], ignore_index=True)
    context_by_user = dict(iter(context.groupby(S_USER_COLUMN)))
    candidates = popular_candidates(frames[Split.TRAIN], pool_size=200)
    lookups = ItemLookups(
        meta=item_meta(frames[Split.TRAIN]),
        candidates=candidates,
        candidate_reps=_candidate_reps(model, candidates, k, device),
    )
    test_users = {int(u) for u in frames[Split.TEST][S_USER_COLUMN].unique()}
    anchors = _anchors(model, meta.architecture, context_by_user, test_users, device)

    config = RunConfig(
        task="phase7-inject-drift",
        description="Controlled sudden/gradual/recurring preference shifts + sanity",
        params={"n_users": args.n_users, "types": list(DRIFT_TYPES)},
        dataset="amazon-subsample",
        tags=["phase7", "injection"],
        seed=args.seed,
    )

    with LabJournal(config) as journal:
        summaries = {}
        for drift_type in DRIFT_TYPES:
            inj_cfg = InjectionConfig(
                drift_type=drift_type, n_users=args.n_users, seed=args.seed
            )
            injected, onsets = DriftInjector(inj_cfg, lookups).inject(
                frames[Split.TEST], anchors
            )
            path = _save(injected, onsets, drift_type)
            journal.event("injection", "saved", detail=f"{drift_type} -> {path.name}")
            journal.metric(
                "injected_users", len(onsets), context=drift_type, unit="users"
            )

            records, best, (false_alarms, seen) = _sanity(
                evaluator, detector, context_by_user, injected,
                frames[Split.TEST], onsets, args.sanity_users,
            )
            summary = summarize_detection(records)
            summaries[drift_type] = (summary, false_alarms, seen)
            journal.metric("detection_rate", summary.detection_rate, context=drift_type)
            journal.metric(
                "false_alarm_rate", false_alarms / max(seen, 1), context=drift_type
            )
            if summary.median_latency is not None:
                journal.metric(
                    "median_detection_latency", summary.median_latency,
                    context=drift_type, unit="interactions",
                )
            if best is not None:
                journal.figure(
                    injection_overlay(best[0], best[1]), f"injection_{drift_type}"
                )

        _summarize(journal, summaries)
    return 0


def _summarize(journal: LabJournal, summaries: dict) -> None:
    """Write the run's narrative results."""
    findings = []
    for drift_type, (summary, false_alarms, seen) in summaries.items():
        latency = (
            f"{summary.median_latency:.0f}"
            if summary.median_latency is not None
            else "n/a"
        )
        findings.append(
            f"{drift_type}: detection_rate={summary.detection_rate:.2f}, "
            f"median_latency={latency}, false_alarm={false_alarms}/{seen}"
        )
    journal.summary(
        objective="Create reproducible preference-sequence shifts with known onsets "
        "and confirm they are detectable, replacing the broken drifted parquet.",
        method="Switch post-onset items to a cluster whose short-term rep is most "
        "opposed to the user's established profile; sudden/gradual/recurring; stream "
        "injected vs control and measure detection latency and false alarms.",
        findings=findings
        + ["injected splits + onset ground truth saved under data/interim/subsample/"],
        limitations=[
            "Strong, clean synthetic shifts (anti-profile cluster); real drift is "
            "weaker/messier. Detection is bounded by the subsample backbone.",
            "Sanity uses static α; full latency/recovery comparison is Phase 8.",
        ],
        next_steps=[
            "Phase 8: control vs injected, all policies; detect/adapt/recover.",
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
