#!/usr/bin/env python3
"""Second-dataset validation — does the adaptation result replicate on Books alone?

The main results mix three product categories. This script re-runs the coherent-drift
adaptation experiment on a single, distinct category (Books by default), which is a
genuinely different data distribution: a fresh backbone is trained on the Books-only
dense cohort and the per-scenario, seed-robust comparison of the CUSUM-driven policy
against static-0.8 and window-B is repeated. If the robust win over window-B replicates,
the effect is not an artefact of the particular mixed corpus.

Usage:
    python scripts/second_dataset.py --category Books --seeds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dense_adaptation import (  # noqa: E402
    FA_TARGET,
    STABLE_ALPHA,
    TOP_K,
    _calibrate,
    _prepare,
    _run_policy,
)
from dense_adaptation_seeds import (  # noqa: E402
    _diff_rows,
    _run_scenario,
    _scenario_rows,
)

from drift_reco.adaptation.policy import AdaptationConfig  # noqa: E402
from drift_reco.data.schema import Split  # noqa: E402
from drift_reco.data.subsample import (  # noqa: E402
    SubsampleBuilder,
    SubsampleConfig,
    read_subsample,
)
from drift_reco.detectors.baselines import (  # noqa: E402
    CusumDetectorConfig,
    CusumDriftDetector,
)
from drift_reco.detectors.signal import get_signal  # noqa: E402
from drift_reco.evaluation.streaming import StreamingEvaluator  # noqa: E402
from drift_reco.models.checkpoint import load_backbone, save_backbone  # noqa: E402
from drift_reco.models.config import ModelConfig, TrainConfig  # noqa: E402
from drift_reco.models.fusion import LongShortFusion  # noqa: E402
from drift_reco.models.sequences import user_sequences  # noqa: E402
from drift_reco.models.train import Trainer  # noqa: E402
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402

SCENARIOS = ("sudden", "gradual", "recurring")


def _build_and_train(args: argparse.Namespace, device: str):
    """Build the single-category dense subsample and train a fresh backbone."""
    out_dir = Path("data/interim") / f"subsample_{args.category.lower()}"
    ckpt = Path("artifacts/models") / f"backbone-{args.category.lower()}"
    cfg = SubsampleConfig(
        k_core=5, min_test_interactions=args.min_test,
        min_train_interactions=args.min_train, category=args.category,
        target_users=5000, extra_train_users=15000, out_dir=out_dir, seed=args.seed,
    )
    manifest = SubsampleBuilder(cfg).build()
    print(f"{args.category}: {manifest.selected_users} dense users, "
          f"vocab {manifest.n_item_vocab}")
    frames, manifest = read_subsample(out_dir)
    model_cfg = ModelConfig(embedding_dim=128)
    if (ckpt / "model.pt").exists() and not args.retrain:
        model, _ = load_backbone(ckpt, device)
        return model, model_cfg, frames
    model = LongShortFusion(manifest.n_item_vocab, model_cfg)
    Trainer(
        model,
        TrainConfig(epochs=args.epochs, alpha_train_mode="random",
                    negatives="uniform", seed=args.seed),
        manifest.n_item_vocab,
    ).fit(user_sequences(frames[Split.TRAIN]))
    save_backbone(model, model_cfg, manifest.n_item_vocab, ckpt)
    return model, model_cfg, frames


def main() -> int:
    """Replicate the coherent-drift adaptation comparison on a single category."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Books")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--min-test", type=int, default=12)
    parser.add_argument("--min-train", type=int, default=40)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_cfg, frames = _build_and_train(args, device)
    evaluator = StreamingEvaluator(
        model, model_cfg, get_signal("short_term_drift"), device, top_k=TOP_K
    )
    ctx_by, control_by, lookups, anchors = _prepare(
        model, model_cfg, frames, device, "coherent"
    )
    control = _run_policy(
        evaluator, CusumDriftDetector(), ctx_by, control_by,
        AdaptationConfig(policy="static", alpha_static=STABLE_ALPHA),
    )
    enter = _calibrate(control, FA_TARGET)
    detector = CusumDriftDetector(CusumDetectorConfig(enter=enter))

    config = RunConfig(
        task="second-dataset",
        description=f"Coherent-drift adaptation replicated on {args.category} alone",
        params={"category": args.category, "seeds": args.seeds},
        dataset=f"amazon-{args.category.lower()}",
        tags=["generalisation", "second-dataset", "coherent"],
        seed=args.seed,
    )

    with LabJournal(config) as journal:
        by_scenario = {
            s: _run_scenario(evaluator, detector, ctx_by, lookups, anchors,
                             frames[Split.TEST], s, args.seeds, args.seed)
            for s in SCENARIOS
        }
        rows = _scenario_rows(by_scenario)
        diffs = _diff_rows(by_scenario)
        journal.table(rows, "second_dataset_means")
        journal.table(diffs, "second_dataset_diffs")
        _summarize(journal, diffs, args.category, enter)
        _print(diffs, args.category)
    return 0


def _summarize(journal: LabJournal, diffs: list, category: str, enter: float) -> None:
    """Write the run's narrative results."""
    journal.summary(
        objective=f"Test whether the adaptation win replicates on {category} alone, a "
        "single distinct product domain rather than the mixed three-category corpus.",
        method=f"Fresh backbone on the {category}-only dense cohort; CUSUM "
        f"enter={enter:g}; per-scenario paired differences with 95% CIs over seeds.",
        findings=[
            f"[{d['scenario']}] vs window-B {d['vs_window_B']} "
            f"({'ROBUST' if d['vs_window_B_robust'] else 'ns'}); "
            f"vs static-0.8 {d['vs_static_08']} "
            f"({'ROBUST' if d['vs_static_08_robust'] else 'ns'})"
            for d in diffs
        ],
        limitations=["Single category; one backbone seed; dense cohort."],
        next_steps=["Additional categories; full-scale replication."],
    )


def _print(diffs: list, category: str) -> None:
    """Echo the replication result to stdout."""
    print(f"\nSecond-dataset replication on {category} (overall NDCG@10):")
    for d in diffs:
        w = "ROBUST" if d["vs_window_B_robust"] else "ns"
        s = "ROBUST" if d["vs_static_08_robust"] else "ns"
        print(f"  [{d['scenario']:<9}] vs window-B {d['vs_window_B']} {w:>6} | "
              f"vs static-0.8 {d['vs_static_08']} {s:>6}")


if __name__ == "__main__":
    raise SystemExit(main())
