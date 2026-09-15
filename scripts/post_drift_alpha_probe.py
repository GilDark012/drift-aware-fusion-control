#!/usr/bin/env python3
"""Diagnostic — what fixed α is optimal *after* the injected drift?

The dense adaptation experiment found that dropping α to 0.2 on drift hurt. Two
explanations: (a) the drift model favours long-term post-drift, so no α<0.8 helps; or
(b) α=0.2 is simply too aggressive and some mid-α (e.g. 0.5) is the real post-drift
optimum. This probe settles it: on the injected sudden streams it sweeps a **fixed** α
and reports quality split by pre-onset vs post-onset steps.

* If α≈0.8 maximises the *post-onset* quality → the drift model favours long-term
  (confirms the injection, not the adaptation target, is the issue).
* If some lower/mid α maximises post-onset quality → the adaptation target was mis-set.

Reuses the dense-adaptation machinery. Assumes the dense subsample and backbone already
exist (run `scripts/dense_adaptation.py` first).

Usage:
    python scripts/post_drift_alpha_probe.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dense_adaptation import (  # noqa: E402
    DENSE_CKPT,
    DENSE_DIR,
    TOP_K,
    _prepare,
    _region_metrics,
    _run_policy,
)

from drift_reco.adaptation.policy import AdaptationConfig  # noqa: E402
from drift_reco.config.constants import S_USER_COLUMN  # noqa: E402
from drift_reco.data.injection import DriftInjector, InjectionConfig  # noqa: E402
from drift_reco.data.schema import Split  # noqa: E402
from drift_reco.data.subsample import read_subsample  # noqa: E402
from drift_reco.detectors.baselines import CusumDriftDetector  # noqa: E402
from drift_reco.detectors.signal import get_signal  # noqa: E402
from drift_reco.evaluation.streaming import StreamingEvaluator  # noqa: E402
from drift_reco.models.checkpoint import load_backbone  # noqa: E402
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402

ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _rows(per_alpha: dict[float, dict]) -> list[dict]:
    """Per-α pre/post/overall NDCG rows."""
    return [
        {
            "alpha": alpha,
            "pre_ndcg": round(regions["pre"].ndcg_at_k, 5),
            "post_ndcg": round(regions["post"].ndcg_at_k, 5),
            "overall_ndcg": round(regions["all"].ndcg_at_k, 5),
        }
        for alpha, regions in per_alpha.items()
    ]


def _verdict(rows: list[dict]) -> str:
    """Interpret the post-onset α-optimum."""
    best = max(rows, key=lambda r: r["post_ndcg"])
    if best["alpha"] >= 0.6:
        return (
            f"DRIFT MODEL FAVOURS LONG-TERM: post-drift optimum is a={best['alpha']} "
            f"({best['post_ndcg']:.5f}) -- dropping alpha cannot help; fix injection."
        )
    return (
        f"ADAPTATION TARGET MIS-SET: post-drift optimum is a={best['alpha']} "
        f"({best['post_ndcg']:.5f}) -- a milder drop (toward that a) should help."
    )


def main() -> int:
    """Sweep fixed α on injected streams and split quality pre/post onset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames, _ = read_subsample(DENSE_DIR)
    model, meta = load_backbone(DENSE_CKPT, device)
    evaluator = StreamingEvaluator(
        model, meta.architecture, get_signal("short_term_drift"), device, top_k=TOP_K
    )
    ctx_by, _control_by, lookups, anchors = _prepare(
        model, meta.architecture, frames, device
    )
    inj_cfg = InjectionConfig(
        drift_type="sudden", n_users=5000, seed=args.seed, min_test_len=8
    )
    injected, onsets = DriftInjector(inj_cfg, lookups).inject(
        frames[Split.TEST], anchors
    )
    inj_by = dict(iter(injected.groupby(S_USER_COLUMN)))
    onset_by_user = {int(o.user): int(o.onset_index) for o in onsets}

    config = RunConfig(
        task="post-drift-alpha-probe",
        description="Fixed-α sweep on injected streams, split pre/post onset (dense)",
        params={"alphas": list(ALPHAS), "n_injected": len(onset_by_user)},
        dataset="amazon-subsample-dense",
        tags=["diagnostic", "alpha-sweep", "post-drift"],
        seed=args.seed,
    )

    detector = CusumDriftDetector()
    with LabJournal(config) as journal:
        per_alpha = {}
        for alpha in ALPHAS:
            results = _run_policy(
                evaluator, detector, ctx_by, inj_by,
                AdaptationConfig(policy="static", alpha_static=alpha),
            )
            per_alpha[alpha] = _region_metrics(results, onset_by_user)
        rows = _rows(per_alpha)
        journal.table(rows, "post_drift_alpha_sweep")
        verdict = _verdict(rows)
        _summarize(journal, rows, verdict)
        _print(rows, verdict)
    return 0


def _summarize(journal: LabJournal, rows: list[dict], verdict: str) -> None:
    """Write the run's narrative results."""
    journal.summary(
        objective="Determine the post-drift-optimal fixed α to distinguish a "
        "long-term-favouring drift model from a mis-set adaptation target.",
        method="Sweep fixed α on the injected sudden streams; split pooled NDCG@10 "
        "into pre-onset and post-onset steps.",
        findings=[verdict] + [
            f"a={r['alpha']}: pre={r['pre_ndcg']:.5f} post={r['post_ndcg']:.5f}"
            for r in rows
        ],
        limitations=["Dense cohort; single seed; sudden scenario."],
        next_steps=["Fix the drift model, or accept and narrow the thesis."],
    )


def _print(rows: list[dict], verdict: str) -> None:
    """Echo the sweep and verdict to stdout."""
    print("\nfixed-alpha sweep on injected streams (dense):")
    print("  alpha   pre       post")
    for r in rows:
        print(f"  {r['alpha']:<6} {r['pre_ndcg']:.5f}   {r['post_ndcg']:.5f}")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    raise SystemExit(main())
