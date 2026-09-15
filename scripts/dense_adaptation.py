#!/usr/bin/env python3
"""The positive-result experiment — CUSUM-driven adaptive fusion on dense users.

Brings the pieces together on the cohort where long-term memory helps (dense, long
histories — see ``docs/dense_cohort_probe.md``):

1. train a fresh backbone on the dense cohort;
2. inject a controlled *sudden* preference shift with known onset;
3. drive the fusion weight α with an **off-the-shelf CUSUM detector** (STABLE→α=0.8,
   CONFIRMED→α=0.2, gradual glide) — detection is a solved component, adaptation is the
   contribution;
4. compare the adaptive policy against **static-α=0.8** (never adapts) and **window-B**
   (pure short-term, α=0) on quality *before* and *after* the drift.

Success: adaptive-α beats **both** baselines on overall quality — good when stable
(uses long-term memory) *and* after the shift (leans on short-term).

Usage:
    python scripts/dense_adaptation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from inject_drift import _anchors, _candidate_reps  # noqa: E402

from drift_reco.adaptation.policy import AdaptationConfig, build_policy  # noqa: E402
from drift_reco.config.constants import S_USER_COLUMN  # noqa: E402
from drift_reco.data.injection import (  # noqa: E402
    DriftInjector,
    InjectionConfig,
    ItemLookups,
    item_meta,
    mid_frequency_candidates,
    popular_candidates,
)
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
from drift_reco.detectors.detector import DriftState  # noqa: E402
from drift_reco.detectors.signal import get_signal  # noqa: E402
from drift_reco.evaluation.metrics import (  # noqa: E402
    RankingMetrics,
    compute_ranking_metrics,
)
from drift_reco.evaluation.plots import recovery_curves  # noqa: E402
from drift_reco.evaluation.recovery import (  # noqa: E402
    build_recovery_curve,
    summarize_recovery,
)
from drift_reco.evaluation.streaming import (  # noqa: E402
    StreamingEvaluator,
    UserRunResult,
)
from drift_reco.models.checkpoint import load_backbone, save_backbone  # noqa: E402
from drift_reco.models.config import ModelConfig, TrainConfig  # noqa: E402
from drift_reco.models.fusion import LongShortFusion  # noqa: E402
from drift_reco.models.sequences import user_sequences  # noqa: E402
from drift_reco.models.train import Trainer  # noqa: E402
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402

DENSE_DIR = Path("data/interim/subsample_dense")
DENSE_CKPT = Path("artifacts/models/dense-backbone")
ENTER_GRID = (2.0, 1.5, 1.0, 0.75, 0.5, 0.3, 0.2, 0.1)
FA_TARGET = 0.10
TOP_K = 10
STABLE_ALPHA = 0.8


def _build_dense(args: argparse.Namespace) -> None:
    """Build the dense long-history cohort to its own directory."""
    cfg = SubsampleConfig(
        k_core=5, min_test_interactions=args.min_test,
        min_train_interactions=args.min_train, target_users=5000,
        extra_train_users=15000, out_dir=DENSE_DIR, seed=args.seed,
    )
    m = SubsampleBuilder(cfg).build()
    print(f"dense cohort: {m.selected_users} users, vocab {m.n_item_vocab}")


def _ckpt(mode: str) -> Path:
    """Mode-specific dense backbone checkpoint directory."""
    return DENSE_CKPT if mode == "ewma" else DENSE_CKPT.with_name(f"dense-{mode}")


def _train_or_load(
    frames: dict, n_items: int, args: argparse.Namespace
) -> LongShortFusion:
    """Train and save a dense backbone (per long-term mode), or load a cached one."""
    ckpt = _ckpt(args.long_term_mode)
    model_cfg = ModelConfig(embedding_dim=128, long_term_mode=args.long_term_mode)
    if (ckpt / "model.pt").exists() and not args.retrain:
        model, _ = load_backbone(ckpt)
        return model
    model = LongShortFusion(n_items, model_cfg)
    train_cfg = TrainConfig(
        epochs=args.epochs, alpha_train_mode="random", negatives="uniform",
        seed=args.seed,
    )
    Trainer(model, train_cfg, n_items).fit(user_sequences(frames[Split.TRAIN]))
    save_backbone(model, model_cfg, n_items, ckpt)
    return model


def _prepare(
    model: LongShortFusion, model_cfg: ModelConfig, frames: dict, device: str,
    mechanism: str = "anti_profile",
):
    """Build context, control, target candidates and per-user anchors.

    The candidate pool matches the mechanism: popular items for ``anti_profile`` (the
    original detectability-optimised shift), mid-frequency items for ``coherent`` (so
    the popularity/item-bias shortcut cannot rank the new preference).
    """
    context = pd.concat([frames[Split.TRAIN], frames[Split.VAL]], ignore_index=True)
    ctx_by = dict(iter(context.groupby(S_USER_COLUMN)))
    control_by = dict(iter(frames[Split.TEST].groupby(S_USER_COLUMN)))
    candidates = (
        mid_frequency_candidates(frames[Split.TRAIN], pool_size=200)
        if mechanism == "coherent"
        else popular_candidates(frames[Split.TRAIN], pool_size=200)
    )
    lookups = ItemLookups(
        meta=item_meta(frames[Split.TRAIN]), candidates=candidates,
        candidate_reps=_candidate_reps(
            model, candidates, model_cfg.short_window, device
        ),
    )
    test_users = {int(u) for u in frames[Split.TEST][S_USER_COLUMN].unique()}
    anchors = _anchors(model, model_cfg, ctx_by, test_users, device)
    return ctx_by, control_by, lookups, anchors


def _run_policy(
    evaluator: StreamingEvaluator,
    detector: CusumDriftDetector,
    ctx_by: dict[int, pd.DataFrame],
    users_by: dict[int, pd.DataFrame],
    policy_cfg: AdaptationConfig,
    subset: set[int] | None = None,
) -> list[UserRunResult]:
    """Stream every user under one α-policy."""
    policy = build_policy(policy_cfg)
    out = []
    for user, frame in users_by.items():
        if user not in ctx_by or (subset is not None and user not in subset):
            continue
        result = evaluator.run_user(ctx_by[user], frame, detector, policy)
        if result is not None:
            out.append(result)
    return out


def _calibrate(control: list[UserRunResult], target: float) -> float:
    """Smallest CUSUM enter with control false-alarm ≤ target (most sensitive)."""
    chosen = ENTER_GRID[0]
    for enter in ENTER_GRID:
        cfg = CusumDetectorConfig(enter=enter)
        fa = np.mean([
            bool((cfg_run(cfg, r).states == int(DriftState.CONFIRMED)).any())
            for r in control
        ]) if control else 0.0
        if fa <= target:
            chosen = enter
    return chosen


def cfg_run(cfg: CusumDetectorConfig, result: UserRunResult):
    """Run a CUSUM detector config on a streamed result's divergence."""
    return CusumDriftDetector(cfg).run(result.divergence)


def _region_metrics(
    results: list[UserRunResult], onset_by_user: dict[int, int]
) -> dict[str, RankingMetrics]:
    """Pool ranks over pre-onset, post-onset and the whole stream."""
    buckets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "pre": [], "post": [], "all": [],
    }
    for r in results:
        onset = onset_by_user.get(r.user)
        if onset is None:
            continue
        idx = np.arange(r.rank.size)
        pre, post = idx < onset, idx >= onset
        buckets["pre"].append((r.rank[pre], r.is_seen[pre]))
        buckets["post"].append((r.rank[post], r.is_seen[post]))
        buckets["all"].append((r.rank, r.is_seen))
    return {name: _pool(parts) for name, parts in buckets.items()}


def _pool(parts: list[tuple[np.ndarray, np.ndarray]]) -> RankingMetrics:
    """Pool (rank, seen) fragments into one metric set."""
    if not parts:
        return compute_ranking_metrics(
            np.zeros(0, np.int64), np.zeros(0, bool), TOP_K
        )
    ranks = np.concatenate([p[0] for p in parts])
    seen = np.concatenate([p[1] for p in parts])
    return compute_ranking_metrics(ranks, seen, TOP_K)


def _rows(per_policy: dict[str, dict[str, RankingMetrics]]) -> list[dict]:
    """Assemble the per-policy pre/post/overall NDCG table."""
    rows = []
    for name, regions in per_policy.items():
        rows.append({
            "policy": name,
            "pre_ndcg": round(regions["pre"].ndcg_at_k, 5),
            "post_ndcg": round(regions["post"].ndcg_at_k, 5),
            "overall_ndcg": round(regions["all"].ndcg_at_k, 5),
            "overall_mrr": round(regions["all"].mrr, 5),
        })
    return rows


def _verdict(rows: list[dict]) -> str:
    """State whether the adaptive policy beats both baselines overall."""
    by = {r["policy"]: r for r in rows}
    prop = by["proposed-CUSUM"]["overall_ndcg"]
    stat = by["static-0.8"]["overall_ndcg"]
    wind = by["window-B"]["overall_ndcg"]
    if prop > stat and prop > wind:
        return (
            f"ADAPTATION WINS: proposed {prop:.5f} > static-0.8 {stat:.5f} "
            f"and > window-B {wind:.5f} on overall NDCG@10."
        )
    return (
        f"NO WIN: proposed {prop:.5f} vs static-0.8 {stat:.5f}, "
        f"window-B {wind:.5f} — adaptation does not dominate."
    )


def main() -> int:
    """Run the CUSUM-driven adaptive-fusion experiment on the dense cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--min-test", type=int, default=12)
    parser.add_argument("--min-train", type=int, default=40)
    parser.add_argument(
        "--mechanism", default="coherent", choices=["anti_profile", "coherent"]
    )
    parser.add_argument(
        "--long-term-mode", default="ewma", choices=["mean", "ewma", "attention"]
    )
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _build_dense(args)
    frames, manifest = read_subsample(DENSE_DIR)
    model = _train_or_load(frames, manifest.n_item_vocab, args)
    model_cfg = model.cfg
    evaluator = StreamingEvaluator(
        model, model_cfg, get_signal("short_term_drift"), device, top_k=TOP_K
    )
    ctx_by, control_by, lookups, anchors = _prepare(
        model, model_cfg, frames, device, args.mechanism
    )

    inj_cfg = InjectionConfig(
        drift_type="sudden", mechanism=args.mechanism, n_users=5000,
        seed=args.seed, min_test_len=8,
    )
    injected, onsets = DriftInjector(inj_cfg, lookups).inject(
        frames[Split.TEST], anchors
    )
    inj_by = dict(iter(injected.groupby(S_USER_COLUMN)))
    onset_by_user = {int(o.user): int(o.onset_index) for o in onsets}

    config = RunConfig(
        task="dense-adaptation-cusum",
        description="CUSUM-driven adaptive fusion vs static-0.8 and window-B (dense)",
        params={"top_k": TOP_K, "stable_alpha": STABLE_ALPHA,
                "mechanism": args.mechanism, "long_term_mode": args.long_term_mode,
                "n_injected": len(onset_by_user)},
        dataset="amazon-subsample-dense",
        tags=["adaptation", "cusum", "dense"],
        seed=args.seed,
    )

    with LabJournal(config) as journal:
        _experiment(journal, evaluator, ctx_by, control_by, inj_by, onset_by_user)
    return 0


def _experiment(
    journal: LabJournal,
    evaluator: StreamingEvaluator,
    ctx_by: dict[int, pd.DataFrame],
    control_by: dict[int, pd.DataFrame],
    inj_by: dict[int, pd.DataFrame],
    onset_by_user: dict[int, int],
) -> None:
    """Calibrate CUSUM, run the three policies, and report the comparison."""
    default_det = CusumDriftDetector()
    injected_users = set(onset_by_user)
    control = _run_policy(
        evaluator, default_det, ctx_by, control_by,
        AdaptationConfig(policy="static", alpha_static=STABLE_ALPHA), injected_users,
    )
    enter = _calibrate(control, FA_TARGET)
    journal.metric("cusum_enter", enter, context="calibrated")
    detector = CusumDriftDetector(CusumDetectorConfig(enter=enter))

    policies = {
        "static-0.8": AdaptationConfig(policy="static", alpha_static=STABLE_ALPHA),
        "window-B": AdaptationConfig(policy="static", alpha_static=0.0),
        "proposed-CUSUM": AdaptationConfig(
            policy="state", alpha_stable=STABLE_ALPHA, alpha_emerging=STABLE_ALPHA,
            alpha_confirmed=0.2, mode="gradual", alpha_step=0.05,
        ),
    }
    per_policy, curves = {}, {}
    for name, cfg in policies.items():
        results = _run_policy(evaluator, detector, ctx_by, inj_by, cfg)
        per_policy[name] = _region_metrics(results, onset_by_user)
        curves[name] = build_recovery_curve(results, onset_by_user, TOP_K)
        summary = summarize_recovery(curves[name], n_users=len(results))
        journal.metric("recovery_latency", float(summary.recovery_latency or -1),
                       context=name)

    rows = _rows(per_policy)
    journal.table(rows, "dense_adaptation")
    journal.figure(recovery_curves(curves), "dense_recovery_curves")
    verdict = _verdict(rows)
    _summarize(journal, rows, enter, verdict)
    _print(rows, enter, verdict)


def _summarize(
    journal: LabJournal, rows: list[dict], enter: float, verdict: str
) -> None:
    """Write the run's narrative results."""
    journal.summary(
        objective="Test whether CUSUM-driven adaptive fusion beats static-α=0.8 and "
        "window-B on the dense cohort where long-term memory helps.",
        method=f"Off-the-shelf CUSUM (enter={enter:g}, calibrated to ≤{FA_TARGET} "
        "control false-alarm) drives α (STABLE 0.8 → CONFIRMED 0.2, gradual); compare "
        "pre/post/overall NDCG@10 against static-0.8 and window-B on a sudden shift.",
        findings=[verdict] + [
            f"{r['policy']}: pre={r['pre_ndcg']:.5f} post={r['post_ndcg']:.5f} "
            f"overall={r['overall_ndcg']:.5f}"
            for r in rows
        ],
        limitations=[
            "Dense cohort is small; single seed; sudden scenario only.",
            "Short test streams limit post-onset resolution; overall pooled NDCG is "
            "the robust headline.",
        ],
        next_steps=[
            "Seed/CI repetition; gradual & recurring scenarios; full-scale backbone.",
        ],
    )


def _print(rows: list[dict], enter: float, verdict: str) -> None:
    """Echo the comparison to stdout."""
    print(f"\nCUSUM enter (calibrated): {enter:g}")
    print("policy          pre      post     overall")
    for r in rows:
        print(f"  {r['policy']:<14} {r['pre_ndcg']:.5f}  {r['post_ndcg']:.5f}  "
              f"{r['overall_ndcg']:.5f}")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    raise SystemExit(main())
