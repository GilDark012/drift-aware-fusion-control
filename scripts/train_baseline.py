#!/usr/bin/env python3
"""Phase 3 — train and evaluate the fixed-α long/short fusion baseline.

Trains the shared backbone on the subsample, then evaluates temporal next-item
recommendation on val and test at a fixed α (Baseline A). Everything is journaled.

Usage:
    python scripts/train_baseline.py --epochs 5 --alpha 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift_reco.config.constants import BASELINE_CHECKPOINT  # noqa: E402
from drift_reco.data.schema import Split  # noqa: E402
from drift_reco.data.subsample import read_subsample  # noqa: E402
from drift_reco.models.checkpoint import save_backbone  # noqa: E402
from drift_reco.models.config import EvalConfig, ModelConfig, TrainConfig  # noqa: E402
from drift_reco.models.evaluate import (  # noqa: E402
    EvalResult,
    NextItemEvaluator,
    build_eval_examples,
)
from drift_reco.models.fusion import LongShortFusion  # noqa: E402
from drift_reco.models.sequences import user_sequences  # noqa: E402
from drift_reco.models.train import Trainer  # noqa: E402
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402


def _seed_everything(seed: int) -> None:
    """Seed numpy and torch for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _combine(
    a: dict[int, np.ndarray], b: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """Concatenate two per-user sequence maps in time order (a precedes b)."""
    empty = np.zeros(0, dtype=np.int64)
    users = set(a) | set(b)
    return {u: np.concatenate([a.get(u, empty), b.get(u, empty)]) for u in users}


def _log_eval(journal: LabJournal, result: EvalResult) -> None:
    """Record every metric of an eval result."""
    for name in (
        "ndcg_at_k",
        "recall_at_k",
        "hit_rate_at_k",
        "mrr",
        "ndcg_at_k_seen",
        "hit_rate_at_k_seen",
    ):
        journal.metric(name, getattr(result, name), split=result.split)
    journal.metric("n_scored", result.n_scored, split=result.split, unit="positions")
    journal.metric(
        "n_seen_target", result.n_seen_target, split=result.split, unit="positions"
    )


def _build_configs(
    args: argparse.Namespace,
) -> tuple[ModelConfig, TrainConfig, EvalConfig]:
    """Assemble the three validated configs from CLI arguments."""
    model_cfg = ModelConfig(
        embedding_dim=args.embedding_dim, short_window=args.short_window
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        alpha_train_mode=args.alpha_train_mode,
        align_weight=args.align_weight,
        negatives=args.negatives,
        seed=args.seed,
    )
    eval_cfg = EvalConfig(top_k=args.top_k, alpha=args.alpha, seed=args.seed)
    return model_cfg, train_cfg, eval_cfg


def main() -> int:
    """Run the Phase 3 baseline training and evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--short-window", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--alpha-train-mode", default="random", choices=["fixed", "random"]
    )
    parser.add_argument("--align-weight", type=float, default=1.0)
    parser.add_argument(
        "--negatives", default="uniform", choices=["uniform", "popularity"]
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _seed_everything(args.seed)

    model_cfg, train_cfg, eval_cfg = _build_configs(args)
    frames, manifest = read_subsample()
    n_items = manifest.n_item_vocab

    config = RunConfig(
        task="phase3-baseline-fixed-alpha",
        description=f"Fixed-α={args.alpha} long/short fusion baseline (subsample)",
        params={
            "model": model_cfg.model_dump(mode="json"),
            "train": train_cfg.model_dump(mode="json"),
            "eval": eval_cfg.model_dump(mode="json"),
            "n_items": n_items,
        },
        dataset="amazon-subsample",
        tags=["phase3", "baseline", "fixed-alpha"],
        seed=args.seed,
    )

    with LabJournal(config) as journal:
        journal.event("data", "subsample_loaded", detail=str(manifest.rows_per_split))
        train_seqs = user_sequences(frames[Split.TRAIN])
        val_seqs = user_sequences(frames[Split.VAL])
        test_seqs = user_sequences(frames[Split.TEST])

        model = LongShortFusion(n_items, model_cfg)
        journal.event("model", "train_start", detail=f"{n_items} items")
        outcome = Trainer(model, train_cfg, n_items).fit(train_seqs)
        journal.metric("train_loss", outcome.final_loss, split="train")
        journal.metric("train_examples", outcome.n_examples, split="train", unit="ex")
        journal.event("model", "train_done", detail=f"device={outcome.device}")

        evaluator = NextItemEvaluator(model, eval_cfg)
        val_ex = build_eval_examples(train_seqs, val_seqs, model_cfg)
        val_res = evaluator.evaluate(val_ex, "val")
        _log_eval(journal, val_res)

        context_tv = _combine(train_seqs, val_seqs)
        test_ex = build_eval_examples(context_tv, test_seqs, model_cfg)
        test_res = evaluator.evaluate(test_ex, "test")
        _log_eval(journal, test_res)

        torch.save(model.state_dict(), journal.run_dir / "model.pt")
        save_backbone(model, model_cfg, n_items, BASELINE_CHECKPOINT)
        journal.event("model", "checkpoint_saved", detail=BASELINE_CHECKPOINT)
        journal.summary(
            objective="Establish the fixed-α long/short fusion baseline (Baseline A).",
            method=f"GRU short-term + EWMA long-term, α-random training, evaluated at "
            f"fixed α={args.alpha} with full-vocabulary temporal ranking.",
            findings=[
                f"train loss {outcome.final_loss:.4f}, {outcome.n_examples:,} examples",
                f"VAL  NDCG@{args.top_k}={val_res.ndcg_at_k:.4f} "
                f"HR@{args.top_k}={val_res.hit_rate_at_k:.4f} MRR={val_res.mrr:.4f} "
                f"(seen NDCG={val_res.ndcg_at_k_seen:.4f})",
                f"TEST NDCG@{args.top_k}={test_res.ndcg_at_k:.4f} "
                f"HR@{args.top_k}={test_res.hit_rate_at_k:.4f} MRR={test_res.mrr:.4f} "
                f"(seen NDCG={test_res.ndcg_at_k_seen:.4f})",
                f"scored: val={val_res.n_scored:,} test={test_res.n_scored:,}",
            ],
            limitations=[
                "Subsample scale; absolute NDCG is low on sparse Amazon data.",
                "Unseen test targets (OOV) are unrankable and counted as misses in the "
                "full metric; the 'seen' metric conditions on rankable targets.",
            ],
            next_steps=[
                "Add divergence signal D_u(t)=dist(L,S) (Phase 4) over this backbone.",
            ],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
