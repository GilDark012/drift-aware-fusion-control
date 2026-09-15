#!/usr/bin/env python3
"""Decisive probe — does long-term memory (α>0) help on a *denser* cohort?

The window-B finding showed that on the default sparse subsample the long-term
representation L is useless: pure short-term (α=0) is optimal at every α, so there is
no fusion balance worth adapting. This probe tests the one precondition the whole
adaptation thesis needs: on a cohort of users with **long histories** (so L can be
informative), is there any α>0 that beats α=0?

It builds a dense subsample (k-core + a train-history floor), trains a fresh backbone
with **random-α** training (both L and S paths independently optimised, giving L its
best chance), and sweeps α at inference on val and test.

* If some α>0 beats α=0 → L carries signal → adaptation has room → proceed.
* If α=0 still wins → L is useless even on dense users → decisive negative result.

Usage:
    python scripts/dense_alpha_probe.py --epochs 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift_reco.data.schema import Split  # noqa: E402
from drift_reco.data.subsample import (  # noqa: E402
    SubsampleBuilder,
    SubsampleConfig,
    read_subsample,
)
from drift_reco.models.config import EvalConfig, ModelConfig, TrainConfig  # noqa: E402
from drift_reco.models.evaluate import (  # noqa: E402
    NextItemEvaluator,
    build_eval_examples,
)
from drift_reco.models.fusion import LongShortFusion  # noqa: E402
from drift_reco.models.sequences import user_sequences  # noqa: E402
from drift_reco.models.train import Trainer  # noqa: E402
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402

DENSE_DIR = Path("data/interim/subsample_dense")
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _combine(
    a: dict[int, np.ndarray], b: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """Concatenate two per-user sequence maps in time order (a precedes b)."""
    empty = np.zeros(0, dtype=np.int64)
    users = set(a) | set(b)
    return {u: np.concatenate([a.get(u, empty), b.get(u, empty)]) for u in users}


def _build_dense(args: argparse.Namespace) -> None:
    """Build the dense, long-history subsample to its own directory."""
    cfg = SubsampleConfig(
        k_core=args.k_core,
        min_test_interactions=args.min_test,
        min_train_interactions=args.min_train,
        target_users=5000,
        extra_train_users=15000,
        out_dir=DENSE_DIR,
        seed=args.seed,
    )
    manifest = SubsampleBuilder(cfg).build()
    print(
        f"dense cohort: {manifest.selected_users} users, "
        f"vocab {manifest.n_item_vocab}, test rows {manifest.rows_per_split['test']}"
    )


def _seed(seed: int) -> None:
    """Seed numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sweep_alpha(
    model: LongShortFusion,
    model_cfg: ModelConfig,
    context: dict[int, np.ndarray],
    eval_seqs: dict[int, np.ndarray],
    split: str,
) -> list[dict]:
    """Evaluate NDCG@10 / MRR at every α on one split."""
    examples = build_eval_examples(context, eval_seqs, model_cfg)
    rows = []
    for alpha in ALPHAS:
        result = NextItemEvaluator(model, EvalConfig(alpha=alpha)).evaluate(
            examples, split
        )
        rows.append({
            "split": split,
            "alpha": alpha,
            "ndcg_at_10": round(result.ndcg_at_k, 5),
            "mrr": round(result.mrr, 5),
            "ndcg_seen": round(result.ndcg_at_k_seen, 5),
            "n_scored": result.n_scored,
        })
    return rows


def _verdict(test_rows: list[dict]) -> tuple[float, float, float, str]:
    """Return (best_alpha, best_ndcg, ndcg_at_alpha0, verdict) for the test sweep."""
    ndcg0 = next(r["ndcg_at_10"] for r in test_rows if r["alpha"] == 0.0)
    best = max(test_rows, key=lambda r: r["ndcg_at_10"])
    lift = (best["ndcg_at_10"] - ndcg0) / ndcg0 if ndcg0 else float("nan")
    if best["alpha"] > 0.0 and best["ndcg_at_10"] > ndcg0:
        verdict = (
            f"L HELPS: best alpha={best['alpha']} beats alpha=0 by {lift:+.1%} "
            "-> adaptation has room."
        )
    else:
        verdict = "L USELESS: alpha=0 still optimal -> decisive negative result."
    return best["alpha"], best["ndcg_at_10"], ndcg0, verdict


def main() -> int:
    """Build a dense cohort, train, and sweep α to test whether L helps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--k-core", type=int, default=5)
    parser.add_argument("--min-test", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=50)
    parser.add_argument(
        "--long-term-mode", default="ewma", choices=["mean", "ewma", "attention"]
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _seed(args.seed)

    _build_dense(args)
    frames, manifest = read_subsample(DENSE_DIR)
    n_items = manifest.n_item_vocab
    model_cfg = ModelConfig(
        embedding_dim=args.embedding_dim, long_term_mode=args.long_term_mode
    )
    train_cfg = TrainConfig(
        epochs=args.epochs, alpha_train_mode="random", negatives="uniform",
        seed=args.seed,
    )

    config = RunConfig(
        task="dense-alpha-probe",
        description="Does α>0 beat α=0 on a dense long-history cohort?",
        params={"dense": {"k_core": args.k_core, "min_test": args.min_test,
                          "min_train": args.min_train},
                "n_items": n_items, "alphas": list(ALPHAS)},
        dataset="amazon-subsample-dense",
        tags=["adaptation", "alpha-sweep", "dense"],
        seed=args.seed,
    )

    with LabJournal(config) as journal:
        train_seqs = user_sequences(frames[Split.TRAIN])
        val_seqs = user_sequences(frames[Split.VAL])
        test_seqs = user_sequences(frames[Split.TEST])

        model = LongShortFusion(n_items, model_cfg)
        journal.event("model", "train_start", detail=f"{n_items} items")
        outcome = Trainer(model, train_cfg, n_items).fit(train_seqs)
        journal.metric("train_loss", outcome.final_loss, split="train")

        val_rows = _sweep_alpha(model, model_cfg, train_seqs, val_seqs, "val")
        context_tv = _combine(train_seqs, val_seqs)
        test_rows = _sweep_alpha(model, model_cfg, context_tv, test_seqs, "test")
        journal.table(val_rows + test_rows, "alpha_sweep")
        for row in test_rows:
            journal.metric("ndcg_at_10", row["ndcg_at_10"], split="test",
                           step=int(row["alpha"] * 100))

        best_alpha, best_ndcg, ndcg0, verdict = _verdict(test_rows)
        _summarize(journal, val_rows, test_rows, best_alpha, best_ndcg, ndcg0, verdict)
        _print(val_rows, test_rows, verdict)
    return 0


def _summarize(
    journal: LabJournal,
    val_rows: list[dict],
    test_rows: list[dict],
    best_alpha: float,
    best_ndcg: float,
    ndcg0: float,
    verdict: str,
) -> None:
    """Write the run's narrative results."""
    journal.summary(
        objective="Test the precondition for the adaptation thesis: on a dense "
        "long-history cohort, does any α>0 (using long-term memory) beat α=0?",
        method="Build a dense subsample (train-history floor); train with random-α so "
        "both paths are optimised; sweep α at inference on val and test.",
        findings=[
            verdict,
            f"test NDCG@10: a=0 -> {ndcg0:.5f}; best a={best_alpha} -> {best_ndcg:.5f}",
            "val sweep: " + ", ".join(
                f"a{r['alpha']}={r['ndcg_at_10']:.5f}" for r in val_rows
            ),
            "test sweep: " + ", ".join(
                f"a{r['alpha']}={r['ndcg_at_10']:.5f}" for r in test_rows
            ),
        ],
        limitations=[
            "Dense cohort is small (long-history users are rare); one trained seed.",
            "EWMA-pool long-term encoder; a stronger L encoder is a separate lever.",
        ],
        next_steps=[
            "If L helps: re-run injection + adaptation (CUSUM) on this cohort.",
            "If not: report the negative result and narrow the thesis to detection.",
        ],
    )


def _print(val_rows: list[dict], test_rows: list[dict], verdict: str) -> None:
    """Echo the sweep and verdict to stdout (ASCII-safe for Windows consoles)."""
    print("\nalpha-sweep NDCG@10 (higher = better):")
    print("  alpha :", "  ".join(f"{a:>5}" for a in ALPHAS))
    print("  val   :", "  ".join(f"{r['ndcg_at_10']:.4f}" for r in val_rows))
    print("  test  :", "  ".join(f"{r['ndcg_at_10']:.4f}" for r in test_rows))
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    raise SystemExit(main())
