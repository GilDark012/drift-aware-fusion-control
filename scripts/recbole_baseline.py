#!/usr/bin/env python3
"""P1.1 (sequential half) — external SASRec / GRU4Rec baselines via RecBole.

The review asks for a real, independently-implemented sequential recommender to rule
out bugs in our backbone and give reference numbers. This exports our subsample to
RecBole's atomic-file format and runs **SASRec** and **GRU4Rec** from RecBole under its
standard sequential protocol (per-user leave-one-out split, full-catalogue ranking),
reporting NDCG@10 / Hit@10 / MRR.

Protocol note: RecBole's sequential task uses a per-user leave-one-out split, which is
*not* identical to this project's global chronological cutoff. The comparison is
therefore an **order-of-magnitude sanity check** on the same data — does an established
SASRec land in the same low range our sparse data forces? — not a same-protocol
head-to-head. Both use full ranking, so the ranges are directly readable.

Usage:
    python scripts/recbole_baseline.py --epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift_reco.config.constants import (  # noqa: E402
    ITEM_IDX_COLUMN,
    OOV_IDX,
    S_ITEM_COLUMN,
    USER_IDX_COLUMN,
)
from drift_reco.data.schema import Split  # noqa: E402
from drift_reco.data.subsample import read_subsample  # noqa: E402

RECBOLE_DIR = Path("data/interim/recbole")
DATASET = "amazon_sub"
MODELS = ("SASRec", "GRU4Rec")


def _export_inter(out_dir: Path) -> int:
    """Write the subsample's (user, item, timestamp) triples as a RecBole .inter file.

    Uses the true ``item_idx`` identity (not the train-vocab ``s_item``, which collapses
    unseen items to a single OOV id). Returns the number of interactions written.
    """
    frames, _ = read_subsample()
    rows = []
    for split in Split:
        frame = frames[split]
        keep = frame[frame[S_ITEM_COLUMN] != OOV_IDX]  # drop unrankable OOV rows
        rows.append(keep[[USER_IDX_COLUMN, ITEM_IDX_COLUMN, "timestamp"]])
    full = pd.concat(rows, ignore_index=True).drop_duplicates()
    full["ts"] = full["timestamp"].astype("int64") // 10**9  # seconds
    out_dir.mkdir(parents=True, exist_ok=True)
    inter = out_dir / f"{DATASET}.inter"
    with inter.open("w", encoding="utf-8") as fh:
        fh.write("user_id:token\titem_id:token\ttimestamp:float\n")
        full[[USER_IDX_COLUMN, ITEM_IDX_COLUMN, "ts"]].to_csv(
            fh, sep="\t", header=False, index=False
        )
    return len(full)


def _config(epochs: int) -> dict:
    """RecBole configuration shared by both sequential models."""
    return {
        "data_path": str(RECBOLE_DIR),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "user_inter_num_interval": "[3,inf)",
        "item_inter_num_interval": "[1,inf)",
        "MAX_ITEM_LIST_LENGTH": 50,
        "eval_args": {
            "split": {"LS": "valid_and_test"},
            "order": "TO",
            "mode": "full",
            "group_by": "user",
        },
        "train_neg_sample_args": None,  # SASRec/GRU4Rec use full-softmax CE loss
        "metrics": ["NDCG", "Hit", "MRR"],
        "topk": [10],
        "valid_metric": "NDCG@10",
        "epochs": epochs,
        "train_batch_size": 512,
        "eval_batch_size": 4096,
        "stopping_step": 5,
        "hidden_size": 128,
        "embedding_size": 128,
        "learning_rate": 1e-3,
        "seed": 42,
        "reproducibility": True,
        "checkpoint_dir": str(RECBOLE_DIR / "saved"),
        "show_progress": False,
        "state": "WARNING",
    }


def _patch_torch_load() -> None:
    """Force ``weights_only=False`` for RecBole's own checkpoint reload.

    RecBole 1.2.1 saves a full checkpoint (with its config object) and reloads it for
    evaluation; PyTorch >= 2.6 defaults ``torch.load`` to ``weights_only=True`` and
    rejects it. The checkpoint is produced by RecBole in this same run, so it is safe.
    """
    import torch  # noqa: PLC0415

    original = torch.load

    def loader(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = loader  # type: ignore[assignment]


def _run_model(model: str, epochs: int) -> dict:
    """Run one RecBole model and return its test metrics."""
    from recbole.quick_start import run_recbole  # noqa: PLC0415

    _patch_torch_load()
    result = run_recbole(
        model=model, dataset=DATASET, config_dict=_config(epochs)
    )
    return dict(result["test_result"])


def main() -> int:
    """Export the subsample and run the RecBole sequential baselines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    n = _export_inter(RECBOLE_DIR / DATASET)
    print(f"exported {n:,} interactions to {RECBOLE_DIR / DATASET}/{DATASET}.inter")

    rows = []
    for model in MODELS:
        print(f"\n=== running {model} ===")
        metrics = _run_model(model, args.epochs)
        rows.append(
            {"model": model, **{k: round(float(v), 5) for k, v in metrics.items()}}
        )

    print("\nRecBole sequential baselines (leave-one-out, full ranking):")
    for r in rows:
        print(
            f"  {r['model']:<9} NDCG@10={r.get('ndcg@10')} "
            f"Hit@10={r.get('hit@10')} MRR={r.get('mrr@10')}"
        )
    print(
        "\nReference: our fusion backbone on its own global-chronological protocol "
        "reaches NDCG@10 ~0.017 vs a popularity floor ~0.012 on this subsample."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
