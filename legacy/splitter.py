# src/data/splitter.py -- Global temporal 70/10/20 split. Raw data is NEVER modified.
# src/data/splitter.py
"""
Global temporal split: TRAIN 70% | VAL 10% | TEST 20%
Split is computed on global timestamp quantiles, then merged per category.
Output: data/processed/{train, val, test}.parquet
Rule: NO user interaction may appear in more than one split.
"""

import pandas as pd
import os
from src.utils.seed import set_all_seeds

set_all_seeds(42)  # SEED_DATA

IN_PATH  = "data/processed/amazon_clean.parquet"
OUT_DIR  = "data/processed"
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
# TEST_RATIO  = 0.20  (remainder)


def temporal_split(df: pd.DataFrame):
    """
    Apply global temporal cutoffs to preserve natural time flow
    and prevent temporal leakage across users.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    t1_idx = int(n * TRAIN_RATIO)
    t2_idx = int(n * (TRAIN_RATIO + VAL_RATIO))

    # Cutoff timestamps (shared across all users — global split)
    T1 = df.loc[t1_idx, "timestamp"]
    T2 = df.loc[t2_idx, "timestamp"]

    train = df[df["timestamp"] < T1].copy()
    val   = df[(df["timestamp"] >= T1) & (df["timestamp"] < T2)].copy()
    test  = df[df["timestamp"] >= T2].copy()

    print(f"[splitter] T1={T1} | T2={T2}")
    print(f"[splitter] TRAIN: {len(train):,} | VAL: {len(val):,} | TEST: {len(test):,}")

    # Safety check: no leakage
    assert len(set(train.index) & set(val.index)) == 0, "Leakage: train ∩ val"
    assert len(set(train.index) & set(test.index)) == 0, "Leakage: train ∩ test"
    assert len(set(val.index) & set(test.index)) == 0, "Leakage: val ∩ test"

    return train, val, test, T1, T2


def save_splits(train, val, test):
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(OUT_DIR, f"{name}.parquet")
        split.to_parquet(path, index=False)
        print(f"[splitter] Saved {name} → {path}")


def run_split():
    df = pd.read_parquet(IN_PATH)
    train, val, test, T1, T2 = temporal_split(df)
    save_splits(train, val, test)
    # Save cutoffs for downstream use
    pd.Series({"T1": str(T1), "T2": str(T2)}).to_json(
        os.path.join(OUT_DIR, "split_cutoffs.json")
    )
    return train, val, test


if __name__ == "__main__":
    run_split()