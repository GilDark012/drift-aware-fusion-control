# src/data/loader.py
"""
Load and filter Amazon Product Reviews into the unified schema.
Scope: Electronics, Clothing, Books | Jan 2014 – Dec 2018
Output: data/processed/amazon_clean.parquet
"""

import pandas as pd
import os
import json
from src.utils.seed import set_all_seeds

set_all_seeds(42)  # SEED_DATA

CATEGORIES = ["Electronics", "Clothing_Shoes_and_Jewelry", "Books"]
RAW_DIR    = "F:/Dataset/raw"
OUT_DIR    = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

MIN_USER_INTERACTIONS = 20
MIN_ITEM_INTERACTIONS = 5
START_DATE = "2014-01-01"
END_DATE   = "2018-12-31"


def load_category(category: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"{category}.jsonl")
    part_dir = os.path.join(OUT_DIR, f"{category}_parts")
    os.makedirs(part_dir, exist_ok=True)

    rows = []
    batch_size = 100_000
    part_idx = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("verified_purchase") is not True:
                continue

            user_id = obj.get("user_id")
            item_id = obj.get("parent_asin")
            rating = obj.get("rating")
            timestamp = obj.get("timestamp")

            if None in (user_id, item_id, rating, timestamp):
                continue

            rows.append({
                "user_id": user_id,
                "item_id": item_id,
                "category": category,
                "preference_score": rating,
                "timestamp": timestamp,
            })

            if len(rows) >= batch_size:
                pd.DataFrame(rows).to_parquet(
                    os.path.join(part_dir, f"part_{part_idx:04d}.parquet"),
                    index=False
                )
                rows = []
                part_idx += 1

    if rows:
        pd.DataFrame(rows).to_parquet(
            os.path.join(part_dir, f"part_{part_idx:04d}.parquet"),
            index=False
        )

    part_files = [
        os.path.join(part_dir, f)
        for f in os.listdir(part_dir)
        if f.endswith(".parquet")
    ]

    if part_files:
        df = pd.concat([pd.read_parquet(p) for p in sorted(part_files)], ignore_index=True)
    else:
        df = pd.DataFrame(columns=["user_id", "item_id", "category", "preference_score", "timestamp"])

    final_path = os.path.join(OUT_DIR, f"{category}_filtered.parquet")
    df.to_parquet(final_path, index=False)
    print(f"[loader] Saved filtered {category} -> {final_path}")
    return df

def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply temporal window, activity thresholds, and dedup."""

    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        sample = df["timestamp"].dropna().iloc[0]
        digit_count = len(str(int(sample)))

        if digit_count >= 13:
            unit = "ms"   # milliseconds (Amazon Reviews 2023 standard)
        elif digit_count >= 10:
            unit = "s"    # seconds
        else:
            raise ValueError(f"Unexpected timestamp format: {sample}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit=unit, errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop rows where conversion failed
    df = df.dropna(subset=["timestamp"])

    # Temporal window
    df = df[(df["timestamp"] >= START_DATE) & (df["timestamp"] <= END_DATE)]

    # Remove duplicates
    df = df.drop_duplicates(subset=["user_id", "item_id", "timestamp"])

    # Sort chronologically — ALWAYS before any split
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Filter low-activity items
    item_counts = df["item_id"].value_counts()
    df = df[df["item_id"].isin(item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index)]

    # Filter low-history users
    user_counts = df["user_id"].value_counts()
    df = df[df["user_id"].isin(user_counts[user_counts >= MIN_USER_INTERACTIONS].index)]

    return df.reset_index(drop=True)


def build_integer_index(df: pd.DataFrame):
    """Map string IDs to sequential integers for embedding layers."""
    user2idx = {u: i for i, u in enumerate(df["user_id"].unique())}
    item2idx = {it: i for i, it in enumerate(df["item_id"].unique())}
    df["user_idx"] = df["user_id"].map(user2idx)
    df["item_idx"] = df["item_id"].map(item2idx)
    return df, user2idx, item2idx


def load_amazon() -> pd.DataFrame:
    frames = []
    for cat in CATEGORIES:
        df_cat = load_category(cat)
        frames.append(df_cat)

    df = pd.concat(frames, ignore_index=True)
    df = filter_data(df)
    df, user2idx, item2idx = build_integer_index(df)

    out_path = os.path.join(OUT_DIR, "amazon_clean.parquet")
    df.to_parquet(out_path, index=False)

    print(
        f"[loader] Saved {len(df):,} interactions | "
        f"{df['user_id'].nunique():,} users | "
        f"{df['item_id'].nunique():,} items -> {out_path}"
    )
    return df


if __name__ == "__main__":
    load_amazon()