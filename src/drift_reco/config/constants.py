"""Project-wide constants.

Column names, directory names and metric identifiers live here and nowhere else,
so that renaming one of them is a single-file change rather than a search across
the codebase.
"""

from __future__ import annotations

from typing import Final

# --- Dataset columns (the processed Amazon parquet schema) -----------------
# Single source of truth: the loader/splitter emit exactly these columns.
USER_COLUMN: Final[str] = "user_id"
ITEM_COLUMN: Final[str] = "item_id"
CATEGORY_COLUMN: Final[str] = "category"
RATING_COLUMN: Final[str] = "preference_score"
TIMESTAMP_COLUMN: Final[str] = "timestamp"
USER_IDX_COLUMN: Final[str] = "user_idx"
ITEM_IDX_COLUMN: Final[str] = "item_idx"

INTERACTION_COLUMNS: Final[tuple[str, ...]] = (
    USER_COLUMN,
    ITEM_COLUMN,
    CATEGORY_COLUMN,
    RATING_COLUMN,
    TIMESTAMP_COLUMN,
    USER_IDX_COLUMN,
    ITEM_IDX_COLUMN,
)

CATEGORIES: Final[tuple[str, ...]] = (
    "Electronics",
    "Clothing_Shoes_and_Jewelry",
    "Books",
)

# --- Canonical data locations ----------------------------------------------
PROCESSED_DIR: Final[str] = "data/processed"
INTERIM_DIR: Final[str] = "data/interim"
SUBSAMPLE_DIR: Final[str] = "data/interim/subsample"
DRIFT_DIR: Final[str] = "data/drift"
CLEAN_PARQUET: Final[str] = "data/processed/amazon_clean.parquet"
TRAIN_PARQUET: Final[str] = "data/processed/train.parquet"
VAL_PARQUET: Final[str] = "data/processed/val.parquet"
TEST_PARQUET: Final[str] = "data/processed/test.parquet"

# --- Dense model vocabulary --------------------------------------------------
# Model-facing integer id columns (dense, subsample-local). Reserved ids:
#   0 = PAD (empty sequence slot), 1 = OOV (item unseen at training time).
S_USER_COLUMN: Final[str] = "s_user"
S_ITEM_COLUMN: Final[str] = "s_item"
PAD_IDX: Final[int] = 0
OOV_IDX: Final[int] = 1
N_RESERVED_ITEM_IDS: Final[int] = 2

# --- Artifact locations ----------------------------------------------------
ARTIFACTS_DIR: Final[str] = "artifacts"
RUNS_DIR: Final[str] = "runs"
INDEX_FILE: Final[str] = "index.csv"
MODELS_DIR: Final[str] = "artifacts/models"
BASELINE_CHECKPOINT: Final[str] = "artifacts/models/baseline-subsample"

DEFAULT_TOP_K: Final[int] = 10
DEFAULT_SEED: Final[int] = 42
