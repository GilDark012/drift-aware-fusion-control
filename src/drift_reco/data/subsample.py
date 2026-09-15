"""Reproducible stratified subsample for fast development iteration.

The full data (≈4M interactions) is expensive to iterate on. This builder selects
a cohort of test-active users, keeps their *full* cross-split history, and re-indexes
users and items into a dense, subsample-local vocabulary. Two properties are
preserved on purpose:

* **the evaluable cohort** (ADR-0001): only users with enough test-period history,
  so per-user latency remains measurable;
* **the unseen-item phenomenon**: the item vocabulary is built from *training* items
  only, so val/test items absent from training map to a reserved OOV id — exactly as
  they would at full scale.

Final results are produced at full scale; the subsample exists to validate the
pipeline and iterate quickly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ..config.constants import (
    CATEGORY_COLUMN,
    ITEM_IDX_COLUMN,
    N_RESERVED_ITEM_IDS,
    OOV_IDX,
    S_ITEM_COLUMN,
    S_USER_COLUMN,
    SUBSAMPLE_DIR,
    USER_IDX_COLUMN,
)
from .schema import Split, SplitPaths, load_all_splits

__all__ = [
    "SubsampleConfig",
    "SubsampleManifest",
    "SubsampleBuilder",
    "read_subsample",
]


def read_subsample(
    out_dir: str | Path = SUBSAMPLE_DIR,
) -> tuple[dict[Split, pd.DataFrame], SubsampleManifest]:
    """Load a previously built subsample and its manifest."""
    root = Path(out_dir)
    frames = {
        split: pd.read_parquet(root / f"{split.value}.parquet") for split in Split
    }
    manifest = SubsampleManifest.model_validate_json(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    return frames, manifest


class SubsampleConfig(BaseModel):
    """Parameters controlling subsample construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 42
    target_users: int = Field(default=3000, gt=0)
    min_test_interactions: int = Field(default=20, ge=1)
    min_train_interactions: int = Field(default=0, ge=0)
    activity_strata: int = Field(default=3, ge=1)
    extra_train_users: int = Field(default=20000, ge=0)
    k_core: int = Field(default=5, ge=1)
    category: str | None = None
    processed_dir: Path = Path("data/processed")
    out_dir: Path = Path(SUBSAMPLE_DIR)


class SubsampleManifest(BaseModel):
    """Provenance of a built subsample."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    selected_users: int
    train_users_total: int
    n_item_vocab: int
    rows_per_split: dict[str, int]
    unseen_item_rate: dict[str, float]
    cohort_pool_size: int


class SubsampleBuilder:
    """Build and persist a stratified, dense-indexed development subsample."""

    def __init__(self, config: SubsampleConfig) -> None:
        """Store the configuration and prepare the split resolver."""
        self.config = config
        self.paths = SplitPaths(config.processed_dir)
        self._frames: dict[Split, pd.DataFrame] = {}
        self._rng = np.random.default_rng(config.seed)

    def _select_cohort(self) -> tuple[NDArray[np.int64], int]:
        """Pick eval-cohort users, stratified by test activity, seeded.

        Returns:
            The selected ``user_idx`` array and the full cohort-pool size.
        """
        test = self._frames[Split.TEST]
        counts = test[USER_IDX_COLUMN].value_counts()
        cohort = counts[counts >= self.config.min_test_interactions]
        if self.config.min_train_interactions > 0:
            train_counts = self._frames[Split.TRAIN][USER_IDX_COLUMN].value_counts()
            long_history = train_counts[
                train_counts >= self.config.min_train_interactions
            ].index
            cohort = cohort[cohort.index.isin(long_history)]
        pool_size = int(cohort.size)
        if pool_size <= self.config.target_users:
            return cohort.index.to_numpy(), pool_size
        strata = pd.qcut(
            cohort.to_numpy(),
            self.config.activity_strata,
            labels=False,
            duplicates="drop",
        )
        return self._sample_strata(cohort.index.to_numpy(), strata), pool_size

    def _sample_strata(
        self, users: NDArray[np.int64], strata: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """Proportionally sample ``target_users`` across activity strata."""
        target = self.config.target_users
        chosen: list[NDArray[np.int64]] = []
        for stratum in np.unique(strata):
            members = users[strata == stratum]
            take = min(max(1, round(target * members.size / users.size)), members.size)
            chosen.append(self._rng.choice(members, size=take, replace=False))
        picked = np.concatenate(chosen)
        if picked.size > target:
            picked = self._rng.choice(picked, size=target, replace=False)
        return np.sort(picked)

    def _extra_train_users(self, cohort: set[int]) -> set[int]:
        """Sample extra train-only users to enrich the item vocabulary."""
        if self.config.extra_train_users == 0:
            return set()
        train_users = set(self._frames[Split.TRAIN][USER_IDX_COLUMN].unique())
        pool = np.array(sorted(train_users - cohort), dtype=np.int64)
        take = min(self.config.extra_train_users, pool.size)
        return set(self._rng.choice(pool, size=take, replace=False).tolist())

    def _build_item_vocab(self, train_sub: pd.DataFrame) -> dict[int, int]:
        """Map training ``item_idx`` to dense ids starting after reserved ids."""
        unique_items = np.sort(train_sub[ITEM_IDX_COLUMN].unique())
        return {
            int(item): rank + N_RESERVED_ITEM_IDS
            for rank, item in enumerate(unique_items)
        }

    def _apply_mapping(
        self,
        frame: pd.DataFrame,
        user_vocab: dict[int, int],
        item_vocab: dict[int, int],
    ) -> pd.DataFrame:
        """Attach dense ``s_user``/``s_item`` columns; unseen items become OOV."""
        out = frame.copy()
        out[S_USER_COLUMN] = out[USER_IDX_COLUMN].map(user_vocab).astype("int64")
        out[S_ITEM_COLUMN] = (
            out[ITEM_IDX_COLUMN].map(item_vocab).fillna(OOV_IDX).astype("int64")
        )
        return out

    def build(self) -> SubsampleManifest:
        """Construct, persist and describe the subsample.

        The eval cohort (test-active users) drives val/test; extra train-only users
        enrich the training frame and item vocabulary but are never evaluated.
        """
        if not self._frames:
            self._frames = load_all_splits(self.paths)
        if self.config.category is not None:
            self._frames = self._filter_category(self._frames)
        self._frames = self._global_kcore(self._frames)
        cohort, pool_size = self._select_cohort()
        cohort_set = set(cohort.tolist())
        included = cohort_set | self._extra_train_users(cohort_set)

        subs = {
            Split.TRAIN: self._filter(Split.TRAIN, included),
            Split.VAL: self._filter(Split.VAL, cohort_set),
            Split.TEST: self._filter(Split.TEST, cohort_set),
        }
        subs = self._apply_kcore(subs)
        user_vocab = {int(u): rank for rank, u in enumerate(sorted(included))}
        item_vocab = self._build_item_vocab(subs[Split.TRAIN])
        mapped = {
            split: self._apply_mapping(frame, user_vocab, item_vocab)
            for split, frame in subs.items()
        }
        self._persist(mapped)
        return self._manifest(
            mapped, item_vocab, len(cohort_set), len(included), pool_size
        )

    def _filter(self, split: Split, users: set[int]) -> pd.DataFrame:
        """Restrict a split to the given user set."""
        frame = self._frames[split]
        return frame[frame[USER_IDX_COLUMN].isin(users)].reset_index(drop=True)

    def _filter_category(
        self, frames: dict[Split, pd.DataFrame]
    ) -> dict[Split, pd.DataFrame]:
        """Restrict every split to a single product category (a second-dataset test)."""
        category = self.config.category
        return {
            split: frame[frame[CATEGORY_COLUMN] == category].reset_index(drop=True)
            for split, frame in frames.items()
        }

    def _global_kcore(
        self, frames: dict[Split, pd.DataFrame]
    ) -> dict[Split, pd.DataFrame]:
        """Keep only items with >= ``k_core`` FULL-train interactions, all splits.

        Standard sequential-recommendation preprocessing (SASRec, GRU4Rec): items with
        too little support cannot be embedded and their occurrences are unrankable.
        Applying this on the *full* data **before** cohort selection means the cohort
        is chosen on the dense test, so its users keep long streams (applying it after
        selection instead shrank cohort streams — see ADR-0006).
        """
        if self.config.k_core <= 1:
            return frames
        support = frames[Split.TRAIN][ITEM_IDX_COLUMN].value_counts()
        kept = set(support[support >= self.config.k_core].index)
        return {
            split: frame[frame[ITEM_IDX_COLUMN].isin(kept)].reset_index(drop=True)
            for split, frame in frames.items()
        }

    def _apply_kcore(
        self, subs: dict[Split, pd.DataFrame]
    ) -> dict[Split, pd.DataFrame]:
        """Local k-core within the subsample train, so its vocabulary is learnable."""
        if self.config.k_core <= 1:
            return subs
        support = subs[Split.TRAIN][ITEM_IDX_COLUMN].value_counts()
        kept = set(support[support >= self.config.k_core].index)
        return {
            split: frame[frame[ITEM_IDX_COLUMN].isin(kept)].reset_index(drop=True)
            for split, frame in subs.items()
        }

    def _persist(self, mapped: dict[Split, pd.DataFrame]) -> None:
        """Write each subsample split to parquet."""
        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        for split, frame in mapped.items():
            path = self.config.out_dir / f"{split.value}.parquet"
            frame.to_parquet(path, index=False)

    def _manifest(
        self,
        mapped: dict[Split, pd.DataFrame],
        item_vocab: dict[int, int],
        n_cohort: int,
        n_included: int,
        pool_size: int,
    ) -> SubsampleManifest:
        """Assemble and persist the subsample manifest."""
        rows = {s.value: int(len(f)) for s, f in mapped.items()}
        unseen = {
            s.value: float((f[S_ITEM_COLUMN] == OOV_IDX).mean())
            for s, f in mapped.items()
        }
        manifest = SubsampleManifest(
            seed=self.config.seed,
            selected_users=n_cohort,
            train_users_total=n_included,
            n_item_vocab=len(item_vocab) + N_RESERVED_ITEM_IDS,
            rows_per_split=rows,
            unseen_item_rate=unseen,
            cohort_pool_size=pool_size,
        )
        (self.config.out_dir / "manifest.json").write_text(
            json.dumps(manifest.model_dump(), indent=2), encoding="utf-8"
        )
        return manifest
