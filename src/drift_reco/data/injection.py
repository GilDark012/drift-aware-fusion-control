"""Controlled preference-drift injection with known onsets.

The point (and the fix for Phase-2 Risk C) is that a shift must change the user's
**preference sequence** — the items they actually engage with — not merely relabel a
category. A controlled shift must also be *unambiguous*, so detection and recovery
latency are measurable.

We switch a user, after a known onset, onto a tight cluster of items whose
**short-term representation is maximally opposed to the user's own established
short-term profile** (real, in-vocabulary, index-consistent). Their recent behaviour
then drives the short-term state far from its slow average, the drift signal rises and
persists, and the ground-truth onset lets us measure how fast the system reacts. This
is a deliberately strong, clean controlled shift; real preference drift is weaker and
messier (a stated limitation).

The item short-term representations and the per-user anchors are computed by the
caller (which owns the model) and passed in, so this module stays pure-data.

Three scenarios: **sudden** (switch at onset), **gradual** (ramp over a window) and
**recurring** (alternating blocks of new/old preference).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ..config.constants import (
    CATEGORY_COLUMN,
    ITEM_COLUMN,
    ITEM_IDX_COLUMN,
    S_ITEM_COLUMN,
    S_USER_COLUMN,
    TIMESTAMP_COLUMN,
)

__all__ = [
    "DriftType",
    "DriftMechanism",
    "InjectionConfig",
    "InjectedOnset",
    "ItemLookups",
    "popular_candidates",
    "mid_frequency_candidates",
    "item_meta",
    "DriftInjector",
]

DriftType = Literal["sudden", "gradual", "recurring"]
DriftMechanism = Literal["anti_profile", "coherent"]
_META_COLUMNS: Final[tuple[str, ...]] = (ITEM_COLUMN, ITEM_IDX_COLUMN, CATEGORY_COLUMN)
_EPS: Final[float] = 1e-8


class InjectionConfig(BaseModel):
    """Parameters controlling controlled-drift injection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    drift_type: DriftType = "sudden"
    mechanism: DriftMechanism = "anti_profile"
    seed: int = 42
    onset_frac: float = Field(default=0.4, gt=0.0, lt=1.0)
    gradual_window: int = Field(default=10, ge=1)
    recurring_period: int = Field(default=8, ge=1)
    min_test_len: int = Field(default=12, ge=4)
    n_users: int = Field(default=300, ge=1)
    profile_size: int = Field(default=8, ge=1)


class InjectedOnset(BaseModel):
    """Ground-truth record of one user's injected shift."""

    model_config = ConfigDict(extra="forbid")

    user: int
    drift_type: str
    onset_index: int
    onset_timestamp: str
    source_category: str
    target_category: str
    n_replaced: int
    test_length: int


@dataclass(frozen=True)
class ItemLookups:
    """Item metadata, candidate items and their short-term representations."""

    meta: pd.DataFrame  # indexed by s_item -> item_id, item_idx, category
    candidates: NDArray[np.int64]  # candidate s_items
    candidate_reps: NDArray[np.float64]  # (n_candidates, d) short-term reps


def item_meta(train: pd.DataFrame) -> pd.DataFrame:
    """s_item -> (item_id, item_idx, category), indexed by s_item."""
    return (
        train.drop_duplicates(S_ITEM_COLUMN)
        .set_index(S_ITEM_COLUMN)[list(_META_COLUMNS)]
        .sort_index()
    )


def popular_candidates(train: pd.DataFrame, pool_size: int) -> NDArray[np.int64]:
    """Union of the ``pool_size`` most popular training items per category."""
    items: list[int] = []
    for _, group in train.groupby(CATEGORY_COLUMN):
        popular = group[S_ITEM_COLUMN].value_counts().head(pool_size).index
        items.extend(int(s) for s in popular)
    return np.array(sorted(set(items)), dtype=np.int64)


def mid_frequency_candidates(
    train: pd.DataFrame, pool_size: int, skip_top: int = 50
) -> NDArray[np.int64]:
    """Union of mid-frequency items per category (skip the ``skip_top`` most popular).

    Coherent-preference injection deliberately avoids the globally popular items: a
    popular target could be ranked well by the long-term path plus the item-bias
    (popularity) term, letting long-term win post-drift regardless of adaptation. Taking
    the frequency band *below* the head removes that shortcut, so post-drift quality is
    decided by whether the short-term encoder tracks the new cluster.
    """
    items: list[int] = []
    for _, group in train.groupby(CATEGORY_COLUMN):
        ranked = group[S_ITEM_COLUMN].value_counts().index
        band = ranked[skip_top : skip_top + pool_size]
        items.extend(int(s) for s in band)
    return np.array(sorted(set(items)), dtype=np.int64)


class DriftInjector:
    """Inject controlled preference shifts into a test split."""

    def __init__(self, config: InjectionConfig, lookups: ItemLookups) -> None:
        """Store the configuration and the item lookups."""
        self.config = config
        self.lookups = lookups
        self._rng = np.random.default_rng(config.seed)
        self._rep_norm = np.linalg.norm(lookups.candidate_reps, axis=1) + _EPS

    def _anti_profile(self, anchor: NDArray[np.float64]) -> NDArray[np.int64]:
        """Candidate items whose short-term rep is most opposed to the anchor."""
        cos = self.lookups.candidate_reps @ anchor / (
            self._rep_norm * (np.linalg.norm(anchor) + _EPS)
        )
        farthest = np.argsort(cos)[: self.config.profile_size]
        profile: NDArray[np.int64] = self.lookups.candidates[farthest]
        return profile

    def _coherent_profile(self, anchor: NDArray[np.float64]) -> NDArray[np.int64]:
        """A tight, coherent cluster far from the user's anchor (a new preference).

        Seed = the candidate whose short-term rep is farthest from the user's
        established profile (so the shift is real and detectable); the cluster is that
        seed plus its nearest neighbours in rep space (so it is *coherent* — engaging a
        few of its items drives the short-term state toward the cluster, which then ranks
        the rest highly). With a mid-frequency candidate pool this makes the new
        preference short-term-predictable but stale for the long-term profile — the
        regime where adaptation can help.
        """
        cos = self.lookups.candidate_reps @ anchor / (
            self._rep_norm * (np.linalg.norm(anchor) + _EPS)
        )
        seed = int(np.argmin(cos))
        seed_rep = self.lookups.candidate_reps[seed]
        sim = self.lookups.candidate_reps @ seed_rep / (
            self._rep_norm * (np.linalg.norm(seed_rep) + _EPS)
        )
        nearest = np.argsort(sim)[::-1][: self.config.profile_size]
        profile: NDArray[np.int64] = self.lookups.candidates[nearest]
        return profile

    def _profile(self, anchor: NDArray[np.float64]) -> NDArray[np.int64]:
        """Select the target cluster for the configured mechanism."""
        if self.config.mechanism == "coherent":
            return self._coherent_profile(anchor)
        return self._anti_profile(anchor)

    def _post_plan(self, m_post: int) -> NDArray[np.bool_]:
        """Decide, for each post-onset step, whether a target item replaces it."""
        cfg = self.config
        idx = np.arange(m_post)
        if cfg.drift_type == "sudden":
            return np.ones(m_post, dtype=bool)
        if cfg.drift_type == "gradual":
            prob = np.clip(idx / cfg.gradual_window, 0.0, 1.0)
            gradual: NDArray[np.bool_] = self._rng.random(m_post) < prob
            return gradual
        recurring: NDArray[np.bool_] = (idx // cfg.recurring_period) % 2 == 0
        return recurring

    def _inject_user(
        self,
        rows: NDArray[np.int64],
        work: pd.DataFrame,
        new_item: NDArray[np.int64],
        anchor: NDArray[np.float64],
    ) -> InjectedOnset | None:
        """Inject one user's shift in place; return its ground-truth onset."""
        onset = int(len(rows) * self.config.onset_frac)
        if onset < 1 or onset >= len(rows):
            return None
        profile = self._profile(anchor)
        post = rows[onset:]
        replace = self._post_plan(post.size)
        draws = self._rng.choice(profile, size=post.size)
        new_item[post[replace]] = draws[replace]
        pre_cats = work[CATEGORY_COLUMN].to_numpy()[rows[:onset]]
        target_cats = self.lookups.meta.loc[profile, CATEGORY_COLUMN]
        return InjectedOnset(
            user=int(work[S_USER_COLUMN].to_numpy()[rows[0]]),
            drift_type=self.config.drift_type,
            onset_index=onset,
            onset_timestamp=str(work[TIMESTAMP_COLUMN].to_numpy()[rows[onset]]),
            source_category=str(pd.Series(pre_cats).mode().iloc[0]),
            target_category=str(target_cats.mode().iloc[0]),
            n_replaced=int(replace.sum()),
            test_length=int(len(rows)),
        )

    def _eligible_users(
        self, work: pd.DataFrame, anchors: dict[int, NDArray[np.float64]]
    ) -> list[NDArray[np.int64]]:
        """Row-index groups for long-enough users that have an anchor, sampled."""
        groups = work.groupby(S_USER_COLUMN, sort=False).indices
        eligible = [
            np.asarray(rows, dtype=np.int64)
            for user, rows in groups.items()
            if rows.size >= self.config.min_test_len and int(user) in anchors
        ]
        if len(eligible) > self.config.n_users:
            chosen = self._rng.choice(len(eligible), self.config.n_users, replace=False)
            eligible = [eligible[i] for i in sorted(chosen)]
        return eligible

    def inject(
        self, test: pd.DataFrame, anchors: dict[int, NDArray[np.float64]]
    ) -> tuple[pd.DataFrame, list[InjectedOnset]]:
        """Return an injected copy of the test split and its ground-truth onsets."""
        work = test.sort_values(
            [S_USER_COLUMN, TIMESTAMP_COLUMN], kind="stable"
        ).reset_index(drop=True)
        new_item = work[S_ITEM_COLUMN].to_numpy().copy()
        onsets = []
        for rows in self._eligible_users(work, anchors):
            user = int(work[S_USER_COLUMN].to_numpy()[rows[0]])
            onset = self._inject_user(rows, work, new_item, anchors[user])
            if onset is not None:
                onsets.append(onset)
        return self._apply(work, new_item), onsets

    def _apply(
        self, work: pd.DataFrame, new_item: NDArray[np.int64]
    ) -> pd.DataFrame:
        """Write replaced items back with index-consistent metadata."""
        injected = work.copy()
        changed = new_item != work[S_ITEM_COLUMN].to_numpy()
        injected[S_ITEM_COLUMN] = new_item
        meta = self.lookups.meta.loc[new_item[changed]]
        for column in _META_COLUMNS:
            values = injected[column].to_numpy().copy()
            values[changed] = meta[column].to_numpy()
            injected[column] = values
        return injected
