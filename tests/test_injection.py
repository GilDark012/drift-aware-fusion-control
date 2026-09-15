"""Tests for controlled-drift injection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drift_reco.data.injection import (
    DriftInjector,
    InjectionConfig,
    ItemLookups,
    item_meta,
    mid_frequency_candidates,
    popular_candidates,
)


def _train() -> pd.DataFrame:
    rows = []
    for s_item, cat, pop in [(10, "Books", 5), (11, "Electronics", 4), (12, "Books", 3)]:
        for _ in range(pop):
            rows.append(
                {
                    "user_id": "u",
                    "item_id": f"I{s_item}",
                    "category": cat,
                    "preference_score": 5.0,
                    "timestamp": pd.Timestamp("2016-01-01"),
                    "user_idx": 0,
                    "item_idx": s_item,
                    "s_user": 0,
                    "s_item": s_item,
                }
            )
    return pd.DataFrame(rows)


def _test_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": "u1",
            "item_id": [f"I{10}"] * 6,
            "category": ["Books"] * 6,
            "preference_score": 5.0,
            "timestamp": pd.date_range("2018-01-01", periods=6, freq="D"),
            "user_idx": 1,
            "item_idx": 10,
            "s_user": 1,
            "s_item": [10, 10, 10, 10, 10, 10],
        }
    )


def _lookups() -> ItemLookups:
    train = _train()
    candidates = np.array([10, 11, 12], dtype=np.int64)
    # reps: item 11 is opposite to the anchor [1,0]
    reps = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    return ItemLookups(meta=item_meta(train), candidates=candidates, candidate_reps=reps)


def test_popular_candidates_and_meta() -> None:
    train = _train()
    cands = popular_candidates(train, pool_size=10)
    assert set(cands.tolist()) == {10, 11, 12}
    assert item_meta(train).loc[11, "category"] == "Electronics"


def test_injection_replaces_post_onset_with_anti_profile() -> None:
    lookups = _lookups()
    cfg = InjectionConfig(drift_type="sudden", min_test_len=4, n_users=5, profile_size=1)
    anchors = {1: np.array([1.0, 0.0])}  # item 11 is most opposed
    injected, onsets = DriftInjector(cfg, lookups).inject(_test_frame(), anchors)
    assert len(onsets) == 1
    onset = onsets[0]
    assert onset.onset_index == int(6 * 0.4)  # == 2
    post = injected.sort_values("timestamp").iloc[onset.onset_index :]
    # all post-onset items switched to item 11 (Electronics) — index-consistent
    assert (post["s_item"] == 11).all()
    assert (post["category"] == "Electronics").all()
    assert (post["item_idx"] == 11).all()


def test_short_users_are_not_injected() -> None:
    lookups = _lookups()
    cfg = InjectionConfig(min_test_len=100, n_users=5)
    _, onsets = DriftInjector(cfg, lookups).inject(_test_frame(), {1: np.array([1.0, 0.0])})
    assert onsets == []


def test_mid_frequency_candidates_skips_the_most_popular() -> None:
    train = _train()  # Books: 10 (pop 5) > 12 (pop 3); Electronics: 11 (pop 4)
    mid = mid_frequency_candidates(train, pool_size=10, skip_top=1)
    assert 10 not in mid.tolist()  # the most popular Books item is dropped
    assert 12 in mid.tolist()  # the next Books item remains


def test_coherent_injection_uses_a_coherent_cluster() -> None:
    lookups = _lookups()  # reps: 10=[1,0], 11=[-1,0], 12=[0,1]
    cfg = InjectionConfig(
        drift_type="sudden", mechanism="coherent", min_test_len=4, n_users=5,
        profile_size=2,
    )
    anchors = {1: np.array([1.0, 0.0])}  # seed = item 11 (farthest); cluster {11, 12}
    injected, onsets = DriftInjector(cfg, lookups).inject(_test_frame(), anchors)
    assert len(onsets) == 1
    post = injected.sort_values("timestamp").iloc[onsets[0].onset_index :]
    assert set(post["s_item"].tolist()) <= {11, 12}  # only cluster items
    assert (post["s_item"] != 10).all()  # never the user's original preference
