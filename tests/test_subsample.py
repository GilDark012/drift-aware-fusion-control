"""Tests for the development subsample builder."""

from __future__ import annotations

from pathlib import Path

from drift_reco.config.constants import OOV_IDX, S_ITEM_COLUMN
from drift_reco.data.schema import Split
from drift_reco.data.subsample import (
    SubsampleBuilder,
    SubsampleConfig,
    read_subsample,
)


def _build(root: Path, out: Path, **kw: object) -> SubsampleConfig:
    params: dict[str, object] = {
        "processed_dir": root / "processed",
        "out_dir": out,
        "min_test_interactions": 1,
        "target_users": 10,
        "activity_strata": 1,
        "extra_train_users": 0,
        "k_core": 1,  # disable k-core for the tiny synthetic fixtures
    }
    params.update(kw)
    cfg = SubsampleConfig(**params)  # type: ignore[arg-type]
    SubsampleBuilder(cfg).build()
    return cfg


def test_subsample_persists_and_reloads(synthetic_dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "sub"
    _build(synthetic_dataset, out)
    frames, manifest = read_subsample(out)
    assert manifest.selected_users >= 1
    assert set(frames) == set(Split)
    assert manifest.n_item_vocab >= 2  # reserved ids at least


def test_category_filter_restricts_to_one_category(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    """A category filter keeps only that category's rows across every split."""
    out = tmp_path / "sub"
    _build(synthetic_dataset, out, category="Books")
    frames, _ = read_subsample(out)
    for frame in frames.values():
        assert (frame["category"] == "Books").all()


def test_unseen_test_item_maps_to_oov(synthetic_dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "sub"
    _build(synthetic_dataset, out)
    frames, _ = read_subsample(out)
    # item i9 appears only in test → must be OOV in the dense vocab.
    assert (frames[Split.TEST][S_ITEM_COLUMN] == OOV_IDX).any()


def test_train_items_are_never_oov(synthetic_dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "sub"
    _build(synthetic_dataset, out)
    frames, _ = read_subsample(out)
    assert not (frames[Split.TRAIN][S_ITEM_COLUMN] == OOV_IDX).any()


def test_kcore_filtering_shrinks_vocabulary(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    _build(synthetic_dataset, tmp_path / "k1", k_core=1)
    _, low = read_subsample(tmp_path / "k1")
    _build(synthetic_dataset, tmp_path / "k2", k_core=2)
    _, high = read_subsample(tmp_path / "k2")
    # dropping items with <2 train interactions removes some of the vocabulary
    assert high.n_item_vocab < low.n_item_vocab
