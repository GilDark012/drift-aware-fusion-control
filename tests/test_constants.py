"""Tests for project-wide constants."""

from __future__ import annotations

from drift_reco.config import constants


def test_interaction_columns_are_the_real_schema() -> None:
    assert constants.RATING_COLUMN == "preference_score"
    expected = ("user_id", "item_id", "category", "timestamp", "user_idx", "item_idx")
    for column in expected:
        assert column in constants.INTERACTION_COLUMNS


def test_three_categories_are_declared() -> None:
    assert len(constants.CATEGORIES) == 3
    assert "Books" in constants.CATEGORIES
