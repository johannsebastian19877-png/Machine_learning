"""Tests for strict temporal splits and label purging."""
import pandas as pd

from src.splits import count_cross_boundary_labels, make_purged_splits


def test_make_purged_splits_are_disjoint_and_purged():
    idx = pd.bdate_range("2023-12-20", "2025-01-14")
    df = pd.DataFrame({"x": range(len(idx))}, index=idx)

    splits = make_purged_splits(
        df,
        train_end="2023-12-31",
        val_start="2024-01-01",
        val_end="2024-12-31",
        test_start="2025-01-01",
        horizon=5,
    )

    assert splits.train.index.max() < pd.Timestamp("2023-12-31")
    assert splits.val.index.min() >= pd.Timestamp("2024-01-01")
    assert splits.val.index.max() < pd.Timestamp("2024-12-31")
    assert splits.test.index.min() >= pd.Timestamp("2025-01-01")
    assert splits.train.index.intersection(splits.val.index).empty
    assert splits.val.index.intersection(splits.test.index).empty


def test_count_cross_boundary_labels_detects_leaky_tail():
    idx = pd.bdate_range("2024-12-20", "2025-01-14")

    count = count_cross_boundary_labels(idx, split_end="2024-12-31", horizon=5)

    assert count == 5
