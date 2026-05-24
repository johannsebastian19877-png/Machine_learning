"""Temporal splitting helpers with purging for forward-looking labels."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplits:
    """Container for strict, non-overlapping temporal splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _drop_tail(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.copy()
    if len(df) <= n:
        return df.iloc[0:0].copy()
    return df.iloc[:-n].copy()


def make_purged_splits(
    df: pd.DataFrame,
    *,
    train_end: str = "2023-12-31",
    val_start: str = "2024-01-01",
    val_end: str = "2024-12-31",
    test_start: str = "2025-01-01",
    horizon: int = 0,
) -> TemporalSplits:
    """Return strict time splits and purge rows whose labels cross boundaries.

    The project target is usually built with ``shift(-horizon)``. The last
    ``horizon`` rows of train and validation therefore contain labels whose
    realization belongs to the next split. Purging removes those rows.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must use a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    train = df.loc[:train_end].copy()
    val = df.loc[val_start:val_end].copy()
    test = df.loc[test_start:].copy()

    train = _drop_tail(train, horizon)
    val = _drop_tail(val, horizon)

    _assert_disjoint(train.index, val.index, "train", "val")
    _assert_disjoint(train.index, test.index, "train", "test")
    _assert_disjoint(val.index, test.index, "val", "test")

    return TemporalSplits(train=train, val=val, test=test)


def _assert_disjoint(left: pd.Index, right: pd.Index, left_name: str, right_name: str) -> None:
    overlap = left.intersection(right)
    if len(overlap) > 0:
        sample = ", ".join(str(x) for x in overlap[:3])
        raise ValueError(f"{left_name} and {right_name} overlap: {sample}")


def count_cross_boundary_labels(
    index: pd.DatetimeIndex,
    *,
    split_end: str,
    horizon: int,
) -> int:
    """Count rows at or before ``split_end`` whose row-based horizon crosses it."""

    if horizon <= 0:
        return 0
    if not index.is_monotonic_increasing:
        index = index.sort_values()
    split_end_ts = pd.Timestamp(split_end)
    count = 0
    positions = [i for i, date in enumerate(index) if date <= split_end_ts]
    for i in positions:
        if i + horizon < len(index) and index[i + horizon] > split_end_ts:
            count += 1
    return count
