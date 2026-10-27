"""Interval-aware purged walk-forward validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedFold:
    number: int
    train_positions: np.ndarray
    validation_positions: np.ndarray
    validation_target_start: int
    validation_target_end: int


def purged_walk_forward_splits(
    sample_index: pd.DataFrame,
    development_positions: np.ndarray,
    *,
    n_splits: int = 3,
    min_train_origins: int = 1,
) -> list[PurgedFold]:
    """Create expanding folds after enough fully purged training origins exist."""
    ordered_positions = np.asarray(development_positions, dtype=int)
    if n_splits < 1 or min_train_origins < 1:
        raise ValueError("n_splits and min_train_origins must be positive")
    if len(ordered_positions) <= n_splits:
        raise ValueError("Not enough development samples for walk-forward splitting")

    target_starts = sample_index.iloc[ordered_positions]["target_start"].to_numpy()
    target_ends = sample_index.iloc[ordered_positions]["target_end"].to_numpy()
    eligible_local = []
    for local_position, validation_start in enumerate(target_starts):
        train_count = int(np.sum(target_ends < validation_start))
        if train_count >= min_train_origins:
            eligible_local.append(local_position)
    if len(eligible_local) < n_splits:
        raise ValueError(
            "Purged walk-forward splitting cannot provide "
            f"{n_splits} folds with at least {min_train_origins} training origins"
        )

    validation_blocks = np.array_split(
        ordered_positions[np.asarray(eligible_local, dtype=int)], n_splits
    )
    folds: list[PurgedFold] = []
    for number, validation in enumerate(validation_blocks, start=1):
        validation_start = int(
            sample_index.iloc[validation]["target_start"].min()
        )
        validation_end = int(sample_index.iloc[validation]["target_end"].max())
        train_target_end = sample_index.iloc[ordered_positions]["target_end"].to_numpy()
        train = ordered_positions[train_target_end < validation_start]
        if len(train) < min_train_origins:
            raise AssertionError("Fold contains fewer than the required training origins")
        if np.any(
            sample_index.iloc[train]["target_end"].to_numpy() >= validation_start
        ):
            raise AssertionError("Purging failed to remove overlapping targets")
        folds.append(
            PurgedFold(number, train, validation, validation_start, validation_end)
        )
    if len(folds) != n_splits:
        raise AssertionError("Purged walk-forward splitting returned the wrong fold count")
    return folds
