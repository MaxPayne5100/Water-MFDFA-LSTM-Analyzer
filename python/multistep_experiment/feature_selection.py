"""Fold-local multi-output feature ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


def _negative_global_rmse(estimator, features: np.ndarray, targets: np.ndarray) -> float:
    predictions = estimator.predict(features)
    return -float(np.sqrt(np.mean(np.square(predictions - targets))))


def rank_features(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    seed: int,
    target_starts: np.ndarray | None = None,
    target_ends: np.ndarray | None = None,
    repeats: int = 10,
) -> pd.DataFrame:
    """Rank features with a purged internal split when intervals permit one."""
    if len(features) < 5:
        raise ValueError("Feature ranking requires at least five samples")
    origin_features = features[:, -1, :]
    flat_targets = targets.reshape(len(targets), -1)
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
    )
    ranking_method = "in_sample_impurity"
    ranking_train_samples = len(features)
    ranking_validation_samples = 0

    if (target_starts is None) != (target_ends is None):
        raise ValueError("target_starts and target_ends must be supplied together")
    if target_starts is not None:
        starts = np.asarray(target_starts, dtype=int).reshape(-1)
        ends = np.asarray(target_ends, dtype=int).reshape(-1)
        if len(starts) != len(features) or len(ends) != len(features):
            raise ValueError("Feature-ranking intervals must match the sample count")
        if np.any(np.diff(starts) < 0):
            raise ValueError("Feature-ranking samples must be chronologically ordered")

        # Tiny technically non-overlapping splits are too unstable to support a
        # scientific ranking. If the outer fold cannot provide both sides of
        # this minimum, use all outer-training rows for impurity ranking instead.
        minimum_train_samples = 10
        minimum_validation_samples = 5
        preferred = max(minimum_train_samples, int(len(features) * 0.8))
        candidates: list[tuple[int, int, int]] = []
        for split in range(
            minimum_validation_samples,
            len(features) - minimum_validation_samples + 1,
        ):
            validation_start = int(starts[split])
            purged_train = np.flatnonzero(ends[:split] < validation_start)
            validation_count = len(features) - split
            if (
                len(purged_train) >= minimum_train_samples
                and validation_count >= minimum_validation_samples
            ):
                candidates.append((abs(split - preferred), split, len(purged_train)))
        if candidates:
            _, split, purged_train_count = min(candidates)
            validation_start = int(starts[split])
            purged_train = np.flatnonzero(ends[:split] < validation_start)
            train_x = origin_features[purged_train]
            train_y = flat_targets[purged_train]
            validation_x = origin_features[split:]
            validation_y = flat_targets[split:]
            model.fit(train_x, train_y)
            result = permutation_importance(
                model,
                validation_x,
                validation_y,
                scoring=_negative_global_rmse,
                n_repeats=repeats,
                random_state=seed,
                n_jobs=-1,
            )
            importance = result.importances_mean
            ranking_method = "purged_permutation"
            ranking_train_samples = purged_train_count
            ranking_validation_samples = len(validation_x)
        else:
            model.fit(origin_features, flat_targets)
            importance = model.feature_importances_
    else:
        # Retained for isolated callers without interval metadata. Production
        # experiment paths always provide intervals.
        split = max(1, int(len(features) * 0.8))
        train_x, validation_x = origin_features[:split], origin_features[split:]
        train_y, validation_y = flat_targets[:split], flat_targets[split:]
        if len(validation_x) >= 2:
            model.fit(train_x, train_y)
            result = permutation_importance(
                model,
                validation_x,
                validation_y,
                scoring=_negative_global_rmse,
                n_repeats=repeats,
                random_state=seed,
                n_jobs=-1,
            )
            importance = result.importances_mean
            ranking_method = "unpurged_permutation_no_intervals"
            ranking_train_samples = len(train_x)
            ranking_validation_samples = len(validation_x)
        else:
            model.fit(origin_features, flat_targets)
            importance = model.feature_importances_

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
                "ranking_method": ranking_method,
                "ranking_train_samples": ranking_train_samples,
                "ranking_validation_samples": ranking_validation_samples,
            }
        )
        .sort_values(["importance", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )


def feature_positions(
    ranking: pd.DataFrame,
    feature_names: tuple[str, ...],
    top_k: int,
) -> tuple[np.ndarray, list[str]]:
    selected = ranking["feature"].head(top_k).tolist()
    lookup = {name: index for index, name in enumerate(feature_names)}
    return np.asarray([lookup[name] for name in selected], dtype=int), selected
