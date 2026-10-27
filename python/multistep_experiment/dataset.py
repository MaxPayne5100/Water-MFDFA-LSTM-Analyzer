"""Leakage-aware direct multi-step dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from data_quality import (
        DataQualityError,
        validate_feature_file,
        validate_output_lineage,
        validate_spectrum_label_file,
        validate_time_series_file,
    )
    from utils import load_config
except ImportError:  # pragma: no cover - package import fallback
    from ..data_quality import (
        DataQualityError,
        validate_feature_file,
        validate_output_lineage,
        validate_spectrum_label_file,
        validate_time_series_file,
    )
    from ..utils import load_config

from .constants import (
    BASE_FEATURE_COLUMNS,
    SPECTRUM_LOG_COLUMNS,
    SPECTRUM_TARGET_COLUMNS,
)


@dataclass(frozen=True)
class MultiStepDataset:
    features: np.ndarray
    values: np.ndarray
    spectra: np.ndarray
    spectrum_mask: np.ndarray
    sample_index: pd.DataFrame
    raw_values: np.ndarray
    raw_dates: pd.DatetimeIndex
    feature_columns: tuple[str, ...]
    horizon: int
    lookback: int
    evaluation_start: int
    evaluation_origins: np.ndarray

    @property
    def development_positions(self) -> np.ndarray:
        return np.flatnonzero(
            self.sample_index["target_end"].to_numpy() < self.evaluation_start
        )

    @property
    def evaluation_positions(self) -> np.ndarray:
        target_starts = self.sample_index["target_start"].to_numpy()
        positions = []
        for origin in self.evaluation_origins:
            matches = np.flatnonzero(target_starts == int(origin))
            if len(matches) != 1:
                raise DataQualityError(
                    f"Expected exactly one sample for evaluation origin {origin}"
                )
            positions.append(int(matches[0]))
        return np.asarray(positions, dtype=int)

    def fractional_context(self, positions: Sequence[int], length: int) -> np.ndarray:
        contexts = []
        for position in positions:
            target_start = int(self.sample_index.iloc[int(position)]["target_start"])
            context_start = target_start - length
            if context_start < 0:
                raise DataQualityError(
                    f"Sample at raw index {target_start} lacks {length} context values"
                )
            contexts.append(self.raw_values[context_start:target_start])
        return np.asarray(contexts, dtype=np.float32)[..., np.newaxis]


@dataclass
class FoldTransformers:
    feature_scaler: StandardScaler
    value_scaler: StandardScaler
    spectrum_scaler: StandardScaler | None

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        values: np.ndarray,
        spectra: np.ndarray | None = None,
        spectrum_mask: np.ndarray | None = None,
    ) -> "FoldTransformers":
        feature_scaler = StandardScaler().fit(features.reshape(-1, features.shape[-1]))
        value_scaler = StandardScaler().fit(values.reshape(-1, 1))
        spectrum_scaler = None
        if spectra is not None:
            transformed = transform_spectra(spectra, spectrum_mask)
            flat = transformed.reshape(-1, transformed.shape[-1])
            if spectrum_mask is not None:
                valid_rows = spectrum_mask.reshape(-1, spectrum_mask.shape[-1]).all(axis=1)
                flat = flat[valid_rows]
            if len(flat) == 0:
                raise DataQualityError("No valid spectrum labels are available for scaling")
            spectrum_scaler = StandardScaler().fit(flat)
        return cls(feature_scaler, value_scaler, spectrum_scaler)

    def transform_features(self, values: np.ndarray) -> np.ndarray:
        shape = values.shape
        return self.feature_scaler.transform(values.reshape(-1, shape[-1])).reshape(shape)

    def transform_values(self, values: np.ndarray) -> np.ndarray:
        return self.value_scaler.transform(values.reshape(-1, 1)).reshape(values.shape)

    def inverse_values(self, values: np.ndarray) -> np.ndarray:
        return self.value_scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)

    def transform_spectra(
        self, values: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        if self.spectrum_scaler is None:
            raise RuntimeError("Spectrum scaler was not fitted")
        transformed = transform_spectra(values, mask)
        shape = transformed.shape
        return self.spectrum_scaler.transform(
            transformed.reshape(-1, shape[-1])
        ).reshape(shape)


def _project_path(config_path: Path, configured_value: str) -> Path:
    value = Path(configured_value)
    return value if value.is_absolute() else config_path.parent.parent / value


def transform_spectra(
    values: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    transformed = np.asarray(values, dtype=float).copy()
    valid_mask = (
        np.ones_like(transformed, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    if valid_mask.shape != transformed.shape:
        raise ValueError("Spectrum mask must have the same shape as spectrum values")
    column_positions = {name: index for index, name in enumerate(SPECTRUM_TARGET_COLUMNS)}
    for column in SPECTRUM_LOG_COLUMNS:
        position = column_positions[column]
        valid = valid_mask[..., position]
        if np.any(transformed[..., position][valid] <= 0):
            raise DataQualityError(f"Spectrum log target must be positive: {column}")
        transformed[..., position] = np.where(
            valid,
            np.log(np.where(valid, transformed[..., position], 1.0)),
            0.0,
        )
    return transformed


def load_multistep_dataset(
    config_path: str | Path,
    *,
    lookback: int,
) -> MultiStepDataset:
    """Load direct targets and configured rolling evaluation origins."""
    config_file = Path(config_path).resolve()
    config = load_config(str(config_file))
    indicator = config["INDICATOR"]
    river = config["RIVER_NAME_VALUE_ENG"]
    horizon = int(config["HORIZON"])
    evaluation_origin_count = int(config.get("EVALUATION_ORIGIN_COUNT", 1))
    evaluation_origin_stride = int(
        config.get("EVALUATION_ORIGIN_STRIDE", horizon)
    )
    mfdfa_window = int(config["MFDFA_WINDOW"])
    if (
        horizon < 1
        or lookback < 1
        or evaluation_origin_count < 1
        or evaluation_origin_stride < 1
    ):
        raise ValueError(
            "HORIZON, lookback, EVALUATION_ORIGIN_COUNT, and "
            "EVALUATION_ORIGIN_STRIDE must be positive"
        )

    curated = _project_path(config_file, config["CURATED_FOLDER"])
    features_folder = _project_path(config_file, config["FEATURES_FOLDER"])
    time_series_path = curated / f"{river}_{indicator}_ts.csv"
    feature_path = features_folder / f"MFDFA_{river}_{indicator}_{mfdfa_window}_window.csv"
    spectrum_path = (
        features_folder
        / f"SpectrumLabels_{river}_{indicator}_{mfdfa_window}_window.csv"
    )

    validate_time_series_file(time_series_path, indicator, horizon)
    validate_output_lineage(time_series_path, config_file, "curated")
    time_series = pd.read_csv(time_series_path)
    validate_feature_file(
        feature_path,
        BASE_FEATURE_COLUMNS,
        time_series_rows=len(time_series),
        mfdfa_window=mfdfa_window,
    )
    validate_output_lineage(feature_path, config_file, "features_mfdfa")
    validate_spectrum_label_file(
        spectrum_path,
        time_series_rows=len(time_series),
        mfdfa_window=mfdfa_window,
    )
    validate_output_lineage(spectrum_path, config_file, "features_spectrum")

    feature_frame = pd.read_csv(feature_path)
    feature_frame = feature_frame.rename(columns={feature_frame.columns[0]: "Index"})
    feature_frame["Index"] = pd.to_numeric(feature_frame["Index"], errors="raise").astype(int)
    feature_frame = feature_frame.set_index("Index", verify_integrity=True)

    spectrum_frame = pd.read_csv(spectrum_path)
    spectrum_frame = spectrum_frame.rename(columns={spectrum_frame.columns[0]: "Index"})
    spectrum_frame["Index"] = pd.to_numeric(
        spectrum_frame["Index"], errors="raise"
    ).astype(int)
    spectrum_frame = spectrum_frame.set_index("Index", verify_integrity=True)

    raw_values = pd.to_numeric(time_series[indicator], errors="raise").to_numpy(dtype=float)
    raw_dates = pd.DatetimeIndex(pd.to_datetime(time_series["date"], errors="raise"))
    first_feature_index = int(feature_frame.index.min())
    final_origin = len(raw_values) - horizon
    first_origin = first_feature_index + lookback - 1
    if final_origin < first_origin:
        raise DataQualityError(
            "No complete direct multi-step sample exists for the configured horizon"
        )
    evaluation_start = final_origin - (
        evaluation_origin_count - 1
    ) * evaluation_origin_stride
    if evaluation_start < first_origin:
        raise DataQualityError(
            "Configured rolling evaluation origins begin before the first aligned "
            "multi-step sample"
        )
    evaluation_origins = np.arange(
        evaluation_start,
        final_origin + 1,
        evaluation_origin_stride,
        dtype=int,
    )
    if len(evaluation_origins) != evaluation_origin_count:
        raise DataQualityError("Failed to construct the configured evaluation origins")

    feature_samples: list[np.ndarray] = []
    value_samples: list[np.ndarray] = []
    spectrum_samples: list[np.ndarray] = []
    spectrum_masks: list[np.ndarray] = []
    index_records: list[dict[str, int]] = []

    for target_start in range(first_origin, final_origin + 1):
        feature_indices = np.arange(target_start - lookback + 1, target_start + 1)
        target_indices = np.arange(target_start, target_start + horizon)
        spectrum_indices = target_indices + 1
        if not np.isin(feature_indices, feature_frame.index).all():
            continue
        if not np.isin(spectrum_indices, spectrum_frame.index).all():
            continue

        feature_values = feature_frame.loc[
            feature_indices, list(BASE_FEATURE_COLUMNS)
        ].to_numpy(dtype=float)
        spectrum_values = spectrum_frame.loc[
            spectrum_indices, list(SPECTRUM_TARGET_COLUMNS)
        ].to_numpy(dtype=float)
        converged = (
            spectrum_frame.loc[spectrum_indices, "Spectrum_Fit_Converged"]
            .astype(str)
            .str.lower()
            .isin(("true", "1", "yes"))
            .to_numpy()
        )
        finite_mask = np.isfinite(spectrum_values)
        mask = finite_mask & converged[:, np.newaxis]
        spectrum_values = np.where(mask, spectrum_values, 0.0)

        feature_samples.append(feature_values)
        value_samples.append(raw_values[target_indices])
        spectrum_samples.append(spectrum_values)
        spectrum_masks.append(mask)
        index_records.append(
            {
                "forecast_origin": target_start - 1,
                "feature_start": int(feature_indices[0]),
                "feature_end": int(feature_indices[-1]),
                "feature_available_through": int(feature_indices[-1] - 1),
                "target_start": int(target_indices[0]),
                "target_end": int(target_indices[-1]),
                "spectrum_start": int(spectrum_indices[0]),
                "spectrum_end": int(spectrum_indices[-1]),
            }
        )

    sample_index = pd.DataFrame(index_records)
    if sample_index.empty:
        raise DataQualityError("No aligned multi-step samples were generated")
    if not (
        sample_index["feature_available_through"] < sample_index["target_start"]
    ).all():
        raise DataQualityError("Feature/target leakage detected in sample alignment")
    if not (
        sample_index["target_end"] - sample_index["target_start"] + 1 == horizon
    ).all():
        raise DataQualityError("Generated target interval has the wrong horizon")

    return MultiStepDataset(
        features=np.asarray(feature_samples, dtype=np.float32),
        values=np.asarray(value_samples, dtype=np.float32)[..., np.newaxis],
        spectra=np.asarray(spectrum_samples, dtype=np.float32),
        spectrum_mask=np.asarray(spectrum_masks, dtype=np.float32),
        sample_index=sample_index,
        raw_values=raw_values.astype(np.float32),
        raw_dates=raw_dates,
        feature_columns=tuple(BASE_FEATURE_COLUMNS),
        horizon=horizon,
        lookback=lookback,
        evaluation_start=int(evaluation_origins[0]),
        evaluation_origins=evaluation_origins,
    )
