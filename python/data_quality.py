"""Data-quality contracts for the Python/R pipeline boundaries.

The functions in this module deliberately raise ``DataQualityError`` instead
of returning a boolean.  This makes a failed validation propagate to the
process exit code and allows Airflow to mark the task as failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from lineage import (
        lineage_sidecar_path,
        load_lineage_context,
        resolve_input_manifest,
        sha256_file,
    )
    from raw_ingestion import (
        DEFAULT_SOURCE_FILE_PREFIXES,
        active_records,
        active_snapshot_checksum,
        load_active_raw_files,
    )
except ImportError:  # pragma: no cover - supports package imports in tests
    from .lineage import (
        lineage_sidecar_path,
        load_lineage_context,
        resolve_input_manifest,
        sha256_file,
    )
    from .raw_ingestion import (
        DEFAULT_SOURCE_FILE_PREFIXES,
        active_records,
        active_snapshot_checksum,
        load_active_raw_files,
    )


class DataQualityError(ValueError):
    """Raised when a pipeline data contract is violated."""


def validate_output_lineage(
    output_path: str | Path,
    config_path: str | Path,
    expected_stage: str,
) -> dict[str, str]:
    """Validate the output sidecar and its raw-manifest identity."""
    output = Path(output_path).resolve()
    sidecar = lineage_sidecar_path(output)
    if not sidecar.is_file():
        raise DataQualityError(
            f"Lineage metadata does not exist for output: {output}"
        )

    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataQualityError(f"Could not read lineage metadata: {sidecar}") from exc

    if metadata.get("stage") != expected_stage:
        raise DataQualityError(
            f"{sidecar} has stage {metadata.get('stage')!r}; "
            f"expected {expected_stage!r}"
        )

    try:
        current_context = load_lineage_context(config_path)
        output_metadata = metadata["output"]
        run_id = metadata["run_id"]
        manifest_checksum = metadata["input_manifest_checksum"]
        output_checksum = output_metadata["sha256"]
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        raise DataQualityError(f"Incomplete lineage metadata: {sidecar}") from exc

    if run_id != current_context["run_id"]:
        raise DataQualityError(
            f"{output} belongs to run {run_id!r}, not current run "
            f"{current_context['run_id']!r}"
        )
    if manifest_checksum != current_context["input_manifest_checksum"]:
        raise DataQualityError(
            f"{output} input-manifest checksum does not match the current "
            "raw-ingestion manifest"
        )
    if output_checksum != sha256_file(output):
        raise DataQualityError(f"Output checksum mismatch: {output}")

    return {
        "run_id": run_id,
        "input_manifest_checksum": manifest_checksum,
    }


def _require_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    missing = sorted(set(required_columns).difference(dataframe.columns))
    if missing:
        raise DataQualityError(
            f"{dataset_name} is missing required column(s): {missing}"
        )


def _require_non_empty(dataframe: pd.DataFrame, dataset_name: str) -> None:
    if dataframe.empty:
        raise DataQualityError(f"{dataset_name} contains no data rows")


def _require_finite_numeric(
    series: pd.Series,
    column_name: str,
    dataset_name: str,
) -> None:
    numeric_values = pd.to_numeric(series, errors="coerce")
    if numeric_values.isna().any():
        invalid_count = int(numeric_values.isna().sum())
        raise DataQualityError(
            f"{dataset_name}.{column_name} contains {invalid_count} non-numeric or missing value(s)"
        )
    if not np.isfinite(numeric_values.to_numpy(dtype=float)).all():
        raise DataQualityError(
            f"{dataset_name}.{column_name} contains non-finite value(s)"
        )


def source_data_files(download_folder: str | Path) -> list[Path]:
    """Return monitoring input files, including the known source typo."""
    folder = Path(download_folder)
    if not folder.exists() or not folder.is_dir():
        raise DataQualityError(f"Raw data directory does not exist: {folder}")

    prefixes = ("monitoring", "monitiring")
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.name.lower().endswith(".csv")
        and path.name.lower().startswith(prefixes)
    )


def validate_raw_directory(
    download_folder: str | Path,
    required_columns: Sequence[str],
) -> dict[str, int]:
    """Validate every downloaded monitoring CSV without loading it all at once."""
    files = source_data_files(download_folder)
    if not files:
        raise DataQualityError(
            f"No monitoring CSV files found in raw data directory: {download_folder}"
        )

    total_rows = 0
    for path in files:
        if path.stat().st_size == 0:
            raise DataQualityError(f"Raw input file is empty: {path}")

        try:
            header = pd.read_csv(path, sep=";", nrows=0)
            _require_columns(header, required_columns, str(path))
            file_rows = 0
            for chunk in pd.read_csv(
                path,
                sep=";",
                usecols=list(required_columns),
                chunksize=10_000,
            ):
                file_rows += len(chunk)
        except DataQualityError:
            raise
        except Exception as exc:
            raise DataQualityError(f"Could not read raw input file: {path}") from exc

        if file_rows == 0:
            raise DataQualityError(f"Raw input file contains no rows: {path}")
        total_rows += file_rows

    return {"files": len(files), "rows": total_rows}


def validate_raw_ingestion_manifest(
    raw_folder: str | Path,
    manifest_folder: str | Path,
    source_filename_prefixes: Sequence[str] = DEFAULT_SOURCE_FILE_PREFIXES,
    input_manifest_path: str | Path | None = None,
) -> dict[str, int | str]:
    """Verify raw files and the active snapshot selected by the latest manifest.

    Historical raw files remain immutable and are allowed to stay in the
    directory. Only the manifest's active records are eligible inputs for
    bronze processing; their files must exist and match their recorded hashes.
    """
    if input_manifest_path is not None:
        manifest_path = Path(input_manifest_path).resolve()
    else:
        manifest_directory = Path(manifest_folder) / "raw_ingestion"
        manifests = sorted(
            manifest_directory.glob("raw_ingestion_*.json"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        if not manifests:
            raise DataQualityError(
                f"No raw-ingestion manifest found in {manifest_directory}"
            )
        manifest_path = manifests[-1]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataQualityError(
            f"Could not read raw-ingestion manifest: {manifest_path}"
        ) from exc

    try:
        selected_records = active_records(manifest)
        selected_files = load_active_raw_files(manifest_path, raw_folder)
        selected_source_files = load_active_raw_files(
            manifest_path,
            raw_folder,
            filename_prefixes=source_filename_prefixes,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise DataQualityError(
            f"Invalid active raw snapshot in {manifest_path.name}: {exc}"
        ) from exc

    calculated_snapshot_checksum = active_snapshot_checksum(manifest)
    stored_snapshot_checksum = manifest.get("active_snapshot_checksum")
    if stored_snapshot_checksum and stored_snapshot_checksum != calculated_snapshot_checksum:
        raise DataQualityError(
            f"Active snapshot checksum mismatch in {manifest_path.name}"
        )
    calculated_source_snapshot_checksum = active_snapshot_checksum(
        manifest,
        source_filename_prefixes,
    )
    stored_source_snapshot_checksum = manifest.get("active_source_snapshot_checksum")
    if (
        stored_source_snapshot_checksum
        and stored_source_snapshot_checksum != calculated_source_snapshot_checksum
    ):
        raise DataQualityError(
            f"Active analytical-source checksum mismatch in {manifest_path.name}"
        )

    records = {
        record.get("stored_filename"): record
        for record in manifest.get("records", [])
        if record.get("stored_filename")
    }
    raw_files = sorted(Path(raw_folder).glob("*.csv"))
    if not raw_files:
        raise DataQualityError(f"No raw CSV files found in {raw_folder}")

    unmatched_legacy = 0
    for raw_file in raw_files:
        record = records.get(raw_file.name)
        if record is None:
            raise DataQualityError(
                f"Raw file is not listed in manifest {manifest_path.name}: {raw_file.name}"
            )
        if not record.get("original_filename"):
            raise DataQualityError(
                f"Manifest record lacks original_filename: {raw_file.name}"
            )
        if not record.get("downloaded_at_utc"):
            raise DataQualityError(
                f"Manifest record lacks downloaded_at_utc: {raw_file.name}"
            )
        actual_checksum = hashlib.sha256(raw_file.read_bytes()).hexdigest()
        if actual_checksum != record.get("sha256"):
            raise DataQualityError(
                f"Checksum mismatch for raw file: {raw_file.name}"
            )
        if record.get("status") == "legacy_unmatched":
            unmatched_legacy += 1
        elif not record.get("source_url") or not record.get("resource_id"):
            raise DataQualityError(
                f"Manifest record lacks source_url/resource_id: {raw_file.name}"
            )

    return {
        "files": len(raw_files),
        "manifest_records": len(records),
        "legacy_unmatched": unmatched_legacy,
        "active_manifest_records": len(selected_records),
        "active_files": len(selected_files),
        "active_source_files": len(selected_source_files),
        "active_snapshot_checksum": calculated_snapshot_checksum,
        "active_source_snapshot_checksum": calculated_source_snapshot_checksum,
    }


def validate_combined_file(
    file_path: str | Path,
    required_columns: Sequence[str],
    *,
    non_null_columns: Sequence[str] = (),
    at_least_one_non_null_columns: Sequence[str] = (),
    filter_column: str | None = None,
    filter_value: str | None = None,
) -> dict[str, int]:
    """Validate a combined or filtered monitoring table."""
    path = Path(file_path)
    if not path.is_file():
        raise DataQualityError(f"Combined data file does not exist: {path}")

    try:
        dataframe = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise DataQualityError(f"Could not read combined data file: {path}") from exc

    dataset_name = str(path)
    _require_non_empty(dataframe, dataset_name)
    _require_columns(dataframe, required_columns, dataset_name)

    for column in non_null_columns:
        null_count = int(dataframe[column].isna().sum())
        if null_count:
            raise DataQualityError(
                f"{dataset_name}.{column} contains {null_count} missing value(s)"
            )

    for column in at_least_one_non_null_columns:
        if not dataframe[column].notna().any():
            raise DataQualityError(
                f"{dataset_name}.{column} contains no usable non-null values"
            )

    if filter_column is not None and filter_value is not None:
        matches = dataframe[filter_column].astype("string").str.contains(
            filter_value,
            regex=False,
            na=False,
        )
        if not matches.all():
            raise DataQualityError(
                f"{dataset_name} contains rows outside the configured filter "
                f"{filter_column!r} contains {filter_value!r}"
            )

    return {"rows": len(dataframe), "columns": len(dataframe.columns)}


def validate_time_series_file(
    file_path: str | Path,
    indicator: str,
    horizon: int,
) -> dict[str, int]:
    """Validate the prepared date/value time-series contract."""
    path = Path(file_path)
    if not path.is_file():
        raise DataQualityError(f"Time-series file does not exist: {path}")

    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:
        raise DataQualityError(f"Could not read time-series file: {path}") from exc

    dataset_name = str(path)
    _require_non_empty(dataframe, dataset_name)
    _require_columns(dataframe, ("date", indicator), dataset_name)

    dates = pd.to_datetime(dataframe["date"], errors="coerce")
    if dates.isna().any():
        raise DataQualityError(f"{dataset_name}.date contains invalid date(s)")
    if dates.duplicated().any():
        raise DataQualityError(f"{dataset_name}.date contains duplicate date(s)")
    if not dates.is_monotonic_increasing:
        raise DataQualityError(f"{dataset_name}.date is not sorted ascending")

    _require_finite_numeric(dataframe[indicator], indicator, dataset_name)
    if len(dataframe) <= horizon:
        raise DataQualityError(
            f"{dataset_name} has {len(dataframe)} rows; it must contain more "
            f"than the forecast horizon ({horizon})"
        )

    return {"rows": len(dataframe), "horizon": horizon}


def validate_feature_file(
    file_path: str | Path,
    feature_columns: Sequence[str],
    *,
    time_series_rows: int,
    mfdfa_window: int,
) -> dict[str, int]:
    """Validate MFDFA features and their expected rolling-window indices."""
    path = Path(file_path)
    if not path.is_file():
        raise DataQualityError(f"Feature file does not exist: {path}")

    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:
        raise DataQualityError(f"Could not read feature file: {path}") from exc

    dataset_name = str(path)
    _require_non_empty(dataframe, dataset_name)
    _require_columns(dataframe, feature_columns, dataset_name)
    for column in feature_columns:
        _require_finite_numeric(dataframe[column], column, dataset_name)

    if time_series_rows <= mfdfa_window:
        raise DataQualityError(
            f"Time series has {time_series_rows} rows, which is not enough for "
            f"MFDFA window {mfdfa_window}"
        )

    expected_rows = time_series_rows - mfdfa_window + 1
    if len(dataframe) != expected_rows:
        raise DataQualityError(
            f"{dataset_name} has {len(dataframe)} rows; expected {expected_rows} "
            f"for {time_series_rows} time-series rows and window {mfdfa_window}"
        )

    index_column = dataframe.columns[0]
    indices = pd.to_numeric(dataframe[index_column], errors="coerce")
    expected_indices = np.arange(mfdfa_window, time_series_rows + 1)
    if indices.isna().any() or not np.array_equal(
        indices.to_numpy(dtype=float), expected_indices
    ):
        raise DataQualityError(
            f"{dataset_name}.{index_column} is not aligned to rolling indices "
            f"{mfdfa_window}..{time_series_rows}"
        )

    return {"rows": len(dataframe), "features": len(feature_columns)}


SPECTRUM_LABEL_COLUMNS = (
    "Spectrum_Alpha_Min",
    "Spectrum_Alpha_Max",
    "Spectrum_Alpha_Width",
    "Spectrum_Alpha_0",
    "Spectrum_Delta_f_alpha",
    "Spectrum_B",
    "Spectrum_F_Min",
    "Spectrum_F_Max",
    "Spectrum_F_Height",
    "Spectrum_A",
    "Spectrum_Beta",
    "Spectrum_Beta2",
    "Spectrum_Eps",
    "Spectrum_Fit_RMSE",
    "Spectrum_Fit_R2",
    "Spectrum_Fit_Converged",
)


def validate_spectrum_label_file(
    file_path: str | Path,
    *,
    time_series_rows: int,
    mfdfa_window: int,
) -> dict[str, int]:
    """Validate the future-window spectrum labels used by the auxiliary head."""
    path = Path(file_path)
    if not path.is_file():
        raise DataQualityError(f"Spectrum label file does not exist: {path}")
    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:
        raise DataQualityError(f"Could not read spectrum label file: {path}") from exc

    dataset_name = str(path)
    _require_non_empty(dataframe, dataset_name)
    _require_columns(dataframe, SPECTRUM_LABEL_COLUMNS, dataset_name)
    numeric_columns = [
        column for column in SPECTRUM_LABEL_COLUMNS if column != "Spectrum_Fit_Converged"
    ]
    for column in numeric_columns:
        _require_finite_numeric(dataframe[column], column, dataset_name)

    expected_rows = time_series_rows - mfdfa_window + 1
    if len(dataframe) != expected_rows:
        raise DataQualityError(
            f"{dataset_name} has {len(dataframe)} rows; expected {expected_rows}"
        )
    index_column = dataframe.columns[0]
    indices = pd.to_numeric(dataframe[index_column], errors="coerce")
    expected_indices = np.arange(mfdfa_window, time_series_rows + 1)
    if indices.isna().any() or not np.array_equal(
        indices.to_numpy(dtype=float), expected_indices
    ):
        raise DataQualityError(
            f"{dataset_name}.{index_column} is not aligned to rolling indices "
            f"{mfdfa_window}..{time_series_rows}"
        )
    converged = dataframe["Spectrum_Fit_Converged"].astype(str).str.lower()
    if not converged.isin(("true", "1", "yes")).all():
        raise DataQualityError(f"{dataset_name} contains non-converged spectrum fits")

    return {"rows": len(dataframe), "labels": len(SPECTRUM_LABEL_COLUMNS) - 1}


def validate_forecast_file(
    file_path: str | Path,
    horizon: int,
    *,
    value_column: str = "bod_mgl",
) -> dict[str, int]:
    """Validate a forecast file written by the R forecast stage."""
    path = Path(file_path)
    if not path.is_file():
        raise DataQualityError(f"Forecast file does not exist: {path}")

    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:
        raise DataQualityError(f"Could not read forecast file: {path}") from exc

    dataset_name = str(path)
    _require_non_empty(dataframe, dataset_name)
    _require_columns(dataframe, ("date", value_column), dataset_name)
    if len(dataframe) != horizon:
        raise DataQualityError(
            f"{dataset_name} has {len(dataframe)} rows; expected forecast horizon {horizon}"
        )

    dates = pd.to_numeric(dataframe["date"], errors="coerce")
    if dates.isna().any() or not np.isfinite(dates.to_numpy(dtype=float)).all():
        raise DataQualityError(f"{dataset_name}.date contains invalid value(s)")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise DataQualityError(f"{dataset_name}.date is not unique and increasing")
    _require_finite_numeric(dataframe[value_column], value_column, dataset_name)

    return {"rows": len(dataframe), "horizon": horizon}


def validate_python_forecast_file(
    file_path: str | Path,
    horizon: int,
) -> dict[str, int]:
    """Validate the CSV published by the Python LSTM forecast stage."""
    path = Path(file_path)
    if not path.is_file():
        raise DataQualityError(f"Python forecast file does not exist: {path}")

    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:
        raise DataQualityError(f"Could not read Python forecast file: {path}") from exc

    dataset_name = str(path)
    _require_non_empty(dataframe, dataset_name)
    _require_columns(dataframe, ("date", "forecast", "actual", "residual"), dataset_name)
    if len(dataframe) != horizon:
        raise DataQualityError(
            f"{dataset_name} has {len(dataframe)} rows; expected forecast horizon {horizon}"
        )

    dates = pd.to_datetime(dataframe["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise DataQualityError(f"{dataset_name}.date is invalid, duplicated, or unsorted")
    for column in ("forecast", "actual", "residual"):
        _require_finite_numeric(dataframe[column], column, dataset_name)

    return {"rows": len(dataframe), "horizon": horizon}


def validate_forecast_metadata(
    file_path: str | Path,
    forecast_csv_name: str,
    horizon: int,
) -> dict[str, int]:
    """Validate the Python forecast metadata contract."""
    path = Path(file_path)
    if not path.is_file():
        raise DataQualityError(f"Forecast metadata does not exist: {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataQualityError(f"Could not read forecast metadata: {path}") from exc

    required = (
        "run_id",
        "input_manifest_checksum",
        "river",
        "indicator",
        "horizon",
        "selected_features",
        "best_params",
        "prediction_csv",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise DataQualityError(f"{path} is missing metadata field(s): {missing}")
    if metadata["horizon"] != horizon:
        raise DataQualityError(
            f"{path}.horizon is {metadata['horizon']}; expected {horizon}"
        )
    if metadata["prediction_csv"] != forecast_csv_name:
        raise DataQualityError(
            f"{path}.prediction_csv does not reference {forecast_csv_name}"
        )
    if not metadata["selected_features"] or not metadata["best_params"]:
        raise DataQualityError(f"{path} contains empty model metadata")
    return {"horizon": horizon}


def _configured_path(config_path: Path, configured_value: str) -> Path:
    value = Path(configured_value)
    if value.is_absolute():
        return value
    return config_path.parent.parent.joinpath(value)


def validate_configured_stage(stage: str, config_path: str | Path) -> dict[str, object]:
    """Run one configured validation stage for Airflow or shell execution."""
    config_file = Path(config_path).resolve()
    with config_file.open(encoding="utf-8") as stream:
        config = json.load(stream)

    fields = config["FIELDS_TO_RETRIEVE"]
    bronze_folder = _configured_path(config_file, config["BRONZE_FOLDER"])
    curated_folder = _configured_path(config_file, config["CURATED_FOLDER"])
    features_folder = _configured_path(config_file, config["FEATURES_FOLDER"])
    if stage == "raw":
        raw_folder = _configured_path(config_file, config["RAW_FOLDER"])
        directory_report = validate_raw_directory(raw_folder, fields)
        manifest_report = validate_raw_ingestion_manifest(
            raw_folder,
            _configured_path(config_file, config["MANIFEST_FOLDER"]),
            config.get("RAW_SOURCE_FILE_PREFIXES", DEFAULT_SOURCE_FILE_PREFIXES),
            input_manifest_path=resolve_input_manifest(config_file),
        )
        return {**directory_report, **manifest_report}
    if stage == "combined":
        path = bronze_folder / "combined_data.csv"
        report = validate_combined_file(path, fields)
        return {**report, **validate_output_lineage(path, config_file, "bronze")}
    if stage == "filtered":
        path = bronze_folder / "combined_filtered_data.csv"
        report = validate_combined_file(
            path,
            fields,
            non_null_columns=(
                "Controle_Date",
                config["COLUMN_FILTER"],
            ),
            at_least_one_non_null_columns=(config["INDICATOR"],),
            filter_column=config["COLUMN_FILTER"],
            filter_value=config["ROW_FILTER"],
        )
        return {**report, **validate_output_lineage(path, config_file, "bronze")}
    if stage == "timeseries":
        indicator = config["INDICATOR"]
        path = curated_folder / (
            f"{config['RIVER_NAME_VALUE_ENG']}_{indicator}_ts.csv"
        )
        report = validate_time_series_file(path, indicator, config["HORIZON"])
        return {
            **report,
            **validate_output_lineage(path, config_file, "curated"),
        }
    if stage == "features":
        indicator = config["INDICATOR"]
        time_series_path = curated_folder / (
            f"{config['RIVER_NAME_VALUE_ENG']}_{indicator}_ts.csv"
        )
        feature_path = features_folder / (
            f"MFDFA_{config['RIVER_NAME_VALUE_ENG']}_{indicator}_"
            f"{config['MFDFA_WINDOW']}_window.csv"
        )
        spectrum_path = features_folder / (
            f"SpectrumLabels_{config['RIVER_NAME_VALUE_ENG']}_{indicator}_"
            f"{config['MFDFA_WINDOW']}_window.csv"
        )
        time_series = pd.read_csv(time_series_path)
        report = validate_feature_file(
            feature_path,
            [
                "H",
                "Delta_H",
                "Delta_alpha",
                "Alpha_0",
                "Delta_f_alpha",
                "B",
                "A",
                "beta",
                "beta2",
                "Mean",
                "Volatility",
                "Skew",
                "Slope",
                "Kurtosis",
                "Entropy",
                "Width",
            ],
            time_series_rows=len(time_series),
            mfdfa_window=config["MFDFA_WINDOW"],
        )
        spectrum_report = validate_spectrum_label_file(
            spectrum_path,
            time_series_rows=len(time_series),
            mfdfa_window=config["MFDFA_WINDOW"],
        )
        return {
            **report,
            **validate_output_lineage(feature_path, config_file, "features_mfdfa"),
            **spectrum_report,
            **validate_output_lineage(spectrum_path, config_file, "features_spectrum"),
        }
    if stage == "forecast":
        indicator = config["INDICATOR"]
        forecast_path = _configured_path(config_file, config["FORECASTING_FOLDER"]) / (
            f"Forecasted_{config['RIVER_NAME_VALUE_ENG']}_{indicator}_"
            f"{config['HORIZON']}_days.csv"
        )
        report = validate_forecast_file(forecast_path, config["HORIZON"])
        return {
            **report,
            **validate_output_lineage(forecast_path, config_file, "forecast_r"),
        }
    if stage == "forecast-python":
        indicator = config["INDICATOR"]
        river_name = config["RIVER_NAME_VALUE_ENG"]
        forecast_folder = _configured_path(config_file, config["FORECASTING_FOLDER"])
        stem = f"Forecasted_{river_name}_{indicator}_{config['HORIZON']}_days_LSTM"
        forecast_path = forecast_folder / f"{stem}.csv"
        metadata_path = forecast_folder / f"{stem}.json"
        report = validate_python_forecast_file(forecast_path, config["HORIZON"])
        metadata_report = validate_forecast_metadata(
            metadata_path,
            forecast_path.name,
            config["HORIZON"],
        )
        return {
            **report,
            **metadata_report,
            **validate_output_lineage(forecast_path, config_file, "forecast_python"),
            **validate_output_lineage(
                metadata_path,
                config_file,
                "forecast_python_metadata",
            ),
        }

    raise ValueError(f"Unknown validation stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "raw",
            "combined",
            "filtered",
            "timeseries",
            "features",
            "forecast",
            "forecast-python",
        ),
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    report = validate_configured_stage(args.stage, args.config)
    print(f"Data quality passed for {args.stage}: {report}")


if __name__ == "__main__":
    main()
