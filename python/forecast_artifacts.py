"""Publish durable artifacts for the Python forecasting stage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from lineage import publish_atomic_output, temporary_output_path, load_lineage_context
except ImportError:  # pragma: no cover - supports package imports in tests
    from .lineage import (
        load_lineage_context,
        publish_atomic_output,
        temporary_output_path,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def _write_json_atomically(
    payload: dict[str, Any],
    output_path: Path,
    config_path: str | Path,
    input_files: Iterable[str | Path],
    stage: str,
) -> Path:
    temporary_path = temporary_output_path(output_path)
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
            + "\n",
            encoding="utf-8",
        )
        return publish_atomic_output(
            temporary_path,
            output_path,
            config_path,
            stage,
            input_files=input_files,
            extra={"artifact_type": "forecast_metadata"},
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def publish_forecast_artifacts(
    *,
    ts_data: pd.DataFrame,
    preds: np.ndarray,
    actuals: np.ndarray,
    horizon: int,
    indicator: str,
    river_name: str,
    best_params: dict[str, Any],
    best_cv_rmse: float,
    test_rmse: float,
    selected_features: list[str],
    stability: pd.DataFrame,
    forecasting_folder: str | Path,
    config_path: str | Path,
    input_files: Iterable[str | Path],
    training_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write CSV, metadata JSON, and PNG artifacts for the holdout forecast."""
    if len(preds) != horizon or len(actuals) != horizon:
        raise ValueError(
            f"Forecast output has {len(preds)} predictions and {len(actuals)} "
            f"actuals; expected horizon {horizon}"
        )
    if "date" not in ts_data.columns or indicator not in ts_data.columns:
        raise ValueError("Time series must contain date and indicator columns")

    output_folder = Path(forecasting_folder).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    config_file = Path(config_path).resolve()
    input_paths = list(input_files)
    lineage = load_lineage_context(config_file)

    dates = pd.to_datetime(ts_data["date"], errors="coerce")
    if dates.isna().any() or len(dates) < horizon:
        raise ValueError("Time-series dates are invalid or shorter than the horizon")
    forecast_dates = dates.iloc[-horizon:].dt.strftime("%Y-%m-%d").tolist()

    stem = f"Forecasted_{river_name}_{indicator}_{horizon}_days_LSTM"
    csv_path = output_folder / f"{stem}.csv"
    metadata_path = output_folder / f"{stem}.json"
    plot_path = output_folder / f"{stem}.png"

    forecast_frame = pd.DataFrame(
        {
            "date": forecast_dates,
            "forecast": np.asarray(preds, dtype=float),
            "actual": np.asarray(actuals, dtype=float),
        }
    )
    forecast_frame["residual"] = forecast_frame["actual"] - forecast_frame["forecast"]

    temporary_csv = temporary_output_path(csv_path)
    try:
        forecast_frame.to_csv(temporary_csv, index=False, encoding="utf-8")
        published_csv = publish_atomic_output(
            temporary_csv,
            csv_path,
            config_file,
            "forecast_python",
            input_files=input_paths,
            extra={
                "artifact_type": "holdout_forecast_csv",
                "forecast_kind": "holdout_evaluation",
                "horizon": horizon,
            },
        )
    finally:
        temporary_csv.unlink(missing_ok=True)

    metadata = {
        "schema_version": 1,
        "artifact_type": "holdout_forecast_metadata",
        "forecast_kind": "holdout_evaluation",
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "run_id": lineage["run_id"],
        "input_manifest_path": lineage["input_manifest_path"],
        "input_manifest_checksum": lineage["input_manifest_checksum"],
        "river": river_name,
        "indicator": indicator,
        "horizon": horizon,
        "forecast_dates": forecast_dates,
        "model": "LSTM + Optuna RMSE baseline",
        "best_params": best_params,
        "selected_features": selected_features,
        "best_cv_rmse": float(best_cv_rmse),
        "test_rmse": float(test_rmse),
        "feature_stability": stability.to_dict(orient="records"),
        "prediction_csv": csv_path.name,
        "training_metadata": training_metadata or {},
    }
    metadata_lineage = _write_json_atomically(
        metadata,
        metadata_path,
        config_file,
        input_paths,
        "forecast_python_metadata",
    )

    temporary_plot = temporary_output_path(plot_path)
    try:
        figure, axis = plt.subplots(figsize=(12, 6))
        axis.plot(
            dates,
            ts_data[indicator].to_numpy(),
            color="black",
            alpha=0.6,
            label="Actual data",
        )
        axis.plot(
            dates.iloc[-horizon:],
            preds,
            color="orange",
            linewidth=2.5,
            label=f"Final {horizon}-point holdout forecast",
        )
        axis.set_title(f"LSTM + Optuna Results: Units={best_params['units']}")
        axis.legend()
        axis.grid(True, alpha=0.3)
        figure.savefig(temporary_plot, format="png", dpi=150, bbox_inches="tight")
        plt.close(figure)
        published_plot = publish_atomic_output(
            temporary_plot,
            plot_path,
            config_file,
            "forecast_python",
            input_files=input_paths,
            extra={
                "artifact_type": "forecast_plot",
                "forecast_kind": "holdout_evaluation",
                "horizon": horizon,
            },
        )
    finally:
        plt.close("all")
        temporary_plot.unlink(missing_ok=True)

    return {
        "csv": csv_path,
        "metadata": metadata_path,
        "metadata_lineage": metadata_lineage,
        "plot": plot_path,
        "plot_lineage": published_plot,
        "csv_lineage": published_csv,
    }
