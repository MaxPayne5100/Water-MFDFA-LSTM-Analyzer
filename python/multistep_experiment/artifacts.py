"""Failure-safe isolated artifacts for multi-step experiments."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from lineage import publish_atomic_output, sha256_file, temporary_output_path
    from utils import load_config
except ImportError:  # pragma: no cover
    from ..lineage import publish_atomic_output, sha256_file, temporary_output_path
    from ..utils import load_config

from .constants import canonical_variant


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def experiment_directory(
    config_path: str | Path,
    experiment_id: str,
    variant: str,
    seed: int,
) -> Path:
    variant = canonical_variant(variant)
    config_file = Path(config_path).resolve()
    config = load_config(str(config_file))
    configured = Path(config["FORECASTING_FOLDER"])
    forecast_root = (
        configured
        if configured.is_absolute()
        else config_file.parent.parent / configured
    )
    return (
        forecast_root
        / "experiments"
        / experiment_id
        / f"horizon_{int(config['HORIZON'])}"
        / f"model_{variant}"
        / f"seed_{seed}"
    )


def horizon_experiment_directory(
    config_path: str | Path, experiment_id: str
) -> Path:
    config_file = Path(config_path).resolve()
    config = load_config(str(config_file))
    configured = Path(config["FORECASTING_FOLDER"])
    forecast_root = (
        configured
        if configured.is_absolute()
        else config_file.parent.parent / configured
    )
    return (
        forecast_root
        / "experiments"
        / experiment_id
        / f"horizon_{int(config['HORIZON'])}"
    )


def locked_parameters_path(
    config_path: str | Path, experiment_id: str
) -> Path:
    return horizon_experiment_directory(config_path, experiment_id) / "locked_parameters.json"


def publish_locked_parameters(
    *,
    config_path: str | Path,
    experiment_id: str,
    payload: dict[str, Any],
    input_files: list[str | Path],
) -> Path:
    config_file = Path(config_path).resolve()
    destination = locked_parameters_path(config_file, experiment_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _publish_json(
        payload,
        destination,
        config_file,
        [Path(path).resolve() for path in input_files],
    )
    return destination


def publish_tuning_trials(
    *,
    config_path: str | Path,
    experiment_id: str,
    trials: pd.DataFrame,
    input_files: list[str | Path],
) -> Path:
    config_file = Path(config_path).resolve()
    destination = horizon_experiment_directory(
        config_file, experiment_id
    ) / "optuna_trials.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _publish_csv(
        trials,
        destination,
        config_file,
        [Path(path).resolve() for path in input_files],
        "optuna_trials_csv",
    )
    return destination


def load_locked_parameters(
    config_path: str | Path, experiment_id: str
) -> dict[str, Any]:
    path = locked_parameters_path(config_path, experiment_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"Locked parameters do not exist for this horizon: {path}. "
            "Run the baseline-only tuning command first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def publish_keras_network(
    network: Any,
    destination: Path,
    config_path: Path,
    input_files: list[Path],
) -> Path:
    """Save a Keras network atomically; custom training logic stays in metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.keras"
    )
    try:
        network.save(temporary)
        return publish_atomic_output(
            temporary,
            destination,
            config_path,
            "forecast_python_multistep_experiment",
            input_files=input_files,
            extra={"artifact_type": "keras_inference_network"},
        )
    finally:
        temporary.unlink(missing_ok=True)


def _publish_json(
    payload: dict[str, Any],
    path: Path,
    config_path: Path,
    input_files: list[Path],
) -> Path:
    temporary = temporary_output_path(path)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
            + "\n",
            encoding="utf-8",
        )
        return publish_atomic_output(
            temporary,
            path,
            config_path,
            "forecast_python_multistep_experiment",
            input_files=input_files,
            extra={"artifact_type": "experiment_metadata"},
        )
    finally:
        temporary.unlink(missing_ok=True)


def _publish_csv(
    dataframe: pd.DataFrame,
    path: Path,
    config_path: Path,
    input_files: list[Path],
    artifact_type: str,
) -> Path:
    temporary = temporary_output_path(path)
    try:
        dataframe.to_csv(temporary, index=False, encoding="utf-8")
        return publish_atomic_output(
            temporary,
            path,
            config_path,
            "forecast_python_multistep_experiment",
            input_files=input_files,
            extra={"artifact_type": artifact_type},
        )
    finally:
        temporary.unlink(missing_ok=True)


def publish_experiment_artifacts(
    *,
    config_path: str | Path,
    experiment_id: str,
    variant: str,
    seed: int,
    forecast_origins: np.ndarray,
    forecast_origin_dates: pd.DatetimeIndex,
    target_dates: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    fold_metrics: pd.DataFrame,
    training_history: pd.DataFrame,
    metadata: dict[str, Any],
    input_files: list[str | Path],
    network: Any | None = None,
) -> dict[str, Path]:
    variant = canonical_variant(variant)
    config_file = Path(config_path).resolve()
    inputs = [Path(path).resolve() for path in input_files]
    output = experiment_directory(config_file, experiment_id, variant, seed)
    output.mkdir(parents=True, exist_ok=True)
    actual_values = np.asarray(actual)
    predicted_values = np.asarray(predicted)
    if actual_values.ndim == 3 and actual_values.shape[-1] == 1:
        actual_values = actual_values[..., 0]
    if predicted_values.ndim == 3 and predicted_values.shape[-1] == 1:
        predicted_values = predicted_values[..., 0]
    if actual_values.ndim == 1:
        actual_values = actual_values[np.newaxis, :]
        predicted_values = predicted_values[np.newaxis, :]
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted rolling forecasts must have equal shapes")
    origin_count, horizon = actual_values.shape
    origins = np.asarray(forecast_origins, dtype=int).reshape(-1)
    origin_dates = pd.DatetimeIndex(forecast_origin_dates)
    date_values = np.asarray(target_dates)
    if (
        len(origins) != origin_count
        or len(origin_dates) != origin_count
        or date_values.shape != (origin_count, horizon)
    ):
        raise ValueError("Rolling-origin dates and forecast arrays are misaligned")
    forecast = pd.DataFrame(
        {
            "forecast_origin": np.repeat(origins, horizon),
            "forecast_origin_date": np.repeat(
                origin_dates.strftime("%Y-%m-%d"), horizon
            ),
            "date": pd.DatetimeIndex(date_values.reshape(-1)).strftime("%Y-%m-%d"),
            "horizon_step": np.tile(np.arange(1, horizon + 1), origin_count),
            "actual": actual_values.reshape(-1),
            "forecast": predicted_values.reshape(-1),
        }
    )
    forecast["residual"] = forecast["actual"] - forecast["forecast"]
    metadata = {
        **metadata,
        "experiment_id": experiment_id,
        "variant": variant,
        "seed": seed,
        "config_checksum": sha256_file(config_file),
    }
    paths = {
        "forecast": output / "forecast.csv",
        "fold_metrics": output / "fold_metrics.csv",
        "training_history": output / "training_history.csv",
        "metadata": output / "metadata.json",
    }
    if network is not None:
        paths["model"] = output / "model.keras"
    sidecars = {
        "forecast_lineage": _publish_csv(
            forecast, paths["forecast"], config_file, inputs, "forecast_csv"
        ),
        "fold_metrics_lineage": _publish_csv(
            fold_metrics,
            paths["fold_metrics"],
            config_file,
            inputs,
            "fold_metrics_csv",
        ),
        "training_history_lineage": _publish_csv(
            training_history,
            paths["training_history"],
            config_file,
            inputs,
            "training_history_csv",
        ),
        "metadata_lineage": _publish_json(
            metadata, paths["metadata"], config_file, inputs
        ),
    }
    if network is not None:
        sidecars["model_lineage"] = publish_keras_network(
            network, paths["model"], config_file, inputs
        )
    return {**paths, **sidecars}


def publish_comparison_artifacts(
    *,
    config_path: str | Path,
    experiment_id: str,
    runs: pd.DataFrame,
    origins: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    input_files: list[str | Path],
) -> dict[str, Path]:
    """Publish same-horizon comparison tables with lineage."""
    config_file = Path(config_path).resolve()
    inputs = [Path(path).resolve() for path in input_files]
    output = horizon_experiment_directory(config_file, experiment_id)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "runs": output / "comparison_runs.csv",
        "origins": output / "comparison_origins.csv",
        "summary": output / "comparison_summary.csv",
        "paired": output / "comparison_paired.csv",
    }
    sidecars = {
        f"{name}_lineage": _publish_csv(
            frame,
            paths[name],
            config_file,
            inputs,
            f"{name}_comparison_csv",
        )
        for name, frame in (
            ("runs", runs),
            ("origins", origins),
            ("summary", summary),
            ("paired", paired),
        )
    }
    return {**paths, **sidecars}
