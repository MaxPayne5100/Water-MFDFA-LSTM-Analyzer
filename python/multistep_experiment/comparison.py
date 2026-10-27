"""Descriptive, paired rolling-origin comparison within one horizon."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from utils import load_config
except ImportError:  # pragma: no cover
    from ..utils import load_config

from .artifacts import horizon_experiment_directory, publish_comparison_artifacts
from .constants import (
    MODEL_VARIANTS,
    SEEDS,
    VALUE_ONLY,
    canonical_variant,
)


REQUIRED_TIMING_FIELDS = (
    "cross_validation_training_seconds",
    "final_training_seconds",
    "total_training_seconds",
    "total_trained_epochs",
)


def _within_horizon_direction_accuracy(forecast: pd.DataFrame) -> float:
    """Fraction of correctly predicted step-to-step directions within horizons."""
    matches: list[np.ndarray] = []
    for _, group in forecast.groupby("forecast_origin", sort=True):
        ordered = group.sort_values("horizon_step")
        actual_change = np.diff(ordered["actual"].to_numpy(dtype=float))
        forecast_change = np.diff(ordered["forecast"].to_numpy(dtype=float))
        if len(actual_change):
            matches.append(np.sign(actual_change) == np.sign(forecast_change))
    if not matches:
        return float("nan")
    return float(np.concatenate(matches).mean())


def _optional_mean(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return float("nan")
    values = group[column].to_numpy(dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _validated_forecast(
    forecast_path: Path,
    *,
    horizon: int,
    expected_origin_count: int,
) -> pd.DataFrame:
    forecast = pd.read_csv(forecast_path)
    required = {
        "forecast_origin",
        "forecast_origin_date",
        "date",
        "horizon_step",
        "actual",
        "forecast",
    }
    if not required.issubset(forecast.columns):
        raise ValueError(
            f"Forecast artifact is not a rolling-origin result: {forecast_path}. "
            "Rerun it with the current experiment implementation."
        )
    numeric = ["forecast_origin", "horizon_step", "actual", "forecast"]
    if forecast[numeric].isna().any().any():
        raise ValueError(f"Forecast artifact contains missing values: {forecast_path}")
    if forecast.duplicated(["forecast_origin", "horizon_step"]).any():
        raise ValueError(f"Forecast artifact contains duplicate points: {forecast_path}")

    origins = forecast["forecast_origin"].drop_duplicates().to_numpy(dtype=int)
    if len(origins) != expected_origin_count:
        raise ValueError(
            f"Expected {expected_origin_count} rolling origins but found {len(origins)} "
            f"in {forecast_path}"
        )
    expected_steps = np.arange(1, horizon + 1)
    for origin, group in forecast.groupby("forecast_origin", sort=True):
        steps = np.sort(group["horizon_step"].to_numpy(dtype=int))
        if not np.array_equal(steps, expected_steps):
            raise ValueError(
                f"Origin {origin} does not contain horizon steps 1..{horizon}: "
                f"{forecast_path}"
            )
    return forecast.sort_values(["forecast_origin", "horizon_step"]).reset_index(
        drop=True
    )


def collect_completed_runs(
    config_path: str | Path, experiment_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    config = load_config(str(Path(config_path).resolve()))
    horizon = int(config["HORIZON"])
    expected_origin_count = int(config.get("EVALUATION_ORIGIN_COUNT", 1))
    expected_seeds = {int(seed) for seed in SEEDS}
    root = horizon_experiment_directory(config_path, experiment_id)
    run_records: list[dict[str, float | int | str]] = []
    origin_records: list[dict[str, float | int | str]] = []
    inputs: list[Path] = []
    reference_targets: pd.DataFrame | None = None

    for variant in MODEL_VARIANTS:
        variant_root = root / f"model_{variant}"
        available_seeds: set[int] = set()
        forecast_paths: dict[int, Path] = {}
        for seed in sorted(expected_seeds):
            forecast_path = variant_root / f"seed_{seed}" / "forecast.csv"
            if forecast_path.is_file():
                available_seeds.add(seed)
                forecast_paths[seed] = forecast_path
        all_seed_folders = list(variant_root.glob("seed_*"))
        unexpected = sorted(
            int(folder.name.removeprefix("seed_"))
            for folder in all_seed_folders
            if folder.name.removeprefix("seed_").isdigit()
            and (folder / "forecast.csv").is_file()
            and int(folder.name.removeprefix("seed_")) not in expected_seeds
        )
        if available_seeds != expected_seeds or unexpected:
            missing = sorted(expected_seeds - available_seeds)
            raise ValueError(
                f"Incomplete {variant} seed panel; missing={missing}, "
                f"unexpected={unexpected}"
            )

        for seed in sorted(expected_seeds):
            forecast_path = forecast_paths[seed]
            metadata_path = forecast_path.with_name("metadata.json")
            if not metadata_path.is_file():
                raise ValueError(f"Run metadata is missing: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            timing = metadata.get("timing")
            if not isinstance(timing, dict):
                raise ValueError(
                    f"Elapsed learning-time metadata is missing: {metadata_path}"
                )
            missing_timing = [
                field for field in REQUIRED_TIMING_FIELDS if field not in timing
            ]
            if missing_timing:
                raise ValueError(
                    f"Timing fields {missing_timing} are missing: {metadata_path}"
                )
            cv_seconds = float(timing["cross_validation_training_seconds"])
            final_seconds = float(timing["final_training_seconds"])
            total_seconds = float(timing["total_training_seconds"])
            total_epochs = int(timing["total_trained_epochs"])
            if (
                not np.isfinite([cv_seconds, final_seconds, total_seconds]).all()
                or min(cv_seconds, final_seconds, total_seconds) < 0
                or total_epochs < 1
                or not np.isclose(total_seconds, cv_seconds + final_seconds)
            ):
                raise ValueError(f"Invalid elapsed learning-time metadata: {metadata_path}")
            forecast = _validated_forecast(
                forecast_path,
                horizon=horizon,
                expected_origin_count=expected_origin_count,
            )
            target_columns = [
                "forecast_origin",
                "forecast_origin_date",
                "date",
                "horizon_step",
            ]
            targets = forecast[[*target_columns, "actual"]].copy()
            if reference_targets is None:
                reference_targets = targets
            elif not (
                targets[target_columns].equals(reference_targets[target_columns])
                and np.allclose(
                    targets["actual"].to_numpy(dtype=float),
                    reference_targets["actual"].to_numpy(dtype=float),
                )
            ):
                raise ValueError(
                    f"Forecast targets differ across variants or seeds: {forecast_path}"
                )

            error = forecast["forecast"].to_numpy(dtype=float) - forecast[
                "actual"
            ].to_numpy(dtype=float)
            absolute_error = np.abs(error)
            run_records.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "horizon": horizon,
                    "origins": expected_origin_count,
                    "forecast_points": len(forecast),
                    "value_rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "value_mae": float(np.mean(absolute_error)),
                    "forecast_bias": float(np.mean(error)),
                    "median_absolute_error": float(np.median(absolute_error)),
                    "p90_absolute_error": float(np.quantile(absolute_error, 0.90)),
                    "within_horizon_direction_accuracy": (
                        _within_horizon_direction_accuracy(forecast)
                    ),
                    "cross_validation_training_seconds": cv_seconds,
                    "final_training_seconds": final_seconds,
                    "total_training_seconds": total_seconds,
                    "total_trained_epochs": total_epochs,
                }
            )
            for origin, group in forecast.groupby("forecast_origin", sort=True):
                origin_error = group["forecast"].to_numpy(dtype=float) - group[
                    "actual"
                ].to_numpy(dtype=float)
                origin_absolute_error = np.abs(origin_error)
                origin_records.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "forecast_origin": int(origin),
                        "forecast_origin_date": str(
                            group["forecast_origin_date"].iloc[0]
                        ),
                        "value_rmse": float(
                            np.sqrt(np.mean(np.square(origin_error)))
                        ),
                        "value_mae": float(np.mean(np.abs(origin_error))),
                        "forecast_bias": float(np.mean(origin_error)),
                        "p90_absolute_error": float(
                            np.quantile(origin_absolute_error, 0.90)
                        ),
                        "within_horizon_direction_accuracy": (
                            _within_horizon_direction_accuracy(group)
                        ),
                    }
                )
            inputs.append(forecast_path)
            inputs.append(metadata_path)

    runs = pd.DataFrame(run_records).sort_values(["variant", "seed"]).reset_index(
        drop=True
    )
    origins = pd.DataFrame(origin_records).sort_values(
        ["variant", "seed", "forecast_origin"]
    ).reset_index(drop=True)
    return runs, origins, inputs


def summarize_runs(
    runs: pd.DataFrame,
    origins: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = runs.copy()
    runs["variant"] = runs["variant"].map(canonical_variant)
    if origins is not None:
        origins = origins.copy()
        origins["variant"] = origins["variant"].map(canonical_variant)
    summary_rows = []
    for variant, group in runs.groupby("variant", sort=False):
        values = group["value_rmse"].to_numpy(dtype=float)
        standard_error = (
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else float("nan")
        )
        critical = (
            float(stats.t.ppf(0.975, len(values) - 1))
            if len(values) > 1
            else float("nan")
        )
        summary_rows.append(
            {
                "variant": variant,
                "seed_runs": len(values),
                "rolling_origins_per_seed": (
                    int(group["origins"].iloc[0]) if "origins" in group else 1
                ),
                "mean_value_rmse": float(np.mean(values)),
                "std_value_rmse_across_seeds": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
                ),
                "median_value_rmse": float(np.median(values)),
                "mean_value_mae": _optional_mean(group, "value_mae"),
                "mean_forecast_bias": _optional_mean(group, "forecast_bias"),
                "mean_p90_absolute_error": _optional_mean(
                    group, "p90_absolute_error"
                ),
                "mean_within_horizon_direction_accuracy": _optional_mean(
                    group, "within_horizon_direction_accuracy"
                ),
                "mean_total_training_seconds": _optional_mean(
                    group, "total_training_seconds"
                ),
                "median_total_training_seconds": (
                    float(group["total_training_seconds"].median())
                    if "total_training_seconds" in group
                    and group["total_training_seconds"].notna().any()
                    else np.nan
                ),
                "mean_total_trained_epochs": _optional_mean(
                    group, "total_trained_epochs"
                ),
                "descriptive_seed_ci95_low": float(
                    np.mean(values) - critical * standard_error
                ),
                "descriptive_seed_ci95_high": float(
                    np.mean(values) + critical * standard_error
                ),
                "inference_note": (
                    "Seed interval measures optimization variability on shared, "
                    "overlapping rolling origins; it is not a generalization interval."
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    baseline_summary = summary.loc[summary["variant"] == VALUE_ONLY]
    if len(baseline_summary) == 1:
        baseline_rmse = float(baseline_summary["mean_value_rmse"].iloc[0])
        baseline_seconds = float(
            baseline_summary["mean_total_training_seconds"].iloc[0]
        )
        summary["mean_rmse_change_vs_baseline_pct"] = (
            100.0 * (summary["mean_value_rmse"] - baseline_rmse) / baseline_rmse
        )
        summary["mean_training_time_ratio_vs_baseline"] = (
            summary["mean_total_training_seconds"] / baseline_seconds
            if np.isfinite(baseline_seconds) and baseline_seconds > 0
            else np.nan
        )

    comparison_source = (
        origins if origins is not None else runs.assign(forecast_origin=0)
    )
    baseline = comparison_source.loc[
        comparison_source["variant"].map(canonical_variant) == VALUE_ONLY,
        ["seed", "forecast_origin", "value_rmse"],
    ].rename(columns={"value_rmse": "baseline_rmse"})
    paired_rows = []
    for variant in MODEL_VARIANTS[1:]:
        candidate = comparison_source.loc[
            comparison_source["variant"].map(canonical_variant) == variant,
            ["seed", "forecast_origin", "value_rmse"],
        ].rename(columns={"value_rmse": "candidate_rmse"})
        matched = baseline.merge(
            candidate,
            on=["seed", "forecast_origin"],
            how="inner",
            validate="one_to_one",
        )
        differences = matched["candidate_rmse"] - matched["baseline_rmse"]
        relative_differences = 100.0 * differences / matched["baseline_rmse"]
        paired_rows.append(
            {
                "candidate": variant,
                "paired_seeds": int(matched["seed"].nunique()),
                "paired_origins": int(matched["forecast_origin"].nunique()),
                "paired_seed_origins": len(matched),
                "mean_origin_rmse_delta_candidate_minus_baseline": (
                    float(differences.mean()) if len(matched) else np.nan
                ),
                "median_origin_rmse_delta_candidate_minus_baseline": (
                    float(differences.median()) if len(matched) else np.nan
                ),
                "wins_lower_origin_rmse": int((differences < 0).sum()),
                "win_rate_lower_origin_rmse": (
                    float((differences < 0).mean()) if len(matched) else np.nan
                ),
                "mean_origin_rmse_change_pct": (
                    float(relative_differences.mean()) if len(matched) else np.nan
                ),
                "inference_note": (
                    "Descriptive matched seed-origin effect; no independence-based "
                    "p-value because rolling forecast windows overlap."
                ),
            }
        )
    return summary, pd.DataFrame(paired_rows)


def compare_and_publish(
    *, config_path: str | Path, experiment_id: str
) -> dict[str, Path]:
    runs, origins, inputs = collect_completed_runs(config_path, experiment_id)
    summary, paired = summarize_runs(runs, origins)
    return publish_comparison_artifacts(
        config_path=config_path,
        experiment_id=experiment_id,
        runs=runs,
        origins=origins,
        summary=summary,
        paired=paired,
        input_files=inputs,
    )
