"""Explicit training entry points for the isolated multi-step experiment.

Importing this module never starts tuning or fitting.  The expensive functions
are reachable only through explicit CLI subcommands in ``__main__.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from lineage import sha256_file
    from utils import load_config
except ImportError:  # pragma: no cover
    from ..lineage import sha256_file
    from ..utils import load_config

from .artifacts import (
    load_locked_parameters,
    publish_experiment_artifacts,
    publish_locked_parameters,
    publish_tuning_trials,
)
from .constants import (
    MODEL_VARIANTS,
    SEEDS,
    VALUE_FRACTIONAL,
    VALUE_FRACTIONAL_SPECTRUM,
    VALUE_ONLY,
    canonical_variant,
)
from .dataset import FoldTransformers, MultiStepDataset, load_multistep_dataset
from .feature_selection import feature_positions, rank_features
from .long_memory import FractionalOrderEstimate, estimate_fractional_order
from .losses import fractional_coefficients, fractional_rmse
from .models import ModelParameters, build_model
from .validation import PurgedFold, purged_walk_forward_splits


N_SPLITS = 3
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20
LR_PATIENCE = 8


@dataclass
class PreparedSplit:
    train_x: np.ndarray
    evaluation_x: np.ndarray
    train_targets: dict[str, np.ndarray]
    evaluation_targets: dict[str, np.ndarray]
    transformers: FoldTransformers
    selected_features: list[str]
    selected_positions: np.ndarray
    feature_ranking_method: str
    feature_ranking_train_samples: int
    feature_ranking_validation_samples: int
    order_estimate: FractionalOrderEstimate | None
    coefficients: np.ndarray | None


@dataclass
class TuningContext:
    """Data, purged folds, and fold-local rankings cached for one lookback."""

    dataset: MultiStepDataset
    folds: list[PurgedFold]
    feature_rankings: dict[int, pd.DataFrame]


@dataclass(frozen=True)
class FractionalPreprocessing:
    """Training-only fractional estimate and finite kernel for one split."""

    order_estimate: FractionalOrderEstimate
    coefficients: np.ndarray
    available_context: int
    bootstrap_repeats: int
    bootstrap_seed: int | None


@dataclass
class MatrixPreprocessingCache:
    """In-memory preprocessing cache scoped to one immutable matrix dataset."""

    dataset: MultiStepDataset
    feature_rankings: dict[tuple[int, tuple[int, ...]], pd.DataFrame] = field(
        default_factory=dict
    )
    fractional: dict[
        tuple[tuple[int, ...], int, int, int | None], FractionalPreprocessing
    ] = field(default_factory=dict)

    @staticmethod
    def _position_key(positions: Sequence[int]) -> tuple[int, ...]:
        return tuple(int(position) for position in positions)

    def feature_ranking(
        self,
        training_positions: Sequence[int],
        *,
        seed: int,
    ) -> pd.DataFrame:
        """Reuse a seed-specific training-fold ranking across variants."""
        train = np.asarray(training_positions, dtype=int)
        key = (int(seed), self._position_key(train))
        if key not in self.feature_rankings:
            intervals = self.dataset.sample_index.iloc[train]
            self.feature_rankings[key] = rank_features(
                self.dataset.features[train],
                self.dataset.values[train],
                self.dataset.feature_columns,
                seed=int(seed),
                target_starts=intervals["target_start"].to_numpy(),
                target_ends=intervals["target_end"].to_numpy(),
            )
        return self.feature_rankings[key]

    def fractional_preprocessing(
        self,
        training_positions: Sequence[int],
        evaluation_positions: Sequence[int],
        *,
        bootstrap_repeats: int,
        seed: int,
    ) -> FractionalPreprocessing:
        """Reuse deterministic estimates globally and bootstrap estimates by seed."""
        train = np.asarray(training_positions, dtype=int)
        evaluation = np.asarray(evaluation_positions, dtype=int)
        available_context = _available_fractional_context(
            self.dataset,
            train,
            evaluation,
        )
        bootstrap_seed = int(seed) if bootstrap_repeats > 0 else None
        key = (
            self._position_key(train),
            available_context,
            int(bootstrap_repeats),
            bootstrap_seed,
        )
        if key not in self.fractional:
            self.fractional[key] = _build_fractional_preprocessing(
                self.dataset,
                train,
                evaluation,
                bootstrap_repeats=int(bootstrap_repeats),
                seed=int(seed),
            )
        return self.fractional[key]


@dataclass(frozen=True)
class VariantRunContext:
    """Shared dataset and cached preprocessing for one matrix cell."""

    lock: dict[str, Any]
    dataset: MultiStepDataset
    folds: tuple[PurgedFold, ...]
    feature_rankings: dict[int, pd.DataFrame]
    final_feature_ranking: pd.DataFrame
    fractional_preprocessing: dict[int, FractionalPreprocessing] | None
    final_fractional_preprocessing: FractionalPreprocessing | None


def report_fold_pruning(
    trial: Any,
    *,
    fold_position: int,
    completed_metric_rows: Sequence[dict[str, Any]],
) -> float:
    """Report one comparable Optuna resource unit per completed purged fold."""
    if fold_position < 1 or not completed_metric_rows:
        raise ValueError("A positive fold position and completed metrics are required")
    running_rmse = float(
        np.mean([float(row["value_rmse"]) for row in completed_metric_rows])
    )
    if not np.isfinite(running_rmse):
        import optuna

        raise optuna.TrialPruned(
            f"Non-finite running value RMSE after fold {fold_position}"
        )
    trial.report(running_rmse, step=fold_position)
    if trial.should_prune():
        import optuna

        raise optuna.TrialPruned(
            f"Trial pruned after fold {fold_position}: "
            f"running mean value RMSE={running_rmse}"
        )
    return running_rmse


def configured_input_paths(config_path: str | Path) -> list[Path]:
    """Return all immutable inputs used to construct comparable samples."""
    config_file = Path(config_path).resolve()
    config = load_config(str(config_file))
    root = config_file.parent.parent

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    river = config["RIVER_NAME_VALUE_ENG"]
    indicator = config["INDICATOR"]
    window = int(config["MFDFA_WINDOW"])
    return [
        config_file,
        resolve(config["CURATED_FOLDER"]) / f"{river}_{indicator}_ts.csv",
        resolve(config["FEATURES_FOLDER"])
        / f"MFDFA_{river}_{indicator}_{window}_window.csv",
        resolve(config["FEATURES_FOLDER"])
        / f"SpectrumLabels_{river}_{indicator}_{window}_window.csv",
    ]


def set_reproducible_seed(seed: int) -> None:
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def global_rmse_numpy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """One RMSE over every sample and horizon position, with equal point weight."""
    return float(np.sqrt(np.mean(np.square(np.asarray(actual) - np.asarray(predicted)))))


def configured_min_train_origins(config_path: str | Path) -> int:
    config = load_config(str(Path(config_path).resolve()))
    minimum = int(config.get("MIN_TRAIN_ORIGINS", 1))
    if minimum < 1:
        raise ValueError("MIN_TRAIN_ORIGINS must be positive")
    return minimum


def _training_raw_values(
    dataset: MultiStepDataset, training_positions: np.ndarray
) -> np.ndarray:
    final_training_target = int(
        dataset.sample_index.iloc[training_positions]["target_end"].max()
    )
    return dataset.raw_values[: final_training_target + 1]


def _available_fractional_context(
    dataset: MultiStepDataset,
    training_positions: np.ndarray,
    evaluation_positions: np.ndarray,
) -> int:
    combined = np.concatenate([training_positions, evaluation_positions])
    if len(combined) == 0:
        raise ValueError("Fractional preprocessing requires at least one sample")
    return int(dataset.sample_index.iloc[combined]["target_start"].min())


def _build_fractional_preprocessing(
    dataset: MultiStepDataset,
    training_positions: Sequence[int],
    evaluation_positions: Sequence[int],
    *,
    bootstrap_repeats: int,
    seed: int,
) -> FractionalPreprocessing:
    """Estimate ``d`` and construct its kernel from training information only."""
    train = np.asarray(training_positions, dtype=int)
    evaluation = np.asarray(evaluation_positions, dtype=int)
    available_context = _available_fractional_context(
        dataset,
        train,
        evaluation,
    )
    estimate = estimate_fractional_order(
        _training_raw_values(dataset, train),
        bootstrap_repeats=int(bootstrap_repeats),
        seed=int(seed),
    )
    coefficients = fractional_coefficients(
        estimate.selected,
        available_context=available_context,
    )
    coefficients.setflags(write=False)
    return FractionalPreprocessing(
        order_estimate=estimate,
        coefficients=coefficients,
        available_context=available_context,
        bootstrap_repeats=int(bootstrap_repeats),
        bootstrap_seed=int(seed) if bootstrap_repeats > 0 else None,
    )


def prepare_split(
    dataset: MultiStepDataset,
    training_positions: Sequence[int],
    evaluation_positions: Sequence[int],
    *,
    parameters: ModelParameters,
    variant: str,
    seed: int,
    d_bootstrap_repeats: int = 0,
    feature_ranking: pd.DataFrame | None = None,
    fractional_preprocessing: FractionalPreprocessing | None = None,
) -> PreparedSplit:
    """Fit every preprocessing decision using the supplied training fold only."""
    variant = canonical_variant(variant)
    train = np.asarray(training_positions, dtype=int)
    evaluation = np.asarray(evaluation_positions, dtype=int)
    ranking = feature_ranking
    if ranking is None:
        ranking_intervals = dataset.sample_index.iloc[train]
        ranking = rank_features(
            dataset.features[train],
            dataset.values[train],
            dataset.feature_columns,
            seed=seed,
            target_starts=ranking_intervals["target_start"].to_numpy(),
            target_ends=ranking_intervals["target_end"].to_numpy(),
        )
    positions, selected = feature_positions(
        ranking, dataset.feature_columns, parameters.top_k
    )

    include_spectrum = variant == VALUE_FRACTIONAL_SPECTRUM
    transformers = FoldTransformers.fit(
        dataset.features[train][:, :, positions],
        dataset.values[train],
        dataset.spectra[train] if include_spectrum else None,
        dataset.spectrum_mask[train] if include_spectrum else None,
    )
    train_x = transformers.transform_features(
        dataset.features[train][:, :, positions]
    ).astype(np.float32)
    evaluation_x = transformers.transform_features(
        dataset.features[evaluation][:, :, positions]
    ).astype(np.float32)
    train_value = transformers.transform_values(dataset.values[train]).astype(np.float32)
    evaluation_value = transformers.transform_values(
        dataset.values[evaluation]
    ).astype(np.float32)
    train_targets: dict[str, np.ndarray] = {"value": train_value}
    evaluation_targets: dict[str, np.ndarray] = {"value": evaluation_value}

    estimate = None
    coefficients = None
    if variant != VALUE_ONLY:
        fractional = fractional_preprocessing
        if fractional is None:
            fractional = _build_fractional_preprocessing(
                dataset,
                train,
                evaluation,
                bootstrap_repeats=d_bootstrap_repeats,
                seed=seed,
            )
        expected_context = _available_fractional_context(
            dataset,
            train,
            evaluation,
        )
        expected_bootstrap_seed = int(seed) if d_bootstrap_repeats > 0 else None
        if (
            fractional.available_context != expected_context
            or fractional.bootstrap_repeats != int(d_bootstrap_repeats)
            or fractional.bootstrap_seed != expected_bootstrap_seed
        ):
            raise ValueError(
                "Cached fractional preprocessing does not match this split, "
                "bootstrap configuration, or seed"
            )
        estimate = fractional.order_estimate
        coefficients = fractional.coefficients
        context_length = len(coefficients) - 1
        train_context = dataset.fractional_context(train, context_length)
        evaluation_context = dataset.fractional_context(
            evaluation, context_length
        )
        train_targets["context"] = transformers.transform_values(
            train_context
        ).astype(np.float32)
        evaluation_targets["context"] = transformers.transform_values(
            evaluation_context
        ).astype(np.float32)

    if include_spectrum:
        train_targets["spectrum"] = transformers.transform_spectra(
            dataset.spectra[train], dataset.spectrum_mask[train]
        ).astype(np.float32)
        evaluation_targets["spectrum"] = transformers.transform_spectra(
            dataset.spectra[evaluation], dataset.spectrum_mask[evaluation]
        ).astype(np.float32)
        train_targets["spectrum_mask"] = dataset.spectrum_mask[train].astype(
            np.float32
        )
        evaluation_targets["spectrum_mask"] = dataset.spectrum_mask[
            evaluation
        ].astype(np.float32)

    return PreparedSplit(
        train_x=train_x,
        evaluation_x=evaluation_x,
        train_targets=train_targets,
        evaluation_targets=evaluation_targets,
        transformers=transformers,
        selected_features=selected,
        selected_positions=positions,
        feature_ranking_method=str(
            ranking.get("ranking_method", pd.Series(["unknown"])).iloc[0]
        ),
        feature_ranking_train_samples=int(
            ranking.get("ranking_train_samples", pd.Series([len(train)])).iloc[0]
        ),
        feature_ranking_validation_samples=int(
            ranking.get("ranking_validation_samples", pd.Series([0])).iloc[0]
        ),
        order_estimate=estimate,
        coefficients=coefficients,
    )


def _callbacks() -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_value_rmse",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_value_rmse",
            mode="min",
            patience=LR_PATIENCE,
            factor=0.5,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def _build_for_split(
    variant: str,
    dataset: MultiStepDataset,
    prepared: PreparedSplit,
    parameters: ModelParameters,
) -> tf.keras.Model:
    return build_model(
        variant,
        lookback=dataset.lookback,
        n_features=prepared.train_x.shape[-1],
        horizon=dataset.horizon,
        parameters=parameters,
        fractional_coefficients=prepared.coefficients,
    )


def fit_prepared_split(
    variant: str,
    dataset: MultiStepDataset,
    prepared: PreparedSplit,
    parameters: ModelParameters,
    *,
    epochs: int = MAX_EPOCHS,
    use_early_stopping: bool = True,
    additional_callbacks: Sequence[tf.keras.callbacks.Callback] = (),
) -> tuple[tf.keras.Model, tf.keras.callbacks.History, float]:
    """Fit one already-prepared split.  This is an explicitly expensive call."""
    training_started = perf_counter()
    model = _build_for_split(variant, dataset, prepared, parameters)
    if hasattr(model, "calibrate_initial_losses"):
        model.calibrate_initial_losses(
            prepared.train_x,
            prepared.train_targets,
        )
    fit_arguments: dict[str, Any] = {
        "x": prepared.train_x,
        "y": prepared.train_targets,
        "epochs": int(epochs),
        "batch_size": parameters.batch_size,
        "shuffle": False,
        "verbose": 0,
    }
    if use_early_stopping:
        fit_arguments.update(
            validation_data=(
                prepared.evaluation_x,
                prepared.evaluation_targets,
            ),
            callbacks=[*_callbacks(), *additional_callbacks],
        )
    else:
        fit_arguments["callbacks"] = [
            tf.keras.callbacks.TerminateOnNaN(),
            *additional_callbacks,
        ]
    history = model.fit(**fit_arguments)
    training_seconds = perf_counter() - training_started
    return model, history, float(training_seconds)


def value_predictions(model: tf.keras.Model, features: np.ndarray) -> np.ndarray:
    outputs = model.predict(features, verbose=0)
    if isinstance(outputs, dict):
        outputs = outputs["value"]
    return np.asarray(outputs)


def _best_epoch(
    history: tf.keras.callbacks.History,
    *,
    require_gradnorm_activation: bool = False,
) -> int:
    values = history.history.get("val_value_rmse")
    if not values:
        return len(history.epoch)
    eligible = np.arange(len(values), dtype=int)
    if require_gradnorm_activation:
        active = history.history.get("val_gradnorm_active")
        if active is None:
            active = history.history.get("gradnorm_active")
        if active is None:
            raise RuntimeError(
                "GradNorm activation history is missing from candidate training"
            )
        if len(active) != len(values):
            raise RuntimeError(
                "GradNorm activation and validation RMSE histories are misaligned"
            )
        eligible = np.flatnonzero(np.asarray(active, dtype=float) >= 0.5)
        if len(eligible) == 0:
            raise RuntimeError(
                "No active GradNorm epoch is eligible for best-epoch selection"
            )
    selected = eligible[int(np.argmin(np.asarray(values)[eligible]))]
    return int(selected) + 1


def _history_frame(
    history: tf.keras.callbacks.History, *, phase: str, fold: int | None
) -> pd.DataFrame:
    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    frame.insert(0, "fold", fold if fold is not None else "final")
    frame.insert(0, "phase", phase)
    return frame


def _fractional_metric(prepared: PreparedSplit, predictions: np.ndarray) -> float:
    if prepared.coefficients is None:
        return float("nan")
    value = fractional_rmse(
        tf.convert_to_tensor(prepared.evaluation_targets["context"]),
        tf.convert_to_tensor(prepared.evaluation_targets["value"]),
        tf.convert_to_tensor(predictions),
        tf.convert_to_tensor(prepared.coefficients),
    )
    return float(value.numpy())


def evaluate_cross_validation(
    *,
    dataset: MultiStepDataset,
    folds: Sequence[PurgedFold],
    variant: str,
    parameters: ModelParameters,
    seed: int,
    d_bootstrap_repeats: int = 0,
    feature_rankings: dict[int, pd.DataFrame] | None = None,
    fractional_preprocessing: dict[int, FractionalPreprocessing] | None = None,
    optuna_trial: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Train and evaluate all purged folds for one frozen configuration."""
    variant = canonical_variant(variant)
    metric_rows: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    best_epochs: list[int] = []
    for fold_position, fold in enumerate(folds, start=1):
        set_reproducible_seed(seed)
        cached_ranking = (
            feature_rankings.get(fold.number)
            if feature_rankings is not None
            else None
        )
        if feature_rankings is not None and cached_ranking is None:
            raise KeyError(f"Missing cached feature ranking for fold {fold.number}")
        cached_fractional = (
            fractional_preprocessing.get(fold.number)
            if fractional_preprocessing is not None
            else None
        )
        if (
            variant != VALUE_ONLY
            and fractional_preprocessing is not None
            and cached_fractional is None
        ):
            raise KeyError(
                f"Missing cached fractional preprocessing for fold {fold.number}"
            )
        prepared = prepare_split(
            dataset,
            fold.train_positions,
            fold.validation_positions,
            parameters=parameters,
            variant=variant,
            seed=seed,
            d_bootstrap_repeats=d_bootstrap_repeats,
            feature_ranking=cached_ranking,
            fractional_preprocessing=cached_fractional,
        )
        model, history, training_seconds = fit_prepared_split(
            variant,
            dataset,
            prepared,
            parameters,
        )
        prediction_scaled = value_predictions(model, prepared.evaluation_x)
        prediction = prepared.transformers.inverse_values(prediction_scaled)
        actual = dataset.values[fold.validation_positions]
        best_epoch = _best_epoch(
            history,
            require_gradnorm_activation=(variant != VALUE_ONLY),
        )
        best_epochs.append(best_epoch)
        metric_row = {
            "fold": fold.number,
            "train_samples": len(fold.train_positions),
            "validation_samples": len(fold.validation_positions),
            "validation_target_start": fold.validation_target_start,
            "validation_target_end": fold.validation_target_end,
            "value_rmse": global_rmse_numpy(actual, prediction),
            "fractional_rmse_standardized": _fractional_metric(
                prepared, prediction_scaled
            ),
            "best_epoch": best_epoch,
            "selected_features": json.dumps(prepared.selected_features),
            "feature_ranking_method": prepared.feature_ranking_method,
            "feature_ranking_train_samples": prepared.feature_ranking_train_samples,
            "feature_ranking_validation_samples": (
                prepared.feature_ranking_validation_samples
            ),
            "fractional_order": (
                prepared.order_estimate.selected
                if prepared.order_estimate is not None
                else np.nan
            ),
            "fractional_truncation": (
                len(prepared.coefficients) - 1
                if prepared.coefficients is not None
                else np.nan
            ),
            "training_seconds": training_seconds,
            "trained_epochs": len(history.epoch),
        }
        metric_rows.append(metric_row)
        histories.append(_history_frame(history, phase="cross_validation", fold=fold.number))
        if optuna_trial is not None:
            report_fold_pruning(
                optuna_trial,
                fold_position=fold_position,
                completed_metric_rows=metric_rows,
            )
    return pd.DataFrame(metric_rows), pd.concat(histories, ignore_index=True), best_epochs


def build_tuning_context(
    config_path: str | Path,
    *,
    lookback: int,
    seed: int,
) -> TuningContext:
    """Build each fold ranking once for reuse by every matching Optuna trial."""
    dataset = load_multistep_dataset(config_path, lookback=lookback)
    folds = purged_walk_forward_splits(
        dataset.sample_index,
        dataset.development_positions,
        n_splits=N_SPLITS,
        min_train_origins=configured_min_train_origins(config_path),
    )
    rankings = {}
    for fold in folds:
        intervals = dataset.sample_index.iloc[fold.train_positions]
        rankings[fold.number] = rank_features(
            dataset.features[fold.train_positions],
            dataset.values[fold.train_positions],
            dataset.feature_columns,
            seed=seed,
            target_starts=intervals["target_start"].to_numpy(),
            target_ends=intervals["target_end"].to_numpy(),
        )
    return TuningContext(dataset=dataset, folds=folds, feature_rankings=rankings)


def tune_baseline(
    *,
    config_path: str | Path,
    experiment_id: str,
    n_trials: int = 40,
    seed: int = 42,
) -> dict[str, Any]:
    """Tune the value-only baseline; other objective bundles reuse its lock."""
    import optuna

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    tuning_cache: dict[int, TuningContext] = {}

    def context_for(lookback: int) -> TuningContext:
        if lookback not in tuning_cache:
            tuning_cache[lookback] = build_tuning_context(
                config_path,
                lookback=lookback,
                seed=seed,
            )
        return tuning_cache[lookback]

    def objective(trial: Any) -> float:
        parameters = ModelParameters(
            encoder_units=trial.suggest_categorical("encoder_units", [16, 32, 64]),
            decoder_units=trial.suggest_categorical("decoder_units", [16, 32, 64]),
            dropout=trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
            learning_rate=trial.suggest_float(
                "learning_rate", 1e-4, 3e-3, log=True
            ),
            batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
            lookback=trial.suggest_categorical("lookback", [3, 4, 5, 6, 8, 10]),
            top_k=trial.suggest_categorical("top_k", [4, 6, 8, 10, 12, 16]),
        )
        context = context_for(parameters.lookback)
        metrics, _, _ = evaluate_cross_validation(
            dataset=context.dataset,
            folds=context.folds,
            variant=VALUE_ONLY,
            parameters=parameters,
            seed=seed,
            feature_rankings=context.feature_rankings,
            optuna_trial=trial,
        )
        return float(metrics["value_rmse"].mean())

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
        interval_steps=1,
    )
    configured_horizon = int(load_config(str(Path(config_path).resolve()))["HORIZON"])
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{experiment_id}_horizon_{configured_horizon}_baseline",
    )
    study.optimize(objective, n_trials=int(n_trials))
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError("Optuna produced no completed baseline trial")
    best = ModelParameters(**study.best_params)
    dataset = load_multistep_dataset(config_path, lookback=best.lookback)
    payload = {
        "objective": "mean purged-fold global value RMSE",
        "direction": "minimize",
        "horizon": dataset.horizon,
        "tuning_variant": VALUE_ONLY,
        "tuning_seed": seed,
        "n_trials": int(n_trials),
        "n_splits": N_SPLITS,
        "completed_trials": len(completed_trials),
        "pruned_trials": sum(
            trial.state == optuna.trial.TrialState.PRUNED
            for trial in study.trials
        ),
        "pruner": {
            "name": "MedianPruner",
            "resource": "completed purged folds",
            "n_startup_trials": 5,
            "n_warmup_steps": 1,
            "interval_steps": 1,
            "monitor": "running mean original-scale value_rmse",
        },
        "feature_ranking_cache": {
            "enabled": True,
            "scope": "in-memory per tuning run, lookback, fold, and seed",
            "cached_lookbacks": sorted(tuning_cache),
        },
        "best_value": float(study.best_value),
        "parameters": asdict(best),
        "config_checksum": sha256_file(config_path),
        "config_snapshot": load_config(str(Path(config_path).resolve())),
    }
    input_files = configured_input_paths(config_path)
    publish_locked_parameters(
        config_path=config_path,
        experiment_id=experiment_id,
        payload=payload,
        input_files=input_files,
    )
    publish_tuning_trials(
        config_path=config_path,
        experiment_id=experiment_id,
        trials=study.trials_dataframe(),
        input_files=input_files,
    )
    return payload


def _model_parameters_from_lock(payload: dict[str, Any]) -> ModelParameters:
    return ModelParameters(**payload["parameters"])


def run_variant(
    *,
    config_path: str | Path,
    experiment_id: str,
    variant: str,
    seed: int,
    d_bootstrap_repeats: int = 0,
    _context: VariantRunContext | None = None,
) -> dict[str, Path]:
    """Run one frozen model/horizon/seed cell and publish isolated artifacts."""
    if d_bootstrap_repeats < 0:
        raise ValueError("d_bootstrap_repeats must be non-negative")
    variant = canonical_variant(variant)
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    lock = (
        _context.lock
        if _context is not None
        else load_locked_parameters(config_path, experiment_id)
    )
    if lock.get("config_checksum") != sha256_file(config_path):
        raise ValueError(
            "The current config differs from the config used to tune the locked "
            "parameters. Re-run value-only baseline tuning for this exact configuration."
        )
    parameters = _model_parameters_from_lock(lock)
    dataset = (
        _context.dataset
        if _context is not None
        else load_multistep_dataset(config_path, lookback=parameters.lookback)
    )
    if int(lock["horizon"]) != dataset.horizon:
        raise ValueError("Locked parameters belong to a different horizon")
    folds: Sequence[PurgedFold] = (
        _context.folds
        if _context is not None
        else purged_walk_forward_splits(
            dataset.sample_index,
            dataset.development_positions,
            n_splits=N_SPLITS,
            min_train_origins=configured_min_train_origins(config_path),
        )
    )
    fold_metrics, histories, best_epochs = evaluate_cross_validation(
        dataset=dataset,
        folds=folds,
        variant=variant,
        parameters=parameters,
        seed=seed,
        d_bootstrap_repeats=d_bootstrap_repeats,
        feature_rankings=(
            _context.feature_rankings if _context is not None else None
        ),
        fractional_preprocessing=(
            _context.fractional_preprocessing if _context is not None else None
        ),
    )

    evaluation_positions = dataset.evaluation_positions
    final_epochs = max(1, int(round(float(np.median(best_epochs)))))
    set_reproducible_seed(seed)
    prepared = prepare_split(
        dataset,
        dataset.development_positions,
        evaluation_positions,
        parameters=parameters,
        variant=variant,
        seed=seed,
        d_bootstrap_repeats=d_bootstrap_repeats,
        feature_ranking=(
            _context.final_feature_ranking if _context is not None else None
        ),
        fractional_preprocessing=(
            _context.final_fractional_preprocessing
            if _context is not None
            else None
        ),
    )
    model, final_history, final_training_seconds = fit_prepared_split(
        variant,
        dataset,
        prepared,
        parameters,
        epochs=final_epochs,
        use_early_stopping=False,
    )
    histories = pd.concat(
        [histories, _history_frame(final_history, phase="final_fit", fold=None)],
        ignore_index=True,
    )
    prediction_scaled = value_predictions(model, prepared.evaluation_x)
    prediction = prepared.transformers.inverse_values(prediction_scaled)
    actual = dataset.values[evaluation_positions]
    task_weights = None
    initial_task_losses = None
    if hasattr(model, "normalized_task_weights"):
        task_weights = {
            name: float(weight)
            for name, weight in zip(
                model.task_names, model.normalized_task_weights().numpy()
            )
        }
        initial_task_losses = {
            name: float(value)
            for name, value in zip(model.task_names, model.initial_losses.numpy())
        }
    metadata = {
        "scientific_role": {
            VALUE_ONLY: "Value-only baseline: value RMSE",
            VALUE_FRACTIONAL: (
                "Value plus fractional-difference RMSE, balanced with GradNorm"
            ),
            VALUE_FRACTIONAL_SPECTRUM: (
                "Value plus fractional-difference plus singularity-spectrum RMSE, "
                "balanced with GradNorm"
            ),
        }[variant],
        "horizon": dataset.horizon,
        "rmse_aggregation": "global over all samples and horizon positions",
        "evaluation_value_rmse": global_rmse_numpy(actual, prediction),
        "evaluation_origin_count": len(evaluation_positions),
        "evaluation_origin_stride": (
            int(np.diff(dataset.evaluation_origins)[0])
            if len(dataset.evaluation_origins) > 1
            else None
        ),
        "evaluation_origins": dataset.evaluation_origins,
        "evaluation_policy": (
            "one fixed model trained before the first rolling origin; no refit "
            "inside the evaluation period"
        ),
        "parameters": asdict(parameters),
        "final_epochs": final_epochs,
        "timing": {
            "clock": "time.perf_counter monotonic wall clock",
            "unit": "seconds",
            "scope": (
                "model build, GradNorm calibration when applicable, and model.fit; "
                "excludes preprocessing, prediction, and artifact publication"
            ),
            "cross_validation_training_seconds": float(
                fold_metrics["training_seconds"].sum()
            ),
            "final_training_seconds": final_training_seconds,
            "total_training_seconds": float(
                fold_metrics["training_seconds"].sum() + final_training_seconds
            ),
            "cross_validation_trained_epochs": int(
                fold_metrics["trained_epochs"].sum()
            ),
            "final_trained_epochs": len(final_history.epoch),
            "total_trained_epochs": int(
                fold_metrics["trained_epochs"].sum() + len(final_history.epoch)
            ),
        },
        "selected_features": prepared.selected_features,
        "feature_ranking": {
            "method": prepared.feature_ranking_method,
            "training_samples": prepared.feature_ranking_train_samples,
            "validation_samples": prepared.feature_ranking_validation_samples,
        },
        "feature_scaler_mean": prepared.transformers.feature_scaler.mean_,
        "feature_scaler_scale": prepared.transformers.feature_scaler.scale_,
        "value_scaler_mean": prepared.transformers.value_scaler.mean_,
        "value_scaler_scale": prepared.transformers.value_scaler.scale_,
        "spectrum_scaler_mean": (
            prepared.transformers.spectrum_scaler.mean_
            if prepared.transformers.spectrum_scaler is not None
            else None
        ),
        "spectrum_scaler_scale": (
            prepared.transformers.spectrum_scaler.scale_
            if prepared.transformers.spectrum_scaler is not None
            else None
        ),
        "fractional_order_estimate": (
            prepared.order_estimate.to_dict()
            if prepared.order_estimate is not None
            else None
        ),
        "fractional_coefficients": prepared.coefficients,
        "gradnorm_final_weights": task_weights,
        "gradnorm_initial_losses": initial_task_losses,
        "gradnorm_initialization": (
            (
                "full training-split calibration before optimizer updates; "
                "active from the first training step"
            )
            if variant != VALUE_ONLY
            else None
        ),
        "gradnorm_gradient_reuse": (
            (
                "one unweighted shared-decoder gradient per task; exact weighted "
                "norm obtained by multiplying by the positive task weight"
            )
            if variant != VALUE_ONLY
            else None
        ),
        "matrix_preprocessing_cache": (
            {
                "scope": "current run_matrix call",
                "feature_rankings": "shared across variants for the same seed and split",
                "fractional_preprocessing": (
                    "not applicable"
                    if variant == VALUE_ONLY
                    else (
                        "shared across variants and seeds"
                        if d_bootstrap_repeats == 0
                        else "shared across variants for the same seed"
                    )
                ),
            }
            if _context is not None
            else None
        ),
        "locked_tuning": lock,
    }
    target_starts = dataset.sample_index.iloc[evaluation_positions][
        "target_start"
    ].to_numpy(dtype=int)
    forecast_origins = dataset.sample_index.iloc[evaluation_positions][
        "forecast_origin"
    ].to_numpy(dtype=int)
    target_dates = np.asarray(
        [
            dataset.raw_dates[start : start + dataset.horizon].to_numpy()
            for start in target_starts
        ]
    )
    forecast_origin_dates = dataset.raw_dates[forecast_origins]
    inference_network = model.network if hasattr(model, "network") else model
    return publish_experiment_artifacts(
        config_path=config_path,
        experiment_id=experiment_id,
        variant=variant,
        seed=seed,
        forecast_origins=forecast_origins,
        forecast_origin_dates=forecast_origin_dates,
        target_dates=target_dates,
        actual=actual,
        predicted=prediction,
        fold_metrics=fold_metrics,
        training_history=histories,
        metadata=metadata,
        input_files=configured_input_paths(config_path),
        network=inference_network,
    )


def run_matrix(
    *,
    config_path: str | Path,
    experiment_id: str,
    variants: Sequence[str] = MODEL_VARIANTS,
    seeds: Sequence[int] = SEEDS,
    d_bootstrap_repeats: int = 0,
) -> list[dict[str, Path]]:
    """Run a complete matrix with fold-local preprocessing reused safely."""
    if d_bootstrap_repeats < 0:
        raise ValueError("d_bootstrap_repeats must be non-negative")
    canonical_variants = tuple(canonical_variant(variant) for variant in variants)
    unknown = [
        variant for variant in canonical_variants if variant not in MODEL_VARIANTS
    ]
    if unknown:
        raise ValueError(f"Unknown model variants: {unknown}")
    if len(set(canonical_variants)) != len(canonical_variants):
        raise ValueError("Model variants contain duplicate canonical names")

    normalized_seeds = tuple(int(seed) for seed in seeds)
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("Seeds must be unique within one matrix run")

    lock = load_locked_parameters(config_path, experiment_id)
    if lock.get("config_checksum") != sha256_file(config_path):
        raise ValueError(
            "The current config differs from the config used to tune the locked "
            "parameters. Re-run value-only baseline tuning for this exact configuration."
        )
    parameters = _model_parameters_from_lock(lock)
    dataset = load_multistep_dataset(config_path, lookback=parameters.lookback)
    if int(lock["horizon"]) != dataset.horizon:
        raise ValueError("Locked parameters belong to a different horizon")
    folds = tuple(
        purged_walk_forward_splits(
            dataset.sample_index,
            dataset.development_positions,
            n_splits=N_SPLITS,
            min_train_origins=configured_min_train_origins(config_path),
        )
    )
    evaluation_positions = dataset.evaluation_positions
    cache = MatrixPreprocessingCache(dataset)

    outputs = []
    for variant in canonical_variants:
        for seed in normalized_seeds:
            fold_rankings = {
                fold.number: cache.feature_ranking(
                    fold.train_positions,
                    seed=seed,
                )
                for fold in folds
            }
            final_ranking = cache.feature_ranking(
                dataset.development_positions,
                seed=seed,
            )
            fold_fractional = None
            final_fractional = None
            if variant != VALUE_ONLY:
                fold_fractional = {
                    fold.number: cache.fractional_preprocessing(
                        fold.train_positions,
                        fold.validation_positions,
                        bootstrap_repeats=d_bootstrap_repeats,
                        seed=seed,
                    )
                    for fold in folds
                }
                final_fractional = cache.fractional_preprocessing(
                    dataset.development_positions,
                    evaluation_positions,
                    bootstrap_repeats=d_bootstrap_repeats,
                    seed=seed,
                )
            context = VariantRunContext(
                lock=lock,
                dataset=dataset,
                folds=folds,
                feature_rankings=fold_rankings,
                final_feature_ranking=final_ranking,
                fractional_preprocessing=fold_fractional,
                final_fractional_preprocessing=final_fractional,
            )
            outputs.append(
                run_variant(
                    config_path=config_path,
                    experiment_id=experiment_id,
                    variant=variant,
                    seed=seed,
                    d_bootstrap_repeats=d_bootstrap_repeats,
                    _context=context,
                )
            )
    return outputs


def validate_design(config_path: str | Path, *, lookback: int = 6) -> dict[str, Any]:
    """Lightweight contract validation; it performs no ranking, fitting, or tuning."""
    dataset = load_multistep_dataset(config_path, lookback=lookback)
    folds = purged_walk_forward_splits(
        dataset.sample_index,
        dataset.development_positions,
        n_splits=N_SPLITS,
        min_train_origins=configured_min_train_origins(config_path),
    )
    return {
        "horizon": dataset.horizon,
        "lookback": dataset.lookback,
        "samples": len(dataset.features),
        "development_samples": len(dataset.development_positions),
        "evaluation_positions": dataset.evaluation_positions.tolist(),
        "evaluation_origins": dataset.evaluation_origins.tolist(),
        "evaluation_origin_count": len(dataset.evaluation_origins),
        "minimum_train_origins": configured_min_train_origins(config_path),
        "folds": [
            {
                "fold": fold.number,
                "train_samples": len(fold.train_positions),
                "validation_samples": len(fold.validation_positions),
                "validation_target_start": fold.validation_target_start,
                "validation_target_end": fold.validation_target_end,
            }
            for fold in folds
        ],
        "feature_target_leakage": False,
        "training_started": False,
    }
