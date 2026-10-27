import os
from typing import List, Tuple

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils import load_config
from data_quality import (
    validate_feature_file,
    validate_output_lineage,
    validate_time_series_file,
)
from forecast_artifacts import publish_forecast_artifacts

try:
    from optuna_integration import TFKerasPruningCallback
except ImportError:  # pragma: no cover - compatibility fallback
    from optuna.integration import TFKerasPruningCallback


SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

BASE_FEATURE_COLS = [
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
]
TARGET_COL = "Future_Target"
TOP_K_OPTIONS = (4, 6, 8, 10, 12, 16)
N_SPLITS = 3
IMPORTANCE_REPEATS = 10
SELECTION_FREQUENCY_THRESHOLD = 0.75
MAX_EPOCHS = 150
N_TRIALS = 40


def project_path(*parts: str) -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, *parts)


def create_sequences(
    data: np.ndarray, target: np.ndarray, lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback, :])
        y.append(target[i + lookback])
    return np.array(X), np.array(y)


def create_validation_sequences(
    train_features: np.ndarray,
    val_features: np.ndarray,
    train_target: np.ndarray,
    val_target: np.ndarray,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    combined_features = np.vstack([train_features[-lookback:], val_features])
    combined_target = np.vstack([train_target[-lookback:], val_target])
    return create_sequences(combined_features, combined_target, lookback)


def rmse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute a differentiable RMSE loss independently for each sample."""
    error = tf.cast(y_pred, tf.float32) - tf.cast(y_true, tf.float32)
    return tf.sqrt(
        tf.reduce_mean(tf.square(error), axis=-1)
        + tf.keras.backend.epsilon()
    )


def create_lstm_model(
    n_steps: int,
    n_features: int,
    units: int,
    dropout_rate: float,
    learning_rate: float,
) -> Model:
    inputs = Input(shape=(n_steps, n_features))
    shared = LSTM(units=units, activation="tanh")(inputs)
    shared = Dropout(dropout_rate)(shared)
    value_output = Dense(1, name="value")(shared)
    model = Model(inputs=inputs, outputs=value_output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=rmse_loss,
        # Keep the metric name aligned with the pruning and early-stopping
        # monitor used by this script: val_value_rmse.
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="value_rmse")],
    )
    return model


def value_predictions(model: Model, features: np.ndarray) -> np.ndarray:
    outputs = model.predict(features, verbose=0)
    return np.asarray(outputs).reshape(-1, 1)


def load_aligned_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    config_params = load_config(project_path("data_lake", "config.json"))
    curated_folder = config_params["CURATED_FOLDER"]
    features_folder = config_params["FEATURES_FOLDER"]
    indicator = config_params["INDICATOR"]
    river_name = config_params["RIVER_NAME_VALUE_ENG"]
    horizon = config_params["HORIZON"]
    mfdfa_window = config_params["MFDFA_WINDOW"]

    ts_data_path = project_path(curated_folder, f"{river_name}_{indicator}_ts.csv")
    mfdfa_results_path = project_path(
        features_folder,
        f"MFDFA_{river_name}_{indicator}_{mfdfa_window}_window.csv",
    )
    validate_time_series_file(ts_data_path, indicator, horizon)
    config_path = project_path("data_lake", "config.json")
    validate_output_lineage(ts_data_path, config_path, "curated")
    ts_data = pd.read_csv(ts_data_path)
    mfdfa_results = pd.read_csv(mfdfa_results_path)
    validate_feature_file(
        mfdfa_results_path,
        BASE_FEATURE_COLS,
        time_series_rows=len(ts_data),
        mfdfa_window=mfdfa_window,
    )
    validate_output_lineage(
        mfdfa_results_path, config_path, "features_mfdfa"
    )
    mfdfa_results = mfdfa_results.rename(columns={mfdfa_results.columns[0]: "Index"})

    # The R rolling output index is the one-based position of the last
    # observation included in each MFDFA window.  That observation is already
    # known when the feature is calculated, so it must predict the next raw
    # observation.  Convert the one-based window-end index to the corresponding
    # zero-based target position directly; do not subtract one and reuse the
    # window's final observation as its own target.
    target_raw_indices = (
        pd.to_numeric(mfdfa_results["Index"], errors="raise").to_numpy(dtype=int)
    )
    valid_indices = target_raw_indices < len(ts_data)
    valid_positions = np.flatnonzero(valid_indices)
    final_target_indices = target_raw_indices[valid_indices]
    final_features_df = mfdfa_results.iloc[valid_positions].reset_index(drop=True).copy()
    final_targets = ts_data[indicator].iloc[final_target_indices].values
    final_features_df[TARGET_COL] = final_targets
    return ts_data, final_features_df, horizon


def rank_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Rank features using a temporal holdout inside the supplied training data.

    The caller must pass only the training portion of an outer CV fold. The
    final fifth of that data is reserved for permutation importance so the
    selector is not evaluated on the same rows used to fit its Random Forest.
    """
    if len(train_df) < 5:
        raise ValueError("Feature ranking requires at least 5 training rows")

    split_index = max(1, int(len(train_df) * 0.8))
    selector_train = train_df.iloc[:split_index]
    selector_validation = train_df.iloc[split_index:]

    # Very small folds cannot support a meaningful inner holdout. In that
    # case, retain the outer-fold training data for fitting and use the
    # model's deterministic impurity importance as a fallback.
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
    )
    if len(selector_validation) >= 2:
        model.fit(selector_train[BASE_FEATURE_COLS], selector_train[TARGET_COL])
        importance = permutation_importance(
            model,
            selector_validation[BASE_FEATURE_COLS],
            selector_validation[TARGET_COL],
            scoring="neg_root_mean_squared_error",
            n_repeats=IMPORTANCE_REPEATS,
            random_state=SEED,
            n_jobs=-1,
        )
        importances = importance.importances_mean
    else:
        model.fit(train_df[BASE_FEATURE_COLS], train_df[TARGET_COL])
        importances = model.feature_importances_

    return (
        pd.DataFrame(
            {"Feature": BASE_FEATURE_COLS, "Importance": importances}
        )
        .sort_values(
            by=["Importance", "Feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def build_cv_folds(train_val_df: pd.DataFrame) -> list[dict]:
    """Build outer temporal folds and cache one feature ranking per fold."""
    folds = []
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(train_val_df), start=1):
        fold_train = train_val_df.iloc[train_idx].reset_index(drop=True)
        fold_val = train_val_df.iloc[val_idx].reset_index(drop=True)
        if len(fold_train) == 0 or len(fold_val) == 0:
            raise ValueError(f"Temporal fold {fold_idx} is empty")
        folds.append(
            {
                "fold": fold_idx,
                "train": fold_train,
                "validation": fold_val,
                "ranking": rank_features(fold_train),
            }
        )
    return folds


def feature_stability(folds: list[dict], top_k: int) -> pd.DataFrame:
    """Summarize selection frequency and mean rank across temporal folds."""
    rank_tables = []
    for fold in folds:
        ranking = fold["ranking"]
        rank_tables.append(
            ranking.assign(
                Fold=fold["fold"],
                Rank=np.arange(1, len(ranking) + 1),
                Selected=ranking["Feature"].isin(ranking["Feature"].head(top_k)),
            )
        )

    stability = pd.concat(rank_tables, ignore_index=True)
    return (
        stability.groupby("Feature", as_index=False)
        .agg(
            Mean_Importance=("Importance", "mean"),
            Mean_Rank=("Rank", "mean"),
            Selection_Frequency=("Selected", "mean"),
        )
        .sort_values(
            by=["Selection_Frequency", "Mean_Rank", "Mean_Importance", "Feature"],
            ascending=[False, True, False, True],
        )
        .reset_index(drop=True)
    )


def select_final_features(
    train_val_df: pd.DataFrame,
    folds: list[dict],
    top_k: int,
) -> tuple[List[str], pd.DataFrame]:
    """Select stable features using only data before the final test window."""
    stability = feature_stability(folds, top_k)
    final_ranking = rank_features(train_val_df)
    stable_features = set(
        stability.loc[
            stability["Selection_Frequency"] >= SELECTION_FREQUENCY_THRESHOLD,
            "Feature",
        ]
    )

    selected_features = [
        feature
        for feature in final_ranking["Feature"]
        if feature in stable_features
    ][:top_k]
    selected_features.extend(
        feature
        for feature in final_ranking["Feature"]
        if feature not in selected_features
    )
    return selected_features[:top_k], stability


def objective(trial: optuna.Trial, folds: list[dict]) -> float:
    lookback = trial.suggest_int("lookback", 3, 8)
    units = trial.suggest_categorical("units", [16, 24, 32, 48, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 24, 32])
    top_k = trial.suggest_categorical("top_k", TOP_K_OPTIONS)

    fold_rmses = []

    for fold in folds:
        fold_idx = fold["fold"]
        fold_train = fold["train"]
        fold_val = fold["validation"]

        if len(fold_train) <= lookback or len(fold_val) == 0:
            raise optuna.TrialPruned()

        selected_features = fold["ranking"]["Feature"].head(top_k).tolist()

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        train_features = feature_scaler.fit_transform(fold_train[selected_features])
        train_target = target_scaler.fit_transform(fold_train[[TARGET_COL]])
        val_features = feature_scaler.transform(fold_val[selected_features])
        val_target = target_scaler.transform(fold_val[[TARGET_COL]])

        X_train, y_train = create_sequences(train_features, train_target, lookback)
        X_val, y_val = create_validation_sequences(
            train_features, val_features, train_target, val_target, lookback
        )

        if len(X_train) == 0 or len(X_val) == 0:
            raise optuna.TrialPruned()

        tf.keras.backend.clear_session()
        model = create_lstm_model(
            n_steps=X_train.shape[1],
            n_features=X_train.shape[2],
            units=units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )

        callbacks = [
            EarlyStopping(
                monitor="val_value_rmse",
                mode="min",
                patience=10,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_value_rmse",
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-5,
            ),
            TFKerasPruningCallback(trial, "val_value_rmse"),
        ]

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=batch_size,
            shuffle=False,
            verbose=0,
            callbacks=callbacks,
        )

        scaled_preds = value_predictions(model, X_val)
        preds = target_scaler.inverse_transform(scaled_preds).ravel()
        actuals = fold_val[TARGET_COL].to_numpy()

        rmse = np.sqrt(mean_squared_error(actuals, preds))
        fold_rmses.append(rmse)

        trial.report(float(np.mean(fold_rmses)), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_rmses))


def fit_final_model(
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_params: dict,
    folds: list[dict],
) -> Tuple[Model, np.ndarray, np.ndarray, List[str]]:
    selected_features, _ = select_final_features(
        train_val_df,
        folds,
        best_params["top_k"],
    )
    lookback = best_params["lookback"]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_features = feature_scaler.fit_transform(train_val_df[selected_features])
    train_target = target_scaler.fit_transform(train_val_df[[TARGET_COL]])
    test_features = feature_scaler.transform(test_df[selected_features])

    X_train, y_train = create_sequences(train_features, train_target, lookback)
    X_test, _ = create_validation_sequences(
        train_features,
        test_features,
        train_target,
        np.zeros((len(test_df), 1)),
        lookback,
    )

    tf.keras.backend.clear_session()
    model = create_lstm_model(
        n_steps=X_train.shape[1],
        n_features=X_train.shape[2],
        units=best_params["units"],
        dropout_rate=best_params["dropout_rate"],
        learning_rate=best_params["learning_rate"],
    )

    val_size = max(1, int(len(X_train) * 0.1))
    if len(X_train) > val_size:
        X_fit, X_val = X_train[:-val_size], X_train[-val_size:]
        y_fit, y_val = y_train[:-val_size], y_train[-val_size:]
        callbacks = [
            EarlyStopping(
                monitor="val_value_rmse",
                mode="min",
                patience=10,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_value_rmse",
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-5,
            ),
        ]
        model.fit(
            X_fit,
            y_fit,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=best_params["batch_size"],
            shuffle=False,
            verbose=0,
            callbacks=callbacks,
        )
    else:
        callbacks = [
            EarlyStopping(
                monitor="value_rmse",
                mode="min",
                patience=10,
                restore_best_weights=True,
            )
        ]
        model.fit(
            X_train,
            y_train,
            epochs=MAX_EPOCHS,
            batch_size=best_params["batch_size"],
            shuffle=False,
            verbose=0,
            callbacks=callbacks,
        )

    scaled_preds = value_predictions(model, X_test)
    preds = target_scaler.inverse_transform(scaled_preds).ravel()
    actuals = test_df[TARGET_COL].to_numpy()
    return model, preds, actuals, selected_features


def main() -> None:
    ts_data, final_features_df, horizon = load_aligned_dataset()
    config_path = project_path("data_lake", "config.json")
    config_params = load_config(config_path)
    indicator = config_params["INDICATOR"]
    river_name = config_params["RIVER_NAME_VALUE_ENG"]
    forecasting_folder = project_path(config_params["FORECASTING_FOLDER"])
    os.makedirs(forecasting_folder, exist_ok=True)
    curated_path = project_path(
        config_params["CURATED_FOLDER"], f"{river_name}_{indicator}_ts.csv"
    )
    features_path = project_path(
        config_params["FEATURES_FOLDER"],
        f"MFDFA_{river_name}_{indicator}_{config_params['MFDFA_WINDOW']}_window.csv",
    )
    train_val_df = final_features_df.iloc[:-horizon].reset_index(drop=True)
    test_df = final_features_df.iloc[-horizon:].reset_index(drop=True)
    cv_folds = build_cv_folds(train_val_df)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED),
        pruner=HyperbandPruner(min_resource=10, max_resource=MAX_EPOCHS, reduction_factor=3),
        study_name="river_lstm_optuna",
    )
    study.optimize(lambda trial: objective(trial, cv_folds), n_trials=N_TRIALS)

    best_params = study.best_trial.params
    model, preds, actuals, selected_features = fit_final_model(
        train_val_df, test_df, best_params, cv_folds
    )
    stability = feature_stability(cv_folds, best_params["top_k"])
    test_rmse = np.sqrt(mean_squared_error(actuals, preds))

    print("--- CHAMPION MODEL ---")
    print(f"Features: {selected_features}")
    print(f"Params: {best_params}")
    print(f"Best CV RMSE: {study.best_value:.4f}")
    print(f"Final Test RMSE (on {horizon} points): {test_rmse:.4f}")
    print("Feature stability across temporal folds:")
    print(stability.to_string(index=False))

    artifacts = publish_forecast_artifacts(
        ts_data=ts_data,
        preds=preds,
        actuals=actuals,
        horizon=horizon,
        indicator=indicator,
        river_name=river_name,
        best_params=best_params,
        best_cv_rmse=study.best_value,
        test_rmse=test_rmse,
        selected_features=selected_features,
        stability=stability,
        forecasting_folder=forecasting_folder,
        config_path=config_path,
        input_files=[curated_path, features_path],
    )
    for artifact_type, artifact_path in artifacts.items():
        print(f"Published {artifact_type}: {artifact_path}")


if __name__ == "__main__":
    main()
