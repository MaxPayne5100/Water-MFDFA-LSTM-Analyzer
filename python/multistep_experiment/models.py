"""Horizon-independent encoder-decoder model variants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .constants import (
    GRADNORM_ALPHA,
    MODEL_VARIANTS,
    SPECTRUM_TARGET_COLUMNS,
    VALUE_FRACTIONAL_SPECTRUM,
    VALUE_ONLY,
    canonical_variant,
)
from .gradnorm import GradNormModel
from .losses import global_rmse


@dataclass(frozen=True)
class ModelParameters:
    encoder_units: int = 32
    decoder_units: int = 32
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 16
    lookback: int = 6
    top_k: int = 16


def build_network(
    *,
    lookback: int,
    n_features: int,
    horizon: int,
    parameters: ModelParameters,
    include_spectrum: bool,
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(lookback, n_features), name="features")
    encoded = tf.keras.layers.LSTM(
        parameters.encoder_units, activation="tanh", name="encoder"
    )(inputs)
    encoded = tf.keras.layers.Dropout(parameters.dropout, name="encoder_dropout")(
        encoded
    )
    repeated = tf.keras.layers.RepeatVector(horizon, name="repeat_horizon")(encoded)
    decoded = tf.keras.layers.LSTM(
        parameters.decoder_units,
        activation="tanh",
        return_sequences=True,
        name="shared_decoder",
    )(repeated)
    decoded = tf.keras.layers.Dropout(parameters.dropout, name="decoder_dropout")(
        decoded
    )
    value = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1), name="value"
    )(decoded)
    outputs: dict[str, tf.Tensor] = {"value": value}
    if include_spectrum:
        outputs["spectrum"] = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(len(SPECTRUM_TARGET_COLUMNS)), name="spectrum"
        )(decoded)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="multistep_lstm")


def build_model(
    variant: str,
    *,
    lookback: int,
    n_features: int,
    horizon: int,
    parameters: ModelParameters,
    fractional_coefficients: np.ndarray | None = None,
) -> tf.keras.Model:
    variant = canonical_variant(variant)
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    include_spectrum = variant == VALUE_FRACTIONAL_SPECTRUM
    network = build_network(
        lookback=lookback,
        n_features=n_features,
        horizon=horizon,
        parameters=parameters,
        include_spectrum=include_spectrum,
    )
    if variant == VALUE_ONLY:
        network.compile(
            optimizer=tf.keras.optimizers.Adam(parameters.learning_rate),
            loss={"value": global_rmse},
            metrics={
                # Keep the metric name identical to the custom GradNorm
                # model. Keras otherwise emits ``val_rmse`` for this
                # single-output dictionary model, while pruning and early
                # stopping consume the stable ``val_value_rmse`` contract.
                "value": [
                    tf.keras.metrics.RootMeanSquaredError(name="value_rmse")
                ]
            },
        )
        return network
    if fractional_coefficients is None:
        raise ValueError(
            "Fractional-difference variants require fractional coefficients"
        )
    model = GradNormModel(
        network,
        fractional_coefficients=fractional_coefficients,
        include_spectrum=include_spectrum,
        alpha=GRADNORM_ALPHA,
    )
    optimizer = tf.keras.optimizers.Adam(parameters.learning_rate)
    # Register the complete functional network explicitly. This is also a
    # defensive guarantee for Keras versions whose Model.compile lifecycle
    # does not build a custom model's optimizer immediately.
    optimizer.build(network.trainable_variables)
    model.compile(
        optimizer=optimizer,
        run_eagerly=True,
    )
    return model
