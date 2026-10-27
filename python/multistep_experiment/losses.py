"""RMSE and fractional-difference operations used by every model variant."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .constants import (
    FRACTIONAL_ENERGY_THRESHOLD,
    FRACTIONAL_MAX_TRUNCATION,
    FRACTIONAL_MIN_TRUNCATION,
)


@tf.keras.utils.register_keras_serializable(package="water_analyzer")
def global_rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """RMSE over all batch, horizon, and output dimensions."""
    error = tf.cast(y_pred, tf.float32) - tf.cast(y_true, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(error)) + tf.keras.backend.epsilon())


def masked_global_rmse(
    y_true: tf.Tensor, y_pred: tf.Tensor, mask: tf.Tensor
) -> tf.Tensor:
    error = tf.cast(y_pred, tf.float32) - tf.cast(y_true, tf.float32)
    valid = tf.cast(mask, tf.float32)
    squared_error = tf.square(error) * valid
    denominator = tf.maximum(tf.reduce_sum(valid), 1.0)
    return tf.sqrt(
        tf.reduce_sum(squared_error) / denominator + tf.keras.backend.epsilon()
    )


def fractional_coefficients(
    order: float,
    *,
    available_context: int,
    energy_threshold: float = FRACTIONAL_ENERGY_THRESHOLD,
    minimum: int = FRACTIONAL_MIN_TRUNCATION,
    maximum: int = FRACTIONAL_MAX_TRUNCATION,
) -> np.ndarray:
    """Generate a deterministic finite fractional kernel from estimated ``d``."""
    if not -0.5 < order < 0.5:
        raise ValueError("Fractional order must lie strictly between -0.5 and 0.5")
    max_order = min(int(maximum), int(available_context))
    if max_order < 1:
        raise ValueError("At least one historical observation is required")
    minimum = min(max(1, int(minimum)), max_order)

    coefficients = np.empty(max_order + 1, dtype=np.float64)
    coefficients[0] = 1.0
    for index in range(1, max_order + 1):
        coefficients[index] = (
            coefficients[index - 1] * (index - 1 - order) / index
        )
    energy = np.square(coefficients)
    cumulative = np.cumsum(energy) / np.sum(energy)
    selected = int(np.searchsorted(cumulative, energy_threshold))
    selected = min(max(selected, minimum), max_order)
    return coefficients[: selected + 1].astype(np.float32)


def fractional_future(
    context: tf.Tensor,
    future: tf.Tensor,
    coefficients: tf.Tensor,
) -> tf.Tensor:
    """Fractionally difference future points using known pre-origin context."""
    context = tf.cast(context, tf.float32)
    future = tf.cast(future, tf.float32)
    coefficients = tf.reshape(tf.cast(coefficients, tf.float32), (-1,))
    required_context = tf.shape(coefficients)[0] - 1
    tf.debugging.assert_equal(
        tf.shape(context)[1],
        required_context,
        message="Fractional context length must equal kernel order",
    )
    combined = tf.concat([context, future], axis=1)
    kernel = tf.reshape(tf.reverse(coefficients, axis=[0]), (-1, 1, 1))
    return tf.nn.conv1d(combined, filters=kernel, stride=1, padding="VALID")


def fractional_rmse(
    context: tf.Tensor,
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    coefficients: tf.Tensor,
) -> tf.Tensor:
    actual = fractional_future(context, y_true, coefficients)
    predicted = fractional_future(context, y_pred, coefficients)
    return global_rmse(actual, predicted)
