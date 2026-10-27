"""Keras model wrapper implementing dynamic GradNorm task balancing."""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from .losses import fractional_rmse, global_rmse, masked_global_rmse


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


class GradNormModel(tf.keras.Model):
    """Balance value, fractional, and optional spectrum RMSE dynamically.

    The implementation intentionally runs eagerly because GradNorm requires
    gradients of per-task gradient norms.  This favors auditability over raw
    throughput for the repository's small experimental dataset.
    """

    def __init__(
        self,
        network: tf.keras.Model,
        *,
        fractional_coefficients: np.ndarray,
        include_spectrum: bool,
        alpha: float = 1.5,
        weight_learning_rate: float = 0.025,
    ) -> None:
        super().__init__(name=f"gradnorm_{network.name}")
        self.network = network
        self.include_spectrum = include_spectrum
        self.task_names = (
            ("value", "fractional", "spectrum")
            if include_spectrum
            else ("value", "fractional")
        )
        self.alpha = float(alpha)
        self.fractional_coefficients = tf.constant(
            fractional_coefficients, dtype=tf.float32
        )
        self.log_task_weights = self.add_weight(
            name="log_task_weights",
            shape=(len(self.task_names),),
            initializer=tf.keras.initializers.Constant(_inverse_softplus(1.0)),
            trainable=True,
        )
        self.initial_losses = self.add_weight(
            name="initial_task_losses",
            shape=(len(self.task_names),),
            initializer="ones",
            trainable=False,
        )
        self._initial_losses_set = False
        self.weight_optimizer = tf.keras.optimizers.Adam(weight_learning_rate)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.value_loss_tracker = tf.keras.metrics.Mean(name="value_loss")
        self.fractional_loss_tracker = tf.keras.metrics.Mean(name="fractional_loss")
        self.value_rmse_metric = tf.keras.metrics.RootMeanSquaredError(
            name="value_rmse"
        )
        self.gradnorm_active_tracker = tf.keras.metrics.Mean(
            name="gradnorm_active"
        )
        self.spectrum_loss_tracker = (
            tf.keras.metrics.Mean(name="spectrum_loss")
            if include_spectrum
            else None
        )
        self.spectrum_rmse_tracker = (
            tf.keras.metrics.Mean(name="spectrum_rmse")
            if include_spectrum
            else None
        )
        self.weight_trackers = {
            name: tf.keras.metrics.Mean(name=f"{name}_weight")
            for name in self.task_names
        }
        self.gradient_norm_trackers = {
            name: tf.keras.metrics.Mean(name=f"{name}_gradient_norm")
            for name in self.task_names
        }
        self.cosine_trackers = {
            f"value_{name}": tf.keras.metrics.Mean(
                name=f"value_{name}_gradient_cosine"
            )
            for name in self.task_names
            if name != "value"
        }

    def compile(self, *args: Any, **kwargs: Any) -> None:
        """Register every optimizer variable before training starts.

        Registering the full functional network prevents Keras from rejecting
        task-head variables that may not receive a gradient in every step.
        """
        super().compile(*args, **kwargs)
        if not self.optimizer.built:
            self.optimizer.build(self.network.trainable_variables)
        if not self.weight_optimizer.built:
            self.weight_optimizer.build([self.log_task_weights])

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        metrics: list[tf.keras.metrics.Metric] = [
            self.total_loss_tracker,
            self.value_loss_tracker,
            self.fractional_loss_tracker,
            self.value_rmse_metric,
            self.gradnorm_active_tracker,
        ]
        if self.spectrum_loss_tracker is not None:
            metrics.extend([self.spectrum_loss_tracker, self.spectrum_rmse_tracker])
        metrics.extend(self.weight_trackers.values())
        metrics.extend(self.gradient_norm_trackers.values())
        metrics.extend(self.cosine_trackers.values())
        return metrics

    def call(self, inputs: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        return self.network(inputs, training=training)

    def normalized_task_weights(self) -> tf.Tensor:
        positive = tf.nn.softplus(self.log_task_weights) + 1e-7
        return positive * len(self.task_names) / tf.reduce_sum(positive)

    def calibrate_initial_losses(
        self,
        features: np.ndarray | tf.Tensor,
        targets: dict[str, np.ndarray | tf.Tensor],
    ) -> dict[str, float]:
        """Calibrate task baselines on the training split without updates."""
        if int(self.optimizer.iterations.numpy()) != 0:
            raise RuntimeError(
                "Initial-loss calibration must happen before optimizer updates"
            )
        tensor_targets = {
            name: tf.convert_to_tensor(value) for name, value in targets.items()
        }
        outputs = self.network(
            tf.convert_to_tensor(features),
            training=False,
        )
        loss_map = self._task_losses(tensor_targets, outputs)
        calibrated = tf.stack([loss_map[name] for name in self.task_names])
        tf.debugging.assert_all_finite(
            calibrated,
            "Non-finite GradNorm initial-loss calibration",
        )
        tf.debugging.assert_positive(
            calibrated,
            "GradNorm initial losses must be positive",
        )
        self.initial_losses.assign(calibrated)
        self._initial_losses_set = True
        return {
            name: float(value)
            for name, value in zip(self.task_names, calibrated.numpy())
        }

    def _task_losses(
        self,
        targets: dict[str, tf.Tensor],
        outputs: dict[str, tf.Tensor],
    ) -> dict[str, tf.Tensor]:
        losses = {
            "value": global_rmse(targets["value"], outputs["value"]),
            "fractional": fractional_rmse(
                targets["context"],
                targets["value"],
                outputs["value"],
                self.fractional_coefficients,
            ),
        }
        if self.include_spectrum:
            losses["spectrum"] = masked_global_rmse(
                targets["spectrum"],
                outputs["spectrum"],
                targets["spectrum_mask"],
            )
        return losses

    @staticmethod
    def _gradient_cosine(
        left: list[tf.Tensor | None], right: list[tf.Tensor | None]
    ) -> tf.Tensor:
        products = []
        left_norms = []
        right_norms = []
        for left_value, right_value in zip(left, right):
            if left_value is None or right_value is None:
                continue
            products.append(tf.reduce_sum(left_value * right_value))
            left_norms.append(tf.reduce_sum(tf.square(left_value)))
            right_norms.append(tf.reduce_sum(tf.square(right_value)))
        if not products:
            return tf.constant(0.0, dtype=tf.float32)
        numerator = tf.add_n(products)
        denominator = tf.sqrt(tf.add_n(left_norms) * tf.add_n(right_norms)) + 1e-7
        return numerator / denominator

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        if not tf.executing_eagerly():
            raise RuntimeError("GradNormModel must be compiled with run_eagerly=True")
        if not self._initial_losses_set:
            raise RuntimeError(
                "Calibrate GradNorm initial losses before model.fit()"
            )
        features, targets = data

        with tf.GradientTape() as weight_tape:
            with tf.GradientTape(persistent=True) as network_tape:
                outputs = self.network(features, training=True)
                loss_map = self._task_losses(targets, outputs)
                loss_vector = tf.stack([loss_map[name] for name in self.task_names])
                task_weights = self.normalized_task_weights()
                network_loss = tf.add_n(
                    [
                        tf.stop_gradient(task_weights[index]) * loss_map[name]
                        for index, name in enumerate(self.task_names)
                    ]
                )

            shared_variables = self.network.get_layer(
                "shared_decoder"
            ).trainable_variables
            per_task_gradients = [
                network_tape.gradient(loss_map[name], shared_variables)
                for name in self.task_names
            ]
            unweighted_gradient_norms = tf.stack(
                [
                    tf.linalg.global_norm(gradients)
                    for gradients in per_task_gradients
                ]
            )
            # Task weights are positive and independent of the shared decoder
            # variables, so ||∇_W(w_i L_i)|| = w_i ||∇_W L_i||. Reusing the
            # gradients above avoids a second reverse pass per task while
            # preserving the GradNorm objective exactly. The base norms are
            # constants for the task-weight update; only w_i is optimized by
            # ``weight_tape``.
            gradient_norms = task_weights * tf.stop_gradient(
                unweighted_gradient_norms
            )

            loss_ratios = loss_vector / tf.maximum(self.initial_losses, 1e-7)
            inverse_training_rates = loss_ratios / tf.reduce_mean(loss_ratios)
            target_norms = tf.stop_gradient(
                tf.reduce_mean(gradient_norms)
                * tf.pow(inverse_training_rates, self.alpha)
            )
            gradnorm_objective = tf.reduce_sum(
                tf.abs(gradient_norms - target_norms)
            )

        network_gradients = network_tape.gradient(
            network_loss, self.network.trainable_variables
        )
        self.optimizer.apply_gradients(
            (gradient, variable)
            for gradient, variable in zip(
                network_gradients, self.network.trainable_variables
            )
            if gradient is not None
        )
        weight_gradient = weight_tape.gradient(
            gradnorm_objective, self.log_task_weights
        )
        if weight_gradient is None:
            raise RuntimeError(
                "GradNorm objective is disconnected from task weights"
            )
        tf.debugging.assert_all_finite(weight_gradient, "Non-finite GradNorm update")
        self.weight_optimizer.apply_gradients(
            [(weight_gradient, self.log_task_weights)]
        )

        del network_tape
        current_weights = self.normalized_task_weights()
        tf.debugging.assert_all_finite(current_weights, "Non-finite task weights")
        self._update_metrics(
            targets,
            outputs,
            loss_map,
            network_loss,
            current_weights,
            gradient_norms,
            per_task_gradients,
        )
        self.gradnorm_active_tracker.update_state(1.0)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        if not self._initial_losses_set:
            raise RuntimeError(
                "Calibrate GradNorm initial losses before model.evaluate()"
            )
        features, targets = data
        outputs = self.network(features, training=False)
        loss_map = self._task_losses(targets, outputs)
        weights = self.normalized_task_weights()
        total = tf.reduce_sum(
            weights * tf.stack([loss_map[name] for name in self.task_names])
        )
        empty_gradients = [[None] for _ in self.task_names]
        self._update_metrics(
            targets,
            outputs,
            loss_map,
            total,
            weights,
            tf.zeros(len(self.task_names), dtype=tf.float32),
            empty_gradients,
            include_gradient_diagnostics=False,
        )
        self.gradnorm_active_tracker.update_state(1.0)
        return {metric.name: metric.result() for metric in self.metrics}

    def _update_metrics(
        self,
        targets: dict[str, tf.Tensor],
        outputs: dict[str, tf.Tensor],
        losses: dict[str, tf.Tensor],
        total_loss: tf.Tensor,
        weights: tf.Tensor,
        gradient_norms: tf.Tensor,
        per_task_gradients: list[list[tf.Tensor | None]],
        *,
        include_gradient_diagnostics: bool = True,
    ) -> None:
        self.total_loss_tracker.update_state(total_loss)
        self.value_loss_tracker.update_state(losses["value"])
        self.fractional_loss_tracker.update_state(losses["fractional"])
        self.value_rmse_metric.update_state(targets["value"], outputs["value"])
        if self.include_spectrum:
            self.spectrum_loss_tracker.update_state(losses["spectrum"])
            self.spectrum_rmse_tracker.update_state(losses["spectrum"])
        for index, name in enumerate(self.task_names):
            self.weight_trackers[name].update_state(weights[index])
            if include_gradient_diagnostics:
                self.gradient_norm_trackers[name].update_state(gradient_norms[index])
        if include_gradient_diagnostics:
            value_gradients = per_task_gradients[0]
            for index, name in enumerate(self.task_names[1:], start=1):
                self.cosine_trackers[f"value_{name}"].update_state(
                    self._gradient_cosine(value_gradients, per_task_gradients[index])
                )
