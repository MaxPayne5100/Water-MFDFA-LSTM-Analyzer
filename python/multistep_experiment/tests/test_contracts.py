from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf

from multistep_experiment.constants import MODEL_VARIANTS
from multistep_experiment.dataset import MultiStepDataset
from multistep_experiment.experiment import (
    _best_epoch,
    prepare_split,
    report_fold_pruning,
)
from multistep_experiment.feature_selection import rank_features
from multistep_experiment.losses import (
    fractional_coefficients,
    fractional_future,
    global_rmse,
)
from multistep_experiment.comparison import summarize_runs
from multistep_experiment.models import ModelParameters, build_model
from multistep_experiment.validation import purged_walk_forward_splits


class FakeTrial:
    def __init__(self, *, prune: bool = False) -> None:
        self.prune = prune
        self.reports: list[tuple[float, int]] = []

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return self.prune


class LossContractsTest(unittest.TestCase):
    def test_global_rmse_has_equal_point_weight(self) -> None:
        actual = tf.zeros((2, 3, 1), dtype=tf.float32)
        predicted = tf.constant([[[0.0], [0.0], [0.0]], [[1.0], [1.0], [1.0]]])
        self.assertAlmostEqual(float(global_rmse(actual, predicted)), np.sqrt(0.5), places=6)

    def test_fractional_future_uses_pre_origin_context(self) -> None:
        context = tf.constant([[[2.0]]])
        future = tf.constant([[[3.0], [5.0]]])
        result = fractional_future(context, future, tf.constant([1.0, -0.5]))
        np.testing.assert_allclose(result.numpy().reshape(-1), [2.0, 3.5])

    def test_truncation_respects_available_history(self) -> None:
        coefficients = fractional_coefficients(0.3, available_context=4)
        self.assertLessEqual(len(coefficients) - 1, 4)


class ValidationContractsTest(unittest.TestCase):
    def test_purge_removes_overlapping_training_targets(self) -> None:
        starts = np.arange(30)
        index = pd.DataFrame({"target_start": starts, "target_end": starts + 9})
        folds = purged_walk_forward_splits(index, np.arange(30), n_splits=3)
        self.assertTrue(folds)
        for fold in folds:
            self.assertTrue(
                np.all(index.iloc[fold.train_positions]["target_end"] < fold.validation_target_start)
            )

    def test_long_horizon_uses_the_same_generic_purge_contract(self) -> None:
        starts = np.arange(600)
        index = pd.DataFrame({"target_start": starts, "target_end": starts + 99})
        folds = purged_walk_forward_splits(index, np.arange(600), n_splits=3)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertLess(
                int(index.iloc[fold.train_positions]["target_end"].max()),
                fold.validation_target_start,
            )

    def test_minimum_training_origins_are_enforced(self) -> None:
        starts = np.arange(200)
        index = pd.DataFrame({"target_start": starts, "target_end": starts + 49})
        folds = purged_walk_forward_splits(
            index, np.arange(200), n_splits=3, min_train_origins=50
        )
        self.assertEqual(len(folds), 3)
        self.assertTrue(all(len(fold.train_positions) >= 50 for fold in folds))


class FeatureSelectionContractsTest(unittest.TestCase):
    def test_feature_ranking_purges_overlapping_internal_targets(self) -> None:
        rng = np.random.default_rng(42)
        starts = np.arange(80)
        ranking = rank_features(
            rng.normal(size=(80, 3, 4)),
            rng.normal(size=(80, 10, 1)),
            ("a", "b", "c", "d"),
            seed=42,
            target_starts=starts,
            target_ends=starts + 9,
            repeats=2,
        )
        self.assertEqual(ranking["ranking_method"].iloc[0], "purged_permutation")
        # Preferred split is 64; purging its overlapping 9-step target tails
        # leaves starts 0..54 for fitting and starts 64..79 for validation.
        self.assertEqual(int(ranking["ranking_train_samples"].iloc[0]), 55)
        self.assertEqual(int(ranking["ranking_validation_samples"].iloc[0]), 16)

    def test_feature_ranking_uses_impurity_when_no_purged_split_exists(self) -> None:
        rng = np.random.default_rng(42)
        starts = np.arange(50)
        ranking = rank_features(
            rng.normal(size=(50, 3, 4)),
            rng.normal(size=(50, 50, 1)),
            ("a", "b", "c", "d"),
            seed=42,
            target_starts=starts,
            target_ends=starts + 49,
            repeats=2,
        )
        self.assertEqual(ranking["ranking_method"].iloc[0], "in_sample_impurity")
        self.assertEqual(int(ranking["ranking_validation_samples"].iloc[0]), 0)

class ModelContractsTest(unittest.TestCase):
    def test_all_variants_have_direct_horizon_output(self) -> None:
        parameters = ModelParameters(lookback=4, top_k=3)
        coefficients = np.asarray([1.0, -0.2], dtype=np.float32)
        for horizon in (10, 50):
            for variant in MODEL_VARIANTS:
                model = build_model(
                    variant,
                    lookback=4,
                    n_features=3,
                    horizon=horizon,
                    parameters=parameters,
                    fractional_coefficients=(
                        coefficients if variant != MODEL_VARIANTS[0] else None
                    ),
                )
                output = model(tf.zeros((2, 4, 3), dtype=tf.float32), training=False)
                self.assertEqual(tuple(output["value"].shape), (2, horizon, 1))
                if variant == MODEL_VARIANTS[2]:
                    self.assertEqual(tuple(output["spectrum"].shape), (2, horizon, 9))

    def test_spectrum_gradnorm_is_active_after_initial_loss_calibration(self) -> None:
        """Calibration must precede immediate, effective GradNorm updates."""
        parameters = ModelParameters(lookback=3, batch_size=4)
        model = build_model(
            MODEL_VARIANTS[2],
            lookback=3,
            n_features=4,
            horizon=5,
            parameters=parameters,
            fractional_coefficients=np.asarray([1.0, -0.2], dtype=np.float32),
        )
        rng = np.random.default_rng(42)
        features = rng.normal(size=(4, 3, 4)).astype(np.float32)
        targets = {
            "value": rng.normal(size=(4, 5, 1)).astype(np.float32),
            "context": np.zeros((4, 1, 1), dtype=np.float32),
            "spectrum": rng.normal(size=(4, 5, 9)).astype(np.float32),
            "spectrum_mask": np.ones((4, 5, 9), dtype=np.float32),
        }

        initial_weights = model.normalized_task_weights().numpy().copy()
        calibrated = model.calibrate_initial_losses(features, targets)
        self.assertEqual(int(model.optimizer.iterations.numpy()), 0)
        self.assertEqual(set(calibrated), {"value", "fractional", "spectrum"})
        self.assertTrue(all(value > 0 for value in calibrated.values()))
        history = model.fit(
            features,
            targets,
            validation_data=(features, targets),
            epochs=3,
            batch_size=4,
            shuffle=False,
            verbose=0,
        )

        self.assertEqual(len(history.epoch), 3)
        self.assertEqual(history.history["gradnorm_active"], [1.0, 1.0, 1.0])
        self.assertEqual(
            history.history["val_gradnorm_active"],
            [1.0, 1.0, 1.0],
        )
        self.assertGreaterEqual(
            _best_epoch(history, require_gradnorm_activation=True),
            1,
        )
        for task_name in ("value", "fractional", "spectrum"):
            gradient_norms = np.asarray(
                history.history[f"{task_name}_gradient_norm"], dtype=float
            )
            self.assertTrue(np.isfinite(gradient_norms).all())
            self.assertGreater(float(gradient_norms.max()), 0.0)
        final_weights = model.normalized_task_weights().numpy()
        self.assertFalse(np.allclose(final_weights, initial_weights))
        self.assertAlmostEqual(float(final_weights.sum()), 3.0, places=5)

    def test_best_epoch_ignores_inactive_gradnorm_epochs(self) -> None:
        history = tf.keras.callbacks.History()
        history.epoch = [0, 1, 2, 3]
        history.history = {
            "val_value_rmse": [0.1, 0.5, 0.4, 0.45],
            "val_gradnorm_active": [0.0, 1.0, 1.0, 1.0],
        }
        self.assertEqual(
            _best_epoch(history, require_gradnorm_activation=True),
            3,
        )


class TuningContractsTest(unittest.TestCase):
    def test_cached_feature_ranking_bypasses_random_forest(self) -> None:
        rng = np.random.default_rng(42)
        feature_names = ("a", "b", "c")
        sample_index = pd.DataFrame(
            {
                "target_start": np.arange(10, 18),
                "target_end": np.arange(11, 19),
            }
        )
        dataset = MultiStepDataset(
            features=rng.normal(size=(8, 2, 3)).astype(np.float32),
            values=rng.normal(size=(8, 2, 1)).astype(np.float32),
            spectra=np.zeros((8, 2, 9), dtype=np.float32),
            spectrum_mask=np.zeros((8, 2, 9), dtype=np.float32),
            sample_index=sample_index,
            raw_values=rng.normal(size=30).astype(np.float32),
            raw_dates=pd.date_range("2020-01-01", periods=30),
            feature_columns=feature_names,
            horizon=2,
            lookback=2,
            evaluation_start=20,
            evaluation_origins=np.asarray([20]),
        )
        ranking = pd.DataFrame(
            {"feature": ["b", "a", "c"], "importance": [0.5, 0.3, 0.2]}
        )
        with patch(
            "multistep_experiment.experiment.rank_features",
            side_effect=AssertionError("ranking should come from cache"),
        ):
            prepared = prepare_split(
                dataset,
                training_positions=np.arange(5),
                evaluation_positions=np.arange(5, 8),
                parameters=ModelParameters(lookback=2, top_k=2),
                variant="value",
                seed=42,
                feature_ranking=ranking,
            )
        self.assertEqual(prepared.selected_features, ["b", "a"])
        self.assertEqual(prepared.train_x.shape[-1], 2)

    def test_pruning_reports_completed_fold_as_resource(self) -> None:
        trial = FakeTrial()
        value = report_fold_pruning(
            trial,
            fold_position=2,
            completed_metric_rows=[{"value_rmse": 1.0}, {"value_rmse": 0.8}],
        )
        self.assertAlmostEqual(value, 0.9)
        self.assertEqual(trial.reports, [(0.9, 2)])

    def test_fold_pruning_raises_trial_pruned(self) -> None:
        with self.assertRaises(optuna.TrialPruned):
            report_fold_pruning(
                FakeTrial(prune=True),
                fold_position=1,
                completed_metric_rows=[{"value_rmse": 1.25}],
            )


class ComparisonContractsTest(unittest.TestCase):
    def test_comparison_is_paired_by_seed(self) -> None:
        runs = pd.DataFrame(
            {
                "variant": ["value"] * 3 + ["fractional"] * 3 + ["spectrum"] * 2,
                "seed": [1, 2, 3, 1, 2, 4, 1, 3],
                "value_rmse": [1.0, 1.2, 1.1, 0.9, 1.3, 0.8, 1.1, 1.0],
            }
        )
        _, paired = summarize_runs(runs)
        counts = dict(zip(paired["candidate"], paired["paired_seeds"]))
        self.assertEqual(
            counts,
            {"value_fractional": 2, "value_fractional_spectrum": 2},
        )

    def test_comparison_pairs_every_seed_and_rolling_origin(self) -> None:
        runs = pd.DataFrame(
            {
                "variant": ["value", "fractional", "spectrum"],
                "seed": [7, 7, 7],
                "origins": [2, 2, 2],
                "value_rmse": [1.0, 0.9, 0.8],
            }
        )
        origins = pd.DataFrame(
            {
                "variant": ["value"] * 2 + ["fractional"] * 2 + ["spectrum"] * 2,
                "seed": [7] * 6,
                "forecast_origin": [100, 110] * 3,
                "value_rmse": [1.0, 1.2, 0.9, 1.1, 0.8, 1.0],
            }
        )
        _, paired = summarize_runs(runs, origins)
        self.assertEqual(paired["paired_origins"].tolist(), [2, 2])
        self.assertEqual(paired["paired_seed_origins"].tolist(), [2, 2])
        self.assertNotIn("exact_sign_flip_pvalue", paired.columns)


if __name__ == "__main__":
    unittest.main()
