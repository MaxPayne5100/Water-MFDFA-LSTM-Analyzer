"""Command-line interface with an explicit guard around expensive operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import MODEL_VARIANTS, SEEDS
from .comparison import compare_and_publish
from .experiment import run_matrix, run_variant, tune_baseline, validate_design


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "data_lake" / "config.json"


def _training_guard(arguments: argparse.Namespace) -> None:
    if not arguments.confirm_training:
        raise SystemExit(
            "Training was not started. Re-run this explicit training command with "
            "--confirm-training after reviewing the configuration."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated direct multi-step RMSE/GradNorm experiment"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment-id", default="multistep_fractal_ablation_v2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="Validate indexing and fold contracts without training"
    )
    validate.add_argument("--lookback", type=int, default=6)

    tune = subparsers.add_parser(
        "tune-baseline", help="Tune the value-only baseline and lock its settings"
    )
    tune.add_argument("--n-trials", type=int, default=40)
    tune.add_argument("--seed", type=int, default=42)
    tune.add_argument("--confirm-training", action="store_true")

    run = subparsers.add_parser(
        "run-variant", help="Run one frozen variant/seed experiment cell"
    )
    run.add_argument("--variant", choices=MODEL_VARIANTS, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--d-bootstrap-repeats", type=int, default=0)
    run.add_argument("--confirm-training", action="store_true")

    matrix = subparsers.add_parser(
        "run-matrix", help="Run all requested variants and seeds"
    )
    matrix.add_argument(
        "--variants", nargs="+", choices=MODEL_VARIANTS, default=list(MODEL_VARIANTS)
    )
    matrix.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    matrix.add_argument("--d-bootstrap-repeats", type=int, default=0)
    matrix.add_argument("--confirm-training", action="store_true")
    subparsers.add_parser(
        "compare", help="Publish descriptive paired seed-origin summaries; no training"
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "validate":
        result = validate_design(arguments.config, lookback=arguments.lookback)
        print(json.dumps(result, indent=2))
        return

    if arguments.command == "compare":
        result = compare_and_publish(
            config_path=arguments.config,
            experiment_id=arguments.experiment_id,
        )
        print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
        return

    _training_guard(arguments)
    if arguments.command == "tune-baseline":
        result = tune_baseline(
            config_path=arguments.config,
            experiment_id=arguments.experiment_id,
            n_trials=arguments.n_trials,
            seed=arguments.seed,
        )
        print(json.dumps(result, indent=2))
    elif arguments.command == "run-variant":
        result = run_variant(
            config_path=arguments.config,
            experiment_id=arguments.experiment_id,
            variant=arguments.variant,
            seed=arguments.seed,
            d_bootstrap_repeats=arguments.d_bootstrap_repeats,
        )
        print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
    elif arguments.command == "run-matrix":
        results = run_matrix(
            config_path=arguments.config,
            experiment_id=arguments.experiment_id,
            variants=arguments.variants,
            seeds=arguments.seeds,
            d_bootstrap_repeats=arguments.d_bootstrap_repeats,
        )
        print(json.dumps([{key: str(value) for key, value in item.items()} for item in results], indent=2))


if __name__ == "__main__":
    main()
