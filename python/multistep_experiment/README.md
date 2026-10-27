# Direct multi-step fractal ablation

This package is an isolated experiment. It does not modify or execute
`forecast_river_data_optuna.py`, which remains the single-step baseline.

The experiment compares, within one configured horizon:

1. `value_only`: direct LSTM trained with global value RMSE.
2. `value_fractional`: the same LSTM trained with value RMSE and
   fractional-difference RMSE, dynamically balanced by GradNorm.
3. `value_fractional_spectrum`: the same LSTM trained with value,
   fractional-difference, and standardized singularity-spectrum RMSE tasks,
   dynamically balanced by GradNorm.

Variant IDs name the complete objective bundle. The `spectrum` output head,
labels, and loss task retain their component name. The comparison step can read
historical `model_value`, `model_fractional`, and `model_spectrum` folders and
reports them under the canonical variant IDs.

All variants use the same encoder-decoder architecture, fold-local feature
selection and scaling, purged walk-forward folds, seeds, and baseline-tuned
hyperparameters. The configured `HORIZON` controls the direct output width.
`EVALUATION_ORIGIN_COUNT` and `EVALUATION_ORIGIN_STRIDE` define a final rolling
evaluation panel. The model is fitted once using only targets ending before the
first evaluation origin and is not refitted inside that panel.

## Environment

From the repository root:

```bash
source .venv-linux/bin/activate
pip install -r python/requirements-experiment.txt
export PYTHONPATH="$PWD/python"
```

## Safe validation (no training)

```bash
python -m multistep_experiment --config data_lake/config.json validate
```

This loads and validates the aligned tables and purged fold boundaries. It does
not perform Random Forest selection, Optuna, model fitting, or prediction.

## Scientific run sequence

The current long-horizon design uses 50 observations, five rolling origins
spaced by 10 observations, and at least 50 fully purged training origins per
validation fold. Regenerate the R feature outputs after changing the corrected
spectrum-asymmetry formula, then tune the value-only baseline:

```bash
python -m multistep_experiment --config data_lake/config.json \
  --experiment-id multistep_fractal_ablation_v2 \
  tune-baseline --n-trials 40 --seed 42 --confirm-training
```

The resulting parameter lock is reused unchanged by every model variant. Run
the matrix only after reviewing that lock:

```bash
python -m multistep_experiment --config data_lake/config.json \
  --experiment-id multistep_fractal_ablation_v2 \
  run-matrix --confirm-training
```

Within one `run_matrix` call, expensive preprocessing is cached without
crossing fold boundaries. Feature rankings are reused across variants only for
the same seed and training split. With fractional bootstrap diagnostics
disabled (the default), each training split's Local Whittle/GPH estimate and
finite kernel are deterministic and reused across both auxiliary variants and
all seeds. If bootstrap repeats are enabled, those cached results remain
seed-specific. The cache is memory-only and ends with the matrix call, so it
cannot silently reuse results after inputs or configuration change.

To switch between the verified short- and long-horizon designs, change only
`HORIZON` between 10 and 50 in `config.json`, restart the notebook kernel, and
run from the top. Unchanged curated/R inputs do not need regeneration. The same
experiment ID may be retained because artifacts and parameter locks are placed
in separate `horizon_10` and `horizon_50` folders. Parameters must still be
tuned and locked independently per horizon, then shared across variants within
that horizon. Never compare the numeric RMSE of horizon 10 directly with horizon
50; model comparisons are made within a horizon on identical seeds and origins.

After a complete matrix finishes, publish run-level summaries, per-origin
metrics, and candidate-minus-baseline deltas matched by seed and origin:

```bash
python -m multistep_experiment --config data_lake/config.json \
  --experiment-id multistep_fractal_ablation_v2 compare
```

The rolling 50-step windows overlap, so seed intervals and paired origin effects
are descriptive. The comparison intentionally does not publish an
independence-based p-value. Report run-level and origin-level effect sizes.

With the current 410 observations and lookback 6, validation has 6, 6, and 5
rolling origins after enforcing 50 purged training origins. This is feasible but
still a constrained long-horizon experiment; the five final evaluation origins
and five neural-network seeds do not become independent replicates merely by
being written as separate rows.

Training commands refuse to run unless `--confirm-training` is supplied.

During baseline tuning, the aligned dataset, purged folds, and complete
fold-local feature rankings are cached in memory for each sampled lookback.
Trials with the same lookback therefore reuse the same leakage-safe ranking;
`top_k` only slices that cached ordering. Ranking uses purged permutation
importance when an internal non-overlapping split retains at least 10 training
and 5 validation samples. If a short outer fold cannot support those minimums,
it uses in-sample Random Forest impurity importance without exposing overlapping
validation labels; the method and sample counts are recorded in fold metrics and
final metadata. A median pruner receives one
comparable resource step after each completed purged fold, using running
original-scale value RMSE. It can stop weak trials before later folds but does
not compare epoch numbers from independently restarted fold models. The lock
records completed/pruned counts and the exact pruning configuration.

## Method contracts

- An MFDFA feature row indexed `j` is available through raw observation `j-1`
  and predicts raw observation `j`; assertions enforce this strict ordering.
- Every training target interval must end before its validation interval starts.
- Every validation fold contains at least `MIN_TRAIN_ORIGINS` fully purged
  training origins.
- Final inference uses all configured rolling origins. Inputs observed between
  origins may be used at later origins, but model weights and fitted
  preprocessing remain fixed after the first origin.
- Feature ranking, scalers, Local Whittle/GPH estimates, and fractional kernel
  truncation are fitted from each training fold only. Any internal permutation
  ranking split is purged by target interval.
- Local Whittle is the primary `d` estimate and GPH is a recorded robustness
  diagnostic. Optional moving-block bootstrap repeats can be requested from the
  run commands, but are disabled by default.
- The fractional kernel is truncated by cumulative coefficient energy, bounded
  only by available history and the global safety maximum.
- Fractional loss receives standardized observations preceding each forecast
  origin, so the convolution has boundary context without validation leakage.
- Value RMSE is one global RMSE over every sample and every horizon point.
  Spectrum RMSE is global over valid standardized labels only.
- GradNorm calibrates initial task losses on the training split before any
  optimizer update and learns positive normalized weights from the first
  training step. No fixed auxiliary loss coefficient is used.
- Each task loss is differentiated against the shared decoder once per batch.
  Its exact weighted norm is the positive task weight multiplied by the
  unweighted norm, and the same gradients supply cosine diagnostics. This
  avoids duplicate reverse passes without changing the GradNorm objective.

Outputs are isolated under:

```text
data_lake/forecasts/experiments/<experiment_id>/horizon_<H>/
  locked_parameters.json
  optuna_trials.csv
  comparison_runs.csv
  comparison_origins.csv
  comparison_summary.csv
  comparison_paired.csv
  model_<variant>/seed_<seed>/
    forecast.csv
    fold_metrics.csv
    training_history.csv
    metadata.json
    model.keras
```

Each `forecast.csv` contains `forecast_origin`, `forecast_origin_date`, target
date, horizon step, actual value, and forecast value for every rolling origin.
Every published file has the repository's lineage sidecar containing the run
ID and input-manifest checksum. Model files contain the inference network; the
fractional kernel, fitted scaler parameters, selected features, estimated `d`,
and final GradNorm weights are recorded in `metadata.json`.

Elapsed learning time is mandatory for new experiment artifacts. Each fold's
duration and trained epoch count are stored in `fold_metrics.csv`.
`metadata.json` stores cross-validation, final-fit, and total learning seconds
plus total epochs. The monotonic wall-clock measurement includes model build,
GradNorm initial-loss calibration, and `model.fit`; it excludes preprocessing,
prediction, and artifact publication. The comparison tables expose these
durations so runtime is compared across models from the same matrix execution.
