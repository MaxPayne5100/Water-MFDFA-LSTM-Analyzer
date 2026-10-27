# Water Analyzer Ukraine

Pipeline for Ukrainian surface-water monitoring data. The project combines
Python data preparation and forecasting with R time-series and multifractal
analysis.

The default configuration analyzes Dnipro `BSK5` data with a forecast horizon
of 10 observations and an MFDFA rolling window of 150 observations.

## Pipeline

```text
download immutable raw data
        ↓
validate raw files and ingestion manifest
        ↓
combine and filter monitoring data
        ↓
validate bronze tables
        ↓
prepare the curated time series
        ↓
R rolling-window MFDFA
        ↓
validate aligned features
        ↓
Python LSTM/Optuna forecasting
        ↓
validate and publish forecast artifacts
```

ARFIMA/ARIMA forecasting is available as an optional R forecasting branch.

## Repository layout

```text
python/
  extract_river_data.py              Download raw resources
  raw_ingestion.py                   Immutable ingestion and manifests
  combine_filter_river_data.py      Build bronze tables
  prepare_visualise_river_data.py   Build curated time series and charts
  data_quality.py                   Cross-stage validation CLI
  forecast_artifacts.py             Forecast CSV, metadata, plot publisher
  forecast_river_data_optuna.py     LSTM/Optuna forecasting
  requirements.txt                   Application dependencies
  requirements-airflow.txt          Airflow 3 dependencies and constraints

r_language/
  multifractal_analysis_rolling_window.r
  arfima_forecast_river_data.r
  R/                                Shared R libraries
  install.r                         R dependency installer

data_lake/
  raw/source_data/                  Immutable downloaded source files
  bronze/combined/                  Combined and filtered CSV tables
  curated/time_series/              Validated river/indicator series
  features/mfdfa/                   R-generated MFDFA features and spectrum labels
  forecasts/                        Forecast CSVs and artifacts
  artifacts/analysis/               Analysis charts
  manifests/raw_ingestion/          Append-only ingestion manifests
  *.lineage.json                    Output lineage sidecars
  config.json                       Pipeline configuration
```

## Linux setup

```bash
sudo apt install python3 python3-venv r-base

python3 -m venv .venv-linux
source .venv-linux/bin/activate
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt

# Install required R packages into the current user's R library.
Rscript r_language/install.r
```

The R pipeline requires packages including `jsonlite`, `MFDFA`, `zoo`,
`moments`, `forecast`, `arfima`, `fracdiff`, and `Metrics`.

Airflow has a separate environment so its scheduler dependencies do not alter
the application environment:

```bash
python3 -m venv .venv-airflow
source .venv-airflow/bin/activate
python -m pip install --upgrade pip
python -m pip install -r python/requirements-airflow.txt
python -m pip check
```

`requirements-airflow.txt` pins Airflow 3.3.1, the standard provider used by
the DAG, and Apache's Python 3.12 constraint set for the complete Airflow
dependency graph. Airflow tasks should invoke the application explicitly with
`.venv-linux/bin/python` and invoke R with `Rscript`.

## Running the pipeline manually

Run commands from the repository root:

```bash
source .venv-linux/bin/activate

# 1. Download resources without overwriting existing raw files.
python python/extract_river_data.py

# 2. Build full and filtered bronze tables.
python python/combine_filter_river_data.py

# 3. Prepare the configured river/indicator time series and charts.
python python/prepare_visualise_river_data.py

# 4. Generate rolling MFDFA features.
Rscript r_language/multifractal_analysis_rolling_window.r

# 5. Run the Python LSTM/Optuna forecast.
python python/forecast_river_data_optuna.py
```

The Python forecast stage publishes the following files under
`data_lake/forecasts/`:

- `Forecasted_<river>_<indicator>_<horizon>_days_LSTM.csv` with the forecast,
  holdout actual value, and residual for each forecast date;
- the matching `.json` metadata file with the run ID, input-manifest checksum,
  selected features, Optuna parameters, and evaluation metrics;
- the matching `.png` diagnostic plot;
- a `.lineage.json` sidecar for each published artifact.

This forecast is a final-horizon holdout evaluation of the configured data,
not an out-of-sample forecast beyond the last observed date. The metadata
contains `forecast_kind: "holdout_evaluation"` to make that distinction
explicit.

The optional R forecasting branch is:

```bash
Rscript r_language/arfima_forecast_river_data.r
```

## Data-quality checks

The validator raises an exception and exits nonzero when a contract fails,
which allows Airflow to mark the task as failed.

```bash
for stage in raw combined filtered timeseries features forecast forecast-python; do
  python python/data_quality.py \
    --stage "$stage" \
    --config data_lake/config.json
done
```

Checks include required columns, non-empty files, valid dates, sorted and
unique time-series dates, finite numeric values, forecast length, MFDFA row
counts, and rolling-index alignment.

## Immutable raw ingestion

Raw downloads are written to a temporary file first and then published only
to a new destination. Existing raw files are never overwritten. If a remote
resource changes, a new versioned filename is created.

Each ingestion creates an append-only JSON manifest in
`data_lake/manifests/raw_ingestion/`. Records contain:

- source URL;
- resource ID;
- original filename;
- stored filename;
- UTC ingestion timestamp;
- SHA-256 checksum;
- file size and ingestion status.

The manifest also records the active raw snapshot and the
`active_source_snapshot_checksum`. Bronze processing uses only active records
whose stored filename matches `RAW_SOURCE_FILE_PREFIXES`; older versions and
dataset structure files remain available for provenance but are not combined
into observation tables.

Older local files whose source metadata is no longer available are marked
`legacy_unmatched` rather than being assigned invented provenance.

## Configuration

Edit [data_lake/config.json](data_lake/config.json) to change:

- source dataset and API URL;
- target river and indicator;
- row and column filters;
- forecast horizon;
- rolling evaluation-origin count and stride;
- minimum number of fully purged training origins per validation fold;
- MFDFA window;
- selected source columns;
- analytical raw-file prefixes used to select observation resources;
- data-lake directories.

Python and R resolve these paths from the repository/configuration location;
Windows-specific absolute paths are not required.

## Airflow integration

There is currently no DAG committed in this repository. The intended Airflow
task order is:

```text
extract_raw
  >> validate_raw
  >> detect_analytical_snapshot_update
  >> [skip all downstream tasks when unchanged]
  >> build_bronze
  >> validate_bronze
  >> prepare_curated
  >> run_mfdfa
  >> validate_features
  >> forecast
  >> validate_forecast
  >> mark_snapshot_processed
```

Pass manifest paths and output paths between tasks, not DataFrames or large
datasets through XCom. Workers must share the data-lake volume or use a shared
object store.

Use classic Airflow 3 operators (no decorators). Have `extract_raw` return the
created manifest path and template it into downstream task environments:

```python
env={
    "WATER_ANALYZER_INPUT_MANIFEST":
        "{{ ti.xcom_pull(task_ids='extract_raw') }}"
}
```

This prevents a downstream task from accidentally selecting a newer manifest
when another ingestion run is present.

The manifest stores both an all-resource checksum and an
`active_source_snapshot_checksum` for the configured monitoring prefixes. The
update gate compares the latter with
`data_lake/manifests/processed/last_successful_snapshot.json`. That marker is
written only after the final forecast validation succeeds; therefore a failed
run is retried on the next schedule, while an unchanged successful snapshot
does not retrain the model.

Derived data is not automatically rebuilt when new raw data arrives. After a
successful new ingestion, rerun the downstream bronze, curated, feature, and
forecast stages.

Every downstream output has a `<output>.lineage.json` sidecar. It records the
raw-ingestion `run_id`, the SHA-256 checksum of the input manifest, checksums of
the direct input files, and the output checksum. Airflow should pass the exact
raw manifest to downstream tasks through `WATER_ANALYZER_INPUT_MANIFEST`; local
CLI execution falls back to the newest raw-ingestion manifest. The data-quality
commands reject outputs whose lineage does not match the selected raw run.

Derived files are published failure-safely: each task writes to a temporary
file in the destination directory, validates it, prepares lineage metadata,
and then atomically replaces the previous output. A failed task therefore
leaves the previous published artifact intact instead of exposing a partial
CSV, image, feature file, or forecast.

## Validation and development checks

```bash
python -m py_compile python/*.py

Rscript -e 'files <- list.files("r_language", pattern="\\.r$", full.names=TRUE, recursive=TRUE); for (f in files) parse(f)'

python python/data_quality.py \
  --stage raw \
  --config data_lake/config.json
```

Notebooks are under `python/` and should be opened after running the relevant
pipeline stages so their outputs correspond to the current data lake.

## Isolated multi-step fractal experiment

The direct horizon-10/horizon-50 ablation implementation is isolated in
[`python/multistep_experiment`](python/multistep_experiment/README.md). It
compares the same encoder-decoder LSTM under `value_only` (value RMSE),
`value_fractional` (value plus fractional-difference RMSE with GradNorm), and
`value_fractional_spectrum` (value plus fractional-difference plus spectrum
RMSE with GradNorm). These IDs name the complete objective bundle.
The existing `forecast_river_data_optuna.py` baseline is not imported or
modified by this experiment.

The current long-horizon configuration forecasts 50 monitoring observations at
five rolling origins. A single model is fitted before the first evaluation
origin and evaluated without refitting. Comparison artifacts include per-origin
RMSE and descriptive candidate-minus-baseline effects matched by seed and
origin; overlapping forecast windows are not treated as independent evidence.

Its safe validation command performs no training:

```bash
export PYTHONPATH="$PWD/python"
python -m multistep_experiment --config data_lake/config.json validate
```

All tuning and fitting CLI commands require `--confirm-training`. A guarded,
cell-by-cell driver is available at
[`python/forecast_river_data_multistep_experiment.ipynb`](python/forecast_river_data_multistep_experiment.ipynb).
