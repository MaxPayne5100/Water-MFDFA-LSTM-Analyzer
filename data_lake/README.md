# Data lake layout

The local data lake is separated by processing stage:

- `raw/source_data/` — downloaded source files; treated as input data.
- `bronze/combined/` — combined and filtered monitoring tables.
- `curated/time_series/` — validated river/indicator time series.
- `features/mfdfa/` — R-generated rolling MFDFA features and separate
  singularity-spectrum labels (`SpectrumLabels_*`). The spectrum labels contain
  normalized-alpha NLS parameters and fit diagnostics; neural-network target
  standardization is intentionally deferred to the fold-aware Python stage.
- `forecasts/` — forecast CSVs and forecast artifacts.
- `artifacts/analysis/` — charts and exploratory analysis outputs.
- `manifests/raw_ingestion/` — append-only raw-ingestion manifests containing
  resource IDs, source URLs, timestamps, original filenames, stored filenames,
  SHA-256 checksums, and ingestion status.

Paths are configured in `config.json`. Pipeline tasks should publish outputs
only to the folder for their stage and pass file paths or manifest metadata to
the next task.

Downstream files publish a sibling `<filename>.lineage.json` sidecar. The
sidecar contains the raw-ingestion run ID, input-manifest SHA-256 checksum,
direct input checksums, and output checksum. Set
`WATER_ANALYZER_INPUT_MANIFEST` when running a downstream task from Airflow so
all tasks use the same immutable raw manifest.

Derived outputs are written to same-directory temporary files and published
with atomic renames only after validation. Failed processing does not truncate
or replace the previous published artifact.
