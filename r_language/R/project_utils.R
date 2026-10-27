# Shared R helpers. This file defines functions only; it does not execute a task.

PROJECT_LIBRARY_DIR <- local({
    file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
    if (length(file_arg) > 0) {
        dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = FALSE))
    } else {
        ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
        if (!is.null(ofile) && length(ofile) > 0 && nzchar(ofile[[1]])) {
            dirname(normalizePath(ofile[[1]], mustWork = FALSE))
        } else {
            getwd()
        }
    }
})

script_directory <- function() {
    file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
    if (length(file_arg) > 0) {
        return(dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = FALSE)))
    }

    ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
    if (!is.null(ofile) && length(ofile) > 0 && nzchar(ofile[[1]])) {
        return(dirname(normalizePath(ofile[[1]], mustWork = FALSE)))
    }

    PROJECT_LIBRARY_DIR
}

find_project_root <- function(start_dir = script_directory()) {
    current_dir <- normalizePath(start_dir, mustWork = FALSE)

    repeat {
        if (file.exists(file.path(current_dir, "data_lake", "config.json"))) {
            return(current_dir)
        }

        parent_dir <- dirname(current_dir)
        if (identical(parent_dir, current_dir)) {
            break
        }
        current_dir <- parent_dir
    }

    stop("Could not locate the repository root containing data_lake/config.json")
}

require_r_packages <- function(packages, context = "R task") {
    missing_packages <- packages[!vapply(
        packages,
        requireNamespace,
        logical(1),
        quietly = TRUE
    )]

    if (length(missing_packages) > 0) {
        stop(
            context,
            " is missing R package(s): ",
            paste(missing_packages, collapse = ", "),
            ". Run Rscript r_language/install.r in the same R installation."
        )
    }
}

sha256_file <- function(file_path) {
    if (!file.exists(file_path)) {
        stop("Cannot checksum missing file: ", file_path)
    }

    result <- system2("sha256sum", args = normalizePath(file_path), stdout = TRUE, stderr = TRUE)
    if (length(result) == 0 || !is.null(attr(result, "status"))) {
        stop("Could not calculate SHA-256 checksum for: ", file_path)
    }
    checksum <- strsplit(trimws(result[[1]]), "[[:space:]]+")[[1]][1]
    if (!grepl("^[0-9a-fA-F]{64}$", checksum)) {
        stop("Unexpected SHA-256 output for: ", file_path)
    }
    tolower(checksum)
}

repository_relative_path <- function(file_path, project_root) {
    normalized_file <- normalizePath(file_path, mustWork = FALSE)
    normalized_root <- normalizePath(project_root, mustWork = FALSE)
    prefix <- paste0(normalized_root, .Platform$file.sep)
    if (startsWith(normalized_file, prefix)) {
        substring(normalized_file, nchar(prefix) + 1)
    } else {
        normalized_file
    }
}

resolve_input_manifest <- function(project_root, manifest_folder) {
    configured_manifest <- Sys.getenv("WATER_ANALYZER_INPUT_MANIFEST", unset = "")
    if (nzchar(configured_manifest)) {
        manifest_path <- normalizePath(configured_manifest, mustWork = FALSE)
        if (!file.exists(manifest_path)) {
            stop("Configured input manifest does not exist: ", manifest_path)
        }
        return(manifest_path)
    }

    manifest_directory <- file.path(project_root, manifest_folder, "raw_ingestion")
    manifests <- sort(list.files(
        manifest_directory,
        pattern = "^raw_ingestion_.*\\.json$",
        full.names = TRUE
    ))
    if (length(manifests) == 0) {
        stop("No raw-ingestion manifest found in: ", manifest_directory)
    }
    normalizePath(manifests[[length(manifests)]], mustWork = TRUE)
}

load_lineage_context <- function(project_root, manifest_folder) {
    require_r_packages("jsonlite", "Lineage metadata")
    manifest_path <- resolve_input_manifest(project_root, manifest_folder)
    manifest <- jsonlite::fromJSON(manifest_path)
    if (is.null(manifest$run_id) || !nzchar(manifest$run_id)) {
        stop("Input manifest lacks run_id: ", manifest_path)
    }
    list(
        run_id = as.character(manifest$run_id),
        input_manifest_path = repository_relative_path(manifest_path, project_root),
        input_manifest_checksum = sha256_file(manifest_path),
        project_root = project_root
    )
}

build_output_lineage <- function(
    output_path,
    checksum_path,
    lineage_context,
    stage,
    input_files = character(),
    extra = list()
) {
    project_root <- lineage_context$project_root
    input_records <- lapply(input_files, function(input_path) {
        input_path <- normalizePath(input_path, mustWork = TRUE)
        list(
            path = repository_relative_path(input_path, project_root),
            sha256 = sha256_file(input_path)
        )
    })

    list(
        schema_version = 1L,
        stage = stage,
        created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
        run_id = lineage_context$run_id,
        input_manifest_path = lineage_context$input_manifest_path,
        input_manifest_checksum = lineage_context$input_manifest_checksum,
        output = list(
            path = repository_relative_path(output_path, project_root),
            sha256 = sha256_file(checksum_path),
            size_bytes = unname(file.info(checksum_path)$size)
        ),
        input_files = input_records,
        metadata = extra
    )
}

write_lineage_json <- function(sidecar, metadata) {
    require_r_packages("jsonlite", "Lineage metadata")
    temporary <- tempfile(
        pattern = paste0(".", basename(sidecar), "."),
        tmpdir = dirname(sidecar),
        fileext = ".tmp"
    )
    on.exit(unlink(temporary), add = TRUE)
    jsonlite::write_json(metadata, temporary, auto_unbox = TRUE, pretty = TRUE)
    if (!file.rename(temporary, sidecar)) {
        stop("Could not publish lineage metadata: ", sidecar)
    }
    sidecar
}

write_output_lineage <- function(
    output_path,
    lineage_context,
    stage,
    input_files = character(),
    extra = list()
) {
    output_path <- normalizePath(output_path, mustWork = TRUE)
    sidecar <- paste0(output_path, ".lineage.json")
    metadata <- build_output_lineage(
        output_path,
        output_path,
        lineage_context,
        stage,
        input_files,
        extra
    )
    write_lineage_json(sidecar, metadata)
}

temporary_output_path <- function(output_path) {
    output_path <- normalizePath(output_path, mustWork = FALSE)
    dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
    tempfile(
        pattern = paste0(".", basename(output_path), "."),
        tmpdir = dirname(output_path),
        fileext = ".tmp"
    )
}

publish_atomic_output <- function(
    temporary_path,
    output_path,
    lineage_context,
    stage,
    input_files = character(),
    extra = list()
) {
    temporary_path <- normalizePath(temporary_path, mustWork = TRUE)
    output_path <- normalizePath(output_path, mustWork = FALSE)
    dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

    metadata <- build_output_lineage(
        output_path,
        temporary_path,
        lineage_context,
        stage,
        input_files,
        extra
    )
    sidecar <- paste0(output_path, ".lineage.json")
    temporary_sidecar <- tempfile(
        pattern = paste0(".", basename(sidecar), "."),
        tmpdir = dirname(sidecar),
        fileext = ".tmp"
    )
    on.exit({
        unlink(temporary_path)
        unlink(temporary_sidecar)
    }, add = TRUE)
    require_r_packages("jsonlite", "Lineage metadata")
    jsonlite::write_json(metadata, temporary_sidecar, auto_unbox = TRUE, pretty = TRUE)

    if (!file.rename(temporary_path, output_path)) {
        stop("Could not publish output: ", output_path)
    }
    if (!file.rename(temporary_sidecar, sidecar)) {
        stop("Output was published but lineage metadata was not: ", sidecar)
    }
    sidecar
}

check_directory_exists <- function(dir) {
    if (!dir.exists(dir)) {
        created <- dir.create(dir, recursive = TRUE, showWarnings = FALSE)
        if (!created && !dir.exists(dir)) {
            stop("Could not create directory: ", dir)
        }
        message("Created new directory: ", dir)
    } else {
        message("Directory already exists: ", dir)
    }
    invisible(dir)
}

validate_prepared_time_series <- function(data, indicator, horizon) {
    required_columns <- c("date", indicator)
    missing_columns <- setdiff(required_columns, names(data))
    if (length(missing_columns) > 0) {
        stop(
            "Prepared time series is missing required column(s): ",
            paste(missing_columns, collapse = ", ")
        )
    }
    if (nrow(data) == 0) {
        stop("Prepared time series contains no rows")
    }

    dates <- as.Date(as.character(data[["date"]]), format = "%Y-%m-%d")
    if (anyNA(dates)) {
        stop("Prepared time series contains invalid date(s)")
    }
    if (anyDuplicated(dates) > 0 || any(diff(dates) <= 0)) {
        stop("Prepared time series dates must be unique and strictly increasing")
    }

    values <- suppressWarnings(as.numeric(as.character(data[[indicator]])))
    if (anyNA(values) || any(!is.finite(values))) {
        stop("Prepared time series contains invalid ", indicator, " value(s)")
    }
    if (nrow(data) <= horizon) {
        stop(
            "Prepared time series has ", nrow(data),
            " rows; it must contain more than the forecast horizon (", horizon, ")"
        )
    }
    invisible(data)
}

validate_mfdfa_features <- function(
    data,
    feature_columns,
    time_series_rows,
    mfdfa_window
) {
    missing_columns <- setdiff(feature_columns, names(data))
    if (length(missing_columns) > 0) {
        stop(
            "MFDFA output is missing required feature column(s): ",
            paste(missing_columns, collapse = ", ")
        )
    }
    if (time_series_rows <= mfdfa_window) {
        stop("MFDFA window must be smaller than the time-series length")
    }

    expected_rows <- time_series_rows - mfdfa_window + 1
    if (nrow(data) != expected_rows) {
        stop(
            "MFDFA output has ", nrow(data), " rows; expected ", expected_rows,
            " for ", time_series_rows, " time-series rows and window ", mfdfa_window
        )
    }

    for (column_name in feature_columns) {
        values <- suppressWarnings(as.numeric(as.character(data[[column_name]])))
        if (anyNA(values) || any(!is.finite(values))) {
            stop("MFDFA output contains invalid value(s) in ", column_name)
        }
    }

    row_indices <- suppressWarnings(as.numeric(rownames(data)))
    expected_indices <- seq(from = mfdfa_window, to = time_series_rows)
    if (
        length(row_indices) != length(expected_indices) ||
        anyNA(row_indices) ||
        !all(row_indices == expected_indices)
    ) {
        stop(
            "MFDFA output row indices are not aligned to rolling indices ",
            mfdfa_window, "..", time_series_rows
        )
    }
    invisible(data)
}

validate_spectrum_labels <- function(
    data,
    label_columns,
    time_series_rows,
    mfdfa_window
) {
    missing_columns <- setdiff(label_columns, names(data))
    if (length(missing_columns) > 0) {
        stop(
            "Spectrum-label output is missing required column(s): ",
            paste(missing_columns, collapse = ", ")
        )
    }
    expected_rows <- time_series_rows - mfdfa_window + 1
    if (nrow(data) != expected_rows) {
        stop(
            "Spectrum-label output has ", nrow(data),
            " rows; expected ", expected_rows
        )
    }

    convergence_column <- "Spectrum_Fit_Converged"
    numeric_columns <- setdiff(label_columns, convergence_column)
    for (column_name in numeric_columns) {
        values <- suppressWarnings(as.numeric(as.character(data[[column_name]])))
        if (anyNA(values) || any(!is.finite(values))) {
            stop("Spectrum-label output contains invalid value(s) in ", column_name)
        }
    }
    if (!all(as.logical(data[[convergence_column]]))) {
        stop("Spectrum-label output contains a failed spectrum fit")
    }

    row_indices <- suppressWarnings(as.numeric(rownames(data)))
    expected_indices <- seq(from = mfdfa_window, to = time_series_rows)
    if (
        length(row_indices) != length(expected_indices) ||
        anyNA(row_indices) ||
        !all(row_indices == expected_indices)
    ) {
        stop(
            "Spectrum-label output row indices are not aligned to rolling indices ",
            mfdfa_window, "..", time_series_rows
        )
    }
    invisible(data)
}

validate_forecast_output <- function(dates, values, horizon) {
    if (length(values) != horizon || length(dates) != horizon) {
        stop(
            "Forecast output has ", length(values),
            " values and ", length(dates),
            " dates; expected ", horizon, " of each"
        )
    }
    numeric_dates <- suppressWarnings(as.numeric(dates))
    numeric_values <- suppressWarnings(as.numeric(values))
    if (
        anyNA(numeric_dates) || any(!is.finite(numeric_dates)) ||
        anyNA(numeric_values) || any(!is.finite(numeric_values))
    ) {
        stop("Forecast output contains invalid date or value(s)")
    }
    if (anyDuplicated(numeric_dates) > 0 || any(diff(numeric_dates) <= 0)) {
        stop("Forecast dates must be unique and strictly increasing")
    }
    invisible(TRUE)
}

get_data <- function(file_name) {
    if (!file.exists(file_name)) {
        stop("Input data file does not exist: ", file_name)
    }

    data <- read.csv(
        file_name,
        header = TRUE,
        na.string = "?",
        encoding = "UTF-8"
    )
    if (nrow(data) == 0) {
        stop("Input data file is empty: ", file_name)
    }
    data
}

get_time_series <- function(ts_table) {
    date_arr <- as.POSIXct(ts_table$date, format = "%Y-%m-%d")
    ts_data <- xts::xts(x = ts_table$bod_mgl, order.by = date_arr)
    tsbox::ts_ts(ts_data)
}

forecasted_to_csv <- function(
    dates,
    forecasted_vals,
    csv_path,
    horizon,
    lineage_context = NULL,
    stage = "forecast_r",
    input_files = character()
) {
    validate_forecast_output(dates, forecasted_vals, horizon)
    if (is.null(lineage_context)) {
        stop("Forecast output requires lineage_context")
    }
    forecasted_df <- data.frame(date = dates, bod_mgl = forecasted_vals)
    temporary_path <- temporary_output_path(csv_path)
    on.exit(unlink(temporary_path), add = TRUE)
    write.csv(
        forecasted_df,
        temporary_path,
        row.names = FALSE,
        quote = FALSE
    )
    publish_atomic_output(
        temporary_path,
        csv_path,
        lineage_context,
        stage,
        input_files = input_files,
        extra = list(horizon = horizon)
    )
}

find_best_arfima_model <- function(
    ar_params,
    ma_params,
    train,
    test,
    horizon,
    estimated_d
) {
    min_mse <- Inf
    return_params <- c(0, 0, 0)

    for (ar in ar_params) {
        for (ma in ma_params) {
            model_arfima_test <- arfima::arfima(
                train,
                order = c(ar, ma, estimated_d),
                numeach = c(1, 1)
            )
            forecasted_arfima_test <- predict(
                model_arfima_test,
                n.ahead = horizon
            )
            forecasted_vals <- forecasted_arfima_test[[1]]$Forecast
            test_mse <- Metrics::rmse(test, forecasted_vals)

            if (min_mse > test_mse) {
                min_mse <- test_mse
                return_params <- c(ar, ma, estimated_d)
            }
        }
    }

    return_params
}
