# Shared time-series context loader. This file defines functions only.

load_time_series_context <- function(project_root = find_project_root()) {
    require_r_packages(c("jsonlite", "MFDFA"), "Time-series MFDFA task")

    config_path <- file.path(project_root, "data_lake", "config.json")
    config <- jsonlite::fromJSON(config_path)

    curated_folder <- file.path(project_root, config$CURATED_FOLDER)
    features_folder <- file.path(project_root, config$FEATURES_FOLDER)
    forecasting_folder <- file.path(project_root, config$FORECASTING_FOLDER)
    indicator <- config$INDICATOR
    horizon <- config$HORIZON
    river_name <- config$RIVER_NAME_VALUE_ENG
    mfdfa_window <- config$MFDFA_WINDOW
    lineage_context <- load_lineage_context(project_root, config$MANIFEST_FOLDER)

    check_directory_exists(curated_folder)
    check_directory_exists(features_folder)
    check_directory_exists(forecasting_folder)

    file_to_process <- paste0(river_name, "_", indicator, "_ts.csv")
    prepared_data_file <- file.path(curated_folder, file_to_process)
    indicator_data <- get_data(prepared_data_file)

    if (ncol(indicator_data) < 2) {
        stop("Prepared time-series data must contain date and indicator columns")
    }

    validate_prepared_time_series(indicator_data, indicator, horizon)

    ts_data <- ts(indicator_data[[indicator]], frequency = 1)

    train <- head(ts_data, -horizon)
    test <- tail(ts_data, horizon)

    min_scale <- 10
    max_scale <- ceiling(length(train) / 100) * 10
    scale <- seq(min_scale, max_scale, by = 10)
    q <- seq(-10, 10, by = 1)
    mfdfa_result <- MFDFA::MFDFA(train, scale = scale, q = q)

    Hq <- mfdfa_result$Hq
    tau_q <- mfdfa_result$tau_q
    spec <- mfdfa_result$spec
    hurst_exponent <- mfdfa_result$Hq[q == 2]
    estimated_d <- hurst_exponent - 0.5
    delta_H <- Hq[q == min(q)] - Hq[q == max(q)]
    alpha <- spec$hq
    f_alpha <- spec$Dq
    delta_alpha <- max(alpha) - min(alpha)
    max_alpha_index <- which.max(alpha)
    min_alpha_index <- which.min(alpha)
    delta_f_alpha <- f_alpha[min_alpha_index] - f_alpha[max_alpha_index]
    alpha_0_index <- which.max(f_alpha)
    alpha_0 <- alpha[alpha_0_index]
    # Use geometric boundaries instead of relying on the order returned by
    # MFDFA. With ascending q, alpha is commonly returned in descending order.
    alpha_L <- min(alpha)
    alpha_R <- max(alpha)
    alpha_width <- alpha_R - alpha_L
    if (!is.finite(alpha_width) || alpha_width <= 0) {
        stop("MFDFA returned a singularity spectrum with non-positive width")
    }
    delta_alpha_L <- alpha_0 - alpha_L
    delta_alpha_R <- alpha_R - alpha_0
    B <- (delta_alpha_L - delta_alpha_R) / alpha_width

    list(
        PROJECT_ROOT = project_root,
        lineage_context = lineage_context,
        run_id = lineage_context$run_id,
        input_manifest_path = lineage_context$input_manifest_path,
        input_manifest_checksum = lineage_context$input_manifest_checksum,
        config = config,
        curated_folder = curated_folder,
        features_folder = features_folder,
        forecasting_folder = forecasting_folder,
        indicator = indicator,
        horizon = horizon,
        river_name = river_name,
        mfdfa_window = mfdfa_window,
        prepared_data_file = prepared_data_file,
        indicator_data = indicator_data,
        ts_data = ts_data,
        train = train,
        test = test,
        mfdfa_result = mfdfa_result,
        Hq = Hq,
        tau_q = tau_q,
        spec = spec,
        q = q,
        scale = scale,
        hurst_exponent = hurst_exponent,
        estimated_d = estimated_d,
        delta_H = delta_H,
        alpha = alpha,
        f_alpha = f_alpha,
        delta_alpha = delta_alpha,
        delta_f_alpha = delta_f_alpha,
        alpha_0 = alpha_0,
        B = B
    )
}

print_time_series_context <- function(context) {
    cat("\n")
    print("Configuration variables loaded successfully:")
    print(paste("Curated Folder:", context$curated_folder))
    print(paste("Features Folder:", context$features_folder))
    print(paste("Forecasting Folder:", context$forecasting_folder))
    print(paste("Indicator:", context$indicator))
    print(paste("Horizon:", context$horizon))
    print(paste("MFDFA window:", context$mfdfa_window))
    print(paste("River name:", context$river_name))
    cat("\n")
    print(paste("Curated Folder Directory:", context$curated_folder))
    print(paste("Features Folder Directory:", context$features_folder))
    print(paste("Forecasting Folder Directory:", context$forecasting_folder))
    print(paste("Max", context$indicator, "data value"))
    print(max(context$indicator_data[[context$indicator]]))
    print(paste("Min", context$indicator, "data value"))
    print(min(context$indicator_data[[context$indicator]]))
}
