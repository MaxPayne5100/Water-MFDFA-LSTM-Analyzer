source(file.path({
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
}, "R", "project_utils.R"))

source(file.path(script_directory(), "R", "time_series_pipeline.R"))

context <- load_time_series_context(find_project_root(script_directory()))
list2env(context, envir = environment())

require_r_packages(
    c("MFDFA", "moments", "pracma", "zoo"),
    "Rolling-window MFDFA task"
)

t_mfdfa <- mfdfa_window
r_step <- 1

# Choose q parameters
q <- seq(-10, 10, by = 1)

# Perform MFDFA
extract_mfdfa_coefficients <- function(window_data) {

  ssf_mean <- mean(window_data, na.rm = TRUE)
  ssf_volatility <- sd(window_data, na.rm = TRUE)

  skew <- moments::skewness(window_data)
  slope <- lm(window_data ~ seq_along(window_data))$coefficients[2]
  kurtosis <- moments::kurtosis(window_data)

  log_diffs <- diff(log(window_data))
  samp_en <- pracma::sample_entropy(log_diffs, edim = 2, r = 0.2 * sd(log_diffs))

  w_max <- max(window_data)
  w_min <- min(window_data)

  w_spread <- (w_max - w_min) / mean(window_data)

  min_scale <- 5
  max_scale <- ceiling(length(window_data) / 4)
  
  scale <- unique(
    round(
      exp(
        seq(log(min_scale), log(max_scale), length.out = 25)
      )
    )
  )

  mfdfa_result <- MFDFA::MFDFA(
      window_data,
      q = q,
      scale = scale
  )
  
  Hq <- mfdfa_result$Hq
  tau_q <- mfdfa_result$tau_q
  spec <- mfdfa_result$spec

  hurst_exponent <- mfdfa_result$Hq[q == 2]

  alpha <- as.numeric(spec$hq)
  f_alpha <- as.numeric(spec$Dq)
  if (
    length(alpha) < 5 ||
    length(alpha) != length(f_alpha) ||
    any(!is.finite(alpha)) ||
    any(!is.finite(f_alpha))
  ) {
    stop("MFDFA returned an invalid singularity spectrum")
  }

  alpha_min <- min(alpha)
  alpha_max <- max(alpha)
  alpha_width <- alpha_max - alpha_min
  if (!is.finite(alpha_width) || alpha_width <= 0) {
    stop("Singularity spectrum has zero or invalid alpha width")
  }

  # Normalize the alpha coordinate before fitting so the shape parameters are
  # comparable across rolling windows with different spectrum widths.
  u <- (alpha - alpha_min) / alpha_width

  delta_alpha <- max(alpha) - min(alpha)

  alpha_0_index <- which.max(f_alpha)
  alpha_0 <- alpha[alpha_0_index]

  delta_H <- Hq[q == min(q)] - Hq[q == max(q)]

  max_alpha_index <- which.max(alpha)
  min_alpha_index <- which.min(alpha)

  delta_f_alpha <- f_alpha[min_alpha_index] - f_alpha[max_alpha_index]

  # Define geometric left/right boundaries independently of the ordering
  # returned by MFDFA. For the configured ascending q sequence, MFDFA returns
  # alpha in descending order, so using alpha[1] as the left boundary reverses
  # the sign of the spectrum asymmetry.
  alpha_L <- alpha_min
  alpha_R <- alpha_max

  # Calculate the positive left and right widths around the spectrum maximum.
  delta_alpha_L <- alpha_0 - alpha_L
  delta_alpha_R <- alpha_R - alpha_0

  # Positive B means that the left side is wider than the right side.
  B <- (delta_alpha_L - delta_alpha_R) / alpha_width

  spectrum_data <- data.frame(u = u, f_alpha = f_alpha)
  fit_model <- tryCatch({
    nls(
      f_alpha ~ eps + A * u^beta * (1 - u)^beta2,
      data = spectrum_data,
      start = list(A = max(f_alpha) - min(f_alpha), beta = 0.5, beta2 = 0.5, eps = min(f_alpha)),
      algorithm = "port",
      lower = c(A = 0, beta = 0.01, beta2 = 0.01, eps = -Inf),
      upper = c(A = Inf, beta = Inf, beta2 = Inf, eps = Inf)
    )
  }, error = function(e) {
    stop("Singularity-spectrum NLS fit failed: ", e$message)
  })

  fitted_spectrum <- as.numeric(predict(fit_model, newdata = spectrum_data))
  residuals <- f_alpha - fitted_spectrum
  fit_rmse <- sqrt(mean(residuals^2))
  total_sum_squares <- sum((f_alpha - mean(f_alpha))^2)
  fit_r_squared <- if (total_sum_squares > 0) {
    1 - sum(residuals^2) / total_sum_squares
  } else {
    0
  }
  coefficients <- coef(fit_model)

  return(unname(c(
    # Existing rolling feature columns are retained for backward compatibility.
    hurst_exponent, delta_H, delta_alpha, alpha_0, delta_f_alpha, B,
    coefficients[["A"]], coefficients[["beta"]], coefficients[["beta2"]],
    ssf_mean, ssf_volatility, skew, slope, kurtosis, samp_en, w_spread,
    # Dedicated spectrum-label columns.
    alpha_min, alpha_max, alpha_width, alpha_0, delta_f_alpha, B,
    min(f_alpha), max(f_alpha), max(f_alpha) - min(f_alpha),
    coefficients[["A"]], coefficients[["beta"]], coefficients[["beta2"]],
    coefficients[["eps"]], fit_rmse, fit_r_squared, TRUE
  )))
}

ts_zoo <- zoo::zoo(ts_data)

mfdfa_coefficients_matrix <- zoo::rollapply(
  data = ts_zoo, 
  width = t_mfdfa,                      
  by = r_step,                          
  FUN = extract_mfdfa_coefficients,
  align = "right",                      
  by.column = FALSE,                   
  fill = NA                         
)


mfdfa_features_df <- data.frame(mfdfa_coefficients_matrix)
feature_columns <- c("H", "Delta_H", "Delta_alpha", "Alpha_0", "Delta_f_alpha", "B", "A", "beta", "beta2", "Mean", "Volatility", "Skew", "Slope", "Kurtosis", "Entropy", "Width")
spectrum_label_columns <- c(
  "Spectrum_Alpha_Min", "Spectrum_Alpha_Max", "Spectrum_Alpha_Width",
  "Spectrum_Alpha_0", "Spectrum_Delta_f_alpha", "Spectrum_B",
  "Spectrum_F_Min", "Spectrum_F_Max", "Spectrum_F_Height",
  "Spectrum_A", "Spectrum_Beta", "Spectrum_Beta2", "Spectrum_Eps",
  "Spectrum_Fit_RMSE", "Spectrum_Fit_R2", "Spectrum_Fit_Converged"
)
colnames(mfdfa_features_df) <- c(feature_columns, spectrum_label_columns)

# Keep only complete rolling windows. Any missing value after this point means
# a valid window failed and must stop the task rather than silently shifting
# feature/target alignment.
valid_window_rows <- seq.int(mfdfa_window, length(ts_data))
mfdfa_features_df <- mfdfa_features_df[valid_window_rows, , drop = FALSE]
rownames(mfdfa_features_df) <- as.character(valid_window_rows)
if (anyNA(mfdfa_features_df)) {
  stop("Rolling MFDFA output contains missing values in a valid window")
}

spectrum_labels_df <- mfdfa_features_df[, spectrum_label_columns, drop = FALSE]

validate_mfdfa_features(
  mfdfa_features_df[, feature_columns, drop = FALSE],
  feature_columns,
  length(ts_data),
  mfdfa_window
)

validate_spectrum_labels(
  spectrum_labels_df,
  spectrum_label_columns,
  length(ts_data),
  mfdfa_window
)

features_output_path <- paste0(
    features_folder,
    "/MFDFA_", river_name, "_", indicator, "_", mfdfa_window, "_window.csv"
)
features_temporary_path <- temporary_output_path(features_output_path)
lineage_path <- tryCatch(
    {
        write.csv(
            mfdfa_features_df[, feature_columns, drop = FALSE],
            features_temporary_path,
            row.names = TRUE,
            quote = FALSE
        )
        publish_atomic_output(
            features_temporary_path,
            features_output_path,
            context$lineage_context,
            "features_mfdfa",
            input_files = c(context$prepared_data_file),
            extra = list(
                feature_count = length(feature_columns),
                row_count = nrow(mfdfa_features_df),
                mfdfa_window = mfdfa_window
            )
        )
    },
    finally = unlink(features_temporary_path)
)
message("Lineage metadata written to: ", lineage_path)

spectrum_output_path <- paste0(
    features_folder,
    "/SpectrumLabels_", river_name, "_", indicator, "_", mfdfa_window, "_window.csv"
)
spectrum_temporary_path <- temporary_output_path(spectrum_output_path)
spectrum_lineage_path <- tryCatch(
    {
        write.csv(
            spectrum_labels_df,
            spectrum_temporary_path,
            row.names = TRUE,
            quote = FALSE
        )
        publish_atomic_output(
            spectrum_temporary_path,
            spectrum_output_path,
            context$lineage_context,
            "features_spectrum",
            input_files = c(context$prepared_data_file),
            extra = list(
                label_count = length(spectrum_label_columns),
                row_count = nrow(spectrum_labels_df),
                mfdfa_window = mfdfa_window,
                alpha_coordinate_normalized = TRUE,
                model = "eps + A * u^beta * (1 - u)^beta2"
            )
        )
    },
    finally = unlink(spectrum_temporary_path)
)
message("Spectrum-label lineage written to: ", spectrum_lineage_path)
