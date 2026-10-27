# Entry point: load the configured time series and run base MFDFA analysis.

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
print_time_series_context(context)

cat("\n")
print("Hurst exponent obtained with MFDFA:")
print(hurst_exponent)

cat("\n")
print("Multifractal parameters obtained with MFDFA:")
print(data.frame(hurst_exponent, delta_H, delta_alpha, delta_f_alpha, B))
