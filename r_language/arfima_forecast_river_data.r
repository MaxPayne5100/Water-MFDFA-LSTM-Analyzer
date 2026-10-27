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
    c("Metrics", "arfima", "forecast", "fracdiff"),
    "ARFIMA forecast task"
)

# ######################################################      R/S ANALYSIS       ######################################################

# # get Hurst exponent using R/S analysis
# print(pracma::hurstexp(train))

# ######################################################      AUTO ARFIMA       ######################################################

# start_time <- Sys.time()

model <- forecast::arfima(train)
# print(summary(model))
# forecast::tsdisplay(model$residuals, main = "Residuals")
forecasted_out <- forecast::forecast(model, level = c(95), h = horizon)

# end_time <- Sys.time()
# time_taken <- end_time - start_time

# cat("\n")
# print(time_taken)

forecasted_dates_numeric <- as.numeric(row.names(as.data.frame(forecasted_out)))

forecasted_values <- forecasted_out[2]$mean

par(c(1, 1))
plot(forecasted_out,
     xaxt = "n",
     xlab = "Day",
     ylab = paste0(indicator),
     xlim = c(time(train)[length(train) - 60],
            forecasted_dates_numeric[length(forecasted_dates_numeric)]),
     main = "ARFIMA model predictions",
     showgap = FALSE,
     shadecols = "#C6DCEC",
     fcol = "black")
axis(1,
     time(ts_data),
     as.numeric(time(ts_data)))

cat("\n")
print(forecast::arimaorder(model))
print("ARFIMA accuracy: ")
print(forecast::accuracy(forecasted_out, test))

# ######################################################      AUTO ARIMA       ######################################################

model_arima <- forecast::auto.arima(train)
forecasted_arima_out <- forecast::forecast(model_arima,
                                  level = c(95),
                                  h = horizon)
cat("\n")
print(forecast::arimaorder(model_arima))
print("ARIMA accuracy: ")
print(forecast::accuracy(forecasted_arima_out, test))
cat("\n")

# ######################################################      MANUAL ARFIMA       ######################################################

diff_train <- fracdiff::diffseries(train, d = estimated_d)

#forecast::tsdisplay(diff_train, main = paste("Differenciated", indicator), ylab=paste0(indicator), xlab = "Measurement number", ci.col = 'black')

acf_res  <- acf(diff_train, plot = FALSE)
pacf_res <- pacf(diff_train, plot = FALSE)

n <- length(diff_train)
conf_limit <- 1.96 / sqrt(n)  # significance threshold

# Significant ACF spikes
sig_acf <- data.frame(
  lag = acf_res$lag[, , 1],
  acf = acf_res$acf[, , 1]
)
sig_acf <- sig_acf[abs(sig_acf$acf) > conf_limit, ]$lag

print("Significant ACF lags:")
print(sig_acf)

# Significant PACF spikes
sig_pacf <- data.frame(
  lag = pacf_res$lag,
  pacf = pacf_res$acf
)
sig_pacf <- sig_pacf[abs(sig_pacf$pacf) > conf_limit, ]$lag

cat("\n")
print("Significant PACF lags:")
print(sig_pacf)

# # from acf and pacf plots get significant spikes
ar_params <- sig_acf
ma_params <- sig_pacf

oldw <- getOption("warn")
options(warn = -1)
param_vec <- find_best_arfima_model(
    ar_params,
    ma_params,
    train,
    test,
    horizon,
    estimated_d
)
options(warn = oldw)

ar_param <- param_vec[1]
ma_param <- param_vec[2]
d_param <- param_vec[3]

model_arfima_man <- arfima::arfima(train,
                                   order = c(ar_param, ma_param, d_param),
                                   numeach = c(1, 1))
forecasted_arfima_man_out <- predict(model_arfima_man,
                                     n.ahead = horizon)

# plot(residuals(model_arfima_man)[[1]],
#                ylab = "",
#                type = "b",
#                main = paste0("Residuals of ARFIMA(",
#                  ar_param, ",", ma_param, ",", d_param, ")"))

# acf(residuals(model_arfima_man)[[1]],
#  main = paste0("Residuals of ARFIMA(",
#   ar_param, ",", ma_param, ",", d_param, ") ACF"))
# pacf(residuals(model_arfima_man)[[1]],
#  main = paste0("Residuals of ARFIMA(",
#   ar_param, ",", ma_param, ",", d_param, ") PACF"))

forecasted_arfima_man_values <- forecasted_arfima_man_out[[1]]$Forecast

forecasted_dates_numeric <- as.numeric(row.names(as.data.frame(forecasted_arfima_man_values)))

cat("\n")
print("p d q")
print(paste(ar_param, d_param, ma_param))
print("ARFIMA manually fitted accuracy: ")
cat("\n")
print(paste0("Test set RMSE: ",
             Metrics::rmse(test, forecasted_arfima_man_values)))
print(paste0("Test set MAPE (%): ",
             Metrics::mape(test, forecasted_arfima_man_values) * 100))

forecasted_to_csv(forecasted_dates_numeric,
                  forecasted_arfima_man_values,
                  paste0(forecasting_folder, "/Forecasted_", river_name, '_', indicator, '_', horizon, "_days.csv"),
                  horizon,
                  lineage_context = context$lineage_context,
                  stage = "forecast_r",
                  input_files = c(context$prepared_data_file))

######################################################      MODELS COMPARISON       ######################################################
print(AIC(model, model_arima, model_arfima_man))
cat("\n")
