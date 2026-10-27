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

require_r_packages(c("Metrics"), "Expanded multifractal analysis task")

library(Metrics)

# ######################################################      MFDFA  Visual Analysis     ######################################################

plot(q, Hq, type = "b",
 main = "The curves of the generalized Hurst exponents H(q) and order q",
 xlab = "q", ylab = "H(q)",
 panel.first = grid(),
 lwd = 2,
 pch = 19,
 cex = 2,
 cex.lab = 1.5,
 cex.axis = 1.5,
 cex.main = 2
 )
lines(q, Hq, col = "#383838")
points(q, Hq, col = "orange", pch = 19)

plot(q, tau_q, type = "b",
 main = "The scaling exponents tau(q) and order q",
 xlab = "q", ylab = "tau(q)",
 panel.first = grid(),
 lwd = 2,
 pch = 19,
 cex = 2,
 cex.lab = 1.5,
 cex.axis = 1.5,
 cex.main = 2
 )
lines(q, tau_q, col = "#383838")
points(q, tau_q, col = "orange", pch = 19)

plot(alpha, f_alpha, type = "b",
 main = "The singularity spectrum f(alpha) and singularity alpha",
 xlab = "alpha", ylab = "f(alpha)",
 panel.first = grid(),
 lwd = 2,
 pch = 19,
 cex = 2,
 cex.lab = 1.5,
 cex.axis = 1.5,
 cex.main = 2
 )
lines(alpha, f_alpha, col = "#383838")
points(alpha, f_alpha, col = "orange", pch = 19)

# ######################################################      Multifractal Strength Analysis       ######################################################

x <- alpha - alpha_0

# Fit the quadratic model (second-order) using least squares
model <- lm(f_alpha ~ (x) + I((x)^2))

print(coefficients(model))
cat("\n")

# Fit the power model using non-linear least squares
model_new <- nls(f_alpha ~ A * (alpha - min(alpha) + eps)**beta *
 (max(alpha) - alpha + eps)**beta2 + eps,
 start = list(A = 2.5, beta = 0.5, beta2 = 0.5, eps = 0.01))

print(model_new)

print(coef(model_new))
cat("\n")
print("B value")
print(B)

x <- alpha

#Plot the original data points
plot(x, f_alpha, type = "b",
 main = "The singularity spectrum f(alpha) and singularity alpha",
 xlab = "aplha", ylab = "f(alpha)",
 panel.first = grid(),
 lwd = 2,
 pch = 19,
 cex.lab = 1.4,
 cex.axis = 1.4,
 cex.main = 2,
 xlim = c(0.4, 1.5),
 ylim = c(0, 1.1)
 )
lines(x, f_alpha, col = "#383838")
points(x, f_alpha, col = "#383838", pch = 19)

#Add the fitted quadratic curve
curve(coefficients(model)[1] + coefficients(model)[2] * (x - alpha_0) + 
        coefficients(model)[3] * (x - alpha_0)^2, add = TRUE, col = "#10b3ef", lwd = 2,
 pch = 19,
 cex = 2)

#Add the fitted power curve
lines(x, predict(model_new), col = "#d83418", lwd = 2,
 pch = 19,
 cex = 2)

# Add a legend
legend("topleft", legend = c("Actual Data", "Fitted Quadratic", "Fitted Power"), col = c("#383838", "#10b3ef", "#d83418"), cex=1.3, pch = 19)

# RMSE approximation comparison
actual_values <- f_alpha
predicted_values_quad <- coefficients(model)[1] + coefficients(model)[2] * (x - alpha_0) + 
        coefficients(model)[3] * (x - alpha_0)^2
predicted_values_exp <- predict(model_new)
rmse_value_quad <- rmse(actual_values, predicted_values_quad)
rmse_value_exp <- rmse(actual_values, predicted_values_exp)

print(alpha)
cat("\n")
print(alpha)
cat("\n")
print(actual_values)
cat("\n")
print(predicted_values_quad)
cat("\n")
print(predicted_values_exp)

cat("\n")
print("RMSE approximation comparison: ")
print(paste0('RMSE value for quadratic approxiamtion: ', rmse_value_quad))
print(paste0('RMSE value for exponential approxiamtion: ', rmse_value_exp))
