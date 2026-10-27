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

require_r_packages(c("LongMemoryTS"), "Brownian multifractal analysis task")

# Generate Fractional Brownian Motion
set.seed(123)  # For reproducibility
gen_N <- 1000
gen_H <- 0.95
gen_d <- gen_H - 0.5

fbm_data <- LongMemoryTS::fBM(n = gen_N, d = gen_d, type = "I")

fbm_data <- fbm_data * 10

cat("\n")
print("Max Fractional Brownian Motion (less fluctuated) data value")
print(max(fbm_data))

print("Min Fractional Brownian Motion (less fluctuated) data value")
print(min(fbm_data))

gen_H <- 0.05
gen_d <- gen_H - 0.5

fbm_data1 <- LongMemoryTS::fBM(n = gen_N, d = gen_d, type = "I")

fbm_data1 <- fbm_data1 * rnorm(length(fbm_data1), mean = 0, sd = 3)

ts_data <- ts(fbm_data, frequency = 1)

cat("\n")
print("Max Fractional Brownian Motion (more fluctuated) data value")
print(max(fbm_data1))

print("Min Fractional Brownian Motion (more fluctuated) data value")
print(min(fbm_data1))

#Plot the generated fBm data
plot(fbm_data, type = "l",
 main = paste("Fractional Brownian Motion (H =", gen_H, ")"),
 xlab = "q", ylab = "H(q)",
 panel.first = grid(),
 lwd = 2,
 pch = 19,
 cex = 2,
 cex.lab = 1.5,
 cex.axis = 1.5,
 cex.main = 2
 )
lines(fbm_data, col = "#383838")
