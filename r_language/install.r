required_packages <- c(
"forecast",
"tsbox",
"xts",
"lubridate",
"pracma",
"arfima",
"Metrics",
"fracdiff",
"MFDFA",
"jsonlite",
"zoo",
"moments"
)

optional_packages <- c("LongMemoryTS")

options(repos = c(CRAN = "https://cloud.r-project.org"))

user_library <- Sys.getenv("R_LIBS_USER")
if (!nzchar(user_library)) {
  user_library <- file.path(
    path.expand("~"),
    "R",
    paste0(R.version$major, ".", R.version$minor),
    "library"
  )
}
dir.create(user_library, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_library, .libPaths()))

packages_to_install <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

if (length(packages_to_install) > 0) {
message("Installing the following packages: ", paste(packages_to_install, collapse = ", "))
 for (package in packages_to_install) {
   if (!requireNamespace(package, quietly = TRUE)) {
     message("Installing ", package)
     install.packages(
       package,
       lib = user_library,
       dependencies = c("Depends", "Imports", "LinkingTo")
     )
   }
 }
message("Installation complete.")
} else {
message("All required packages are already installed.")
}

missing_required <- required_packages[!vapply(
  required_packages,
  requireNamespace,
  logical(1),
  quietly = TRUE
)]
if (length(missing_required) > 0) {
  stop(
    "Required R package installation failed: ",
    paste(missing_required, collapse = ", ")
  )
}

missing_optional <- optional_packages[!vapply(
  optional_packages,
  requireNamespace,
  logical(1),
  quietly = TRUE
)]
if (length(missing_optional) > 0) {
  warning(
    "Optional R package(s) unavailable: ",
    paste(missing_optional, collapse = ", "),
    ". Brownian multifractal analysis will remain unavailable."
  )
}

message("All required packages are installed and ready to use.")
