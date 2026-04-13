# SPDX-License-Identifier: GPL-2.0-or-later
# Install R dependencies for the APMODE nlmixr2 harness.
# Run once: Rscript install_deps.R

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "https://cran.r-project.org")
}

if (!requireNamespace("nlmixr2", quietly = TRUE)) {
  install.packages("nlmixr2", repos = "https://cran.r-project.org")
}

message("APMODE R dependencies installed successfully.")
message(paste("  jsonlite:", packageVersion("jsonlite")))
message(paste("  nlmixr2:", packageVersion("nlmixr2")))
message(paste("  R:", R.version.string))
