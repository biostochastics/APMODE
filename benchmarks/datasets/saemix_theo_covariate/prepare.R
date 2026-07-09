# SPDX-License-Identifier: GPL-2.0-or-later
# saemix::theo.saemix — theophylline with SIMULATED Sex covariate
# Package license: GPL (>= 2)
# Reference: Comets, Lavenu, Lavielle (saemix); Boeckmann/Sheiner/Beal 1994 (base)

suppressMessages(library(saemix))
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) > 0) args[1] else "benchmarks/datasets/saemix_theo_covariate"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

data(theo.saemix, package = "saemix")
write.csv(theo.saemix, file.path(out, "theo_saemix.csv"), row.names = FALSE)
cat(sprintf("theo.saemix written: %d rows × %d cols → %s\n",
            nrow(theo.saemix), ncol(theo.saemix),
            file.path(out, "theo_saemix.csv")))
