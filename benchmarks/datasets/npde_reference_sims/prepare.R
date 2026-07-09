# SPDX-License-Identifier: GPL-2.0-or-later
# npde reference simulated sets — for VPC/NPE calibration against Comets 2008
# Package license: GPL (>= 2)
# Reference: Comets E, Brendel K, Mentré F. (2008) Comp Meth Prog Biomed 90(2):154-166

suppressMessages(library(npde))
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) > 0) args[1] else "benchmarks/datasets/npde_reference_sims"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

data(theopp, package = "npde")
write.csv(theopp, file.path(out, "theopp.csv"), row.names = FALSE)
cat(sprintf("theopp: %d rows × %d cols\n", nrow(theopp), ncol(theopp)))

data(warfarin, package = "npde")
write.csv(warfarin, file.path(out, "warfarin_pk.csv"), row.names = FALSE)
cat(sprintf("warfarin: %d rows × %d cols\n", nrow(warfarin), ncol(warfarin)))

data(simwarfarinCov, package = "npde")
write.csv(simwarfarinCov, file.path(out, "simwarfarinCov.csv"), row.names = FALSE)
cat(sprintf("simwarfarinCov: %d rows × %d cols (1000 sims × 247 obs)\n",
            nrow(simwarfarinCov), ncol(simwarfarinCov)))

cat("\nNOTE: simwarfarinBase is NOT in CRAN npde v3.5. Fetch from:\n")
cat("  https://github.com/ecomets/npde30/blob/main/keep/data/simwarfarinBase.tab\n")
