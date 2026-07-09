# SPDX-License-Identifier: GPL-2.0-or-later
# Extended nlmixr2data grid — datasets not already covered by
# nlmixr2data_{schoemaker,theophylline,warfarin,mavoglurant,phenobarbital,oral_1cpt}.
#
# Usage: Rscript prepare.R [output_dir]
# Reference: Schoemaker et al. ACoP 2016; nlmixr2data v2.0.9
# License: GPL-3.0-only (inherited from nlmixr2data)

suppressMessages(library(nlmixr2data))

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) > 0) args[1] else "benchmarks/datasets/nlmixr2data_extended"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

targets <- c(
  "Oral_2CPT","Oral_1CPTMM","Oral_2CPTMM",
  "Bolus_1CPT","Bolus_2CPT","Bolus_1CPTMM","Bolus_2CPTMM",
  "Infusion_1CPT","Infusion_2CPT","Infusion_1CPTMM","Infusion_2CPTMM",
  "wbcSim","nimoData","metabolite","Wang2007","nmtest",
  "invgaussian","rats","pump","theo_md"
)

summary_rows <- list()
for (ds in targets) {
  e <- new.env()
  tryCatch({
    data(list = ds, package = "nlmixr2data", envir = e)
    x <- e[[ds]]
    if (is.null(x)) x <- get(ds, envir = e)
    fp <- file.path(out, paste0(ds, ".csv"))
    write.csv(x, fp, row.names = FALSE)
    summary_rows[[ds]] <- data.frame(dataset = ds, rows = nrow(x),
                                     cols = ncol(x),
                                     bytes = file.info(fp)$size)
  }, error = function(e) {
    message(sprintf("FAILED %s: %s", ds, e$message))
  })
}
summary_df <- do.call(rbind, summary_rows)
write.csv(summary_df, file.path(out, "_MANIFEST.csv"), row.names = FALSE)
cat(sprintf("Wrote %d datasets to %s\n", nrow(summary_df), out))
print(summary_df)
