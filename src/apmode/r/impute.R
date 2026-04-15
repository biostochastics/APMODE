# SPDX-License-Identifier: GPL-2.0-or-later
# APMODE multiple-imputation harness
#
# Usage:
#   Rscript impute.R <request.json> <response.json>
#
# Request JSON:
#   {
#     "source_csv": "/abs/path/data.csv",
#     "output_dir": "/abs/path/imputed",
#     "method":    "pmm" | "missRanger",
#     "m":         integer >= 1,
#     "seed":      integer,
#     "covariates": ["WT", "AGE", ...],     # columns to impute (subject-level)
#     "id_column": "NMID"                     # subject id (one value per subject)
#   }
#
# Response JSON:
#   {
#     "status":       "success" | "error",
#     "error_type":   NULL | "package_missing" | "imputation_failed" | "crash",
#     "message":      NULL | "<details>",
#     "imputed_csvs": [ "/abs/path/imp_1.csv", ... ],
#     "m":            integer,
#     "method":       "pmm" | "missRanger"
#   }
#
# Method semantics:
#   - pmm: FCS via the mice package with Predictive Mean Matching.
#     The covariate imputation is performed on per-subject rows; imputed
#     values are then broadcast back onto the full observation-level data.
#   - missRanger: ranger-backed random-forest imputation with PMM
#     (pmm.k = 10, num.trees = 100) — fast alternative to missForest,
#     Mayer CRAN 2.6.x. For multiple imputation we run m independent
#     calls with different seeds following the missRanger multiple-
#     imputation vignette; the PMM step restores between-imputation
#     variance so Rubin's rules apply downstream.
#
# Security: no arbitrary R code is evaluated — only mice::mice() and
# missRanger::missRanger() are invoked with typed arguments.

suppressPackageStartupMessages({
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript impute.R <request.json> <response.json>")
}

request_path  <- args[1]
response_path <- args[2]

.write_response <- function(path, status, error_type = NULL, message = NULL,
                            imputed_csvs = list(), m = 0L, method = "pmm") {
  resp <- list(
    status       = status,
    error_type   = error_type,
    message      = message,
    imputed_csvs = as.list(imputed_csvs),
    m            = as.integer(m),
    method       = method
  )
  write_json(resp, path, auto_unbox = TRUE, null = "null", pretty = TRUE)
}

tryCatch({
  req <- fromJSON(request_path, simplifyVector = TRUE)

  required_fields <- c("source_csv", "output_dir", "method", "m", "seed",
                       "covariates", "id_column")
  missing_fields <- setdiff(required_fields, names(req))
  if (length(missing_fields) > 0) {
    .write_response(response_path, "error", "crash",
                    paste("Missing request fields:",
                          paste(missing_fields, collapse = ", ")))
    quit(status = 1, save = "no")
  }

  method  <- req$method
  m       <- as.integer(req$m)
  seed    <- as.integer(req$seed)
  cov_names <- as.character(req$covariates)
  id_col    <- as.character(req$id_column)

  if (!dir.exists(req$output_dir)) {
    dir.create(req$output_dir, recursive = TRUE)
  }

  # Load and validate data
  full_df <- read.csv(req$source_csv, header = TRUE,
                      stringsAsFactors = FALSE, check.names = FALSE)
  if (!(id_col %in% names(full_df))) {
    .write_response(response_path, "error", "crash",
                    paste("id_column", id_col, "not in data"))
    quit(status = 1, save = "no")
  }

  missing_covs <- setdiff(cov_names, names(full_df))
  if (length(missing_covs) > 0) {
    .write_response(response_path, "error", "crash",
                    paste("Covariates not in data:",
                          paste(missing_covs, collapse = ", ")))
    quit(status = 1, save = "no")
  }

  # Subject-level slice: one row per subject with covariate values. mice
  # operates here; imputed values are broadcast back onto observations.
  subj_df <- aggregate(full_df[, cov_names, drop = FALSE],
                       by = list(.id = full_df[[id_col]]),
                       FUN = function(x) x[1])
  names(subj_df)[1] <- id_col

  imputed_csvs <- character(0)

  if (method == "pmm") {
    if (!requireNamespace("mice", quietly = TRUE)) {
      .write_response(response_path, "error", "package_missing",
                      "mice package required for PMM imputation")
      quit(status = 1, save = "no")
    }

    set.seed(seed)
    # Predictive Mean Matching with mice. m completed datasets.
    imp <- mice::mice(subj_df[, cov_names, drop = FALSE],
                      m = m, method = "pmm", seed = seed,
                      printFlag = FALSE)

    for (i in seq_len(m)) {
      completed <- mice::complete(imp, action = i)
      # Broadcast subject-level imputations onto the full observation data
      merged <- full_df
      for (cn in cov_names) {
        lookup <- setNames(completed[[cn]], subj_df[[id_col]])
        merged[[cn]] <- lookup[as.character(merged[[id_col]])]
      }
      out_path <- file.path(req$output_dir, sprintf("imp_%03d.csv", i))
      write.csv(merged, out_path, row.names = FALSE)
      imputed_csvs <- c(imputed_csvs, normalizePath(out_path))
    }

  } else if (method == "missRanger") {
    if (!requireNamespace("missRanger", quietly = TRUE)) {
      .write_response(response_path, "error", "package_missing",
                      "missRanger package required for ranger-backed RF imputation")
      quit(status = 1, save = "no")
    }

    # missRanger is a fast ranger-backed alternative to missForest (Mayer,
    # CRAN 2.6.x). It iterates RF imputation until out-of-bag error stops
    # improving, then applies predictive mean matching (pmm.k neighbours)
    # to keep imputed values in-range and restore variance. For multiple
    # imputation we run m independent calls with different seeds, as
    # recommended by the missRanger multiple-imputation vignette — large
    # pmm.k (~10) at that call site restores between-imputation variance.
    for (i in seq_len(m)) {
      local_seed <- seed + i - 1L
      set.seed(local_seed)
      completed <- missRanger::missRanger(
        subj_df[, cov_names, drop = FALSE],
        num.trees = 100,
        pmm.k = 10,
        seed = local_seed,
        verbose = 0
      )
      completed <- as.data.frame(completed)
      merged <- full_df
      for (cn in cov_names) {
        lookup <- setNames(completed[[cn]], subj_df[[id_col]])
        merged[[cn]] <- lookup[as.character(merged[[id_col]])]
      }
      out_path <- file.path(req$output_dir, sprintf("imp_%03d.csv", i))
      write.csv(merged, out_path, row.names = FALSE)
      imputed_csvs <- c(imputed_csvs, normalizePath(out_path))
    }

  } else {
    .write_response(response_path, "error", "crash",
                    paste("Unknown method:", method))
    quit(status = 1, save = "no")
  }

  .write_response(response_path, "success",
                  error_type = NULL, message = NULL,
                  imputed_csvs = imputed_csvs, m = m, method = method)

}, error = function(e) {
  .write_response(response_path, "error", "imputation_failed", e$message)
  quit(status = 1, save = "no")
})
