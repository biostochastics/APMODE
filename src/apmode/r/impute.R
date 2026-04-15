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
#     "method":    "pmm" | "missForest",
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
#     "method":       "pmm" | "missForest"
#   }
#
# Method semantics:
#   - pmm: FCS via the mice package with Predictive Mean Matching.
#     The covariate imputation is performed on per-subject rows; imputed
#     values are then broadcast back onto the full observation-level data.
#   - missForest: single-imputation random-forest imputation applied m
#     times with different seeds to produce an ensemble usable under the
#     MI framework (Bräm CPT:PSP 2022). Not "true" MI because each run is
#     independent, but appropriate for nonlinear covariate structure.
#
# Security: no arbitrary R code is evaluated — only mice::mice() and
# missForest::missForest() are invoked with typed arguments.

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

  } else if (method == "missForest") {
    if (!requireNamespace("missForest", quietly = TRUE)) {
      .write_response(response_path, "error", "package_missing",
                      "missForest package required for RF imputation")
      quit(status = 1, save = "no")
    }

    # missForest is not natively multiple-imputation. We run m independent
    # draws with different seeds and treat the ensemble as MI. Coverage
    # properties are comparable to PMM for MAR covariates (Bräm 2022) but
    # do not satisfy Rubin's within/between decomposition exactly.
    for (i in seq_len(m)) {
      local_seed <- seed + i - 1L
      set.seed(local_seed)
      imp_out <- missForest::missForest(
        subj_df[, cov_names, drop = FALSE],
        verbose = FALSE
      )
      completed <- as.data.frame(imp_out$ximp)
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
