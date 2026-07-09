# SPDX-License-Identifier: GPL-2.0-or-later
"""Benchmark suite scaffolding (PRD §5).

Three-tier benchmark system:

  Suite A:          Synthetic recovery (A1-A8) — known ground truth
  Suite A-External: Schoemaker 2019 standard grid (nlmixr2data, 12 datasets)
  Suite B:          NODE validation (B1-B3, mock) + real-data anchors (B4-B12)
  Suite C:          Literature-anchor head-to-head (5 MLE + 2 Bayesian)
  Suite C Synthetic Panel:
                    Synthetic-panel methodology validation (NOT a
                    real-expert benchmark; blinded-human-expert head-to-head
                    depends on external collaborator coordination)

Supporting infrastructure:

  models:                       DatasetCard, BenchmarkCase, BenchmarkScore,
                                PerturbationRecipe, PerturbationType,
                                LiteratureFixture, LiteratureReference
  perturbations:                Pure functions for controlled data
                                modification — BLQ, outliers, sparse
                                absorption, null covariates, sparsify,
                                protocol pooling, occasion labels,
                                covariate missingness (PRD §5), and the
                                four PRD §10 stress surfaces (BSV scaling,
                                saturable clearance, TMDD, flip-flop)
  scoring:                      Backend-agnostic evaluation harness
                                (structure recovery, parameter bias,
                                NPE, prediction-interval calibration,
                                fraction-beats-median-expert)
  suite_b_runner:               Live Suite B runner — perturb → multi-seed
                                fit → score, with the PRD R8 cross-seed
                                stability monitor on parameter estimates
  suite_b_cli:                  Score-only CLI for Suite B results JSON
  suite_c_phase1_runner:        Live Suite C runner — held-out NPE per fold
                                + fixed-THETA literature comparator
  suite_c_phase1_scoring:       Suite C scoring math (FixtureScore,
                                fraction-beats-literature-median)
  suite_c_phase1_cli:           Suite C scorer CLI used by the weekly workflow
  suite_c_phase2_synthetic:     Synthetic-panel methodology validation
                                scaffold; explicitly NOT a real-expert claim
"""
