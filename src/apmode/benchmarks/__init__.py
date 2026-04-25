# SPDX-License-Identifier: GPL-2.0-or-later
"""Benchmark suite scaffolding (PRD §5).

Three-tier benchmark system:

  Suite A:          Synthetic recovery (A1-A8) — known ground truth
  Suite A-External: Schoemaker 2019 standard grid (nlmixr2data, 12 datasets)
  Suite B:          NODE validation (B1-B3, mock) + real-data anchors (B4-B9)
  Suite C Phase 1:  Literature-anchor head-to-head (5 MLE + 2 Bayesian)
  Suite C Phase 2:  Synthetic-panel methodology validation (NOT a real-expert
                    benchmark; the blinded-human-expert head-to-head is gated
                    on external collaborator coordination and out of v0.6 scope)

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
  suite_c_phase1_runner:        Live Phase-1 runner — held-out NPE per fold
                                + fixed-THETA literature comparator
  suite_c_phase1_scoring:       Phase-1 scoring math (FixtureScore,
                                fraction-beats-literature-median)
  suite_c_phase1_cli:           Phase-1 scorer CLI used by the weekly workflow
  suite_c_phase2_synthetic:     Synthetic-panel methodology validation
                                scaffold for the Phase 2 metric;
                                explicitly NOT a real-expert claim
"""
