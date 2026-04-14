# SPDX-License-Identifier: GPL-2.0-or-later
"""Benchmark suite scaffolding (PRD §5).

Three-tier benchmark system:

  Suite A:          Synthetic recovery (A1-A7) — known ground truth
  Suite A-External: Schoemaker 2019 standard grid (nlmixr2data, 12 datasets)
  Suite B:          NODE validation (B1-B3) + real-data anchors (B4-B9)
  Suite C:          Expert head-to-head comparison (mavoglurant, gentamicin, propofol)

Supporting infrastructure:
  models:         DatasetCard, BenchmarkCase, BenchmarkScore, PerturbationRecipe
  perturbations:  Pure functions for controlled data modification
  scoring:        Backend-agnostic evaluation harness
"""
