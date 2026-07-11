# SPDX-License-Identifier: GPL-2.0-or-later
"""Shared, non-test helpers for the APMODE test suite.

This package is the single home for factories/builders and fixture data that
were previously copy-pasted across test modules (or imported cross-module from
sibling test files). It is deliberately NOT test-prefixed so pytest does not
collect it.

Submodules:
- ``builders``  — Pydantic factories (make_spec, make_backend_result, manifests, scoring contracts)
- ``policies``  — POLICY_DIR + load_policy(lane)
- ``bundles``   — reproducibility-bundle builders (build_submission_bundle, seal/reseal helpers)
- ``strategies``— shared Hypothesis strategies for property tests
"""
