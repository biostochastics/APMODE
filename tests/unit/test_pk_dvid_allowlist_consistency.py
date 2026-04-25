# SPDX-License-Identifier: GPL-2.0-or-later
"""Cross-module PK-DVID allowlist consistency guardrail.

Three modules independently filter NONMEM-style observation rows down to
"PK rows":

  1. ``apmode.data.profiler`` — uses ``_PK_DVIDS`` derived from
     ``policies/profiler.json`` (versioned policy artifact).
  2. ``apmode.data.adapters.to_nlmixr2_format`` — uses
     ``PK_DVID_ALLOWLIST`` (hard-coded, exported) to drop PD rows
     before handing the CSV to nlmixr2.
  3. ``apmode.data.initial_estimates.NCAEstimator`` — imports the same
     ``PK_DVID_ALLOWLIST`` from ``adapters`` so the per-subject NCA
     does not blow up on PD concentrations (the warfarin
     ``DVID="cp"``/``"pca"`` failure mode that motivated this guard).

The three sources MUST agree. Any divergence opens a class of bugs
where the runner ships nlmixr2 a PK-only CSV but the NCA initialised
its parameters from a PK+PD blend (or vice versa), and the Phase-1
fits silently traverse 4 OoMs in the outer optimiser.

This file pins the equality at test time so a future edit to one
module's allowlist trips a unit-test red rather than landing as a
silent regression on the next live Phase-1 run.
"""

from __future__ import annotations

from apmode.data.adapters import _PK_DVID_ALLOWLIST, PK_DVID_ALLOWLIST
from apmode.data.initial_estimates import PK_DVID_ALLOWLIST as NCA_FILTER
from apmode.data.profiler import _PK_DVIDS


class TestPKDVIDAllowlistConsistency:
    """The three module-level allowlists must agree element-for-element."""

    def test_adapters_public_alias_matches_underscored(self) -> None:
        # Backwards-compat alias kept for in-tree consumers that grew
        # an import on the underscored name before promotion.
        assert _PK_DVID_ALLOWLIST is PK_DVID_ALLOWLIST or (_PK_DVID_ALLOWLIST == PK_DVID_ALLOWLIST)

    def test_initial_estimates_imports_from_adapters(self) -> None:
        # NCAEstimator's filter source IS the adapter's allowlist by
        # construction (single import). This pins the structural
        # property: a refactor that copies the set instead of importing
        # it would trip the next test on drift.
        assert NCA_FILTER is PK_DVID_ALLOWLIST

    def test_profiler_policy_matches_adapter_allowlist(self) -> None:
        # Profiler reads its allowlist from a JSON policy file
        # (``policies/profiler.json``); adapters hard-code the same
        # set. They MUST be element-equal. Drift here is a class of
        # silent bugs: the profiler classifies a row as PK while the
        # adapter strips it (or vice versa), so the data the
        # downstream backend sees is no longer what the profiler
        # measured.
        assert _PK_DVIDS == PK_DVID_ALLOWLIST, (
            f"profiler._PK_DVIDS ({sorted(_PK_DVIDS)}) drifts from "
            f"adapters.PK_DVID_ALLOWLIST ({sorted(PK_DVID_ALLOWLIST)}). "
            "Update either policies/profiler.json or "
            "src/apmode/data/adapters.py so the two sources agree."
        )

    def test_allowlist_contains_canonical_tokens(self) -> None:
        # Lowercase / stripped string tokens by contract. If a future
        # edit accidentally casefolds upstream and forgets to update
        # this allowlist, the runner's ``str(...).strip().str.lower()``
        # comparison silently misses every row.
        for token in PK_DVID_ALLOWLIST:
            assert token == token.strip().lower(), (
                f"PK_DVID_ALLOWLIST contains non-canonical token "
                f"{token!r}; tokens must be already lowercased + "
                "stripped because the runner casefolds the data side."
            )

    def test_allowlist_includes_minimum_pk_synonyms(self) -> None:
        # Defence-in-depth: the canonical four tokens MUST be in the
        # set so that mixed-endpoint datasets like warfarin
        # (``DVID="cp"``) are routed correctly. Adding tokens is fine;
        # removing one of these four is a regression.
        required = {"cp", "1", "conc", "concentration"}
        missing = required - PK_DVID_ALLOWLIST
        assert not missing, f"required PK-DVID tokens missing from allowlist: {sorted(missing)}"
