# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the code-derived DSL emitter capability matrix (P0.7).

Covers:
- Every CapabilityTag is classified (SUPPORTS xor EXPLICITLY_UNSUPPORTED)
  by every registered emitter — no silent gaps.
- ``tags_for_spec`` derives the expected tags from a compiled DSLSpec,
  including the block-structure / maturation-form feature tags.
- ``report`` accepts both a DSLSpec and an explicit tag set.
- ``scripts/verify_capability_coverage.py`` exits 0 (CI-facing contract).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from apmode.dsl.ast_models import (
    IIV,
    Additive,
    CovariateLink,
    DSLSpec,
    FirstOrder,
    LinearElim,
    NODEAbsorption,
    ObservationEndpoint,
    OneCmt,
    Proportional,
)
from apmode.dsl.capabilities import (
    CapabilityTag,
    registered_emitters,
    report,
    tags_for_spec,
)


def _minimal_spec(**overrides: object) -> DSLSpec:
    defaults: dict[str, object] = {
        "model_id": "capability-matrix-test",
        "absorption": FirstOrder(),
        "distribution": OneCmt(),
        "elimination": LinearElim(),
        "variability": [IIV(params=["CL"], structure="diagonal")],
        "observation": Proportional(sigma_prop=0.2),
    }
    defaults.update(overrides)
    return DSLSpec(**defaults)


class TestCoverage:
    """Pins the no-silent-gap contract for every registered emitter."""

    def test_every_tag_classified_by_every_emitter(self) -> None:
        for emitter in registered_emitters():
            classified = emitter.supports | emitter.explicitly_unsupported
            missing = [tag for tag in CapabilityTag if tag not in classified]
            assert not missing, f"{emitter.name} has unclassified tags: {missing}"

    def test_supports_and_unsupported_are_disjoint(self) -> None:
        for emitter in registered_emitters():
            overlap = emitter.supports & emitter.explicitly_unsupported
            assert not overlap, (
                f"{emitter.name} marks tags as both supported and "
                f"explicitly unsupported: {overlap}"
            )

    def test_at_least_three_emitters_registered(self) -> None:
        names = {emitter.name for emitter in registered_emitters()}
        assert {"nlmixr2", "stan", "frem"}.issubset(names)

    def test_verify_script_exits_zero(self) -> None:
        script = Path(__file__).resolve().parents[3] / "scripts" / "verify_capability_coverage.py"
        assert script.is_file()
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "COVERAGE GAP" not in result.stdout


class TestTagsForSpec:
    def test_minimal_spec_tags(self) -> None:
        tags = tags_for_spec(_minimal_spec())
        assert tags == {
            CapabilityTag.ABSORPTION_FIRST_ORDER,
            CapabilityTag.DISTRIBUTION_ONE_CMT,
            CapabilityTag.ELIMINATION_LINEAR,
            CapabilityTag.OBSERVATION_PROPORTIONAL,
            CapabilityTag.VARIABILITY_IIV,
        }

    def test_block_structure_feature_tag(self) -> None:
        spec = _minimal_spec(variability=[IIV(params=["CL", "V"], structure="block")])
        tags = tags_for_spec(spec)
        assert CapabilityTag.VARIABILITY_IIV in tags
        assert CapabilityTag.VARIABILITY_IIV_BLOCK_STRUCTURE in tags

    def test_diagonal_structure_omits_block_tag(self) -> None:
        tags = tags_for_spec(_minimal_spec())
        assert CapabilityTag.VARIABILITY_IIV_BLOCK_STRUCTURE not in tags

    def test_maturation_covariate_feature_tag(self) -> None:
        spec = _minimal_spec(
            variability=[IIV(params=["CL"], structure="diagonal")],
            covariates=[
                CovariateLink(param="CL", covariate="PMA", form="maturation", tm50=45.0, hill=3.0),
            ],
        )
        tags = tags_for_spec(spec)
        assert CapabilityTag.VARIABILITY_COVARIATE_LINK in tags
        assert CapabilityTag.VARIABILITY_COVARIATE_MATURATION_FORM in tags

    def test_non_maturation_covariate_omits_maturation_tag(self) -> None:
        spec = _minimal_spec(
            variability=[IIV(params=["CL"], structure="diagonal")],
            covariates=[
                CovariateLink(param="CL", covariate="WT", form="power", theta=0.75, ref=70.0),
            ],
        )
        tags = tags_for_spec(spec)
        assert CapabilityTag.VARIABILITY_COVARIATE_LINK in tags
        assert CapabilityTag.VARIABILITY_COVARIATE_MATURATION_FORM not in tags

    def test_node_absorption_tag(self) -> None:
        spec = _minimal_spec(
            absorption=NODEAbsorption(dim=2, constraint_template="bounded_positive")
        )
        tags = tags_for_spec(spec)
        assert CapabilityTag.ABSORPTION_NODE in tags

    def test_multi_analyte_observations_tag(self) -> None:
        spec = _minimal_spec(
            observations={
                "plasma": ObservationEndpoint(
                    name="plasma",
                    dvid=1,
                    prediction="C_central",
                    error=Proportional(sigma_prop=0.2),
                ),
                "metabolite": ObservationEndpoint(
                    name="metabolite",
                    dvid=2,
                    prediction="C_central",
                    error=Additive(sigma_add=0.3),
                ),
            }
        )
        tags = tags_for_spec(spec)
        assert CapabilityTag.OBSERVATION_MULTI_ANALYTE in tags
        assert CapabilityTag.OBSERVATION_PROPORTIONAL in tags
        assert CapabilityTag.OBSERVATION_ADDITIVE in tags

    def test_no_observations_omits_multi_analyte_tag(self) -> None:
        tags = tags_for_spec(_minimal_spec())
        assert CapabilityTag.OBSERVATION_MULTI_ANALYTE not in tags


class TestReport:
    def test_report_accepts_spec(self) -> None:
        result = report(_minimal_spec())
        assert set(result) == {"nlmixr2", "stan", "frem"}
        assert result["nlmixr2"][CapabilityTag.ABSORPTION_FIRST_ORDER.value] == "supported"
        assert result["stan"][CapabilityTag.ABSORPTION_FIRST_ORDER.value] == "supported"
        assert result["frem"][CapabilityTag.ABSORPTION_FIRST_ORDER.value] == "supported"

    def test_report_accepts_explicit_tag_set(self) -> None:
        result = report({CapabilityTag.ABSORPTION_NODE})
        for emitter_name in ("nlmixr2", "stan", "frem"):
            assert (
                result[emitter_name][CapabilityTag.ABSORPTION_NODE.value]
                == "experimental_no_stable_backend"
            )

    def test_report_surfaces_stan_iov_as_supported(self) -> None:
        result = report({CapabilityTag.VARIABILITY_IOV})
        assert result["stan"][CapabilityTag.VARIABILITY_IOV.value] == "supported"
        assert result["nlmixr2"][CapabilityTag.VARIABILITY_IOV.value] == "supported"

    def test_report_surfaces_stan_block_iiv_gap_as_explicitly_unsupported(self) -> None:
        result = report({CapabilityTag.VARIABILITY_IIV_BLOCK_STRUCTURE})
        assert (
            result["stan"][CapabilityTag.VARIABILITY_IIV_BLOCK_STRUCTURE.value]
            == "explicitly_unsupported"
        )
        assert (
            result["nlmixr2"][CapabilityTag.VARIABILITY_IIV_BLOCK_STRUCTURE.value] == "supported"
        )

    def test_report_surfaces_multi_analyte_observations_status(self) -> None:
        """P1.7: nlmixr2 supports multi-analyte observations:; Stan/FREM are Phase 2 gaps."""
        result = report({CapabilityTag.OBSERVATION_MULTI_ANALYTE})
        assert result["nlmixr2"][CapabilityTag.OBSERVATION_MULTI_ANALYTE.value] == "supported"
        assert (
            result["stan"][CapabilityTag.OBSERVATION_MULTI_ANALYTE.value]
            == "explicitly_unsupported"
        )
        assert (
            result["frem"][CapabilityTag.OBSERVATION_MULTI_ANALYTE.value]
            == "explicitly_unsupported"
        )
