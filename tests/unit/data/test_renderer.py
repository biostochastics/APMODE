# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for apmode.report.renderer HTML export (rc1)."""

from __future__ import annotations

from apmode.bundle.models import (
    ColumnMapping,
    CovariateSpec,
    DataManifest,
    EvidenceManifest,
    ImputationStabilityEntry,
    ImputationStabilityManifest,
    MissingDataDirective,
)
from apmode.report.renderer import render_markdown_to_html, render_run_report


def test_render_markdown_to_html_produces_standalone_document() -> None:
    md = "# Run Report\n\n**Bold**\n\n- item 1\n- item 2\n"
    html = render_markdown_to_html(md, title="Test Run")
    assert html.startswith("<!DOCTYPE html>")
    assert "<title>Test Run</title>" in html
    assert "Run Report" in html
    assert "Bold" in html
    # Must be self-contained (no external references).
    assert "http://" not in html
    assert "</html>" in html


def test_render_markdown_to_html_handles_tables() -> None:
    md = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    html = render_markdown_to_html(md)
    # Rich renders tables as preformatted text (pre or code block).
    assert "1" in html and "2" in html
    assert "<html>" in html and "</html>" in html


def test_render_markdown_to_html_default_title() -> None:
    html = render_markdown_to_html("# hi")
    assert "<title>APMODE Run Report</title>" in html


def test_run_report_includes_missing_data_plan() -> None:
    manifest = DataManifest(
        data_sha256="0" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="NMID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
        ),
        n_subjects=10,
        n_observations=30,
        n_doses=10,
    )
    evidence = EvidenceManifest(
        route_certainty="confirmed",
        absorption_complexity="simple",
        nonlinear_clearance_evidence_strength="none",
        richness_category="moderate",
        identifiability_ceiling="medium",
        covariate_burden=1,
        covariate_correlated=False,
        covariate_missingness=CovariateSpec(
            pattern="MAR",
            fraction_incomplete=0.2,
            strategy="MI-PMM",
        ),
        blq_burden=0.05,
        protocol_heterogeneity="single-study",
        absorption_phase_coverage="adequate",
        elimination_phase_coverage="adequate",
    )
    directive = MissingDataDirective(
        covariate_method="MI-PMM",
        m_imputations=5,
        blq_method="M7+",
        rationale=["Covariate missingness 20.00% within MI-PMM ceiling 30.00%."],
    )
    stability = ImputationStabilityManifest(
        m=5,
        method="MI-PMM",
        entries=[
            ImputationStabilityEntry(
                candidate_id="m1",
                convergence_rate=0.8,
                rank_stability=0.6,
            )
        ],
    )

    report = render_run_report(
        run_id="run_test",
        lane="discovery",
        manifest=manifest,
        evidence=evidence,
        ranked=[],
        missing_data_directive=directive,
        imputation_stability=stability,
    )

    assert "### Missing Data" in report
    assert "| Covariate method | MI-PMM |" in report
    assert "| Imputations | 5 |" in report
    assert "| Minimum MI convergence rate | 80.0% |" in report
