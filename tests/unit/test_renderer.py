# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for apmode.report.renderer HTML export (rc1)."""

from __future__ import annotations

from apmode.report.renderer import render_markdown_to_html


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
