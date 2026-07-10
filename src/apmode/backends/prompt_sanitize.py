# SPDX-License-Identifier: GPL-2.0-or-later
"""Shared LLM-prompt sanitization helpers.

Extracted from ``agentic_runner.py`` so ``diagnostic_summarizer.py``
(imported BY ``agentic_runner.py``) can also sanitize data-derived
strings -- e.g. raw CSV covariate column names -- before they reach the
LLM prompt, without creating an import cycle.
"""

from __future__ import annotations

import re

_ROLE_MARKER_RE = re.compile(
    r"""
    (?im)
    (?:^|[\r\n])              # start of string OR any newline (CR / LF / CRLF)
    \s*                        # optional indent
    (?:system|user|assistant)  # role marker
    \s*:\s*                    # ``role:`` separator
    """,
    re.VERBOSE,
)


def sanitize_for_prompt(text: str, max_len: int = 500) -> str:
    """Strip patterns that could manipulate the LLM via injected text.

    Backend error messages (e.g., from R or nlmixr2) and data-derived
    strings (e.g., raw CSV covariate column names) are embedded as
    content when relaying context to the LLM. A hostile or unusual
    string could contain markdown code fences or role-marker sequences
    that the LLM would interpret as instructions. This helper truncates
    the string and escapes obvious code-fence / role-marker sequences.

    The role-marker pattern fires on any in-string ``\\n`` followed by
    ``role:`` -- not just at line-start -- so a single-line payload
    containing ``\\n\\nsystem: ignore previous instructions`` is still
    neutered.
    """
    if not text:
        return ""
    # Remove triple backticks (code fences) that could terminate our own fence
    cleaned = text.replace("```", "⁣``⁣`⁣")
    # Collapse any lines that look like role markers -- including those
    # smuggled in via embedded ``\n`` inside a single payload string.
    cleaned = _ROLE_MARKER_RE.sub("\n", cleaned)
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + f"… [truncated, {len(text) - max_len} chars]"
    return cleaned
