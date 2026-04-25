# SPDX-License-Identifier: GPL-2.0-or-later
"""Runtime-vs-declared version drift detection for ``apmode --version``.

APMODE has two version notions that disagree during development:

* **Declared**: the milestone version users see on the README badge and in
  ``CHANGELOG.md`` (most recent ``## [X.Y.Z]`` header). Sourced by
  ``scripts/sync_readme.py::collect_version`` for marker substitution. This is
  the "what release am I building toward" identity.
* **Runtime**: whatever ``importlib.metadata.version("apmode")`` reports —
  derived from VCS tags by ``hatch-vcs`` at build time. In an editable install
  ahead of the next release tag this looks like
  ``0.3.0rc4.dev77+gfcf87e16e.d20260425``. This is the "what bytes are
  installed" identity.

When the two agree (after PEP-440 normalisation, ignoring the ``+local``
segment), ``apmode --version`` prints a single line::

    apmode 0.6.0-rc1

When they disagree, both are surfaced so the operator can correlate the
installed bits to a tag::

    apmode 0.6.0-rc1 (runtime 0.3.0rc4.dev77+gfcf87e16e.d20260425)

This is intentional: drift is normal during pre-tag development. The CLI
shows it rather than hiding it; once a tag is cut, ``hatch-vcs`` regenerates
``_version.py`` and the values realign automatically.
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version
from pathlib import Path

from packaging.version import InvalidVersion, Version

# CHANGELOG.md header form: ``## [0.6.0-rc1] — 2026-04-24`` (the trailing date
# is decorative and varies). The pre-release suffix accepts both ``-`` and
# ``.`` separators with one or more dot-segmented identifiers — covers
# ``-rc1``, ``-rc1.2``, ``.dev5``, ``.post1``, etc. PEP 440 normalises the
# rendering when a ``packaging.version.Version`` is built.
_CHANGELOG_HEADER_RE = re.compile(
    r"^##\s+\[(?P<v>[0-9]+\.[0-9]+\.[0-9]+(?:[.\-][a-z0-9]+)*)\]",
    re.M,
)

# In editable installs the package is at ``<repo>/src/apmode/_version_drift.py``,
# so ``parents[2]`` is the repo root that contains ``CHANGELOG.md``. In wheels
# this path will not contain CHANGELOG.md and ``collect_declared_version``
# returns None — the right answer for a packaged install.
_REPO_ROOT_FROM_HERE = Path(__file__).resolve().parents[2]

_RUNTIME_UNKNOWN = "unknown"


def collect_runtime_version(dist_name: str = "apmode") -> str:
    """Return the runtime version, ``"unknown"`` if it cannot be determined.

    Resolution order:

    1. ``importlib.metadata.version(dist_name)`` — the authoritative source
       once the package is installed (wheel, sdist, or ``pip install -e .``).
    2. ``apmode._version.__version__`` — the file ``hatch-vcs`` writes during
       build. Present in editable installs, missing in source checkouts that
       have not been built.
    3. ``"unknown"`` — neither source resolved.
    """
    try:
        return _metadata_version(dist_name)
    except PackageNotFoundError:
        pass
    try:
        from apmode._version import __version__

        return __version__
    except (ModuleNotFoundError, ImportError):
        return _RUNTIME_UNKNOWN


def collect_declared_version(repo_root: Path = _REPO_ROOT_FROM_HERE) -> str | None:
    """Return the most recent ``## [X.Y.Z]`` header from ``CHANGELOG.md``.

    Returns ``None`` when the changelog is unreadable (typical in a wheel
    install that does not ship ``CHANGELOG.md`` alongside ``src/``). The CLI
    treats ``None`` as "can't compare; just show runtime", which is the right
    UX for users who installed via ``pip install apmode``.
    """
    changelog = repo_root / "CHANGELOG.md"
    try:
        text = changelog.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    m = _CHANGELOG_HEADER_RE.search(text)
    return m.group("v") if m else None


def is_drifted(declared: str | None, runtime: str) -> bool:
    """True iff declared and runtime disagree after PEP-440 canonicalisation.

    The local segment (``+gHASH.dDATE``) is dropped before comparison — it is
    pure provenance, not part of release identity. The dev segment (``.devN``)
    IS retained, so a build downstream of the most-recent tag is reported as
    drifted relative to a tagged-release declaration. That is the desired
    behaviour: a ``0.6.0rc1.dev5`` install is *not* the same artefact as the
    eventual ``0.6.0rc1`` tag, and the operator should see that.

    Returns ``False`` when ``declared is None`` (cannot determine drift) and
    ``True`` when ``runtime == "unknown"`` (declared is known but runtime is
    not — definitionally drift).
    """
    if declared is None:
        return False
    if runtime == _RUNTIME_UNKNOWN:
        return True

    try:
        d = Version(declared)
        r = Version(runtime)
    except InvalidVersion:
        # Either side is non-PEP-440; fall back to lexical compare so we
        # still flag obvious mismatches without raising.
        return declared != runtime

    # ``Version.public`` strips the ``+local`` provenance segment from
    # *each* side. A symmetric strip avoids false drift in the (unusual
    # but valid) case where the declared CHANGELOG header itself carried
    # a ``+local`` annotation. Rebuilding through ``Version`` keeps the
    # comparison structural rather than string-based.
    return Version(d.public) != Version(r.public)


def format_version_line(*, declared: str | None, runtime: str) -> str:
    """Render the ``apmode --version`` first line.

    * Aligned (or declared unknown): ``apmode <single-version>``
    * Drifted: ``apmode <declared> (runtime <runtime>)`` — the runtime string
      is preserved verbatim, including ``+gHASH.dDATE``, so the operator has
      the exact provenance handle.
    """
    if declared is None:
        return f"apmode {runtime}"
    if not is_drifted(declared, runtime):
        return f"apmode {declared}"
    return f"apmode {declared} (runtime {runtime})"
