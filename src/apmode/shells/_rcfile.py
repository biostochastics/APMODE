# SPDX-License-Identifier: GPL-2.0-or-later
"""Shared rc-file machinery for the shell strategies.

Everything that needs to know how a marker block is laid out, written
atomically, or removed without disturbing the surrounding rc-file bytes
lives here. The shell strategies consume this module; they never touch
the filesystem directly. That keeps the byte-level safety properties
(atomic write, single ``.bak`` per session, version-aware update) in one
place that is exhaustively unit-tested.

Marker shape — every block looks like::

    <prelude bytes...>
    # >>> apmode completion >>>            ← MARKER_OPEN
    # version: 1                            ← schema signature line
    <generated completion source>
    # <<< apmode completion <<<            ← MARKER_CLOSE
    <postlude bytes...>

The ``# version:`` line is what makes idempotent re-install fast: when a
re-install renders the same body and signature, nothing is written. When
the signature differs (operator upgraded APMODE) the block is rewritten
in place, but the prelude / postlude bytes are preserved exactly.

Atomic writes — every write goes through a sibling ``<file>.tmp`` and
``os.replace``. That guarantees the rc file is either entirely the old
content or entirely the new content — never a half-written state that
could break shell startup. The temp file is created in the same
directory so the rename is atomic on POSIX (cross-device renames would
fall back to copy+unlink which is *not* atomic).

One-shot ``.bak`` — the first time we write a given file in this
process, we copy the current bytes to ``<file>.bak``. Subsequent writes
within the same Python process do not re-create it. That keeps test
runs from spamming backup files while still leaving the operator one
recoverable copy of whatever was there before.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — runtime Path() used by atomic_write tmp swap

from apmode.shells import COMPLETION_SCHEMA, MARKER_CLOSE, MARKER_OPEN

# Pattern that matches the entire marker block, capturing the body so
# callers can inspect it for version drift. ``re.DOTALL`` is required so
# ``.`` spans newlines inside the body.
_BLOCK_RE = re.compile(
    rf"^{re.escape(MARKER_OPEN)}\n(?P<body>.*?)\n{re.escape(MARKER_CLOSE)}\n?",
    re.DOTALL | re.MULTILINE,
)

# Per-process bookkeeping of which files have already been backed up.
# Sets are sufficient: a ``Path`` round-trips through ``str(path)`` so
# resolving the absolute path before insertion is enough to deduplicate
# across symlink variants.
_BACKED_UP: set[str] = set()


@dataclass(frozen=True)
class BlockUpdate:
    """Outcome of :func:`upsert_block`.

    ``action`` carries the change classification consumed by the
    :class:`InstallResult` envelope:

    * ``"installed"`` — the block was absent before; it was appended.
    * ``"already_installed"`` — the block existed and matched bytes-for-bytes.
    * ``"updated"`` — the block existed but with a different version
      signature; it was rewritten in place.
    """

    new_text: str
    action: str


@dataclass(frozen=True)
class BlockRemoval:
    """Outcome of :func:`remove_block`.

    * ``"uninstalled"`` — a block was present and was removed.
    * ``"absent"`` — no block was present; ``new_text == old text``.
    """

    new_text: str
    action: str


def _render_block(body: str) -> str:
    """Wrap ``body`` between the open / close markers with a version line.

    The body itself is the verbatim shell-specific completion source. The
    version-signature comment lives between the open marker and the body
    so :func:`upsert_block` can detect version drift without parsing the
    completion source.
    """
    return f"{MARKER_OPEN}\n# version: {COMPLETION_SCHEMA}\n{body}\n{MARKER_CLOSE}\n"


def upsert_block(text: str, body: str) -> BlockUpdate:
    """Insert or refresh the apmode block in ``text``.

    The function is idempotent across (a) verbatim re-install (no change),
    (b) version bump (rewrite block), and (c) first-time install (append).
    Bytes outside the marked block are preserved exactly.
    """
    rendered = _render_block(body)
    match = _BLOCK_RE.search(text)
    if match is None:
        # First-time install. Make sure the file ends with a newline so
        # the appended block does not glue itself onto whatever the
        # operator's last line was.
        prefix = text if text.endswith("\n") or text == "" else text + "\n"
        if prefix and not prefix.endswith("\n\n"):
            # One blank line of breathing room before our block, so the
            # rc file stays readable when an operator opens it.
            prefix += "\n"
        return BlockUpdate(new_text=prefix + rendered, action="installed")

    existing = match.group(0)
    if existing == rendered:
        return BlockUpdate(new_text=text, action="already_installed")

    # Replace the matched span with the new rendering. ``re.sub`` would
    # also work but slicing avoids re-scanning the whole text and keeps
    # the replacement length-stable in Python's mind.
    new_text = text[: match.start()] + rendered + text[match.end() :]
    return BlockUpdate(new_text=new_text, action="updated")


def remove_block(text: str) -> BlockRemoval:
    """Delete the apmode block; surrounding bytes preserved.

    A trailing blank line introduced by :func:`upsert_block` is also
    cleaned up so an install/uninstall cycle is round-trip safe up to a
    *single* trailing newline. (If the original file ended without a
    newline, the cycle leaves one trailing ``\\n`` — :func:`upsert_block`
    must end the prefix with a newline so the appended block parses
    cleanly. This is a one-byte difference; full byte-for-byte recovery
    of a no-trailing-newline rc file is left to the operator.)
    """
    match = _BLOCK_RE.search(text)
    if match is None:
        return BlockRemoval(new_text=text, action="absent")
    new_text = text[: match.start()] + text[match.end() :]
    # Collapse the leading blank line we may have introduced on install.
    # We only collapse a single blank line so an operator's existing
    # multi-line spacing is preserved.
    if new_text.endswith("\n\n"):
        new_text = new_text[:-1]
    return BlockRemoval(new_text=new_text, action="uninstalled")


def read_rc(path: Path) -> str:
    """Return the file contents, or ``""`` if the file does not exist."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (temp file + ``os.replace``).

    The temp filename includes ``os.getpid()`` so two ``apmode completion
    install`` invocations running concurrently do not race on a shared
    ``<file>.tmp`` and clobber each other's writes. ``os.replace`` is
    atomic on POSIX so the destination file is either entirely the old
    content or entirely the new content — never a half-written state.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        # If os.replace raised mid-flight, leave the world clean rather
        # than dropping a stale ``<file>.tmp.PID`` next to the operator's
        # rc file. The unlink is best-effort: if tmp was already
        # consumed by os.replace, ``missing_ok=True`` keeps it silent.
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()


def backup_once(path: Path) -> Path | None:
    """Copy ``path`` to ``<path>.bak`` once per process. Returns the .bak path
    when a copy was made, ``None`` when the path is missing or already backed up.

    A ``FileNotFoundError`` between the existence check and the copy
    (file deleted by another process / dotfile manager / sync daemon)
    is treated identically to the "no file to back up" case — the
    operator sees a clean install rather than a Python traceback.
    """
    abs_str = str(path.resolve())
    if abs_str in _BACKED_UP:
        return None
    if not path.exists():
        # Track it so we don't try again, but there is nothing to copy.
        _BACKED_UP.add(abs_str)
        return None
    bak = path.with_name(path.name + ".bak")
    try:
        shutil.copy2(path, bak)
    except FileNotFoundError:
        # TOCTOU: file disappeared between exists() and copy2.
        _BACKED_UP.add(abs_str)
        return None
    _BACKED_UP.add(abs_str)
    return bak


def reset_backup_state_for_tests() -> None:
    """Test hook: clear the per-process backup tracker.

    Tests invoke ``install`` against tmp_path rc files; clearing the
    tracker between tests guarantees the ``.bak`` copy semantics are
    exercised consistently rather than depending on test ordering.
    """
    _BACKED_UP.clear()


__all__ = [
    "BlockRemoval",
    "BlockUpdate",
    "atomic_write",
    "backup_once",
    "read_rc",
    "remove_block",
    "reset_backup_state_for_tests",
    "upsert_block",
]
