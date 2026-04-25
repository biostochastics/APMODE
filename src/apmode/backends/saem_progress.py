# SPDX-License-Identifier: GPL-2.0-or-later
"""SAEM iteration-progress parser for nlmixr2 stderr output.

The streaming format that nlmixr2 5.x emits during a SAEM fit is::

    params: tka  tcl  tv  V(eta.ka)  V(eta.cl)  V(eta.v)  add.sd
    001: 0.508466  0.889380  3.462163  0.570000  0.285000  0.095000  2.312013
    002: 0.445913  1.052222  3.517213  0.541500  0.270750  0.090250  1.555306
    ...
    060: 0.463163  1.011791  3.460486  0.390666  0.062332  0.022821  0.699135

Header: ``params:`` followed by tab-separated parameter names.
Iteration lines: 3-digit zero-padded iteration number, ``: ``, then a
tab-separated row of float values aligned with the header.

Notes that informed parser design:

* nlmixr2 does **not** print OBJF per iteration (it is computed only at
  the end of SAEM via FOCEi or Gaussian quadrature). The streaming
  signal therefore consists of (iteration, parameter values, phase).
* Phase is implicit: the first ``nBurn`` iterations are simulated-
  annealing burn-in; the next ``nEm`` are the main EM phase. The R
  control object knows the boundary; the parser does not. We surface
  ``phase=None`` until the caller passes ``nburn`` to
  :class:`SAEMLineParser` (which is the case in the runner — it has
  the ``saemControl`` settings on hand).
* nlmixr2 also writes ODE-compilation progress bars
  (``[====|====|...] 0:00:00``) and chatty unicode arrows
  (``→ loading...``) on stderr. Those are not iteration data; the
  parser returns ``None`` for them, and the runner forwards them to
  the raw audit log only.

The parser is designed to **never raise** on arbitrary input — that is
the contract the Hypothesis property test in
``tests/property/test_saem_parser.py`` pins.  Callers can then run the
parser blindly over every line that crosses the stderr pipe and trust
that malformed bytes will produce ``None`` rather than blow up the
backend.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal

# Iteration line: "001: 0.508466\t0.889380\t..."
# Strict form first; fallback strips whitespace if the strict pattern
# misses (e.g. nlmixr2 ever changes the digit padding). The character
# class accepts ``Inf`` / ``-Inf`` / ``inf`` so a divergent SAEM
# iteration whose parameter overflowed (rare but reproducible with
# ill-conditioned data + tight ``itmax``) is still surfaced — the
# parser then rejects the value at float() time and the line drops
# out, but the regex match itself succeeds.
_STRICT_ITER_RE = re.compile(r"^\s*(?P<iter>\d{1,5}):\s*(?P<values>[\-\d\.eE\+\sNAaNnIiFf]+?)\s*$")
_TOLERANT_ITER_RE = re.compile(
    r"^\s*(?P<iter>\d{1,5})\s*[:\-]\s*(?P<values>[\-\d\.eE\+\sNAaNnIiFf]+)\s*$"
)

# Header line: "params: tka\ttcl\t..."
_HEADER_RE = re.compile(r"^\s*params:\s*(?P<names>.+?)\s*$")

# Phase labels surfaced through ``SAEMState.phase``. ``None`` is a
# distinct value meaning "phase boundary not declared by the caller";
# downstream consumers should treat it as "unknown" rather than
# defaulting to one of the named phases.
SAEMPhase = Literal["burnin", "main"]


@dataclass(frozen=True)
class SAEMState:
    """One observation of SAEM progress, parsed from a single stderr line.

    Attributes:
        iteration: 1-based iteration counter as printed by nlmixr2.
        param_names: Names from the most recent ``params:`` header
            (empty tuple if the parser hasn't seen one yet — useful when
            a stream is replayed mid-fit).
        param_values: Float values aligned with ``param_names``. NaN
            tokens (``"NA"`` / ``"NaN"``) round-trip as ``float('nan')``.
        phase: ``"burnin"`` for iters 1..nBurn, ``"main"`` for nBurn+1
            onward, ``None`` when ``nBurn`` was not provided to the
            parser.
        objf: Optional objective-function value. nlmixr2 SAEM does not
            stream OBJF per iteration; this field exists so a future
            backend (or post-fit summary line) can populate it without
            a schema change.
        timestamp: Unix epoch seconds when the line was parsed. Useful
            for both wall-clock progress display and NDJSON envelopes.
    """

    iteration: int
    param_names: tuple[str, ...] = field(default_factory=tuple)
    param_values: tuple[float, ...] = field(default_factory=tuple)
    phase: SAEMPhase | None = None
    objf: float | None = None
    timestamp: float = field(default_factory=time.time)


class SAEMLineParser:
    """Stateful line-by-line SAEM progress parser.

    The parser remembers the last ``params:`` header so iteration
    lines can be paired with parameter names. Lines that don't match
    either the header or the iteration pattern return ``None`` —
    callers tee them to a raw log untouched.

    Args:
        nburn: Optional burn-in iteration count from ``saemControl``.
            When provided, iterations 1..nburn carry ``phase="burnin"``
            and the rest ``phase="main"``. When omitted, ``phase`` is
            ``None`` — explicitly "unknown", not a default.
    """

    def __init__(self, *, nburn: int | None = None) -> None:
        self._nburn = nburn
        self._param_names: tuple[str, ...] = ()
        # ``last_iter`` lets a caller compute ETA / iters-per-second
        # without the parser growing a clock dependency.
        self._last_iter: int | None = None

    @property
    def param_names(self) -> tuple[str, ...]:
        return self._param_names

    def parse(self, line: str | bytes) -> SAEMState | None:
        """Return a :class:`SAEMState` if ``line`` is an iteration line.

        ``line`` may be either ``str`` or ``bytes``. ``bytes`` are
        decoded as UTF-8 with replacement to keep the parser
        non-raising even for truly garbage input. Returns ``None``
        for header lines, blank lines, and anything else.
        """
        text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
        text = text.rstrip("\r\n")
        if not text:
            return None

        # Header takes precedence — many runs never repeat it but the
        # parser must not interpret a header as iteration data.
        header = _HEADER_RE.match(text)
        if header is not None:
            self._capture_header(header.group("names"))
            return None

        match = _STRICT_ITER_RE.match(text) or _TOLERANT_ITER_RE.match(text)
        if match is None:
            return None

        try:
            iteration = int(match.group("iter"))
        except (TypeError, ValueError):
            return None
        if iteration <= 0 or iteration > 1_000_000:
            # Keeps a stray "999999999: ..." cosmic-ray line from
            # producing a state that sentinels never hit.
            return None

        values = self._parse_values(match.group("values"))
        if not values and not self._param_names:
            # Both empty → no useful payload. ``params:`` not yet seen
            # AND the iteration row had no parseable numbers; return
            # None so the runner does not emit a hollow event.
            return None
        if self._param_names and not values:
            # Header was previously captured but the values column
            # is unparseable on this line (e.g. a stray diagnostic
            # row that the tolerant regex matched but ``_parse_values``
            # rejected). Returning a hollow ``SAEMState`` would
            # silently break ``zip(param_names, param_values)``
            # downstream — drop the line instead.
            return None
        if self._param_names and len(values) != len(self._param_names):
            # Column-count mismatch is the same hazard: an iteration
            # row with the wrong arity points to a parser drift, not
            # a real iteration. Surfacing it as ``None`` keeps the
            # invariant that emitted states have aligned arrays.
            return None

        phase = self._classify_phase(iteration)
        self._last_iter = iteration
        return SAEMState(
            iteration=iteration,
            param_names=self._param_names,
            param_values=tuple(values),
            phase=phase,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _capture_header(self, names_field: str) -> None:
        # nlmixr2 separates names with TAB but a future version could
        # use whitespace; ``split`` with no argument handles both.
        parts = tuple(piece.strip() for piece in names_field.split() if piece.strip())
        if parts:
            self._param_names = parts

    def _classify_phase(self, iteration: int) -> SAEMPhase | None:
        if self._nburn is None:
            return None
        return "burnin" if iteration <= self._nburn else "main"

    @staticmethod
    def _parse_values(values_field: str) -> list[float]:
        out: list[float] = []
        for token in values_field.split():
            try:
                # ``float`` natively handles "1.5", "-2e-3", "inf",
                # "Inf", "-Inf", but not "NA" — translate that case
                # explicitly so an NA cell round-trips as NaN.
                if token in {"NA", "NaN", "nan"}:
                    out.append(float("nan"))
                else:
                    out.append(float(token))
            except ValueError:
                # A non-numeric token in the values column means the
                # line wasn't really an iteration row; abandon what we
                # collected so the caller sees an empty list.
                return []
        return out


__all__ = ["SAEMLineParser", "SAEMPhase", "SAEMState"]
