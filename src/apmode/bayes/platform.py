# SPDX-License-Identifier: GPL-2.0-or-later
"""Platform-adaptive cmdstanpy kwargs.

Rationale
---------
cmdstanpy exposes three knobs that interact with the host platform:

* ``force_one_process_per_chain`` (keyword-only in ``CmdStanModel.sample``):
  when ``True`` cmdstanpy spawns one OS process per chain rather than running
  them in-process. This is roughly 3-4x slower on POSIX, but is the only
  reliable configuration on Windows where the MinGW toolchain has a
  long-standing ``thread_local`` performance bug
  (cmdstanpy issue #895).
* ``cpp_options["STAN_THREADS"]``: enables Stan's thread-based within-chain
  parallelism. Has no effect unless the Stan program actually uses
  ``reduce_sum`` / ``map_rect``, and can interact poorly with forked children
  on platforms that don't like ``fork`` + threads (cmdstanpy issue #780).
* ``save_cmdstan_config`` (not set here; set at the call-site alongside the
  output CSVs): required so a fit can be reconstructed from disk without the
  original Python ``CmdStanModel`` instance (cmdstanpy issue #848).

This helper encodes the safe defaults so each backend doesn't have to
re-derive them. Confirmed against cmdstanpy ``model.py`` develop branch
(2026-01) — the keyword is ``force_one_process_per_chain``, not
``one_process_per_chain``; calling with the wrong name raises ``TypeError``.
"""

from __future__ import annotations

import platform
from typing import Any


def cmdstan_run_kwargs(*, uses_reduce_sum: bool) -> dict[str, Any]:
    """Return sampler kwargs appropriate for the current host platform.

    Args:
        uses_reduce_sum: Whether the Stan program relies on ``reduce_sum``
            (or ``map_rect``) for within-chain parallelism. Set this based
            on a static scan of the emitted Stan program text.

    Returns:
        A mapping that can be spread into ``CmdStanModel.sample(**kw)``.
        Always contains ``force_one_process_per_chain`` (a required
        keyword-only argument); may additionally contain ``cpp_options``
        with ``STAN_THREADS`` on non-Windows hosts when
        ``uses_reduce_sum`` is true.
    """
    sys_name = platform.system()
    if sys_name == "Windows":
        # Windows + STAN_THREADS triggers the cmdstanpy #895 TBB/thread_local
        # bug; the only safe configuration is one process per chain with
        # threading disabled, regardless of ``uses_reduce_sum``.
        return {"force_one_process_per_chain": True}

    kw: dict[str, Any] = {"force_one_process_per_chain": False}
    if uses_reduce_sum:
        kw["cpp_options"] = {"STAN_THREADS": True}
    return kw
