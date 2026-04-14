# SPDX-License-Identifier: GPL-2.0-or-later
"""Root conftest: JAX fork-safety and shared fixtures.

JAX initializes a POSIX thread pool on first import. If os.fork() is called
after that (e.g., by the nlmixr2 R subprocess runner), the forked child
inherits dead mutex state, causing a potential deadlock.

Fix: set XLA_FLAGS to disable multi-threading before JAX loads, and
suppress the RuntimeWarning in the test suite. In production, the
orchestrator runs R subprocesses BEFORE dispatching to the NODE backend,
so the ordering is safe.
"""

from __future__ import annotations

import os
import warnings

# Tell XLA/JAX to use a single thread — prevents the fork deadlock.
# This must happen before any `import jax` in the test process.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Suppress the fork warning in tests (we've mitigated it above)
warnings.filterwarnings(
    "ignore",
    message="os.fork\\(\\) was called.*JAX is multithreaded",
    category=RuntimeWarning,
)
