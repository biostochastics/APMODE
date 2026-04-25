# SPDX-License-Identifier: GPL-2.0-or-later
"""HTTP API surface for APMODE (plan Block 3 — Tasks 31-37).

Optional install — requires the ``[api]`` extra (fastapi + uvicorn +
aiosqlite). Importing :mod:`apmode.api.store` does not pull FastAPI in;
``store.py`` only depends on ``aiosqlite`` so that the run registry can
be exercised by unit tests without the full server stack.
"""
