# SPDX-License-Identifier: GPL-2.0-or-later
"""APMODE — Adaptive Pharmacokinetic Model Discovery Engine."""

__version__: str
try:
    from apmode._version import __version__
except ModuleNotFoundError:  # editable install without build
    __version__ = "0.6.0-rc1"
