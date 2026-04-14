# SPDX-License-Identifier: GPL-2.0-or-later
"""Structured logging configuration (ARCHITECTURE.md §2.8).

JSON-structured logs via structlog, context-bound with run_id, candidate_id,
gate. Logs are operational mirrors — the bundle is the source of truth.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(*, json_output: bool = True, level: int = logging.INFO) -> None:
    """Configure structlog for APMODE.

    Args:
        json_output: True for JSON lines, False for human-readable console.
        level: Logging level (default INFO).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a context-bound structlog logger."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
