# SPDX-License-Identifier: GPL-2.0-or-later
"""Backend error taxonomy per ARCHITECTURE.md §4.1."""


class BackendError(Exception):
    """Base error for all backend failures."""


class ConvergenceError(BackendError):
    """Estimation algorithm did not converge."""

    def __init__(
        self,
        message: str,
        *,
        method: str | None = None,
        iterations: int | None = None,
        gradient_norm: float | None = None,
    ) -> None:
        super().__init__(message)
        self.method = method
        self.iterations = iterations
        self.gradient_norm = gradient_norm


class BackendTimeoutError(BackendError):
    """Backend process exceeded its time budget.

    Named to avoid shadowing ``builtins.TimeoutError`` (which asyncio raises
    from ``wait_for``) at callsites that import from this module.
    """

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: int | None = None,
        pid: int | None = None,
    ) -> None:
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.pid = pid


class CrashError(BackendError):
    """Backend process crashed (segfault, OOM, unexpected exit)."""

    def __init__(
        self,
        message: str,
        *,
        exit_code: int | None = None,
        stderr_tail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail


class AgenticExhaustionError(BackendError):
    """Agentic loop completed all iterations without a converged result.

    Distinct from transient ``BackendError`` failures (network, crash,
    timeout) so the orchestrator can surface "LLM could not find a valid
    model in N iterations" as a meaningful outcome rather than a
    generic transient failure.
    """

    def __init__(
        self,
        message: str,
        *,
        iterations: int | None = None,
    ) -> None:
        super().__init__(message)
        self.iterations = iterations


class APMODEConfigError(Exception):
    """Configuration loading failed (missing or malformed policy files).

    M4: raised by modules whose import triggers config loading — wrapping
    ``FileNotFoundError`` / ``ValidationError`` in a typed exception with
    the resolved path gives end users an actionable message instead of a
    bare traceback through unrelated internals.
    """

    def __init__(self, message: str, *, resolved_path: str | None = None) -> None:
        super().__init__(message)
        self.resolved_path = resolved_path


class InvalidSpecError(BackendError):
    """DSL spec failed validation before backend dispatch."""

    def __init__(
        self,
        message: str,
        *,
        spec_id: str | None = None,
        violations: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.spec_id = spec_id
        self.violations = violations or []
