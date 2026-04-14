# SPDX-License-Identifier: GPL-2.0-or-later
"""nlmixr2 backend runner via R subprocess (ARCHITECTURE.md S4.2).

Spawns Rscript as a child process, communicates via JSON files:
  request.json  -> R reads, runs nlmixr2, writes ->  response.json

Exit codes: 0=success, 1=R error, 137=killed (SIGKILL), 139=segfault (SIGSEGV).

Security: Uses asyncio.create_subprocess_exec (not shell=True) to prevent
shell injection. All arguments are passed as a list, not interpolated.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING

from apmode.backends.r_schemas import RSubprocessRequest, RSubprocessResponse
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2
from apmode.errors import BackendTimeoutError, ConvergenceError, CrashError
from apmode.ids import generate_run_id

if TYPE_CHECKING:
    from apmode.bundle.models import BackendResult, DataManifest
    from apmode.dsl.ast_models import DSLSpec

# Location of the R harness script, relative to this package
_DEFAULT_HARNESS = Path(__file__).parent.parent / "r" / "harness.R"


class Nlmixr2Runner:
    """BackendRunner implementation for nlmixr2 via R subprocess.

    Communication is file-based (not stdout) to avoid R startup message
    contamination. Each run creates a unique work directory under work_dir.
    """

    def __init__(
        self,
        work_dir: Path,
        r_executable: str = "Rscript",
        harness_path: Path | None = None,
        estimation: list[str] | None = None,
    ) -> None:
        self.work_dir = work_dir
        self.r_executable = r_executable
        self.harness_path = harness_path or _DEFAULT_HARNESS
        self.estimation = estimation or ["saem", "focei"]

    async def run(
        self,
        spec: DSLSpec,
        data_manifest: DataManifest,
        initial_estimates: dict[str, float],
        seed: int,
        timeout_seconds: int | None = None,
        *,
        data_path: Path | None = None,
        split_manifest: dict[str, object] | None = None,
    ) -> BackendResult:
        """Run nlmixr2 estimation via R subprocess.

        Creates a unique work directory, writes request.json with the compiled
        R code, spawns Rscript, and reads response.json.

        Args:
            data_path: Absolute path to the nlmixr2-ready CSV file. Required.

        Raises:
            BackendTimeoutError: If R process exceeds timeout.
            CrashError: If R process crashes or produces no response.
            ConvergenceError: If nlmixr2 reports convergence failure.
            ValueError: If data_path is not provided.
        """
        if data_path is None:
            msg = "data_path is required for Nlmixr2Runner"
            raise ValueError(msg)

        request_id = generate_run_id()
        run_dir = self.work_dir / request_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Compile R code from DSL spec
        compiled_r_code = emit_nlmixr2(spec, initial_estimates=initial_estimates)

        # Build and write request
        request = RSubprocessRequest(
            schema_version="1.0",
            request_id=request_id,
            run_id=request_id,
            candidate_id=spec.model_id,
            spec=spec,
            data_path=str(data_path),
            seed=seed,
            rng_kind="L'Ecuyer-CMRG",
            initial_estimates=initial_estimates,
            estimation=self.estimation,
            compiled_r_code=compiled_r_code,
            split_manifest=split_manifest,
        )

        request_path = run_dir / "request.json"
        request_path.write_text(request.model_dump_json(indent=2))

        # Spawn R subprocess
        response_path = run_dir / "response.json"
        exit_code = await self._spawn_r(request_path, response_path, timeout_seconds)

        # Parse response
        return self._parse_response(response_path, exit_code, spec.model_id)

    async def _spawn_r(
        self,
        request_path: Path,
        response_path: Path,
        timeout_seconds: int | None,
    ) -> int:
        """Spawn Rscript subprocess. Returns exit code.

        Uses create_subprocess_exec (not shell) for security.
        Creates a new process group via os.setsid for clean timeout kill.
        """
        proc = await asyncio.create_subprocess_exec(
            self.r_executable,
            str(self.harness_path),
            str(request_path),
            str(response_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=os.setsid,
        )

        try:
            _stdout, _stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except TimeoutError:
            # Kill the entire process group
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            await proc.wait()
            raise BackendTimeoutError(
                f"R subprocess timed out after {timeout_seconds}s",
                timeout_seconds=timeout_seconds,
                pid=proc.pid,
            ) from None

        return proc.returncode or 0

    def _parse_response(
        self,
        response_path: Path,
        exit_code: int,
        model_id: str,
    ) -> BackendResult:
        """Parse response.json and map to BackendResult or raise error."""
        from apmode.bundle.models import BackendResult

        # No response file means crash
        if not response_path.exists():
            raise CrashError(
                f"R subprocess exited with code {exit_code} and no response.json",
                exit_code=exit_code,
            )

        raw = json.loads(response_path.read_text())
        response = RSubprocessResponse.model_validate(raw)

        if response.status == "error":
            if response.error_type == "convergence":
                raise ConvergenceError(
                    f"nlmixr2 convergence failure for {model_id}",
                    method=None,
                )
            raise CrashError(
                f"R backend error ({response.error_type}) for {model_id}",
                exit_code=exit_code,
            )

        # Success: parse result dict into BackendResult
        if response.result is None:
            raise CrashError(
                "R subprocess returned success but no result payload",
                exit_code=exit_code,
            )

        return BackendResult.model_validate(response.result)
