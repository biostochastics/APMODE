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
import logging
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from apmode.backends.predictive_summary import (
    SubjectSimulation,
    build_predictive_diagnostics,
)
from apmode.backends.r_schemas import (
    PredictedSimulationsSubject,
    RSubprocessRequest,
    RSubprocessResponse,
)
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2
from apmode.errors import BackendTimeoutError, ConvergenceError, CrashError
from apmode.ids import generate_run_id

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from apmode.bundle.models import BackendResult, DataManifest, NCASubjectDiagnostic
    from apmode.dsl.ast_models import DSLSpec
    from apmode.governance.policy import Gate3Config

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
        compiled_code_override: str | None = None,
        gate3_policy: Gate3Config | None = None,
        nca_diagnostics: list[NCASubjectDiagnostic] | None = None,
    ) -> BackendResult:
        """Run nlmixr2 estimation via R subprocess.

        Creates a unique work directory, writes request.json with the compiled
        R code, spawns Rscript, and reads response.json.

        Args:
            data_path: Absolute path to the nlmixr2-ready CSV file. Required.
            compiled_code_override: Optional pre-compiled R model code to use
                instead of ``emit_nlmixr2(spec)``. Used by the FREM path
                where the model is produced by ``emit_nlmixr2_frem`` with
                augmented covariate endpoints. When provided, ``spec``
                still supplies ``model_id`` and ``data_manifest`` for
                metadata, but its AST is not re-emitted.
            gate3_policy: When provided, requests
                ``n_posterior_predictive_sims`` simulation replicates
                from the R harness and populates
                ``diagnostics.{vpc,npe_score,auc_cmax_be_score,
                auc_cmax_source}`` via
                :func:`apmode.backends.predictive_summary.
                build_predictive_diagnostics`. When ``None`` (default),
                simulation is skipped and Gate 3 ranking falls back to
                the CWRES NPE proxy.
            nca_diagnostics: Per-subject NCA QC records (from the
                orchestrator's initial-estimates stage). Used as the
                per-subject NCA-eligibility mask for AUC/Cmax BE
                scoring. Subjects without a matching record are treated
                as ineligible (mask-dropped).

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

        # Compile R code from DSL spec (unless the caller pre-compiled, e.g.
        # the FREM path).
        compiled_r_code = (
            compiled_code_override
            if compiled_code_override is not None
            else emit_nlmixr2(spec, initial_estimates=initial_estimates)
        )

        n_sims_req = (
            int(gate3_policy.n_posterior_predictive_sims) if gate3_policy is not None else 0
        )

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
            n_posterior_predictive_sims=n_sims_req,
        )

        request_path = run_dir / "request.json"
        request_path.write_text(request.model_dump_json(indent=2))

        # Spawn R subprocess
        response_path = run_dir / "response.json"
        exit_code = await self._spawn_r(request_path, response_path, timeout_seconds)

        # Parse response + optionally enrich with simulation-based diagnostics
        return self._parse_response(
            response_path,
            exit_code,
            spec.model_id,
            gate3_policy=gate3_policy,
            nca_diagnostics=nca_diagnostics,
        )

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
        *,
        gate3_policy: Gate3Config | None = None,
        nca_diagnostics: list[NCASubjectDiagnostic] | None = None,
    ) -> BackendResult:
        """Parse response.json and map to BackendResult or raise error.

        When ``gate3_policy`` is supplied, any ``predicted_simulations``
        payload emitted by the R harness is atomically converted into
        VPC/NPE/AUC-Cmax diagnostics via
        :func:`apmode.backends.predictive_summary.build_predictive_diagnostics`
        and merged onto the returned :class:`BackendResult`. A simulation
        failure on the R side (``predicted_simulations = NULL``) is
        non-fatal: the BackendResult is returned without simulation-based
        diagnostics and Gate 3 falls back to the CWRES proxy.
        """
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

        # Strip out-of-band predicted_simulations before BackendResult
        # validation — the field is consumed here and not part of the
        # BackendResult schema.
        result_dict = dict(response.result)
        predicted_sims_raw = result_dict.pop("predicted_simulations", None)
        backend_result = BackendResult.model_validate(result_dict)

        if gate3_policy is None:
            # Caller did not request simulation-based diagnostics — silent
            # fallback is the contract.
            return backend_result

        if (
            predicted_sims_raw is None
            or not isinstance(predicted_sims_raw, list)
            or not predicted_sims_raw
        ):
            # A Gate3Config was supplied but no sims arrived (R rxSolve
            # tryCatch swallowed the error in harness.R). This is a
            # non-fatal audit-trail event — log structured so the bundle
            # has provenance for why Gate 3 fell back to the CWRES proxy.
            logger.warning(
                "nlmixr2 posterior-predictive simulation absent for model %s "
                "(n_posterior_predictive_sims=%d requested); Gate 3 will use "
                "the CWRES NPE proxy and uniform-drop the AUC/Cmax component.",
                model_id,
                gate3_policy.n_posterior_predictive_sims,
            )
            return backend_result

        diagnostics_by_subject = {d.subject_id: d for d in (nca_diagnostics or [])}
        subject_sims: list[SubjectSimulation] = []
        for entry in predicted_sims_raw:
            payload = PredictedSimulationsSubject.model_validate(entry)
            sims_matrix = np.array(payload.sims_at_observed, dtype=float)
            subject_sims.append(
                SubjectSimulation(
                    subject_id=payload.subject_id,
                    t_observed=np.array(payload.t_observed, dtype=float),
                    observed_dv=np.array(payload.observed_dv, dtype=float),
                    sims_at_observed=sims_matrix,
                    nca_diagnostic=diagnostics_by_subject.get(payload.subject_id),
                )
            )

        if not subject_sims:
            return backend_result

        predictive = build_predictive_diagnostics(subject_sims, policy=gate3_policy)
        backend_result.diagnostics = backend_result.diagnostics.model_copy(
            update={
                "vpc": predictive.vpc,
                "pit_calibration": predictive.pit_calibration,
                "npe_score": predictive.npe_score,
                "auc_cmax_be_score": predictive.auc_cmax_be_score,
                "auc_cmax_source": predictive.auc_cmax_source,
            }
        )
        return backend_result
