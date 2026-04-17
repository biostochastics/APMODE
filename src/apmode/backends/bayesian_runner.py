# SPDX-License-Identifier: GPL-2.0-or-later
"""Bayesian backend runner via Stan/Torsten (Phase 2+, PRD §4.2.2).

Compiles DSLSpec -> Stan via stan_emitter.emit_stan, spawns a Python harness
subprocess that drives cmdstanpy, parses draws + diagnostics into BackendResult.

Mirrors the Nlmixr2Runner pattern: file-based JSON request/response, asyncio
subprocess with SIGKILL on process-group timeout, no shell.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from apmode.bundle.models import SamplerConfig
from apmode.dsl.stan_emitter import emit_stan
from apmode.errors import BackendTimeoutError, ConvergenceError, CrashError
from apmode.ids import generate_run_id

if TYPE_CHECKING:
    from apmode.bundle.models import BackendResult, DataManifest, NCASubjectDiagnostic
    from apmode.dsl.ast_models import DSLSpec
    from apmode.governance.policy import Gate3Config


_DEFAULT_HARNESS = Path(__file__).parent.parent / "bayes" / "harness.py"


class BayesianSubprocessRequest(BaseModel):
    """Request written to {run_dir}/request.json for the Python harness."""

    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    run_id: str
    candidate_id: str
    spec: Any  # DSLSpec model_dump'd to avoid schema cycles
    data_path: str
    seed: int
    initial_estimates: dict[str, float]
    compiled_stan_code: str
    sampler_config: SamplerConfig
    output_draws_path: str


class BayesianSubprocessResponse(BaseModel):
    """Response read from {run_dir}/response.json after harness completes."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["success", "error"]
    error_type: Literal["convergence", "crash", "compile_error", "invalid_spec"] | None = None
    error_detail: str | None = None
    result: dict[str, Any] | None = None
    session_info: dict[str, str] = Field(default_factory=dict)


class BayesianRunner:
    """BackendRunner implementation for Stan/Torsten via Python subprocess."""

    def __init__(
        self,
        work_dir: Path,
        python_executable: str | None = None,
        harness_path: Path | None = None,
        cmdstan_path: Path | None = None,
        torsten_path: Path | None = None,
        default_sampler_config: SamplerConfig | None = None,
    ) -> None:
        self.work_dir = work_dir
        # #22: resolve the Python interpreter and harness script to
        # absolute paths up front. sys.executable is already absolute;
        # only validate a caller-supplied override.
        py_candidate = python_executable or sys.executable
        if not Path(py_candidate).is_absolute():
            resolved_py = shutil.which(py_candidate)
            if resolved_py is None:
                msg = (
                    f"Python executable {py_candidate!r} not found on PATH; "
                    "cannot initialise BayesianRunner"
                )
                raise FileNotFoundError(msg)
            py_candidate = resolved_py
        if not Path(py_candidate).exists():
            msg = f"Python executable not found at {py_candidate}"
            raise FileNotFoundError(msg)
        self.python_executable = py_candidate
        harness = harness_path or _DEFAULT_HARNESS
        if not harness.exists():
            msg = f"Bayesian harness script not found at {harness}"
            raise FileNotFoundError(msg)
        self.harness_path = harness
        self.cmdstan_path = cmdstan_path
        self.torsten_path = torsten_path
        self.sampler_config = default_sampler_config or SamplerConfig()

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
        gate3_policy: Gate3Config | None = None,
        nca_diagnostics: list[NCASubjectDiagnostic] | None = None,
        fixed_parameter: bool = False,
    ) -> BackendResult:
        """Run Stan sampling via Python subprocess.

        Raises ValueError if spec contains NODE modules (not supported by the
        Stan emitter) or if data_path is missing.

        ``gate3_policy`` and ``nca_diagnostics`` are accepted for
        ``BackendRunner``-protocol conformance. Scope 2 (Stan
        ``generated quantities`` y_pred emission) wires them into
        :func:`apmode.bayes.harness.build_predictive_from_draws` so the
        BayesianRunner populates VPC/NPE/AUC-Cmax atomically via
        :func:`apmode.backends.predictive_summary.build_predictive_diagnostics`.
        Until that lands the kwargs are recorded and ignored.
        """
        _ = gate3_policy, nca_diagnostics  # Scope 2: Stan y_pred wiring
        if fixed_parameter:
            msg = (
                "fixed_parameter=True not yet honoured by BayesianRunner "
                "(requires Stan `generate_quantities` fix-param stage — see "
                "PRD \u00a78 Phase 3 / loro_cv.py). Refusing to evaluate to avoid "
                "silent train/test leakage."
            )
            raise NotImplementedError(msg)
        if data_path is None:
            raise ValueError("data_path is required for BayesianRunner")

        if spec.has_node_modules():
            raise ValueError(
                f"BayesianRunner cannot execute spec {spec.model_id} - "
                "NODE modules are not supported by the Stan emitter"
            )

        request_id = generate_run_id()
        run_dir = self.work_dir / request_id
        run_dir.mkdir(parents=True, exist_ok=True)

        compiled_stan = emit_stan(spec, initial_estimates=initial_estimates)
        (run_dir / "model.stan").write_text(compiled_stan)

        draws_path = run_dir / "posterior_draws.parquet"

        request = BayesianSubprocessRequest(
            request_id=request_id,
            run_id=request_id,
            candidate_id=spec.model_id,
            spec=spec.model_dump(mode="json"),
            data_path=str(data_path),
            seed=seed,
            initial_estimates=initial_estimates,
            compiled_stan_code=compiled_stan,
            sampler_config=self.sampler_config.model_copy(update={"seed": seed}),
            output_draws_path=str(draws_path),
        )

        request_path = run_dir / "request.json"
        request_path.write_text(request.model_dump_json(indent=2))

        response_path = run_dir / "response.json"
        exit_code = await self._spawn_harness(request_path, response_path, timeout_seconds)
        return self._parse_response(response_path, exit_code, spec)

    async def _spawn_harness(
        self,
        request_path: Path,
        response_path: Path,
        timeout_seconds: int | None,
    ) -> int:
        """Spawn the Python harness subprocess; returns exit code.

        See Nlmixr2Runner._spawn_r for the identical pattern and rationale
        (asyncio + setsid process group + SIGKILL on timeout, no shell).
        """
        env = os.environ.copy()
        if self.cmdstan_path is not None:
            env["CMDSTAN"] = str(self.cmdstan_path)
        if self.torsten_path is not None:
            env["TORSTEN"] = str(self.torsten_path)

        # start_new_session=True is the cross-platform equivalent of
        # preexec_fn=os.setsid and avoids the thread-safety caveat of
        # preexec_fn on forking platforms.
        proc = await asyncio.create_subprocess_exec(
            self.python_executable,
            str(self.harness_path),
            str(request_path),
            str(response_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            env=env,
        )

        try:
            _stdout, _stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            await proc.wait()
            raise BackendTimeoutError(
                f"Bayesian harness timed out after {timeout_seconds}s",
                timeout_seconds=timeout_seconds,
                pid=proc.pid,
            ) from None

        return proc.returncode or 0

    def _parse_response(
        self,
        response_path: Path,
        exit_code: int,
        spec: DSLSpec,
    ) -> BackendResult:
        """Parse response.json and build BackendResult, or raise a classified error.

        The returned :class:`BackendResult` carries the Stan-specific
        :class:`~apmode.bundle.models.ScoringContract`
        (``nlpd_integrator="hmc_nuts"``) so
        :meth:`BackendResult.validate_backend_scoring_contract_consistency`
        accepts it and Gate 3 groups it correctly.
        """
        from apmode.bundle.models import BackendResult
        from apmode.bundle.scoring_contract import attach_scoring_contract

        if not response_path.exists():
            raise CrashError(
                f"Bayesian harness exited with code {exit_code} and no response.json",
                exit_code=exit_code,
            )

        raw = json.loads(response_path.read_text())
        response = BayesianSubprocessResponse.model_validate(raw)

        if response.status == "error":
            if response.error_type == "convergence":
                raise ConvergenceError(
                    f"Stan convergence failure for {spec.model_id}: {response.error_detail}",
                    method="nuts",
                )
            raise CrashError(
                f"Bayesian backend error ({response.error_type}) for {spec.model_id}: "
                f"{response.error_detail}",
                exit_code=exit_code,
            )

        if response.result is None:
            raise CrashError(
                "Bayesian harness returned success but no result payload",
                exit_code=exit_code,
            )

        # Inject the Stan scoring contract BEFORE model_validate so the
        # backend_scoring_contract_consistency validator passes. The raw
        # response carries a DiagnosticBundle with a default (classical
        # FOCEI) contract; we overwrite it inline.
        from apmode.bundle.scoring_contract import (
            _contract_for_backend,
            _obs_from_spec,
        )

        raw_result = (
            dict(response.result) if isinstance(response.result, dict) else response.result
        )
        if isinstance(raw_result, dict) and "diagnostics" in raw_result:
            _nlpd_kind, _re_treatment, _integrator, _precision = _contract_for_backend(
                "bayesian_stan"
            )
            diagnostics_dict = dict(raw_result["diagnostics"])
            blq = diagnostics_dict.get("blq") or {}
            blq_method = (
                blq.get("method") if isinstance(blq, dict) else getattr(blq, "method", "none")
            ) or "none"
            diagnostics_dict["scoring_contract"] = {
                "contract_version": 1,
                "nlpd_kind": _nlpd_kind,
                "re_treatment": _re_treatment,
                "nlpd_integrator": _integrator,
                "blq_method": blq_method,
                "observation_model": _obs_from_spec(spec),
                "float_precision": _precision,
            }
            raw_result["diagnostics"] = diagnostics_dict

        result = BackendResult.model_validate(raw_result)
        return attach_scoring_contract(result, spec)


# Harness contract (implemented in src/apmode/bayes/harness.py):
#   1. Read/validate request.json as BayesianSubprocessRequest.
#   2. Convert {data_path, spec} into the Stan data dict matching the
#      stan_emitter data block (event_subject/event_time/event_amt/event_cmt/
#      event_evid/event_rate arrays + per-subject index ranges; cens+loq for
#      BLQM3/BLQM4; per-covariate vectors).
#   3. Compile the Stan program via cmdstanpy.CmdStanModel.
#   4. Sample with the provided SamplerConfig; include Torsten patches when
#      TORSTEN env var is set.
#   5. Compute diagnostics via arviz: R-hat, ESS_bulk/tail, n_divergent,
#      n_max_treedepth, E-BFMI, MCSE for headline params, Pareto-k via loo.
#   6. Write posterior draws to output_draws_path in long-form Parquet.
#   7. Aggregate ParameterEstimate (posterior_mean, sd, q05/q50/q95); derive
#      eta_shrinkage from posterior eta draws.
#   8. Build BackendResult dict with backend="bayesian_stan",
#      posterior_diagnostics, sampler_config, posterior_draws_path.
#   9. Emit status="error",error_type="convergence" when catastrophic sampling
#      failure is detected (all chains stuck, R-hat>2, divergences>25%).
#  10. On uncaught exception -> error_type="crash"; Stan compile failure ->
#      error_type="compile_error".
#  11. VPC/NPE/AUC-Cmax predictive diagnostics: the
#      apmode.bayes.harness.build_predictive_from_draws helper plumbs
#      Stan's posterior-predictive y_pred draws into the shared
#      apmode.backends.predictive_summary.build_predictive_diagnostics
#      helper. The stan_emitter generated-quantities block must first
#      emit y_pred[n] — tracked as a follow-up commit per CHANGELOG.
#      Until that lands, BayesianRunner does not populate
#      diagnostics.{vpc, npe_score, auc_cmax_be_score} and Gate 3 falls
#      back to the CWRES proxy for Bayesian candidates.
