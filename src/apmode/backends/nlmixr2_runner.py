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
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from apmode.backends.predictive_summary import (
    SubjectSimulation,
    build_predictive_diagnostics,
)
from apmode.backends.process_lifecycle import terminate_process_group
from apmode.backends.r_schemas import (
    PredictedSimulationsSubject,
    RSubprocessRequest,
    RSubprocessResponse,
)
from apmode.data.adapters import to_nlmixr2_format
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2
from apmode.errors import BackendTimeoutError, ConvergenceError, CrashError
from apmode.ids import generate_run_id

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import IO

    from apmode.bundle.models import BackendResult, DataManifest, NCASubjectDiagnostic
    from apmode.dsl.ast_models import DSLSpec
    from apmode.governance.policy import Gate3Config


async def _drain_pipe(
    stream: asyncio.StreamReader,
    log_handle: IO[bytes] | None,
    on_line: Callable[[bytes], None] | None,
) -> bytes:
    """Read ``stream`` line-by-line until EOF; tee to log + callback.

    Returns the full body of bytes that crossed the pipe so callers
    that previously relied on ``proc.communicate()`` can still inspect
    it after the fact. The line-by-line drain is what enables PR4's
    streaming SAEM progress: a parser sees iteration N as it is
    emitted, not at process exit.

    Robustness contract: a callback that raises **must not** stop the
    drain. Stopping the drain would leave the OS pipe full and
    deadlock R when its kernel buffer fills (default 64 KB on Linux,
    typically larger on macOS but always finite). The callback's
    exception is logged at WARNING and the drain continues.
    """
    buffer = bytearray()
    # Tracks whether the last byte we wrote to the audit log was a
    # newline. Because two drain coroutines (stdout + stderr) share the
    # same handle, a stream that ends mid-line on one pipe could
    # otherwise be glued to the next pipe's line. We append ``\n``
    # before any non-newline-terminated chunk to guarantee stream
    # boundaries are visible to a forensic reader.
    last_byte_is_newline = True
    while True:
        try:
            line = await stream.readline()
        except (asyncio.LimitOverrunError, ValueError):
            # Default StreamReader limit was bumped to 1 MiB at spawn
            # time, so this branch is reached only for pathological
            # output (single message > 1 MiB). Python 3.12+ re-raises
            # the internal ``LimitOverrunError`` as ``ValueError`` from
            # ``StreamReader.readline``, so we catch both. We then
            # drain repeatedly until we hit a newline (or EOF) so a
            # multi-MiB stack trace surfaces as one logical chunk
            # rather than oscillating between exception and recovery
            # for the rest of the stream.
            chunks: list[bytes] = []
            while True:
                fragment = await stream.read(65536)
                if not fragment:
                    break
                chunks.append(fragment)
                if b"\n" in fragment:
                    break
            line = b"".join(chunks)
        if not line:
            break
        buffer.extend(line)
        if log_handle is not None:
            try:
                if not last_byte_is_newline and not line.startswith(b"\n"):
                    # Sentinel newline so the audit log never glues two
                    # streams' bytes together within a single line.
                    log_handle.write(b"\n")
                log_handle.write(line)
                last_byte_is_newline = line.endswith(b"\n")
            except (OSError, ValueError):
                # ``ValueError`` covers the race where the surrounding
                # ``finally`` closed the handle before the drain task
                # finished its last iteration (cancellation path).
                logger.warning("audit-log write failed; suppressing tee", exc_info=True)
                log_handle = None
        if on_line is not None:
            try:
                on_line(line)
            except Exception:
                logger.warning("on_line callback raised; continuing drain", exc_info=True)
    return bytes(buffer)


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
        *,
        progress_callback: Callable[[bytes], None] | None = None,
        audit_log_dir: Path | None = None,
    ) -> None:
        self.work_dir = work_dir
        # #22: resolve the R executable to an absolute path via shutil.which
        # up front rather than letting asyncio.create_subprocess_exec do a
        # PATH lookup at spawn time. Defence-in-depth — a compromised PATH
        # or an unexpected shadowing binary is caught immediately instead
        # of silently executing untrusted content on first run().
        resolved = (
            shutil.which(r_executable) if not Path(r_executable).is_absolute() else r_executable
        )
        if resolved is None or not Path(resolved).exists():
            msg = (
                f"R executable {r_executable!r} not found on PATH or at that "
                f"absolute path — cannot initialise Nlmixr2Runner"
            )
            raise FileNotFoundError(msg)
        self.r_executable = resolved
        harness = harness_path or _DEFAULT_HARNESS
        if not harness.exists():
            msg = f"R harness script not found at {harness}"
            raise FileNotFoundError(msg)
        self.harness_path = harness
        self.estimation = estimation or ["saem", "focei"]
        # PR4 streaming hooks. Both are optional; when unset, ``_spawn_r``
        # behaves exactly like the pre-PR4 contract (silent stdout/stderr,
        # exit code via response.json). When set, the operator gets a
        # per-line callback for streaming UI / NDJSON emission, and a
        # raw byte-for-byte audit log written under ``audit_log_dir``.
        self.progress_callback = progress_callback
        self.audit_log_dir = audit_log_dir

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
        fixed_parameter: bool = False,
        test_data_path: Path | None = None,
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

        if test_data_path is not None and not test_data_path.is_absolute():
            msg = f"test_data_path must be absolute, got {test_data_path!r}"
            raise ValueError(msg)

        request_id = generate_run_id()
        run_dir = self.work_dir / request_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Adapt the on-disk CSV to the nlmixr2-ready shape and write the
        # adapted copy alongside request.json. Two transformations are
        # critical for FOCEI / SAEM stability and have caused indefinite
        # hangs when missed in the past:
        #   1. ``NMID`` -> ``ID`` rename. nlmixr2 / rxode2 only recognise
        #      the literal column ``ID`` for subject identity. Without the
        #      rename the engine treats every row as a single subject and
        #      FOCEI enters a "Theta reset (ETA drift)" loop forever.
        #   2. ``DVID`` PK-row filtering + drop on single-endpoint models.
        #      The default warfarin / FREM joint datasets carry a ``DVID``
        #      column ("cp" for PK, other values for PD); single-endpoint
        #      DSL specs raise "mis-match in nbr endpoints in model & in
        #      data" or, worse, hang FOCEI on the ambiguous endpoint count.
        # The persisted train/test CSVs (and the bundle digest contributions
        # they make at the orchestrator layer) are unchanged: the adapted
        # copy lives in this run's scratch ``run_dir``. The ``data_path``
        # field on RSubprocessRequest still points at the (adapted)
        # path the harness reads.
        adapted_data_path = run_dir / "data_nlmixr2.csv"
        to_nlmixr2_format(pd.read_csv(data_path)).to_csv(adapted_data_path, index=False)
        adapted_test_data_path: Path | None = None
        if test_data_path is not None:
            adapted_test_data_path = run_dir / "test_data_nlmixr2.csv"
            to_nlmixr2_format(pd.read_csv(test_data_path)).to_csv(
                adapted_test_data_path, index=False
            )

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
            data_path=str(adapted_data_path),
            seed=seed,
            rng_kind="L'Ecuyer-CMRG",
            initial_estimates=initial_estimates,
            estimation=self.estimation,
            compiled_r_code=compiled_r_code,
            split_manifest=split_manifest,
            n_posterior_predictive_sims=n_sims_req,
            test_data_path=(
                str(adapted_test_data_path) if adapted_test_data_path is not None else None
            ),
            fixed_parameter=fixed_parameter,
        )

        request_path = run_dir / "request.json"
        request_path.write_text(request.model_dump_json(indent=2))

        # Spawn R subprocess
        response_path = run_dir / "response.json"
        # Resolve the per-run audit-log path lazily so a missing
        # ``audit_log_dir`` simply turns the streaming tee off.
        audit_log_path: Path | None = None
        if self.audit_log_dir is not None:
            audit_log_path = self.audit_log_dir / f"{spec.model_id}.log"
        exit_code = await self._spawn_r(
            request_path,
            response_path,
            timeout_seconds,
            audit_log_path=audit_log_path,
            on_stderr_line=self.progress_callback,
        )

        # Parse response + optionally enrich with simulation-based diagnostics.
        # ``spec`` is forwarded so that ``build_predictive_diagnostics`` can
        # derive the matching NPE residual scaling from ``spec.observation``;
        # without it the NPE defaults to additive raw-MedAE which inflates
        # by ~3 OoMs on ng/mL-scaled fixtures vs mg/L fixtures.
        result = self._parse_response(
            response_path,
            exit_code,
            spec.model_id,
            gate3_policy=gate3_policy,
            nca_diagnostics=nca_diagnostics,
            spec=spec,
        )
        from apmode.bundle.scoring_contract import attach_scoring_contract

        return attach_scoring_contract(result, spec)

    async def _spawn_r(
        self,
        request_path: Path,
        response_path: Path,
        timeout_seconds: int | None,
        *,
        audit_log_path: Path | None = None,
        on_stderr_line: Callable[[bytes], None] | None = None,
    ) -> int:
        """Spawn Rscript subprocess. Returns exit code.

        Uses create_subprocess_exec (not shell) for security.
        Creates a new process group via os.setsid for clean timeout kill.

        PR4 streaming additions (both default to ``None`` so existing
        callers are unaffected):

        * ``audit_log_path`` — when set, every byte of stdout AND
          stderr is teed to this file in arrival order.
        * ``on_stderr_line`` — when set, every full line on stderr is
          passed to the callback. Callback exceptions are logged
          and do not interrupt the drain.
        """
        proc = await asyncio.create_subprocess_exec(
            self.r_executable,
            str(self.harness_path),
            str(request_path),
            str(response_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=os.setsid,
            # asyncio's default StreamReader buffer is 64 KiB; a long
            # nlmixr2 stack trace without internal newlines triggers
            # ``LimitOverrunError`` and then loses the buffered bytes.
            # 1 MiB is enough for every realistic R diagnostic message
            # while still capping pathological memory growth.
            limit=1_048_576,
        )

        log_handle: IO[bytes] | None = None
        if audit_log_path is not None:
            audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            # Unbuffered so ``tail -f`` sees lines as they arrive.
            log_handle = audit_log_path.open("wb", buffering=0)

        try:
            async with asyncio.timeout(timeout_seconds):
                # Two concurrent drain coroutines so stdout cannot
                # starve stderr (the iteration progress lives on
                # stderr; if stdout outran it without a separate
                # reader, the OS pipe could fill).
                stream_stdout = proc.stdout
                stream_stderr = proc.stderr
                assert stream_stdout is not None  # PIPE was set above
                assert stream_stderr is not None
                drain_stdout = asyncio.create_task(
                    _drain_pipe(stream_stdout, log_handle, None),
                    name=f"drain-stdout-{proc.pid}",
                )
                drain_stderr = asyncio.create_task(
                    _drain_pipe(stream_stderr, log_handle, on_stderr_line),
                    name=f"drain-stderr-{proc.pid}",
                )

                # Race a kill(0)-based watchdog against the drain
                # gather. Background:
                #
                #   rxode2 invokes gcc/clang to compile the generated C
                #   model code. Those grandchildren inherit R's stdout
                #   and stderr FDs. If R exits while a grandchild is
                #   still alive (it segfaulted, was SIGKILLed, or just
                #   takes a long time to flush), the OS pipes stay
                #   open. Both ``await proc.wait()`` and
                #   ``await asyncio.gather(drains)`` then hang
                #   indefinitely — asyncio's ``BaseSubprocessTransport.
                #   _try_finish`` gates the wait future on ALL pipe
                #   transports closing, so an orphaned grandchild can
                #   block the immediate-child reap path forever. The
                #   outer ``asyncio.timeout(timeout_seconds)`` would
                #   eventually fire, but a per-fit timeout of 600 s
                #   means a 600 s wall-clock penalty per orphaned fit.
                #
                # The watchdog uses ``os.kill(pid, 0)`` to detect when
                # the *immediate* child is gone independent of pipe
                # state (kill(0) checks process existence; the OS
                # reaps the immediate child on exit even if pipes
                # stay open). Once the immediate child is gone, we
                # give the drains a short grace and then force-EOF
                # the pipes so the StreamReader unblocks and the
                # asyncio transport finally resolves proc.wait().
                async def _await_drains() -> None:
                    await asyncio.gather(drain_stdout, drain_stderr)

                async def _wait_immediate_child() -> None:
                    while True:
                        try:
                            os.kill(proc.pid, 0)
                        except (ProcessLookupError, PermissionError):
                            return
                        await asyncio.sleep(0.5)

                drains_task: asyncio.Task[None] = asyncio.create_task(
                    _await_drains(),
                    name=f"drains-await-{proc.pid}",
                )
                watchdog: asyncio.Task[None] = asyncio.create_task(
                    _wait_immediate_child(),
                    name=f"child-watchdog-{proc.pid}",
                )
                done, _pending = await asyncio.wait(
                    {drains_task, watchdog},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if watchdog in done and drains_task not in done:
                    # Immediate child gone but drains still draining
                    # (orphaned grandchild holds the pipes). Give a
                    # short grace for any tail of legitimate output
                    # and then force-EOF the streams so the drain
                    # coroutines complete and the asyncio transport
                    # can resolve proc.wait().
                    _drain_grace = 5.0
                    try:
                        await asyncio.wait_for(drains_task, timeout=_drain_grace)
                    except TimeoutError:
                        logger.warning(
                            "drain_pipes_hung_after_proc_exit_force_eof",
                            extra={
                                "pid": proc.pid,
                                "grace_seconds": _drain_grace,
                            },
                        )
                        # feed_eof() makes StreamReader.readline()
                        # return b"" and the drain loop exits cleanly.
                        with contextlib.suppress(Exception):
                            stream_stdout.feed_eof()
                        with contextlib.suppress(Exception):
                            stream_stderr.feed_eof()
                        with contextlib.suppress(BaseException):
                            await asyncio.gather(
                                drain_stdout,
                                drain_stderr,
                                return_exceptions=True,
                            )
                else:
                    # Drains completed normally — cancel the watchdog
                    # so it doesn't leak.
                    watchdog.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await watchdog

                # By this point the immediate child has exited and the
                # drains have finished (cleanly or via force-EOF). The
                # transport can now resolve proc.wait() — but with a
                # safety timeout in case the asyncio transport's
                # _try_finish is still wedged on a pipe that we
                # couldn't close. After this short wait, proceed with
                # whatever returncode the OS already reaped (kill(0)
                # confirmed the immediate child is gone).
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
        except TimeoutError:
            # Route the timeout path through ``terminate_process_group``
            # (SIGTERM, 5 s grace, then SIGKILL) just like the
            # cancellation path below. Going straight to SIGKILL leaves
            # nlmixr2's intermediate ``~/.cache/R/...`` artefacts behind;
            # the SIGTERM pass lets R clean up before we escalate.
            await terminate_process_group(proc)
            raise BackendTimeoutError(
                f"R subprocess timed out after {timeout_seconds}s",
                timeout_seconds=timeout_seconds,
                pid=proc.pid,
            ) from None
        except asyncio.CancelledError:
            # Plan Task 33 — DELETE /runs/{id} cancelled the parent
            # task while R was running. SIGTERM the whole process group
            # (5 s grace, then SIGKILL); R cleans up its temp files on
            # SIGTERM but not on SIGKILL, so the grace pass is worth
            # the wait. ``terminate_process_group`` is idempotent and
            # handles the race where R exited between the wait_for
            # raising and us looking at the PID.
            await terminate_process_group(proc)
            raise
        finally:
            if log_handle is not None:
                with contextlib.suppress(OSError):
                    log_handle.close()

        return proc.returncode or 0

    def _parse_response(
        self,
        response_path: Path,
        exit_code: int,
        model_id: str,
        *,
        gate3_policy: Gate3Config | None = None,
        nca_diagnostics: list[NCASubjectDiagnostic] | None = None,
        spec: DSLSpec | None = None,
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

        # Pass the DSL spec so build_predictive_diagnostics can derive
        # the matching NPE residual scaling from spec.observation. Without
        # this hint the rc8 raw-MedAE path inflates NPE by ~3 OoMs on
        # ng/mL-scaled fixtures (mavoglurant) vs mg/L fixtures (theo)
        # even when the model fit quality is comparable; the
        # dimensionless rescaling collapses the cross-fixture skew.
        predictive = build_predictive_diagnostics(subject_sims, policy=gate3_policy, spec=spec)
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
