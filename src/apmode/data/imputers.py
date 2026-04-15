# SPDX-License-Identifier: GPL-2.0-or-later
"""Imputation providers backed by the R harness (PRD §4.2.1).

Concrete implementations of the ``ImputationProvider`` protocol defined
in ``apmode.search.stability``. Each provider spawns an Rscript process
against ``src/apmode/r/impute.R``, which invokes ``mice`` (PMM) or
``missForest`` and returns m imputed CSV paths.

FREM is a structural model path rather than a preprocessor; a
placeholder provider is included so that routing/backends can be wired
uniformly, but calling its ``impute`` raises ``NotImplementedError``
pointing to the nlmixr2 emitter work still needed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from apmode.errors import BackendTimeoutError, CrashError

if TYPE_CHECKING:
    from collections.abc import Sequence

_DEFAULT_IMPUTE_SCRIPT = Path(__file__).parent.parent / "r" / "impute.R"


@dataclass(frozen=True)
class _ImputeResponse:
    """Parsed response from the R imputation harness."""

    status: str
    error_type: str | None
    message: str | None
    imputed_csvs: list[str]
    m: int
    method: str


async def _spawn_rscript(
    r_executable: str,
    script_path: Path,
    request_path: Path,
    response_path: Path,
    timeout_seconds: int | None,
) -> int:
    """Spawn Rscript for the imputation harness. Returns process exit code.

    Shares the security conventions of ``Nlmixr2Runner._spawn_r``:
    ``create_subprocess_exec`` (no shell), new process group via
    ``os.setsid`` for clean timeout kill.
    """
    proc = await asyncio.create_subprocess_exec(
        r_executable,
        str(script_path),
        str(request_path),
        str(response_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    try:
        await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        await proc.wait()
        raise BackendTimeoutError(
            f"Imputation Rscript timed out after {timeout_seconds}s",
            timeout_seconds=timeout_seconds,
            pid=proc.pid,
        ) from None
    return proc.returncode or 0


@dataclass
class _RImputerBase:
    """Shared state for mice/missForest imputation providers."""

    work_dir: Path
    covariates: Sequence[str]
    id_column: str = "NMID"
    r_executable: str = "Rscript"
    script_path: Path = field(default_factory=lambda: _DEFAULT_IMPUTE_SCRIPT)
    timeout_seconds: int | None = 600
    method: str = "pmm"  # overridden by subclasses

    async def impute(
        self,
        source_csv: Path,
        m: int,
        seed: int,
    ) -> list[Path]:
        """Produce m imputed CSVs. Returns absolute paths."""
        if m < 1:
            msg = f"m must be >= 1, got {m}"
            raise ValueError(msg)
        if not source_csv.is_absolute():
            msg = f"source_csv must be absolute, got {source_csv}"
            raise ValueError(msg)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        request_path = self.work_dir / "impute_request.json"
        response_path = self.work_dir / "impute_response.json"
        output_dir = self.work_dir / "imputed"

        request_payload = {
            "source_csv": str(source_csv),
            "output_dir": str(output_dir),
            "method": self.method,
            "m": m,
            "seed": seed,
            "covariates": list(self.covariates),
            "id_column": self.id_column,
        }
        request_path.write_text(json.dumps(request_payload, indent=2))

        exit_code = await _spawn_rscript(
            self.r_executable,
            self.script_path,
            request_path,
            response_path,
            self.timeout_seconds,
        )

        if not response_path.exists():
            raise CrashError(
                f"Imputation Rscript exited with code {exit_code} and no response.json",
                exit_code=exit_code,
            )

        response = _parse_response(response_path)
        if response.status != "success":
            raise CrashError(
                f"Imputation failed ({response.error_type}): {response.message}",
                exit_code=exit_code,
            )
        if len(response.imputed_csvs) != m:
            raise CrashError(
                f"Imputer returned {len(response.imputed_csvs)} datasets, expected {m}",
                exit_code=exit_code,
            )
        return [Path(p) for p in response.imputed_csvs]


def _parse_response(path: Path) -> _ImputeResponse:
    raw = json.loads(path.read_text())
    imputed_csvs_field = raw.get("imputed_csvs") or []
    # jsonlite auto_unbox produces a scalar for single-element arrays;
    # coerce to list for consistent handling.
    if isinstance(imputed_csvs_field, str):
        imputed_csvs_field = [imputed_csvs_field]
    return _ImputeResponse(
        status=str(raw.get("status")),
        error_type=raw.get("error_type"),
        message=raw.get("message"),
        imputed_csvs=[str(p) for p in imputed_csvs_field],
        m=int(raw.get("m", 0)),
        method=str(raw.get("method", "")),
    )


@dataclass
class R_MiceImputer(_RImputerBase):
    """Predictive Mean Matching via the R `mice` package."""

    method: str = "pmm"


@dataclass
class R_MissForestImputer(_RImputerBase):
    """Random-forest imputation via the R `missForest` package.

    Produces an m-draw ensemble (each draw uses a different seed).
    Coverage is comparable to PMM under MAR for nonlinear covariate
    relations; not a formal Rubin-decomposable multiple imputation.
    """

    method: str = "missForest"


@dataclass
class R_FREMProvider:
    """Placeholder FREM provider.

    FREM is a single-fit structural-model path — not a preprocessing
    imputer. This stub exists so routing code can uniformly refer to
    an ``ImputationProvider`` for any covariate strategy; calling
    ``impute`` raises ``NotImplementedError`` with a pointer to the
    nlmixr2 emitter work required.
    """

    work_dir: Path

    async def impute(
        self,
        source_csv: Path,
        m: int,
        seed: int,
    ) -> list[Path]:
        del source_csv, m, seed
        msg = (
            "FREM is a single-fit structural path, not a preprocessing imputer. "
            "Wire FREM into the nlmixr2 emitter (src/apmode/dsl/nlmixr2_emitter.py) "
            "and dispatch directly from the orchestrator — do not call through "
            "the MI stability loop."
        )
        raise NotImplementedError(msg)
