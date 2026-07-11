# SPDX-License-Identifier: GPL-2.0-or-later
"""Regression tests for per-run R imputation response isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apmode.data import imputers
from apmode.data.imputers import R_MiceImputer
from apmode.errors import CrashError


@pytest.mark.asyncio
async def test_nonzero_current_run_cannot_reuse_stale_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    source.write_text("NMID,WT\n1,70\n")
    stale_output = tmp_path / "stale.csv"
    stale_output.write_text("NMID,WT\n1,71\n")
    (tmp_path / "impute_response.json").write_text(
        json.dumps(
            {
                "status": "success",
                "imputed_csvs": [str(stale_output)],
                "m": 1,
                "method": "pmm",
            }
        )
    )

    async def failed_run(*_args: object, **_kwargs: object) -> int:
        return 1

    monkeypatch.setattr(imputers, "_spawn_rscript", failed_run)
    provider = R_MiceImputer(work_dir=tmp_path, covariates=["WT"])
    with pytest.raises(CrashError, match="nonzero code 1"):
        await provider.impute(source.resolve(), m=1, seed=42)


@pytest.mark.asyncio
async def test_response_identity_and_output_paths_are_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    source.write_text("NMID,WT\n1,70\n")

    async def mismatched_run(
        _r: str,
        _script: Path,
        request_path: Path,
        response_path: Path,
        _timeout: int | None,
    ) -> int:
        request = json.loads(request_path.read_text())
        output_dir = Path(request["output_dir"])
        output_dir.mkdir(parents=True)
        output = output_dir / "imputed_1.csv"
        output.write_text("NMID,WT\n1,71\n")
        response_path.write_text(
            json.dumps(
                {
                    "status": "success",
                    "imputed_csvs": [str(output)],
                    "m": 1,
                    "method": "wrong-method",
                }
            )
        )
        return 0

    monkeypatch.setattr(imputers, "_spawn_rscript", mismatched_run)
    provider = R_MiceImputer(work_dir=tmp_path, covariates=["WT"])
    with pytest.raises(CrashError, match="does not match the current request"):
        await provider.impute(source.resolve(), m=1, seed=42)
