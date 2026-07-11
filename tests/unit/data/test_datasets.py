# SPDX-License-Identifier: GPL-2.0-or-later
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from apmode.data import datasets
from apmode.data.datasets import DATASET_REGISTRY

if TYPE_CHECKING:
    import pytest


def test_bolus_1cptmm_registered() -> None:
    info = DATASET_REGISTRY["Bolus_1CPTMM"]
    assert info.route == "iv_bolus"
    assert info.elimination == "michaelis_menten"
    assert info.compartments == 1
    assert info.n_rows == 7920


def test_infusion_2cptmm_registered() -> None:
    info = DATASET_REGISTRY["Infusion_2CPTMM"]
    assert info.route == "iv_infusion"
    assert info.elimination == "michaelis_menten"
    assert info.compartments == 2
    assert "RATE" in info.columns


def test_r_extraction_passes_output_path_as_data_not_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    malicious = tmp_path / 'out"); cat("APMODE_INJECTED\\n"); #.csv'
    captured: list[str] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.extend(args)
        Path(args[-2]).write_text("NMID,TIME\n1,0\n")
        return subprocess.CompletedProcess(args, 0, stdout="1", stderr="")

    monkeypatch.setattr(datasets.subprocess, "run", fake_run)
    datasets._fetch_from_nlmixr2data("theo", malicious)
    assert str(malicious) == captured[-2]
    assert str(malicious) not in captured[2]
    assert "output_path <- args[[2]]" in captured[2]


def test_normalized_and_raw_fetches_have_distinct_cache_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = next(iter(DATASET_REGISTRY))
    calls: list[tuple[Path, bool]] = []

    def fake_fetch(_name: str, path: Path, *, normalize: bool = True) -> None:
        calls.append((path, normalize))
        path.write_text("x\n1\n")

    monkeypatch.setattr(datasets, "_fetch_from_nlmixr2data", fake_fetch)
    normalized = datasets.fetch_dataset(name, tmp_path, normalize_columns=True)
    raw = datasets.fetch_dataset(name, tmp_path, normalize_columns=False)
    assert normalized != raw
    assert normalized.name == f"{name}.csv"
    assert raw.name == f"{name}.raw.csv"
    assert calls == [(normalized, True), (raw, False)]
