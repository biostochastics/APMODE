# SPDX-License-Identifier: GPL-2.0-or-later
from apmode.data.datasets import DATASET_REGISTRY


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
