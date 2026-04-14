# SPDX-License-Identifier: GPL-2.0-or-later
"""Public PK dataset discovery and download.

Provides access to real and validated simulated PK datasets from nlmixr2data
(R package) and converts them to NONMEM-format CSVs compatible with
``apmode run`` and ``apmode validate``.

Requires: R with nlmixr2data package installed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 — used at runtime in function bodies
from typing import Literal


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for a public PK dataset."""

    name: str
    source: str  # e.g. "nlmixr2data", "pharmpy"
    description: str
    n_subjects: int
    n_rows: int
    route: Literal["oral", "iv_bolus", "iv_infusion", "sc", "mixed"]
    elimination: Literal["linear", "michaelis_menten", "mixed", "unknown"]
    compartments: int
    columns: list[str]
    covariates: list[str] = field(default_factory=list)
    has_occasion: bool = False
    reference: str = ""


# ---------------------------------------------------------------------------
# Dataset registry — manually curated metadata for well-known datasets
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, DatasetInfo] = {
    "theo_sd": DatasetInfo(
        name="theo_sd",
        source="nlmixr2data",
        description=(
            "Theophylline single-dose oral PK (12 subjects, dense sampling)."
            " Classic pharmacometrics teaching dataset."
        ),
        n_subjects=12,
        n_rows=144,
        route="oral",
        elimination="linear",
        compartments=1,
        columns=["ID", "TIME", "DV", "AMT", "EVID", "CMT", "WT"],
        covariates=["WT"],
        reference="Boeckmann AJ, Sheiner LB, Beal SL. NONMEM Users Guide.",
    ),
    "warfarin": DatasetInfo(
        name="warfarin",
        source="nlmixr2data",
        description="Warfarin PK/PD dataset. Rich oral PK with age, sex, weight covariates.",
        n_subjects=32,
        n_rows=515,
        route="oral",
        elimination="linear",
        compartments=1,
        columns=["id", "time", "amt", "dv", "dvid", "evid", "wt", "age", "sex"],
        covariates=["wt", "age", "sex"],
        reference="O'Reilly RA. Warfarin metabolism and drug interactions.",
    ),
    "Oral_1CPT": DatasetInfo(
        name="Oral_1CPT",
        source="nlmixr2data",
        description=(
            "Simulated 1-compartment oral PK (120 subjects). Ground truth: first-order"
            " absorption, linear elimination."
        ),
        n_subjects=120,
        n_rows=7920,
        route="oral",
        elimination="linear",
        compartments=1,
        columns=[
            "ID",
            "TIME",
            "DV",
            "LNDV",
            "MDV",
            "AMT",
            "EVID",
            "DOSE",
            "V",
            "CL",
            "KA",
            "SS",
            "II",
            "SD",
            "CMT",
        ],
        reference="ACOP 2016 simulation.",
    ),
    "Oral_1CPTMM": DatasetInfo(
        name="Oral_1CPTMM",
        source="nlmixr2data",
        description=(
            "Simulated 1-compartment oral PK with Michaelis-Menten elimination (120 subjects)."
            " Ground truth: nonlinear clearance."
        ),
        n_subjects=120,
        n_rows=7920,
        route="oral",
        elimination="michaelis_menten",
        compartments=1,
        columns=[
            "ID",
            "TIME",
            "DV",
            "LNDV",
            "MDV",
            "AMT",
            "EVID",
            "DOSE",
            "V",
            "VM",
            "KM",
            "KA",
            "SS",
            "II",
            "SD",
            "CMT",
        ],
        reference="ACOP 2016 simulation.",
    ),
    "Oral_2CPT": DatasetInfo(
        name="Oral_2CPT",
        source="nlmixr2data",
        description=(
            "Simulated 2-compartment oral PK (120 subjects). Ground truth: two-compartment"
            " distribution, linear elimination."
        ),
        n_subjects=120,
        n_rows=7920,
        route="oral",
        elimination="linear",
        compartments=2,
        columns=[
            "ID",
            "TIME",
            "DV",
            "LNDV",
            "MDV",
            "AMT",
            "EVID",
            "DOSE",
            "V1",
            "CL",
            "Q",
            "V2",
            "KA",
            "SS",
            "II",
            "SD",
            "CMT",
        ],
        reference="ACOP 2016 simulation.",
    ),
    "Oral_2CPTMM": DatasetInfo(
        name="Oral_2CPTMM",
        source="nlmixr2data",
        description="Simulated 2-compartment oral PK with MM elimination (120 subjects).",
        n_subjects=120,
        n_rows=7920,
        route="oral",
        elimination="michaelis_menten",
        compartments=2,
        columns=[
            "ID",
            "TIME",
            "DV",
            "LNDV",
            "MDV",
            "AMT",
            "EVID",
            "DOSE",
            "V",
            "VM",
            "KM",
            "Q",
            "V2",
            "KA",
            "SS",
            "II",
            "SD",
            "CMT",
        ],
        reference="ACOP 2016 simulation.",
    ),
    "Bolus_1CPT": DatasetInfo(
        name="Bolus_1CPT",
        source="nlmixr2data",
        description="Simulated 1-compartment IV bolus (120 subjects). Simple linear PK.",
        n_subjects=120,
        n_rows=7920,
        route="iv_bolus",
        elimination="linear",
        compartments=1,
        columns=[
            "ID",
            "TIME",
            "DV",
            "LNDV",
            "MDV",
            "AMT",
            "EVID",
            "DOSE",
            "V",
            "CL",
            "SS",
            "II",
            "SD",
            "CMT",
        ],
        reference="ACOP 2016 simulation.",
    ),
    "Bolus_2CPT": DatasetInfo(
        name="Bolus_2CPT",
        source="nlmixr2data",
        description="Simulated 2-compartment IV bolus (120 subjects).",
        n_subjects=120,
        n_rows=7920,
        route="iv_bolus",
        elimination="linear",
        compartments=2,
        columns=[
            "ID",
            "TIME",
            "DV",
            "LNDV",
            "MDV",
            "AMT",
            "EVID",
            "DOSE",
            "V1",
            "CL",
            "Q",
            "V2",
            "SS",
            "II",
            "SD",
            "CMT",
        ],
        reference="ACOP 2016 simulation.",
    ),
    "Bolus_2CPTMM": DatasetInfo(
        name="Bolus_2CPTMM",
        source="nlmixr2data",
        description="Simulated 2-compartment IV bolus with Michaelis-Menten elimination.",
        n_subjects=120,
        n_rows=7920,
        route="iv_bolus",
        elimination="michaelis_menten",
        compartments=2,
        columns=[
            "ID",
            "TIME",
            "DV",
            "LNDV",
            "MDV",
            "AMT",
            "EVID",
            "DOSE",
            "V",
            "VM",
            "KM",
            "Q",
            "V2",
            "SD",
            "CMT",
        ],
        reference="ACOP 2016 simulation.",
    ),
    "Infusion_1CPT": DatasetInfo(
        name="Infusion_1CPT",
        source="nlmixr2data",
        description="Simulated 1-compartment IV infusion (120 subjects).",
        n_subjects=120,
        n_rows=7920,
        route="iv_infusion",
        elimination="linear",
        compartments=1,
        columns=[
            "ID",
            "TIME",
            "DV",
            "LNDV",
            "MDV",
            "AMT",
            "EVID",
            "DOSE",
            "V",
            "CL",
            "RATE",
            "SS",
            "II",
            "SD",
            "CMT",
        ],
        reference="ACOP 2016 simulation.",
    ),
    "mavoglurant": DatasetInfo(
        name="mavoglurant",
        source="nlmixr2data",
        description=(
            "Mavoglurant (AFQ056) real clinical PK data (120 subjects). Oral, with occasion"
            " variability. Phase I/II mGluR5 antagonist."
        ),
        n_subjects=120,
        n_rows=2678,
        route="oral",
        elimination="unknown",
        compartments=1,
        columns=[
            "ID",
            "CMT",
            "EVID",
            "MDV",
            "DV",
            "AMT",
            "TIME",
            "DOSE",
            "OCC",
            "RATE",
            "AGE",
            "SEX",
            "WT",
            "HT",
        ],
        covariates=["AGE", "SEX", "WT", "HT"],
        has_occasion=True,
        reference="Wendling T et al. J Pharmacokinet Pharmacodyn. 2015.",
    ),
    "pheno_sd": DatasetInfo(
        name="pheno_sd",
        source="nlmixr2data",
        description=(
            "Phenobarbital neonatal PK (59 subjects). Sparse pediatric data, IV, classic"
            " NONMEM example."
        ),
        n_subjects=59,
        n_rows=744,
        route="iv_bolus",
        elimination="linear",
        compartments=1,
        columns=["ID", "TIME", "AMT", "WT", "APGR", "DV", "MDV", "EVID"],
        covariates=["WT", "APGR"],
        reference="Grasela TH, Donn SM. Clin Pharmacol Ther. 1985;38:396-400.",
    ),
    "nimoData": DatasetInfo(
        name="nimoData",
        source="nlmixr2data",
        description=(
            "Nimotuzumab (anti-EGFR mAb) real clinical PK. Monoclonal antibody with"
            " potential target-mediated disposition."
        ),
        n_subjects=40,
        n_rows=580,
        route="iv_infusion",
        elimination="unknown",
        compartments=2,
        columns=["ID", "TIME", "AMT", "DV", "EVID", "CMT", "MDV", "DOSE", "WT"],
        covariates=["WT"],
        reference="Crombet T et al. Nimotuzumab PK.",
    ),
    "theo_md": DatasetInfo(
        name="theo_md",
        source="nlmixr2data",
        description=(
            "Theophylline multiple-dose oral PK. Extends theo_sd with steady-state dosing."
        ),
        n_subjects=12,
        n_rows=348,
        route="oral",
        elimination="linear",
        compartments=1,
        columns=["ID", "TIME", "DV", "AMT", "EVID", "CMT", "WT", "SS", "II"],
        covariates=["WT"],
        reference="Boeckmann AJ, Sheiner LB, Beal SL. NONMEM Users Guide.",
    ),
}


# ---------------------------------------------------------------------------
# Discovery and download
# ---------------------------------------------------------------------------


def list_datasets(
    *,
    route: str | None = None,
    elimination: str | None = None,
    min_subjects: int = 0,
) -> list[DatasetInfo]:
    """List available public PK datasets, optionally filtered.

    Args:
        route: Filter by route ("oral", "iv_bolus", "iv_infusion").
        elimination: Filter by elimination ("linear", "michaelis_menten").
        min_subjects: Minimum number of subjects.

    Returns:
        List of matching DatasetInfo records.
    """
    results: list[DatasetInfo] = []
    for info in DATASET_REGISTRY.values():
        if route and info.route != route:
            continue
        if elimination and info.elimination != elimination:
            continue
        if info.n_subjects < min_subjects:
            continue
        results.append(info)
    return results


def fetch_dataset(
    name: str,
    output_dir: Path,
    *,
    normalize_columns: bool = True,
) -> Path:
    """Download/extract a dataset and save as NONMEM-format CSV.

    Extracts data from R's nlmixr2data package and writes a CSV file
    with standardized NONMEM column names (ID, TIME, DV, AMT, EVID, MDV, CMT).

    Args:
        name: Dataset name (key in DATASET_REGISTRY).
        output_dir: Directory to write the CSV file.
        normalize_columns: If True, normalize column names to uppercase NONMEM standard.

    Returns:
        Path to the written CSV file.

    Raises:
        ValueError: If dataset name is unknown.
        RuntimeError: If R extraction fails.
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        msg = f"Unknown dataset '{name}'. Available: {available}"
        raise ValueError(msg)

    info = DATASET_REGISTRY[name]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.csv"

    # Cache check: skip fetch if file already exists
    if out_path.exists():
        return out_path

    if info.source == "nlmixr2data":
        _fetch_from_nlmixr2data(name, out_path, normalize=normalize_columns)
    else:
        msg = f"Source '{info.source}' not yet supported for fetch."
        raise NotImplementedError(msg)

    return out_path


def _fetch_from_nlmixr2data(name: str, out_path: Path, *, normalize: bool = True) -> None:
    """Extract a dataset from R's nlmixr2data package to CSV."""
    # Column name normalization mapping
    normalize_script = ""
    if normalize:
        normalize_script = """
    # Normalize column names to uppercase NONMEM standard
    names(d) <- toupper(names(d))
    # APMODE canonical schema uses NMID (not ID)
    if ("ID" %in% names(d) && !"NMID" %in% names(d)) {
      names(d)[names(d) == "ID"] <- "NMID"
    }
    # Ensure required columns exist
    if (!"MDV" %in% names(d) && "EVID" %in% names(d)) {
      d$MDV <- as.integer(d$EVID != 0)
    }
    if (!"CMT" %in% names(d)) {
      d$CMT <- 1L
    }
    # Normalize EVID: nlmixr2 uses 101 for oral dose, NONMEM uses 1
    if ("EVID" %in% names(d)) {
      d$EVID[d$EVID == 101] <- 1L
      d$EVID[d$EVID == 102] <- 1L  # infusion dose
    }
    # Observation CMT should be 1 (central) for concentration data
    if ("CMT" %in% names(d)) {
      d$CMT[d$EVID == 0] <- 1L
    }
"""

    r_script = f"""
    library(nlmixr2data)
    d <- {name}
    {normalize_script}
    write.csv(d, "{out_path}", row.names = FALSE)
    cat(nrow(d))
    """

    result = subprocess.run(
        ["Rscript", "-e", r_script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        msg = f"R extraction failed for '{name}': {result.stderr.strip()}"
        raise RuntimeError(msg)

    if not out_path.exists():
        msg = f"R script completed but output file not found: {out_path}"
        raise RuntimeError(msg)


def fetch_all(
    output_dir: Path,
    *,
    route: str | None = None,
    elimination: str | None = None,
) -> list[Path]:
    """Fetch all matching datasets to a directory.

    Args:
        output_dir: Directory to write CSV files.
        route: Optional route filter.
        elimination: Optional elimination filter.

    Returns:
        List of paths to written CSV files.
    """
    datasets = list_datasets(route=route, elimination=elimination)
    paths: list[Path] = []
    for info in datasets:
        path = fetch_dataset(info.name, output_dir)
        paths.append(path)
    return paths


def dataset_summary_table() -> str:
    """Return a formatted summary table of all available datasets."""
    lines: list[str] = []
    lines.append(f"{'Name':<20} {'Subj':>5} {'Route':<12} {'Elim':<18} {'Cmt':>3}  Description")
    lines.append("-" * 100)
    for info in DATASET_REGISTRY.values():
        desc = info.description[:50] + "..." if len(info.description) > 50 else info.description
        lines.append(
            f"{info.name:<20} {info.n_subjects:>5} {info.route:<12} "
            f"{info.elimination:<18} {info.compartments:>3}  {desc}"
        )
    return "\n".join(lines)
