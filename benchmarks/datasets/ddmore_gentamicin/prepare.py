# SPDX-License-Identifier: GPL-2.0-or-later
"""Prepare DDMoRe gentamicin IOV dataset (DDMODEL00000238).

Downloads from the DDMoRe Model Repository (CC0 license) and
canonicalizes to NONMEM-style CSV for APMODE ingestion.

Citation: Germovsek E et al. (2017). Development and evaluation of a
gentamicin pharmacokinetic model that facilitates opportunistic
gentamicin therapeutic drug monitoring in neonates and infants.
Antimicrob Agents Chemother 61(8):e00481-17.

Source: http://repository.ddmore.eu/model/DDMODEL00000238
License: CC0 Public Domain
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

# Default output directory
OUTPUT_DIR = Path(__file__).parent
CANONICAL_CSV = "gentamicin_iov.csv"


def download_dataset(output_dir: Path) -> Path:
    """Download gentamicin dataset from DDMoRe repository.

    The DDMoRe repository provides NONMEM-format data files.
    This function downloads the dataset CSV from the model package.

    Returns the path to the downloaded raw data file.
    """
    # DDMoRe model package URL (DDMODEL00000238)
    # The data file is typically named something like gentamicin_data.csv
    # within the model package archive
    url = "http://repository.ddmore.eu/model/DDMODEL00000238#"

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    readme = raw_dir / "README.txt"
    readme.write_text(
        "DDMoRe gentamicin dataset (DDMODEL00000238)\n"
        "License: CC0 Public Domain\n"
        "Source: http://repository.ddmore.eu/model/DDMODEL00000238\n"
        "\n"
        "Download the model package manually from the DDMoRe repository\n"
        "and place the NONMEM dataset CSV in this directory as 'gentamicin_raw.csv'.\n"
        "\n"
        "The DDMoRe repository may require browser-based download.\n"
        f"URL: {url}\n"
    )

    raw_file = raw_dir / "gentamicin_raw.csv"
    if not raw_file.exists():
        print(
            f"DDMoRe dataset not found at {raw_file}.\n"
            f"Please download manually from {url}\n"
            f"and save as {raw_file}"
        )
        return raw_file

    return raw_file


def canonicalize(raw_file: Path, output_dir: Path) -> Path:
    """Canonicalize DDMoRe gentamicin data to APMODE format.

    Expected raw columns (NONMEM format):
      ID, TIME, DV, AMT, EVID, CMT, MDV, RATE, WT, GA, PNA, SCR, OCC
    """
    if not raw_file.exists():
        print(f"Raw file not found: {raw_file}")
        sys.exit(1)

    df = pd.read_csv(raw_file)

    # Standard column rename map
    rename_map: dict[str, str] = {
        "ID": "NMID",
        "id": "NMID",
        "TIME": "TIME",
        "DV": "DV",
        "AMT": "AMT",
        "EVID": "EVID",
        "CMT": "CMT",
        "MDV": "MDV",
        "RATE": "RATE",
        "WT": "WT",
        "GA": "GA",  # Gestational age
        "PNA": "PNA",  # Postnatal age
        "SCR": "SCR",  # Serum creatinine
        "OCC": "OCCASION",  # Inter-occasion variability
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure MDV
    if "MDV" not in df.columns:
        df["MDV"] = (df["EVID"] == 1).astype(int)

    # Sort
    df = df.sort_values(["NMID", "TIME", "EVID"], ascending=[True, True, False])

    # Write canonical CSV
    out_path = output_dir / CANONICAL_CSV
    df.to_csv(out_path, index=False)

    # Compute SHA-256
    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"Gentamicin IOV dataset written: {len(df)} rows, {df['NMID'].nunique()} subjects")
    print(f"SHA-256: {sha256}")
    print(f"Output: {out_path}")

    return out_path


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_file = download_dataset(output_dir)
    if raw_file.exists():
        canonicalize(raw_file, output_dir)
    else:
        print("Skipping canonicalization (raw data not available).")
        print("Place gentamicin_raw.csv in raw/ subdirectory and re-run.")


if __name__ == "__main__":
    main()
