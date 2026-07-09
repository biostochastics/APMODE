# SPDX-License-Identifier: GPL-2.0-or-later
"""Prepare DDMoRe gentamicin IOV dataset (DDMODEL00000238).

Downloads the simulated dataset and canonicalizes it to NONMEM-style CSV
for APMODE ingestion.

Citation: Germovsek E, Kent A, Metsvaht T, Lutsar I, Klein N, Turner MA,
Sharland M, Nielsen EI, Heath PT, Standing JF. (2016) Development and
Evaluation of a Gentamicin Pharmacokinetic Model That Facilitates
Opportunistic Gentamicin Therapeutic Drug Monitoring in Neonates and
Infants. Antimicrob Agents Chemother 60(8):4869-4877.
doi:10.1128/AAC.00577-16 (PMC4958175, PMID 27270281). Verified against
independent citing reviews -- prior versions of this docstring (and of
benchmarks/suite_c/gentamicin_germovsek_2017.yaml) cited two different,
mutually-inconsistent, and nonexistent "2017" DOIs for this same paper.

Source
------
The original DDMoRe Model Repository URL
(http://repository.ddmore.eu/model/DDMODEL00000238) has been retired.
Its successor, http://repository.ddmore.foundation/model/DDMODEL00000238,
returned 503 across all URLs as of 2026-07-08 (see
../ddmore_wayback/README.md) and, even when reachable, serves model
detail pages through JS-rendered/Eclipse-based tooling rather than a
directly fetchable data file. This script instead downloads the model
package's ``Simulated_simdataDDM.csv`` from a long-standing community
GitHub mirror of the DDMoRe repository
(https://github.com/dpastoor/ddmore_scraping, scraped 2016), served via
raw.githubusercontent.com. The file is pinned by SHA-256
(``EXPECTED_SHA256``) so a future change to the mirror is detected and
refused rather than silently canonicalizing a different dataset under
the same name.

The data is itself simulated ("simdataDDM = simulated data using the
median values for covariates" per the mirror's own Readme_ddmore.txt),
not real patient records, so no patient re-identification concern
applies. Every one of the 205 simulated subjects shares an identical
covariate vector (GA=34wk, WT=2.12kg, CREAT=78, PNA=5.4d, PMA=33wk,
GIRL=0); only dosing history, occasion count, and the random-effect
-driven concentration trajectory vary by subject -- consistent with
this fixture's own documented scope (see the Suite C fixture YAML: the
Phase-1 comparison targets disposition-parameter agreement, not
covariate-effect recovery, precisely because this simulated dataset
carries no covariate variation to recover an effect from).
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path
from urllib.error import URLError

import pandas as pd

# Default output directory
OUTPUT_DIR = Path(__file__).parent
CANONICAL_CSV = "gentamicin_iov.csv"

# GitHub mirror of the DDMoRe model repository (dpastoor/ddmore_scraping).
# Verified 2026-07-09: 205 subjects, 2788 rows, matching the population
# described in benchmarks/suite_c/gentamicin_germovsek_2017.yaml and in
# the source paper's Table 1 (model-building dataset, n=205).
RAW_CSV_URL = (
    "https://raw.githubusercontent.com/dpastoor/ddmore_scraping/"
    "master/238/Simulated_simdataDDM.csv"
)
EXPECTED_SHA256 = "2abb995b8cf56406e39793506799eca8d2f849d180607664f2109d5a1d07131c"
_DOWNLOAD_TIMEOUT_SECONDS = 30.0

# Raw NONMEM columns, verified against the actual downloaded file rather
# than assumed from documentation. Order does not matter here; presence
# does -- canonicalize() fails loudly if the upstream file's shape drifts.
_EXPECTED_RAW_COLUMNS = {
    "ID",
    "GA",
    "GIRL",
    "TIME",
    "RATE",
    "EVID",
    "AMT",
    "WT",
    "CREAT",
    "DV",
    "PNA",
    "PMA",
    "TCREA",
    "OCC",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manual_fallback_readme(raw_dir: Path) -> None:
    (raw_dir / "README.txt").write_text(
        "DDMoRe gentamicin dataset (DDMODEL00000238)\n"
        "Automated download failed -- see prepare.py's docstring for why\n"
        "the original repository.ddmore.eu / repository.ddmore.foundation\n"
        "URLs are not used directly.\n"
        "\n"
        f"Manual fallback: download {RAW_CSV_URL}\n"
        f"and save it as {raw_dir / 'gentamicin_raw.csv'}\n"
    )


def download_dataset(output_dir: Path) -> Path:
    """Download the gentamicin dataset from the DDMoRe GitHub mirror.

    Verifies the download against ``EXPECTED_SHA256``. On any network
    failure, degrades to writing manual-download instructions rather
    than raising -- a CI environment without network access should get
    an actionable message, not a crash. On a hash mismatch (the mirror
    served something other than the pinned file), exits loudly instead:
    that is a "the upstream content changed" signal that needs human
    review, not a condition to silently paper over.

    Returns the path to the raw data file (may not exist if the
    automated download failed -- callers must check).
    """
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "gentamicin_raw.csv"

    if raw_file.exists():
        # Cache hit: verify integrity of what's already there rather than
        # trusting an unknown prior download.
        existing = _sha256(raw_file.read_bytes())
        if existing == EXPECTED_SHA256:
            return raw_file
        print(
            f"WARNING: cached {raw_file} has sha256={existing}, "
            f"expected {EXPECTED_SHA256}. Re-downloading."
        )

    print(f"Downloading gentamicin dataset from {RAW_CSV_URL} ...")
    try:
        req = urllib.request.Request(
            RAW_CSV_URL, headers={"User-Agent": "APMODE-DDMoRe-Fetch/1.0"}
        )
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as resp:
            data = resp.read()
    except URLError as e:
        print(f"Download failed: {e}")
        _write_manual_fallback_readme(raw_dir)
        return raw_file

    actual_sha256 = _sha256(data)
    if actual_sha256 != EXPECTED_SHA256:
        msg = (
            f"Downloaded file sha256={actual_sha256} does not match "
            f"expected {EXPECTED_SHA256} -- the upstream mirror may have "
            "changed. Refusing to write a silently-different dataset; "
            "verify the source at RAW_CSV_URL and update EXPECTED_SHA256 "
            "in this file if the change is legitimate."
        )
        print(msg)
        sys.exit(1)

    raw_file.write_bytes(data)
    print(f"Downloaded and verified: {raw_file} (sha256={actual_sha256})")
    return raw_file


def canonicalize(raw_file: Path, output_dir: Path) -> Path:
    """Canonicalize DDMoRe gentamicin data to APMODE format.

    Notes on covariates (all constant across all 205 simulated subjects,
    per raw/README.txt -- see module docstring):
      - ``GIRL`` is 1=girl/0=boy in the raw DDMoRe convention. Left
        unrenamed rather than mapped to APMODE's inferred "SEX" naming
        used elsewhere (male=1/female=0) -- a naive rename would
        silently flip the encoded sex. The Germovsek model does not use
        sex as a covariate at all (see the $PK block referenced in the
        NONMEM control stream), so this column is inert either way.
      - ``CREAT`` / ``TCREA`` are left under their DDMoRe names (matching
        the published NONMEM code's own variable names) rather than
        renamed to a generic "SCR". ``TCREA`` is the population-typical
        creatinine the model substitutes when ``CREAT`` is reported
        missing (a negative sentinel in the original patient data); that
        branch never fires in this simulated file since ``CREAT`` has no
        missing/negative rows here.

    ``CMT`` is not present in the raw file (single-endpoint model --
    concentration is the only observed variable) -- added as a constant
    1 for every row.

    ``MDV`` is derived as ``EVID != 0``, not ``EVID == 1``: the raw file
    contains two ``EVID == 2`` ("other") rows in addition to the
    ``EVID == 1`` dose rows, and both carry ``DV == 0`` exactly like the
    dose rows. A narrower ``EVID == 1`` check would leave those two rows
    marked ``MDV=0`` with a meaningless ``DV=0``, violating
    ``CanonicalPKSchema``'s ``dv_present_when_mdv_0`` invariant.
    """
    if not raw_file.exists():
        print(f"Raw file not found: {raw_file}")
        sys.exit(1)

    df = pd.read_csv(raw_file)

    missing = _EXPECTED_RAW_COLUMNS - set(df.columns)
    if missing:
        msg = (
            f"Raw file {raw_file} is missing expected column(s) {sorted(missing)} "
            f"-- got {sorted(df.columns)}. The upstream file may have changed "
            "shape; update this function before trusting its output."
        )
        raise ValueError(msg)

    df = df.rename(columns={"ID": "NMID", "OCC": "OCCASION"})
    df["MDV"] = (df["EVID"] != 0).astype(int)
    df["CMT"] = 1

    df = df.sort_values(["NMID", "TIME", "EVID"], ascending=[True, True, False])

    out_path = output_dir / CANONICAL_CSV
    df.to_csv(out_path, index=False)

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
