#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Retrieve Open Systems Pharmacology observed-data database.
#
# License caveat: NO LICENSE file at repo HEAD (verified 2026-07-08).
# README explicitly says "PK data have carefully been transferred and digitized
# from the referenced original sources" — the curation is unlicensed effort,
# but each row cites its original PMID so the underlying data IS public via
# the primary literature.
#
# Reference: Open Systems Pharmacology consortium (Bayer).
#   https://github.com/Open-Systems-Pharmacology/Database-for-observed-data
set -euo pipefail
OUT_DIR="${1:-benchmarks/datasets/osp_observed_data}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

if [ ! -d _cloned ]; then
    git clone --depth 1 https://github.com/Open-Systems-Pharmacology/Database-for-observed-data _cloned
fi

for f in DDI.csv Pediatrics.csv ObsDataPK_OSP.xlsx obsDataPK_OSP_DISL.docx README.md; do
    if [ -f "_cloned/$f" ]; then
        rows=$(if [[ "$f" == *.csv ]]; then tail -n +2 "_cloned/$f" | wc -l | tr -d ' '; else echo "-"; fi)
        echo "$f: $rows rows"
    fi
done
