#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Retrieve pmxNODE reference examples (Bräm 2025 NODE benchmark fixtures).
#
# License: GPL (>= 3) — verified via pmxNODE DESCRIPTION 2026-07-08.
# GPL-3 is compatible with APMODE's GPL-2-or-later stance.
#
# References (both in pmxNODE inst/CITATION):
#   Bräm et al. (2025) CPT:PSP 14:5-16. doi:10.1002/psp4.13265
#   Bräm et al. (2024) J Pharmacokinet Pharmacodyn 51:123-140.
#     doi:10.1007/s10928-023-09886-4
set -euo pipefail
OUT_DIR="${1:-benchmarks/datasets/pmxnode_examples}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

if [ ! -d _cloned ]; then
    git clone --depth 1 https://github.com/braemd/pmxnode _cloned
fi

# Copy the 4 examples' data + model files into a flat layout
for i in 1 2 3 4; do
    for f in "data_example${i}_mlx.csv" "data_example${i}_nm.csv" \
             "mlx_example${i}_model.txt" "nm_example${i}_model.ctl"; do
        if [ -f "_cloned/inst/$f" ]; then
            cp "_cloned/inst/$f" "$f"
        fi
    done
done

cp _cloned/inst/CITATION CITATION.R
cp _cloned/LICENSE.md LICENSE.md
cp _cloned/DESCRIPTION DESCRIPTION

echo "Retrieved files:"
ls -1 *.csv *.txt *.ctl 2>/dev/null
