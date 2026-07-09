#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Retrieve Metrum MeRGE Expo 1 (NONMEM/FOCE) data + model files.
#
# License caveat: the upstream repo has NO LICENSE file at HEAD (verified
# via GitHub API 2026-07-08). Do not vendor into APMODE distribution — this
# is a `tier: reference_only` fixture. Reproducing methodology is fine;
# code/data cannot be redistributed. See JUSTIFICATIONS.md §1.6.
#
# Reference: Metrum Research Group (2024). MeRGE Expo 1: NONMEM/FOCE PopPK.
#   https://merge.metrumrg.com/expo/expo1-nonmem-foce/
#
# On disk this dataset is intentionally NOT stored — we only clone on demand.
set -euo pipefail
OUT_DIR="${1:-benchmarks/datasets/metrum_expo1_nonmem}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

if [ ! -d _cloned ]; then
    git clone --depth 1 https://github.com/metrumresearchgroup/expo1-nonmem-foce _cloned
fi

DATA="_cloned/data/derived/pk.csv"
if [ -f "$DATA" ]; then
    rows=$(tail -n +2 "$DATA" | wc -l | tr -d ' ')
    subjects=$(tail -n +2 "$DATA" | awk -F, '{print $3}' | sort -u | wc -l | tr -d ' ')
    md5sum "$DATA" 2>/dev/null || md5 "$DATA"
    echo "pk.csv: $rows rows, $subjects subjects"
    echo "Model files under _cloned/model/pk/: $(ls _cloned/model/pk/*.yaml 2>/dev/null | wc -l) YAML"
fi
