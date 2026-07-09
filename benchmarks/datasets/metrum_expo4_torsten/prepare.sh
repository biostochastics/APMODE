#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Retrieve Metrum MeRGE Expo 4 (Torsten Bayesian) data + model files.
#
# License caveat: NO LICENSE file at HEAD (verified 2026-07-08). Reference-only.
#
# Reference: Metrum Research Group (2024). MeRGE Expo 4: Torsten Bayesian PopPK.
#   https://merge.metrumrg.com/expo/expo4-torsten-poppk/
#
# Replaces the dead bugs_model_library (Bitbucket 404) as the
# canonical Bayesian PKPD reference set.
#
# The `pk.csv` shipped with Expo 4 is byte-identical to Expo 1's (MD5
# 84a596d0fd81c2d187a64bc62d96b433, verified 2026-07-08). This is intentional —
# same dataset lets classical FOCE and Bayesian NUTS methods be compared apples-to-apples.
set -euo pipefail
OUT_DIR="${1:-benchmarks/datasets/metrum_expo4_torsten}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

if [ ! -d _cloned ]; then
    git clone --depth 1 https://github.com/metrumresearchgroup/expo4-torsten-poppk _cloned
fi

DATA="_cloned/data/derived/pk.csv"
if [ -f "$DATA" ]; then
    rows=$(tail -n +2 "$DATA" | wc -l | tr -d ' ')
    subjects=$(tail -n +2 "$DATA" | awk -F, '{print $3}' | sort -u | wc -l | tr -d ' ')
    md5sum "$DATA" 2>/dev/null || md5 "$DATA"
    echo "pk.csv: $rows rows, $subjects subjects"
    echo "Torsten library dependency: github.com/metrumresearchgroup/Torsten"
fi
