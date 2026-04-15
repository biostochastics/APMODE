#!/usr/bin/env bash
# Full end-to-end benchmark: runs `apmode run` on all Suite A scenarios
# plus the three real datasets (warfarin, theo_sd, mavoglurant), writes
# bundles to benchmarks/runs/full-<timestamp>/, and summarises.

set -u -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/benchmarks/runs/full-$STAMP"
mkdir -p "$OUT"

echo "Full benchmark run -> $OUT"
echo

FIXTURES=(
  "$ROOT/tests/fixtures/suite_a/a1_1cmt_oral_linear.csv:submission"
  "$ROOT/tests/fixtures/suite_a/a2_2cmt_iv_parallel_mm.csv:discovery"
  "$ROOT/tests/fixtures/suite_a/a3_transit_1cmt_linear.csv:submission"
  "$ROOT/tests/fixtures/suite_a/a4_1cmt_oral_mm.csv:discovery"
  "$ROOT/tests/fixtures/suite_a/a5_tmdd_qss.csv:discovery"
  "$ROOT/tests/fixtures/suite_a/a6_1cmt_covariates.csv:submission"
  "$ROOT/tests/fixtures/suite_a/a7_2cmt_node_absorption.csv:discovery"
  "$ROOT/benchmarks/data/warfarin.csv:submission"
  "$ROOT/benchmarks/data/theo_sd.csv:submission"
  "$ROOT/benchmarks/data/mavoglurant.csv:discovery"
)

for entry in "${FIXTURES[@]}"; do
  csv="${entry%%:*}"
  lane="${entry##*:}"
  name="$(basename "$csv" .csv)"
  bundle="$OUT/$name"
  mkdir -p "$bundle"
  echo "[run] $name (lane=$lane) -> $bundle"
  uv run apmode run "$csv" --lane "$lane" --output "$bundle" \
    > "$bundle/stdout.log" 2> "$bundle/stderr.log" \
    && echo "  OK" || echo "  FAIL (see stderr.log)"
done

echo
echo "Bundles written to $OUT"
echo "Summary:"
for entry in "${FIXTURES[@]}"; do
  csv="${entry%%:*}"
  name="$(basename "$csv" .csv)"
  bundle="$OUT/$name"
  manifest="$bundle/evidence_manifest.json"
  if [[ -f "$manifest" ]]; then
    strength=$(uv run python -c "import json; m=json.load(open('$manifest')); print(m.get('nonlinear_clearance_evidence_strength','?'),m.get('compartmentality','?'),m.get('flip_flop_risk','?'))" 2>/dev/null || echo "? ? ?")
    echo "  $name: $strength"
  else
    echo "  $name: no manifest"
  fi
done
