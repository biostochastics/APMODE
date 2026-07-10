# policies/

Versioned gate-threshold JSON. Each file is a `GatePolicy`
(`src/apmode/governance/policy.py`), one per lane
(`submission.json`, `discovery.json`, `optimization.json`) plus
`profiler.json` for `EvidenceManifest` richness/coverage thresholds.

Gate thresholds are policy artifacts, not hard-coded constants — edit the
JSON (or pass `--policy <path>` to `apmode run`) to change a threshold.
Never patch a numeric literal into `governance/gates.py`,
`governance/policy.py`, or a backend module instead.

## Gate calibration reports are a reference, not an auto-tuner

`benchmarks/calibration/reports/gate{1,2}_calibration_<policy_version>.json`
(produced by `scripts/run_gate_calibration.py`, backed by
`src/apmode/governance/calibration.py`) cross-tabulates each Gate 1/2
check's pass/fail verdict against a ground-truth correctness label
derived from Suite A `reference_params.json` (and, once populated, Suite
B perturbed fixtures) — giving a per-check `false_pass_rate` /
`false_fail_rate` a policy author can look at before hand-tuning a
threshold.

That report is descriptive, not prescriptive:

- It never writes to a `policies/*.json` file itself.
- A high `false_pass_rate` or `false_fail_rate` is a signal to a human
  policy author to *consider* a threshold change — it is not applied
  automatically, and the calibration harness has no write path into this
  directory.
- Any threshold change made in response to a calibration finding must
  still go through the normal `policies/*.json` edit + `policy_version`
  bump (see `GatePolicy.policy_version`), same as any other threshold
  change, so the change is itself versioned and auditable.
- At Suite A/B's expected small per-scenario n (~21 Suite A scenarios;
  Suite B fixtures not yet generated as of this writing —
  `benchmarks/suite_b/` is a README stub), per-check rates are a
  directional signal, not a statistically tight estimate. The report
  deliberately omits Wilson/Clopper-Pearson confidence intervals for this
  reason (see `GateCalibrationReport.notes`).

See `docs/plans/2026-07-09-qaqc-remediation.md` ("Empirically calibrate
Gate1/2/3 thresholds against Suite A/B false-pass/fail rates") for the
design rationale.
