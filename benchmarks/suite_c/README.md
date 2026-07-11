# Suite C — Phase-1 Literature-Anchor Fixtures

> APMODE plan Tasks 38–43, blueprint in `docs/plans/2026-04-24-apmode-v0.6-completion.md`.

Phase-1 of Suite C compares APMODE's PK fits against published, peer-reviewed
reference parameterizations and selected ground-truth reference datasets. Each fixture pairs a
NONMEM-style CSV (resolved via `dataset_id` against
`benchmarks/datasets/registry.yaml`) with a `DSLSpec` JSON and a
`LiteratureFixture` YAML capturing the reference parameter values plus the
parameterization mapping needed to translate published symbol names
(e.g. `TVCL`) into APMODE's DSL-canonical names (e.g. `CL`).

## Fixtures (loaded by `apmode.benchmarks.literature_loader.PHASE1_MLE_FIXTURE_IDS`)

| dataset_id | route | DSL skeleton | reference DOI |
|------------|-------|--------------|---------------|
| `theophylline_boeckmann_1992` | oral | 1-cmt + FO ka | [10.1002/psp4.12471](https://doi.org/10.1002/psp4.12471) (Schoemaker et al. 2019, nlmixr SAEM/FOCEI grid which contains the Theoph fit) |
| `warfarin_funaki_2018` | oral | 1-cmt + lagged-FO ka | [10.1002/psp4.12445](https://doi.org/10.1002/psp4.12445) (Fidler et al. 2019, nlmixr documentation including Holford warfarin) |
| `mavoglurant_wendling_2015` | oral | 2-cmt + FO ka | [10.1007/s11095-014-1574-1](https://doi.org/10.1007/s11095-014-1574-1) (Wendling et al. 2015, mavoglurant population PK; the Phase-1 fixture uses the simpler 2-cmt approximation, while SumIG absorption remains a v0.7 preview path) |
| `phenobarbital_grasela_1985` | iv_bolus | 1-cmt | [10.1159/000457062](https://doi.org/10.1159/000457062) (Grasela & Donn 1985, phenobarbital neonatal PK) |
| `oral_1cpt_acop_2016` | oral | 1-cmt + FO ka | [10.32614/CRAN.package.nlmixr2data](https://doi.org/10.32614/CRAN.package.nlmixr2data) (ACOP 2016 / `nlmixr2data::Oral_1CPT` ground-truth recovery fixture) |
| `gentamicin_germovsek_2017` | iv_bolus | 1-cmt | [10.1128/AAC.00577-16](https://doi.org/10.1128/AAC.00577-16) (Germovsek et al. 2016, gentamicin IOV neonates) |
| `schoemaker_nlmixr2_tutorial` | iv_bolus | 1-cmt | [10.1002/psp4.12471](https://doi.org/10.1002/psp4.12471) (Schoemaker et al. 2019, `pkBolus1cmt` reference values) |

## Why the names changed from the original plan draft

The plan called for `mavoglurant_wang_2007.yaml` and `gentamicin_decock_2014.yaml`.
A literature search produced no Wang 2007 mavoglurant publication — mavoglurant
(Novartis AFQ056) population PK was first published by Wendling et al. (2015).
For gentamicin, the De Cock 2014 paper covers gentamicin/tobramycin/vancomycin
*jointly* (DOI [10.1007/s11095-014-1361-z](https://doi.org/10.1007/s11095-014-1361-z))
and reports population CL for a 4-kg full-term neonate (`Cldrug = 0.21 L/h`),
but the gentamicin-specific IOV model with a separate intercompartmental
covariance structure that the `ddmore_gentamicin` dataset is fit to is
Germovsek et al. 2016 (DOI
[10.1128/AAC.00577-16](https://doi.org/10.1128/AAC.00577-16)).
Both papers are cited in the registry; the fixture uses the Germovsek IOV
parameterization because it matches the dataset prepared by
`benchmarks/datasets/ddmore_gentamicin/prepare.py`.

## Metric definition and current scorecard status

The `npe_apmode` / `npe_literature` fields in `phase1_npe_inputs.json`
are APMODE's own **NPE** (Nonparametric Prediction Error): the median
absolute prediction error from posterior-predictive simulations, per
`apmode.benchmarks.scoring.compute_npe`. **This is not the classical
Comets/Mentré NPDE** (Normalised Prediction Distribution Error — a
mean≈0/variance≈1 z-score-like diagnostic from full Monte-Carlo
simulation; Comets, Brendel & Mentré 2008,
[10.1016/j.cmpb.2007.12.002](https://doi.org/10.1016/j.cmpb.2007.12.002)).
The two share the "NPE" substring by coincidence of naming, not
methodology — do not present a Suite C win/loss as an NPDE-validated
result; it is a proprietary, residual-scaled proxy comparison between
two fits of the same DSL spec on the same held-out fold (see
`suite_c_phase1_runner.py`'s "Honest mode" docstring for exactly what
is and isn't controlled for).

`phase1_npe_inputs.json` also carries `pit_calibration_apmode` /
`pit_calibration_literature` — the median-across-folds PIT/NPDE-lite
calibration check (`PITCalibrationSummary`, the same mechanism Gate 1
uses: does the predictive CDF hit its nominal coverage, `calibration["p50"]
≈ 0.50`?). It was already computed by `build_predictive_diagnostics` for
every Suite C fit but discarded before this field existed. It is a real
calibration diagnostic (closer in spirit to NPDE than the NPE point-
accuracy number above, though still without decorrelation or the formal
distributional test battery) and is reported for visibility in
`render_markdown_summary`'s output — it does **not** feed the win/loss
gate.

**True decorrelated NPDE now exists** as of the "Implement true NPDE"
plan gap: `apmode.backends.predictive_summary._compute_npde` Cholesky-
decorrelates each subject's simulated replicate covariance (Comets &
Brendel 2008 eq. 3-5), ECDF-ranks and normal-quantile-transforms, and
pools across subjects into `DiagnosticBundle.npde`
(`NPDESummary`) — a Wilcoxon (mean=0) / Shapiro-Wilk (normality) /
chi-square (var=1) battery with Bonferroni correction, **not** a KS
test (correcting an earlier draft of this note that misnamed the
battery). It is opt-in at Gate 1 via `Gate1Config.npde_required`
(default `False`) and additive to, not a replacement for, the
PIT/NPDE-lite check above. **`phase1_npe_inputs.json` / `suite_c_phase1_runner.py`
/ `suite_c_phase1_scoring.py` have not been extended to surface
`npde` fields** — that wiring is a still-open follow-up, not part of
this pass. Until it lands, Suite C scorecards continue to report only
`npe_*` (point-accuracy proxy) and `pit_calibration_*` (marginal
calibration); do not read their absence as "NPDE was checked and
passed."

The committed `phase1_npe_inputs.json` was last regenerated
2026-04-25 and reports `fraction_beats_literature_median = 40%
(2/5)` — **below** the `>= 60%` CI target
(`PHASE1_FRACTION_BEATS_TARGET`). **There is no scheduled CI job for
Suite C.** An earlier weekly cron workflow
(`.github/workflows/suite_c_phase1.yml`) was removed because it only
re-scored this static committed JSON via
`suite_c_phase1_cli.py` — pure arithmetic, no R, no `nlmixr2`, no live
fit — and a perpetually-green weekly check created a false impression
of ongoing live validation. The scoring math it ran is still available
and correct; it is just not automated on a schedule.

Regenerating and scoring the snapshot is a manual, on-demand operator
step:

1. Regenerate `phase1_npe_inputs.json` with a live run:
   `python -m apmode.benchmarks.suite_c_phase1_runner` (requires R 4.4+
   with a compiled `nlmixr2`; see "Metric definition" above for what
   the honest-mode fit contract guarantees).
2. Score the regenerated snapshot:
   `python -m apmode.benchmarks.suite_c_phase1_cli --inputs
   benchmarks/suite_c/phase1_npe_inputs.json --out scorecard.json
   --markdown-summary scorecard.md`.
3. Commit the refreshed `phase1_npe_inputs.json` alongside the change
   that motivated the re-run, and update the "last regenerated" date
   and scorecard numbers in this README.

Before citing Suite C results as evidence of methodology improvement,
confirm (a) the scorecard has been regenerated via a live
`suite_c_phase1_runner.py` run since the change in question landed, and
(b) `apmode.data.initial_estimates.NCAEstimator` was not seeded with
`fallback_estimates` derived from the fixture's own
`reference_params` (see
`tests/unit/test_suite_c_phase1_runner.py::test_run_fixture_apmode_side_never_seeded_from_literature_reference_params`
for the regression test that pins this).

## Adding a new fixture

1. Decide on a stable `dataset_id` (snake_case, lowercase).
2. Add a `<dataset_id>.dsl.json` next to this README — emit the DSL spec via
   `DSLSpec(...).model_dump_json(indent=2)`.
3. Add a `<dataset_id>.yaml` carrying the `LiteratureReference`,
   `reference_params`, and `parameterization_mapping`.
4. Append the new id to `PHASE1_MLE_FIXTURE_IDS` in
   `src/apmode/benchmarks/literature_loader.py` so the integration test
   picks it up.
5. `suite_c_phase1_runner.py` and `suite_c_phase1_cli.py` iterate over
   `PHASE1_MLE_FIXTURE_IDS` directly — no additional registration
   required. Re-run both manually per the "Metric definition" section
   above to refresh the committed scorecard.
