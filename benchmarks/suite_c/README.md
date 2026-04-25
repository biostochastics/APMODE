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
| `gentamicin_germovsek_2017` | iv_bolus | 1-cmt | [10.1128/AAC.02659-16](https://doi.org/10.1128/AAC.02659-16) (Germovsek et al. 2017, gentamicin IOV neonates) |
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
Germovsek et al. 2017. Both papers are cited in the registry; the fixture
uses the Germovsek IOV parameterization because it matches the dataset card
already on disk (`benchmarks/datasets/ddmore_gentamicin`).

## Adding a new fixture

1. Decide on a stable `dataset_id` (snake_case, lowercase).
2. Add a `<dataset_id>.dsl.json` next to this README — emit the DSL spec via
   `DSLSpec(...).model_dump_json(indent=2)`.
3. Add a `<dataset_id>.yaml` carrying the `LiteratureReference`,
   `reference_params`, and `parameterization_mapping`.
4. Append the new id to `PHASE1_MLE_FIXTURE_IDS` in
   `src/apmode/benchmarks/literature_loader.py` so the integration test
   picks it up.
5. The Suite C scoring CI workflow (Task 41) iterates over
   `PHASE1_MLE_FIXTURE_IDS` directly — no additional registration required.
