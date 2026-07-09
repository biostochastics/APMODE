<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# Retrieval Summary — what is actually on disk

**Generated:** 2026-07-08. All artefacts produced by running each dataset's
`prepare.{R,py,sh}` script. Total: **89 MB** under `benchmarks/datasets/`.

## Verification: everything below has been fetched, not just documented

| Directory | Files retrieved | Size | Data on disk? | License | Committable? |
|---|---|---|---|---|---|
| `nlmixr2data_extended/` | 20 CSV + `_MANIFEST.csv` | 5.4 MB | ✅ | GPL-3.0-only | ✅ |
| `npde_reference_sims/` | `theopp.csv`, `warfarin_pk.csv`, `simwarfarinCov.csv` (247k rows) | 5.3 MB | ✅ | GPL (≥ 2) | ✅ |
| `medicaldata_indometh/` | `indometh.csv`, `indo_rct.csv` | 156 KB | ✅ | MIT + file LICENSE | ✅ |
| `r_base_datasets/` | `Theoph.csv`, `Indometh.csv` | 12 KB | ✅ | GPL-2 (R core) | ✅ |
| `saemix_theo_covariate/` | `theo_saemix.csv` (120 rows) | 8 KB | ✅ | GPL (≥ 2) | ✅ |
| `pmxnode_examples/` | 4 examples × (mlx.csv + nm.csv + mlxtran + ctl) = 16 files | 3.2 MB | ✅ | GPL (≥ 3) | ✅ (source files copied out of `_cloned/`; the clone itself is git-ignored) |
| `pkdb_api/` | `index.json` (88 open studies) + `index_summary.csv` + `studies/*.json` (70 full payloads with dataset/groupset/individualset/interventionset/outputset) | 1.5 MB | ✅ | Per-study `licence:open` (redistributable) | ✅ |
| `ddmore_wayback/` | 7 HTML pages (models index + 6 canonical model IDs from Wayback 2025-03-27 snapshot) | 548 KB | ✅ | Wayback / DDMoRe (Apache-2.0 spec) | ✅ (archived HTML) |
| `metrum_expo1_nonmem/_cloned/` | full repo (pk.csv 4360×34×160, 8 NONMEM model YAMLs) | 60 MB | ✅ | UNSPECIFIED | ❌ git-ignored — reproduce via `prepare.sh` |
| `metrum_expo4_torsten/_cloned/` | full repo (pk.csv identical to Expo 1 by MD5) | 3.5 MB | ✅ | UNSPECIFIED | ❌ git-ignored — reproduce via `prepare.sh` |
| `osp_observed_data/_cloned/` | DDI.csv (635 rows), Pediatrics.csv (277 rows), ObsDataPK_OSP.xlsx | 9.3 MB | ✅ | UNSPECIFIED | ❌ git-ignored — reproduce via `prepare.sh` |

## Directories previously in-tree (untouched, listed for completeness)

- `nlmixr2data_theophylline/`, `nlmixr2data_warfarin/`, `nlmixr2data_mavoglurant/`, `nlmixr2data_schoemaker/` — existing `prepare.R` scripts (no committed data)
- `ddmore_gentamicin/` — existing `prepare.py` (blocked on 503; use `ddmore_wayback/` as fallback)
- `opentci_propofol/` — existing `prepare.R`
- `mimic_vancomycin/` — existing `README.md` (Tier-2 credentialed, never in CI)

## How to reproduce (from scratch)

```bash
cd /Users/biostochastics/APMODE

# 1) R data packages — installs saemix/npde/medicaldata + writes all CSVs
Rscript benchmarks/datasets/nlmixr2data_extended/prepare.R
Rscript benchmarks/datasets/npde_reference_sims/prepare.R
Rscript benchmarks/datasets/medicaldata_indometh/prepare.R
Rscript benchmarks/datasets/r_base_datasets/prepare.R
Rscript benchmarks/datasets/saemix_theo_covariate/prepare.R

# 2) Git-based clones (reference-only where unlicensed)
bash benchmarks/datasets/metrum_expo1_nonmem/prepare.sh
bash benchmarks/datasets/metrum_expo4_torsten/prepare.sh
bash benchmarks/datasets/osp_observed_data/prepare.sh
bash benchmarks/datasets/pmxnode_examples/prepare.sh

# 3) API pulls
python3 benchmarks/datasets/pkdb_api/prepare.py     # 88 open studies + 70 payloads (~2 min)
python3 benchmarks/datasets/ddmore_wayback/prepare.py  # 7 pages from 2025-03-27 archive
```

## Coverage against APMODE lanes

| Lane / suite | Datasets now materialised |
|---|---|
| Submission (Suite A, ground-truth recovery) | `nlmixr2data_extended/` × 20 files (all closed-form structural fixtures) |
| Submission (Suite B, real-data anchors) | `r_base_datasets/` (Theoph + Indometh), `medicaldata_indometh/`, existing `nlmixr2data_theophylline` + `nlmixr2data_warfarin` |
| Discovery (Suite C, literature anchors) | `metrum_expo1_nonmem/_cloned/pk.csv` (via `prepare.sh`), `ddmore_wayback/` for 6 canonical DDMoRe IDs |
| Discovery — NODE benchmarks | `pmxnode_examples/` (4 fixtures × Monolix + NONMEM formats) |
| Discovery — VPC/NPE calibration | `npde_reference_sims/simwarfarinCov.csv` (247k rows) |
| Discovery — covariate model finding | `saemix_theo_covariate/theo_saemix.csv` (synthetic Sex ground truth) |
| Optimization / MIPD | `metrum_expo4_torsten/_cloned/pk.csv` (Bayesian workflow), `pkdb_api/` (88 studies for cross-validation cohorts) |
| Evidence Manifest stress (§4.2.1) | `pkdb_api/` (heterogeneous richness — 15 open studies without individual data → sparse; 36 with individual → rich) |

## What is NOT retrieved (deliberate)

- **T-DM1 raw data** — Genentech patient privacy; `Neural_PK` code repo is MIT but data is restricted. Use `Lu 2014 T-DM1 PopPK paper` (CC-BY) for prior elicitation instead.
- **DDMoRe live models** — 503 outage; Wayback snapshot from 2025-03-27 substituted.
- **`simwarfarinBase`** — not in CRAN npde v3.5; ships only on github.com/ecomets/npde30. Not critical (`simwarfarinCov` is the load-bearing calibration set).
- **PK-DB `closed` license studies** — 92.5% of PK-DB corpus. Metadata is public but data redistribution is per-study restricted.

## Full audit trail

- Every retrieval command logged in `fetch_ledger.jsonl` (68 records).
- Every citation in `CITATIONS.bib` (32 BibTeX entries).
- License provenance in `manifest.json` (25 datasets) with `license_verified_via` field.
