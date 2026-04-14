# MIMIC-IV Vancomycin PK Dataset (Tier 2: Credentialed Access)

## Access Requirements

This dataset requires credentialed access to MIMIC-IV via PhysioNet:

1. Register at https://physionet.org/
2. Complete CITI Human Subjects Research training
3. Request access to MIMIC-IV: https://physionet.org/content/mimiciv/

## Dataset Description

- **Source**: MIMIC-IV v2.2+ (PhysioNet)
- **Population**: ~4,059 ICU sepsis patients (Sepsis-3 criteria)
- **Drug**: Vancomycin IV
- **Sampling**: Sparse TDM (predominantly troughs)
- **Covariates**: Renal function (SCr, eGFR, CrCl), body weight, age, SOFA/APACHE

## Reference

Zhang et al. (2025). "Machine learning and population pharmacokinetics:
a hybrid approach for optimizing vancomycin therapy in sepsis patients."
Microbiology Spectrum. doi:10.1128/spectrum.00499-25

## Extraction

See `prepare.sql` for the SQL extraction query against MIMIC-IV tables.
The `prepare.py` script converts the extracted data to APMODE canonical format.

## CI Policy

This dataset is **never** included in public CI. Set the environment variable
`APMODE_BENCH_DATA_ROOT` to include MIMIC-IV data in local benchmark runs.
