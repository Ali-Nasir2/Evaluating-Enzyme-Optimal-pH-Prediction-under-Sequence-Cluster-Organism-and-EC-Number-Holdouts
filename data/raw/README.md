# Raw Dataset

Place the original enzyme optimal-pH dataset files in this directory.

Do not edit files in this directory. The pipeline writes cleaned outputs to `data/processed/`.

Expected tabular columns are configured in `config/default.yaml`:

- `protein_id`
- `sequence`
- `optimal_ph`
- `organism`
- `taxonomy`
- `ec_number`

The only strictly required columns for Phase 1 are the sequence column and the experimental optimal-pH target column.

## Provenance

`pHopt_data.csv` is the pHopt enzyme optimal-pH dataset from the EpHod release:

- Gado, J. E., Knotts, M., Shaw, A. Y., et al. *Machine learning prediction of enzyme optimal pH*. Nature Machine Intelligence 7, 716--729 (2025). DOI: [10.1038/s42256-025-01026-6](https://doi.org/10.1038/s42256-025-01026-6).
- Dataset record: *Machine learning prediction of enzyme optimal pH*, Zenodo record [10.5281/zenodo.14252615](https://doi.org/10.5281/zenodo.14252615), published 2023-06-06, version 1.
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).
- File: `pHopt_data.csv`, 9,855 records, 4,841,831 bytes.
- Published MD5: `d5d0887710e70f2aa5153c15178a6d08`.
- Local SHA-256: `6C2CE7F68E28E4130904A96FC071DBF7FE78E090BD1E408408038F01FA75A3E9`.
- Access date for this analysis: 2026-10-22.
