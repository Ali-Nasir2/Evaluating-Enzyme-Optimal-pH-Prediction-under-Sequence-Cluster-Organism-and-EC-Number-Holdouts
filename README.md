# Evaluating Enzyme Optimal-pH Prediction under Sequence-Cluster, Organism, and EC-Number Holdouts

This repository contains the code, configurations, split definitions, selected
metrics, and diagnostic artifacts for evaluating enzyme optimal-pH prediction
under increasingly demanding distribution shifts. The evaluated strategies are:

- generated random and exact-sequence-disjoint splits;
- MMseqs2 sequence-cluster holdouts at 90%, 70%, and 50% nominal identity;
- organism-held-out splits;
- complete EC-number-held-out splits; and
- the dataset-provided train/validation/test partition.

The repository contains reproducibility materials, not the manuscript. The
paper source and PDF are intentionally excluded.

## Dataset and provenance

The original data are the EpHod/pHopt enzyme optimal-pH dataset:

- Gado, J. E., Knotts, M., Shaw, A. Y., et al. “Machine learning prediction
  of enzyme optimal pH.” *Nature Machine Intelligence* 7, 716–729 (2025).
  [Publication DOI](https://doi.org/10.1038/s42256-025-01026-6)
- Zenodo record: [10.5281/zenodo.14252615](https://doi.org/10.5281/zenodo.14252615)
- Dataset version: 1, published 2023-06-06
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Download `pHopt_data.csv` from Zenodo and place it in `data/raw/`. The raw
dataset is not redistributed in this repository. The complete provenance
record, expected file details, and checksums are in
[`data/raw/README.md`](data/raw/README.md).

The analysis used 9,855 cleaned records. The original dataset-provided labels
contain 7,124 training, 760 validation, and 1,971 testing records. The
generated random split contains 6,898 training, 1,478 validation, and 1,479
testing records; these are different partitions and should not be conflated.

## Repository contents

- [`src/`](src/) — feature extraction, embeddings, splitting, models, and
  evaluation utilities.
- [`scripts/`](scripts/) — data preparation, split creation, training,
  evaluation, source-label, similarity-audit, and repeated-seed commands.
- [`config/`](config/) — experiment configuration.
- [`tests/`](tests/) — automated tests.
- [`data/splits/`](data/splits/) — generated partition assignments and parsed
  cluster assignments.
- [`outputs/metrics/`](outputs/metrics/) — saved metrics, confidence intervals,
  paired comparisons, subgroup results, and selected diagnostics.
- [`outputs/metrics/cluster_similarity_audit_20261025/`](outputs/metrics/cluster_similarity_audit_20261025/)
  — returned-hit MMseqs2 similarity audit, settings, commands, missing-hit
  records, and partition copies.
- [`outputs/metrics/repeated_grouped_300tree_20261025/`](outputs/metrics/repeated_grouped_300tree_20261025/)
  — documented five-seed (42–46) 300-tree grouped evaluation, target
  distributions, settings, commands, and partition assignments.
- [`reports/`](reports/) — data-quality, experiment-summary, and source-label
  audit reports.

## Environment and installation

The main environment files are [`requirements.txt`](requirements.txt),
[`environment.yml`](environment.yml), and
[`environment-embeddings.yml`](environment-embeddings.yml). The saved
embedding run used Python 3.12.13, PyTorch 2.11.0+cu128, and
`facebook/esm2_t6_8M_UR50D`.

Create an environment and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the test suite with:

```powershell
pytest
```

## Reproduction workflow

After downloading the source CSV into `data/raw/`, the main workflow is:

```powershell
python scripts/prepare_data.py --config config/default.yaml
python scripts/generate_features.py --config config/default.yaml
python scripts/create_splits.py --config config/default.yaml
python scripts/train_baselines.py --config config/default.yaml --include-dataset-split
python scripts/generate_embeddings.py --config config/default.yaml
python scripts/train_embedding_models.py --config config/default.yaml --include-dataset-split
python scripts/evaluate_models.py --config config/default.yaml
python scripts/analyze_errors.py --config config/default.yaml
python scripts/generate_figures.py --config config/default.yaml
```

The checked-in metrics and diagnostics are the saved outputs from the completed
analysis. Re-running the workflow may require the local dataset, the ESM-2
model cache, and MMseqs2.

## Models and features

The handcrafted input contains 35 numeric sequence features. Metadata and
targets (`Accession`, `pHopt`, `Organism`, `EC Number`, and `Split`) are
excluded from model inputs. The learned models use median imputation; Ridge
also uses standard scaling. The main random forest uses 300 trees and
`random_state=42`. Histogram gradient boosting is configured with
`max_iter=300` and library-default early-stopping behavior.

The embedding model is frozen ESM-2
`facebook/esm2_t6_8M_UR50D`. Embeddings use the final hidden state with mean
pooling over residue tokens, excluding special and padding tokens.

## Sequence-cluster evaluation and similarity audit

MMseqs2 clustering was run with `easy-cluster`, `-c 0.80`,
`--cov-mode 0`, `--threads 4`, and nominal minimum identities of 0.50, 0.70,
and 0.90. The saved executable hash and exact commands are recorded in the
similarity-audit output directory.

The saved audit is explicitly a **returned-hit audit**: it used
`easy-search --max-seqs 1`, so it does not prove that the returned hit is the
globally highest-scoring qualifying training hit. Identity is reported in
percentage units by MMseqs2. Coverage-qualified exceedance counts and
100%-identity returned hits are documented in the saved audit files. These
results should not be described as exhaustive nearest-neighbor verification.

## Statistical and robustness artifacts

[`outputs/metrics/paired_feature_group_comparisons.csv`](outputs/metrics/paired_feature_group_comparisons.csv)
contains 40 paired comparison rows: eight split strategies across five model
types. The saved Holm correction family therefore contains 40 rows, including
24 learned-model comparisons. Paired MAE differences and bootstrap confidence
intervals are included.

The full documented repeated grouped evaluation is in
[`outputs/metrics/repeated_grouped_300tree_20261025/`](outputs/metrics/repeated_grouped_300tree_20261025/).
It evaluates organism-held-out and EC-held-out strategies for seeds 42–46
using the main 300-tree random-forest configuration. Sample standard
deviations are calculated with `ddof=1`.

The source-label audit, including the questioned B5LXD7 record and sensitivity
outputs, is documented in
[`reports/source_label_audit.csv`](reports/source_label_audit.csv) and
[`outputs/metrics/source_label_sensitivity.csv`](outputs/metrics/source_label_sensitivity.csv).

## Reproducibility and limitations

Split assignments, commands, settings, and generated results are preserved
where available. The repository does not claim that cluster-disjoint
assignments alone prove a complete homology exclusion. In particular, the
returned-hit audit has the sensitivity limitation described above. The
published source-label critique also means that disagreement with an extreme
reported pH value should not automatically be interpreted as model failure.
