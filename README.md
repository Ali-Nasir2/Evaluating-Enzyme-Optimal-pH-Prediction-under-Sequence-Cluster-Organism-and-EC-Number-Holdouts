# Evaluating Enzyme Optimal-pH Prediction under Sequence-Cluster, Organism, and EC-Number Holdouts

This repository is a reproducible research codebase for studying whether enzyme optimal-pH prediction models generalize beyond closely related proteins, organisms, enzyme classes, or sequence clusters represented during training.

The project starts with conservative data loading and validation. It does not fabricate datasets, metrics, citations, or biological conclusions.

## Current Status

Phase 1 scaffold is implemented:

- Repository structure
- YAML configuration
- Dependency files
- Dataset loading module
- Dataset validation module
- Data-quality report generation
- Unit tests for loading and validation
- LaTeX paper scaffold
- Handcrafted sequence-feature generation
- Initial baseline training on the dataset-provided split
- Reproducible random and exact-duplicate-safe split generation
- FASTA export and external sequence-clustering command generation
- Baseline training across generated split strategies
- Organism-held-out split generation

The current baseline metrics are preliminary. Generated random, exact-duplicate-safe, 90% MMseqs2 cluster-safe, and organism-held-out split definitions are available in `data/splits/`. EC-held-out splits are a later phase.

## Dataset Placement

Place the real enzyme optimal-pH dataset in:

```text
data/raw/
```

Supported input formats:

- `.csv`
- `.tsv`
- `.txt` tab-separated files
- `.json`
- `.jsonl`
- `.xlsx` or `.xls`
- `.fasta`, `.fa`, `.faa`, `.fas`

For tabular datasets, the expected columns are configured in `config/default.yaml`.

Default expected column names:

| Field | Default column | Required |
| --- | --- | --- |
| Protein or enzyme identifier | `protein_id` | recommended |
| Amino-acid sequence | `sequence` | yes |
| Experimental optimal pH | `optimal_ph` | yes |
| Organism | `organism` | recommended |
| Taxonomy | `taxonomy` | optional |
| EC number | `ec_number` | optional |

If your dataset uses different names, edit `config/default.yaml` before running the pipeline.

For FASTA files, sequence identifiers are read from the FASTA headers, but optimal-pH labels and metadata usually require a separate tabular metadata file. The validator will clearly report missing target values if only FASTA sequences are present.

## Initial Data Inspection

After placing the dataset in `data/raw/`, run:

```bash
python scripts/prepare_data.py --config config/default.yaml
```

This will:

- Detect dataset files in `data/raw/`
- Load supported file formats
- Validate required columns
- Check sequence validity
- Check duplicate identifiers and sequences
- Check pH range
- Save a cleaned dataset to `data/processed/cleaned_dataset.csv`
- Save a validation summary to `reports/data_quality_report.md`

The raw dataset is never modified.

## Installation

Create a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Alternatively, with conda:

```bash
conda env create -f environment.yml
conda activate enzyme-ph-generalization
```

For Phase 7 protein embeddings, use a Python 3.12 environment because PyTorch on Windows currently supports Python 3.9-3.12:

```bash
conda env create -f environment-embeddings.yml
conda activate enzyme-ph-embeddings
```

## Tests

Run:

```bash
pytest
```

## Planned Workflow

The intended command workflow is:

```bash
python scripts/prepare_data.py --config config/default.yaml
python scripts/generate_features.py --config config/default.yaml
python scripts/generate_embeddings.py --config config/default.yaml
python scripts/create_splits.py --config config/default.yaml
python scripts/train_baselines.py --config config/default.yaml
python scripts/train_embedding_models.py --config config/default.yaml
python scripts/evaluate_models.py --config config/default.yaml
python scripts/run_full_pipeline.py --config config/default.yaml
```

Only `prepare_data.py` is implemented in Phase 1. Later scripts are placeholders until their phases are reached.

Currently implemented:

```bash
python scripts/prepare_data.py --config config/default.yaml
python scripts/generate_features.py --config config/default.yaml
python scripts/cluster_sequences.py --config config/default.yaml
python scripts/create_splits.py --config config/default.yaml
python scripts/train_baselines.py --config config/default.yaml --include-dataset-split
python scripts/generate_embeddings.py --config config/default.yaml
python scripts/train_embedding_models.py --config config/default.yaml --include-dataset-split
python scripts/evaluate_models.py --config config/default.yaml
python scripts/analyze_errors.py --config config/default.yaml
python scripts/generate_figures.py --config config/default.yaml
```

## Sequence Clustering

Phase 5 prepares clustering inputs and parsers, but real sequence-similarity clusters require an external protein clustering tool.

Create the FASTA file and clustering command file:

```bash
python scripts/cluster_sequences.py --config config/default.yaml
```

This writes:

```text
data/processed/clusters/sequences.fasta
data/processed/clusters/run_clustering_commands.ps1
```

Install MMseqs2 or CD-HIT, then run the command block for one tool and threshold from `run_clustering_commands.ps1`. After the raw clustering output exists, parse it, for example:

```bash
python scripts/cluster_sequences.py --config config/default.yaml --parse-mmseqs data/processed/clusters/mmseqs_identity_90_cluster.tsv --identity 0.90
python scripts/create_splits.py --config config/default.yaml
```

Cluster-based split files are generated only after real clustering assignments are available.

## Baseline Split Comparison

After cluster assignments exist, train baselines across all generated split definitions:

```bash
python scripts/train_baselines.py --config config/default.yaml --include-dataset-split
```

This writes:

```text
outputs/metrics/baseline_metrics_by_strategy.csv
outputs/metrics/baseline_generalization_comparison.csv
outputs/predictions/baseline_predictions_by_strategy.csv
```

## Protein Embeddings

Phase 7 supports frozen ESM-2 embeddings using Hugging Face Transformers. The default model is:

```text
facebook/esm2_t6_8M_UR50D
```

Generate cached embeddings:

```bash
python scripts/generate_embeddings.py --config config/default.yaml
```

This writes:

```text
data/processed/embeddings/esm2_t6_8M_UR50D_embeddings.npy
data/processed/embeddings/esm2_t6_8M_UR50D_metadata.json
```

Train embedding-based regressors across split strategies:

```bash
python scripts/train_embedding_models.py --config config/default.yaml --include-dataset-split
```

This writes:

```text
outputs/metrics/embedding_metrics_by_strategy.csv
outputs/metrics/embedding_generalization_comparison.csv
outputs/predictions/embedding_predictions_by_strategy.csv
```

Feature-group comparison outputs:

```text
outputs/metrics/feature_group_test_metrics.csv
outputs/metrics/feature_group_mae_comparison.csv
```

## Evaluation Tables

Create consistent evaluation outputs after baseline and embedding predictions exist:

```bash
python scripts/evaluate_models.py --config config/default.yaml
```

This writes:

```text
outputs/metrics/bootstrap_confidence_intervals.csv
outputs/metrics/model_comparison_table.csv
outputs/metrics/performance_by_ph_range.csv
outputs/metrics/performance_by_subgroup.csv
outputs/metrics/paired_feature_group_comparisons.csv
outputs/predictions/top_50_errors_by_model.csv
reports/tables/model_comparison_table.tex
reports/tables/performance_by_ph_range.tex
paper/tables/model_comparison_table.tex
paper/tables/performance_by_ph_range.tex
```

## Error Analysis and Figures

Generate focused error-analysis tables:

```bash
python scripts/analyze_errors.py --config config/default.yaml
```

Generate publication figures:

```bash
python scripts/generate_figures.py --config config/default.yaml
```

Figures are saved as both PNG and PDF in:

```text
reports/figures/
```

## Paper

The LaTeX manuscript scaffold is in `paper/`.

Preferred build command:

```bash
cd paper
latexmk -pdf main.tex
```

Alternative:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The paper intentionally contains placeholders for results, tables, figures, and citations until real experiments are complete.

## Scientific Safeguards

This project will not:

- Generate fake biological data
- Invent metrics or results
- Split exact duplicate sequences across train and test sets
- Fit scalers or feature-selection logic before splitting
- Tune models on test data
- Claim biological mechanisms from correlations alone
- Claim significance without statistical analysis
