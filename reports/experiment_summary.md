# Experiment Summary

## Research Question

Do enzyme optimal-pH prediction models maintain their performance when evaluated on protein families, sequence clusters, enzyme classes, or organisms that were not represented during training?

## Dataset Description

The input is the EpHod pHopt enzyme optimal-pH dataset: 9,855 protein sequences with accession, organism, EC number, experimentally measured optimal-pH, source split labels, and sequence columns. It was obtained from Zenodo record `10.5281/zenodo.14252615`, version 1, under CC BY 4.0, accessed 2026-10-22. The published file MD5 is `d5d0887710e70f2aa5153c15178a6d08`; the local file SHA-256 is `6C2CE7F68E28E4130904A96FC071DBF7FE78E090BD1E408408038F01FA75A3E9`. The associated publication is Gado et al., *Nature Machine Intelligence* 7, 716--729 (2025), DOI `10.1038/s42256-025-01026-6`.

## Cleaning Decisions

[Insert cleaning decisions from `reports/data_quality_report.md`.]

## Source-label audit

The published data-quality critique by Zhou and Fang (Briefings in Bioinformatics, 2026, DOI `10.1093/bib/bbag543`) identifies disputed pH values and protein-identity assignments in the EpHod pHopt resource. The named accession B5LXD7 is present once in this dataset as *Bispora sp.*, EC 3.2.1.78, with pHopt 11.5. It appears among the largest saved prediction errors and is in the test sets for the 50% cluster-safe and EC-held-out strategies. The cited underlying Bispora study is Luo et al. (2009, DOI `10.1007/s00253-008-1766-x`), which is consistent with the acidic-source concern reported by Zhou and Fang. No label was silently corrected. The reproducible audit is in `scripts/audit_source_labels.py`, with record-level findings in `reports/source_label_audit.csv` and leave-one-flagged-record-out sensitivity results in `outputs/metrics/source_label_sensitivity.csv`. For the best ESM-2 models, excluding B5LXD7 changed MAE by -0.0038 on the 50% split and -0.0037 on the EC-held-out split.

## Split Strategies

Phase 4 generated two reproducible split definitions in `data/splits/`:

| Strategy | Train | Validation | Test | Leakage groups |
| --- | ---: | ---: | ---: | ---: |
| Random | 6898 | 1478 | 1479 | 39 exact-duplicate groups |
| Exact-duplicate-safe | 6895 | 1480 | 1480 | 0 exact-duplicate groups |
| MMseqs2 90\% cluster-safe | 6902 | 1480 | 1473 | 0 cluster groups |
| Organism-held-out | 6899 | 1478 | 1478 | 0 organism groups |

The random split is intentionally row-wise and is retained as a comparison condition. The exact-duplicate-safe split assigns all identical amino-acid sequences to the same partition.

Sequence-similarity cluster split creation is implemented for MMseqs2 outputs. A 90\% identity MMseqs2 cluster-safe split has been generated from external clustering output. Phase 6 added organism-held-out and EC-held-out splits using the `Organism` and `EC Number` columns. Taxonomy-level splits beyond organism remain planned work.

Phase 5 clustering preparation exported cleaned sequences to `data/processed/clusters/sequences.fasta` and wrote external MMseqs2/CD-HIT commands to `data/processed/clusters/run_clustering_commands.ps1`. The 90\% identity MMseqs2 clustering output assigned 9855 sequences to 9025 clusters.

## Models

Phase 3 implemented initial handcrafted-feature baselines:

- Mean predictor
- Ridge regression
- Random forest regressor
- Histogram gradient boosting regressor

Phase 7 implemented frozen protein-embedding support using the configured ESM-2 model followed by the same regression heads. Embedding generation is cached and records model name, sequence checksum, embedding dimension, device, batch size, and maximum sequence length.

## Hyperparameters

Current baseline hyperparameters are stored in `config/default.yaml`.

## Metrics

Primary metric: mean absolute error in pH units.

Additional metrics: RMSE, R-squared, Spearman correlation, Pearson correlation.

## Preliminary Phase 3 Baseline Results

These results use the dataset-provided split in `pHopt_data.csv`. They are not yet leakage-controlled generalization results.

| Model | Evaluation split | MAE | RMSE | R-squared | Spearman | Pearson |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mean predictor | Test | 0.8481 | 1.1552 | -0.0011 | NA | NA |
| Ridge regression | Test | 0.7573 | 1.0203 | 0.2191 | 0.3809 | 0.4728 |
| Random forest | Test | 0.6727 | 0.9343 | 0.3451 | 0.4831 | 0.5951 |
| Histogram gradient boosting | Test | 0.6883 | 0.9445 | 0.3307 | 0.4511 | 0.5781 |

## Preliminary Split-Strategy Baseline Comparison

These results use handcrafted sequence features only. They compare generated row-wise random, exact-duplicate-safe, MMseqs2 50\%, 70\%, and 90\% cluster-safe, and organism-held-out splits. Statistical uncertainty has not yet been estimated.

| Model | Random MAE | Exact-duplicate-safe MAE | MMseqs2 90\% cluster-safe MAE | Organism-held-out MAE |
| --- | ---: | ---: | ---: | ---: |
| Mean predictor | 0.8444 | 0.8371 | 0.8178 | 0.7099 |
| Ridge regression | 0.7817 | 0.7518 | 0.7560 | 0.6593 |
| Random forest | 0.6984 | 0.6638 | 0.6813 | 0.5944 |
| Histogram gradient boosting | 0.7091 | 0.6797 | 0.7052 | 0.6052 |

In this preliminary 90\% identity experiment, the cluster-safe split did not show a larger MAE than the generated row-wise random split. The organism-held-out split also showed lower MAE than the generated random split, but its test target distribution has lower variability: pH standard deviation 0.9477 versus 1.1845 for the generated random test split. These results should not be interpreted as evidence against leakage effects until lower-identity cluster splits, confidence intervals, and subgroup analyses are complete.

The lower-identity cluster experiments are now complete. The 70\% cluster-safe test set contained 1481 examples and the 50\% cluster-safe test set contained 1544 examples. For ESM-2 embeddings, the best 70\% result was histogram gradient boosting with MAE 0.6282, while the best 50\% result was histogram gradient boosting with MAE 0.6822. The corresponding best handcrafted results were random forest with MAE 0.6954 at 70\% and 0.7376 at 50\%. These results indicate a performance decrease for the more distant 50\% split relative to the 70\% split, while ESM-2 embeddings remain better than handcrafted features.

The EC-held-out split assigned all records sharing an EC number to one partition and contained 6899 training, 1478 validation, and 1478 test records with no EC-group leakage. ESM-2 random forest was strongest, with MAE 0.8321 and 95\% bootstrap CI [0.7913, 0.8773]. The best handcrafted model was random forest with MAE 0.8844. EC-held-out performance was substantially worse than random and sequence-cluster evaluation, indicating that enzyme-class generalization is a harder setting.

## Main Findings

[Insert findings only after final experiments and statistical analysis.]

## Phase 7 Status

Frozen ESM-2 embeddings were generated externally in Colab using CUDA and saved to `data/processed/embeddings/esm2_t6_8M_UR50D_embeddings.npy`. The embedding matrix has shape 9855 by 320 and uses `facebook/esm2_t6_8M_UR50D`.

Embedding-based regressors were trained across all available split strategies, including the new 50\% and 70\% MMseqs2 cluster-safe splits. Test-set MAE values are:

| Model | Random MAE | Exact-duplicate-safe MAE | MMseqs2 90\% cluster-safe MAE | Organism-held-out MAE |
| --- | ---: | ---: | ---: | ---: |
| Ridge regression | 0.7035 | 0.6863 | 0.6786 | 0.6209 |
| Random forest | 0.6250 | 0.6073 | 0.6164 | 0.5360 |
| Histogram gradient boosting | 0.6335 | 0.6028 | 0.6148 | 0.5386 |

Relative to handcrafted features, ESM-2 embeddings reduced test MAE for all non-mean regressors across all split strategies. Statistical uncertainty has not yet been estimated.

## Phase 8 Evaluation Status

Consistent evaluation tables were generated from saved prediction files. Outputs include bootstrap confidence intervals, model comparison tables, pH-range summaries, subgroup summaries for organism and EC number, paired handcrafted-versus-embedding absolute-error comparisons, and top-error tables.

Selected ESM-2 embedding model bootstrap intervals for test MAE:

| Split strategy | Model | MAE | 95\% CI |
| --- | --- | ---: | --- |
| Random | Random forest | 0.6250 | [0.5930, 0.6574] |
| Random | Histogram gradient boosting | 0.6335 | [0.6034, 0.6622] |
| MMseqs2 90\% cluster-safe | Random forest | 0.6164 | [0.5837, 0.6532] |
| MMseqs2 90\% cluster-safe | Histogram gradient boosting | 0.6148 | [0.5826, 0.6474] |
| Organism-held-out | Random forest | 0.5360 | [0.5079, 0.5609] |
| Organism-held-out | Histogram gradient boosting | 0.5386 | [0.5111, 0.5636] |

Paired test-set comparisons showed lower absolute error for ESM-2 embeddings than handcrafted features for all non-mean regressors across all available split strategies. Wilcoxon tests were recorded in `outputs/metrics/paired_feature_group_comparisons.csv`, with Holm--Bonferroni adjustment applied across all 32 paired tests. All 24 non-mean comparisons remained significant at adjusted alpha 0.05; the eight mean-predictor comparisons were unchanged because the paired errors are identical. Error and subgroup diagnostics still warrant cautious interpretation.

## Phase 9 Error Analysis and Figures

Focused error-analysis tables were generated from the combined prediction file. Outputs include overall top errors, top errors for the best ESM-2 embedding model per split strategy, low-pH and high-pH error tables, error by sequence-length bin, and rare-versus-common organism/EC summaries.

Publication figures were generated in both PNG and PDF format under `reports/figures/`. The figure set includes target-pH and sequence-length distributions, dataset composition, model comparison plots with bootstrap intervals, split-strategy comparisons, random-versus-cluster and random-versus-organism comparisons, embedding-versus-handcrafted MAE change, actual-versus-predicted plots, residual distributions, error-versus-true-pH plots, pH-range performance, subgroup error figures, and ESM-2 embedding PCA colored by optimal pH.

## Paper Integration Status

The LaTeX manuscript now imports generated tables from `paper/tables/` and generated PDF figures from `paper/figures/`. Dataset, methodology, experimental setup, results, error-analysis, discussion, limitations, conclusion, abstract, and supplementary reproducibility sections have been updated with measured values from the current pipeline. The raw dataset SHA-256 is `6C2CE7F68E28E4130904A96FC071DBF7FE78E090BD1E408408038F01FA75A3E9`; the cleaned dataset has checksum `7413D3F48E8A097C4687BA0621F9E4E1D237682FCCAA54B41B952C16E606B3A7`. Dataset provenance is verified in `data/raw/README.md`; the manuscript lists Ali Nasir and Ayesha Rizwan as independent researchers. The workspace does not expose a usable Git commit identifier.

## Limitations

[Insert limitations after data inspection and experiments.]

## Reproducibility Details

CPU-side evaluation used Python 3.14.6 on Windows 11 with NumPy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0, SciPy 1.18.0, Biopython 1.87, PyYAML 6.0.3, Matplotlib 3.11.1, seaborn 0.13.2, Jinja2 3.1.6, and pytest 9.1.1. ESM-2 embeddings were generated with Python 3.12.13, PyTorch 2.11.0+cu128, CUDA, and `facebook/esm2_t6_8M_UR50D`; full embedding metadata is stored in `data/processed/embeddings/esm2_t6_8M_UR50D_metadata.json`. All runs used `config/default.yaml`.

Final validation completed with 47 pytest tests passing. The manuscript was compiled with MiKTeX using `pdflatex`, `bibtex`, and two final `pdflatex` passes; the resulting `paper/main.pdf` is 21 pages and has no fatal LaTeX errors or unresolved citations/references. `latexmk` is installed but cannot run on this Windows environment because its Perl script engine is unavailable, so the equivalent explicit compilation sequence was used.
