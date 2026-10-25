# Split Summary

Generated split definitions for reproducible experiments.

| Strategy | Train | Validation | Test | Group leakage count | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| random | 6898 | 1478 | 1479 | 39 | Row-wise random split; exact duplicate sequences may cross splits. |
| exact_duplicate_safe | 6895 | 1480 | 1480 | 0 | All exact duplicate sequences are assigned to one split. |
| organism_held_out | 6899 | 1478 | 1478 | 0 | All records from the same Organism are assigned to one split. |
| ec_held_out | 6899 | 1478 | 1478 | 0 | All records from the same EC Number are assigned to one split. |
| cluster_safe_mmseqs_identity_50 | 6872 | 1439 | 1544 | 0 | Cluster-safe split from cluster_assignments_mmseqs_identity_50.csv. |
| cluster_safe_mmseqs_identity_70 | 6908 | 1466 | 1481 | 0 | Cluster-safe split from cluster_assignments_mmseqs_identity_70.csv. |
| cluster_safe_mmseqs_identity_90 | 6902 | 1480 | 1473 | 0 | Cluster-safe split from cluster_assignments_mmseqs_identity_90.csv. |
