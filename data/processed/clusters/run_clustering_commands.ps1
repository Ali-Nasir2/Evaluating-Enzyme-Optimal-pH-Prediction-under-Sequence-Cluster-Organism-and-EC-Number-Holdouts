# Run the commands for one clustering tool after installing MMseqs2 or CD-HIT.
# Do not treat this file as evidence that clustering has already completed.
# Recommended with conda/mamba:
#   conda install -c bioconda -c conda-forge mmseqs2 cd-hit

# mmseqs at 90% identity
mmseqs easy-cluster "data\processed\clusters\sequences.fasta" "data\processed\clusters\mmseqs_identity_90" "data\processed\clusters\tmp_mmseqs_identity_90" --min-seq-id 0.90 -c 0.80 --cov-mode 0 --threads 4
# Expected raw output: data/processed/clusters/mmseqs_identity_90_cluster.tsv
# Parse to assignments with: python scripts/cluster_sequences.py --config config/default.yaml --parse-mmseqs "data/processed/clusters/mmseqs_identity_90_cluster.tsv" --identity 0.90

# mmseqs at 70% identity
mmseqs easy-cluster "data\processed\clusters\sequences.fasta" "data\processed\clusters\mmseqs_identity_70" "data\processed\clusters\tmp_mmseqs_identity_70" --min-seq-id 0.70 -c 0.80 --cov-mode 0 --threads 4
# Expected raw output: data/processed/clusters/mmseqs_identity_70_cluster.tsv
# Parse to assignments with: python scripts/cluster_sequences.py --config config/default.yaml --parse-mmseqs "data/processed/clusters/mmseqs_identity_70_cluster.tsv" --identity 0.70

# mmseqs at 50% identity
mmseqs easy-cluster "data\processed\clusters\sequences.fasta" "data\processed\clusters\mmseqs_identity_50" "data\processed\clusters\tmp_mmseqs_identity_50" --min-seq-id 0.50 -c 0.80 --cov-mode 0 --threads 4
# Expected raw output: data/processed/clusters/mmseqs_identity_50_cluster.tsv
# Parse to assignments with: python scripts/cluster_sequences.py --config config/default.yaml --parse-mmseqs "data/processed/clusters/mmseqs_identity_50_cluster.tsv" --identity 0.50

# cd-hit at 90% identity
cd-hit -i "data\processed\clusters\sequences.fasta" -o "data\processed\clusters\cdhit_identity_90.fasta" -c 0.90 -n 5 -d 0 -T 4 -M 0
# Expected raw output: data/processed/clusters/cdhit_identity_90.fasta.clstr
# Parse to assignments with: python scripts/cluster_sequences.py --config config/default.yaml --parse-cdhit "data/processed/clusters/cdhit_identity_90.fasta.clstr" --identity 0.90

# cd-hit at 70% identity
cd-hit -i "data\processed\clusters\sequences.fasta" -o "data\processed\clusters\cdhit_identity_70.fasta" -c 0.70 -n 5 -d 0 -T 4 -M 0
# Expected raw output: data/processed/clusters/cdhit_identity_70.fasta.clstr
# Parse to assignments with: python scripts/cluster_sequences.py --config config/default.yaml --parse-cdhit "data/processed/clusters/cdhit_identity_70.fasta.clstr" --identity 0.70

# cd-hit at 50% identity
cd-hit -i "data\processed\clusters\sequences.fasta" -o "data\processed\clusters\cdhit_identity_50.fasta" -c 0.50 -n 3 -d 0 -T 4 -M 0
# Expected raw output: data/processed/clusters/cdhit_identity_50.fasta.clstr
# Parse to assignments with: python scripts/cluster_sequences.py --config config/default.yaml --parse-cdhit "data/processed/clusters/cdhit_identity_50.fasta.clstr" --identity 0.50
