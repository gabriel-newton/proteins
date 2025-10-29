#!/bin/bash
#SBATCH --job-name=v6_db_prep
#SBATCH --time=04:00:00      # 4 hours should be sufficient for vacuuming
#SBATCH --mem=32G
#SBATCH --partition=nodes
#SBATCH --output=slurm_logs/v6_db_prep_%j.log
#SBATCH --error=slurm_logs/v6_db_prep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

# This script prepares the database for the v6 pipeline.
# It drops all v5 tables, keeping only 'invariants_filtered'.

# --- Configuration ---
DB_PATH="/users/sggnewto/fastscratch/proteins_v7.db"
PROJECT_DIR="/users/sggnewto/_KmerV6/"
CONDA_ENV_NAME="v5"

# --- Job ---
set -e # Exit immediately if a command fails.

echo "=== JOB START @ $(date) ==="
echo "Preparing database for v6 pipeline: ${DB_PATH}"

echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${PROJECT_DIR}" || { echo "Error: Project directory not found"; exit 1; }
mkdir -p slurm_logs

# Use a 'here document' to pass a series of commands to sqlite3
sqlite3 "${DB_PATH}" <<EOF

-- Dropping all old v5 tables
DROP TABLE IF EXISTS v5_invariant_limits;
DROP TABLE IF EXISTS v5_pairwise;
DROP TABLE IF EXISTS v5_pairwise_cache;
DROP TABLE IF EXISTS v5_pairwise_stats;
DROP TABLE IF EXISTS v5_quads;
DROP TABLE IF EXISTS v5_quints;
DROP TABLE IF EXISTS v5_triplets;

-- Reclaim the disk space from all the dropped tables.
VACUUM;

EOF

echo ""
echo "--- PREPARATION COMPLETE ---"
echo "The database at '${DB_PATH}' now contains only 'invariants_filtered'."
