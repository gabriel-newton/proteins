#!/bin/bash
#SBATCH --job-name=v6_pipeline3_cleanup
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --partition=nodes
#SBATCH --output=slurm_logs/pipeline3_%j.log
#SBATCH --error=slurm_logs/pipeline3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# --- Configuration ---
# NOTE: Cleanup is a single-threaded process and requires low resources but must wait for cache generation.
PYTHON_SCRIPT="pipeline_03_cleanup.py"
CONDA_ENV_NAME="v5"
PROJECT_DIR="/users/sggnewto/_KmerV6/"
# IMPORTANT: This must match the DB used in pipeline_01 and pipeline_02
DB_PATH="/users/sggnewto/fastscratch/proteins_v7.db"

# --- Job ---
set -eo pipefail # Exit immediately if a command exits with a non-zero status.
echo "=== JOB START (DATABASE CLEANUP) @ $(date) ==="
cd "${PROJECT_DIR}" || { echo "Error: Project directory not found"; exit 1; }
mkdir -p slurm_logs

echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

echo "Executing cleanup script on database: ${DB_PATH}"

# Execute the cleanup script, passing the database path as the required argument.
# VACUUM operations can be time-consuming, so the time limit is set generously.
if ! time python -u "${PYTHON_SCRIPT}" "${DB_PATH}"; then
    echo "--- PYTHON SCRIPT FAILED ---"
    echo "Check the error log for details: slurm_logs/pipeline3_${SLURM_JOB_ID}.err"
    exit 1
fi

echo "=== JOB COMPLETE (DATABASE CLEANUP) @ $(date) ==="
exit 0
