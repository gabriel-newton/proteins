#!/bin/bash
#SBATCH --job-name=v6_pipeline2_cache
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --partition=nodes
#SBATCH --output=slurm_logs/pipeline2_cache_%j.log
#SBATCH --error=slurm_logs/pipeline2_cache_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

# --- Configuration ---
PYTHON_SCRIPT="pipeline_02_cache.py" # Corrected filename
CONDA_ENV_NAME="v5" # Assuming env name is the same
PROJECT_DIR="/users/sggnewto/_KmerV6/"
DB_PATH="/users/sggnewto/fastscratch/proteins_v7.db" # Assuming DB path is the same

# --- Job ---
set -eo pipefail # Exit immediately if a command exits with a non-zero status.
echo "=== JOB START (CACHE GENERATION) @ $(date) ==="
cd "${PROJECT_DIR}" || { echo "Error: Project directory not found"; exit 1; }
mkdir -p slurm_logs

echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

echo "Executing script on database: ${DB_PATH}"
echo "NOTE: --recalculate flag is active. Existing cache and limits tables will be dropped."
# The Python script is designed to use the number of allocated CPUs via SLURM_CPUS_PER_TASK.
# The --recalculate flag forces a full data regeneration.
# Removed --visual flag as it's not an argument for the main pipeline script.
if ! time python -u "${PYTHON_SCRIPT}" "${DB_PATH}" --recalculate; then
    echo "--- PYTHON SCRIPT FAILED ---"
    echo "The Python script exited with an error during recalculation."
    echo "Check the error log for details: slurm_logs/pipeline2_cache_${SLURM_JOB_ID}.err"
    exit 1
fi

echo "=== JOB COMPLETE (CACHE GENERATION) @ $(date) ==="
echo "IMPORTANT: Remove the --recalculate flag from this script for future runs to use the cache."
exit 0