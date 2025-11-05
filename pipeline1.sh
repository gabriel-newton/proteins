#!/bin/bash
#SBATCH --job-name=v6_pipeline1_joins
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --partition=nodes
#SBATCH --output=slurm_logs/pipeline1_%j.log
#SBATCH --error=slurm_logs/pipeline1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# --- Configuration ---
PYTHON_SCRIPT="pipeline_01_joins.py"
CONDA_ENV_NAME="v5"
PROJECT_DIR="/users/sggnewto/_KmerV6/"
DB_PATH="/users/sggnewto/fastscratch/proteins_v7.db"

# --- Job ---
set -eo pipefail
echo "=== JOB START @ $(date) ==="
cd "${PROJECT_DIR}" || { echo "Error: Project directory not found"; exit 1; }
mkdir -p slurm_logs

echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

echo "Executing script on database: ${DB_PATH}"
if ! time python -u "${PYTHON_SCRIPT}" "${DB_PATH}"; then
    echo "--- PYTHON SCRIPT FAILED ---"
    echo "Check the error log for details: slurm_logs/pipeline1_${SLURM_JOB_ID}.err"
    exit 1
fi

echo "=== JOB COMPLETE @ $(date) ==="
exit 0