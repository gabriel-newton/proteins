#!/bin/bash
#SBATCH --job-name=create_db_index
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=nodes
#SBATCH --output=slurm_logs/create_index_%j.log
#SBATCH --error=slurm_logs/create_index_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# --- Configuration ---
PYTHON_SCRIPT="create_index.py" 
CONDA_ENV_NAME="kmers" 

# --- Job ---
set -eo pipefail # Exit on error

echo "=== JOB START: CREATING DATABASE INDEX ==="
# IMPORTANT: Update this path to your project directory if it's different
cd /users/sggnewto/_KmerDashboard/ || exit 1

mkdir -p slurm_logs

echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

echo "Executing script: ${PYTHON_SCRIPT}"
time python ${PYTHON_SCRIPT}

echo "=== JOB COMPLETE ==="
exit $?