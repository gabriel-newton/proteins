#!/bin/bash
#SBATCH --job-name=archive_db_files
#SBATCH --time=00:30:00      # 30 mins, file moving is fast
#SBATCH --mem=2G
#SBATCH --partition=nodes
#SBATCH --output=slurm_logs/archive_db_%j.log
#SBATCH --error=slurm_logs/archive_db_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# This script archives old database files from fastscratch to a data directory,
# leaving ONLY proteins_v7.db, .conda, and .mirrored.

# --- Configuration ---
SOURCE_DIR="/users/sggnewto/fastscratch/"
DEST_DIR="/users/sggnewto/data/"
LOG_DIR="/users/sggnewto/_KmerV6/slurm_logs" # Log directory

# --- Job ---
set -e # Exit immediately if a command fails.

echo "=== JOB START @ $(date) ==="
echo "Archiving old databases from ${SOURCE_DIR} to ${DEST_DIR}"

# Ensure log and destination directories exist
mkdir -p "${LOG_DIR}"
mkdir -p "${DEST_DIR}"

# Use rsync to move files
# --remove-source-files: deletes files from source after copy (i.e., "moves" them)
# --exclude: Specifies patterns to NOT move.
echo "Moving files, excluding 'proteins_v7.db', '.conda', and '.mirrored'..."
rsync -av --remove-source-files \
    --exclude='proteins_v7.db' \
    --exclude='.conda' \
    --exclude='.mirrored' \
    "${SOURCE_DIR}" "${DEST_DIR}"

echo ""
echo "--- ARCHIVE COMPLETE ---"
echo "Old database files moved to '${DEST_DIR}'."
echo "Only 'proteins_v7.db', '.conda', and '.mirrored' remain in '${SOURCE_DIR}'."