#!/bin/bash

# This script monitors the SLURM queue for a specific user.
# It refreshes every 5 minutes until you stop it with Ctrl+C.

USER_TO_WATCH="sggnewto"
DELAY_SECONDS=1 # 5 minutes = 300 seconds

# --- trap handler ---
# This function will run when the script receives the EXIT signal,
# which includes being interrupted by Ctrl+C.
cleanup() {
    echo "" # Add a newline for cleaner exit
    echo "Monitoring stopped."
    exit 0
}

# Register the cleanup function to be called on script exit/interrupt.
trap cleanup SIGINT EXIT

# --- Main loop ---
echo "Starting queue monitor for user '${USER_TO_WATCH}'."
echo "Press Ctrl+C to stop."

while true; do
    clear # Clears the terminal for a clean view
    echo "Last updated: $(date)"
    echo "-------------------------------------"
    squeue -u "${USER_TO_WATCH}"
    echo "-------------------------------------"
    echo "Refreshing in 5 minutes... (Press Ctrl+C to stop)"
    sleep "${DELAY_SECONDS}"
done
