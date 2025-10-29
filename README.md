# Protein Plot Panel - Data Pipeline

This repository contains the Python scripts for the data generation and aggregation pipeline that powers the **Protein Plot Panel** dashboard.

Its primary function is to process a table of pre-calculated protein invariants (`invariants_filtered`) and generate a compact, optimized SQLite database (`proteins_app.db`) containing all pre-computed statistics and binned plot data needed by the front-end application.

## ⚙️ Pipeline Overview

The pipeline consists of three main stages, executed sequentially:

1.  **Join Generation (`pipeline_01_joins.py`)**: Creates intermediate tables by joining the `invariants_filtered` table with itself at different residue offsets (+1 to +4). These tables (`v6_offset_1` to `v6_offset_4`) facilitate efficient pairwise comparisons in the next step.
2.  **Cache & Stats Generation (`pipeline_02_cache.py`)**: This is the core processing step.
    * It determines the full list of required data products (3D plots, 1D histograms + stats, stats-only) based on invariant types, residue contexts, and offset, excluding redundant combinations.
    * It uses multiprocessing to parallelize the calculation of statistics and plot data for each combination by calling `tools/generate_visualizations.py`.
    * It saves the results into the final V7 schema tables:
        * `v7_stats`: Unified table containing all pre-computed 1D and 2D statistics.
        * `v7_3D_cache`: Binned data for 3D heatmap plots (Torsion vs. Torsion), stored as JSON strings in the 'points' format.
        * `v7_histo_cache`: Binned data for 1D histograms (Torsion vs. Non-Torsion), stored as JSON strings.
    * This script is resumable and checks for existing `plot_key` entries in `v7_stats` before reprocessing.
3.  **Database Cleanup (`pipeline_03_cleanup.py`)**: Separates the generated cache data from the raw input data.
    * It creates a copy of the main database, named `proteins_app.db`.
    * It **removes** the raw `invariants_filtered` table and the intermediate `v6_offset_*` tables from `proteins_app.db`, leaving only the `v7_*` cache tables needed by the front end.
    * It **removes** all `v6_offset_*` and `v7_*` tables from the **original** database, leaving only the raw `invariants_filtered` table.
    * Both databases are vacuumed to reclaim disk space.

## Prerequisites

* A **SQLite database** containing a table named `invariants_filtered`. This table must include columns for `chain_id`, `position`, `residue`, and all invariants listed in `tools/pipeline_constants.py` (e.g., `tau_NA`, `angle_A`, `length_C`, etc.). This table represents the clean, high-quality input data.

## 🛠️ Dependencies

The pipeline relies on several Python libraries:

* `pandas`
* `numpy`
* `tqdm` (for progress bars)

Install them using pip:
```bash
pip install pandas numpy tqdm

🏃‍♂️ How to Run

    Ensure Input: Make sure your SQLite database containing the invariants_filtered table exists.

    Run Sequentially: Execute the scripts in order, passing the path to your database as a command-line argument.
    Bash

    # Step 1: Create offset tables
    python pipeline_01_joins.py /path/to/your_database.db

    # Step 2: Generate stats and cache data (can take a long time)
    # Use --recalculate to force regeneration if needed
    python pipeline_02_cache.py /path/to/your_database.db [--recalculate]

    # Step 3: Create the final app.db and clean up
    python pipeline_03_cleanup.py /path/to/your_database.db

    Output: After successful execution, you will have two databases:

        /path/to/your_database.db: Contains only the raw invariants_filtered table.

        proteins_app.db: Contains only the v7_stats, v7_3D_cache, and v7_histo_cache tables, ready to be used by the Protein Plot Panel dashboard.

🧩 Supporting Modules

    tools/pipeline_constants.py: Defines shared lists of invariants, residues, resolution levels, bin sizes, and static invariant limits.

    tools/pipeline_utils.py: Contains helper functions for database connections, querying data, calculating bin numbers, etc.

    tools/generate_visualizations.py: The core calculation engine called by pipeline_02_cache.py. It takes specific query parameters, fetches data using pipeline_utils, calculates unified stats, 3D heatmap points, and 1D histograms.

🗃️ Final App Database Schema (proteins_app.db)

    v7_stats:

        plot_key (TEXT PRIMARY KEY)

        job_type (TEXT: '3D_VIZ', 'STATS_AND_HISTO', 'STATS_ONLY')

        population (INTEGER)

        mean_x, variance_x, median_x, min_x, max_x, freq_at_mean_x (REAL/INTEGER)

        mean_y, variance_y, median_y, min_y, max_y, freq_at_mean_y (REAL/INTEGER)

        covariance (REAL)

        peak_x, peak_y (REAL)

        peak_freq (INTEGER)

    v7_3D_cache:

        plot_key (TEXT PRIMARY KEY)

        data (JSON TEXT: {"points": [[x, y, z], ...]})

    v7_histo_cache:

        plot_key (TEXT)

        axis (TEXT: 'x' or 'y')

        data (JSON TEXT: {"bins": [...], "counts": [...]})

        PRIMARY KEY (plot_key, axis)