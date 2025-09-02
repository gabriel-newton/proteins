import pandas as pd
import sqlite3
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

def prepare_plot_data(df: pd.DataFrame) -> dict:
    """
    Takes a dataframe for a single residue and calculates the 2D frequency
    grids for both 180 and 360 degree ranges.
    """
    required_cols = ['tau_NA', 'tau_AC']
    if not all(col in df.columns for col in required_cols):
        return {} # Return empty dict if data is incomplete

    # --- 180 Degree Range ---
    angle_data_180 = df[required_cols].dropna().copy()
    angle_data_180['phi'] = angle_data_180['tau_NA'].round().astype(int)
    angle_data_180['psi'] = angle_data_180['tau_AC'].round().astype(int)
    full_range_180 = np.arange(-180, 181)
    
    freq_counts_180 = angle_data_180.groupby(['phi', 'psi']).size().reset_index(name='count')
    z_data_180 = freq_counts_180.pivot_table(index='psi', columns='phi', values='count', fill_value=0)
    z_data_180 = z_data_180.reindex(index=full_range_180, columns=full_range_180, fill_value=0)
    
    # --- 360 Degree Range ---
    angle_data_360 = df[required_cols].dropna().copy()
    angle_data_360['phi'] = (angle_data_360['tau_NA'].round() % 360).astype(int)
    angle_data_360['psi'] = (angle_data_360['tau_AC'].round() % 360).astype(int)
    full_range_360 = np.arange(0, 361)

    freq_counts_360 = angle_data_360.groupby(['phi', 'psi']).size().reset_index(name='count')
    z_data_360 = freq_counts_360.pivot_table(index='psi', columns='phi', values='count', fill_value=0)
    z_data_360 = z_data_360.reindex(index=full_range_360, columns=full_range_360, fill_value=0)

    # --- Prepare final data dictionary ---
    plot_data = {
        "x_180": z_data_180.columns.tolist(),
        "y_180": z_data_180.index.tolist(),
        "z_180": np.where(z_data_180 > 0, z_data_180, None).tolist(),
        "log_z_180": np.where(z_data_180 > 0, np.log10(z_data_180.astype(float)), None).tolist(),
        "x_360": z_data_360.columns.tolist(),
        "y_360": z_data_360.index.tolist(),
        "z_360": np.where(z_data_360 > 0, z_data_360, None).tolist(),
        "log_z_360": np.where(z_data_360 > 0, np.log10(z_data_360.astype(float)), None).tolist(),
    }
    return plot_data

def main():
    parser = argparse.ArgumentParser(description="Export Ramachandran plot data from a SQLite DB to JSON files for a static web viewer.")
    parser.add_argument("db_path", type=str, help="Path to the protein_geometry_invariants.db SQLite file.")
    parser.add_argument("k_value", type=int, help="The k-mer length (e.g., 1, 2) to export.")
    parser.add_argument("-o", "--output", type=str, default="visualizations", help="Directory to save the output files.")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    output_dir = Path(args.output)
    data_dir = output_dir / "data"
    
    if not db_path.is_file():
        print(f"Error: Database file not found at '{db_path}'", file=sys.stderr)
        sys.exit(1)

    # Create output directories
    data_dir.mkdir(parents=True, exist_ok=True)
    
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        print(f"--- Exporting data for k-mer length: {args.k_value} ---")
        
        # Get all relevant data from the database in one query
        print("Fetching data from database...")
        query = "SELECT residue, tau_NA, tau_AC FROM invariants WHERE LENGTH(residue) = ?"
        full_df = pd.read_sql_query(query, conn, params=(args.k_value,))
        
        if full_df.empty:
            print(f"No residues of length {args.k_value} found in the database.")
            return
            
        residue_names = sorted(full_df['residue'].unique().tolist())
        
        print(f"Found {len(residue_names)} unique residues. Preparing and exporting JSON files...")
        for res_name in tqdm(residue_names, desc="Exporting JSON"):
            residue_df = full_df[full_df['residue'] == res_name]
            plot_data = prepare_plot_data(residue_df)
            
            if plot_data:
                output_path = data_dir / f"{res_name}.json"
                with open(output_path, 'w') as f:
                    json.dump(plot_data, f)
        
        # Create a manifest file for the web app
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(residue_names, f)
            
        print(f"\nExport complete. Data saved in '{data_dir}'")
        print(f"Manifest file created at '{manifest_path}'")

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()