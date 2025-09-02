import pandas as pd
import sqlite3
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys

# The 20 standard amino acids, in alphabetical order, defining the grid
AMINO_ACIDS = sorted(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])

def _create_plot_data_structure(z_data: pd.DataFrame) -> dict:
    """
    Helper function to generate the final data dictionary with multiple coordinate and scale options.
    """
    # Create z_log before replacing zeros with None
    # np.log1p calculates log(1 + x), handling x=0 correctly by returning log(1)=0
    z_data_log = np.log1p(z_data)
    
    # Check which original cells were greater than 0
    mask = z_data > 0
    
    return {
        "x_180": z_data.columns.tolist(),
        "y_180": z_data.index.tolist(),
        "x_360": list(range(0, 361)),
        "y_360": list(range(0, 361)),
        "z_linear": np.where(mask, z_data, None).tolist(),
        "z_log": np.where(mask, z_data_log, None).tolist()
    }

def process_central_residue(db_connection, central_residue: str) -> dict:
    """
    Fetches and processes data for a central residue, ignoring context,
    returning a dictionary with multiple coordinate and scale options.
    """
    query = "SELECT tau_NA, tau_AC FROM invariants WHERE residue = ?"
    df = pd.read_sql_query(query, db_connection, params=(central_residue,))
    
    if df.empty:
        return {}

    angle_data = df[['tau_NA', 'tau_AC']].dropna().copy()
    angle_data['phi'] = angle_data['tau_NA'].round().astype(int)
    angle_data['psi'] = angle_data['tau_AC'].round().astype(int)
    
    full_range = np.arange(-180, 181)
    freq_counts = angle_data.groupby(['phi', 'psi']).size().reset_index(name='count')
    z_data = freq_counts.pivot_table(index='psi', columns='phi', values='count', fill_value=0)
    z_data = z_data.reindex(index=full_range, columns=full_range, fill_value=0)

    return _create_plot_data_structure(z_data)

def process_context(db_connection, central_residue: str, context: str) -> dict:
    """
    Fetches and processes all data for a given central residue and context,
    returning a dictionary containing 20 individual plot structures.
    """
    if context not in ['before', 'after']:
        raise ValueError("Context must be 'before' or 'after'")

    context_k_value = len(central_residue) + 1
    pattern = f"_{central_residue}" if context == 'before' else f"{central_residue}_"
    query = "SELECT residue, tau_NA, tau_AC FROM invariants WHERE LENGTH(residue) = ? AND residue LIKE ?"
    
    df = pd.read_sql_query(query, db_connection, params=(context_k_value, pattern))
    if df.empty:
        return {}

    faceted_data = {}
    grouped = df.groupby('residue')

    for neighbor_aa in AMINO_ACIDS:
        kmer = f"{neighbor_aa}{central_residue}" if context == 'before' else f"{central_residue}{neighbor_aa}"
        
        if kmer not in grouped.groups:
            continue

        residue_df = grouped.get_group(kmer)
        
        angle_data = residue_df[['tau_NA', 'tau_AC']].dropna().copy()
        angle_data['phi'] = angle_data['tau_NA'].round().astype(int)
        angle_data['psi'] = angle_data['tau_AC'].round().astype(int)
        
        full_range = np.arange(-180, 181)
        freq_counts = angle_data.groupby(['phi', 'psi']).size().reset_index(name='count')
        z_data = freq_counts.pivot_table(index='psi', columns='phi', values='count', fill_value=0)
        z_data = z_data.reindex(index=full_range, columns=full_range, fill_value=0)

        faceted_data[neighbor_aa] = _create_plot_data_structure(z_data)
        
    return faceted_data

def main():
    parser = argparse.ArgumentParser(description="Export Ramachandran data (overall and contextual) for residues of a given length to JSON files.")
    parser.add_argument("db_path", type=str, help="Path to the SQLite database file.")
    parser.add_argument("central_k_value", type=int, help="The length of the central residue/k-mer to analyze (e.g., 1 for single amino acids).")
    parser.add_argument("-o", "--output", type=str, default="ramachandran_data", help="Directory to save the output JSON files.")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not db_path.is_file():
        print(f"Error: Database file not found at '{db_path}'", file=sys.stderr)
        sys.exit(1)
        
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        
        print("✅ Database connection successful.")
        print("Fetching central residues... (This may take a while)")
        kmer_query = "SELECT DISTINCT residue FROM invariants WHERE LENGTH(residue) = ?"
        central_residues_df = pd.read_sql_query(kmer_query, conn, params=(args.central_k_value,))
        central_residues = sorted(central_residues_df['residue'].tolist())
        print(f"✅ Found {len(central_residues)} residues to process. Starting main loop.")
        
        if not central_residues:
            print(f"No residues of length {args.central_k_value} found in the database.")
            return

        for central_aa in tqdm(central_residues, desc="Processing Central Residues"):
            # 1. Process and save the central data
            central_plot_data = process_central_residue(conn, central_aa)
            if central_plot_data:
                output_path = output_dir / f"{central_aa}.json"
                with open(output_path, 'w') as f:
                    json.dump(central_plot_data, f)
            
            # 2. Process and save the "before" context data
            pre_context_data = process_context(conn, central_aa, 'before')
            if pre_context_data:
                output_path = output_dir / f"{central_aa}_befr.json"
                with open(output_path, 'w') as f:
                    json.dump(pre_context_data, f)

            # 3. Process and save the "next" (after) context data
            pos_context_data = process_context(conn, central_aa, 'after')
            if pos_context_data:
                output_path = output_dir / f"{central_aa}_next.json"
                with open(output_path, 'w') as f:
                    json.dump(pos_context_data, f)

        print(f"\nExport complete. JSON files have been saved to '{output_dir}'.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()