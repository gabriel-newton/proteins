# File: db_utils.py
import sqlite3
import pandas as pd
import time

DB_PATH = '/users/sggnewto/fastscratch/protein_geometry_invariants.db'

def get_angles_for_kmer(kmer: str, focal_index: int, limit: int) -> pd.DataFrame:
    """
    Queries the database with a user-defined row limit.
    """
    if not kmer or not isinstance(focal_index, int):
        return pd.DataFrame({'phi': [], 'psi': []})

    try:
        limit = abs(int(limit))
    except (ValueError, TypeError):
        limit = 1000000 

    k = len(kmer)
    if not (0 <= focal_index < k):
        raise ValueError("focal_index must be within the bounds of the k-mer length.")

    if k == 1:
        query = f"SELECT tau_NA AS phi, tau_AC AS psi FROM invariants WHERE residue = ? LIMIT {limit};"
        params = (kmer,)
    else:
        kmer_context_sql = "residue || " + " || ".join([f"LEAD(residue, {i}) OVER w" for i in range(1, k)])
        query = f"""
            WITH KmerContext AS (
                SELECT chain_id, position, {kmer_context_sql} AS kmer_context
                FROM invariants
                WINDOW w AS (PARTITION BY chain_id ORDER BY position)
                LIMIT {limit}
            )
            SELECT inv.tau_NA AS phi, inv.tau_AC AS psi
            FROM KmerContext kc
            JOIN invariants inv ON kc.chain_id = inv.chain_id AND kc.position + {focal_index} = inv.position
            WHERE kc.kmer_context = ?
        """
        params = (kmer,)

    try:
        with sqlite3.connect(DB_PATH) as con:
            print(f"--- Starting database query with LIMIT {limit}... ---")
            start_time = time.time()
            df = pd.read_sql_query(query, con, params=params)
            end_time = time.time()
            print(f"--- Query finished in {end_time - start_time:.2f} seconds. ---")
    except Exception as e:
        print(f"Database query failed: {e}")
        return pd.DataFrame({'phi': [], 'psi': []})
    return df