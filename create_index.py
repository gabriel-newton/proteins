# File: create_index.py
import sqlite3
import time

DB_PATH = '/users/sggnewto/fastscratch/protein_geometry_invariants.db'
INDEX_NAME = 'idx_chain_position'
TABLE_NAME = 'invariants'

print(f"Connecting to the database at '{DB_PATH}'...")
con = sqlite3.connect(DB_PATH)
cursor = con.cursor()

try:
    print(f"Creating index '{INDEX_NAME}' on table '{TABLE_NAME}'...")
    start_time = time.time()
    
    sql_command = f"CREATE INDEX {INDEX_NAME} ON {TABLE_NAME} (chain_id, position);"
    cursor.execute(sql_command)
    
    con.commit()
    end_time = time.time()
    
    print(f"--- Index created successfully in {end_time - start_time:.2f} seconds. ---")

except sqlite3.OperationalError as e:
    if "already exists" in str(e):
        print(f"Index '{INDEX_NAME}' already exists. No action taken.")
    else:
        raise e
finally:
    con.close()
    print("Database connection closed.")