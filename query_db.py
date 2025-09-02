import sqlite3
import pandas as pd
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run a read-only SQL query on a SQLite database and print the results.")
    parser.add_argument("db_path", type=Path, help="Path to the SQLite database file.")
    parser.add_argument("query", type=str, help="The SQL query to execute.")
    args = parser.parse_args()

    if not args.db_path.is_file():
        print(f"Error: Database file not found at '{args.db_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        conn = sqlite3.connect(f"file:{args.db_path}?mode=ro", uri=True)
        df = pd.read_sql_query(args.query, conn)
        print(f"--- Results for query: ---\n{args.query}\n")
        if df.empty:
            print("Query returned no results.")
        else:
            print(df.to_string())
    
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    main()

