import pandas as pd
import numpy as np
import sys
import os

# --- CONFIGURATION ---
# The input file containing the 4D angular data.
INPUT_ANGLES_FILE = '2mer_angles.csv'

# The columns to be converted.
ANGLE_COLUMNS = ['phi_i', 'psi_i', 'phi_i+1', 'psi_i+1']

# The name for the new output file.
OUTPUT_8D_FILE = '2mer_angles_8d.csv'


def convert_angles_to_8d(input_file: str, output_file: str):
    """
    Reads a CSV with angular data, converts specified columns to their
    sin/cos representation, and saves the result to a new CSV.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path for the new output CSV file.
    """
    # --- 1. Validate and Load Data ---
    if not os.path.exists(input_file):
        print(f"Error: The input file '{input_file}' was not found.")
        sys.exit(1)

    print(f"Loading data from '{input_file}'...")
    try:
        df_angles = pd.read_csv(input_file)
    except Exception as e:
        print(f"Failed to load data file: {e}")
        sys.exit(1)
        
    print(f"Successfully loaded {len(df_angles)} data points.")

    # --- 2. Perform the 8D Conversion ---
    print("Converting 4D angular data to 8D sin/cos representation...")
    
    # Create an empty DataFrame to store the new 8D data.
    df_8d = pd.DataFrame()

    # Loop through each of the four angle columns.
    for col in ANGLE_COLUMNS:
        if col not in df_angles.columns:
            print(f"Warning: Column '{col}' not found in the input file. Skipping.")
            continue
            
        # Convert degrees to radians for numpy's sin/cos functions.
        radians = np.deg2rad(df_angles[col])
        
        # Calculate the sin and cos values and add them as new columns.
        df_8d[f'{col}_cos'] = np.round(np.cos(radians), 4)
        df_8d[f'{col}_sin'] = np.round(np.sin(radians), 4)

    # --- 3. Save the New 8D Data ---
    if df_8d.empty:
        print("No data was converted. Exiting without saving.")
        sys.exit(1)
        
    print(f"Saving 8D data to '{output_file}'...")
    try:
        df_8d.to_csv(output_file, index=False)
        print("Done.")
    except Exception as e:
        print(f"Failed to save output file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # This script runs without needing command-line arguments.
    # It uses the filenames defined in the configuration section.
    convert_angles_to_8d(INPUT_ANGLES_FILE, OUTPUT_8D_FILE)
