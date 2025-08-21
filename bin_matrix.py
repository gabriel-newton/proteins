import csv
from collections import Counter
import sys
import math
import os

def process_angles(input_path, output_path, bin_size):
    """
    Reads a CSV of dihedral angles, bins them, counts frequencies of 4D bins,
    and writes the sorted, non-zero frequencies to an output CSV.

    This function processes the input file in a streaming fashion to minimize
    memory usage, making it suitable for very large files.

    Args:
        input_path (str): The path to the input CSV file.
        output_path (str): The path where the output CSV file will be saved.
        bin_size (float): The size of each bin for the angle values.
    """
    print(f"Starting processing of '{input_path}' with bin size {bin_size}...")

    # A Counter is a specialized dictionary subclass for counting hashable objects.
    # The key will be a tuple of the four binned angles, e.g., (-161.6, 154.4, -123.9, 162.2)
    frequencies = Counter()
    
    # Calculate the precision needed for rounding based on the bin size.
    # This helps normalize the floating-point keys in the dictionary.
    try:
        precision = int(-math.log10(bin_size))
    except ValueError:
        precision = 10 # Default precision for very small bin sizes

    try:
        with open(input_path, 'r', newline='') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Skip the header row

            # Process each row in the input file
            for i, row in enumerate(reader, 1):
                # Provide progress updates for the user, as this can be a long process.
                if i % 5_000_000 == 0:
                    print(f"  ...processed {i:,} rows.")

                try:
                    # Unpack and convert angles to float
                    angles = [float(angle) for angle in row]

                    # Create a key by binning each angle.
                    # We use math.floor(angle / bin_size) to find which multiple of
                    # bin_size the angle belongs to, effectively finding the 'floor'
                    # of the bin. Rounding helps create consistent float keys.
                    binned_tuple = tuple(
                        round(math.floor(angle / bin_size) * bin_size, precision)
                        for angle in angles
                    )

                    # Increment the count for this specific combination
                    frequencies[binned_tuple] += 1

                except (ValueError, IndexError):
                    print(f"Warning: Skipping malformed row {i+1}: {row}", file=sys.stderr)
                    continue

        print(f"Finished reading file. Found {len(frequencies):,} unique angle combinations.")

        # Sort the frequencies from most common to least common.
        # The .most_common() method is highly efficient for this.
        print("Sorting results by frequency...")
        sorted_frequencies = frequencies.most_common()

        # Write the sorted results to the output file
        print(f"Writing sorted results to '{output_path}'...")
        with open(output_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)

            # Write the header for the output file
            writer.writerow(['phi_i_binned', 'psi_i_binned', 'phi_i+1_binned', 'psi_i+1_binned', 'frequency'])

            # Write the data rows
            for (angles_tuple, freq) in sorted_frequencies:
                writer.writerow(list(angles_tuple) + [freq])

        print("Processing complete.")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'", file=sys.stderr)
        # For demonstration, create a dummy file if it doesn't exist
        print("Creating a dummy 'data/2mer_angles.csv' file for you.")
        if not os.path.exists('data'):
            os.makedirs('data')
        with open(input_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['phi_i','psi_i','phi_i+1','psi_i+1'])
            writer.writerows([
                [-161.61,154.43,-123.94,162.24],
                [-123.94,162.24,-65.81,167.11],
                [-65.81,167.11,-134.92,-22.43],
                [-134.92,-22.43,-81.21,14.88],
                [-81.21,14.88,-62.52,153.87],
                [-62.52,153.87,-91.29,-0.29],
                # Add a duplicate row to test frequency counting
                [-81.21,14.88,-62.52,153.87],
            ])
        print("Dummy file created. Please run the script again to process it.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


# --- Main execution block ---
if __name__ == '__main__':
    # --- Configuration ---
    # Ensure the input data is in a folder named 'data' in the same directory as the script.
    INPUT_FILE = 'data/2mer_angles.csv'
    OUTPUT_FILE = 'binned_frequencies_0_5.csv'
    BIN_SIZE = 0.5

    process_angles(INPUT_FILE, OUTPUT_FILE, BIN_SIZE)