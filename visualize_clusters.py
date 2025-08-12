import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# --- CONFIGURATION ---
# The original data file with the 4D angular data.
ANGLES_FILE = '2mer_angles.csv'

# The columns that contain the angular data for plotting.
ANGLE_COLUMNS = ['phi_i', 'psi_i', 'phi_i+1', 'psi_i+1']

# The output file name for the plot.
OUTPUT_PLOT_FILE = 'cluster_pair_plot.png'


def create_cluster_plot(assignments_file: str):
    """
    Loads angular data and cluster assignments, merges them,
    and generates a 4x4 pair plot colored by cluster.

    Args:
        assignments_file (str): The path to the cluster_assignments.csv file.
    """
    # --- 1. Validate Input Files ---
    if not os.path.exists(ANGLES_FILE):
        print(f"Error: The angular data file '{ANGLES_FILE}' was not found.")
        sys.exit(1)

    if not os.path.exists(assignments_file):
        print(f"Error: The cluster assignments file '{assignments_file}' was not found.")
        sys.exit(1)

    # --- 2. Load and Merge Data ---
    print("Loading and merging data...")
    try:
        df_angles = pd.read_csv(ANGLES_FILE)
        df_clusters = pd.read_csv(assignments_file)
    except Exception as e:
        print(f"Failed to load data files: {e}")
        sys.exit(1)

    # The 'Observation_Identifier' corresponds to the index of the original angles file.
    # We use the index of df_angles to merge with the identifier from df_clusters.
    df_clusters['Observation_Identifier'] = df_clusters['Observation_Identifier'].astype(int)
    
    # Merge the two dataframes based on the identifier/index.
    df_final = pd.merge(df_angles, df_clusters, 
                        left_index=True, 
                        right_on='Observation_Identifier')

    print(f"Successfully merged {len(df_final)} data points.")

    # --- 3. Generate the 4x4 Pair Plot ---
    print(f"Generating the pair plot... this may take some time for large datasets.")
    
    # Use seaborn's pairplot to create the 4x4 grid.
    # `vars` specifies the dimensions to plot.
    # `hue` specifies the column to use for coloring.
    pair_plot = sns.pairplot(df_final,
                             vars=ANGLE_COLUMNS,
                             hue='Assigned_Cluster',
                             palette='viridis',  # Use a colorblind-friendly and distinct palette
                             plot_kws={'s': 5, 'alpha': 0.6}) # Style scatter plots

    # --- 4. Save the Plot ---
    print(f"Saving plot to '{OUTPUT_PLOT_FILE}'...")
    pair_plot.savefig(OUTPUT_PLOT_FILE, dpi=300) # Save with high resolution
    print("Done.")


if __name__ == "__main__":
    # The script expects one command-line argument: the path to the cluster assignments file.
    if len(sys.argv) != 2:
        print("Usage: python visualize_clusters.py <path_to_cluster_assignments.csv>")
        sys.exit(1)

    cluster_assignments_path = sys.argv[1]
    create_cluster_plot(cluster_assignments_path)