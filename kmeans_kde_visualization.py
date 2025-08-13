print("=== PYTHON SCRIPT STARTED ===\n")
print("=== PYTHON SCRIPT STARTED ===\n")
print("=== PYTHON SCRIPT STARTED ===\n")
print("=== PYTHON SCRIPT STARTED ===\n")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import gc
import psutil

# --- CONFIGURATION ---
# Note: These are now defaults. It's better to pass them as arguments.
ANGLES_FILE_DEFAULT = '2mer_angles.csv'
ANGLE_COLUMNS_DEFAULT = ['phi_i', 'psi_i', 'phi_i+1', 'psi_i+1']
OUTPUT_PLOT_FILE_DEFAULT = 'cluster_pair_plot3.png'
MAX_POINTS = 1_000_000

def memory_report():
    return f"Memory used: {psutil.Process().memory_info().rss / (1024**2):.2f} MB"

def create_cluster_plot(assignments_file: str,
                        angles_file: str = ANGLES_FILE_DEFAULT,
                        output_plot_file: str = OUTPUT_PLOT_FILE_DEFAULT,
                        angle_columns: list = ANGLE_COLUMNS_DEFAULT):
    """
    Loads angular data and cluster assignments, merges them, and generates a
    high-resolution pair plot visualization.
    """
    print("=== PYTHON SCRIPT STARTED ===")
    print(f"Initial memory: {memory_report()}")
    
    # 1. Validate Input Files
    if not os.path.exists(angles_file):
        raise FileNotFoundError(f"Error: Angular data file '{angles_file}' not found")
    if not os.path.exists(assignments_file):
        raise FileNotFoundError(f"Error: Cluster assignments file '{assignments_file}' not found")

    # 2. Load Data with Memory Optimization
    print("Loading data with optimized dtypes...")
    df_angles = pd.read_csv(angles_file, dtype={
        'phi_i': 'float32',
        'psi_i': 'float32',
        'phi_i+1': 'float32',
        'psi_i+1': 'float32'
    })
    print(f"Angles loaded. {memory_report()}")

    df_clusters = pd.read_csv(assignments_file, dtype={
        'Observation_Identifier': 'int32',
        'Assigned_Cluster': 'category'
    })
    print(f"Clusters loaded. {memory_report()}")

    # 3. Merge and Downsample
    print("Merging datasets...")
    # Use Observation_Identifier as index for faster merge
    df_clusters.set_index('Observation_Identifier', inplace=True)
    df_final = pd.merge(df_angles, df_clusters, left_index=True, right_index=True)
    
    # if len(df_final) > MAX_POINTS:
    #     print(f"Downsampling from {len(df_final)} to {MAX_POINTS} points...")
    #     df_final = df_final.sample(n=MAX_POINTS, random_state=42)
    
    del df_angles, df_clusters
    gc.collect()
    print(f"Merged data. {memory_report()}")

    # 4. HD Visualization
    print("Creating HD pair plot...")
    try:
        pair_plot = sns.pairplot(
            df_final,
            vars=angle_columns,
            hue='Assigned_Cluster',
            palette='viridis',
            plot_kws={
                's': 1,
                'alpha': 0.15,
                'linewidth': 0,
                'rasterized': True
            },
            diag_kind='kde',
            corner=True,
            height=4
        )

        plt.tight_layout()
        pair_plot.savefig(
            output_plot_file,
            dpi=600,
            bbox_inches='tight'
        )
        plt.close('all')
        gc.collect()
        print(f"Done. Final memory: {memory_report()}")
    except Exception as e:
        raise RuntimeError(f"Plotting failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_clusters.py <cluster_assignments.csv>")
        sys.exit(1)
    
    try:
        create_cluster_plot(sys.argv[1])
    except (FileNotFoundError, RuntimeError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)