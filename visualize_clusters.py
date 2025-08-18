print("=== PYTHON SCRIPT STARTED ===\n")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import gc
import psutil

# --- CONFIGURATION ---
ANGLES_FILE = '2mer_angles.csv'
ANGLE_COLUMNS = ['phi_i', 'psi_i', 'phi_i+1', 'psi_i+1']
OUTPUT_PLOT_FILE = 'cluster_pair_plot4.png'  # Change to .pdf for vector format
MAX_POINTS = 1_000_000

def memory_report():
    return f"Memory used: {psutil.Process().memory_info().rss / (1024**2):.2f} MB"

def create_cluster_plot(assignments_file: str):
    print("=== PYTHON SCRIPT STARTED ===")
    print(f"Initial memory: {memory_report()}")
    
    # 1. Validate Input Files
    if not os.path.exists(ANGLES_FILE):
        print(f"Error: Angular data file '{ANGLES_FILE}' not found")
        sys.exit(1)
    if not os.path.exists(assignments_file):
        print(f"Error: Cluster assignments file '{assignments_file}' not found")
        sys.exit(1)

    # 2. Load Data with Memory Optimization
    print("Loading data with optimized dtypes...")
    try:
        df_angles = pd.read_csv(ANGLES_FILE, dtype={
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

    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    # 3. Merge and Downsample
    print("Merging datasets...")
    df_clusters['Observation_Identifier'] = df_clusters['Observation_Identifier'].astype(int)
    df_final = pd.merge(df_angles, df_clusters, 
                       left_index=True,
                       right_on='Observation_Identifier')
    
    if len(df_final) > MAX_POINTS:
        print(f"Downsampling from {len(df_final)} to {MAX_POINTS} points...")
        df_final = df_final.sample(n=MAX_POINTS, random_state=42)
    
    del df_angles, df_clusters
    gc.collect()
    print(f"Merged data. {memory_report()}")

    # 4. HD Visualization
    print("Creating HD pair plot...")
    try:
        # Set up figure
        plt.figure(figsize=(16, 16))
        
        # Create plot with small points
        pair_plot = sns.pairplot(
            df_final,
            vars=ANGLE_COLUMNS,
            hue='Assigned_Cluster',
            palette='viridis',
            plot_kws={
                's': 1,           # Very small points
                'alpha': 0.15,    # Increased transparency
                'linewidth': 0,   # No point borders
                'rasterized': True # Better HD rendering
            },
            diag_kind='kde',
            corner=True,
            height=4             # Subplot size in inches
        )

        # Save with high resolution
        plt.tight_layout()
        pair_plot.savefig(
            OUTPUT_PLOT_FILE,
            dpi=600,             # Ultra-high DPI
            bbox_inches='tight', # No extra whitespace
            quality=100          # Maximum quality
        )
        plt.close('all')
        gc.collect()
        
    except Exception as e:
        print(f"Plotting failed: {e}")
        sys.exit(1)

    print(f"Done. Final memory: {memory_report()}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_clusters.py <cluster_assignments.csv>")
        sys.exit(1)
    
    create_cluster_plot(sys.argv[1])