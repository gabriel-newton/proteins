import pandas as pd
import plotly.express as px
import sys
import os
import gc
import psutil

# --- CONFIGURATION ---
ANGLES_FILE_DEFAULT = '2mer_angles.csv'
OUTPUT_PLOT_FILE_DEFAULT = 'cluster_3d_plot.html'
MAX_POINTS = 1_000_000

def memory_report():
    return f"Memory used: {psutil.Process().memory_info().rss / (1024**2):.2f} MB"

def create_cluster_plot(assignments_file: str,
                        angles_file: str = ANGLES_FILE_DEFAULT,
                        output_plot_file: str = OUTPUT_PLOT_FILE_DEFAULT,
                        x_var: str = 'phi_i',
                        y_var: str = 'psi_i',
                        z_var: str = 'phi_i+1'):
    """
    Loads angular data and cluster assignments, merges them, and generates an
    interactive 3D Plotly Express scatter plot.
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
    df_clusters.set_index('Observation_Identifier', inplace=True)
    df_final = pd.merge(df_angles, df_clusters, left_index=True, right_index=True)
    
    if len(df_final) > MAX_POINTS:
        print(f"Downsampling from {len(df_final)} to {MAX_POINTS} points...")
        df_final = df_final.sample(n=MAX_POINTS, random_state=42)
    
    del df_angles, df_clusters
    gc.collect()
    print(f"Merged data. {memory_report()}")

    # 4. Create 3D Plotly Visualization
    print("Creating 3D Plotly scatter plot...")
    try:
        fig = px.scatter_3d(
            df_final,
            x=x_var,
            y=y_var,
            z=z_var,
            color='Assigned_Cluster',
            symbol='Assigned_Cluster',
            title=f"3D Cluster Plot: {x_var} vs {y_var} vs {z_var}",
            labels={x_var: x_var.upper(), y_var: y_var.upper(), z_var: z_var.upper()},
            opacity=0.5,
            height=800
        )
        
        fig.update_traces(marker_size=1)
        
        # Save the interactive plot to an HTML file
        fig.write_html(output_plot_file)
        
        print(f"Done. Plot saved to {output_plot_file}")
        print(f"Final memory: {memory_report()}")
    except Exception as e:
        raise RuntimeError(f"Plotting with Plotly failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_clusters.py <cluster_assignments.csv>")
        sys.exit(1)
    
    try:
        # To plot a different combination, change the keyword arguments.
        # Example: create_cluster_plot(sys.argv[1], x_var='phi_i', y_var='phi_i+1', z_var='psi_i+1')
        create_cluster_plot(sys.argv[1], x_var='phi_i', y_var='psi_i', z_var='phi_i+1')
    except (FileNotFoundError, RuntimeError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)