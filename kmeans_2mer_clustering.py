#!/usr/bin/env python

import os
import sys
import json
from datetime import datetime

# # Major Library Imports
import numpy as np
import matplotlib.pyplot as plt

# # Dask and RAPIDS/cuML Imports for Multi-GPU operations
import dask
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import dask_cudf
import dask_cupy as dcp
from cuml.dask.cluster import KMeans as DaskKMeans
from cuml.dask.decomposition import PCA as DaskPCA

# # Supporting library for finding the elbow point
from kneed import KneeLocator

# --- CONFIGURATION ---
# These features will be used for clustering.
FEATURES_TO_CLUSTER = ['phi_i', 'psi_i', 'phi_i+1', 'psi_i+1']
# Whether to transform dihedral angles into a circular representation (sin/cos).
HANDLE_PERIODIC_DATA = True
# The maximum number of clusters to test for the elbow method.
MAX_K_TO_TEST = 10
# Sample size for visualization only. Plotting millions of points is slow and uninformative.
VIZ_SAMPLE_SIZE = 500_000

# --- DATA LOADING (DASK) ---
def load_and_prepare_data_dask(filepath, feature_cols, handle_periodic):
    """Loads a single CSV using Dask-cuDF and prepares it for clustering across multiple GPUs."""
    print(f"Loading data from '{filepath}' using Dask-cuDF...")
    try:
        # Use dask_cudf to read the CSV into multiple partitions across the GPUs.
        ddf = dask_cudf.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data with Dask-cuDF: {e}")
        return None, None

    if not all(col in ddf.columns for col in feature_cols):
        print(f"Error: Input file must contain the columns: {feature_cols}")
        return None, None

    # Keep original identifiers as a Dask Series for later.
    identifiers = ddf.index.astype(str)
    # Select feature columns and ensure data is float32 to conserve memory.
    data_points = ddf[feature_cols].astype(np.float32)

    if handle_periodic:
        print("Applying periodic transformation (cos/sin)...")
        # Perform sin/cos transformation. These operations are executed in parallel on the GPUs.
        radians_df = data_points * (np.pi / 180.0)
        transformed_data = dask_cudf.DataFrame()
        for col in radians_df.columns:
            transformed_data[f'{col}_cos'] = dcp.cos(radians_df[col])
            transformed_data[f'{col}_sin'] = dcp.sin(radians_df[col])
        # Persist the transformed data in GPU memory to avoid re-computation.
        return transformed_data.persist(), identifiers.persist()
    else:
        return data_points.persist(), identifiers.persist()

def find_optimal_clusters_dask(data_dask, max_k):
    """Finds the optimal number of clusters using the Elbow Method on the full Dask dataset."""
    print("\nFinding the optimal number of clusters using the Elbow Method...")
    inertias = []
    k_range = range(2, max_k + 1)

    print(f"Testing k from 2 to {max_k} on the full dataset...")
    for k in k_range:
        kmeans = DaskKMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(data_dask)
        inertias.append(kmeans.inertia_)
        print(f"  k={k}, Inertia={inertias[-1]:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plot_filename = 'elbow_plot_dihedrals.png'

    try:
        kneedle = KneeLocator(k_range, inertias, S=1.0, curve="convex", direction="decreasing")
        optimal_k = kneedle.elbow
        print(f"Found optimal number of clusters: {optimal_k}")
        plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles='--', color='r', label=f'Elbow at k={optimal_k}')
        plt.legend()
    except Exception as e:
        optimal_k = None
        print(f"Could not automatically find elbow point: {e}. Manual inspection required.")

    plt.savefig(plot_filename)
    print(f"Saved elbow plot to {plot_filename}")
    plt.close()
    return optimal_k

def save_results_dask(identifiers_dask, labels_dask, kmeans_model, config_params, output_dir):
    """Saves the distributed clustering results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to '{output_dir}/'")

    results_ddf = dask_cudf.DataFrame({
        'Observation_Identifier': identifiers_dask,
        'Assigned_Cluster': labels_dask
    })
    
    assignments_filepath = os.path.join(output_dir, 'kmeans_cluster_assignments.csv')
    results_ddf.to_csv(assignments_filepath, single_file=True, index=False)
    print(f"  - Saved cluster assignments to {assignments_filepath}")

    summary_data = {
        "configuration": config_params,
        "cluster_centers": kmeans_model.cluster_centers_.tolist(),
        "inertia": float(kmeans_model.inertia_)
    }
    summary_filepath = os.path.join(output_dir, 'run_summary.json')
    with open(summary_filepath, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"  - Saved run summary and centroids to {summary_filepath}")

def plot_clusters_dask(data_dask, labels_dask, centers, n_clusters, output_dir):
    """Visualizes the clusters using PCA, sampling the data for a readable plot."""
    print("\nRunning PCA on full dataset for visualization...")
    pca = DaskPCA(n_components=2)
    data_2d_dask = pca.fit_transform(data_dask)

    plot_df_dask = dask_cudf.DataFrame({
        'pca1': data_2d_dask[0],
        'pca2': data_2d_dask[1],
        'labels': labels_dask
    })
    
    print(f"Sampling {VIZ_SAMPLE_SIZE} points for plotting...")
    plot_df_cpu = plot_df_dask.sample(frac=VIZ_SAMPLE_SIZE / len(data_dask)).compute()

    centers_2d_cpu = pca.transform(centers).compute()
    
    plt.figure(figsize=(12, 10))
    plt.scatter(plot_df_cpu['pca1'], plot_df_cpu['pca2'], c=plot_df_cpu['labels'], cmap='viridis', alpha=0.7)
    plt.scatter(centers_2d_cpu[:, 0], centers_2d_cpu[:, 1], c='red', s=250, marker='X', label='Centroids')
    plt.title(f'PCA of K-Means++ Clustering (Dihedrals, {n_clusters} clusters, {VIZ_SAMPLE_SIZE} points shown)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_filepath = os.path.join(output_dir, 'cluster_visualization.png')
    plt.savefig(plot_filepath)
    print(f"Saved cluster visualization to {plot_filepath}")
    plt.close()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_csv_file>")
        sys.exit(1)

    input_filepath = sys.argv[1]
    if not os.path.exists(input_filepath):
        print(f"Error: File not found at '{input_filepath}'")
        sys.exit(1)
        
    # --- DASK SETUP ---
    # This sets up a Dask cluster using the GPUs allocated by SLURM.
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print(f"Dask client ready: {client.dashboard_link}")
    # ---

    print("--- Running Dihedral Angle Clustering on MULTI-GPU ---")
    data_points_dask, identifiers_dask = load_and_prepare_data_dask(input_filepath, FEATURES_TO_CLUSTER, HANDLE_PERIODIC_DATA)

    if data_points_dask is None:
        print("Data loading failed. Exiting.")
        sys.exit(1)
    
    # Trigger computation to get the total number of points.
    n_points = len(data_points_dask)
    print(f"Successfully loaded and processed {n_points} data points across the cluster.")
    
    optimal_k = find_optimal_clusters_dask(data_points_dask, MAX_K_TO_TEST)

    if optimal_k and n_points > optimal_k:
        print(f"\nRunning final K-Means++ with {optimal_k} clusters on the full dataset...")
        # Run the final clustering on the full distributed dataset.
        kmeans_final = DaskKMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
        labels_dask = kmeans_final.fit_predict(data_points_dask)
        
        output_directory = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config_params = {
            'input_file': input_filepath,
            'features_clustered': FEATURES_TO_CLUSTER,
            'handled_periodic_data': HANDLE_PERIODIC_DATA,
            'num_clusters_found': int(optimal_k)
        }
        
        save_results_dask(identifiers_dask, labels_dask, kmeans_final, config_params, output_directory)
        
        plot_clusters_dask(data_points_dask, labels_dask, kmeans_final.cluster_centers_, optimal_k, output_directory)
        
        print("\n--- Analysis Complete ---")
    else:
        print("\nCould not determine optimal k or not enough data to cluster. Exiting.")

    # --- DASK SHUTDOWN ---
    client.close()
    cluster.close()
    print("Dask cluster has been shut down.")