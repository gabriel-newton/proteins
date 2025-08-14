import os
import pandas as pd
import cudf
import cupy as cp
from cuml.cluster import KMeans
from cuml.decomposition import PCA as cuMLPCA
import matplotlib.pyplot as plt
from kneed import KneeLocator
import json
from datetime import datetime
import sys
import numpy as np

# --- CONFIGURATION ---
FEATURES_TO_CLUSTER = ['phi_i', 'psi_i', 'phi_i+1', 'psi_i+1']
K_VALUE = 2
HANDLE_PERIODIC_DATA = True
MAX_K_TO_TEST = 10

# --- DATA LOADING ---
def load_and_prepare_data(filepath, feature_cols, handle_periodic):
    """Loads a single CSV and prepares it for clustering."""
    print(f"Loading data from '{filepath}'...")
    try:
        df = cudf.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data with cuDF: {e}")
        return None, None

    if not all(col in df.columns for col in feature_cols):
        print(f"Error: Input file must contain the columns: {feature_cols}")
        return None, None

    identifiers = df.index.astype(str)
    data_points = df[feature_cols]

    if handle_periodic:
        print("Applying periodic transformation (cos/sin)...")
        transformed_data = cudf.DataFrame()
        radians_df = data_points * (np.pi / 180.0)
        for col in radians_df.columns:
            transformed_data[f'{col}_cos'] = cp.cos(radians_df[col])
            transformed_data[f'{col}_sin'] = cp.sin(radians_df[col])
        return transformed_data, identifiers
    else:
        return data_points, identifiers

# --- OPTIMAL K-VALUE FINDER ---
def find_optimal_clusters(data, max_k):
    """Finds the optimal number of clusters using the Elbow Method."""
    print("\nFinding the optimal number of clusters using the Elbow Method...")
    inertias = []
    upper_k_bound = min(max_k, data.shape[0] - 1)
    k_range = range(2, upper_k_bound + 1)

    if len(k_range) < 1:
        print("Not enough data points to determine an optimal k > 1.")
        return 1

    print(f"Testing k from 2 to {upper_k_bound}...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42, output_type='cupy')
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k (Input: 2-mer Angles)')
    
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

# --- SAVE RESULTS ---
def save_results(identifiers, labels, kmeans_model, config_params, output_dir):
    """Saves the clustering results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to '{output_dir}/'")

    results_df = cudf.DataFrame({
        'Observation_Identifier': identifiers,
        'Assigned_Cluster': labels
    })
    
    assignments_filepath = os.path.join(output_dir, 'kmeans_cluster_assignments.csv')
    results_df.to_csv(assignments_filepath, index=False)
    print(f"  - Saved cluster assignments to {assignments_filepath}")

    summary_data = {
        "configuration": config_params,
        "cluster_centers": kmeans_model.cluster_centers_.get().tolist(),
        "inertia": float(kmeans_model.inertia_)
    }
    summary_filepath = os.path.join(output_dir, 'run_summary.json')
    with open(summary_filepath, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"  - Saved run summary and centroids to {summary_filepath}")
    return results_df

# --- VISUALIZATION ---
def plot_clusters(data, labels, centers, n_clusters, output_dir):
    """Visualizes the clusters using PCA and saves the plot."""
    if data.shape[0] < 2: return
    
    title = f'K-Means++ Clustering (Dihedrals, {n_clusters} clusters)'
    
    if data.shape[1] > 2:
        print("Data has >2 dimensions, running PCA for visualization...")
        pca = cuMLPCA(n_components=2)
        data_2d = pca.fit_transform(data)
        centers_2d = pca.transform(centers)
        xlabel, ylabel = 'Principal Component 1', 'Principal Component 2'
        title = f'PCA of K-Means++ Clustering (Dihedrals, {n_clusters} clusters)'
    else:
        data_2d, centers_2d = data, centers
        xlabel, ylabel = 'Feature 1', 'Feature 2'

    data_2d_cpu = data_2d.get()
    labels_cpu = labels.get()
    centers_2d_cpu = centers_2d.get()

    plt.figure(figsize=(12, 10))
    plt.scatter(data_2d_cpu[:, 0], data_2d_cpu[:, 1], c=labels_cpu, cmap='viridis', alpha=0.7)
    plt.scatter(centers_2d_cpu[:, 0], centers_2d_cpu[:, 1], c='red', s=250, marker='X', label='Centroids')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_filepath = os.path.join(output_dir, 'cluster_visualization.png')
    plt.savefig(plot_filepath)
    print(f"\nSaved cluster visualization to {plot_filepath}")
    plt.close()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dihedral_cluster.py <path_to_csv_file>")
        sys.exit(1)

    input_filepath = sys.argv[1]
    if not os.path.exists(input_filepath):
        print(f"Error: File not found at '{input_filepath}'")
        sys.exit(1)

    print("--- Running Dihedral Angle Clustering on GPU ---")
    
    data_points_gpu, identifiers_gpu = load_and_prepare_data(input_filepath, FEATURES_TO_CLUSTER, HANDLE_PERIODIC_DATA)

    if data_points_gpu is None or data_points_gpu.empty:
        print("Data loading failed or resulted in empty dataset. Exiting.")
        sys.exit(1)
        
    print(f"Successfully loaded and processed {len(data_points_gpu)} data points.")
    
    if isinstance(data_points_gpu, cudf.DataFrame):
        data_points_gpu = data_points_gpu.to_cupy()

    optimal_k = find_optimal_clusters(data_points_gpu, MAX_K_TO_TEST)

    if optimal_k and data_points_gpu.shape[0] > optimal_k:
        print(f"\nRunning final K-Means++ with {optimal_k} clusters on GPU...")
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42, output_type='cupy')
        labels = kmeans.fit_predict(data_points_gpu)
        
        output_directory = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # --- FIX APPLIED HERE ---
        config_params = {
            'input_file': input_filepath,
            'features_clustered': FEATURES_TO_CLUSTER,  # Corrected variable name
            'handled_periodic_data': HANDLE_PERIODIC_DATA,
            'num_clusters_found': int(optimal_k),      # Cast to standard Python int
            'k_mer_size': K_VALUE
        }
        # --- END OF FIX ---
        
        results_df = save_results(identifiers_gpu, labels, kmeans, config_params, output_directory)
        
        print("\n--- Clustering Results (head) ---")
        print(results_df.head().to_pandas().to_string())

        plot_clusters(data_points_gpu, labels, kmeans.cluster_centers_, optimal_k, output_directory)
        
        print("\n--- Analysis Complete ---")
    else:
        print("\nCould not determine optimal k or not enough data to cluster. Exiting.")