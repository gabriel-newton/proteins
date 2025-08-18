import os
import pandas as pd
import cudf # Use cuDF for potential GPU-accelerated data loading
import cupy as cp # cuML uses cupy arrays
from cuml.cluster import KMeans # Import KMeans from cuML
from cuml.decomposition import PCA
import matplotlib.pyplot as plt
from kneed import KneeLocator
import json
from datetime import datetime
import glob
import sys

# --- CONFIGURATION ---
INVARIANTS_TO_CLUSTER = ['tau(NA)', 'tau(AC)']
HANDLE_PERIODIC_DATA = True
MAX_K_TO_TEST = 10

# --- DATA LOADING AND PROCESSING ---
def process_data_resumable(k_val, invariants, handle_periodic, progress_file):
    base_dir = f'k{k_val}'
    processed_files = set()
    
    if os.path.exists(progress_file):
        print(f"Found existing progress file: '{progress_file}'. Resuming.")
        progress_df = pd.read_csv(progress_file)
        if not progress_df.empty:
            processed_files = set(progress_df['Observation_Identifier'].apply(lambda x: x.split(':')[0]))
            print(f"Found {len(processed_files)} already processed k-mer files. They will be skipped.")
    else:
        if handle_periodic:
            dim = k_val * len(invariants) * 2
        else:
            dim = k_val * len(invariants)
        header_cols = ['Observation_Identifier'] + [f'feature_{i}' for i in range(dim)]
        pd.DataFrame(columns=header_cols).to_csv(progress_file, index=False)

    print(f"\nScanning and processing data from '{base_dir}/'...")
    if not os.path.isdir(base_dir):
        print(f"Error: Directory '{base_dir}' not found.")
        return

    all_csv_files = glob.glob(os.path.join(base_dir, '*.csv'))
    files_to_process = [f for f in all_csv_files if os.path.splitext(os.path.basename(f))[0] not in processed_files]

    if not files_to_process:
        print("All files have already been processed.")
        return

    print(f"Processing {len(files_to_process)} new k-mer files...")
    for file_path in files_to_process:
        try:
            kmer_filename = os.path.splitext(os.path.basename(file_path))[0]
            df = pd.read_csv(file_path) # Read with pandas, as it's often faster for small files
            grouped = df.groupby(['source_file', 'start_location'])
            file_results = []

            for (source, start_loc), group_df in grouped:
                if len(group_df) != k_val:
                    continue

                group_df = group_df.sort_values(by='position_in_kmer')
                point_df = group_df[invariants]

                if handle_periodic:
                    radians_df = np.deg2rad(point_df)
                    transformed_data = pd.DataFrame()
                    for col in radians_df.columns:
                        transformed_data[f'{col}_cos'] = np.cos(radians_df[col])
                        transformed_data[f'{col}_sin'] = np.sin(radians_df[col])
                    flattened_point = transformed_data.to_numpy().flatten()
                else:
                    flattened_point = point_df.to_numpy().flatten()
                
                source_id = source.split('\\')[-1]
                identifier = f"{kmer_filename}:{source_id}:{start_loc}"
                file_results.append([identifier] + flattened_point.tolist())

            if file_results:
                results_df = pd.DataFrame(file_results)
                results_df.to_csv(progress_file, mode='a', header=False, index=False)

        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")

    print("\nData processing step complete.")

# --- OPTIMAL K-VALUE FINDER ---
def find_optimal_clusters(data, max_k, k_val):
    print("\nFinding the optimal number of clusters using the Elbow Method...")
    inertias = []
    upper_k_bound = min(max_k, data.shape[0] - 1)
    k_range = range(2, upper_k_bound + 1)

    if len(k_range) < 1:
        print("Not enough data points to determine an optimal k > 1.")
        return 1

    print(f"Testing k from 2 to {upper_k_bound}...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title(f'Elbow Method For Optimal k (k-mer size = {k_val})')
    
    plot_filename = f'elbow_plot_k{k_val}.png'

    try:
        kneedle = KneeLocator(k_range, inertias, S=1.0, curve="convex", direction="decreasing")
        optimal_k = kneedle.elbow
        print(f"Found optimal number of clusters: {optimal_k}")
        plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles='--', color='r', label=f'Elbow at k={optimal_k}')
        plt.legend()
        plt.savefig(plot_filename)
        print(f"Saved elbow plot to {plot_filename}")
        plt.close()
        return optimal_k
    except Exception as e:
        print(f"Could not automatically find elbow point: {e}")
        plt.savefig(plot_filename)
        print(f"Saved elbow plot to {plot_filename}")
        plt.close()
        return None

# --- SAVE RESULTS ---
def save_results(results_df, kmeans_model, config_params, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to '{output_dir}/'")

    assignments_filepath = os.path.join(output_dir, 'cluster_assignments.csv')
    # If results_df is a cuDF dataframe, convert to pandas to save
    if hasattr(results_df, 'to_pandas'):
        results_df.to_pandas().to_csv(assignments_filepath, index=False)
    else:
        results_df.to_csv(assignments_filepath, index=False)
    print(f"  - Saved cluster assignments to {assignments_filepath}")

    summary_data = {
        "configuration": config_params,
        "cluster_centers": kmeans_model.cluster_centers_.get().tolist(), # Use .get() to move from GPU to CPU
        "inertia": float(kmeans_model.inertia_) # Cast inertia to float
    }
    summary_filepath = os.path.join(output_dir, 'run_summary.json')
    with open(summary_filepath, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"  - Saved run summary and centroids to {summary_filepath}")

# --- VISUALIZATION ---
def plot_clusters(data, labels, centers, k_val, n_clusters, output_dir):
    if data.shape[0] < 1: return
    
    # Move data to CPU for plotting with Matplotlib
    data_cpu = data.get()
    labels_cpu = labels.get()
    centers_cpu = centers.get()

    title = f'K-Means++ Clustering (k={k_val}, {n_clusters} clusters)'
    
    if data_cpu.shape[1] > 2:
        # Use scikit-learn PCA for plotting as it's on the CPU
        from sklearn.decomposition import PCA as SklearnPCA
        pca = SklearnPCA(n_components=2)
        data_2d = pca.fit_transform(data_cpu)
        centers_2d = pca.transform(centers_cpu)
        xlabel, ylabel = 'Principal Component 1', 'Principal Component 2'
        title = f'PCA of K-Means++ Clustering (k={k_val}, {n_clusters} clusters)'
    else:
        data_2d, centers_2d = data_cpu, centers_cpu
        xlabel, ylabel = 'Feature 1', 'Feature 2'

    plt.figure(figsize=(12, 10))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels_cpu, cmap='viridis', alpha=0.7)
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=250, marker='X', label='Centroids')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_filepath = os.path.join(output_dir, 'cluster_visualization.png')
    plt.savefig(plot_filepath)
    print(f"Saved cluster visualization to {plot_filepath}")
    plt.close()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python kmer_clusters.py <k_value>")
        sys.exit(1)

    try:
        K_VALUE = int(sys.argv[1])
    except ValueError:
        print(f"Error: Invalid k_value '{sys.argv[1]}'. It must be an integer.")
        sys.exit(1)

    PROGRESS_CSV = f'processed_data_k{K_VALUE}.csv'
    print(f"--- Running K-mer Clustering for k = {K_VALUE} on GPU ---")

    process_data_resumable(K_VALUE, INVARIANTS_TO_CLUSTER, HANDLE_PERIODIC_DATA, PROGRESS_CSV)

    print(f"\nLoading all processed data from '{PROGRESS_CSV}' for analysis...")
    if not os.path.exists(PROGRESS_CSV) or os.path.getsize(PROGRESS_CSV) == 0:
        print("Error: Progress file is empty or does not exist. Cannot proceed.")
    else:
        # Load data using pandas, then clean
        full_data_df_pd = pd.read_csv(PROGRESS_CSV)
        initial_rows = len(full_data_df_pd)
        full_data_df_pd.dropna(inplace=True)
        final_rows = len(full_data_df_pd)
        if initial_rows != final_rows:
            print(f"\nCleaned data: Removed {initial_rows - final_rows} rows containing NaN values.")

        if full_data_df_pd.empty:
            print("No data points left after cleaning.")
        else:
            # Separate identifiers (on CPU) from data (to be moved to GPU)
            processed_identifiers = full_data_df_pd['Observation_Identifier'].values
            data_points_pd = full_data_df_pd.drop('Observation_Identifier', axis=1)

            # Move data to GPU using cuDF/CuPy for cuML
            print("Moving data to GPU...")
            data_points_gpu = cp.asarray(data_points_pd)

            optimal_k = find_optimal_clusters(data_points_gpu, MAX_K_TO_TEST, K_VALUE)

            if optimal_k and data_points_gpu.shape[0] > optimal_k:
                print(f"\nRunning final K-Means++ with {optimal_k} clusters on GPU...")
                kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
                kmeans.fit(data_points_gpu)
                
                # Create results dataframe using pandas as identifiers are on CPU
                results_df = pd.DataFrame({
                    'Observation_Identifier': processed_identifiers,
                    'Assigned_Cluster': kmeans.labels_.get() # .get() moves labels from GPU to CPU
                })
                print("\n--- Clustering Results (head) ---")
                print(results_df.head().to_string())

                config_params = {
                    'k_value': K_VALUE,
                    'invariants_clustered': INVARIANTS_TO_CLUSTER,
                    'handled_periodic_data': HANDLE_PERIODIC_DATA,
                    'optimal_clusters_found': int(optimal_k) if optimal_k is not None else None
                }
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                output_dir = f"results_k{config_params['k_value']}_{timestamp}"
                
                save_results(results_df, kmeans, config_params, output_dir)
                plot_clusters(data_points_gpu, kmeans.labels_, kmeans.cluster_centers_, K_VALUE, optimal_k, output_dir)
            
            elif optimal_k:
                print(f"\nCannot perform clustering. Data points ({data_points_gpu.shape[0]}) must be > clusters ({optimal_k}).")
