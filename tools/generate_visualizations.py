"""
Standalone Tool & Pipeline Module: Data Generator (v7.2 - Restored Points Format)

Generates 3D heatmap data, unified 1D/2D statistics, and
1D context-dependent histograms for a pair of invariants.

This is the primary calculation engine for pipeline_02_cache_v7.2.py.

Example (standalone):
python tools/generate_visualizations.py /path/to/db.db --inv1 tau_NA --inv2 tau_AC --offset 1 --res1 A --res2 C --level level_1
"""
import pandas as pd
import numpy as np
import argparse
import json
from tools.pipeline_utils import query_data_for_comparison, get_invariant_limits, get_resolution_bins
from tools.pipeline_constants import (
    ALL_INVARIANTS, RESIDUE_CONTEXTS, RESOLUTION_LEVELS,
    TORSION_INVARIANTS, INVARIANT_TYPES, RESOLUTION_BINS
)

# --- Helper function to calculate 1D histogram ---
def _calculate_1d_histo(data_series, invariant_name):
    """Calculates 1D histogram for a pandas Series."""
    if data_series.empty or data_series.isnull().all():
        return None

    limits = get_invariant_limits()
    limit_def = limits.get(invariant_name, {'limit_min': -180, 'limit_max': 180})
    bin_min, bin_max = limit_def['limit_min'], limit_def['limit_max']

    # Use 360 bins for torsion angles
    counts, bin_edges = np.histogram(data_series, bins=360, range=(bin_min, bin_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {
        'bins': [round(b, 2) for b in bin_centers],
        'counts': [int(c) for c in counts]
    }

# --- Helper function for unified stats ---

def _get_bin_width(invariant_name):
    """Helper to get the physical bin size for freq_at_mean calc."""
    try:
        inv_type = INVARIANT_TYPES[invariant_name]
        res_level_1 = RESOLUTION_LEVELS[0]
        bin_width = RESOLUTION_BINS[res_level_1][inv_type]
        return bin_width
    except KeyError:
        return 1.0

def _calculate_raw_stats(data_x, data_y, inv1, inv2):
    """
    Calculates all statistics derived from the raw (unbinned)
    data_x and data_y pandas Series.
    """
    stats = {}
    stats['population'] = int(data_x.count())

    if stats['population'] == 0:
        return stats # Return early if no data

    stats['mean_x'] = round(data_x.mean(), 3)
    stats['variance_x'] = round(data_x.var(), 3)
    stats['median_x'] = round(data_x.median(), 3)
    stats['min_x'] = round(data_x.min(), 3)
    stats['max_x'] = round(data_x.max(), 3)

    stats['mean_y'] = round(data_y.mean(), 3)
    stats['variance_y'] = round(data_y.var(), 3)
    stats['median_y'] = round(data_y.median(), 3)
    stats['min_y'] = round(data_y.min(), 3)
    stats['max_y'] = round(data_y.max(), 3)

    stats['covariance'] = round(data_x.cov(data_y), 3)

    bin_width_x = _get_bin_width(inv1)
    window_x = bin_width_x / 2.0
    freq_x = data_x[(data_x >= stats['mean_x'] - window_x) & (data_x < stats['mean_x'] + window_x)].count()
    stats['freq_at_mean_x'] = int(freq_x)

    bin_width_y = _get_bin_width(inv2)
    window_y = bin_width_y / 2.0
    freq_y = data_y[(data_y >= stats['mean_y'] - window_y) & (data_y < stats['mean_y'] + window_y)].count()
    stats['freq_at_mean_y'] = int(freq_y)

    return stats


# --- Main Function ---
def generate_visualization_data(db_path, inv1, inv2, offset, res1, res2, res_level, TORSION_INVARIANTS_LIST):
    """
    Generates all data for a given invariant pair context:
    1. 3D Heatmap (as list of points)
    2. Unified 1D and 3D Stats (Raw + Binned)
    3. 1D Histograms (for Torsions)

    Returns:
    (heatmap_data_dict, stats_data_dict, histo_data_x, histo_data_y)
    """

    # 1. Query the data
    df = query_data_for_comparison(db_path, inv1, inv2, offset, res1, res2)

    heatmap_data = {'points': []} # Initialize with points key
    histo_data_x = None
    histo_data_y = None

    df.dropna(subset=['x', 'y'], inplace=True)
    data_x = df['x']
    data_y = df['y']

    # 2. Calculate ALL Raw Stats
    stats_data = _calculate_raw_stats(data_x, data_y, inv1, inv2)
    population = stats_data.get('population', 0)

    if population == 0:
        return heatmap_data, stats_data, histo_data_x, histo_data_y

    # 3. Calculate 1D Histograms
    if inv1 in TORSION_INVARIANTS_LIST:
        histo_data_x = _calculate_1d_histo(data_x, inv1)
    if inv2 in TORSION_INVARIANTS_LIST:
        histo_data_y = _calculate_1d_histo(data_y, inv2)

    # 4. Calculate 3D Heatmap Data
    limits = get_invariant_limits()
    x_bins = get_resolution_bins(inv1, res_level)
    y_bins = get_resolution_bins(inv2, res_level)
    x_lim = (limits[inv1]['limit_min'], limits[inv1]['limit_max'])
    y_lim = (limits[inv2]['limit_min'], limits[inv2]['limit_max'])

    H, xedges, yedges = np.histogram2d(
        data_x,
        data_y,
        bins=[x_bins, y_bins],
        range=[x_lim, y_lim]
    )

    # 5. Calculate Binned 2D Stats (Peak Stats)
    if H.size > 0: # Check if histogram is not empty
        peak_freq_raw = H.max()
        peak_indices = np.unravel_index(H.argmax(), H.shape)
        peak_x_center = (xedges[peak_indices[0]] + xedges[peak_indices[0] + 1]) / 2
        peak_y_center = (yedges[peak_indices[1]] + yedges[peak_indices[1] + 1]) / 2
        stats_data['peak_x'] = round(peak_x_center, 2)
        stats_data['peak_y'] = round(peak_y_center, 2)
        stats_data['peak_freq'] = int(peak_freq_raw)
    else: # Handle empty histogram case for peak stats
        stats_data['peak_x'] = None
        stats_data['peak_y'] = None
        stats_data['peak_freq'] = 0


    # --- 6. Format Heatmap Data for JSON (RESTORED 'points' format) ---
    points_list = []
    # Iterate through the histogram bins (H shape is (xbins, ybins))
    for i in range(H.shape[0]): # Iterate x bins
        for j in range(H.shape[1]): # Iterate y bins
            count = H[i, j]
            if count > 0: # Only include bins with data
                x_center = (xedges[i] + xedges[i+1]) / 2
                y_center = (yedges[j] + yedges[j+1]) / 2
                # Append tuple: (x, y, count) - rounded
                points_list.append((round(x_center, 2), round(y_center, 2), int(count)))

    heatmap_data = {'points': points_list}
    # --- END RESTORED 'points' format ---

    # 7. Return all four items
    return heatmap_data, stats_data, histo_data_x, histo_data_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D heatmap (points), 1D stats, and 1D histo data.")
    parser.add_argument("db_path", type=str, help="Path to the SQLite database file.")
    parser.add_argument("--inv1", type=str, required=True, choices=ALL_INVARIANTS, help="Invariant 1 (x-axis).")
    parser.add_argument("--inv2", type=str, required=True, choices=ALL_INVARIANTS, help="Invariant 2 (y-axis).")
    parser.add_argument("--offset", type=int, default=0, choices=range(5), help="Residue offset (0-4).")
    parser.add_argument("--res1", type=str, default="Any", choices=RESIDUE_CONTEXTS, help="Residue context for invariant 1.")
    parser.add_argument("--res2", type=str, default="Any", choices=RESIDUE_CONTEXTS, help="Residue context for invariant 2.")
    parser.add_argument("--level", type=str, default="level_1", choices=RESOLUTION_LEVELS, help="Resolution level for heatmap.")
    args = parser.parse_args()

    print(f"Generating data for {args.inv1} vs {args.inv2} (Offset +{args.offset}, {args.res1}-{args.res2}, {args.level} res)")

    heatmap, stats, histo_x, histo_y = generate_visualization_data(
        args.db_path, args.inv1, args.inv2, args.offset,
        args.res1, args.res2, args.level, TORSION_INVARIANTS
    )

    print("\n--- Unified Stats ---")
    print(json.dumps(stats, indent=4))

    if histo_x:
        print(f"\n--- Histo X ({args.inv1}) ---")
        # Ensure histo_x is not None before accessing keys
        if histo_x: print(f"Bins: {len(histo_x.get('bins', []))}, Total Count: {sum(histo_x.get('counts', []))}")

    if histo_y:
        print(f"\n--- Histo Y ({args.inv2}) ---")
        if histo_y: print(f"Bins: {len(histo_y.get('bins', []))}, Total Count: {sum(histo_y.get('counts', []))}")

    output_file = f"vizdata_{args.inv1}_vs_{args.inv2}+{args.offset}_{args.res1}_{args.res2}_{args.level}.json"
    with open(output_file, 'w') as f:
        json.dump({
            # Saving heatmap under 'heatmap' key still, but content is now {'points':...}
            "heatmap": heatmap,
            "statistics": stats,
            "histogram_x": histo_x,
            "histogram_y": histo_y
        }, f, indent=4)

    print(f"\nFull data saved to {output_file}")