import pandas as pd
import numpy as np
import plotly.graph_objects as go
import argparse
import os
import sys

def create_toroidal_heatmap(df: pd.DataFrame, x_cos_col: str, x_sin_col: str, y_cos_col: str, y_sin_col: str, title: str, output_path: str):
    """
    Generates and saves an interactive, bumpy 3D toroidal heatmap with log/linear scaling,
    reconstructing angles from sin/cos columns.

    Args:
        df (pd.DataFrame): DataFrame with angle data in sin/cos format.
        x_cos_col (str): Column for the cosine of the major angle (phi).
        x_sin_col (str): Column for the sine of the major angle (phi).
        y_cos_col (str): Column for the cosine of the minor angle (psi).
        y_sin_col (str): Column for the sine of the minor angle (psi).
        title (str): Title for the plot.
        output_path (str): File path to save the HTML.
    """
    print(f"Processing data for: {title}...")

    # --- 1. Reconstruct Angles from Sin/Cos ---
    print("Reconstructing angles from sin/cos values...")
    x_angles_rad = np.arctan2(df[x_sin_col], df[x_cos_col])
    y_angles_rad = np.arctan2(df[y_sin_col], df[y_cos_col])
    
    reconstructed_df = pd.DataFrame({
        'phi_deg': np.rad2deg(x_angles_rad),
        'psi_deg': np.rad2deg(y_angles_rad)
    })

    # --- 2. Bin the data into a 2D frequency map ---
    print("Binning reconstructed angle data...")
    reconstructed_df['phi_round'] = reconstructed_df['phi_deg'].round().astype(int)
    reconstructed_df['psi_round'] = reconstructed_df['psi_deg'].round().astype(int)
    
    full_range = np.arange(-180, 181)
    freq_counts = reconstructed_df.groupby(['phi_round', 'psi_round']).size().reset_index(name='count')
    z_data = freq_counts.pivot_table(index='psi_round', columns='phi_round', values='count', fill_value=0)
    z_data = z_data.reindex(index=full_range, columns=full_range, fill_value=0)
    
    heatmap_values = z_data.values.astype(float)
    heatmap_values[heatmap_values == 0] = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        log_heatmap_values = np.log10(heatmap_values)

    # --- 3. Define Toroidal Coordinates ---
    # phi is the major angle (around the main ring), psi is the minor angle (around the tube)
    phi_deg = z_data.columns.values
    psi_deg = z_data.index.values
    
    phi = np.deg2rad(phi_deg)
    psi = np.deg2rad(psi_deg)
    
    phi, psi = np.meshgrid(phi, psi)

    # --- 4. Calculate Radii for "Bumpy" Effect ---
    R = 2.0  # Major radius (distance from center of hole to center of tube)
    r_base = 1.0 # Base minor radius (radius of the tube)
    bump_scaling_factor = 0.5 # Controls how much the surface juts out

    norm_linear = np.nan_to_num(heatmap_values) / np.nanmax(heatmap_values)
    norm_log = np.nan_to_num(log_heatmap_values) / np.nanmax(log_heatmap_values)

    r_linear = r_base + (norm_linear * bump_scaling_factor)
    r_log = r_base + (norm_log * bump_scaling_factor)

    # --- 5. Convert to Cartesian Coordinates using Toroidal Equations ---
    def to_cartesian(R_major, r_minor, phi_major, psi_minor):
        x = (R_major + r_minor * np.cos(psi_minor)) * np.cos(phi_major)
        y = (R_major + r_minor * np.cos(psi_minor)) * np.sin(phi_major)
        z = r_minor * np.sin(psi_minor)
        return x, y, z

    x_linear, y_linear, z_linear = to_cartesian(R, r_linear, phi, psi)
    x_log, y_log, z_log = to_cartesian(R, r_log, phi, psi)

    # --- 6. Build the Plotly Figure ---
    print(f"Building plot: {title}...")
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=x_linear, y=y_linear, z=z_linear,
        surfacecolor=heatmap_values,
        colorscale='Jet', cmin=1, cmax=np.nanpercentile(heatmap_values, 99.5),
        colorbar=dict(title='Frequency'),
        showscale=True,
        name='Linear Scale',
        visible=True
    ))

    fig.add_trace(go.Surface(
        x=x_log, y=y_log, z=z_log,
        surfacecolor=log_heatmap_values,
        colorscale='Jet', cmin=0, cmax=np.nanpercentile(log_heatmap_values, 99.5),
        colorbar=dict(title='log10(Frequency)'),
        showscale=True,
        name='Log Scale',
        visible=False
    ))

    # --- 7. Add Interactive Controls ---
    fig.update_layout(
        title_text=title,
        scene=dict(
            xaxis=dict(showticklabels=False, title='', backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(showticklabels=False, title='', backgroundcolor="rgb(230, 230,230)"),
            zaxis=dict(showticklabels=False, title='', backgroundcolor="rgb(230, 230,230)"),
            aspectratio=dict(x=1, y=1, z=0.5) # Adjust aspect ratio for better viewing
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1, xanchor="left", y=1.1, yanchor="top",
                buttons=[
                    dict(label="Linear Scale", method="update", args=[{"visible": [True, False]}, {"title": f"{title} (Linear Scale)"}]),
                    dict(label="Log Scale", method="update", args=[{"visible": [False, True]}, {"title": f"{title} (Log Scale)"}])
                ]
            )
        ]
    )

    fig.write_html(output_path, full_html=True, include_plotlyjs='cdn')
    print(f"Successfully saved to '{output_path}'")

def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive toroidal heatmap from sin/cos dihedral angle data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_csv", type=str, help="Path to the 2mer_angles8d.csv file.")
    parser.add_argument("--output_dir", type=str, default="toroidal_plots", help="Directory to save the output HTML file.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        column_names = [
            'phi_i_cos', 'phi_i_sin', 'psi_i_cos', 'psi_i_sin',
            'phi_i+1_cos', 'phi_i+1_sin', 'psi_i+1_cos', 'psi_i+1_sin'
        ]
        df = pd.read_csv(args.input_csv, header=None, names=column_names, skiprows=1, engine='python')
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_csv}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    output_file = os.path.join(args.output_dir, "toroidal_ramachandran.html")
    create_toroidal_heatmap(
        df=df,
        x_cos_col='phi_i_cos',
        x_sin_col='phi_i_sin',
        y_cos_col='psi_i_cos',
        y_sin_col='psi_i_sin',
        title='Toroidal Ramachandran Plot (phi_i vs psi_i)',
        output_path=output_file
    )

if __name__ == "__main__":
    main()
