import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.colors as pcolors
from tqdm import tqdm
import sys
import argparse

class KmerVisualizer:
    """
    A class to load k-mer data and generate interactive 3D visualizations
    of Ramachandran plots, saved as self-contained HTML files.
    """
    def __init__(self, kmer: str):
        """
        Initializes the visualizer by loading the k-mer data from a CSV file.

        Args:
            kmer (str): The k-mer string (e.g., 'ALA', 'GLY').
        """
        self.kmer = kmer
        k_value = len(kmer)
        self.input_file = os.path.join(f"data/k{k_value}", f"{kmer}.csv")
        self.df = None
        
        try:
            self.df = pd.read_csv(self.input_file)
            if not sys.argv[1].isdigit():
                print(f"Successfully loaded data for k-mer '{self.kmer}' from '{self.input_file}'.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file '{self.input_file}' was not found.")
        except pd.errors.EmptyDataError:
            print(f"Warning: The file '{self.input_file}' is empty.")
            self.df = pd.DataFrame()

    def _build_figure(self, initial_colorscale: str = 'Magma', initial_log_scale: bool = False) -> go.Figure:
        """
        Constructs the Plotly figure with interactive controls for colorscale, log/linear scaling, and angle range.
        """
        if self.df is None or self.df.empty:
            return None

        required_cols = ['tau(NA)', 'tau(AC)']
        if not all(col in self.df.columns for col in required_cols):
            print(f"Error: Missing required columns in {self.input_file}", file=sys.stderr)
            return None

        def _prepare_range_data(use_360: bool):
            """Helper function to generate plot data for a specific angle range."""
            angle_data = self.df[required_cols].dropna().copy()
            if use_360:
                angle_data['phi'] = (angle_data['tau(NA)'].round() % 360).astype(int)
                angle_data['psi'] = (angle_data['tau(AC)'].round() % 360).astype(int)
                full_range = np.arange(0, 361)
            else:
                angle_data['phi'] = angle_data['tau(NA)'].round().astype(int)
                angle_data['psi'] = angle_data['tau(AC)'].round().astype(int)
                full_range = np.arange(-180, 181)
            
            freq_counts = angle_data.groupby(['phi', 'psi']).size().reset_index(name='count')
            z_data = freq_counts.pivot_table(index='psi', columns='phi', values='count', fill_value=0)
            z_data = z_data.reindex(index=full_range, columns=full_range, fill_value=0)
            
            z_numeric = z_data.values.astype(float)
            z_numeric[z_numeric == 0] = np.nan
            with np.errstate(invalid='ignore'):
                log_z = np.log10(z_numeric)
            return z_data.columns, z_data.index, z_numeric, log_z

        # --- Data Preparation for both ranges ---
        x_180, y_180, z_180, log_z_180 = _prepare_range_data(use_360=False)
        x_360, y_360, z_360, log_z_360 = _prepare_range_data(use_360=True)

        # --- Manually define colorscales for reliability ---
        def reverse_colorscale(colorscale_name):
            """Gets a Plotly colorscale and reverses it."""
            original = pcolors.get_colorscale(colorscale_name)
            return [[1 - stop, color] for stop, color in reversed(original)]

        # MODIFIED: Replaced the custom rainbow scale with your specific definition
        custom_rainbow = [
            [0.0, 'rgb(0,0,200)'],
            [0.125,  'rgb(0,25,255)'],
            [0.25, 'rgb(0,152,255)'],
            [0.375,   'rgb(44,255,150)'],
            [0.5, 'rgb(151,255,0)'],
            [0.625,  'rgb(255,234,0)'],
            [0.75, 'rgb(255,111,0)'],
            [0.875,   'rgb(255,0,0)'],
            [1.0,   'rgb(0,0,0)']
        ]

        # MODIFIED: Updated the list of available colorscales
        available_colorscales = {
            'Rainbow': custom_rainbow
            'Magma': 'Magma',
            'Magma (Reverse)': reverse_colorscale('Magma'),
            'Viridis': 'Viridis'
        }

        # The 'args' now target both traces by providing a list of two values.
        colorscale_dropdown = dict(buttons=[dict(label=name, method="restyle", args=[{"colorscale": [scale, scale]}]) for name, scale in available_colorscales.items()],
                                 direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.05, xanchor="left", y=1.15, yanchor="top")

        scale_buttons = dict(type="buttons", direction="left", pad={"r": 10, "t": 10}, showactive=True, x=0.35, xanchor="left", y=1.15, yanchor="top",
                             buttons=[
                                 dict(label="Linear", method="update", args=[{"surfacecolor": [z_180, z_360]}, {"scene.zaxis.type": "linear", "scene.zaxis.title": "Frequency Count"}]),
                                 dict(label="Log", method="update", args=[{"surfacecolor": [log_z_180, log_z_360]}, {"scene.zaxis.type": "log", "scene.zaxis.title": "Frequency Count (log)"}])
                             ])
        
        range_buttons = dict(type="buttons", direction="left", pad={"r": 10, "t": 10}, showactive=True, x=0.65, xanchor="left", y=1.15, yanchor="top",
                             buttons=[
                                 dict(label="-180° to 180°", method="update", args=[{"visible": [True, False]}, {"scene.xaxis.range": [-180, 180], "scene.yaxis.range": [-180, 180]}]),
                                 dict(label="0° to 360°", method="update", args=[{"visible": [False, True]}, {"scene.xaxis.range": [0, 360], "scene.yaxis.range": [0, 360]}])
                             ])

        # --- Figure Creation ---
        fig = go.Figure()
        # Add trace for -180 to 180 range (visible by default)
        fig.add_trace(go.Surface(z=z_180, x=x_180, y=y_180,
                                 surfacecolor=log_z_180 if initial_log_scale else z_180,
                                 colorscale=initial_colorscale, cmin=np.nanmin(log_z_180) if initial_log_scale else 1,
                                 colorbar=dict(title='Count', len=0.75), connectgaps=False, visible=True))
        # Add trace for 0 to 360 range (hidden by default)
        fig.add_trace(go.Surface(z=z_360, x=x_360, y=y_360,
                                 surfacecolor=log_z_360 if initial_log_scale else z_360,
                                 colorscale=initial_colorscale, cmin=np.nanmin(log_z_360) if initial_log_scale else 1,
                                 colorbar=dict(title='Count', len=0.75), connectgaps=False, visible=False))
        
        fig.update_layout(
            title=f"3D Ramachandran Plot for k-mer: '{self.kmer}'",
            scene=dict(
                xaxis_title='Phi (φ) / tau(NA) [degrees]', yaxis_title='Psi (ψ) / tau(AC) [degrees]',
                zaxis_title='Frequency Count (log)' if initial_log_scale else 'Frequency Count',
                xaxis=dict(range=[-180, 180], tickmode='linear', dtick=60), 
                yaxis=dict(range=[-180, 180], tickmode='linear', dtick=60),
                zaxis_type='log' if initial_log_scale else 'linear'
            ),
            width=900, height=800, margin=dict(l=65, r=50, b=65, t=120),
            updatemenus=[colorscale_dropdown, scale_buttons, range_buttons]
        )
        return fig
            
    def save_interactive_html(self, colorscale: str = 'Magma', use_log_scale: bool = False):
        output_dir = "visualizations"
        os.makedirs(output_dir, exist_ok=True)
        fig = self._build_figure(initial_colorscale=colorscale, initial_log_scale=use_log_scale)
        if fig:
            output_path = os.path.join(output_dir, f"{self.kmer}_ramachandran.html")
            fig.write_html(output_path, include_plotlyjs='cdn')
            if not sys.argv[1].isdigit():
                print(f"Successfully saved interactive plot to '{output_path}'.")

def process_kmer(kmer_name, args):
    try:
        visualizer = KmerVisualizer(kmer=kmer_name)
        if visualizer.df is not None and not visualizer.df.empty:
            visualizer.save_interactive_html(colorscale=args.colorscale, use_log_scale=args.log_scale)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while processing '{kmer_name}': {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate interactive 3D Ramachandran plots for k-mers and save them as HTML files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("target", type=str, help="A k-mer string (e.g. 'ALA') or a k-value (e.g. '3') for batch processing.")
    parser.add_argument("--colorscale", type=str, default="Magma", help="Initial Plotly colorscale to use.")
    parser.add_argument("--log_scale", action="store_true", help="Set the initial view to a logarithmic scale.")
    args = parser.parse_args()

    if args.target.isdigit():
        k_value = int(args.target)
        kmer_dir = f"data/k{k_value}"
        if not os.path.isdir(kmer_dir):
            print(f"Error: Directory '{kmer_dir}' not found.", file=sys.stderr)
            sys.exit(1)
        
        csv_files = sorted([f for f in os.listdir(kmer_dir) if f.lower().endswith('.csv')])
        if not csv_files:
            print(f"No .csv files found in '{kmer_dir}'.")
            sys.exit(0)
            
        print(f"--- Batch Mode: Found {len(csv_files)} k-mers to process in '{kmer_dir}' ---")
        for filename in tqdm(csv_files, desc="Saving HTML plots"):
            kmer_name = os.path.splitext(filename)[0]
            process_kmer(kmer_name.upper(), args)
        print(f"--- Batch processing complete. Files saved in 'visualizations/' directory. ---")
    else:
        kmer = args.target.upper()
        print(f"--- Single Mode: Generating HTML plot for '{kmer}' ---")
        process_kmer(kmer, args)

