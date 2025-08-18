import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.colors as pcolors
from tqdm import tqdm
import sys
import argparse
from typing import List, Optional

class ResidueVisualizer:
    """
    A class to load residue data and generate interactive 3D visualizations
    of Ramachandran plots, saved as self-contained HTML files.
    """
    def __init__(self, residue: str):
        """
        Initializes the visualizer by loading the residue data from a CSV file.

        Args:
            residue (str): The residue string (e.g., 'ALA', 'GLY').
        """
        self.residue = residue
        k_value = len(residue)
        self.input_file = os.path.join(f"data/k{k_value}", f"{self.residue}.csv")
        self.df = None
        
        try:
            self.df = pd.read_csv(self.input_file)
            # This check is a flaw, but retained as requested.
            if len(sys.argv) > 1 and not sys.argv[1].isdigit():
                print(f"Successfully loaded data for residue '{self.residue}' from '{self.input_file}'.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file '{self.input_file}' was not found.")
        except pd.errors.EmptyDataError:
            print(f"Warning: The file '{self.input_file}' is empty.")
            self.df = pd.DataFrame()

    def _build_figure(self, initial_colorscale: str = 'nRainbow', initial_log_scale: bool = False) -> go.Figure:
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

        x_180, y_180, z_180, log_z_180 = _prepare_range_data(use_360=False)
        x_360, y_360, z_360, log_z_360 = _prepare_range_data(use_360=True)

        def reverse_colorscale(colorscale_name):
            """Gets a Plotly colorscale and reverses it."""
            original = pcolors.get_colorscale(colorscale_name)
            return [[1 - stop, color] for stop, color in reversed(original)]

        nRainbow = [
            [0.0, 'rgb(0,0,200)'], [0.125, 'rgb(0,25,255)'], [0.25, 'rgb(0,152,255)'],
            [0.375, 'rgb(44,255,150)'], [0.5, 'rgb(151,255,0)'], [0.625, 'rgb(255,234,0)'],
            [0.75, 'rgb(255,111,0)'], [0.875, 'rgb(255,0,0)'], [1.0, 'rgb(0,0,0)']
        ]

        available_colorscales = {
            'nRainbow': nRainbow,
            'Magma': pcolors.get_colorscale('Magma'),
            'Magma (Reverse)': reverse_colorscale('Magma'),
            'Viridis': pcolors.get_colorscale('Viridis')
        }

        resolved_initial_colorscale = available_colorscales[initial_colorscale]

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

        fig = go.Figure()
        fig.add_trace(go.Surface(z=z_180, x=x_180, y=y_180,
                                 surfacecolor=log_z_180 if initial_log_scale else z_180,
                                 colorscale=resolved_initial_colorscale, cmin=np.nanmin(log_z_180) if initial_log_scale else 1,
                                 colorbar=dict(title='Count', len=0.75), connectgaps=False, visible=True))
        fig.add_trace(go.Surface(z=z_360, x=x_360, y=y_360,
                                 surfacecolor=log_z_360 if initial_log_scale else z_360,
                                 colorscale=resolved_initial_colorscale, cmin=np.nanmin(log_z_360) if initial_log_scale else 1,
                                 colorbar=dict(title='Count', len=0.75), connectgaps=False, visible=False))
        
        fig.update_layout(
            title=None, # <<< MODIFICATION: Title removed from plot
            scene=dict(
                xaxis_title='Phi (φ) / tau(NA) [degrees]', yaxis_title='Psi (ψ) / tau(AC) [degrees]',
                zaxis_title='Frequency Count (log)' if initial_log_scale else 'Frequency Count',
                xaxis=dict(range=[-180, 180], tickmode='linear', dtick=60), 
                yaxis=dict(range=[-180, 180], tickmode='linear', dtick=60),
                zaxis_type='log' if initial_log_scale else 'linear'
            ),
            width=None, height=None,
            margin=dict(l=65, r=50, b=65, t=90), # Adjusted top margin
            updatemenus=[colorscale_dropdown, scale_buttons, range_buttons]
        )
        return fig
            
    def save_interactive_html(self, colorscale: str = 'nRainbow', use_log_scale: bool = False, all_residues: Optional[List[str]] = None):
        output_dir = "visualizations"
        os.makedirs(output_dir, exist_ok=True)
        fig = self._build_figure(initial_colorscale=colorscale, initial_log_scale=use_log_scale)
        if fig:
            output_path = os.path.join(output_dir, f"{self.residue}_ramachandran.html")
            
            plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn', default_height="80vh")

            # --- MODIFICATION: Generate navigation bar if residue list is provided ---
            nav_bar_html = ""
            if all_residues and len(all_residues) > 1:
                buttons_html = ""
                for res_name in sorted(all_residues):
                    file_name = f"{res_name}_ramachandran.html"
                    active_class = "active" if res_name == self.residue else ""
                    buttons_html += f'<a href="{file_name}" class="residue-btn {active_class}">{res_name}</a>\n'
                
                nav_bar_html = f"""
                <div class="controls-container">
                    <h2 id="plot-title">3D Ramachandran Plot for Residue: '{self.residue}'</h2>
                    <div class="residue-btn-wrapper">
                        {buttons_html}
                    </div>
                </div>
                """

            html_template = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Ramachandran Plot: {self.residue}</title>
                <style>
                    body, html {{
                        margin: 0; padding: 0;
                        width: 100%; height: 100%;
                        overflow-x: hidden;
                        font-family: sans-serif;
                    }}
                    .plot-container {{
                        width: 100vw;
                        height: 80vh;
                    }}
                    /* --- MODIFICATION: CSS for new elements --- */
                    .controls-container {{ text-align: center; padding: 15px 0; }}
                    #plot-title {{ margin-top: 0; margin-bottom: 15px; font-size: 24px; font-weight: bold; }}
                    .residue-btn-wrapper {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; max-width: 90%; margin: auto; }}
                    .residue-btn {{
                        padding: 8px 12px;
                        border: 1px solid #ccc;
                        background-color: #f0f0f0;
                        border-radius: 16px; /* Rounded edges */
                        cursor: pointer;
                        font-size: 14px;
                        color: #333;
                        text-decoration: none;
                        transition: background-color 0.2s, border-color 0.2s;
                    }}
                    .residue-btn:hover {{ background-color: #e0e0e0; border-color: #bbb; }}
                    .residue-btn.active {{
                        background-color: #d0d0d0;
                        border-color: #999;
                        font-weight: bold;
                    }}
                </style>
            </head>
            <body>
                <div class="plot-container">
                    {plot_div}
                </div>
                {nav_bar_html}
            </body>
            </html>
            """
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_template)

            # This check is a flaw, but retained as requested.
            if len(sys.argv) > 1 and not sys.argv[1].isdigit():
                print(f"Successfully saved interactive plot to '{output_path}'.")

def process_residue(residue_name: str, args, all_residues: Optional[List[str]] = None):
    try:
        visualizer = ResidueVisualizer(residue=residue_name)
        if visualizer.df is not None and not visualizer.df.empty:
            # --- MODIFICATION: Pass the list of all residues ---
            visualizer.save_interactive_html(
                colorscale=args.colorscale, 
                use_log_scale=args.log_scale,
                all_residues=all_residues 
            )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while processing '{residue_name}': {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate interactive 3D Ramachandran plots for residues and save them as HTML files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("target", type=str, help="A residue string (e.g. 'ALA') or a k-value (e.g. '3') for batch processing.")
    parser.add_argument("--colorscale", type=str, default="nRainbow", help="Initial Plotly colorscale to use.")
    parser.add_argument("--log_scale", action="store_true", help="Set the initial view to a logarithmic scale.")
    args = parser.parse_args()

    if args.target.isdigit():
        k_value = int(args.target)
        kmer_dir = f"data/k{k_value}"
        if not os.path.isdir(kmer_dir):
            print(f"Error: Directory '{kmer_dir}' not found.", file=sys.stderr)
            sys.exit(1)
        
        all_files = sorted([f for f in os.listdir(kmer_dir) if f.lower().endswith('.csv')])
        
        # In this specific case, we only want the 20 standard amino acids for the nav bar
        if k_value == 1:
            valid_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            csv_files = [f for f in all_files if os.path.splitext(f)[0] in valid_amino_acids]
        else:
            # For other k-values, all found CSVs will be linked
            csv_files = all_files

        if not csv_files:
            print(f"No valid residue .csv files found in '{kmer_dir}'.")
            sys.exit(0)
            
        # --- MODIFICATION: Get list of all residue names before processing ---
        residue_names_for_nav = [os.path.splitext(f)[0].upper() for f in csv_files]

        print(f"--- Batch Mode: Found {len(csv_files)} residues to process in '{kmer_dir}' ---")
        for filename in tqdm(csv_files, desc="Saving HTML plots"):
            residue_name = os.path.splitext(filename)[0]
            process_residue(residue_name.upper(), args, all_residues=residue_names_for_nav)
        print(f"--- Batch processing complete. Files saved in 'visualizations/' directory. ---")
    else:
        # Running in single mode, no navigation bar will be created
        residue = args.target.upper()
        print(f"--- Single Mode: Generating HTML plot for '{residue}' ---")
        process_residue(residue, args, all_residues=None)