import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.colors as pcolors
from tqdm import tqdm
import sys
import argparse
from typing import List, Optional
import json

class ResidueVisualizer:
    """
    A class to load residue data and generate interactive 3D visualizations
    of Ramachandran plots, saved as self-contained HTML files.
    """
    def __init__(self, residue: str):
        self.residue = residue
        k_value = len(residue)
        self.input_file = os.path.join(f"data/k{k_value}", f"{self.residue}.csv")
        self.df = None
        
        try:
            self.df = pd.read_csv(self.input_file, engine='python')
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file '{self.input_file}' was not found.")
        except pd.errors.EmptyDataError:
            print(f"Warning: The file '{self.input_file}' is empty.")
            self.df = pd.DataFrame()

    def _build_figure(self) -> go.Figure:
        if self.df is None or self.df.empty:
            return None

        required_cols = ['tau(NA)', 'tau(AC)']
        if not all(col in self.df.columns for col in required_cols):
            print(f"Error: Missing required columns in {self.input_file}", file=sys.stderr)
            return None

        def _prepare_range_data(use_360: bool):
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
            original = pcolors.get_colorscale(colorscale_name)
            return [[1 - stop, color] for stop, color in reversed(original)]

        nRainbow = [
            [0.0, 'rgb(0,0,200)'], [0.125, 'rgb(0,25,255)'], [0.25, 'rgb(0,152,255)'],
            [0.375, 'rgb(44,255,150)'], [0.5, 'rgb(151,255,0)'], [0.625, 'rgb(255,234,0)'],
            [0.75, 'rgb(255,111,0)'], [0.875, 'rgb(255,0,0)'], [1.0, 'rgb(0,0,0)']
        ]

        self.available_colorscales = {
            'nRainbow': nRainbow, 'Magma': pcolors.get_colorscale('Magma'),
            'Magma_Reverse': reverse_colorscale('Magma'), 'Viridis': pcolors.get_colorscale('Viridis')
        }
        
        z_180_list = np.where(np.isfinite(z_180), z_180, None).tolist()
        z_360_list = np.where(np.isfinite(z_360), z_360, None).tolist()
        log_z_180_list = np.where(np.isfinite(log_z_180), log_z_180, None).tolist()
        log_z_360_list = np.where(np.isfinite(log_z_360), log_z_360, None).tolist()
        
        self.scale_button_args = [
            [{"surfacecolor": [z_180_list, z_360_list]}, {"scene.zaxis.type": "linear", "scene.zaxis.title": "Frequency Count"}],
            [{"surfacecolor": [log_z_180_list, log_z_360_list]}, {"scene.zaxis.type": "log", "scene.zaxis.title": "Frequency Count (log)"}]
        ]
        
        self.range_button_args = [
            [{"visible": [True, False]}, {"scene.xaxis.range": [-180, 180], "scene.yaxis.range": [-180, 180]}],
            [{"visible": [False, True]}, {"scene.xaxis.range": [0, 360], "scene.yaxis.range": [0, 360]}]
        ]
        
        colorscale_buttons = [dict(label=name.replace('_', ' '), method="restyle", args=[{"colorscale": [scale, scale]}]) for name, scale in self.available_colorscales.items()]
        colorscale_dropdown = dict(buttons=colorscale_buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.05, xanchor="left", y=1.15, yanchor="top")
        scale_buttons = dict(type="buttons", direction="left", pad={"r": 10, "t": 10}, showactive=True, x=0.35, xanchor="left", y=1.15, yanchor="top",
                             buttons=[dict(label="Linear", method="update", args=self.scale_button_args[0]), dict(label="Log", method="update", args=self.scale_button_args[1])])
        range_buttons = dict(type="buttons", direction="left", pad={"r": 10, "t": 10}, showactive=True, x=0.65, xanchor="left", y=1.15, yanchor="top",
                             buttons=[dict(label="-180° to 180°", method="update", args=self.range_button_args[0]), dict(label="0° to 360°", method="update", args=self.range_button_args[1])])

        fig = go.Figure()
        fig.add_trace(go.Surface(z=z_180, x=x_180, y=y_180, surfacecolor=z_180, colorscale=nRainbow, cmin=1, colorbar=dict(title='Count', len=0.75), connectgaps=False, visible=True))
        fig.add_trace(go.Surface(z=z_360, x=x_360, y=y_360, surfacecolor=z_360, colorscale=nRainbow, cmin=1, colorbar=dict(title='Count', len=0.75), connectgaps=False, visible=False))
        
        fig.update_layout(title=None, scene=dict(xaxis_title='Phi (φ) / tau(NA) [degrees]', yaxis_title='Psi (ψ) / tau(AC) [degrees]', zaxis_title='Frequency Count', xaxis=dict(range=[-180, 180], tickmode='linear', dtick=60), yaxis=dict(range=[-180, 180], tickmode='linear', dtick=60), zaxis_type='linear'), width=None, height=None, margin=dict(l=65, r=50, b=65, t=90), updatemenus=[colorscale_dropdown, scale_buttons, range_buttons])
        return fig
            
    def save_interactive_html(self, all_residues: Optional[List[str]] = None):
        output_dir = "visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        fig = self._build_figure()
        if not fig: return

        plot_div_id = "ramachandranPlot"
        output_path = os.path.join(output_dir, f"{self.residue}_ramachandran.html")
        plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn', default_height="80vh", div_id=plot_div_id)

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
        
        colorscales_json = json.dumps(self.available_colorscales)
        scale_args_json = json.dumps(self.scale_button_args)
        range_args_json = json.dumps(self.range_button_args)

        # --- JAVASCRIPT CORRECTION USING URL HASH ---
        javascript_code = f"""
        <script>
            (function() {{
                const plotDiv = document.getElementById('{plot_div_id}');
                if (!plotDiv) return;

                const initializeInteractivity = () => {{
                    const colorscaleMap = {colorscales_json};
                    const scaleArgs = {scale_args_json};
                    const rangeArgs = {range_args_json};
                    
                    const currentState = {{
                        scale: 'linear', range: '180', colorscale: 'nRainbow'
                    }};

                    const updateNavLinks = () => {{
                        const params = new URLSearchParams(currentState).toString();
                        document.querySelectorAll('.residue-btn').forEach(a => {{
                            const baseUrl = a.href.split('#')[0];
                            a.href = baseUrl + '#' + params;
                        }});
                    }};

                    // Step 1: Read URL hash and apply settings
                    const hash = window.location.hash.substring(1);
                    const params = new URLSearchParams(hash);
                    const scale = params.get('scale');
                    const range = params.get('range');
                    const colorscale = params.get('colorscale');

                    if (scale === 'log') {{
                        Plotly.update(plotDiv, scaleArgs[1][0], scaleArgs[1][1]);
                        currentState.scale = 'log';
                    }}
                    if (range === '360') {{
                        Plotly.update(plotDiv, rangeArgs[1][0], rangeArgs[1][1]);
                        currentState.range = '360';
                    }}
                    if (colorscale && colorscaleMap[colorscale]) {{
                       Plotly.restyle(plotDiv, {{'colorscale': [colorscaleMap[colorscale], colorscaleMap[colorscale]]}});
                       currentState.colorscale = colorscale;
                    }}
                    
                    // Step 2: Update nav links immediately
                    updateNavLinks();

                    // Step 3: Attach listeners for new clicks
                    plotDiv.on('plotly_update', (updateData) => {{
                        if (updateData && updateData.scene && updateData.scene.zaxis && updateData.scene.zaxis.type) {{
                            currentState.scale = updateData.scene.zaxis.type;
                        }}
                        if (updateData && updateData.scene && updateData.scene.xaxis && updateData.scene.xaxis.range) {{
                            currentState.range = (updateData.scene.xaxis.range[1] > 200) ? '360' : '180';
                        }}
                        updateNavLinks();
                    }});

                    plotDiv.on('plotly_restyle', (restyleData) => {{
                        try {{
                            const updateObject = restyleData[0];
                            if (updateObject && updateObject.colorscale) {{
                                const newColorscaleData = JSON.stringify(updateObject.colorscale[0]);
                                for (const name in colorscaleMap) {{
                                    if (JSON.stringify(colorscaleMap[name]) === newColorscaleData) {{
                                        currentState.colorscale = name;
                                        break;
                                    }}
                                }}
                            }}
                            updateNavLinks();
                        }} catch(e) {{}}
                    }});
                }};

                plotDiv.once('plotly_afterplot', initializeInteractivity);

            }})();
        </script>
        """
        # --- END JAVASCRIPT CORRECTION ---

        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ramachandran Plot: {self.residue}</title>
            <style>
                body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow-x: hidden; font-family: sans-serif; }}
                .plot-container {{ width: 100vw; height: 80vh; }}
                .controls-container {{ text-align: center; padding: 15px 0; }}
                #plot-title {{ margin-top: 0; margin-bottom: 15px; font-size: 24px; font-weight: bold; }}
                .residue-btn-wrapper {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; max-width: 90%; margin: auto; }}
                .residue-btn {{ padding: 8px 12px; border: 1px solid #ccc; background-color: #f0f0f0; border-radius: 16px; cursor: pointer; font-size: 14px; color: #333; text-decoration: none; transition: background-color 0.2s, border-color 0.2s; }}
                .residue-btn:hover {{ background-color: #e0e0e0; border-color: #bbb; }}
                .residue-btn.active {{ background-color: #d0d0d0; border-color: #999; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="plot-container">{plot_div}</div>
            {nav_bar_html}
            {javascript_code}
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)

def process_residue(residue_name: str, all_residues: Optional[List[str]] = None):
    try:
        visualizer = ResidueVisualizer(residue=residue_name)
        if visualizer.df is not None and not visualizer.df.empty:
            visualizer.save_interactive_html(all_residues=all_residues)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while processing '{residue_name}': {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interactive 3D Ramachandran plots for residues and save them as HTML files.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("target", type=str, help="A residue string (e.g. 'ALA') or a k-value (e.g. '1') for batch processing.")
    args = parser.parse_args()

    if args.target.isdigit():
        k_value = int(args.target)
        kmer_dir = f"data/k{k_value}"
        if not os.path.isdir(kmer_dir):
            print(f"Error: Directory '{kmer_dir}' not found.", file=sys.stderr)
            sys.exit(1)
        
        all_files = sorted([f for f in os.listdir(kmer_dir) if f.lower().endswith('.csv')])
        
        if k_value == 1:
            valid_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            csv_files = [f for f in all_files if os.path.splitext(f)[0] in valid_amino_acids]
        else:
            csv_files = all_files

        if not csv_files:
            print(f"No valid residue .csv files found in '{kmer_dir}'.")
            sys.exit(0)
            
        residue_names_for_nav = [os.path.splitext(f)[0].upper() for f in csv_files]

        print(f"--- Batch Mode: Found {len(csv_files)} residues to process in '{kmer_dir}' ---")
        for filename in tqdm(csv_files, desc="Saving HTML plots"):
            residue_name = os.path.splitext(filename)[0]
            process_residue(residue_name.upper(), all_residues=residue_names_for_nav)
        print(f"--- Batch processing complete. Files saved in 'visualizations/' directory. ---")
    else:
        residue = args.target.upper()
        print(f"--- Single Mode: Generating HTML plot for '{residue}' ---")
        process_residue(residue, all_residues=None)

