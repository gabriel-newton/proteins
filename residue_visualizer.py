import pandas as pd
import sqlite3
import numpy as np
import os
import plotly.graph_objects as go
import plotly.colors as pcolors
from tqdm import tqdm
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional

class ResidueVisualizer:
    """
    A class to load residue data from a SQLite database and generate
    interactive 3D visualizations of Ramachandran plots with a custom HTML interface.
    """
    def __init__(self, residue: str, db_connection):
        self.residue = residue
        self.df = None
        
        try:
            query = "SELECT tau_NA, tau_AC FROM invariants WHERE residue = ?"
            self.df = pd.read_sql_query(query, db_connection, params=(self.residue,))
            if self.df.empty:
                print(f"Warning: No data found for residue '{self.residue}' in the database.")
        except Exception as e:
            print(f"Error reading data for '{self.residue}' from database: {e}", file=sys.stderr)
            self.df = pd.DataFrame()

    def _prepare_range_data(self, use_360: bool):
        angle_data = self.df[['tau_NA', 'tau_AC']].dropna().copy()
        if use_360:
            angle_data['phi'] = (angle_data['tau_NA'].round() % 360).astype(int)
            angle_data['psi'] = (angle_data['tau_AC'].round() % 360).astype(int)
            full_range = np.arange(0, 361)
        else:
            angle_data['phi'] = angle_data['tau_NA'].round().astype(int)
            angle_data['psi'] = angle_data['tau_AC'].round().astype(int)
            full_range = np.arange(-180, 181)
        
        freq_counts = angle_data.groupby(['phi', 'psi']).size().reset_index(name='count')
        z_data = freq_counts.pivot_table(index='psi', columns='phi', values='count', fill_value=0)
        z_data = z_data.reindex(index=full_range, columns=full_range, fill_value=0)
        
        z_numeric = z_data.values.astype(float)
        z_numeric[z_numeric == 0] = np.nan
        with np.errstate(invalid='ignore'):
            log_z = np.log10(z_numeric)
        return z_data.columns, z_data.index, z_numeric, log_z

    def _build_figure(self) -> Optional[go.Figure]:
        if self.df is None or self.df.empty: return None

        def reverse_colorscale(colorscale_name):
            original = pcolors.get_colorscale(colorscale_name)
            return [[1 - stop, color] for stop, color in reversed(original)]

        x_180, y_180, z_180, log_z_180 = self._prepare_range_data(use_360=False)
        x_360, y_360, z_360, log_z_360 = self._prepare_range_data(use_360=True)

        nRainbow = [[0.0,'rgb(0,0,200)'],[0.125,'rgb(0,25,255)'],[0.25,'rgb(0,152,255)'],[0.375,'rgb(44,255,150)'],[0.5,'rgb(151,255,0)'],[0.625,'rgb(255,234,0)'],[0.75,'rgb(255,111,0)'],[0.875,'rgb(255,0,0)'],[1.0,'rgb(0,0,0)']]
        
        self.available_colorscales = {'nRainbow': nRainbow, 'Magma': pcolors.get_colorscale('Magma'), 'Magma_Reverse': reverse_colorscale('Magma'), 'Viridis': pcolors.get_colorscale('Viridis')}
        
        fig = go.Figure()
        # Create only two traces, one for each range. Their surfacecolor will be updated by the buttons.
        fig.add_trace(go.Surface(z=z_180, x=x_180, y=y_180, surfacecolor=z_180, colorscale=nRainbow, cmin=1, colorbar=dict(title='Count'), visible=True))
        fig.add_trace(go.Surface(z=z_360, x=x_360, y=y_360, surfacecolor=z_360, colorscale=nRainbow, cmin=1, colorbar=dict(title='Count'), visible=False))
        
        # Convert numpy arrays to lists for JSON serialization in the HTML
        z_180_list = np.where(np.isfinite(z_180), z_180, None).tolist()
        log_z_180_list = np.where(np.isfinite(log_z_180), log_z_180, None).tolist()
        z_360_list = np.where(np.isfinite(z_360), z_360, None).tolist()
        log_z_360_list = np.where(np.isfinite(log_z_360), log_z_360, None).tolist()
        
        colorscale_buttons = [dict(label=name.replace('_', ' '), method="restyle", args=[{"colorscale": scale}]) for name, scale in self.available_colorscales.items()]
        
        fig.update_layout(
            title=None, # The title is now in the custom HTML
            scene=dict(xaxis_title='Phi (φ) / tau_NA [degrees]', yaxis_title='Psi (ψ) / tau_AC [degrees]', zaxis_title='Frequency Count', xaxis=dict(range=[-180, 180]), yaxis=dict(range=[-180, 180])),
            margin=dict(l=65, r=50, b=65, t=90),
            updatemenus=[
                dict(buttons=colorscale_buttons, direction="down", showactive=True, x=0.05, xanchor="left", y=1.1, yanchor="top"),
                dict(type="buttons", direction="left", showactive=True, x=0.35, xanchor="left", y=1.1, yanchor="top",
                     buttons=[dict(label="Linear Scale", method="update", args=[{"surfacecolor": [z_180_list, z_360_list]}, {"scene.zaxis.type": "linear"}]),
                              dict(label="Log Scale", method="update", args=[{"surfacecolor": [log_z_180_list, log_z_360_list]}, {"scene.zaxis.type": "log"}])]),
                dict(type="buttons", direction="left", showactive=True, x=0.65, xanchor="left", y=1.1, yanchor="top",
                     buttons=[dict(label="-180° to 180°", method="update", args=[{"visible": [True, False]}, {"scene.xaxis.range": [-180, 180], "scene.yaxis.range": [-180, 180]}]),
                              dict(label="0° to 360°", method="update", args=[{"visible": [False, True]}, {"scene.xaxis.range": [0, 360], "scene.yaxis.range": [0, 360]}])])
            ])
        return fig
            
    def save_interactive_html(self, output_dir: str, all_residues: Optional[List[str]] = None):
        os.makedirs(output_dir, exist_ok=True)
        fig = self._build_figure()
        if not fig: return

        plot_div_id = "ramachandranPlot"
        output_path = Path(output_dir) / f"{self.residue}_ramachandran.html"
        plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn', default_height="80vh", div_id=plot_div_id)

        nav_bar_html = f"""<div class="controls-container">
            <h2 id="plot-title">3D Ramachandran Plot for Residue: '{self.residue}'</h2>
        </div>"""
        if all_residues and len(all_residues) > 1:
            buttons_html = "".join([f'<a href="{r}_ramachandran.html" class="residue-btn {"active" if r == self.residue else ""}">{r}</a>\n' for r in sorted(all_residues)])
            nav_bar_html = f"""<div class="controls-container">
                <h2 id="plot-title">3D Ramachandran Plot for Residue: '{self.residue}'</h2>
                <div class="residue-btn-wrapper">{buttons_html}</div>
            </div>"""

        colorscales_json = json.dumps(self.available_colorscales)
        javascript_code = f"""<script>
            (function() {{
                const plotDiv = document.getElementById('{plot_div_id}');
                if (!plotDiv) return;

                const initializeInteractivity = () => {{
                    const colorscaleMap = {colorscales_json};
                    const currentState = {{ scale: 'linear', range: '180', colorscale: 'nRainbow' }};

                    const updateNavLinks = () => {{
                        const params = new URLSearchParams(currentState).toString();
                        document.querySelectorAll('.residue-btn').forEach(a => {{
                            const baseUrl = a.href.split('#')[0];
                            a.href = baseUrl + '#' + params;
                        }});
                    }};

                    const hash = window.location.hash.substring(1);
                    if (hash) {{
                        const params = new URLSearchParams(hash);
                        // Find the Plotly update menus from the figure layout
                        const menus = plotDiv.layout.updatemenus;
                        if (params.get('scale') === 'log') {{ Plotly.update(plotDiv, menus[1].buttons[1].args[0], menus[1].buttons[1].args[1]); }}
                        if (params.get('range') === '360') {{ Plotly.update(plotDiv, menus[2].buttons[1].args[0], menus[2].buttons[1].args[1]); }}
                        const cs = params.get('colorscale');
                        if (cs && colorscaleMap[cs]) {{ Plotly.restyle(plotDiv, {{'colorscale': colorscaleMap[cs]}}); }}
                    }}

                    plotDiv.on('plotly_update', (updateData) => {{
                        if(updateData.layout.scene) {{
                            const scene = updateData.layout.scene;
                            if (scene.xaxis && scene.xaxis.range) {{ currentState.range = (scene.xaxis.range[1] > 200) ? '360' : '180'; }}
                            if (scene.zaxis && scene.zaxis.type) {{ currentState.scale = scene.zaxis.type; }}
                        }}
                        updateNavLinks();
                    }});

                    plotDiv.on('plotly_restyle', (restyleData) => {{
                        try {{
                            if (restyleData[0].colorscale) {{
                                const newCs = JSON.stringify(restyleData[0].colorscale[0]);
                                for (const name in colorscaleMap) {{
                                    if (JSON.stringify(colorscaleMap[name]) === newCs) {{
                                        currentState.colorscale = name;
                                        break;
                                    }}
                                }}
                            }}
                        }} catch(e) {{}}
                        updateNavLinks();
                    }});
                    updateNavLinks();
                }};
                plotDiv.once('plotly_afterplot', initializeInteractivity);
            }})();
        </script>"""
        
        html_template = f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ramachandran Plot: {self.residue}</title>
            <style>
                body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow-x: hidden; font-family: sans-serif; background-color: #f8f9fa;}}
                .plot-container {{ width: 100vw; }}
                .controls-container {{ text-align: center; padding: 15px 0; background-color: #fff; border-bottom: 1px solid #dee2e6; }}
                #plot-title {{ margin-top: 0; margin-bottom: 15px; font-size: 24px; font-weight: bold; color: #343a40; }}
                .residue-btn-wrapper {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; max-width: 90%; margin: auto; }}
                .residue-btn {{ padding: 8px 12px; border: 1px solid #ccc; background-color: #f0f0f0; border-radius: 16px; cursor: pointer; font-size: 14px; color: #333; text-decoration: none; transition: all 0.2s; }}
                .residue-btn:hover {{ background-color: #e0e0e0; border-color: #bbb; transform: translateY(-2px); }}
                .residue-btn.active {{ background-color: #007bff; border-color: #007bff; color: white; font-weight: bold; }}
            </style>
        </head>
        <body>
            {nav_bar_html}
            <div class="plot-container">{plot_div}</div>
            {javascript_code}
        </body>
        </html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)

def main():
    parser = argparse.ArgumentParser(description="Generate 3D Ramachandran plots from a SQLite database.")
    parser.add_argument("db_path", type=str, help="Path to the protein_geometry_invariants.db SQLite file.")
    parser.add_-oargument("target", type=str, help="A specific residue (e.g., 'ALA') or a k-value (e.g., '1' or '2') for batch processing.")
    parser.add_argument("-o", "--output", type=str, default="visualizations", help="Directory to save HTML files.")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.is_file():
        print(f"Error: Database file not found at '{db_path}'", file=sys.stderr); sys.exit(1)

    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) # Open in read-only mode
        
        if args.target.isdigit():
            k_value = int(args.target)
            print(f"--- Batch Mode: Processing all residues of length {k_value} ---")
            
            residues_df = pd.read_sql_query("SELECT DISTINCT residue FROM invariants WHERE LENGTH(residue) = ?", conn, params=(k_value,))
            residue_names = sorted(residues_df['residue'].tolist())

            if not residue_names: print(f"No residues of length {k_value} found."); return
            
            print(f"Found {len(residue_names)} residues to process.")
            for res_name in tqdm(residue_names, desc="Saving HTML plots"):
                visualizer = ResidueVisualizer(res_name, conn)
                visualizer.save_interactive_html(args.output, all_residues=residue_names)
        else:
            residue = args.target.upper()
            print(f"--- Single Mode: Generating plot for '{residue}' ---")
            visualizer = ResidueVisualizer(residue, conn)
            # In single mode, still generate nav bar for that k-mer length
            residues_df = pd.read_sql_query("SELECT DISTINCT residue FROM invariants WHERE LENGTH(residue) = ?", conn, params=(len(residue),))
            all_residues = sorted(residues_df['residue'].tolist())
            visualizer.save_interactive_html(args.output, all_residues=all_residues)
            
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        if conn: conn.close(); print("Database connection closed.")

if __name__ == "__main__":
    main()