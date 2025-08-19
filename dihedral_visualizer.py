import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.colors as pcolors
from tqdm import tqdm
import sys
import argparse
from typing import List, Optional, Dict
import json

class DihedralVisualizer:
    """
    Loads 2-mer dihedral angle data and generates interactive 3D surface plots
    representing the frequency distribution of angle pairs.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the visualizer with the pre-loaded DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the 2-mer angle data.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("A valid, non-empty DataFrame must be provided.")
        self.df = df

    def _build_figure(self, x_col: str, y_col: str) -> go.Figure:
        """
        Constructs the Plotly figure with interactive controls for a given
        pair of dihedral angle columns.
        """
        
        def _prepare_range_data(use_360: bool):
            """Helper function to bin data for a specific angle range."""
            angle_data = self.df[[x_col, y_col]].dropna().copy()
            
            # Create rounded columns for binning
            angle_data['x_round'] = angle_data[x_col].round().astype(int)
            angle_data['y_round'] = angle_data[y_col].round().astype(int)

            if use_360:
                angle_data['x_binned'] = angle_data['x_round'] % 360
                angle_data['y_binned'] = angle_data['y_round'] % 360
                full_range = np.arange(0, 361)
            else:
                angle_data['x_binned'] = angle_data['x_round']
                angle_data['y_binned'] = angle_data['y_round']
                full_range = np.arange(-180, 181)
            
            freq_counts = angle_data.groupby(['x_binned', 'y_binned']).size().reset_index(name='count')
            z_data = freq_counts.pivot_table(index='y_binned', columns='x_binned', values='count', fill_value=0)
            z_data = z_data.reindex(index=full_range, columns=full_range, fill_value=0)
            
            z_numeric = z_data.values.astype(float)
            z_numeric[z_numeric == 0] = np.nan
            with np.errstate(divide='ignore', invalid='ignore'):
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
        
        fig.update_layout(title=None, scene=dict(xaxis_title=f'{x_col} (degrees)', yaxis_title=f'{y_col} (degrees)', zaxis_title='Frequency Count', xaxis=dict(range=[-180, 180], tickmode='linear', dtick=60), yaxis=dict(range=[-180, 180], tickmode='linear', dtick=60), zaxis_type='linear'), width=None, height=None, margin=dict(l=65, r=50, b=65, t=90), updatemenus=[colorscale_dropdown, scale_buttons, range_buttons])
        return fig
            
    def save_interactive_html(self, plot_title: str, output_path: str, all_plots: List[Dict]):
        """
        Builds the figure and saves it as a self-contained HTML file with navigation.
        """
        x_col, y_col = plot_title.split(' vs ')
        fig = self._build_figure(x_col, y_col)
        if not fig: return

        plot_div_id = "dihedralPlot"
        plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn', default_height="80vh", div_id=plot_div_id)

        nav_bar_html = ""
        if all_plots and len(all_plots) > 1:
            buttons_html = ""
            for plot_info in all_plots:
                active_class = "active" if plot_info['title'] == plot_title else ""
                buttons_html += f'<a href="{plot_info["filename"]}" class="residue-btn {active_class}">{plot_info["title"]}</a>\n'
            
            nav_bar_html = f"""
            <div class="controls-container">
                <h2 id="plot-title">3D Dihedral Plot: '{plot_title}'</h2>
                <div class="residue-btn-wrapper">
                    {buttons_html}
                </div>
            </div>
            """
        
        colorscales_json = json.dumps(self.available_colorscales)
        scale_args_json = json.dumps(self.scale_button_args)
        range_args_json = json.dumps(self.range_button_args)

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
                    
                    updateNavLinks();

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

        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Dihedral Plot: {plot_title}</title>
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

def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive 3D surface plots for 2-mer dihedral angle relationships.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_csv", type=str, help="Path to the 2mer_angles.csv file.")
    parser.add_argument("--output_dir", type=str, default="dihedral_surface_plots", help="Directory to save the output HTML files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(args.input_csv, header=None, names=['phi_i', 'psi_i', 'phi_i+1', 'psi_i+1'], skiprows=1)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_csv}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    visualizer = DihedralVisualizer(df)

    plots_to_generate = [
        {'x': 'phi_i',   'y': 'psi_i',   'title': 'phi_i vs psi_i'},
        {'x': 'phi_i',   'y': 'phi_i+1', 'title': 'phi_i vs phi_i+1'},
        {'x': 'phi_i',   'y': 'psi_i+1', 'title': 'phi_i vs psi_i+1'},
        {'x': 'psi_i',   'y': 'phi_i+1', 'title': 'psi_i vs phi_i+1'},
    ]

    # Add filenames to the list for navigation links
    for plot in plots_to_generate:
        plot['filename'] = f"{plot['x']}_vs_{plot['y']}_surface.html"

    for plot_info in tqdm(plots_to_generate, desc="Generating plots"):
        output_file_path = os.path.join(args.output_dir, plot_info['filename'])
        
        visualizer.save_interactive_html(
            plot_title=plot_info['title'],
            output_path=output_file_path,
            all_plots=plots_to_generate
        )

    print(f"\n--- Batch processing complete. Files saved in '{args.output_dir}/' directory. ---")

if __name__ == "__main__":
    main()
