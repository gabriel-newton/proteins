# File: callbacks.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update
from db_utils import get_angles_for_kmer
from io import StringIO

def register_callbacks(app: Dash, cache):
    """Registers all callbacks for the app."""

    @app.callback(
        Output('data-store', 'data'),
        Output('ramachandran-plot', 'figure', allow_duplicate=True), 
        Input('submit-button', 'n_clicks'),
        State('kmer-input', 'value'),
        State('position-dropdown', 'value'),
        State('limit-input', 'value'),
        prevent_initial_call=True
    )
    @cache.memoize()
    def query_database(n_clicks, kmer, focal_index, limit):
        """
        This is the slow, expensive callback. It runs ONLY when the analyze
        button is clicked. It queries the database and stores the result.
        """
        if not kmer or focal_index is None:
            return no_update, no_update
        
        df = get_angles_for_kmer(kmer, focal_index, limit)
        return df.to_json(date_format='iso', orient='split'), no_update

    @app.callback(
        Output('ramachandran-plot', 'figure'),
        Input('data-store', 'data'),
        Input('scale-switch', 'value'),
        Input('colormap-dropdown', 'value'),
    )
    def update_figure(jsonified_data, use_log_scale, selected_colormap):
        """
        This is the fast callback. It runs whenever the stored data changes OR
        when a visualization control is changed. It does no database work.
        """
        if jsonified_data is None:
            return go.Figure()

        df = pd.read_json(StringIO(jsonified_data), orient='split')
        
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title=f'No data found')
            return fig

        df.dropna(subset=['phi', 'psi'], inplace=True)
        df['phi_int'] = df['phi'].round().astype(int)
        df['psi_int'] = df['psi'].round().astype(int)
        freq_counts = df.groupby(['phi_int', 'psi_int']).size().reset_index(name='count')
        z_data = freq_counts.pivot_table(index='psi_int', columns='phi_int', values='count', fill_value=0)
        
        full_range = np.arange(-180, 181)
        z_data = z_data.reindex(index=full_range, columns=full_range, fill_value=0)
        
        z_data.replace(0, np.nan, inplace=True)
        z_axis_title = "Frequency"
        if use_log_scale:
            z_data = np.log10(z_data)
            z_axis_title = "Log(Frequency)"
        
        fig = go.Figure(data=[go.Surface(
            z=z_data.values, x=z_data.columns, y=z_data.index,
            colorscale=selected_colormap, cmin=0 if not use_log_scale else None,
            colorbar=dict(title=z_axis_title)
        )])
        
        fig.update_layout(
            title=f'Ramachandran Plot',
            scene=dict(
                xaxis_title='Phi (ϕ)', yaxis_title='Psi (ψ)', zaxis_title=z_axis_title,
                xaxis=dict(range=[-180, 180]), yaxis=dict(range=[-180, 180]),
            ),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig

    @app.callback(Output('position-dropdown', 'options'), Input('kmer-input', 'value'))
    def update_position_dropdown(kmer_value):
        if not kmer_value: return []
        return [{'label': f'{i+1}: {r}', 'value': i} for i, r in enumerate(kmer_value)]