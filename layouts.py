# File: layouts.py
from dash import dcc, html
import dash_bootstrap_components as dbc

PLOTLY_COLORSCALES = [
    "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
    "Blues", "Greens", "Reds", "YlOrRd",
    "Jet", "Turbo", "Rainbow",
]

def main_layout():
    """Defines the main layout of the dashboard."""
    
    sidebar = html.Div(
        id="sidebar",
        className="sidebar",
        children=[
            html.H2("Controls"),
            html.Hr(),
            dbc.Label("K-mer Sequence"),
            dbc.Input(id='kmer-input', type='text', placeholder='Enter K-mer...'),
            html.Br(),
            
            dbc.Label("Focal Residue Position"),
            dcc.Dropdown(id='position-dropdown', placeholder='Select Position...'),
            html.Br(),

            dbc.Label("Row Limit (for testing)"),
            dbc.Input(id='limit-input', type='number', value=1000000, min=1000, step=1000),
            html.Br(),

            dbc.Button('Analyze', id='submit-button', n_clicks=0, color="primary", className="w-100"),
            html.Hr(),

            html.H4("Visualization Options"),
            
            dbc.Label("Scale"),
            dbc.Switch(id='scale-switch', label="Logarithmic", value=False),
            html.Br(),

            dbc.Label("Colorscale"),
            dcc.Dropdown(
                id='colormap-dropdown',
                options=[{'label': cs, 'value': cs} for cs in PLOTLY_COLORSCALES],
                value='Viridis'
            ),
        ]
    )

    main_content = html.Div(
        id="main-content",
        className="main-content",
        children=[
            dcc.Store(id='data-store'),
            dcc.Loading(
                id="loading-spinner",
                type="circle",
                children=dcc.Graph(id='ramachandran-plot')
            )
        ]
    )

    return html.Div(className="app-container", children=[sidebar, main_content])