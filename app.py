# File: app.py
from dash import Dash
import dash_bootstrap_components as dbc
from flask_caching import Cache
import layouts
import callbacks

# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Configure Caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

# Set the app layout and register callbacks
app.layout = layouts.main_layout()
callbacks.register_callbacks(app, cache) # Pass the cache object to callbacks

# Run the app
if __name__ == '__main__':
    app.run(debug=True)