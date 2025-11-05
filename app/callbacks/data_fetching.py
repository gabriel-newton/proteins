import json
import sqlite3
import pandas as pd
from dash import Input, Output, State, no_update
import time
import math
from constants import (
    DB_PATH, INVARIANT_SHORTHAND, INVARIANT_ORDER,
    TORSION_INVARIANTS, NON_TORSION_INVARIANTS
)


def get_plot_key_for_query(inv1, inv2, offset, res1, res2):
    """
    Determines the correct plot_key to query the v7 database.
    Handles sorting for offset 0 Torsion/Torsion and Any/Any.
    Adds '_level_1' suffix for potential 3D jobs.
    """
    inv1_type = 'TORSION' if inv1 in TORSION_INVARIANTS else 'NON_TORSION'
    inv2_type = 'TORSION' if inv2 in TORSION_INVARIANTS else 'NON_TORSION'

    is_any_any = (res1 == 'Any' and res2 == 'Any')
    is_torsion_torsion = (inv1_type == 'TORSION' and inv2_type == 'TORSION')

    # Handle offset 0 sorting for Torsion/Torsion or Any/Any
    if (is_any_any or is_torsion_torsion) and offset == 0:
        try:
            idx1 = INVARIANT_ORDER.index(inv1)
            idx2 = INVARIANT_ORDER.index(inv2)
            if idx2 < idx1:
                inv1, inv2 = inv2, inv1 # Swap to match the key in the DB
        except ValueError:
            print(f"Warning: Invariant not found in INVARIANT_ORDER: {inv1} or {inv2}")

    if is_any_any or is_torsion_torsion:
        plot_key = f"{inv1}_vs_{inv2}+{offset}_{res1}_{res2}_level_1"
    else:
        plot_key = f"{inv1}_vs_{inv2}+{offset}_{res1}_{res2}"

    return plot_key

def fetch_v7_data(conn, plot_key):
    """
    Fetches all available data (stats, 3D cache, histo cache)
    for a single plot_key from the v7 DB tables.
    """
    print(f"DEBUG: Querying v7 database with plot_key: '{plot_key}'")

    stats_query = "SELECT * FROM v7_stats WHERE plot_key = ?"
    stats_df = pd.read_sql_query(stats_query, conn, params=(plot_key,))

    if stats_df.empty:
        print(f"DEBUG: No entry found in v7_stats for key '{plot_key}'")
        raise ValueError("No data found for this comparison (Population may be 0).")

    stats_data = stats_df.to_dict('records')[0]
    job_type_v7 = stats_data['job_type']

    all_data = {
        'stats_v7': stats_data,
        'job_type_v7': job_type_v7,
        'figure_data_3d': None,
        'figure_data_histo_x': None,
        'figure_data_histo_y': None
    }

    if job_type_v7 == '3D_VIZ':
        cache_query = "SELECT data FROM v7_3D_cache WHERE plot_key = ?"
        cache_df = pd.read_sql_query(cache_query, conn, params=(plot_key,))
        if not cache_df.empty:
            try:
                parsed_data = json.loads(cache_df.iloc[0]['data'])
                if isinstance(parsed_data, dict):
                    all_data['figure_data_3d'] = parsed_data
                else:
                    print(f"WARNING: Unexpected 3D cache data format for '{plot_key}'")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"ERROR decoding 3D cache JSON for '{plot_key}': {e}")
        else:
            print(f"WARNING: No 3D_cache data found for 3D_VIZ job '{plot_key}'")

    if job_type_v7 == '3D_VIZ' or job_type_v7 == 'STATS_AND_HISTO':
        histo_query = "SELECT axis, data FROM v7_histo_cache WHERE plot_key = ?"
        histo_df = pd.read_sql_query(histo_query, conn, params=(plot_key,))

        for _, row in histo_df.iterrows():
            try:
                parsed_histo = json.loads(row['data'])
                if row['axis'] == 'x':
                    all_data['figure_data_histo_x'] = parsed_histo
                elif row['axis'] == 'y':
                    all_data['figure_data_histo_y'] = parsed_histo
            except (json.JSONDecodeError, TypeError) as e:
                print(f"ERROR decoding Histo cache JSON for '{plot_key}', axis {row['axis']}: {e}")


    return all_data

def _transform_v7_stats_to_v6_1d_stats(stats_v7, axis='x'):
    """ Transforms unified v7 stats into old v6 1D stats format. """
    prefix = axis
    return {
        'population': stats_v7.get('population'),
        'mean': stats_v7.get(f'mean_{prefix}'),
        'variance': stats_v7.get(f'variance_{prefix}'),
        'freq_at_mean': stats_v7.get(f'freq_at_mean_{prefix}'),
        'quartiles': {
            'min': stats_v7.get(f'min_{prefix}'),
            'median': stats_v7.get(f'median_{prefix}'),
            'max': stats_v7.get(f'max_{prefix}')
        } ## Consider 25 75
    }

def _transform_v7_stats_to_v6_3d_overlay_stats(stats_v7):
    """ Transforms unified v7 stats into old v6 3D overlay format. """
    return {
        'population': stats_v7.get('population'),
        'peak_x': stats_v7.get('peak_x'),
        'peak_y': stats_v7.get('peak_y'),
        'peak_freq': stats_v7.get('peak_freq'),
        'mean_x': stats_v7.get('mean_x'),
        'mean_y': stats_v7.get('mean_y'),
        'variance_x': stats_v7.get('variance_x'),
        'variance_y': stats_v7.get('variance_y'),
    }

def register_data_fetching_callbacks(app):
    @app.callback(
        Output('panel-states-store', 'data'),
        Output('status-message-store', 'data', allow_duplicate=True),
        Input('generate-graph-button', 'n_clicks'),
        State('inv1-dropdown', 'value'), State('inv2-dropdown', 'value'),
        State('offset-dropdown', 'value'), State('res1-dropdown', 'value'),
        State('res2-dropdown', 'value'), State('xaxis-min-input', 'value'),
        State('xaxis-max-input', 'value'), State('yaxis-min-input', 'value'),
        State('yaxis-max-input', 'value'),
        State('scale-switch', 'value'),
        State('colormap-dropdown', 'value'),
        State('active-panel-store', 'data'),
        State('panel-states-store', 'data'), prevent_initial_call=True
    )
    def generate_panel_data(n_clicks, inv1, inv2, offset, res1, res2, x_min, x_max, y_min, y_max,
                            scale_bool, colormap, active_panel_index, panel_states_json):
        """ Fetches v7 data, transforms to v6 format, saves to store. """
        panel_states = json.loads(panel_states_json or '{}')

        try:
            plot_key = get_plot_key_for_query(inv1, inv2, offset, res1, res2)

            inv1_label = INVARIANT_SHORTHAND.get(inv1, inv1); inv2_label = INVARIANT_SHORTHAND.get(inv2, inv2);
            res1_label = "" if res1 == 'Any' else f"({res1})"; res2_label = "" if res2 == 'Any' else f"({res2})";
            if offset == 0:
                res2_label = res1_label
            title = f"{inv1_label} {res1_label} vs {inv2_label} {res2_label} +{offset}";

            with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
                fetched_data = fetch_v7_data(conn, plot_key)

            stats_v7 = fetched_data['stats_v7']
            pearson_val = None
            try:
                var_x = stats_v7.get('variance_x')
                var_y = stats_v7.get('variance_y')
                cov = stats_v7.get('covariance')
                
                if all(isinstance(v, (int, float)) for v in [var_x, var_y, cov]):
                    if var_x > 0 and var_y > 0:
                        std_dev_product = math.sqrt(var_x * var_y)
                        if std_dev_product != 0:
                            pearson_val = cov / std_dev_product
            except (TypeError, ValueError, ZeroDivisionError) as e:
                print(f"Warning: Could not calculate Pearson's correlation: {e}")
            
            fetched_data['stats_v7']['pearson_correlation'] = pearson_val

            new_panel_state = {
                'title': title, 'inv1': inv1, 'inv2': inv2, 'offset': offset,
                'res1': res1, 'res2': res2, 'x_lims': [x_min, x_max],
                'y_lims': [y_min, y_max], 'uirevision_key': str(time.time()),
                'full_v7_stats': fetched_data['stats_v7'],
                'log_scale': scale_bool,
                'colormap': colormap,
            }

            v7_job_type = fetched_data['job_type_v7']

            if v7_job_type == '3D_VIZ':
                new_panel_state['job_type'] = '3D_HEATMAP'
                new_panel_state['figure_data'] = fetched_data['figure_data_3d']
                new_panel_state['stats'] = _transform_v7_stats_to_v6_3d_overlay_stats(stats_v7)
                new_panel_state['view'] = 'graph'

            elif v7_job_type == 'STATS_AND_HISTO':
                inv1_is_torsion = inv1 in TORSION_INVARIANTS
                histo_data = fetched_data['figure_data_histo_x'] if inv1_is_torsion else fetched_data['figure_data_histo_y']
                
                if inv1_is_torsion:
                    new_panel_state['job_type'] = '1D_HISTO_VS_STATS'
                    new_panel_state['figure_data_histo'] = histo_data
                    new_panel_state['figure_data_stats'] = _transform_v7_stats_to_v6_1d_stats(stats_v7, axis='y')
                else:
                    new_panel_state['job_type'] = '1D_STATS_VS_HISTO'
                    new_panel_state['figure_data_stats'] = _transform_v7_stats_to_v6_1d_stats(stats_v7, axis='x')
                    new_panel_state['figure_data_histo'] = histo_data
                
                new_panel_state['view'] = 'stats'

            elif v7_job_type == 'STATS_ONLY':
                new_panel_state['job_type'] = '1D_STATS_VS_STATS'
                new_panel_state['figure_data_stats1'] = _transform_v7_stats_to_v6_1d_stats(stats_v7, axis='x')
                new_panel_state['figure_data_stats2'] = _transform_v7_stats_to_v6_1d_stats(stats_v7, axis='y')

            panel_states[str(active_panel_index)] = new_panel_state
            return json.dumps(panel_states), f"Panel {active_panel_index + 1} updating..."

        except ValueError as e:
            error_state = {'error': str(e), 'title': 'No Data'}
            panel_states[str(active_panel_index)] = error_state
            print(f"INFO during generate_panel_data: {e}")
            return json.dumps(panel_states), f"No data found for this combination."
        except Exception as e:
            error_state = {'error': str(e), 'title': 'Error'}
            panel_states[str(active_panel_index)] = error_state
            print(f"ERROR during generate_panel_data: {e}")
            import traceback
            traceback.print_exc()
            return json.dumps(panel_states), f"Error: {e}"