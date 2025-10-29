# Protein Plot Panel

Protein Plot Panel is an interactive Dash dashboard for the dynamic visualization of pre-computed protein backbone and residue invariants. It provides a high-density, multi-panel interface for exploring conformational data, toggling between 3D Ramachandran-style plots, 1D histograms, and detailed statistical summaries.

## 🚀 Features

* **Multi-Panel Grid**: View up to 6 plots or statistical tables simultaneously.
* **Dynamic Panel Content**: Renders three types of content:
    * **3D Heatmaps**: For Torsion-vs-Torsion comparisons.
    * **1D Histograms**: For Torsion-vs-Non-Torsion comparisons.
    * **Statistical Tables**: A comprehensive summary for all comparison types.
* **"Flipper" Interface**: Seamlessly toggle between graph and stats views on any 3D or 1D panel.
* **Context-Aware Downloads**: Download the current view.
    * Graphs are downloaded as interactive `.html` files.
    * Statistics are downloaded as a clean `.csv` file.
* **Intelligent UI**: The configuration panel dynamically hides/shows options.
    * Disables `X vs. X` invariant selection at offset 0.
    * Hides the "Residue 2" dropdown at offset 0.
    * Hides visual options (colormap, axis limits) for stats-only queries.
* **Per-Panel Visuals**: Log/Linear scale and Colormap settings are saved independently for each panel.

## Prerequisites

This application **does not** generate data. It is a visualization tool for a pre-existing database.

* You must have a SQLite database named `proteins_app.db` in the root directory.
* This database must match the "v7" schema, containing the tables:
    * `v7_stats`
    * `v7_3D_cache`
    * `v7_histo_cache`

## ⚙️ Installation

1.  Clone this repository:
    ```bash
    git clone [your-repo-url]
    cd [your-repo-name]
    ```

2.  Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  Install the required dependencies:
    ```bash
    # (Create a requirements.txt file with the following content)
    # pip install -r requirements.txt
    ```
    **`requirements.txt`:**
    ```
    dash
    dash-bootstrap-components
    pandas
    numpy
    plotly
    ```

## 🏃‍♂️ Running the Application

Once your `proteins_app.db` file is in place and dependencies are installed, run the app:

```bash
python app.py