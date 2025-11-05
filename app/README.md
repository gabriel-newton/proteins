# OriViz Release v0.7

First testing release of the Protein Offset-Residue Invariant Visualization Dashboard (OriViz).
## How to set up

### 1. Install Python 3.10 or newer
* Download from [python.org/downloads](https://www.python.org/downloads/)
* During installation (Windows): check “Add Python to PATH”.

### 2. Download application Code

* **`OriViz-v0.7.zip`** - The front-end Dash application (found below).
    *Un-zip to desired location (e.g. `Downloads/OriViz/`)*

### 3. Required Database (Hugging Face)

* **[Download `proteins_app.db` from my Hugging Face datasets](https://huggingface.co/datasets/gabriel-newton/proteins/resolve/main/proteins_app.db?download=true)**  `3.11 GB`
    *Place this file in the same directory as `app.py` (e.g. inside `Downloads/OriViz/`)*

## How to Run

### 1. Run `run.py`
* Navigate to the application folder.
* Double-click `run.py` to run it with Python, or open a terminal in that folder and type `python run.py`.
* The script will first install all required Python packages (see list below).
* It will then start the application server on `http://127.0.0.1:8050`.

### 2. Access the Application
* Open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your preferred web browser.

### 3. Stop the Application
* To quit, go to the terminal where `run.py` is running and press `CTRL+C`.

## How to Use

The side panel controls the query. The main window shows the output.

### 1. Panel Selection
* Select a desired panel to configure (default: Panel 1).

### 2. Offset Selection
* Select a backbone offset `i` vs `i+n` (default: 0).
* This choice changes the "Residue Context Filtering" options below.

### 3. Invariant Parameter Selection
* Select 2 invariant parameters to compare (default: $\phi$ vs $\psi$).

### 4. Residue Context Filtering
* **If Offset = 0:** One "Residue" drop-down appears. This filters all data to a single residue type (default: Any).
* **If Offset > 0:** Two dropdowns ("Residue 1" and "Residue 2") appear. This query is **directional**. (default: Any vs Any)
    * `Residue 1` refers to the residue at position `i`.
    * `Residue 2` refers to the residue at position `i+n`.
    * *Example:* Offset = 2, Res 1 = A, Res 2 = P. This *only* shows data for Alanine (A) followed by Proline (P) two steps later.

### 5. Analyze Visual Output
* By default, "Any vs Any" residue comparisons produce a **3D Heatmap**.
* Otherwise, the output will change based on your invariant selection:
    * **3D Heatmap & Stats Panel:** Renders if two torsion angles ($\phi$, $\psi$, $\omega$) are selected.
    * **Stats Panel & 1D Torsion Histogram:** Renders if only one torsion invariant is selected (e.g., $\phi$ vs $\alpha(N)$).
    * **Stats Panel:** Renders if any other combination is selected (e.g., $\alpha(N)$ vs $L(N)$).

### 6. Toggle and Download
- Stats/Graph toggle button is the first button available unless only a Stats Panel is produced. This button switched between the available graph and stats views.
* Focus button opens a full screen modal of whatever is displayed on the relevant panel.
* The Download button is context-aware and downloads interactive HTML files when a graph is being viewed, otherwise (i.e. when statistics are being viewed) a CSV of the statistics table is downloaded instead.
* The Config (cog) button focuses the left configuration section on the appropriate panel.
* The Clear button removes the given visualization (confirmation required).

## Dependencies (for information only)
The `run.py` script will automatically install these packages for you:
* `dash`
* `dash-bootstrap-components`
* `pandas`
* `numpy`
* `plotly`
