"""Patch preprocess_ugmrt.ipynb: add interactive plot support and save examples."""
import json, shutil, pathlib

NB = pathlib.Path('preprocess_ugmrt.ipynb')
shutil.copy(NB, NB.with_suffix('.ipynb.bak2'))

nb = json.load(open(NB))
ids = {c['id']: i for i, c in enumerate(nb['cells'])}

def set_src(cell_id, new_source):
    idx = ids[cell_id]
    lines = new_source.splitlines(keepends=True)
    # ensure last line has no trailing newline added by splitlines
    nb['cells'][idx]['source'] = lines
    print(f'  patched {cell_id} (cell {idx+1})')

# ── Cell 3 [9fa21fcc] — Setup code ─────────────────────────────────────────
set_src('9fa21fcc', """\
# Choose plotting backend BEFORE importing ugmrt_query.
#   %matplotlib inline   → static PNG embedded in notebook output (default)
#   %matplotlib widget   → interactive pan/zoom/resize  (requires: pip install ipympl)
#
# Switch interactivity at any time by re-running this cell with the other magic.
# Important: after switching, kernel-restart is NOT needed — just re-run affected plot cells.
%matplotlib inline
# %matplotlib widget     # ← uncomment for interactive plots

import importlib
import sys
import numpy as np
from pathlib import Path

if 'ugmrt_query' in sys.modules:
    importlib.reload(sys.modules['ugmrt_query'])
import ugmrt_query as q

print('ugmrt_query loaded')
""")

# ── Cell 6 [d079bc05] — §3 Source Observability markdown ──────────────────
set_src('d079bc05', """\
## 3 · Source Observability

`query_source` extracts every observable-quality property from the FITS file:
elevation/azimuth track, hour angle, parallactic angle, UV coverage  
statistics, sensitivity estimate, and data-selection advice.

```python
result = q.query_source(fits_path_or_index, source, azel_time_step_s=60.0)
```

- Pass the FITS path or a pre-built row index (avoids re-opening the file).
- `print_source_query(result)` prints a human-readable summary.
- `plot_source_query(result)` produces a four-panel diagnostic figure:  
  top-left El/Az, top-right UV plane, bottom-left HA/PA, bottom-right baseline histogram.

**Interactive / save:**
```python
fig = q.plot_source_query(result, figsize=(18, 14))
fig.tight_layout()
fig.savefig(WORK_DIR / '3c48_source_query.png', dpi=150, bbox_inches='tight')
```
Switch to `%matplotlib widget` in the Setup cell for interactive pan/zoom.

**Key outputs used downstream:**

| Key | Use |
|-----|-----|
| `result['data_selection_advice']['recommended_elevation_min_deg']` | Set `SOLVE_ELEVATION_MIN_DEG` |
| `result['uv_coverage']['b_max_klambda']` | Upper limit for `uvrange_klambda` in `DataSelection` |
| `result['timing']['scans']` | Identify scan boundaries for time-range cuts |
""")

# ── Cell 7 [33b48ce6] — §3 Source Observability code ──────────────────────
set_src('33b48ce6', """\
# ── Known dud antennas for this FITS file ────────────────────────────────────
# C07 (antenna_no=31) and S05 (antenna_no=32) have ghost STABXYZ entries in
# the AIPS AN table: their positions are identical to W06 (~1 m apart).
# They are non-functional and must be excluded from all counts and solves.
# Keep this list in sync with DUD_ANTENNA_NAMES in the §5 configuration cell.
DUD_ANTENNA_NAMES = ['C07', 'S05']

# ── Build row index once (reused for source query and all downstream steps) ──
# Adjust SOURCE_QUERY to the calibrator you want to inspect.
SOURCE_QUERY = '3C48'   # change to '3C286', 'SRCNAME', etc.

# Use CAL_FITS from the cell above, or re-define it here.
result = q.query_source(
    CAL_FITS,
    SOURCE_QUERY,
    azel_time_step_s=60.0,
    override_dud_names=DUD_ANTENNA_NAMES,
)

# ── Printed summary ───────────────────────────────────────────────────────────
q.print_source_query(result)

# ── Diagnostic plot (El/Az, UV, HA/PA, baseline histogram) ───────────────────
fig = q.plot_source_query(result, figsize=(18, 14))
fig.tight_layout()
# fig.savefig(WORK_DIR / f'{SOURCE_QUERY.lower()}_source_query.png', dpi=150, bbox_inches='tight')
""")

# ── Cell 8 [906f0dbb] — §4 Array Layout markdown ──────────────────────────
set_src('906f0dbb', """\
## 4 · Array Layout

`plot_antenna_positions` draws the GMRT array from the ENU coordinates in the AIPS AN table.

```python
fig = q.plot_antenna_positions(
    fits_path_or_index,           # FITS path or row index
    flagged_antenna_names=['C11'],# highlight flagged antennas in red
    highlight_pairs_closer_than_m=100.0,  # draw lines + standing-wave annotation
)
```

**Interactive / save:**
```python
fig.savefig(WORK_DIR / 'gmrt_array.png', dpi=150, bbox_inches='tight')
```
Switch to `%matplotlib widget` in the Setup cell for interactive pan/zoom.

Useful to visually confirm which antennas are short-spaced (susceptible to standing waves)  
and which have already been flagged in a flag table.
""")

# ── Cell 9 [8a2d7bca] — §4 Array Layout code ──────────────────────────────
set_src('8a2d7bca', """\
# Optionally mark currently flagged antennas from the session flag table.
import json as _json

_flagged_names = []
_flag_table_path = Path('/Users/raj030/DATA/gmrt_40_014/work/3c48_flag_table_session.json')
if _flag_table_path.exists():
    _ft = _json.loads(_flag_table_path.read_text())
    _flagged_names = list(_ft.get('bad_antennas', {}).keys())
    print('Flagged antennas from session table:', _flagged_names)

fig = q.plot_antenna_positions(
    CAL_FITS,
    flagged_antenna_names=_flagged_names,
    highlight_pairs_closer_than_m=200.0,
    title=f'GMRT array — {CAL_FITS.name}',
)
# fig.savefig(WORK_DIR / 'gmrt_array.png', dpi=150, bbox_inches='tight')
""")

# ── Cell 18 [50e963a6] — §9 Per-antenna Gain Grid markdown ────────────────
set_src('50e963a6', """\
## 9 · Per-antenna Gain Grid

`plot_bandpass_solution_grid` renders every antenna on a rows × cols page.  
Each antenna cell shows:
- **Top panel** — amplitude $|G_{a,\\nu}|$ (raw + smoothed)
- **Bottom panel** — phase $\\angle G_{a,\\nu}$ in degrees (raw + smoothed)

The reference antenna is pinned to phase = 0° across all channels (post-solve gauge choice).

```python
fig = q.plot_bandpass_solution_grid(
    solution,
    rows=6, cols=5,
    figsize=(32, 40),
    skip_edge_channels=(5, 5),     # ignore edge artefacts in auto-scaling
    phase_ylim=(-200, 200),
    canonical_antenna_names=sorted_names,  # fix panel order across iterations
    save_path=WORK_DIR / 'bandpass_grid.png',   # save directly inside the function
)
```

**Interactive / save — two equivalent approaches:**

*Option A — pass `save_path` directly (saves without displaying):*
```python
fig = q.plot_bandpass_solution_grid(..., save_path=WORK_DIR / 'grid.png')
```

*Option B — capture `fig` and call `savefig` (also works with `%matplotlib widget`):*
```python
fig = q.plot_bandpass_solution_grid(...)   # display in notebook
fig.savefig(WORK_DIR / 'grid.png', dpi=100, bbox_inches='tight')
```

Switch to `%matplotlib widget` in the Setup cell for interactive pan/zoom on the full grid.  
The grid is large (32×40 inches); `%matplotlib widget` lets you zoom into individual panels.
""")

# ── Cell 19 [cb96b70c] — §9 Per-antenna Gain Grid code ───────────────────
set_src('cb96b70c', """\
_src = SOURCE.lower()
BANDPASS_GRID_PATH = WORK_DIR / f'{_src}_bandpass_grid_{ITER_TAG}.png'

fig = q.plot_bandpass_solution_grid(
    bandpass_sol,
    rows  = 6,
    cols  = 5,
    figsize = (32, 40),
    skip_edge_channels = (5, 5),
    phase_ylim = (-200.0, 200.0),
    title = f'{SOURCE} bandpass solution | {ITER_TAG}',
    # save_path = BANDPASS_GRID_PATH,  # ← uncomment to save without displaying
)

# Save after display (works with both %matplotlib inline and %matplotlib widget):
# fig.savefig(BANDPASS_GRID_PATH, dpi=100, bbox_inches='tight')
# print('Saved:', BANDPASS_GRID_PATH)
""")

# ── Cell 20 [1b295490] — §10 Diagnostics markdown ────────────────────────
set_src('1b295490', """\
## 10 · Diagnostics

`run_bandpass_diagnostics` applies the bandpass solution to the loaded visibilities  
and produces the multi-page diagnostic PDF / PNG:

- **Page 1** — corrected amp vs UV distance for each antenna (top N worst highlighted)
- **Page 2** — baselines ranked by outlier score
- **Page 3** — per-channel residuals vs model flux

Key outputs in the returned `diag` dict:

| Key | Description |
|-----|-------------|
| `diag['antenna_scores']` | Dict `{antenna_name: score_jy}` for the active metrics |
| `diag['baseline_scores']` | Dict `{(ant1, ant2): score_jy}` |
| `diag['freqs_hz']` / `diag['chan_mask']` | Frequency axis and valid-channel mask |
| `diag['plot_freq_min_mhz']` / `diag['plot_freq_max_mhz']` | Plotted frequency range |

**Save:** pass `save_path=DIAG_PLOT_BASE` (already set in the code cell below) to write the  
multi-panel figure directly to disk.  The figure is still displayed inline.  
Switch to `%matplotlib widget` in the Setup cell for interactive pan/zoom.
""")

# ── Cell 24 [21d6d805] — §12 Raw Visibility Inspection markdown ───────────
set_src('21d6d805', """\
## 12 · Raw Visibility Inspection

`load_vis_for_source` loads a block of raw visibilities into memory as a structured dict.  
Pass a `DataSelection` to apply time/UV/elevation cuts before loading.

Three complementary views:

| Function | X axis | Notes |
|----------|--------|-------|
| `plot_vis_amp_vs_uvdist(vis)` | UV distance (kλ) | Shows amplitude envelope → source structure |
| `plot_vis_amp_vs_time(vis)` | Time from start (min) | Shows time-variable RFI or elevation effects |
| `plot_vis_amp_vs_channel(vis)` | Frequency (MHz) | Shows channel-specific RFI or roll-off |

Set `show_phase=True` on any of the above to add a phase panel below the amplitude panel.  
Use `amp_ylim=(0, X)` to cap the amplitude axis for zooming into the noise floor.

**Interactive / save:**  
Each function returns a `fig` object — capture it and call `fig.savefig(...)`, or switch to  
`%matplotlib widget` in the Setup cell for interactive pan/zoom.

```python
fig_uv  = q.plot_vis_amp_vs_uvdist(vis_raw, ...)
fig_t   = q.plot_vis_amp_vs_time(vis_raw, ...)
fig_ch  = q.plot_vis_amp_vs_channel(vis_raw, ...)
fig_uv.savefig(WORK_DIR / 'raw_vis_uvdist.png',   dpi=150, bbox_inches='tight')
fig_t.savefig(WORK_DIR  / 'raw_vis_time.png',     dpi=150, bbox_inches='tight')
fig_ch.savefig(WORK_DIR / 'raw_vis_channel.png',  dpi=150, bbox_inches='tight')
```
""")

# ── Cell 25 [94c4157d] — §12 Raw Visibility code ─────────────────────────
set_src('94c4157d', """\
# Load raw visibilities (pre-bandpass, pre-correction)
vis_raw = q.load_vis_for_source(
    row_index,
    source    = SOURCE,
    chan_range = CHAN_RANGE,
    stokes    = STOKES,
    max_rows  = 60_000,
    flag_all_corrs_if_any_rawvis_flagged = FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
)

print(f'Loaded: {vis_raw["amp"].shape[0]} rows × {vis_raw["amp"].shape[1]} channels × {vis_raw["amp"].shape[2]} pols')
print(f'Freq range: {vis_raw["freqs_hz"].min()/1e6:.3f} – {vis_raw["freqs_hz"].max()/1e6:.3f} MHz')

_src = SOURCE.lower()

# ── Amplitude vs UV distance ──────────────────────────────────────────────────
fig_uv = q.plot_vis_amp_vs_uvdist(
    vis_raw,
    title     = f'{SOURCE} raw vis — amp vs UV distance | {ITER_TAG}',
    show_phase = False,
    alpha     = 0.10,
)
# fig_uv.savefig(WORK_DIR / f'{_src}_raw_vis_uvdist_{ITER_TAG}.png', dpi=150, bbox_inches='tight')

# ── Amplitude vs time ─────────────────────────────────────────────────────────
fig_t = q.plot_vis_amp_vs_time(
    vis_raw,
    title     = f'{SOURCE} raw vis — amp vs time | {ITER_TAG}',
    show_phase = False,
    alpha     = 0.10,
)
# fig_t.savefig(WORK_DIR / f'{_src}_raw_vis_time_{ITER_TAG}.png', dpi=150, bbox_inches='tight')

# ── Amplitude vs channel ──────────────────────────────────────────────────────
fig_ch = q.plot_vis_amp_vs_channel(
    vis_raw,
    title     = f'{SOURCE} raw vis — amp vs channel | {ITER_TAG}',
    show_phase = False,
    alpha     = 0.10,
)
# fig_ch.savefig(WORK_DIR / f'{_src}_raw_vis_channel_{ITER_TAG}.png', dpi=150, bbox_inches='tight')
""")

# ── Cell 26 [cd26de3a] — §13 Corrected Visibility markdown ────────────────
set_src('cd26de3a', """\
## 13 · Corrected Visibility Inspection

After a bandpass solve, apply the solution in memory and inspect the corrected visibilities.

```python
corrected = q.apply_bandpass_solution(vis_raw, bandpass_sol)
# corrected has: 'vis_complex_corrected', 'amp_corrected', 'phase_deg_corrected', …
```

Two convenience plot wrappers:

| Function | What it shows |
|----------|---------------|
| `plot_bandpass_corrected_vis_amp_vs_uvdist(vis, sol)` | Corrected amplitudes vs UV distance → should be flat for an unresolved calibrator |
| `plot_corrected_vector_avg_spectrum(vis, sol)` | Vector-averaged Re(V) vs Perley-Butler 2017 flux model + residuals → residuals should be near zero for a clean calibrator |

A flat UV-distance plot and near-zero residuals mean the bandpass is correct  
to within noise. Structured residuals point to a bad antenna or RFI.

**Interactive / save:**  
Both functions return a `fig` — capture it and call `fig.savefig(...)`.  
`plot_corrected_vector_avg_spectrum` also accepts `save_path=` directly  
(already wired when `DRY_RUN_BANDPASS=False` in the code cell below).  
Switch to `%matplotlib widget` in the Setup cell for interactive pan/zoom.
""")

# ── Cell 27 [025597df] — §13 Corrected Visibility code ────────────────────
set_src('025597df', """\
_src = SOURCE.lower()
GAIN_PLOT_BASE    = WORK_DIR / f'{_src}_corrected_spectrum_{ITER_TAG}.png'
CORRECTED_UV_PATH = WORK_DIR / f'{_src}_corrected_uvdist_{ITER_TAG}.png'

# Corrected amplitude vs UV distance
fig_cuv = q.plot_bandpass_corrected_vis_amp_vs_uvdist(
    vis_raw,
    bandpass_sol,
    title          = f'{SOURCE} corrected vis — amp vs UV dist | {ITER_TAG}',
    exclude_antennas = EXCLUDE_FOR_PLOTS,
    show_phase     = False,
    alpha          = 0.10,
)
# fig_cuv.savefig(CORRECTED_UV_PATH, dpi=150, bbox_inches='tight')

# Vector-averaged spectrum vs Perley-Butler 2017 model
# save_path is written only when DRY_RUN_BANDPASS=False (controls disk commit).
fig_sp = q.plot_corrected_vector_avg_spectrum(
    vis_raw,
    bandpass_sol,
    title            = f'{SOURCE} corrected vector-avg spectrum | {ITER_TAG}',
    exclude_antennas = EXCLUDE_FOR_PLOTS,
    skip_edge_channels = SKIP_EDGE_CHANNELS,
    save_path        = GAIN_PLOT_BASE if not DRY_RUN_BANDPASS else None,
)
# fig_sp.savefig(GAIN_PLOT_BASE, dpi=150, bbox_inches='tight')  # ← force save regardless of dry-run
""")

json.dump(nb, open(NB, 'w'), indent=1)
print(f'Written: {NB}')
print('Verify with:  grep -c savefig preprocess_ugmrt.ipynb')
