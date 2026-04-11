#!/usr/bin/env python3
"""
rebuild_notebook.py  –  patch preprocess_ugmrt.ipynb to add DUD_ANTENNA_NAMES.

Runs entirely outside VS Code.  Call from terminal:
    python rebuild_notebook.py
"""
import json, sys, pathlib, shutil, datetime

NB = pathlib.Path(__file__).parent / 'preprocess_ugmrt.ipynb'

# ── New cell sources ──────────────────────────────────────────────────────────

CELL7_SRC = """\
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
"""

CELL11_SRC = """\
# ── Iteration book-keeping ────────────────────────────────────────────────────
ITER_TAG = 'iter00'

# ── File paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path('/Users/raj030/DATA/gmrt_40_014')
WORK_DIR = BASE_DIR / 'work'
DATA_DIR = BASE_DIR / 'data'

CAL_FITS    = DATA_DIR / '40_014_25jul2021_gsb.FITS'
INDEX_CACHE = WORK_DIR / '40_014_25jul2021_gsb.row_index_cache.npz'

# Row-index cache validation mode:
#   'fast'      -> compare file size + modification time (default for most sessions)
#   'fast+sha'  -> fast check first; on mismatch, verify SHA before rebuild
#   'sha256'    -> always recompute SHA256 (slowest, strictest)
#   'none'      -> trust cache unconditionally
INDEX_VALIDATION_MODE = 'fast+sha'

# ── Known dud antennas ────────────────────────────────────────────────────────
# C07 (antenna_no=31) and S05 (antenna_no=32) have ghost STABXYZ entries in
# the AIPS AN table: their positions are identical to W06 (~1 m apart).
# They are non-functional and must be excluded from all counts and solves.
# Prefix matching is used: 'C07' matches 'C07:31', 'S05' matches 'S05:32', etc.
DUD_ANTENNA_NAMES = ['C07', 'S05']

# ── Source ────────────────────────────────────────────────────────────────────
# Registered calibrators: 3C48, 3C286 (both use Perley-Butler 2017 Table 2).
SOURCE = '3C48'

# Source-derived output file stems (change SOURCE above, paths follow automatically)
_src = SOURCE.lower()
FLAG_TABLE_BASE    = WORK_DIR / f'{_src}_flag_table.json'
FLAG_TABLE_SESSION = WORK_DIR / f'{_src}_flag_table_session.json'
FLAG_TABLE_PATHS   = [p for p in [FLAG_TABLE_BASE, FLAG_TABLE_SESSION] if p.exists()]

# In-memory (dry-run) flag proposals from previous iterations
if 'PENDING_FLAG_TABLES' not in globals():
    PENDING_FLAG_TABLES = []

USE_PENDING_FLAG_TABLES    = True
CLEAR_PENDING_FLAG_TABLES  = False
if CLEAR_PENDING_FLAG_TABLES:
    PENDING_FLAG_TABLES = []

DRY_RUN_BANDPASS  = True
DRY_RUN_FLAG_WRITE = True

# ── Solve options ─────────────────────────────────────────────────────────────
STOKES        = ('RR', 'LL')
CHAN_RANGE     = (64, 191)     # 0-based inclusive channel indices to solve over
MAX_ROWS_SOLVE = 150_000
SMOOTH_WINDOW  = 5             # window size for Re/Im bandpass smoothing
MIN_BASELINES  = 20

# Symmetrize per-sample flags across correlations at load time:
# whenever any correlation is natively flagged (weight ≤ 0) for a given
# (row, channel), all correlations are forced flagged at that point.
FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED = True

# ── Diagnostics options ───────────────────────────────────────────────────────
EXCLUDE_FOR_PLOTS           = []
SKIP_EDGE_CHANNELS          = (0, 0)
TOP_N                       = 12
DIAG_APPLY_FLAGS_ON_THE_FLY     = True
DIAG_SAVE_UNFLAGGED_COMPARISON  = False

# ── Outlier detection metrics ─────────────────────────────────────────────────
#
# OUTLIER_METRIC — signal(s) used to score each antenna/baseline.
#   'RR'  → |Re(V_RR^corrected) − S_ν|  Perley-Butler residual (registered sources only)
#   'LL'  → |Re(V_LL^corrected) − S_ν|  same
#   'V'   → |RR − LL|  Stokes-V proxy, sky-model-independent (works for any source)
#
# OUTLIER_METRIC_MERGE_STRATEGY — when multiple metrics are active:
#   'union'        → flag if threshold exceeded in ANY metric  (recommended)
#   'intersection' → flag only if exceeded in ALL metrics      (conservative)
OUTLIER_METRIC                = ('RR', 'LL', 'V')
OUTLIER_METRIC_MERGE_STRATEGY = 'union'

# ── Flag thresholds ───────────────────────────────────────────────────────────
# V (|RR−LL|) scores are much smaller than RR/LL residuals for clean data.
# A realistic bad antenna on an unpolarised calibrator produces ~5–50 Jy of
# differential signal, so V needs a much lower threshold.
ANTENNA_FLAG_THRESHOLD_JY  = {'RR': 180.0, 'LL': 180.0, 'V': 30.0}   # Jy
BASELINE_FLAG_THRESHOLD_JY = {'RR': 800.0, 'LL': 800.0, 'V': 150.0}  # Jy

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'SOURCE           : {SOURCE}')
print(f'DUD_ANTENNA_NAMES: {DUD_ANTENNA_NAMES}')
print(f'ITER_TAG         : {ITER_TAG}')
print(f'CAL_FITS         : {CAL_FITS}')
print(f'INDEX_CACHE      : {INDEX_CACHE}')
print(f'FLAG_TABLE_BASE  : {FLAG_TABLE_BASE}')
print(f'FLAG_TABLE_SESSION: {FLAG_TABLE_SESSION}')
print(f'FLAG_TABLE_PATHS : {FLAG_TABLE_PATHS}')
print(f'Pending in-memory flag tables: {len(PENDING_FLAG_TABLES)}')
print(f'USE_PENDING_FLAG_TABLES : {USE_PENDING_FLAG_TABLES}')
print(f'DRY_RUN_BANDPASS : {DRY_RUN_BANDPASS}')
print(f'DRY_RUN_FLAG_WRITE: {DRY_RUN_FLAG_WRITE}')
print(f'FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED: {FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED}')
print(f'OUTLIER_METRIC   : {OUTLIER_METRIC}')
print(f'ANTENNA_FLAG_THRESHOLD_JY : {ANTENNA_FLAG_THRESHOLD_JY}')
print(f'BASELINE_FLAG_THRESHOLD_JY: {BASELINE_FLAG_THRESHOLD_JY}')
"""

CELL13_SRC = """\
row_index = q.get_or_build_row_index(
    CAL_FITS,
    cache_path=INDEX_CACHE,
    force_rebuild=False,
    validation_mode=INDEX_VALIDATION_MODE,
    write_cache=True,
    override_dud_names=DUD_ANTENNA_NAMES,
)

print('Index cache path  :', row_index.get('index_cache_path', INDEX_CACHE))
print('Source identity   :', row_index.get('source_identity'))
print('Source SHA256     :', row_index.get('source_sha256'))
print('Dud antennas      :', row_index.get('dud_antenna_names', []))
print('Active antennas   :', len(row_index.get('active_antennas', row_index['antennas'])))
print('Sources in index  :', {v: k for k, v in row_index['id_to_name'].items()})
"""

# ── Cell ID → new source mapping ─────────────────────────────────────────────
PATCHES = {
    '33b48ce6': CELL7_SRC,   # cell 7  – §3 Source Observability
    'dcc3923b': CELL11_SRC,  # cell 11 – §5 Configuration
    '76ede542': CELL13_SRC,  # cell 13 – §6 Row-Index Cache
}

# ── Load, patch, write ────────────────────────────────────────────────────────
nb = json.loads(NB.read_text())

patched = 0
for cell in nb['cells']:
    cid = cell.get('id', '')
    if cid in PATCHES:
        # Notebook format stores source as a list of lines (each ending in \n
        # except the last).  Split on newlines, re-add \n to all but last.
        lines = PATCHES[cid].splitlines(keepends=True)
        cell['source'] = lines
        # Clear any stale outputs
        cell['outputs'] = []
        cell['execution_count'] = None
        print(f'  patched cell id={cid}')
        patched += 1

if patched != len(PATCHES):
    print(f'ERROR: expected {len(PATCHES)} patches but applied {patched}', file=sys.stderr)
    sys.exit(1)

# Back up original
bak = NB.with_suffix('.ipynb.bak')
shutil.copy2(NB, bak)

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f'Written: {NB}  (backup: {bak})')
print('Verify with:  grep -c DUD preprocess_ugmrt.ipynb')
