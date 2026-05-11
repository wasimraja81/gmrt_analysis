"""Patch notebook: SOLVE_ELEVATION_MIN_DEG + aligned V thresholds."""
import json, shutil, pathlib

NB = pathlib.Path('preprocess_ugmrt.ipynb')
shutil.copy(NB, NB.with_suffix('.ipynb.bak3'))
nb = json.load(open(NB))

def find_cell(nb, *markers):
    for c in nb['cells']:
        src = ''.join(c.get('source', []))
        if all(m in src for m in markers):
            return c
    return None

config_c = find_cell(nb, 'ITER_TAG', 'ANTENNA_FLAG_THRESHOLD_JY')
solve_c  = find_cell(nb, 'derive_bandpass_iteration', 'BANDPASS_OUT')
diag_c   = find_cell(nb, 'run_bandpass_diagnostics', 'DIAG_PLOT_BASE')

print('config cell id:', config_c['id'] if config_c else 'NOT FOUND')
print('solve  cell id:', solve_c['id']  if solve_c  else 'NOT FOUND')
print('diag   cell id:', diag_c['id']   if diag_c   else 'NOT FOUND')

# ── Config cell ───────────────────────────────────────────────────────────────
config_new = """\
# ── Iteration book-keeping ────────────────────────────────────────────────────
ITER_TAG = 'iter02'

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

DRY_RUN_BANDPASS   = True
DRY_RUN_FLAG_WRITE = True

# ── Solve options ─────────────────────────────────────────────────────────────
STOKES        = ('RR', 'LL')
CHAN_RANGE     = (64, 191)     # 0-based inclusive channel indices to solve over
MAX_ROWS_SOLVE = 150_000
SMOOTH_WINDOW  = 5             # window size for Re/Im bandpass smoothing
MIN_BASELINES  = 20

# Elevation cut applied uniformly to the solve, diagnostics load, and vis load.
# Matches --set "SOLVE_ELEVATION_MIN_DEG=25.0" in the v-based-outlier-detection script.
SOLVE_ELEVATION_MIN_DEG = 25.0

# Symmetrize per-sample flags across correlations at load time:
# whenever any correlation is natively flagged (weight <= 0) for a given
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
#   'RR'  -> |Re(V_RR^corrected) - S_nu|  Perley-Butler residual (registered sources only)
#   'LL'  -> |Re(V_LL^corrected) - S_nu|  same
#   'V'   -> |RR - LL|  Stokes-V proxy, sky-model-independent (works for any source)
#
# OUTLIER_METRIC_MERGE_STRATEGY — when multiple metrics are active:
#   'union'        -> flag if threshold exceeded in ANY metric  (recommended)
#   'intersection' -> flag only if exceeded in ALL metrics      (conservative)
OUTLIER_METRIC                = ('RR', 'LL', 'V')
OUTLIER_METRIC_MERGE_STRATEGY = 'union'

# ── Flag thresholds ───────────────────────────────────────────────────────────
# V thresholds match the v-based-outlier-detection.sh defaults (5 Jy for both
# antenna and baseline).  RR/LL model-residual thresholds are wider because
# field confusion + model errors add scatter; tighten once the data is clean.
#
# Equivalent shell invocation:
#   --set "ANTENNA_FLAG_THRESHOLD_JY={'V': 5.0}"
#   --set "BASELINE_FLAG_THRESHOLD_JY={'V': 5.0}"
ANTENNA_FLAG_THRESHOLD_JY  = {'RR': 50.0, 'LL': 50.0, 'V': 5.0}   # Jy
BASELINE_FLAG_THRESHOLD_JY = {'RR': 200.0, 'LL': 200.0, 'V': 5.0}  # Jy

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
print(f'SOLVE_ELEVATION_MIN_DEG: {SOLVE_ELEVATION_MIN_DEG}')
print(f'FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED: {FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED}')
print(f'OUTLIER_METRIC   : {OUTLIER_METRIC}')
print(f'ANTENNA_FLAG_THRESHOLD_JY : {ANTENNA_FLAG_THRESHOLD_JY}')
print(f'BASELINE_FLAG_THRESHOLD_JY: {BASELINE_FLAG_THRESHOLD_JY}')
"""

# ── Solve cell ────────────────────────────────────────────────────────────────
solve_new = """\
_src = SOURCE.lower()
BANDPASS_OUT = WORK_DIR / f'{_src}_bandpass_25jul_gsb.npz'

active_pending = PENDING_FLAG_TABLES if USE_PENDING_FLAG_TABLES else []

bandpass_run = q.derive_bandpass_iteration(
    fits_path          = CAL_FITS,
    index              = row_index,
    bandpass_out       = BANDPASS_OUT,
    source             = SOURCE,
    stokes             = STOKES,
    chan_range         = CHAN_RANGE,
    max_rows           = MAX_ROWS_SOLVE,
    smooth_window      = SMOOTH_WINDOW,
    min_baselines      = MIN_BASELINES,
    ignore_autos       = True,
    elevation_min_deg  = SOLVE_ELEVATION_MIN_DEG,
    flag_table_path    = FLAG_TABLE_PATHS if FLAG_TABLE_PATHS else None,
    flag_table         = active_pending if active_pending else None,
    flag_all_corrs_if_any_rawvis_flagged = FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
    iteration_tag      = ITER_TAG,
    dry_run            = DRY_RUN_BANDPASS,
)

bandpass_sol      = bandpass_run['solution']
bandpass_out_path = bandpass_run['bandpass_out']

print('dry_run            :', bandpass_run['dry_run'])
print('bandpass_out       :', bandpass_out_path)
print('on-disk flags used :', FLAG_TABLE_PATHS)
print('pending flags used :', len(active_pending))
print('merged flag count  :', bandpass_sol.get('flag_table_count', 0))
print('rows by flag tables:', bandpass_sol.get('solve_dropped_rows_by_flag_table', 0))
"""

# ── Diagnostics cell ──────────────────────────────────────────────────────────
diag_new = """\
_src = SOURCE.lower()
DIAG_PLOT_BASE = WORK_DIR / f'{_src}_bandpass_diagnostics.png'

diag = q.run_bandpass_diagnostics(
    row_index,
    bandpass_sol,
    source    = SOURCE,
    chan_range = CHAN_RANGE,
    stokes    = STOKES,
    max_rows  = MAX_ROWS_SOLVE,
    exclude_antennas        = EXCLUDE_FOR_PLOTS,
    apply_flag_tables       = DIAG_APPLY_FLAGS_ON_THE_FLY,
    flag_table_path         = FLAG_TABLE_PATHS if FLAG_TABLE_PATHS else None,
    flag_table              = active_pending if active_pending else None,
    flag_all_corrs_if_any_rawvis_flagged = FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
    elevation_min_deg       = SOLVE_ELEVATION_MIN_DEG,
    skip_edge_channels      = SKIP_EDGE_CHANNELS,
    top_n                   = TOP_N,
    ranking_metric          = OUTLIER_METRIC,
    title                   = f'{SOURCE} diagnostics | {ITER_TAG}',
    save_path               = DIAG_PLOT_BASE,
)

# ── Channel/frequency sanity check ────────────────────────────────────────────
_freqs = diag['freqs_hz']
_mask  = diag['chan_mask']
_chan_range_expected = CHAN_RANGE[1] - CHAN_RANGE[0] + 1
print(f'CHAN_RANGE          : {CHAN_RANGE}  ({_chan_range_expected} expected)')
print(f'Channels loaded     : {_freqs.size}')
print(f'Channels plotted    : {int(_mask.sum())}  (SKIP_EDGE_CHANNELS={SKIP_EDGE_CHANNELS})')
print(f'Freq range plotted  : {diag["plot_freq_min_mhz"]:.3f} - {diag["plot_freq_max_mhz"]:.3f} MHz')
print(f'Bandwidth plotted   : {diag["plot_freq_max_mhz"] - diag["plot_freq_min_mhz"]:.3f} MHz')
"""

def apply(cell, new_source):
    cell['source'] = new_source.splitlines(keepends=True)
    print(f'  patched cell id={cell["id"]}')

apply(config_c, config_new)
apply(solve_c,  solve_new)
apply(diag_c,   diag_new)

json.dump(nb, open(NB, 'w'), indent=1)
print(f'Written: {NB}')
