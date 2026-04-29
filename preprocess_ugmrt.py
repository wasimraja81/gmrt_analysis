#!/usr/bin/env python3
"""
preprocess_ugmrt.py  –  command-line driver for the uGMRT bandpass workflow.

All configuration lives in preprocess_ugmrt.cfg (Python syntax).
Pass a different config file with --config if needed.

Usage examples
--------------
# One interactive pass of all steps, labelled iter01
./run_preprocess.sh --step all --iter-tag iter01

# Interactive loop: N iterations, flags accumulate in memory, blocking plot
# and accept/reject prompt after each.  Add --dry-run to preview only.
./run_preprocess.sh --step all --n-iters 5
./run_preprocess.sh --step all --n-iters 5 --start-iter 3   # iter03..iter07
./run_preprocess.sh --step all --n-iters 5 --dry-run        # preview, nothing written

# Run individual steps  (--iter-tag names the run)
./run_preprocess.sh --step 1                       # build index only
./run_preprocess.sh --step 2 --iter-tag iter01     # bandpass only
./run_preprocess.sh --step 3 --iter-tag iter01     # diagnostics / plot only
./run_preprocess.sh --step 4 --iter-tag iter01     # flag proposals only

# Batch automated loop — fully unattended, writes outputs to disk
./run_preprocess.sh --step all --auto --n-iters 10
./run_preprocess.sh --step all --auto --n-iters 10 --dry-run        # preview only
./run_preprocess.sh --step all --auto --n-iters 5 --start-iter 6

# Use a custom config file
./run_preprocess.sh --config /path/to/base.cfg --step all --iter-tag iter01

# Override individual config keys on the CLI (Python expressions accepted)
./run_preprocess.sh --config base.cfg \
    --set "CAL_FITS=Path('/data/3c147_aug.FITS')" \
    --set "WORK_DIR=Path('/data/3c147/work')" \
    --set "SOURCE='3C147'" \
    --step all --n-iters 5

# Default is to write outputs to disk.  Pass --dry-run to preview without writing.

Plot behaviour
--------------
Diagnostic plots are ALWAYS saved to disk as PNG files (WORK_DIR/<stem>_<iter>.png)
regardless of --dry-run or any other flag.

Whether a popup window also opens is controlled by the MPLBACKEND environment variable:

  MPLBACKEND=MacOSX  → popup window per iteration (blocking until closed)  [macOS default]
  MPLBACKEND=TkAgg   → popup window per iteration (blocking until closed)  [Linux]
  MPLBACKEND=Agg     → no popup windows; PNGs only

Override at any time by prefixing the command:
  MPLBACKEND=Agg     ./run_preprocess.sh --step all --n-iters 5   # suppress popups
  MPLBACKEND=MacOSX  ./run_preprocess.sh --step all --n-iters 5   # force popups
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

from workflow_common import (
    add_common_config_set_arguments,
    add_log_level_argument,
    apply_overrides_to_globals,
    bootstrap_config_from_cli,
    derive_index_cache,
    format_table_application_note,
    load_config_into_globals,
    resolve_versioned_path,
)

# Module-level logger – configured in main() once log level / paths are known.
log = logging.getLogger('ugmrt.preprocess')

# Module-level config globals – populated by _load_config() before any step runs.
# INDEX_CACHE is not in the config; it is derived from CAL_FITS in main().
CAL_FITS = INDEX_CACHE = BANDPASS_OUT = DIAG_PLOT_BASE = DIAG_PLOT_UNFLAGGED = None
FLAG_TABLE_SESSION = FLAG_TABLE_PATHS = None
INDEX_VALIDATION_MODE = None
DUD_ANTENNA_NAMES = None
SOURCE = STOKES = CHAN_RANGE = None
MAX_ROWS_SOLVE = SMOOTH_WINDOW = MIN_BASELINES = None
ITER_TAG = None
FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED = None
DIAG_APPLY_FLAGS_ON_THE_FLY = DIAG_SAVE_UNFLAGGED_COMPARISON = None
EXCLUDE_FOR_PLOTS = SKIP_EDGE_CHANNELS = TOP_N = MAX_ROWS_DIAG = None
OUTLIER_METRIC = OUTLIER_METRIC_MERGE_STRATEGY = None
ANTENNA_FLAG_THRESHOLD_JY = BASELINE_FLAG_THRESHOLD_JY = None
FLAG_WHAT_TO_FLAG = MAX_ANTENNAS_TO_FLAG = MAX_BASELINES_TO_FLAG = None
AUTO_N_ITERS = AUTO_START_ITER = AUTO_ITER_PREFIX = AUTO_ITER_WIDTH = None
DRY_RUN_BANDPASS = DRY_RUN_FLAG_WRITE = None
GAIN_PLOT_BASE = None
CONVERGENCE_EPSILON              = 0.0
CONVERGENCE_MIN_ITERS            = 3
COMPARE_METRICS_FOR_CONVERGENCE  = ['V', 'Model']
CONVERGENCE_COMBINE_STRATEGY     = 'any'
RUN_ITER0_DIAGNOSTIC             = True
LOG_LEVEL = 'INFO'
DOCAL = 'on'
DOFLAG = 'on'
CALVER = 'latest'
FLAGVER = 'latest'


def _effective_flag_table_paths() -> list:
    return list(globals().get('_FLAG_TABLE_PATHS_EFFECTIVE', FLAG_TABLE_PATHS or []))


def _refresh_effective_flag_paths() -> None:
    _flag_paths = [Path(p) for p in (FLAG_TABLE_PATHS or [])]
    if str(globals().get('DOFLAG', 'on')).lower() == 'off':
        _flag_paths = []
    else:
        _flagver = str(globals().get('FLAGVER', 'latest'))
        if _flagver != 'latest':
            _src = str(globals().get('SOURCE', 'src')).lower()
            _cands = list(_flag_paths)
            _cands.extend(Path(globals().get('WORK_DIR', '.')).glob(f'{_src}_flag_table*.json'))
            _flag_paths = [resolve_versioned_path(
                candidates=_cands,
                version=_flagver,
                default_path=(_flag_paths[-1] if _flag_paths else None),
                label='flag table',
            )]
    globals()['_FLAG_TABLE_PATHS_EFFECTIVE'] = _flag_paths

# ── Final clustering stage (post-convergence, optional) ───────────────────────
# When RUN_FINAL_CLUSTERING=True the auto workflow runs one extra pass of the
# outlier_detection.py clustering algorithm *after* the residual-based loop
# has converged (stopped on C1 or an epsilon criterion).  The corrected
# visibilities are already in memory so no extra FITS I/O is needed.
RUN_FINAL_CLUSTERING                     = False
CLUSTERING_CORR                          = 'V'
CLUSTERING_THRESHOLD_JY                  = 5.0
CLUSTERING_THRESHOLD_LOW_JY              = None   # e.g. {'RR': 0.5, 'LL': 0.5} to catch dead antennas
CLUSTERING_MIN_CLUSTER_FRACTION          = 0.80
CLUSTERING_MIN_DISTINCT_BASELINES_FOR_ANT = 3
CLUSTERING_MIN_BURST_BASELINE_FRACTION   = 0.50
CLUSTERING_WHOLE_SCAN_BAD_FRACTION       = 0.70
CLUSTERING_PER_SCAN_ANT_FRACTION         = 0.60
CLUSTERING_PER_SCAN_ANT_BL_FRACTION      = 0.50
CLUSTERING_BASELINE_MAX_BAD_SAMPLES      = 100
CLUSTERING_MAX_GAP_SAMPLES               = 2
CLUSTERING_MAX_GAP_MINUTES               = 30.0
CLUSTERING_SCAN_GAP_MINUTES              = 2.0

# Ordered list of every config key this driver knows about.
# Used to write reproducible config snapshots to the log directory.
_CONFIG_KEYS = (
    'CAL_FITS', 'WORK_DIR', 'BANDPASS_OUT', 'DIAG_PLOT_BASE',
    'DIAG_PLOT_UNFLAGGED', 'INDEX_VALIDATION_MODE',
    'FLAG_TABLE_SESSION', 'FLAG_TABLE_PATHS',
    'SOURCE', 'STOKES', 'CHAN_RANGE',
    'MAX_ROWS_SOLVE', 'SMOOTH_WINDOW', 'MIN_BASELINES',
    'FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED',
    'ITER_TAG',
    'DIAG_APPLY_FLAGS_ON_THE_FLY', 'DIAG_SAVE_UNFLAGGED_COMPARISON',
    'EXCLUDE_FOR_PLOTS', 'SKIP_EDGE_CHANNELS', 'TOP_N', 'MAX_ROWS_DIAG',
    'OUTLIER_METRIC', 'OUTLIER_METRIC_MERGE_STRATEGY',
    'ANTENNA_FLAG_THRESHOLD_JY', 'BASELINE_FLAG_THRESHOLD_JY',
    'FLAG_WHAT_TO_FLAG', 'MAX_ANTENNAS_TO_FLAG', 'MAX_BASELINES_TO_FLAG',
    'AUTO_N_ITERS', 'AUTO_START_ITER', 'AUTO_ITER_PREFIX', 'AUTO_ITER_WIDTH',
    'GAIN_PLOT_BASE', 'CONVERGENCE_EPSILON', 'CONVERGENCE_MIN_ITERS',
    'COMPARE_METRICS_FOR_CONVERGENCE', 'CONVERGENCE_COMBINE_STRATEGY',
    'RUN_ITER0_DIAGNOSTIC',
    'LOG_LEVEL',
    'RUN_FINAL_CLUSTERING', 'CLUSTERING_CORR', 'CLUSTERING_THRESHOLD_JY',
    'CLUSTERING_THRESHOLD_LOW_JY',
    'CLUSTERING_MIN_CLUSTER_FRACTION', 'CLUSTERING_MIN_DISTINCT_BASELINES_FOR_ANT',
    'CLUSTERING_MIN_BURST_BASELINE_FRACTION', 'CLUSTERING_WHOLE_SCAN_BAD_FRACTION',
    'CLUSTERING_PER_SCAN_ANT_FRACTION', 'CLUSTERING_PER_SCAN_ANT_BL_FRACTION',
    'CLUSTERING_BASELINE_MAX_BAD_SAMPLES', 'CLUSTERING_MAX_GAP_SAMPLES',
    'CLUSTERING_MAX_GAP_MINUTES', 'CLUSTERING_SCAN_GAP_MINUTES',
)


def _data_coverage_summary(index, cumulative_flag_table,
                           n_obs_active_bls=None, n_obs_active_ants=None):
    """Return a log string summarising the flagging impact on data coverage.

    Uses observed unique (ant1,ant2) pairs from the most recent bandpass
    solve (``n_obs_active_bls``, ``n_obs_active_ants``) when available.
    Falls back to the DUD-corrected ``active_antennas`` count from the index
    (better than the raw antenna-table count which includes dead hardware).
    """
    # Denominator: observed > DUD-corrected theoretical > raw theoretical
    active_ants = index.get('active_antennas', index.get('antennas', []))
    n_ants_theory = len(active_ants)
    n_bls_theory  = n_ants_theory * max(0, n_ants_theory - 1) // 2

    n_ants_denom = n_obs_active_ants if n_obs_active_ants is not None else n_ants_theory
    n_bls_denom  = n_obs_active_bls  if n_obs_active_bls  is not None else n_bls_theory
    obs_label = '' if n_obs_active_bls is not None else ' (est.)'

    flagged_ants  = list(cumulative_flag_table.get('bad_antennas', []))
    flagged_bases = list(cumulative_flag_table.get('bad_baselines', []))
    n_fa = len(flagged_ants)
    n_fb = len(flagged_bases)
    n_rem_ants = n_ants_denom - n_fa
    n_eff_rem  = n_rem_ants * max(0, n_rem_ants - 1) // 2 - n_fb
    frac = n_eff_rem / max(1, n_bls_denom)
    warn = '  *** WARNING: >50% baselines affected — solutions may be unreliable ***' if frac < 0.5 else ''
    return (
        f'  Coverage: {n_fa}/{n_ants_denom} antennas flagged {flagged_ants or "—"} | '
        f'{n_fb} explicit baseline flags | '
        f'≈{n_eff_rem}/{n_bls_denom}{obs_label} baselines remain ({frac*100:.0f}%)'
        + (f'\n{warn}' if warn else '')
    )


def _load_config(config_path: str) -> None:
    """
    Exec the config file and inject every public name into this module's globals.
    The config file is plain Python: Path(), tuples, dicts, booleans all work.
    """
    load_config_into_globals(config_path, globals())


def _apply_overrides(overrides: list) -> None:
    """Apply --set KEY=expr overrides to module globals after the config is loaded.

    Each item must be a string of the form ``KEY=<python-expression>``.  The
    expression is evaluated in a namespace that has ``Path`` pre-imported so
    path values can be written naturally::

        --set "CAL_FITS=Path('/data/3c147_aug.FITS')"
        --set "SOURCE='3C147'"
        --set "CHAN_RANGE=(64,191)"
        --set "ANTENNA_FLAG_THRESHOLD_JY={'RR':180,'LL':180,'V':30}"
    """
    apply_overrides_to_globals(overrides, globals(), log=log)


def _rederive_source_paths() -> None:
    """Re-compute all output paths that embed the source name.

    Called after --set overrides are applied so that ``--set "SOURCE='3C286'"``
    correctly updates BANDPASS_OUT, DIAG_PLOT_BASE, FLAG_TABLE_* etc.  Without
    this step the paths would remain frozen to whatever SOURCE was set to when
    the config file was exec'd.
    """
    g = globals()
    src = str(g.get('SOURCE', '3C48')).lower()
    work = Path(g.get('WORK_DIR', '.'))        # already a Path from config exec

    # Derive the stem used by the config (everything after the source prefix).
    # Strategy: read the original BANDPASS_OUT stem and replace the leading
    # token up to the first '_bandpass' with the new source name.
    # Fallback: build sensible defaults if the existing value can't be parsed.
    def _restem(old_path, new_prefix, marker):
        """Replace the source-name prefix in a filename."""
        old = Path(old_path)
        name = old.name
        idx = name.find(marker)
        if idx != -1:
            return work / (new_prefix + name[idx:])
        # Couldn't find marker — fall back to <src><marker><suffix>
        return work / (new_prefix + marker + old.suffix)

    g['BANDPASS_OUT']        = _restem(g.get('BANDPASS_OUT',        work / f'{src}_bandpass.npz'),         src, '_bandpass')
    g['DIAG_PLOT_BASE']      = _restem(g.get('DIAG_PLOT_BASE',      work / f'{src}_bandpass_diagnostics.png'), src, '_bandpass')
    g['DIAG_PLOT_UNFLAGGED'] = _restem(g.get('DIAG_PLOT_UNFLAGGED', work / f'{src}_bandpass_diagnostics_unflagged.png'), src, '_bandpass')
    g['FLAG_TABLE_BASE']     = _restem(g.get('FLAG_TABLE_BASE',     work / f'{src}_flag_table.json'),      src, '_flag_table')
    g['FLAG_TABLE_SESSION']  = _restem(g.get('FLAG_TABLE_SESSION',  work / f'{src}_flag_table_session.json'), src, '_flag_table')
    g['FLAG_TABLE_PATHS']    = [p for p in [g['FLAG_TABLE_BASE'], g['FLAG_TABLE_SESSION']] if Path(p).exists()]
    if g.get('GAIN_PLOT_BASE') is not None:
        g['GAIN_PLOT_BASE']  = _restem(g['GAIN_PLOT_BASE'], src, '_bandpass')


def _derive_index_cache() -> Path:
    """Return the index-cache path derived from CAL_FITS.

    Convention: <WORK_DIR>/<cal_fits_stem>.index.npz
    Falls back to the same directory as CAL_FITS if WORK_DIR is not set.
    If INDEX_CACHE was set explicitly in the config file that value wins.
    """
    return derive_index_cache(INDEX_CACHE, CAL_FITS, WORK_DIR)


def _setup_logging(work_dir: Path, iter_tag: str, log_level: str,
                   timestamp: str = None) -> Path:
    """Configure console + file logging for the run.  Returns the log directory.

    Log files are written to <work_dir>/logs/ and named:
        preprocess_<iter_tag>_<YYYYMMDD_HHMMSS>.log
    A symlink ``preprocess_latest.log`` in the same directory always points
    to the most recent log file for convenience.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    log.setLevel(logging.DEBUG)          # capture everything; handlers filter

    fmt = logging.Formatter(
        fmt='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Console handler – respects the requested level.
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric_level)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    # File handler – always DEBUG so the log captures full detail.
    log_dir = work_dir / 'logs'
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts         = timestamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file   = log_dir / f'preprocess_{iter_tag}_{ts}.log'
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        log.addHandler(fh)
        # Convenience symlink: preprocess_latest.log
        latest = log_dir / 'preprocess_latest.log'
        try:
            latest.unlink(missing_ok=True)
            latest.symlink_to(log_file.name)
        except OSError:
            pass
        log.debug('Log file: %s', log_file)
    except OSError as exc:
        log.warning('Could not create log file in %s: %s', log_dir, exc)
    return log_dir


def _save_config_snapshot(log_dir: Path, timestamp: str,
                          config_path: str, overrides: list) -> None:
    """Write the effective (post-override) config to a Python file in log_dir.

    Filename convention::

        config_<FITS_stem>_<YYYYMMDD_HHMMSS>.py

    where FITS_stem is derived from CAL_FITS so the dataset identity is
    immediately visible in the filename.  A symlink ``config_latest.py``
    always points to the most recent snapshot.

    Keys that were modified by --set are annotated with an inline comment.
    """
    import shutil
    override_keys = set()
    for item in (overrides or []):
        if '=' in item:
            override_keys.add(item.partition('=')[0].strip())

    # Snapshot filename encodes dataset + timestamp.
    fits_stem = Path(CAL_FITS).stem if CAL_FITS else 'unknown'
    fname = f'config_{fits_stem}_{timestamp}.py'
    snap  = log_dir / fname

    g = globals()
    lines = [
        f'# Effective config snapshot',
        f'# Base config : {config_path}',
        f'# Timestamp   : {timestamp}',
        f'# Dataset     : {fits_stem}',
    ]
    if override_keys:
        lines.append(f'# CLI overrides: {", ".join(sorted(override_keys))}')
    lines += ['', 'from pathlib import Path', '']

    for key in _CONFIG_KEYS:
        val = g.get(key)
        suffix = '  # << --set override' if key in override_keys else ''
        try:
            lines.append(f'{key} = {val!r}{suffix}')
        except Exception:
            lines.append(f'# {key} : <unrepresentable>')

    try:
        snap.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        latest = log_dir / 'config_latest.py'
        try:
            latest.unlink(missing_ok=True)
            latest.symlink_to(snap.name)
        except OSError:
            pass
        log.info('Config snapshot : %s', snap)
    except OSError as exc:
        log.warning('Could not write config snapshot: %s', exc)


def _log_run_parameters(config_path: str, args) -> None:
    """Emit all runtime parameters at INFO level for reproducibility."""
    sep = '─' * 56
    log.info(sep)
    log.info('uGMRT preprocess  |  step=%-6s  tag=%s', args.step, args.iter_tag)
    log.info(sep)
    log.info('[invocation]')
    log.info('  config         : %s', config_path)
    log.info('  step           : %s', args.step)
    log.info('  iter_tag       : %s', args.iter_tag)
    log.info('  dry_run        : %s', args.dry_run)
    log.info('  log_level      : %s', args.log_level)
    log.info('[paths]')
    log.info('  CAL_FITS            : %s', CAL_FITS)
    log.info('  INDEX_CACHE (eff.)  : %s', _derive_index_cache())
    log.info('  BANDPASS_OUT        : %s', BANDPASS_OUT)
    log.info('  DIAG_PLOT_BASE      : %s', DIAG_PLOT_BASE)
    log.info('  WORK_DIR            : %s', WORK_DIR)
    log.info('[index]')
    log.info('  INDEX_VALIDATION_MODE : %s', INDEX_VALIDATION_MODE)
    log.info('[solve]')
    log.info('  SOURCE         : %s', SOURCE)
    log.info('  STOKES         : %s', STOKES)
    log.info('  CHAN_RANGE      : %s', CHAN_RANGE)
    log.info('  MAX_ROWS_SOLVE  : %s', MAX_ROWS_SOLVE)
    log.info('  SMOOTH_WINDOW   : %s', SMOOTH_WINDOW)
    log.info('  MIN_BASELINES   : %s', MIN_BASELINES)
    log.info('  FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED : %s', FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED)
    log.info('[flags]')
    log.info('  FLAG_TABLE_PATHS : %s', FLAG_TABLE_PATHS)
    log.info('[diagnostics]')
    log.info('  MAX_ROWS_DIAG                  : %s', MAX_ROWS_DIAG)
    log.info('  SKIP_EDGE_CHANNELS             : %s', SKIP_EDGE_CHANNELS)
    log.info('  TOP_N                          : %s', TOP_N)
    log.info('  EXCLUDE_FOR_PLOTS              : %s', EXCLUDE_FOR_PLOTS)
    log.info('  DIAG_APPLY_FLAGS_ON_THE_FLY    : %s', DIAG_APPLY_FLAGS_ON_THE_FLY)
    log.info('  DIAG_SAVE_UNFLAGGED_COMPARISON : %s', DIAG_SAVE_UNFLAGGED_COMPARISON)
    log.info('[outlier scoring]')
    log.info('  OUTLIER_METRIC                : %s', OUTLIER_METRIC)
    log.info('  OUTLIER_METRIC_MERGE_STRATEGY : %s', OUTLIER_METRIC_MERGE_STRATEGY)
    log.info('  ANTENNA_FLAG_THRESHOLD_JY     : %s', ANTENNA_FLAG_THRESHOLD_JY)
    log.info('  BASELINE_FLAG_THRESHOLD_JY    : %s', BASELINE_FLAG_THRESHOLD_JY)
    log.info('[flag proposals]')
    log.info('  FLAG_WHAT_TO_FLAG    : %s', FLAG_WHAT_TO_FLAG)
    log.info('  MAX_ANTENNAS_TO_FLAG : %s', MAX_ANTENNAS_TO_FLAG)
    log.info('  MAX_BASELINES_TO_FLAG: %s', MAX_BASELINES_TO_FLAG)
    if args.auto:
        log.info('[auto-iteration / batch]')
        log.info('  n_iters      : %s', args.n_iters)
        log.info('  start_iter   : %s', args.start_iter)
        log.info('  AUTO_ITER_PREFIX : %s', AUTO_ITER_PREFIX)
        log.info('  AUTO_ITER_WIDTH  : %s', AUTO_ITER_WIDTH)
    elif args.n_iters > 1:
        log.info('[interactive loop]')
        log.info('  n_iters      : %s', args.n_iters)
        log.info('  start_iter   : %s', args.start_iter)
        log.info('  AUTO_ITER_PREFIX : %s', AUTO_ITER_PREFIX)
        log.info('  AUTO_ITER_WIDTH  : %s', AUTO_ITER_WIDTH)
    log.info(sep)


def _import_ugmrt():
    """Import (or reload) ugmrt_query from the same directory as this script."""
    import importlib, sys
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    if 'ugmrt_query' in sys.modules:
        return importlib.reload(sys.modules['ugmrt_query'])
    import ugmrt_query
    return ugmrt_query


def step_1_index(q):
    """Build or load the persistent row-index cache."""
    cache = _derive_index_cache()
    log.info('── Step 1: row index')
    log.debug('  cache path: %s', cache)
    index = q.get_or_build_row_index(
        CAL_FITS,
        cache_path=cache,
        force_rebuild=False,
        validation_mode=INDEX_VALIDATION_MODE,
        write_cache=True,
        override_dud_names=DUD_ANTENNA_NAMES if DUD_ANTENNA_NAMES is not None else None,
    )
    log.info('  index_cache_path : %s', index.get('index_cache_path', cache))
    log.info('  source_identity  : %s', index.get('source_identity'))
    log.info('  dud_antennas     : %s', index.get('dud_antenna_names', []))
    log.info('  active_antennas  : %d', len(index.get('active_antennas', index['antennas'])))
    log.debug('  source_sha256    : %s', index.get('source_sha256'))
    return index


def step_2_bandpass(q, index, iter_tag, dry_run, flag_tables_extra=None):
    """Derive bandpass solution."""
    log.info('── Step 2: bandpass')
    _flag_paths = _effective_flag_table_paths()
    run = q.derive_bandpass_iteration(
        fits_path=CAL_FITS,
        index=index,
        bandpass_out=BANDPASS_OUT,
        source=SOURCE,
        stokes=STOKES,
        chan_range=CHAN_RANGE,
        max_rows=MAX_ROWS_SOLVE,
        smooth_window=SMOOTH_WINDOW,
        min_baselines=MIN_BASELINES,
        ignore_autos=True,
        flag_table_path=_flag_paths if _flag_paths else None,
        flag_table=flag_tables_extra if flag_tables_extra else None,
        flag_all_corrs_if_any_rawvis_flagged=FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
        iteration_tag=iter_tag,
        dry_run=dry_run,
    )
    sol = run['solution']
    log.info('  dry_run              : %s', run['dry_run'])
    log.info('  output path          : %s', run['bandpass_out'])
    log.info('  on-disk flag tables  : %s', _flag_paths)
    log.info('  flag table count     : %s', sol.get('flag_table_count', 0))
    log.info('  rows dropped by flags: %s', sol.get('solve_dropped_rows_by_flag_table', 0))
    return run


def step_3_diagnostics(q, index, bandpass_sol, iter_tag, flag_tables_extra=None, cal_path=None):
    """Run bandpass diagnostics and produce plots."""
    log.info('── Step 3: diagnostics')
    _flag_paths = _effective_flag_table_paths()
    _apply_corr = str(globals().get('DOCAL', 'on')).lower() == 'on'
    _table_note = format_table_application_note(
        docal=DOCAL,
        doflag=DOFLAG,
        calver=CALVER,
        flagver=FLAGVER,
        cal_path=cal_path,
        flag_paths=_flag_paths,
    )
    _diag_solution = bandpass_sol if _apply_corr else None
    diag = q.run_bandpass_diagnostics(
        index,
        _diag_solution,
        source=SOURCE,
        chan_range=CHAN_RANGE,
        stokes=STOKES,
        max_rows=MAX_ROWS_DIAG,
        exclude_antennas=EXCLUDE_FOR_PLOTS,
        apply_flag_tables=DIAG_APPLY_FLAGS_ON_THE_FLY and (str(globals().get('DOFLAG', 'on')).lower() == 'on'),
        flag_table_path=_flag_paths if _flag_paths else None,
        flag_table=flag_tables_extra if flag_tables_extra else None,
        apply_correction=_apply_corr,
        flag_all_corrs_if_any_rawvis_flagged=FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
        skip_edge_channels=SKIP_EDGE_CHANNELS,
        top_n=TOP_N,
        ranking_metric=OUTLIER_METRIC,
        title=f'{SOURCE} diagnostics | {iter_tag}{_table_note}',
        save_path=DIAG_PLOT_BASE.with_name(
            f'{DIAG_PLOT_BASE.stem}_{iter_tag}{DIAG_PLOT_BASE.suffix}'
        ) if DIAG_PLOT_BASE is not None else None,
    )
    n_loaded  = diag['freqs_hz'].size
    n_plotted = int(diag['chan_mask'].sum())
    expected  = CHAN_RANGE[1] - CHAN_RANGE[0] + 1
    log.info('  plot saved       : %s', diag['save_path'] or '(not saved)')
    log.info('  tables applied   : %s', _table_note.strip())
    # Bandpass gain grid plot (amp + phase per antenna)
    if GAIN_PLOT_BASE is not None:
        import numpy as np
        _gain_plot_path = GAIN_PLOT_BASE.with_name(
            f'{GAIN_PLOT_BASE.stem}_{iter_tag}{GAIN_PLOT_BASE.suffix}'
        )
        _canon_ants = list(index.get('active_antennas', index.get('antennas', [])))
        _canon_ids = [int(a['antenna_no']) for a in _canon_ants if a.get('antenna_no') is not None]
        _canon_names = [str(a.get('name', f'Ant{int(a["antenna_no"])}')) for a in _canon_ants if a.get('antenna_no') is not None]
        _sol_ids = {int(x) for x in np.asarray(bandpass_sol.get('antenna_ids', []), dtype=np.int32)}
        _missing_slots = len([aid for aid in _canon_ids if aid not in _sol_ids])
        log.info('  gains grid slots : %d canonical, %d placeholder(s) not in solution',
                 len(_canon_ids), _missing_slots)
        q.plot_bandpass_solution_grid(
            bandpass_sol,
            title=f'{SOURCE} bandpass gains | {iter_tag}{_table_note}',
            skip_edge_channels=SKIP_EDGE_CHANNELS,
            save_path=_gain_plot_path,
            canonical_antenna_ids=_canon_ids if _canon_ids else None,
            canonical_antenna_names=_canon_names if _canon_names else None,
        )
        log.info('  gains plot saved  : %s', _gain_plot_path)
        log.info('  tables applied    : %s', _table_note.strip())
    log.info('  CHAN_RANGE       : %s  (%d channels expected)', CHAN_RANGE, expected)
    log.info('  channels loaded  : %d', n_loaded)
    log.info('  channels plotted : %d  (SKIP_EDGE_CHANNELS=%s)', n_plotted, SKIP_EDGE_CHANNELS)
    log.info('  freq range       : %.3f – %.3f MHz  (%.3f MHz bw)',
             diag['plot_freq_min_mhz'], diag['plot_freq_max_mhz'],
             diag['plot_freq_max_mhz'] - diag['plot_freq_min_mhz'])
    return diag


def step_4_propose_flags(q, diag):
    """Propose flag-table updates from diagnostic outliers."""
    log.info('── Step 4: flag proposals')
    proposal = q.propose_flag_updates_from_diagnostics(
        diag,
        outlier_metric=OUTLIER_METRIC,
        outlier_metric_merge_strategy=OUTLIER_METRIC_MERGE_STRATEGY,
        mode=FLAG_WHAT_TO_FLAG,
        antenna_flag_threshold_jy=ANTENNA_FLAG_THRESHOLD_JY,
        baseline_flag_threshold_jy=BASELINE_FLAG_THRESHOLD_JY,
        max_antennas_to_flag=MAX_ANTENNAS_TO_FLAG,
        max_baselines_to_flag=MAX_BASELINES_TO_FLAG,
    )
    ant   = proposal.get('candidate_antennas', [])
    bases = proposal.get('candidate_baselines', [])
    log.info('  candidate antennas  : %s', ', '.join(ant) if ant else '—')
    log.info('  candidate baselines : %s',
             ', '.join(f'{a}-{b}' for a, b in bases) if bases else '—')
    return proposal


def _iter_tags(prefix: str, start: int, width: int, n: int):
    """Yield n iteration tags from AUTO_ITER_* config settings.

    _iter_tags('iter', 1, 2, 5)  →  iter01, iter02, iter03, iter04, iter05
    _iter_tags('iter', 3, 2, 3)  →  iter03, iter04, iter05
    """
    for i in range(n):
        yield f'{prefix}{start + i:0{width}d}'


def _build_multipage_pdf_from_pngs(image_paths, pdf_path: Path, title_prefix: str = '') -> int:
    """Write a multi-page PDF from an ordered list of PNG files.

    Returns the number of pages written.
    """
    from matplotlib import pyplot as _plt
    from matplotlib.backends.backend_pdf import PdfPages as _PdfPages

    valid_paths = []
    for p in (image_paths or []):
        if p is None:
            continue
        _p = Path(p)
        if _p.exists():
            valid_paths.append(_p)

    if not valid_paths:
        return 0

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with _PdfPages(str(pdf_path)) as _pdf:
        for img_path in valid_paths:
            img = _plt.imread(str(img_path))
            if hasattr(img, 'shape') and len(img.shape) >= 2:
                h, w = int(img.shape[0]), int(img.shape[1])
            else:
                h, w = 900, 1400
            fig_w = max(6.0, min(14.0, w / 160.0))
            fig_h = max(4.0, min(10.0, h / 160.0))
            fig, ax = _plt.subplots(figsize=(fig_w, fig_h))
            ax.imshow(img)
            ax.axis('off')
            if title_prefix:
                fig.suptitle(f'{title_prefix} | {img_path.stem}', fontsize=10)
            _pdf.savefig(fig, bbox_inches='tight')
            _plt.close(fig)

    return len(valid_paths)


def _prompt_accept_flags(proposal: dict, dry_run: bool) -> str:
    """Show flag proposals and ask the user to accept / skip / quit.

    Returns 'y' (accept), 'n' (skip this iteration), or 'q' (quit loop).
    """
    ants  = proposal.get('candidate_antennas', [])
    bases = proposal.get('candidate_baselines', [])
    print()
    print('  ┌─ Flag proposals ──────────────────────────────────────────')
    print(f'  │  Antennas  : {", ".join(ants) if ants else "—"}')
    print(f'  │  Baselines : ')
    for a, b in bases:
        print(f'  │    {a} – {b}')
    if not bases:
        print('  │    —')
    if dry_run:
        print('  │  [dry-run: accepted flags stay in memory only — nothing written to disk]')
    else:
        print(f'  │  [writes: accepted flags will be appended to {FLAG_TABLE_SESSION}]')
    print('  └' + '─' * 52)
    while True:
        try:
            ans = input('  Accept and continue? [y = yes / n = skip / q = quit loop]: ').strip().lower()
        except EOFError:
            ans = 'q'
        if ans in ('y', 'n', 'q'):
            return ans
        print('  Please enter y, n, or q.')


def run_manual(args):
    """Run the interactive workflow.

    --step all (n_iters == 1)
        One pass: build index → solve bandpass → diagnostics (blocking plot)
        → propose flags.  Use --iter-tag to name the run.

    --step all --n-iters N
        N interactive iterations.  After each diagnostic plot you are prompted
        to accept or reject the proposed flags:
          y  – accepted flags accumulate in memory and are fed into the next
               iteration's solve.  Without --dry-run they are also written to
               FLAG_TABLE_SESSION on disk.
          n  – flags are skipped; the next iteration still solves with all
               previously accepted flags.
          q  – quit the loop early.
        With --dry-run nothing is written to disk; the loop is a
        safe scratch pad for tuning thresholds.
        Tags are auto-generated (iter01, iter02, …) from the config
        AUTO_ITER_PREFIX / AUTO_ITER_WIDTH and --start-iter.

    --step 1/2/3/4
        Run only that step once.  --iter-tag names the run.
        --n-iters is ignored for individual steps.
    """
    q       = _import_ugmrt()
    dry_run = args.dry_run
    steps   = {1, 2, 3, 4} if args.step == 'all' else {int(args.step)}

    # Log plot behaviour early so it's visible before the first iteration.
    import os
    _mpl = os.environ.get('MPLBACKEND', '(not set — matplotlib will use its default)')
    _interactive = _mpl.lower() not in ('agg', 'pdf', 'svg', 'ps', 'cairo')
    if _interactive:
        log.info('── Plot mode: MPLBACKEND=%s → popup windows ENABLED (blocking) + PNG saved to disk', _mpl)
        log.info('   To suppress popups: MPLBACKEND=Agg ./v-based-outlier-detection.sh')
    else:
        log.info('── Plot mode: MPLBACKEND=%s → no popup windows; PNG saved to disk only', _mpl)
        log.info('   To enable popups: MPLBACKEND=MacOSX ./v-based-outlier-detection.sh  (macOS)')
    if DIAG_PLOT_BASE is not None:
        log.info('   PNG location: %s', DIAG_PLOT_BASE.parent / f'{DIAG_PLOT_BASE.stem}_<iter>.png')

    # Multi-iteration loop only applies to --step all.
    n_iters = args.n_iters if args.step == 'all' else 1

    # Tag sequence.
    if n_iters > 1:
        tags = list(_iter_tags(AUTO_ITER_PREFIX, args.start_iter, AUTO_ITER_WIDTH, n_iters))
    else:
        tags = [args.iter_tag]

    # In-memory flag accumulator: accepted proposals from previous iterations.
    # Passed as flag_tables_extra to every subsequent solve so the solve sees
    # all accepted flags even in dry-run mode.
    accepted_flags: list = []
    _diag_png_paths: list = []

    # Build the row index once — reused across all iterations.
    index = None
    if steps & {1, 2, 3, 4}:
        index = step_1_index(q)

    for iter_num, iter_tag in enumerate(tags, 1):
        # Refresh FLAG_TABLE_PATHS so that FLAG_TABLE_SESSION created or
        # updated by the previous iteration is automatically included in
        # the current solve without requiring a script restart.
        _rederive_source_paths()
        _refresh_effective_flag_paths()
        if n_iters > 1:
            log.info('═' * 60)
            log.info('  Iteration %d / %d : %s', iter_num, n_iters, iter_tag)
            if accepted_flags:
                cum_ants  = sorted({a for ft in accepted_flags
                                    for a in ft.get('bad_antennas', [])})
                cum_bases = sorted({f"{b[0]}-{b[1]}" for ft in accepted_flags
                                    for b in ft.get('bad_baselines', [])})
                log.info('  Accumulated flags — antennas: %s  baselines: %s',
                         cum_ants or '—', cum_bases or '—')
            log.info('═' * 60)

        bandpass_run = None
        diag         = None
        proposal     = None

        if 2 in steps:
            bandpass_run = step_2_bandpass(
                q, index, iter_tag, dry_run,
                flag_tables_extra=accepted_flags if accepted_flags else None,
            )

        if 3 in steps:
            if bandpass_run is None:
                import numpy as np
                _bp_path = Path(BANDPASS_OUT)
                if CALVER != 'latest':
                    try:
                        _cands = [_bp_path]
                        _cands.extend(_bp_path.parent.glob(f'{_bp_path.stem}_*.npz'))
                        _bp_path = resolve_versioned_path(
                            candidates=_cands,
                            version=CALVER,
                            default_path=_bp_path,
                            label='bandpass table',
                        )
                    except ValueError as exc:
                        raise RuntimeError(str(exc)) from exc
                log.info('  [step 3] loading bandpass from %s', _bp_path)
                try:
                    bandpass_sol = q.load_bandpass_solution(_bp_path)
                except Exception:
                    raw = np.load(str(_bp_path), allow_pickle=True)
                    bandpass_sol = dict(raw)
                    if 'antenna_names' in bandpass_sol:
                        bandpass_sol['antenna_names'] = np.asarray(bandpass_sol['antenna_names']).astype(str).tolist()
                    if 'stokes_labels' in bandpass_sol:
                        bandpass_sol['stokes_labels'] = np.asarray(bandpass_sol['stokes_labels']).astype(str).tolist()
                    if 'reference_antenna' not in bandpass_sol:
                        _ant_ids = np.asarray(bandpass_sol.get('antenna_ids', []), dtype=np.int32)
                        bandpass_sol['reference_antenna'] = int(_ant_ids[0]) if _ant_ids.size else 0
                    bandpass_sol.setdefault('source_name', SOURCE)
                    bandpass_sol.setdefault('source_file', str(CAL_FITS))
            else:
                bandpass_sol = bandpass_run['solution']
            _bp_for_note = None
            if bandpass_run is not None:
                _bp_for_note = bandpass_run.get('bandpass_out')
            elif str(globals().get('DOCAL', 'on')).lower() == 'on':
                _bp_for_note = _bp_path
            diag = step_3_diagnostics(
                q, index, bandpass_sol, iter_tag,
                flag_tables_extra=accepted_flags if accepted_flags else None,
                cal_path=_bp_for_note,
            )
            _sp = diag.get('save_path') if isinstance(diag, dict) else None
            if _sp:
                _diag_png_paths.append(Path(_sp))

        if 4 in steps:
            if diag is None:
                raise RuntimeError('Step 4 requires diagnostics (step 3) to have run first.')
            proposal = step_4_propose_flags(q, diag)

        # Prompt to accept / skip / quit — only in multi-iteration loop.
        if n_iters > 1 and proposal is not None:
            ans = _prompt_accept_flags(proposal, dry_run)
            if ans == 'q':
                log.info('Loop quit by user after %s.', iter_tag)
                break
            elif ans == 'y':
                accepted_flags.append(proposal['proposal'])
                log.info('  Flags accepted — accumulated in memory (%d table(s) so far).',
                         len(accepted_flags))
                _cum_ft = {}
                for ft in accepted_flags:
                    _cum_ft.setdefault('bad_antennas', []).extend(ft.get('bad_antennas', []))
                    _cum_ft.setdefault('bad_baselines', []).extend(ft.get('bad_baselines', []))
                _sol = (bandpass_run or {}).get('solution', {})
                log.info(_data_coverage_summary(
                    index, _cum_ft,
                    n_obs_active_bls=_sol.get('obs_active_baselines'),
                    n_obs_active_ants=_sol.get('obs_active_antennas'),
                ))
                if not dry_run:
                    q.update_flag_table(
                        FLAG_TABLE_SESSION,
                        add_antennas=proposal['proposal'].get('bad_antennas', []),
                        add_baselines=proposal['proposal'].get('bad_baselines', []),
                        notes=f'Accepted interactively: {iter_tag}',
                        dry_run=False,
                    )
                    log.info('  Written to %s', FLAG_TABLE_SESSION)
            else:
                log.info('  Flags skipped for %s — next iteration uses same accumulated flags.',
                         iter_tag)

    if DIAG_PLOT_BASE is not None and _diag_png_paths:
        _diag_pdf = DIAG_PLOT_BASE.with_name(f'{DIAG_PLOT_BASE.stem}_all_iters.pdf')
        _n_pages = _build_multipage_pdf_from_pngs(
            _diag_png_paths,
            _diag_pdf,
            title_prefix=f'{SOURCE} diagnostics',
        )
        if _n_pages > 0:
            log.info('  multipage diagnostics PDF: %s  (%d page(s))', _diag_pdf, _n_pages)


def _run_final_clustering_step(q, workflow_result: dict, dry_run: bool) -> dict:
    """Run outlier_detection.run_clustering_detection after residual-loop convergence.

    Loads corrected visibilities using the last accumulated flag set and the
    final bandpass solution (both already in memory from the workflow result).
    Merges the per-channel clustering flags, optionally writes them to
    FLAG_TABLE_SESSION, then runs one final bandpass solve + diagnostic plot.

    Returns the dict returned by run_clustering_detection.
    """
    import importlib.util

    # ── Import outlier_detection from the same directory as this script ─────
    _od_path = Path(__file__).resolve().parent / 'outlier_detection.py'
    if not _od_path.exists():
        raise FileNotFoundError(
            f'outlier_detection.py not found next to preprocess_ugmrt.py: {_od_path}'
        )
    _spec = importlib.util.spec_from_file_location('outlier_detection', _od_path)
    od    = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(od)

    index       = workflow_result['index']
    bandpass_sol = workflow_result['last_bandpass_run']['solution']
    pending      = list(workflow_result.get('pending_flag_tables', []))

    session_path = Path(FLAG_TABLE_SESSION) if FLAG_TABLE_SESSION else None
    active_disk  = [Path(p) for p in (FLAG_TABLE_PATHS or [])]
    if session_path is not None and session_path.exists() and session_path not in active_disk:
        active_disk.append(session_path)

    # ── Build ant_name_map from row index ───────────────────────────────────
    ant_name_map = {
        int(a['antenna_no']): str(a['name'])
        for a in index.get('antennas', [])
    }

    # ── Load vis (final accumulated flags applied) ──────────────────────────
    # Build union of needed Stokes from all requested corrs
    _corr_list = [CLUSTERING_CORR] if isinstance(CLUSTERING_CORR, str) else list(CLUSTERING_CORR)
    needed_stokes: list = []
    for _c in _corr_list:
        for _s in od._CORR_STOKES_NEEDED[_c]:
            if _s not in needed_stokes:
                needed_stokes.append(_s)
    log.info('[final-clustering] Loading vis for %s (stokes=%s) ...', SOURCE, needed_stokes)
    vis_raw = q.load_vis_for_source(
        index,
        source           = SOURCE,
        chan_range        = CHAN_RANGE,
        stokes            = needed_stokes,
        max_rows          = MAX_ROWS_SOLVE,
        flag_all_corrs_if_any_rawvis_flagged = FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
        elevation_min_deg = SOLVE_ELEVATION_MIN_DEG,
    )
    if active_disk or pending:
        vis_raw, _flag_stats = q.apply_flag_tables_to_vis(
            vis_raw, ant_name_map,
            flag_table_paths=active_disk if active_disk else None,
            flag_tables=pending if pending else None,
        )
        log.info('[final-clustering] flags applied: dropped %d rows (%d kept)',
                 _flag_stats['dropped_rows'], _flag_stats['kept_rows'])
    vis_corr = q.apply_bandpass_solution(vis_raw, bandpass_sol)
    log.info('[final-clustering] Bandpass correction applied; shape=%s', vis_corr['amp'].shape)

    # ── Run clustering detection ────────────────────────────────────────────
    log.info('[final-clustering] Running per-channel clustering (corr=%s, thr=%s Jy) ...',
             CLUSTERING_CORR, CLUSTERING_THRESHOLD_JY)
    cluster_result = od.run_clustering_detection(
        vis_corr,
        ant_name_map,
        corr                            = CLUSTERING_CORR,
        threshold_jy                    = CLUSTERING_THRESHOLD_JY,
        threshold_low_jy                = CLUSTERING_THRESHOLD_LOW_JY,
        min_cluster_fraction            = CLUSTERING_MIN_CLUSTER_FRACTION,
        min_distinct_baselines_for_ant  = CLUSTERING_MIN_DISTINCT_BASELINES_FOR_ANT,
        min_burst_baseline_fraction     = CLUSTERING_MIN_BURST_BASELINE_FRACTION,
        whole_scan_bad_fraction         = CLUSTERING_WHOLE_SCAN_BAD_FRACTION,
        per_scan_ant_fraction           = CLUSTERING_PER_SCAN_ANT_FRACTION,
        per_scan_ant_bl_fraction        = CLUSTERING_PER_SCAN_ANT_BL_FRACTION,
        baseline_max_bad_samples        = CLUSTERING_BASELINE_MAX_BAD_SAMPLES,
        max_gap_samples                 = CLUSTERING_MAX_GAP_SAMPLES,
        max_gap_minutes                 = CLUSTERING_MAX_GAP_MINUTES,
        scan_gap_minutes                = CLUSTERING_SCAN_GAP_MINUTES,
        source                          = SOURCE,
        obs_jd_start                    = float(vis_raw['jd'].min()),
        obs_jd_end                      = float(vis_raw['jd'].max()),
        elevation_min_deg               = SOLVE_ELEVATION_MIN_DEG,
        verbose                         = True,
    )
    ft = cluster_result['flag_table']
    new_ants  = ft.get('bad_antennas', [])
    new_bases = ft.get('bad_baselines', [])
    log.info('[final-clustering] New flags — antennas: %s  baselines: %s',
             new_ants or '—', [f'{b[0]}-{b[1]}' for b in new_bases] or '—')

    # ── Persist to FLAG_TABLE_SESSION — ALL six flag-table sections ──────────
    # merge_flag_table_into_file merges bad_antennas, bad_baselines,
    # bad_antenna_timeranges, bad_baseline_timeranges, bad_scan_timeranges,
    # and bad_burst_timeranges into the session file, preserving flags
    # written by earlier iterations.  Unlike update_flag_table this does
    # NOT silently discard the chanrange-annotated per-channel entries
    # produced by run_clustering_detection.
    _has_any_flags = (
        ft.get('bad_antennas')
        or ft.get('bad_baselines')
        or ft.get('bad_antenna_timeranges')
        or ft.get('bad_baseline_timeranges')
        or ft.get('bad_scan_timeranges')
        or ft.get('bad_burst_timeranges')
    )
    if not dry_run and _has_any_flags:
        q.merge_flag_table_into_file(
            FLAG_TABLE_SESSION,
            ft,
            notes   = 'Final clustering stage (post-convergence)',
            dry_run = False,
        )
        log.info('[final-clustering] Full flag table (all 6 sections) written to %s',
                 FLAG_TABLE_SESSION)
        log.info('[final-clustering]   antennas         : %s', new_ants or '—')
        log.info('[final-clustering]   baselines        : %s',
                 [f'{b[0]}-{b[1]}' for b in new_bases] or '—')
        log.info('[final-clustering]   ant  time-ranges : %d key(s)',
                 len(ft.get('bad_antenna_timeranges', {})))
        log.info('[final-clustering]   bl   time-ranges : %d key(s)',
                 len(ft.get('bad_baseline_timeranges', {})))
        log.info('[final-clustering]   scan time-ranges : %d',
                 len(ft.get('bad_scan_timeranges', [])))
        log.info('[final-clustering]   burst time-ranges: %d',
                 len(ft.get('bad_burst_timeranges', [])))
    elif dry_run:
        log.info('[final-clustering] dry-run — flag table NOT written')

    # ── Final bandpass solve with clustering flags active ──────────────────
    final_tag = f'{AUTO_ITER_PREFIX}final_clustering'
    log.info('[final-clustering] Final bandpass solve tagged %s ...', final_tag)
    # Include the new clustering flag table in-memory so it applies even
    # in dry-run mode (we still want to see the corrected solution).
    final_pending = pending + [ft]
    final_bp = q.derive_bandpass_iteration(
        fits_path        = CAL_FITS,
        index            = index,
        bandpass_out     = BANDPASS_OUT,
        source           = SOURCE,
        stokes           = STOKES,
        chan_range        = CHAN_RANGE,
        max_rows         = MAX_ROWS_SOLVE,
        smooth_window    = SMOOTH_WINDOW,
        min_baselines    = MIN_BASELINES,
        flag_table_path  = active_disk if active_disk else None,
        flag_table       = final_pending,
        flag_all_corrs_if_any_rawvis_flagged = FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
        iteration_tag    = final_tag,
        dry_run          = dry_run,
        elevation_min_deg = SOLVE_ELEVATION_MIN_DEG,
        elevation_max_deg = SOLVE_ELEVATION_MAX_DEG,
        uvrange_m         = SOLVE_UVRANGE_M,
        uvrange_klambda   = SOLVE_UVRANGE_KLAMBDA,
        timerange         = SOLVE_TIMERANGE,
    )

    # ── Final diagnostic plot ───────────────────────────────────────────────
    if DIAG_PLOT_BASE is not None:
        _plot_path = q.tagged_output_path(DIAG_PLOT_BASE, final_tag)
        log.info('[final-clustering] Final diagnostic plot → %s', _plot_path)
        import matplotlib.pyplot as _plt
        _final_note = format_table_application_note(
            docal='on',
            doflag=DOFLAG,
            calver='iterative-final',
            flagver=FLAGVER,
            cal_path=final_bp.get('bandpass_out'),
            flag_paths=active_disk,
        )
        _fig = q.run_bandpass_diagnostics(
            index,
            final_bp['solution'],
            source           = SOURCE,
            chan_range        = CHAN_RANGE,
            stokes           = STOKES,
            max_rows         = MAX_ROWS_DIAG,
            elevation_min_deg = SOLVE_ELEVATION_MIN_DEG,
            exclude_antennas  = EXCLUDE_FOR_PLOTS,
            apply_flag_tables = DIAG_APPLY_FLAGS_ON_THE_FLY,
            flag_table_path   = active_disk if active_disk else None,
            flag_table        = final_pending,
            flag_all_corrs_if_any_rawvis_flagged = FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
            skip_edge_channels = SKIP_EDGE_CHANNELS,
            top_n              = TOP_N,
            title              = f'{SOURCE} FINAL — post-clustering | {final_tag}{_final_note}',
            save_path          = _plot_path,
        )
        log.info('[final-clustering] tables applied: %s', _final_note.strip())
        if _fig is not None:
            _plt.close(_fig)

    return cluster_result


def run_auto(args):
    """Run the automated iterative workflow."""
    q = _import_ugmrt()
    dry_run_bp   = args.dry_run
    dry_run_flag = args.dry_run

    if str(globals().get('DOCAL', 'on')).lower() == 'off':
        log.warning('DOCAL=off requested, but auto workflow diagnostics currently use corrected vis; '
                    'DOCAL is only honored in step_3_diagnostics/manual path.')

    log.info('── Auto workflow: %d iterations', args.n_iters)
    log.info('  dry_run_bandpass=%s  dry_run_flag_write=%s', dry_run_bp, dry_run_flag)

    result = q.run_iterative_bandpass_workflow(
        fits_path=CAL_FITS,
        index_cache_path=INDEX_CACHE,
        index_validation_mode=INDEX_VALIDATION_MODE,
        write_index_cache=True,
        bandpass_out_base=BANDPASS_OUT,
        diag_plot_base=DIAG_PLOT_BASE,
        diag_plot_unflagged_base=DIAG_PLOT_UNFLAGGED,
        gain_plot_base=GAIN_PLOT_BASE,
        flag_table_session_path=FLAG_TABLE_SESSION,
        base_flag_table_paths=_effective_flag_table_paths(),
        pending_flag_tables=[],
        use_pending_flag_tables=True,
        start_iteration=AUTO_START_ITER,
        n_iterations=args.n_iters,
        iter_prefix=AUTO_ITER_PREFIX,
        iter_width=AUTO_ITER_WIDTH,
        dry_run_bandpass=dry_run_bp,
        dry_run_flag_write=dry_run_flag,
        source=SOURCE,
        stokes=STOKES,
        chan_range=CHAN_RANGE,
        max_rows_solve=MAX_ROWS_SOLVE,
        smooth_window=SMOOTH_WINDOW,
        min_baselines=MIN_BASELINES,
        max_rows_diag=MAX_ROWS_DIAG,
        exclude_for_plots=EXCLUDE_FOR_PLOTS,
        diag_apply_flag_tables=DIAG_APPLY_FLAGS_ON_THE_FLY,
        diag_save_unflagged_comparison=DIAG_SAVE_UNFLAGGED_COMPARISON,
        flag_all_corrs_if_any_rawvis_flagged=FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
        skip_edge_channels=SKIP_EDGE_CHANNELS,
        top_n=TOP_N,
        outlier_metric=OUTLIER_METRIC,
        outlier_metric_merge_strategy=OUTLIER_METRIC_MERGE_STRATEGY,
        proposal_mode=FLAG_WHAT_TO_FLAG,
        antenna_flag_threshold_jy=ANTENNA_FLAG_THRESHOLD_JY,
        baseline_flag_threshold_jy=BASELINE_FLAG_THRESHOLD_JY,
        max_antennas_to_flag=MAX_ANTENNAS_TO_FLAG,
        max_baselines_to_flag=MAX_BASELINES_TO_FLAG,
        strict_flag_table=False,
        convergence_epsilon=CONVERGENCE_EPSILON,
        convergence_min_iters=CONVERGENCE_MIN_ITERS,
        compare_metrics_for_convergence=COMPARE_METRICS_FOR_CONVERGENCE,
        convergence_combine_strategy=CONVERGENCE_COMBINE_STRATEGY,
        run_iter0_diagnostic=RUN_ITER0_DIAGNOSTIC,
        elevation_min_deg=SOLVE_ELEVATION_MIN_DEG,
        elevation_max_deg=SOLVE_ELEVATION_MAX_DEG,
        uvrange_m=SOLVE_UVRANGE_M,
        uvrange_klambda=SOLVE_UVRANGE_KLAMBDA,
        timerange=SOLVE_TIMERANGE,
        table_note_suffix=format_table_application_note(
            docal='on',
            doflag=DOFLAG,
            calver='iterative',
            flagver=FLAGVER,
            cal_path=None,
            flag_paths=_effective_flag_table_paths(),
        ),
    )

    _index = result.get('index', {})
    for h in result['history']:
        ants  = h['candidate_antennas']
        bases = h['candidate_baselines']
        log.info(
            '  %s: ant(%d)=[%s]  base(%d)=[%s]  cum_ant=%d  cum_base=%d',
            h['iteration_tag'],
            len(ants), ', '.join(ants) or '—',
            len(bases), ', '.join(f'{a}-{b}' for a, b in bases) or '—',
            h['cumulative_bad_antenna_count'],
            h['cumulative_bad_baseline_count'],
        )
    # Final data coverage summary
    _final_ft = result.get('last_flag_update', {}).get('flag_table', {})
    _last_sol = (result.get('last_bandpass_run') or {}).get('solution', {})
    if _index and _final_ft:
        log.info(_data_coverage_summary(
            _index, _final_ft,
            n_obs_active_bls=_last_sol.get('obs_active_baselines'),
            n_obs_active_ants=_last_sol.get('obs_active_antennas'),
        ))
    log.info('Iterations executed : %d  (of %d requested)', len(result['history']), args.n_iters)
    log.info('Stop reason         : %s', result.get('stop_reason', 'unknown'))

    # Build one multi-page diagnostics PDF across all iterations.
    if DIAG_PLOT_BASE is not None:
        _diag_pngs = []
        if RUN_ITER0_DIAGNOSTIC:
            _iter0_tag = f'{AUTO_ITER_PREFIX}00'
            _diag_pngs.append(q.tagged_output_path(DIAG_PLOT_BASE, _iter0_tag))
        _diag_pngs.extend(
            Path(h['diagnostics_plot_path'])
            for h in result.get('history', [])
            if h.get('diagnostics_plot_path')
        )
        _diag_pdf = DIAG_PLOT_BASE.with_name(f'{DIAG_PLOT_BASE.stem}_all_iters.pdf')
        _n_pages = _build_multipage_pdf_from_pngs(
            _diag_pngs,
            _diag_pdf,
            title_prefix=f'{SOURCE} diagnostics',
        )
        if _n_pages > 0:
            log.info('Multipage diagnostics PDF: %s  (%d page(s))', _diag_pdf, _n_pages)

    # ── Persist final iteration bandpass to BANDPASS_OUT (no iter suffix) ────
    # run_iterative_bandpass_workflow writes <BASE>_iter01.npz, _iter02.npz, …
    # but never the base path itself.  Copy the last iteration's file to
    # BANDPASS_OUT so that a subsequent --phase=audit / run_clustering.py step
    # finds the file where it expects it.
    # (If RUN_FINAL_CLUSTERING below re-derives and overwrites BANDPASS_OUT,
    # that polished version takes precedence — the copy is just a safe default.)
    if not dry_run_bp and result['history']:
        import shutil as _shutil
        _last_tag = result['history'][-1]['iteration_tag']
        _last_bp  = (BANDPASS_OUT.parent
                     / f'{BANDPASS_OUT.stem}_{_last_tag}{BANDPASS_OUT.suffix}')
        if _last_bp.exists():
            _shutil.copy2(str(_last_bp), str(BANDPASS_OUT))
            log.info('Final bandpass  : %s  →  %s  (ready for audit phase)',
                     _last_bp.name, BANDPASS_OUT.name)
        else:
            log.warning(
                'Last-iter bandpass %s not found on disk — BANDPASS_OUT not updated. '
                'Check that dry_run_bandpass=False reached the library.',
                _last_bp,
            )

    # ── Optional post-convergence clustering step ────────────────────────────
    if RUN_FINAL_CLUSTERING:
        _stop = result.get('stop_reason', '')
        _converged = 'C1 —' in _stop or 'Epsilon convergence' in _stop
        if _converged:
            log.info('[final-clustering] Convergence confirmed — starting clustering stage.')
            _run_final_clustering_step(q, result, args.dry_run)
        else:
            log.warning(
                '[final-clustering] RUN_FINAL_CLUSTERING=True but loop ended without '
                'convergence ("%s"). Clustering step skipped. '
                'Increase AUTO_N_ITERS or CONVERGENCE_EPSILON and re-run.', _stop
            )


def main():
    # Parse --config and --set first so overrides are available before full parse.
    _default_cfg = str(Path(__file__).resolve().parent / 'preprocess_ugmrt.cfg')
    pre_args = bootstrap_config_from_cli(_default_cfg, globals(), log=log)
    _rederive_source_paths()   # re-sync output paths if SOURCE was overridden

    parser = argparse.ArgumentParser(
        description='uGMRT bandpass calibration workflow driver.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_config_set_arguments(
        parser,
        config_default=pre_args.config,
        config_help='Path to config file (default: preprocess_ugmrt.cfg next to this script).',
        set_help=(
            'Override a config key with a Python expression, e.g. '
            '--set "SOURCE=\'3C147\'" or --set "CHAN_RANGE=(64,191)". '
            'Path values: --set "CAL_FITS=Path(\'/data/file.FITS\')". '
            'Repeatable.  Applied after the config file is loaded.'
        ),
        config_metavar='FILE',
        set_metavar='KEY=expr',
    )
    parser.add_argument(
        '--step',
        default='all',
        choices=['1', '2', '3', '4', 'all'],
        help=(
            'Which step(s) to run: '
            '1=build index  2=solve bandpass  3=diagnostics+plot  4=propose flags  '
            'all=steps 1-4.  '
            'Combine with --auto to run the full automated iterative loop.  '
            '(default: all)'
        ),
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        default=False,
        help=(
            'Batch mode: fully unattended iterative loop with progressive flag '
            'accumulation (requires --step all). No interactive prompts or blocking '
            'plots. Diagnostic plots are saved to disk as image files each iteration. '
            'Bandpass solutions and flag tables are written unless --dry-run is set. '
            'Without --auto, --step all runs an interactive loop that shows a blocking '
            'plot and prompts for flag acceptance after each pass.'
        ),
    )
    parser.add_argument(
        '--iter-tag',
        default=ITER_TAG,
        metavar='TAG',
        help=(
            f'Label used in plot titles and log filenames. '
            f'Used for single-pass runs (--step all --n-iters 1) or individual steps. '
            f'Ignored when --n-iters > 1 (tags are auto-generated from '
            f'AUTO_ITER_PREFIX/AUTO_ITER_WIDTH + --start-iter). '
            f'(default from config: {ITER_TAG})'
        ),
    )
    parser.add_argument(
        '--n-iters',
        type=int,
        default=1,
        metavar='N',
        help=(
            f'Number of iterations. '
            f'Without --auto: interactive loop — blocking plot and accept/reject '
            f'prompt after each pass; flags accumulate in memory (--dry-run) or are '
            f'also written to FLAG_TABLE_SESSION (default). '
            f'With --auto: fully unattended batch loop. '
            f'(default: 1 for interactive; set AUTO_N_ITERS in config for --auto)'
        ),
    )
    parser.add_argument(
        '--start-iter',
        type=int,
        default=AUTO_START_ITER,
        metavar='N',
        help=(
            f'Starting iteration number. '
            f'--n-iters 3 --start-iter 1 → iter01, iter02, iter03. '
            f'--n-iters 3 --start-iter 4 → iter04, iter05, iter06. '
            f'(default from config: AUTO_START_ITER={AUTO_START_ITER})'
        ),
    )
    parser.add_argument(
        '--dry-run',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Preview only — do not write bandpass solutions or flag tables to disk. '
            'Use --no-dry-run to override a hardcoded --dry-run (e.g. from a '
            'wrapper script that defaults to dry-run for safety).'
        ),
    )
    parser.add_argument(
        '--docal', choices=['on', 'off'], default=str(globals().get('DOCAL', 'on')),
        help='Apply calibration table in diagnostics where applicable (AIPS-like DOCAL).',
    )
    parser.add_argument(
        '--doflag', choices=['on', 'off'], default=str(globals().get('DOFLAG', 'on')),
        help='Apply on-disk flag table(s) before solve/diagnostics (AIPS-like DOFLAG).',
    )
    parser.add_argument(
        '--calver', default=str(globals().get('CALVER', 'latest')),
        metavar='VER',
        help='Calibration table selector for standalone diagnostics: latest | N | tag | /path/to/table.npz',
    )
    parser.add_argument(
        '--flagver', default=str(globals().get('FLAGVER', 'latest')),
        metavar='VER',
        help='Flag table selector: latest | N | tag | /path/to/table.json',
    )

    add_log_level_argument(
        parser,
        default=None,  # None → read from config LOG_LEVEL
        help_text='Logging level: DEBUG | INFO | WARNING | ERROR  (default from config)',
        metavar='LEVEL',
    )

    args = parser.parse_args()

    # Re-apply overrides in case full parser reset anything (it doesn't, but
    # this keeps the effective globals consistent with what was pre-parsed).
    _apply_overrides(args.set_overrides)

    global DOCAL, DOFLAG, CALVER, FLAGVER
    DOCAL = str(args.docal).lower()
    DOFLAG = str(args.doflag).lower()
    CALVER = str(args.calver)
    FLAGVER = str(args.flagver)

    try:
        _refresh_effective_flag_paths()
    except ValueError as exc:
        sys.exit(f'ERROR: {exc}')

    # --auto: default n_iters to AUTO_N_ITERS from config if user didn't supply it.
    if args.auto and args.n_iters == 1:
        args.n_iters = AUTO_N_ITERS

    # Derive index cache now that CAL_FITS and WORK_DIR are known.
    global INDEX_CACHE
    INDEX_CACHE = _derive_index_cache()

    # Resolve effective log level: CLI wins over config.
    effective_log_level = (args.log_level or LOG_LEVEL or 'INFO').upper()
    args.log_level = effective_log_level

    # Shared timestamp ties the log file and config snapshot together.
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Log filename uses the first iteration tag (or iter-tag for single runs).
    if args.n_iters > 1:
        first_tag = f'{AUTO_ITER_PREFIX}{args.start_iter:0{AUTO_ITER_WIDTH}d}'
    else:
        first_tag = args.iter_tag
    log_dir = _setup_logging(Path(WORK_DIR), first_tag, effective_log_level, timestamp)
    _save_config_snapshot(log_dir, timestamp, args.config, args.set_overrides)
    _log_run_parameters(args.config, args)
    log.info('DOCAL/DOFLAG  : %s / %s   CALVER/FLAGVER: %s / %s', DOCAL, DOFLAG, CALVER, FLAGVER)

    if args.auto:
        if args.step != 'all':
            parser.error('--auto requires --step all')
        run_auto(args)
    else:
        run_manual(args)

    log.info('Done.')


if __name__ == '__main__':
    main()
