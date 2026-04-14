#!/usr/bin/env python3
"""run_clustering.py — Phase-2 standalone clustering detection.

Loads the saved bandpass solution and accumulated flag tables produced by
Phase-1 (v-based-outlier-detection.sh), then runs per-channel clustering
detection and shows six before/after vis-amplitude plots for interactive
threshold tuning — replicating notebook cell 15F as a script.

Default mode: DRY-RUN.  Plots are shown without writing any new flags to
disk.  Safe for exploring CLUSTERING_THRESHOLD_JY repeatedly.

Usage
─────
    ./run_clustering.sh                           # dry-run, interactive plots
    ./run_clustering.sh --set "CLUSTERING_THRESHOLD_JY=3.0"
    ./run_clustering.sh --set "SOURCE='3C286'"
    ./run_clustering.sh --config /path/to/other.cfg
    ./run_clustering.sh --no-dry-run              # write clustering flags to FLAG_TABLE_SESSION
    ./run_clustering.sh --refit                   # also re-solve bandpass with clustering flags
    ./run_clustering.sh --save-plots              # save 6 PNGs to WORK_DIR

    # Identical interface — direct Python invocation also works:
    python run_clustering.py --set "CLUSTERING_THRESHOLD_JY=3.0"

Description of the six plots
─────────────────────────────
    BEFORE (3 plots): corrected vis using saved BANDPASS_OUT, no clustering flags
        1. Amp vs UV-dist (RR + LL)
        2. Vector-avg spectrum vs PB2017 model
        3. Stokes-V amp vs UV-dist  (threshold line shown)

    AFTER (3 plots): same corrected vis but with clustered rows masked out
        4. Amp vs UV-dist (RR + LL)
        5. Vector-avg spectrum vs PB2017 model
        6. Stokes-V amp vs UV-dist
    
    Without --refit both BEFORE and AFTER use the same saved BANDPASS_OUT
    solution (fast; good for threshold exploration).  Add --refit to
    re-solve the bandpass with clustering flags active (slower but shows
    the effect on the solution itself).
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

# ── locate siblings robustly regardless of cwd ───────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import ugmrt_query as q  # noqa: E402

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level config globals — populated by _load_config() before main() runs.
# Defaults here are only fallbacks; the .cfg file is the real source of truth.
# ─────────────────────────────────────────────────────────────────────────────

# Paths (derived/None before config is loaded)
CAL_FITS:          Path | None  = None
INDEX_CACHE:       Path | None  = None   # explicit override; derived otherwise
BANDPASS_OUT:      Path | None  = None
WORK_DIR:          Path         = Path('.')
DIAG_PLOT_BASE:    Path | None  = None

# Observation / selection
SOURCE:                              str   = '3C48'
STOKES:                              tuple = ('RR', 'LL')
CHAN_RANGE:                          tuple = (0, 255)
PLOT_CHAN_RANGE:                      tuple | None = None
MAX_ROWS_SOLVE:                      int   = 500_000
MAX_ROWS_DIAG:                       int   = 500_000
SMOOTH_WINDOW:                       int   = 5
MIN_BASELINES:                       int   = 10
FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED: bool  = True
SOLVE_ELEVATION_MIN_DEG:             float = 20.0
SOLVE_ELEVATION_MAX_DEG:             float = 90.0
SOLVE_UVRANGE_M:                     tuple | None = None
SOLVE_UVRANGE_KLAMBDA:               tuple | None = None
SOLVE_TIMERANGE:                     tuple | None = None
SKIP_EDGE_CHANNELS:                  tuple = (0, 0)
EXCLUDE_FOR_PLOTS:                   list  = []
DUD_ANTENNA_NAMES:                   list | None = None

# Flag tables
FLAG_TABLE_PATHS:   list        = []
FLAG_TABLE_SESSION: Path | None = None
INDEX_VALIDATION_MODE: str      = 'warn'

# Iteration labelling (used for the optional refit tag)
AUTO_ITER_PREFIX: str = 'iter'

# Clustering (all params mirror preprocess_ugmrt.cfg defaults)
CLUSTERING_CORR:                           Union[str, List]   = 'V'
CLUSTERING_THRESHOLD_JY:                   Union[float, dict] = 5.0
CLUSTERING_THRESHOLD_LOW_JY:              Union[float, dict, None] = None
CLUSTERING_MIN_CLUSTER_FRACTION:           float = 0.80
CLUSTERING_MIN_DISTINCT_BASELINES_FOR_ANT: int   = 3
CLUSTERING_MIN_BURST_BASELINE_FRACTION:    float = 0.50
CLUSTERING_WHOLE_SCAN_BAD_FRACTION:        float = 0.70
CLUSTERING_PER_SCAN_ANT_FRACTION:          float = 0.60
CLUSTERING_PER_SCAN_ANT_BL_FRACTION:       float = 0.50
CLUSTERING_BASELINE_MAX_BAD_SAMPLES:       int   = 100
CLUSTERING_MAX_GAP_SAMPLES:                int   = 2
CLUSTERING_MAX_GAP_MINUTES:                float = 30.0
CLUSTERING_SCAN_GAP_MINUTES:               float = 2.0

LOG_LEVEL: str = 'INFO'

# ─────────────────────────────────────────────────────────────────────────────
# Config helpers — deliberately mirrors preprocess_ugmrt.py for consistency
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(config_path: str) -> None:
    """Exec the config file and inject every public name into module globals."""
    path = Path(config_path).resolve()
    if not path.exists():
        sys.exit(f'ERROR: config file not found: {path}')
    ns: dict = {}
    with open(path) as fh:
        exec(compile(fh.read(), str(path), 'exec'), ns)  # noqa: S102
    g = globals()
    for key, val in ns.items():
        if not key.startswith('_'):
            g[key] = val


def _apply_overrides(overrides: list) -> None:
    """Apply --set KEY=expr overrides to module globals (same contract as preprocess_ugmrt.py)."""
    if not overrides:
        return
    from pathlib import Path as _Path
    g = globals()
    for item in overrides:
        if '=' not in item:
            sys.exit(f'ERROR: --set requires KEY=VALUE format, got: {item!r}')
        key, _, expr = item.partition('=')
        key = key.strip()
        if not key.isidentifier():
            sys.exit(f'ERROR: --set key must be a valid Python identifier, got: {key!r}')
        try:
            val = eval(expr.strip(), {'Path': _Path, '__builtins__': __builtins__})  # noqa: S307
        except Exception as exc:
            sys.exit(f'ERROR: could not evaluate --set {key}={expr!r}: {exc}')
        g[key] = val
        log.debug('  --set %s = %r', key, val)


def _derive_index_cache() -> Path:
    """Mirror preprocess_ugmrt._derive_index_cache: <WORK_DIR>/<cal_fits_stem>.index.npz."""
    if INDEX_CACHE is not None:
        return Path(INDEX_CACHE)
    stem = Path(CAL_FITS).stem
    base = Path(WORK_DIR) if WORK_DIR is not None else Path(CAL_FITS).parent
    return base / f'{stem}.index.npz'


def _load_outlier_detection():
    """Import outlier_detection.py from the same directory as this script."""
    od_path = _SCRIPT_DIR / 'outlier_detection.py'
    if not od_path.exists():
        raise FileNotFoundError(f'outlier_detection.py not found at {od_path}')
    spec = importlib.util.spec_from_file_location('outlier_detection', od_path)
    od   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(od)
    return od


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _default_cfg = str(_SCRIPT_DIR / 'preprocess_ugmrt.cfg')

    # ── pre-parse: load config + overrides before the full parser sees defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=_default_cfg)
    pre.add_argument('--set', dest='set_overrides', action='append', default=[])
    pre_args, _ = pre.parse_known_args()
    _load_config(pre_args.config)
    _apply_overrides(pre_args.set_overrides)

    # ── full parser ───────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        prog='run_clustering.py',
        description=(
            'Phase-2 clustering detection.  '
            'Loads saved bandpass + flags from Phase-1 and shows BEFORE/AFTER '
            'vis-amplitude plots.  Default: dry-run (plots only, nothing written).'
        ),
    )
    parser.add_argument(
        '--config', default=_default_cfg,
        help='Path to .cfg file (default: preprocess_ugmrt.cfg next to this script)',
    )
    parser.add_argument(
        '--set', dest='set_overrides', action='append', default=[],
        metavar='KEY=expr',
        help=(
            'Override any config key after loading.  E.g.:\n'
            '  --set "CLUSTERING_THRESHOLD_JY=3.0"\n'
            '  --set "SOURCE=\'3C286\'"\n'
            '  --set "PLOT_CHAN_RANGE=(100,180)"\n'
            '  --set "SOLVE_ELEVATION_MIN_DEG=25.0"'
        ),
    )
    parser.add_argument(
        '--dry-run',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Default: dry-run — plots only, nothing written to disk. '
            'Use --no-dry-run to write clustering flags to FLAG_TABLE_SESSION.'
        ),
    )
    parser.add_argument(
        '--refit', action='store_true', default=False,
        help='Re-solve the bandpass with clustering flags active and use that '
             'solution for the AFTER plots.  Without this flag both BEFORE and '
             'AFTER use the same saved BANDPASS_OUT solution (faster for threshold '
             'exploration).',
    )
    parser.add_argument(
        '--save-plots', action='store_true', default=False,
        help='Save the six plots as PNGs to WORK_DIR.',
    )
    parser.add_argument(
        '--log-level', default=None,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging verbosity (default from config LOG_LEVEL or INFO).',
    )
    args = parser.parse_args()
    _apply_overrides(args.set_overrides)

    # ── logging ───────────────────────────────────────────────────────────────
    effective_level = args.log_level or globals().get('LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=getattr(logging, effective_level, logging.INFO),
        format='%(asctime)s  %(levelname)-8s  %(message)s',
    )

    dry_run = args.dry_run
    if dry_run:
        log.info('DRY-RUN mode — clustering flags will NOT be written to disk')
    else:
        log.info('REAL RUN — clustering flags will be written to %s',
                 FLAG_TABLE_SESSION)

    log.info('Source        : %s', SOURCE)
    log.info('CLUSTERING_CORR=%s  threshold=%s Jy  threshold_low=%s Jy',
             CLUSTERING_CORR, CLUSTERING_THRESHOLD_JY, CLUSTERING_THRESHOLD_LOW_JY)

    od = _load_outlier_detection()

    # ── resolve active disk flag tables (same logic as preprocess_ugmrt.py) ──
    session_path = Path(FLAG_TABLE_SESSION) if FLAG_TABLE_SESSION else None
    active_disk  = [Path(p) for p in (FLAG_TABLE_PATHS or [])]
    if session_path is not None and session_path.exists() and session_path not in active_disk:
        active_disk.append(session_path)
    log.info('Active disk flag tables: %s', active_disk or '(none)')

    # ── load row index ────────────────────────────────────────────────────────
    if CAL_FITS is None:
        sys.exit('ERROR: CAL_FITS is not set — check your .cfg file or pass '
                 '--set "CAL_FITS=Path(\'/path/to/file.FITS\')"')
    cache = _derive_index_cache()
    log.info('Loading row index (cache=%s) ...', cache)
    index = q.get_or_build_row_index(
        CAL_FITS,
        cache_path           = cache,
        force_rebuild        = False,
        validation_mode      = INDEX_VALIDATION_MODE,
        write_cache          = True,
        override_dud_names   = DUD_ANTENNA_NAMES or None,
    )
    ant_name_map = {
        int(a['antenna_no']): str(a['name'])
        for a in index.get('antennas', [])
    }
    log.info('  active antennas: %d', len(index.get('active_antennas', index['antennas'])))

    # ── load saved bandpass solution ──────────────────────────────────────────
    bp_path = Path(BANDPASS_OUT) if BANDPASS_OUT else None
    if bp_path is None or not bp_path.exists():
        sys.exit(f'ERROR: BANDPASS_OUT not found: {bp_path}  '
                 '(Has Phase-1 run and committed at least one iteration?)')
    log.info('Loading bandpass solution from %s ...', bp_path)
    bandpass_sol = q.load_bandpass_solution(bp_path)

    # ── load vis (Phase-1 flags already on disk, applied here) ───────────────
    # Build union of needed Stokes from all requested corrs
    _corr_list = [CLUSTERING_CORR] if isinstance(CLUSTERING_CORR, str) else list(CLUSTERING_CORR)
    needed_stokes: list = []
    for _c in _corr_list:
        for _s in od._CORR_STOKES_NEEDED[_c]:
            if _s not in needed_stokes:
                needed_stokes.append(_s)
    log.info('Loading vis for %s (stokes=%s) ...', SOURCE, needed_stokes)
    vis_raw = q.load_vis_for_source(
        index,
        source            = SOURCE,
        chan_range         = CHAN_RANGE,
        stokes             = needed_stokes,
        max_rows           = MAX_ROWS_SOLVE,
        flag_all_corrs_if_any_rawvis_flagged = FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
        elevation_min_deg  = SOLVE_ELEVATION_MIN_DEG,
    )
    if active_disk:
        vis_raw, _flag_stats = q.apply_flag_tables_to_vis(
            vis_raw, ant_name_map, flag_table_paths=active_disk,
        )
        log.info('  Phase-1 flags applied: dropped %d rows (%d kept)',
                 _flag_stats['dropped_rows'], _flag_stats['kept_rows'])
    vis_corr = q.apply_bandpass_solution(vis_raw, bandpass_sol)
    log.info('Vis loaded; shape=%s', vis_corr['amp'].shape)

    # ── run clustering detection ──────────────────────────────────────────────
    log.info('Running per-channel clustering (corr=%s, thr=%s Jy) ...',
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
    ft_new   = cluster_result['flag_table']
    new_ants = ft_new.get('bad_antennas', [])
    new_bases = ft_new.get('bad_baselines', [])
    log.info('New flags — antennas: %s  baselines: %s',
             new_ants or '—', [f'{b[0]}-{b[1]}' for b in new_bases] or '—')

    # ── print_summary + CASA commands ─────────────────────────────────────────
    print()
    od.print_summary(ft_new)
    print()
    print(od.format_casa_commands(ft_new))
    print()

    # ── optionally commit clustering flags to disk — ALL six sections ─────────
    # merge_flag_table_into_file persists bad_antennas, bad_baselines,
    # bad_antenna_timeranges, bad_baseline_timeranges, bad_scan_timeranges,
    # and bad_burst_timeranges.  update_flag_table (old code) only persisted
    # bad_antennas + bad_baselines, silently discarding all time-range and
    # chanrange entries — causing a re-run at --phase=audit to see the same
    # RFIs again because those flags were never on disk.
    _has_any_flags = (
        new_ants
        or new_bases
        or ft_new.get('bad_antenna_timeranges')
        or ft_new.get('bad_baseline_timeranges')
        or ft_new.get('bad_scan_timeranges')
        or ft_new.get('bad_burst_timeranges')
    )
    if not dry_run and _has_any_flags:
        q.merge_flag_table_into_file(
            FLAG_TABLE_SESSION,
            ft_new,
            notes   = (
                f'Clustering detection — corr={CLUSTERING_CORR} '
                f'thr={CLUSTERING_THRESHOLD_JY} Jy'
            ),
            dry_run = False,
        )
        log.info('Clustering flags (all 6 sections) written to %s', FLAG_TABLE_SESSION)
        log.info('  antennas         : %s', new_ants or '—')
        log.info('  baselines        : %s',
                 [f'{b[0]}-{b[1]}' for b in new_bases] or '—')
        log.info('  ant  time-ranges : %d key(s)',
                 len(ft_new.get('bad_antenna_timeranges', {})))
        log.info('  bl   time-ranges : %d key(s)',
                 len(ft_new.get('bad_baseline_timeranges', {})))
        log.info('  scan time-ranges : %d',
                 len(ft_new.get('bad_scan_timeranges', [])))
        log.info('  burst time-ranges: %d',
                 len(ft_new.get('bad_burst_timeranges', [])))
    elif _has_any_flags:
        log.info('dry-run — clustering flags NOT written to disk  '
                 '(re-run with --no-dry-run to persist)')
    else:
        log.info('No new clustering flags proposed.')

    # ── optional bandpass refit with clustering flags ─────────────────────────
    if args.refit:
        _refit_tag = f'{AUTO_ITER_PREFIX}_final_clustering'
        log.info('Post-clustering bandpass re-solve (tag=%s) ...', _refit_tag)
        final_bp = q.derive_bandpass_iteration(
            fits_path         = CAL_FITS,
            index             = index,
            bandpass_out      = BANDPASS_OUT,
            source            = SOURCE,
            stokes            = STOKES,
            chan_range         = CHAN_RANGE,
            max_rows          = MAX_ROWS_SOLVE,
            smooth_window     = SMOOTH_WINDOW,
            min_baselines     = MIN_BASELINES,
            flag_table_path   = active_disk if active_disk else None,
            flag_table        = [ft_new],
            flag_all_corrs_if_any_rawvis_flagged = FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
            iteration_tag     = _refit_tag,
            dry_run           = True,   # never overwrite BANDPASS_OUT from here
            elevation_min_deg = SOLVE_ELEVATION_MIN_DEG,
            elevation_max_deg = SOLVE_ELEVATION_MAX_DEG,
            uvrange_m         = SOLVE_UVRANGE_M,
            uvrange_klambda   = SOLVE_UVRANGE_KLAMBDA,
            timerange         = SOLVE_TIMERANGE,
        )
        bandpass_sol_after = final_bp['solution']
        _after_label = _refit_tag
    else:
        bandpass_sol_after = bandpass_sol
        _after_label = 'saved bandpass (no refit)'

    # ── channel-sliced vis_raw for plotting ───────────────────────────────────
    plot_cr = PLOT_CHAN_RANGE or CHAN_RANGE
    _c0 = plot_cr[0] - CHAN_RANGE[0]
    _c1 = plot_cr[1] - CHAN_RANGE[0] + 1

    _vis_plot: dict = {
        **vis_raw,
        'amp':       vis_raw['amp'][:, _c0:_c1, :],
        'phase_deg': vis_raw['phase_deg'][:, _c0:_c1, :],
        'freqs_hz':  vis_raw['freqs_hz'][_c0:_c1],
    }
    # chan_indices must be sliced so expand_flag_table_to_mask can map absolute
    # chanrange entries in flag table to local column indices.
    if 'chan_indices' in vis_raw:
        _vis_plot['chan_indices'] = vis_raw['chan_indices'][_c0:_c1]
    if 'uvdist_per_chan' in vis_raw:
        _vis_plot['uvdist_per_chan'] = vis_raw['uvdist_per_chan'][:, _c0:_c1]
    if 'vis_complex' in vis_raw:
        _vis_plot['vis_complex'] = vis_raw['vis_complex'][:, _c0:_c1, :]

    log.info('Plotting %d channels (PLOT_CHAN_RANGE=%s)', _c1 - _c0, plot_cr)

    # ── AFTER vis: 2-D (row × channel) masking via expand_flag_table_to_mask ─
    # apply_flag_tables_to_vis is row-only (bad_antennas + bad_baselines).
    # expand_flag_table_to_mask also covers bad_antenna_timeranges and
    # bad_baseline_timeranges with their chanrange fields, which is where the
    # channel-selective RFI detections live.
    import numpy as _np
    _flag_mask_2d = q.expand_flag_table_to_mask(
        _vis_plot, ft_new, ant_name_map,
    )
    _vis_plot_after = {**_vis_plot}
    _vis_plot_after['amp'] = _np.where(
        _flag_mask_2d[:, :, _np.newaxis], _np.nan, _vis_plot['amp'],
    )
    if 'vis_complex' in _vis_plot:
        _vis_plot_after['vis_complex'] = _np.where(
            _flag_mask_2d[:, :, _np.newaxis], _np.nan + 0j, _vis_plot['vis_complex'],
        )
    if 'phase_deg' in _vis_plot:
        _vis_plot_after['phase_deg'] = _np.where(
            _flag_mask_2d[:, :, _np.newaxis], _np.nan, _vis_plot['phase_deg'],
        )
    _n_flagged = int(_flag_mask_2d.sum())
    _n_total   = int(_flag_mask_2d.size)
    log.info('AFTER masking: %d / %d vis cells flagged (%.1f%%)  '
             '[wholesale + chanrange time-range entries]',
             _n_flagged, _n_total, 100.0 * _n_flagged / _n_total)

    # ── save-path helpers ─────────────────────────────────────────────────────
    _src = SOURCE.lower()
    def _save(stem: str) -> Path | None:
        if not args.save_plots:
            return None
        return Path(WORK_DIR) / stem

    # ── 6-plot layout: 3 BEFORE, 3 AFTER ─────────────────────────────────────
    import matplotlib.pyplot as plt

    # — BEFORE (1/3): corrected amp vs UV-dist ————————————————————————————————
    print('== BEFORE clustering flags ==')
    q.plot_bandpass_corrected_vis_amp_vs_uvdist(
        _vis_plot, bandpass_sol,
        title            = f'{SOURCE} corrected vis — amp vs UV dist  BEFORE clustering',
        exclude_antennas = EXCLUDE_FOR_PLOTS,
        show_phase       = False,
        alpha            = 0.10,
    )

    # — BEFORE (2/3): vector-avg spectrum ————————————————————————————————————
    q.plot_corrected_vector_avg_spectrum(
        _vis_plot, bandpass_sol,
        title              = f'{SOURCE} corrected spectrum  BEFORE clustering',
        exclude_antennas   = EXCLUDE_FOR_PLOTS,
        skip_edge_channels = SKIP_EDGE_CHANNELS,
    )

    # — BEFORE (3/3): Stokes-V amp vs UV-dist (threshold line) ───────────────
    q.plot_vis_amp_vs_uvdist(
        q.compute_stokes_vis(_vis_plot, bandpass_sol, output_stokes='V', signed=True),
        title      = f'{SOURCE} Stokes-V corrected  BEFORE clustering',
        show_phase = False,
        alpha      = 0.10,
        hline_jy   = CLUSTERING_THRESHOLD_JY if isinstance(CLUSTERING_THRESHOLD_JY, float) else CLUSTERING_THRESHOLD_JY.get('V', 5.0),
        signed     = True,
    )

    # — AFTER (4/3): corrected amp vs UV-dist ————————————————————————————————
    print(f'\n== AFTER clustering flags  [{_after_label}] ==')
    q.plot_bandpass_corrected_vis_amp_vs_uvdist(
        _vis_plot_after, bandpass_sol_after,
        title            = (f'{SOURCE} corrected vis — amp vs UV dist  '
                            f'AFTER clustering  [{_after_label}]'),
        exclude_antennas = EXCLUDE_FOR_PLOTS,
        show_phase       = False,
        alpha            = 0.10,
        save_path        = _save(f'{_src}_clustering_uvdist_after.png'),
    )

    # — AFTER (5/3): vector-avg spectrum ─────────────────────────────────────
    q.plot_corrected_vector_avg_spectrum(
        _vis_plot_after, bandpass_sol_after,
        title              = (f'{SOURCE} corrected spectrum  '
                              f'AFTER clustering  [{_after_label}]'),
        exclude_antennas   = EXCLUDE_FOR_PLOTS,
        skip_edge_channels = SKIP_EDGE_CHANNELS,
        save_path          = _save(f'{_src}_clustering_spectrum_after.png'),
    )

    # — AFTER (6/3): Stokes-V amp vs UV-dist (threshold line) ────────────────
    q.plot_vis_amp_vs_uvdist(
        q.compute_stokes_vis(_vis_plot_after, bandpass_sol_after, output_stokes='V', signed=True),
        title      = (f'{SOURCE} Stokes-V corrected  '
                      f'AFTER clustering  [{_after_label}]'),
        show_phase = False,
        alpha      = 0.10,
        hline_jy   = CLUSTERING_THRESHOLD_JY if isinstance(CLUSTERING_THRESHOLD_JY, float) else CLUSTERING_THRESHOLD_JY.get('V', 5.0),
        signed     = True,
        save_path  = _save(f'{_src}_clustering_stokesV_after.png'),
    )

    if args.save_plots:
        log.info('Plots saved to %s/', WORK_DIR)

    plt.show()
    log.info('Done.')


if __name__ == '__main__':
    main()
