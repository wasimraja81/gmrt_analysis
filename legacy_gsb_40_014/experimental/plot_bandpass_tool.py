#!/usr/bin/env python3
"""Standalone bandpass diagnostics plotting utility.

This tool is intentionally independent of pipeline_cli phases. It lets you
render one diagnostic plot for a user-selected visibility file + optional
calibration/flag tables.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

if not os.environ.get('MPLBACKEND'):
    os.environ['MPLBACKEND'] = 'Agg'

import ugmrt_query as q
from workflow_common import (
    add_common_config_set_arguments,
    add_log_level_argument,
    apply_overrides_to_globals,
    bootstrap_config_from_cli,
    derive_index_cache,
    format_table_application_note,
    resolve_versioned_path,
)

log = logging.getLogger(__name__)

# Config defaults (overridden by cfg + --set)
CAL_FITS: Path | None = None
INDEX_CACHE: Path | None = None
WORK_DIR: Path = Path('.')
SOURCE: str = '3C48'
BANDPASS_OUT: Path | None = None
DIAG_PLOT_BASE: Path | None = None
STOKES: tuple = ('RR', 'LL')
CHAN_RANGE: tuple = (0, 255)
MAX_ROWS_DIAG: int = 60_000
FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED: bool = True
SKIP_EDGE_CHANNELS: tuple = (0, 0)
EXCLUDE_FOR_PLOTS: list = []
DUD_ANTENNA_NAMES: list | None = None
SOLVE_ELEVATION_MIN_DEG: float | None = None
SOLVE_ELEVATION_MAX_DEG: float | None = None
SOLVE_UVRANGE_M: tuple | None = None
SOLVE_UVRANGE_KLAMBDA: tuple | None = None
SOLVE_TIMERANGE: tuple | None = None
FLAG_TABLE_PATHS: list = []
FLAG_TABLE_SESSION: Path | None = None
INDEX_VALIDATION_MODE: str = 'warn'
LOG_LEVEL: str = 'INFO'
DOCAL: str = 'on'
DOFLAG: str = 'off'
CALVER: str = 'latest'
FLAGVER: str = 'latest'


def _apply_overrides(overrides: list) -> None:
    apply_overrides_to_globals(overrides, globals(), log=log)


def _derive_index_cache() -> Path:
    return derive_index_cache(INDEX_CACHE, CAL_FITS, WORK_DIR)


def _build_identity_bandpass(vis_raw: dict, index: dict, source_name: str) -> dict:
    import numpy as _np

    ant_ids = sorted(
        set(_np.asarray(vis_raw['ant1'], dtype=_np.int32).tolist())
        | set(_np.asarray(vis_raw['ant2'], dtype=_np.int32).tolist())
    )
    stokes_labels = list(vis_raw['stokes_labels'])
    freqs_hz = _np.asarray(vis_raw['freqs_hz'], dtype=_np.float64)
    nchan, nant, npol = len(freqs_hz), len(ant_ids), len(stokes_labels)
    ref_ant = int(ant_ids[0]) if ant_ids else 0
    ant_name = {int(a['antenna_no']): str(a['name']) for a in index.get('antennas', [])}

    return {
        'freqs_hz': freqs_hz,
        'chan_indices': _np.asarray(vis_raw.get('chan_indices', list(range(nchan))), dtype=_np.int32),
        'antenna_ids': _np.asarray(ant_ids, dtype=_np.int32),
        'antenna_names': [ant_name.get(int(i), str(i)) for i in ant_ids],
        'stokes_labels': list(stokes_labels),
        'gains': _np.ones((nchan, nant, npol), dtype=_np.complex128),
        'valid': _np.ones((nchan, nant, npol), dtype=bool),
        'reference_antenna': ref_ant,
        'source_name': source_name,
    }


def _parse_args(pre_args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='plot_bandpass_tool.py',
        description='Standalone plot tool for diagnostics using selected vis/cal/flag inputs.',
    )
    add_common_config_set_arguments(
        parser,
        config_default=pre_args.config,
        config_help='Path to .cfg file (default: preprocess_ugmrt.cfg next to this script)',
        set_help='Override any config key after loading: --set "KEY=expr"',
        set_metavar='KEY=expr',
    )
    parser.add_argument('--fits', type=Path, default=None,
                        help='Input UVFITS file path. Defaults to CAL_FITS from config.')
    parser.add_argument('--source', default=None,
                        help='Source/calibrator name. Defaults to SOURCE from config.')
    parser.add_argument('--plot-kind', choices=['vis', 'diagnostics'], default='vis',
                        help='Plot type: vis (uvdist+spectrum+stokesV) or diagnostics panel. Default: vis.')
    parser.add_argument('--docal', choices=['on', 'off'], default=str(globals().get('DOCAL', 'on')),
                        help='Apply calibration table.')
    parser.add_argument('--doflag', choices=['on', 'off'], default='off',
                        help='Apply flag table(s). Default: off (standalone-safe).')
    parser.add_argument('--cal-table', type=Path, default=None,
                        help='Explicit calibration table path (.npz). Overrides --calver.')
    parser.add_argument('--flag-table', type=Path, action='append', default=[],
                        help='Explicit flag table path (.json). Repeatable. Overrides --flagver.')
    parser.add_argument('--calver', default=str(globals().get('CALVER', 'latest')),
                        help='Calibration table selector: latest | N | tag | /path/to/table.npz')
    parser.add_argument('--flagver', default=str(globals().get('FLAGVER', 'latest')),
                        help='Flag table selector: latest | N | tag | /path/to/table.json')
    parser.add_argument('--elevation-min', type=float, default=None,
                        help='Minimum elevation (deg) selection threshold.')
    parser.add_argument('--elevation-max', type=float, default=None,
                        help='Maximum elevation (deg) selection threshold.')
    parser.add_argument('--uvrange-m', nargs=2, type=float, metavar=('MIN_M', 'MAX_M'), default=None,
                        help='UV-range selection in metres.')
    parser.add_argument('--uvrange-klambda', nargs=2, type=float, metavar=('MIN_KL', 'MAX_KL'), default=None,
                        help='UV-range selection in kilolambda.')
    parser.add_argument('--timerange', nargs=2, metavar=('START', 'END'), default=None,
                        help='UTC timerange selection. Example: --timerange "2021-07-25 19:00:00" "2021-07-25 23:00:00"')
    parser.add_argument('--chan-range', nargs=2, type=int, metavar=('START', 'END'), default=None,
                        help='Channel range [start end] for diagnostics load.')
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Maximum visibility rows for diagnostics. Default: no limit (no downsampling).')
    parser.add_argument('--stokes-v-threshold-jy', type=float, default=5.0,
                        help='Horizontal threshold line for Stokes-V vis plot (Jy).')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output path prefix. For vis plots writes *_uvdist.png, *_spectrum.png, *_stokesV.png.')
    add_log_level_argument(
        parser,
        default=None,
        help_text='Logging verbosity (default from config LOG_LEVEL or INFO).',
        metavar='LEVEL',
    )
    return parser.parse_args()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_cfg = str(script_dir / 'preprocess_ugmrt.cfg')
    pre_args = bootstrap_config_from_cli(default_cfg, globals(), log=log)
    args = _parse_args(pre_args)
    _apply_overrides(args.set_overrides)

    effective_level = args.log_level or globals().get('LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=getattr(logging, effective_level, logging.INFO),
        format='%(asctime)s  %(levelname)-8s  %(message)s',
    )

    fits_path = Path(args.fits) if args.fits is not None else (Path(CAL_FITS) if CAL_FITS is not None else None)
    if fits_path is None:
        sys.exit('ERROR: no input vis provided. Set --fits or CAL_FITS in config.')
    if not fits_path.exists():
        sys.exit(f'ERROR: input vis not found: {fits_path}')

    source = str(args.source) if args.source is not None else str(globals().get('SOURCE', '3C48'))
    chan_range = tuple(args.chan_range) if args.chan_range is not None else tuple(globals().get('CHAN_RANGE', (0, 255)))
    max_rows = int(args.max_rows) if args.max_rows is not None else 0

    elevation_min = args.elevation_min if args.elevation_min is not None else globals().get('SOLVE_ELEVATION_MIN_DEG', None)
    elevation_max = args.elevation_max if args.elevation_max is not None else globals().get('SOLVE_ELEVATION_MAX_DEG', None)
    uvrange_m = tuple(args.uvrange_m) if args.uvrange_m is not None else globals().get('SOLVE_UVRANGE_M', None)
    uvrange_kl = tuple(args.uvrange_klambda) if args.uvrange_klambda is not None else globals().get('SOLVE_UVRANGE_KLAMBDA', None)
    timerange = tuple(args.timerange) if args.timerange is not None else globals().get('SOLVE_TIMERANGE', None)

    out_base = args.out
    if out_base is None:
        src = source.lower().replace(' ', '_')
        out_base = Path(WORK_DIR) / f'{src}_plot_tool'

    if out_base.suffix:
        out_prefix = out_base.with_suffix('')
    else:
        out_prefix = out_base

    log.info('Input vis      : %s', fits_path)
    log.info('Source         : %s', source)
    log.info('Plot kind      : %s', args.plot_kind)
    log.info('DOCAL/DOFLAG   : %s / %s', args.docal, args.doflag)
    if max_rows:
        log.info('Max rows       : %d', max_rows)
    else:
        log.info('Max rows       : no limit (no downsampling)')
    log.info('Selection      : el=[%s,%s] uvm=%s uvk=%s timerange=%s', elevation_min, elevation_max, uvrange_m, uvrange_kl, timerange)

    cache = _derive_index_cache()
    index = q.get_or_build_row_index(
        fits_path,
        cache_path=cache,
        force_rebuild=False,
        validation_mode=INDEX_VALIDATION_MODE,
        write_cache=True,
        override_dud_names=DUD_ANTENNA_NAMES or None,
    )

    session_path = Path(FLAG_TABLE_SESSION) if FLAG_TABLE_SESSION else None
    active_disk = [Path(p) for p in (FLAG_TABLE_PATHS or [])]
    if session_path is not None and session_path.exists() and session_path not in active_disk:
        active_disk.append(session_path)

    if args.flag_table:
        selected_flag_paths = [Path(p).resolve() for p in args.flag_table]
    elif args.doflag == 'off':
        selected_flag_paths = []
    elif args.flagver != 'latest':
        _src = source.lower()
        _flag_candidates = list(active_disk)
        _flag_candidates.extend(Path(WORK_DIR).glob(f'{_src}_flag_table*.json'))
        try:
            selected_flag_paths = [
                resolve_versioned_path(
                    candidates=_flag_candidates,
                    version=args.flagver,
                    default_path=(active_disk[-1] if active_disk else None),
                    label='flag table',
                )
            ]
        except ValueError as exc:
            sys.exit(f'ERROR: {exc}')
    else:
        selected_flag_paths = active_disk

    for fp in selected_flag_paths:
        if not Path(fp).exists():
            sys.exit(f'ERROR: flag table not found: {fp}')

    bp_path = None
    if args.cal_table is not None:
        bp_path = Path(args.cal_table).resolve()
    elif args.docal == 'on':
        base_bp = Path(BANDPASS_OUT) if BANDPASS_OUT else None
        try:
            bp_path = resolve_versioned_path(
                candidates=([base_bp] + list(base_bp.parent.glob(f'{base_bp.stem}_*.npz')) if base_bp is not None else []),
                version=args.calver,
                default_path=base_bp,
                label='bandpass table',
            )
        except ValueError as exc:
            sys.exit(f'ERROR: {exc}')

    if args.docal == 'on':
        if bp_path is None or not bp_path.exists():
            sys.exit(f'ERROR: calibration table not found: {bp_path}')
        solution = q.load_bandpass_solution(bp_path)
        apply_correction = True
    else:
        solution = None
        apply_correction = False
        log.info('DOCAL=off: plotting raw vis amplitudes (no calibration application).')

    vis_for_plots = None
    if args.plot_kind == 'vis':
        vis_for_plots = q.load_vis_for_source(
            index,
            source=source,
            stokes=list(STOKES),
            max_rows=max_rows,
            chan_range=chan_range,
            flag_all_corrs_if_any_rawvis_flagged=FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
            elevation_min_deg=elevation_min,
            elevation_max_deg=elevation_max,
            timerange=timerange,
            uvrange_m=uvrange_m,
            uvrange_klambda=uvrange_kl,
        )
        ant_name_map = {
            int(a['antenna_no']): str(a['name'])
            for a in index.get('antennas', [])
            if a.get('antenna_no') is not None
        }
        if args.doflag == 'on' and selected_flag_paths:
            vis_for_plots, _plot_flag_stats = q.apply_flag_tables_to_vis(
                vis_for_plots,
                antenna_name_map=ant_name_map,
                flag_table_paths=selected_flag_paths,
            )
            log.info(
                'Applied flags   : dropped %d rows, kept %d rows',
                int(_plot_flag_stats.get('dropped_rows', 0)),
                int(_plot_flag_stats.get('kept_rows', 0)),
            )

        if solution is None:
            solution = _build_identity_bandpass(vis_for_plots, index, source)

    table_note = format_table_application_note(
        docal=args.docal,
        doflag=args.doflag,
        calver=args.calver,
        flagver=args.flagver,
        cal_path=(bp_path if args.docal == 'on' else None),
        flag_paths=selected_flag_paths,
    )
    title = f'Plot Tool | Source={source} | el=[{elevation_min},{elevation_max}]' + table_note

    if args.plot_kind == 'diagnostics':
        result = q.run_bandpass_diagnostics(
            index,
            solution,
            source=source,
            chan_range=chan_range,
            stokes=tuple(STOKES),
            max_rows=max_rows,
            exclude_antennas=EXCLUDE_FOR_PLOTS,
            apply_flag_tables=(args.doflag == 'on'),
            flag_table_path=selected_flag_paths,
            apply_correction=apply_correction,
            flag_all_corrs_if_any_rawvis_flagged=FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
            skip_edge_channels=SKIP_EDGE_CHANNELS,
            title=title,
            save_path=out_prefix.with_suffix('.png'),
            timerange=timerange,
            uvrange_m=uvrange_m,
            uvrange_klambda=uvrange_kl,
            elevation_min_deg=elevation_min,
            elevation_max_deg=elevation_max,
        )
        log.info('Saved plot     : %s', result.get('save_path') or out_prefix.with_suffix('.png'))
        fig = result.get('figure')
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)
    else:
        uv_path = Path(f'{out_prefix}_uvdist.png')
        spec_path = Path(f'{out_prefix}_spectrum.png')
        v_path = Path(f'{out_prefix}_stokesV.png')

        q.plot_bandpass_corrected_vis_amp_vs_uvdist(
            vis_for_plots,
            solution,
            title=f'{source} corrected vis — amp vs UV dist\n{table_note}',
            exclude_antennas=EXCLUDE_FOR_PLOTS,
            show_phase=False,
            alpha=0.10,
            save_path=uv_path,
        )
        q.plot_corrected_vector_avg_spectrum(
            vis_for_plots,
            solution,
            title=f'{source} corrected spectrum\n{table_note}',
            exclude_antennas=EXCLUDE_FOR_PLOTS,
            skip_edge_channels=SKIP_EDGE_CHANNELS,
            save_path=spec_path,
        )
        q.plot_vis_amp_vs_uvdist(
            q.compute_stokes_vis(vis_for_plots, solution, output_stokes='V', signed=True),
            title=f'{source} Stokes-V corrected\n{table_note}',
            show_phase=False,
            alpha=0.10,
            hline_jy=float(args.stokes_v_threshold_jy),
            signed=True,
            save_path=v_path,
        )
        log.info('Saved vis plots: %s | %s | %s', uv_path, spec_path, v_path)

    log.info('Flag tables    : %s', [str(p) for p in selected_flag_paths] if selected_flag_paths else '(none)')
    log.info('Cal table      : %s', str(bp_path) if bp_path is not None else '(identity/no cal table)')


if __name__ == '__main__':
    main()
