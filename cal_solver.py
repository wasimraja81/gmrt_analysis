#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np

import ugmrt_query as q
from workflow_common import (
    add_common_config_set_arguments,
    add_log_level_argument,
    apply_overrides_to_globals,
    bootstrap_config_from_cli,
    derive_index_cache,
    resolve_versioned_path,
)

log = logging.getLogger(__name__)

# Config-backed defaults (overridden by cfg + --set)
CAL_FITS: Path | None = None
INDEX_CACHE: Path | None = None
WORK_DIR: Path = Path('.')
SOURCE: str = '3C48'
CHAN_RANGE: tuple = (64, 191)
STOKES: tuple = ('RR', 'LL')
MAX_ROWS_SOLVE: int = 150_000
SMOOTH_WINDOW: int = 5
MIN_BASELINES: int = 20
FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED: bool = True
SOLVE_ELEVATION_MIN_DEG: float | None = None
SOLVE_ELEVATION_MAX_DEG: float | None = None
SOLVE_UVRANGE_M: tuple | None = None
SOLVE_UVRANGE_KLAMBDA: tuple | None = None
SOLVE_TIMERANGE: tuple | None = None
FLAG_TABLE_PATHS: list = []
FLAG_TABLE_SESSION: Path | None = None
INDEX_VALIDATION_MODE: str = 'warn'
FLAGVER: str = 'latest'
LOG_LEVEL: str = 'INFO'


def _apply_overrides(overrides: list) -> None:
    apply_overrides_to_globals(overrides, globals(), log=log)


def _derive_index_cache(fits_path: Path, explicit_index_cache: Path | None) -> Path:
    if explicit_index_cache is not None:
        return Path(explicit_index_cache)
    return derive_index_cache(INDEX_CACHE, fits_path, WORK_DIR)


def _source_tag(source: str) -> str:
    return str(source).strip().lower().replace(' ', '_')


def _resolve_flag_tables(args: argparse.Namespace, source: str) -> list[Path]:
    if args.flag_table:
        selected = [Path(p).expanduser().resolve() for p in args.flag_table]
        missing = [str(p) for p in selected if not p.exists()]
        if missing:
            sys.exit(f'ERROR: explicit --flag-table path(s) not found: {missing}')
        return selected

    session_path = Path(FLAG_TABLE_SESSION).resolve() if FLAG_TABLE_SESSION else None
    configured = [Path(p).resolve() for p in (FLAG_TABLE_PATHS or []) if Path(p).exists()]
    if session_path is not None and session_path.exists() and session_path not in configured:
        configured.append(session_path)

    src = _source_tag(source)
    discovered = sorted(Path(WORK_DIR).glob(f'{src}_flag_table*.json'))
    candidates = configured + [p.resolve() for p in discovered if p.exists()]

    if not candidates:
        return []

    if args.flagver and str(args.flagver).strip().lower() != 'latest':
        try:
            return [
                resolve_versioned_path(
                    candidates=candidates,
                    version=args.flagver,
                    default_path=(session_path if session_path and session_path.exists() else None),
                    label='flag table',
                )
            ]
        except ValueError as exc:
            sys.exit(f'ERROR: {exc}')

    return candidates


def _parse_args(pre_args: argparse.Namespace) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Unified calibration solver entrypoint')
    add_common_config_set_arguments(
        parser,
        config_default=pre_args.config,
        config_help='Path to .cfg file (default: preprocess_ugmrt.cfg)',
        set_help='Override any config key after loading: --set "KEY=expr"',
        set_metavar='KEY=expr',
    )

    parser.add_argument('--fits', type=Path, default=None, help='Input UVFITS path (defaults to CAL_FITS from config).')
    parser.add_argument('--index-cache', type=Path, default=None, help='Row-index cache path (optional override).')
    parser.add_argument('--source', default=None, help='Calibration source (defaults to SOURCE from config).')
    parser.add_argument('--mode', choices=['primary_bandpass', 'phase_only', 'delay_phase'], required=True)
    parser.add_argument('--bpcal', default=None, help='Optional primary bandpass table metadata dependency (not hardcoded).')
    parser.add_argument('--flag-table', type=Path, action='append', default=[], help='Explicit flag table path(s). Repeatable.')
    parser.add_argument('--flagver', default=str(globals().get('FLAGVER', 'latest')), help='Flag table selector: latest | N | tag | /path/to/file.json')
    parser.add_argument('--strict-flag-table', action='store_true', help='Fail on malformed flag tables instead of best-effort parsing.')
    parser.add_argument('--solint-mode', dest='solint_mode', choices=['all', 'minutes', 'scan'], default='scan',
                        help='Solution interval strategy: all | minutes | scan (default: scan)')
    parser.add_argument('--solint', dest='solint_mode', choices=['all', 'minutes', 'scan'],
                        help=argparse.SUPPRESS)
    parser.add_argument('--solint-minutes', type=float, default=10.0,
                        help='Minutes per solution interval when --solint-mode=minutes.')
    parser.add_argument('--scan-gap-minutes', type=float, default=None,
                        help='Optional manual gap threshold (minutes) for --solint-mode=scan. Default: auto-derived from timestamp cadence.')
    parser.add_argument('--timerange', nargs=2, default=None, metavar=('START', 'END'))
    parser.add_argument('--chan-range', nargs=2, type=int, default=None, metavar=('START', 'END'))
    parser.add_argument('--max-rows', type=int, default=None, help='Max rows for solving; default from config.')
    parser.add_argument('--elevation-min', type=float, default=None)
    parser.add_argument('--elevation-max', type=float, default=None)
    parser.add_argument('--uvrange-m', nargs=2, type=float, metavar=('MIN_M', 'MAX_M'), default=None)
    parser.add_argument('--uvrange-klambda', nargs=2, type=float, metavar=('MIN_KL', 'MAX_KL'), default=None)
    parser.add_argument('--out', required=True, help='Output calibration table path (.npz).')
    add_log_level_argument(
        parser,
        default=None,
        help_text='Logging verbosity (default from config LOG_LEVEL or INFO).',
        metavar='LEVEL',
    )
    return parser.parse_args()


def _jd_to_utc(jd: float) -> str:
    unix = (float(jd) - 2440587.5) * 86400.0
    ts = dt.datetime.fromtimestamp(unix, tz=dt.timezone.utc)
    return ts.strftime('%Y-%m-%d %H:%M:%S')


def _log_planned_intervals(intervals: list[tuple[float, float, str]]) -> None:
    log.info('[cal-solver] planned %d solution interval(s)', len(intervals))
    for i, (t0_jd, t1_jd, label) in enumerate(intervals, 1):
        if np.isfinite(t0_jd) and np.isfinite(t1_jd):
            dur_min = max(0.0, (float(t1_jd) - float(t0_jd)) * 1440.0)
            log.info(
                '[cal-solver] interval %02d %-8s %s -> %s (%.2f min)',
                i,
                label,
                _jd_to_utc(float(t0_jd)),
                _jd_to_utc(float(t1_jd)),
                dur_min,
            )
        else:
            log.info('[cal-solver] interval %02d %-8s full selected timerange', i, label)


def _plan_intervals_jd(
    *,
    index: dict,
    source: str,
    chan_range: tuple,
    timerange,
    uvrange_m,
    uvrange_kl,
    elevation_min,
    elevation_max,
    solint: str,
    solint_minutes: float,
    scan_gap_minutes: float | None,
) -> list[tuple[float, float, str]]:
    if solint == 'all':
        ch0 = int(chan_range[0]) if chan_range else 0
        stokes0 = list(STOKES[:1] if STOKES else ('RR',))
        probe = q.load_vis_for_source(
            index,
            source=source,
            stokes=stokes0,
            chan_range=(ch0, ch0),
            max_rows=0,
            flag_all_corrs_if_any_rawvis_flagged=FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
            timerange=timerange,
            uvrange_m=uvrange_m,
            uvrange_klambda=uvrange_kl,
            elevation_min_deg=elevation_min,
            elevation_max_deg=elevation_max,
        )
        jd = np.asarray(probe.get('jd', []), dtype=np.float64)
        jd = jd[np.isfinite(jd)]
        if jd.size == 0:
            raise ValueError('No valid timestamps after selection; cannot build all-interval range.')
        return [(float(np.min(jd)), float(np.max(jd)), 'all')]

    ch0 = int(chan_range[0]) if chan_range else 0
    stokes0 = list(STOKES[:1] if STOKES else ('RR',))
    probe = q.load_vis_for_source(
        index,
        source=source,
        stokes=stokes0,
        chan_range=(ch0, ch0),
        max_rows=0,
        flag_all_corrs_if_any_rawvis_flagged=FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
        timerange=timerange,
        uvrange_m=uvrange_m,
        uvrange_klambda=uvrange_kl,
        elevation_min_deg=elevation_min,
        elevation_max_deg=elevation_max,
    )
    jd = np.asarray(probe.get('jd', []), dtype=np.float64)
    jd = jd[np.isfinite(jd)]
    if jd.size == 0:
        raise ValueError('No valid timestamps after selection; cannot build solint intervals.')

    uniq = np.unique(np.sort(jd))
    if uniq.size == 1:
        return [(float(uniq[0]), float(uniq[0]), 'all')]

    if solint == 'scan':
        dt_sec = np.diff(uniq) * 86400.0
        dt_sec = dt_sec[np.isfinite(dt_sec) & (dt_sec > 0.0)]

        if dt_sec.size == 0:
            gap_sec = 5.0
            gap_origin = 'fallback'
        elif scan_gap_minutes is not None:
            gap_sec = max(float(scan_gap_minutes) * 60.0, 0.0)
            gap_origin = 'user'
        else:
            cadence_sec = float(np.median(dt_sec))
            large_gaps = dt_sec[dt_sec > (5.0 * cadence_sec)]
            if large_gaps.size:
                gap_sec = max(5.0 * cadence_sec, 0.5 * float(np.min(large_gaps)))
            else:
                gap_sec = max(10.0 * cadence_sec, 5.0)
            gap_origin = 'auto'

        gap_days = gap_sec / 86400.0
        log.info('[cal-solver] scan gap (%s): %.2f s', gap_origin, gap_sec)
        split_idx = np.where(np.diff(uniq) > gap_days)[0] + 1
        groups = np.split(np.arange(uniq.size), split_idx)
        out = []
        for idx_g, grp in enumerate(groups, 1):
            if grp.size == 0:
                continue
            t0 = float(uniq[int(grp[0])])
            t1 = float(uniq[int(grp[-1])])
            out.append((t0, t1, f'scan{idx_g:02d}'))
        return out or [(float(uniq[0]), float(uniq[-1]), 'all')]

    step_min = float(solint_minutes)
    if step_min <= 0.0:
        raise ValueError('--solint-minutes must be > 0 for --solint-mode=minutes')
    step_days = step_min / 1440.0

    start = float(uniq[0])
    stop = float(uniq[-1])
    edges = np.arange(start, stop + step_days, step_days, dtype=np.float64)
    if edges.size < 2:
        edges = np.asarray([start, stop], dtype=np.float64)

    out: list[tuple[float, float, str]] = []
    for i in range(edges.size - 1):
        t0 = float(edges[i])
        t1 = float(edges[i + 1])
        if i == edges.size - 2:
            sel = (uniq >= t0) & (uniq <= t1)
        else:
            sel = (uniq >= t0) & (uniq < t1)
        if not np.any(sel):
            continue
        out.append((float(uniq[sel][0]), float(uniq[sel][-1]), f'm{i+1:03d}'))

    return out or [(start, stop, 'all')]


def _apply_primary_transfer(solution: dict, primary_solution: dict) -> dict:
    gains = np.asarray(solution['gains'], dtype=np.complex128)
    gains_raw = np.asarray(solution.get('gains_raw', solution['gains']), dtype=np.complex128)
    valid = np.asarray(solution['valid'], dtype=bool)

    p_gains = np.asarray(primary_solution['gains'], dtype=np.complex128)
    p_valid = np.asarray(primary_solution['valid'], dtype=bool)

    chan_s = np.asarray(solution.get('chan_indices', np.arange(gains.shape[0])), dtype=np.int32)
    chan_p = np.asarray(primary_solution.get('chan_indices', np.arange(p_gains.shape[0])), dtype=np.int32)
    ant_s = np.asarray(solution['antenna_ids'], dtype=np.int32)
    ant_p = np.asarray(primary_solution['antenna_ids'], dtype=np.int32)
    pol_s = [str(x) for x in solution['stokes_labels']]
    pol_p = [str(x) for x in primary_solution['stokes_labels']]

    ch_map_s = {int(c): i for i, c in enumerate(chan_s.tolist())}
    ch_map_p = {int(c): i for i, c in enumerate(chan_p.tolist())}
    ant_map_s = {int(a): i for i, a in enumerate(ant_s.tolist())}
    ant_map_p = {int(a): i for i, a in enumerate(ant_p.tolist())}
    pol_map_s = {p: i for i, p in enumerate(pol_s)}
    pol_map_p = {p: i for i, p in enumerate(pol_p)}

    common_ch = sorted(set(ch_map_s).intersection(ch_map_p))
    common_ant = sorted(set(ant_map_s).intersection(ant_map_p))
    common_pol = sorted(set(pol_map_s).intersection(pol_map_p))
    if not common_ch or not common_ant or not common_pol:
        raise ValueError('No overlapping channel/antenna/stokes coverage between secondary and primary solutions.')

    for ch in common_ch:
        si = ch_map_s[ch]
        pi = ch_map_p[ch]
        for ant in common_ant:
            sa = ant_map_s[ant]
            pa = ant_map_p[ant]
            for pol in common_pol:
                sp = pol_map_s[pol]
                pp = pol_map_p[pol]
                p_gain = p_gains[pi, pa, pp]
                p_ok = bool(p_valid[pi, pa, pp]) and np.isfinite(p_gain.real) and np.isfinite(p_gain.imag) and (abs(p_gain) > 0.0)
                if not p_ok:
                    valid[si, sa, sp] = False
                    gains[si, sa, sp] = np.nan + 1j * np.nan
                    gains_raw[si, sa, sp] = np.nan + 1j * np.nan
                    continue
                gains[si, sa, sp] = gains[si, sa, sp] / p_gain
                gains_raw[si, sa, sp] = gains_raw[si, sa, sp] / p_gain
                valid[si, sa, sp] = bool(valid[si, sa, sp]) and p_ok

    out = dict(solution)
    out['gains'] = gains
    out['gains_raw'] = gains_raw
    out['valid'] = valid
    out['notes'] = (
        f'{solution.get("notes", "")} '
        f'Primary transfer applied using {primary_solution.get("source_name", "unknown")} table.'
    ).strip()
    return out


def _project_delay_phase(solution: dict) -> dict:
    gains = np.asarray(solution['gains'], dtype=np.complex128)
    valid = np.asarray(solution['valid'], dtype=bool)
    freqs = np.asarray(solution['freqs_hz'], dtype=np.float64)
    nchan, nant, npol = gains.shape

    out_gains = np.full_like(gains, np.nan + 1j * np.nan)
    out_valid = np.zeros_like(valid, dtype=bool)
    phi0 = np.full((nant, npol), np.nan, dtype=np.float64)
    tau = np.full((nant, npol), np.nan, dtype=np.float64)

    x_all = 2.0 * np.pi * freqs
    for ant in range(nant):
        for pol in range(npol):
            m = valid[:, ant, pol] & np.isfinite(gains[:, ant, pol].real) & np.isfinite(gains[:, ant, pol].imag)
            if int(np.count_nonzero(m)) < 2:
                continue
            x = x_all[m]
            y = np.unwrap(np.angle(gains[m, ant, pol]))
            slope, intercept = np.polyfit(x, y, 1)
            model = intercept + slope * x_all
            out_gains[:, ant, pol] = np.exp(1j * model)
            out_valid[:, ant, pol] = True
            phi0[ant, pol] = float(intercept)
            tau[ant, pol] = float(slope)

    out = dict(solution)
    out['gains_raw'] = gains
    out['gains'] = out_gains
    out['valid'] = out_valid
    out['delay_phase_phi0_rad'] = phi0
    out['delay_phase_tau_s'] = tau
    out['kind'] = 'secondary_delay_phase'
    out['notes'] = (
        f'{solution.get("notes", "")} '
        'Mode=delay_phase; gains projected to phi0 + 2*pi*tau*nu model (single interval).'
    ).strip()
    return out


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_cfg = str(script_dir / 'preprocess_ugmrt.cfg')
    pre_args = bootstrap_config_from_cli(default_cfg, globals(), log=log)
    args = _parse_args(pre_args)
    _apply_overrides(args.set_overrides)

    effective_level = args.log_level or globals().get('LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=getattr(logging, str(effective_level).upper(), logging.INFO),
        format='%(asctime)s  %(levelname)-8s  %(message)s',
    )

    fits_path = Path(args.fits) if args.fits is not None else (Path(CAL_FITS) if CAL_FITS is not None else None)
    if fits_path is None:
        sys.exit('ERROR: no input vis provided. Set --fits or CAL_FITS in config.')
    if not fits_path.exists():
        sys.exit(f'ERROR: input vis not found: {fits_path}')

    source = str(args.source) if args.source is not None else str(globals().get('SOURCE', '3C48'))
    chan_range = tuple(args.chan_range) if args.chan_range is not None else tuple(globals().get('CHAN_RANGE', (64, 191)))
    max_rows = int(args.max_rows) if args.max_rows is not None else int(globals().get('MAX_ROWS_SOLVE', 150_000))

    timerange = tuple(args.timerange) if args.timerange is not None else globals().get('SOLVE_TIMERANGE', None)
    elevation_min = args.elevation_min if args.elevation_min is not None else globals().get('SOLVE_ELEVATION_MIN_DEG', None)
    elevation_max = args.elevation_max if args.elevation_max is not None else globals().get('SOLVE_ELEVATION_MAX_DEG', None)
    uvrange_m = tuple(args.uvrange_m) if args.uvrange_m is not None else globals().get('SOLVE_UVRANGE_M', None)
    uvrange_kl = tuple(args.uvrange_klambda) if args.uvrange_klambda is not None else globals().get('SOLVE_UVRANGE_KLAMBDA', None)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    index_cache = _derive_index_cache(fits_path=fits_path, explicit_index_cache=args.index_cache)
    index = q.get_or_build_row_index(
        fits_path,
        cache_path=index_cache,
        force_rebuild=False,
        validation_mode=INDEX_VALIDATION_MODE,
        write_cache=True,
    )

    selected_flag_paths = _resolve_flag_tables(args, source)
    if not selected_flag_paths:
        log.warning('No flag tables resolved for source=%s; solve will run unflagged.', source)

    primary_solution = None
    if args.bpcal:
        bp = Path(str(args.bpcal)).expanduser().resolve()
        if not bp.exists():
            sys.exit(f'ERROR: --bpcal not found: {bp}')
        primary_solution = q.load_bandpass_solution(bp)
        log.info('Primary bandpass input: %s', bp)

    if args.mode in {'phase_only', 'delay_phase'} and primary_solution is None:
        log.warning('Secondary mode requested without --bpcal; solving directly on raw vis (no primary transfer).')

    model_flux_jy = None
    if args.mode in {'phase_only', 'delay_phase'}:
        if chan_range is not None:
            c0, c1 = int(chan_range[0]), int(chan_range[1])
            chan_idx = np.arange(max(0, c0), min(int(index['naxis4']), c1 + 1), dtype=np.int32)
        else:
            chan_idx = np.arange(int(index['naxis4']), dtype=np.int32)
        freqs_hz = np.asarray(index['chan_freqs_hz'], dtype=np.float64)[chan_idx]
        model_flux_jy = np.ones_like(freqs_hz, dtype=np.float64)

    try:
        intervals = _plan_intervals_jd(
            index=index,
            source=source,
            chan_range=chan_range,
            timerange=timerange,
            uvrange_m=uvrange_m,
            uvrange_kl=uvrange_kl,
            elevation_min=elevation_min,
            elevation_max=elevation_max,
            solint=args.solint_mode,
            solint_minutes=args.solint_minutes,
            scan_gap_minutes=args.scan_gap_minutes,
        )
    except Exception as exc:
        sys.exit(f'ERROR: could not build solint intervals: {exc}')

    _log_planned_intervals(intervals)

    saved: list[Path] = []
    for i, (t0_jd, t1_jd, label) in enumerate(intervals, 1):
        if np.isfinite(t0_jd) and np.isfinite(t1_jd):
            timerange_i = (_jd_to_utc(t0_jd), _jd_to_utc(t1_jd))
        else:
            timerange_i = timerange

        solution = q.derive_point_source_bandpass(
            index,
            source=source,
            chan_range=chan_range,
            stokes=tuple(STOKES),
            max_rows=max_rows,
            model_flux_jy=model_flux_jy,
            smooth_window=int(globals().get('SMOOTH_WINDOW', 5)),
            min_baselines=int(globals().get('MIN_BASELINES', 20)),
            flag_table_path=[str(p) for p in selected_flag_paths] if selected_flag_paths else None,
            strict_flag_table=bool(args.strict_flag_table),
            flag_all_corrs_if_any_rawvis_flagged=FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
            timerange=timerange_i,
            uvrange_m=uvrange_m,
            uvrange_klambda=uvrange_kl,
            elevation_min_deg=elevation_min,
            elevation_max_deg=elevation_max,
        )

        if args.mode in {'phase_only', 'delay_phase'} and primary_solution is not None:
            solution = _apply_primary_transfer(solution, primary_solution)

        if args.mode == 'phase_only':
            gains = np.asarray(solution['gains'], dtype=np.complex128)
            valid = np.asarray(solution['valid'], dtype=bool)
            phase_only = np.exp(1j * np.angle(gains))
            phase_only[~valid] = np.nan + 1j * np.nan
            solution['gains_raw'] = gains.copy()
            solution['gains'] = phase_only
            solution['kind'] = 'secondary_phase_only'
            solution['notes'] = (
                f'{solution.get("notes", "")} '
                'Mode=phase_only; saved gains are phase-only (unit amplitude).'
            ).strip()
        elif args.mode == 'delay_phase':
            solution = _project_delay_phase(solution)

        if len(intervals) == 1:
            out_i = out_path
        else:
            out_i = out_path.with_name(f'{out_path.stem}_{label}{out_path.suffix}')

        solution['solution_time_start_jd'] = float(t0_jd)
        solution['solution_time_end_jd'] = float(t1_jd)
        solution['solution_time_center_jd'] = float(0.5 * (float(t0_jd) + float(t1_jd)))
        solution['solution_time_start_utc'] = _jd_to_utc(float(t0_jd))
        solution['solution_time_end_utc'] = _jd_to_utc(float(t1_jd))
        solution['solution_time_center_utc'] = _jd_to_utc(float(0.5 * (float(t0_jd) + float(t1_jd))))
        solution['solution_interval_label'] = str(label)

        q.save_bandpass_solution(solution, out_i)
        saved.append(out_i)
        print(f'[cal-solver] interval {i}/{len(intervals)} saved: {out_i}')

    print('[cal-solver] done')
    print(f'[cal-solver] mode={args.mode} source={source} solint_mode={args.solint_mode} n_intervals={len(saved)}')
    print(f'[cal-solver] flags={len(selected_flag_paths)} table(s)')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
