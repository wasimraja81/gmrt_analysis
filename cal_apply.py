#!/usr/bin/env python3

from __future__ import annotations

import argparse
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
)

log = logging.getLogger(__name__)

# Config-backed defaults
CAL_FITS: Path | None = None
INDEX_CACHE: Path | None = None
WORK_DIR: Path = Path('.')
SOURCE: str = '3C48'
CHAN_RANGE: tuple = (64, 191)
STOKES: tuple = ('RR', 'LL')
MAX_ROWS_DIAG: int = 60_000
FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED: bool = True
SOLVE_ELEVATION_MIN_DEG: float | None = None
SOLVE_ELEVATION_MAX_DEG: float | None = None
SOLVE_UVRANGE_M: tuple | None = None
SOLVE_UVRANGE_KLAMBDA: tuple | None = None
SOLVE_TIMERANGE: tuple | None = None
INDEX_VALIDATION_MODE: str = 'warn'
LOG_LEVEL: str = 'INFO'


TIME_INTERP_SCHEMES = [
    'nearest',
    'linear',
    'slinear',
    'quadratic',
    'cubic',
    'pchip',
    'akima',
]


def _apply_overrides(overrides: list) -> None:
    apply_overrides_to_globals(overrides, globals(), log=log)


def _derive_index_cache(fits_path: Path, explicit_index_cache: Path | None) -> Path:
    if explicit_index_cache is not None:
        return Path(explicit_index_cache)
    return derive_index_cache(INDEX_CACHE, fits_path, WORK_DIR)


def _parse_args(pre_args: argparse.Namespace) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Unified calibration applicator entrypoint')
    add_common_config_set_arguments(
        parser,
        config_default=pre_args.config,
        config_help='Path to .cfg file (default: preprocess_ugmrt.cfg)',
        set_help='Override any config key after loading: --set "KEY=expr"',
        set_metavar='KEY=expr',
    )
    parser.add_argument('--fits', default=None)
    parser.add_argument('--index-cache', default=None)
    parser.add_argument('--source', default=None)
    parser.add_argument('--tables', required=True, nargs='+', help='One or more calibration tables (apply in order)')
    parser.add_argument(
        '--time-interp-scheme',
        dest='time_interp_scheme',
        choices=TIME_INTERP_SCHEMES,
        default='nearest',
        help='Time interpolation scheme for multi-time solutions.',
    )
    parser.add_argument(
        '--time-interp',
        dest='time_interp_scheme',
        choices=TIME_INTERP_SCHEMES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--time-extrapolation',
        choices=['hold', 'nearest', 'none'],
        default='hold',
        help='Behavior outside solved time range: hold (default), nearest, or none (invalid).',
    )
    parser.add_argument('--chan-range', nargs=2, type=int, default=None, metavar=('START', 'END'))
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument(
        '--strict-time-checks',
        action='store_true',
        help='Fail if timed solution tables look inconsistent (order/overlap/gaps).',
    )
    parser.add_argument('--out', default=None, help=argparse.SUPPRESS)
    add_log_level_argument(
        parser,
        default=None,
        help_text='Logging verbosity (default from config LOG_LEVEL or INFO).',
        metavar='LEVEL',
    )
    return parser.parse_args()


def _group_key(solution: dict) -> tuple:
    return (
        str(solution.get('kind', '')),
        str(solution.get('source_name', '')),
        tuple(int(x) for x in np.asarray(solution['chan_indices'], dtype=np.int32).tolist()),
        tuple(int(x) for x in np.asarray(solution['antenna_ids'], dtype=np.int32).tolist()),
        tuple(str(x) for x in solution['stokes_labels']),
    )


def _group_tables_in_order(solutions: list[dict]) -> list[list[dict]]:
    if not solutions:
        return []
    grouped: list[list[dict]] = []
    current = [solutions[0]]
    current_key = _group_key(solutions[0])
    for sol in solutions[1:]:
        key = _group_key(sol)
        if key == current_key:
            current.append(sol)
        else:
            grouped.append(current)
            current = [sol]
            current_key = key
    grouped.append(current)
    return grouped


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _table_time_info(solution: dict) -> tuple[bool, float | None, float | None, float | None]:
    center = _safe_float(solution.get('solution_time_center_jd', None))
    start = _safe_float(solution.get('solution_time_start_jd', None))
    end = _safe_float(solution.get('solution_time_end_jd', None))
    has_time = center is not None
    return has_time, center, start, end


def _timed_domain_jd(group_timed: list[dict], centers_t: np.ndarray) -> tuple[float, float]:
    starts: list[float] = []
    ends: list[float] = []
    for sol in group_timed:
        _, _, s, e = _table_time_info(sol)
        if s is not None:
            starts.append(float(s))
        if e is not None:
            ends.append(float(e))
    lo = float(np.min(starts)) if starts else float(np.min(centers_t))
    hi = float(np.max(ends)) if ends else float(np.max(centers_t))
    return lo, hi


def _check_timed_tables(group: list[dict], strict: bool) -> None:
    timed: list[tuple[int, dict, float, float | None, float | None]] = []
    for idx, sol in enumerate(group):
        has_time, center, start, end = _table_time_info(sol)
        if has_time and center is not None:
            timed.append((idx, sol, float(center), start, end))
    if len(timed) <= 1:
        return

    issues: list[str] = []

    centers_in_input = np.asarray([x[2] for x in timed], dtype=np.float64)
    if np.any(np.diff(centers_in_input) < 0):
        issues.append('timed table centers are not in ascending input order (they will be internally sorted).')

    timed_sorted = sorted(timed, key=lambda x: x[2])
    centers = np.asarray([x[2] for x in timed_sorted], dtype=np.float64)
    if np.unique(centers).size != centers.size:
        issues.append('duplicate solution_time_center_jd values detected among timed tables.')

    for i in range(len(timed_sorted) - 1):
        _, sol_i, c_i, s_i, e_i = timed_sorted[i]
        _, sol_j, c_j, s_j, e_j = timed_sorted[i + 1]

        if s_i is not None and e_i is not None and e_i < s_i:
            issues.append(
                f'invalid interval in {sol_i.get("_path", "<table>")}: '
                f'start>end ({s_i:.8f} > {e_i:.8f} JD).'
            )
        if s_j is not None and e_j is not None and e_j < s_j:
            issues.append(
                f'invalid interval in {sol_j.get("_path", "<table>")}: '
                f'start>end ({s_j:.8f} > {e_j:.8f} JD).'
            )

        if e_i is not None and s_j is not None:
            if e_i > s_j:
                issues.append(
                    f'overlap between timed tables: {sol_i.get("_path", "<table>")} '
                    f'[{(s_i if s_i is not None else c_i):.8f} .. {e_i:.8f}] JD and '
                    f'{sol_j.get("_path", "<table>")} '
                    f'[{s_j:.8f} .. {(e_j if e_j is not None else c_j):.8f}] JD.'
                )

    if not issues:
        return

    msg = '[cal-apply] timed-table consistency check:\n  - ' + '\n  - '.join(issues)
    if strict:
        raise ValueError(msg)
    log.warning(msg)


def _nearest_indices(query_t: np.ndarray, centers_t: np.ndarray) -> np.ndarray:
    right = np.searchsorted(centers_t, query_t, side='left')
    right = np.clip(right, 0, centers_t.size - 1)
    left = np.clip(right - 1, 0, centers_t.size - 1)
    dleft = np.abs(query_t - centers_t[left])
    dright = np.abs(query_t - centers_t[right])
    return np.where(dleft <= dright, left, right).astype(np.int32)


def _nearest_interp(
    times_q: np.ndarray,
    centers_t: np.ndarray,
    gains_t: np.ndarray,
    valid_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    idx = _nearest_indices(times_q, centers_t)
    return gains_t[idx], valid_t[idx]


def _linear_interp(
    times_q: np.ndarray,
    centers_t: np.ndarray,
    gains_t: np.ndarray,
    valid_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    m = times_q.size
    out_g = np.empty((m,) + gains_t.shape[1:], dtype=np.complex128)
    out_v = np.zeros((m,) + valid_t.shape[1:], dtype=bool)

    right = np.searchsorted(centers_t, times_q, side='right')
    right = np.clip(right, 1, centers_t.size - 1)
    left = right - 1

    t0 = centers_t[left]
    t1 = centers_t[right]
    den = np.maximum(t1 - t0, np.finfo(np.float64).eps)
    alpha = ((times_q - t0) / den).astype(np.float64)
    alpha = np.clip(alpha, 0.0, 1.0)

    g0 = gains_t[left]
    g1 = gains_t[right]
    out_g[:] = (1.0 - alpha)[:, None, None, None] * g0 + alpha[:, None, None, None] * g1
    out_v[:] = valid_t[left] & valid_t[right]
    out_g[~out_v] = np.nan + 1j * np.nan
    return out_g, out_v


def _scipy_interp(
    times_q: np.ndarray,
    centers_t: np.ndarray,
    gains_t: np.ndarray,
    valid_t: np.ndarray,
    scheme: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.interpolate import Akima1DInterpolator, PchipInterpolator, interp1d
    except Exception:
        log.warning('[cal-apply] scipy unavailable; falling back to linear for %s interpolation.', scheme)
        return _linear_interp(times_q, centers_t, gains_t, valid_t)

    if scheme in {'linear', 'slinear', 'quadratic', 'cubic'}:
        kind = 'linear' if scheme == 'linear' else scheme
        min_points = {'linear': 2, 'slinear': 2, 'quadratic': 3, 'cubic': 4}[kind]
        if centers_t.size < min_points:
            log.warning('[cal-apply] scheme=%s needs >=%d points; falling back to linear.', scheme, min_points)
            return _linear_interp(times_q, centers_t, gains_t, valid_t)
        fn_re = interp1d(centers_t, np.real(gains_t), axis=0, kind=kind, bounds_error=False, fill_value=np.nan)
        fn_im = interp1d(centers_t, np.imag(gains_t), axis=0, kind=kind, bounds_error=False, fill_value=np.nan)
        out_g = fn_re(times_q) + 1j * fn_im(times_q)
        near_idx = _nearest_indices(times_q, centers_t)
        out_v = valid_t[near_idx]
        out_g[~out_v] = np.nan + 1j * np.nan
        return out_g.astype(np.complex128), out_v

    if scheme == 'pchip':
        if centers_t.size < 2:
            return _nearest_interp(times_q, centers_t, gains_t, valid_t)
        fn_re = PchipInterpolator(centers_t, np.real(gains_t), axis=0, extrapolate=True)
        fn_im = PchipInterpolator(centers_t, np.imag(gains_t), axis=0, extrapolate=True)
        out_g = fn_re(times_q) + 1j * fn_im(times_q)
        near_idx = _nearest_indices(times_q, centers_t)
        out_v = valid_t[near_idx]
        out_g[~out_v] = np.nan + 1j * np.nan
        return out_g.astype(np.complex128), out_v

    if scheme == 'akima':
        if centers_t.size < 2:
            return _nearest_interp(times_q, centers_t, gains_t, valid_t)
        fn_re = Akima1DInterpolator(centers_t, np.real(gains_t), axis=0, method='akima')
        fn_im = Akima1DInterpolator(centers_t, np.imag(gains_t), axis=0, method='akima')
        out_g = fn_re(times_q) + 1j * fn_im(times_q)
        near_idx = _nearest_indices(times_q, centers_t)
        out_v = valid_t[near_idx]
        out_g[~out_v] = np.nan + 1j * np.nan
        return out_g.astype(np.complex128), out_v

    return _linear_interp(times_q, centers_t, gains_t, valid_t)


def _evaluate_group_gains(
    times_q: np.ndarray,
    centers_t: np.ndarray,
    gains_t: np.ndarray,
    valid_t: np.ndarray,
    *,
    scheme: str,
    extrapolation: str,
    valid_time_domain: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    t_query = np.asarray(times_q, dtype=np.float64).copy()
    if valid_time_domain is None:
        outside = (t_query < centers_t[0]) | (t_query > centers_t[-1])
    else:
        outside = (t_query < float(valid_time_domain[0])) | (t_query > float(valid_time_domain[1]))

    if extrapolation in {'hold', 'nearest'}:
        t_query = np.clip(t_query, centers_t[0], centers_t[-1])

    if centers_t.size == 1:
        out_g = np.broadcast_to(gains_t[0], (t_query.size,) + gains_t.shape[1:]).copy()
        out_v = np.broadcast_to(valid_t[0], (t_query.size,) + valid_t.shape[1:]).copy()
    elif scheme == 'nearest':
        out_g, out_v = _nearest_interp(t_query, centers_t, gains_t, valid_t)
    elif scheme == 'linear':
        out_g, out_v = _linear_interp(t_query, centers_t, gains_t, valid_t)
    else:
        out_g, out_v = _scipy_interp(t_query, centers_t, gains_t, valid_t, scheme)

    if extrapolation == 'none' and np.any(outside):
        out_v[outside] = False
        out_g[outside] = np.nan + 1j * np.nan

    return out_g, out_v


def _stack_group_payload(group: list[dict], vis_labels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    chan_common = set(np.asarray(group[0]['chan_indices'], dtype=np.int32).tolist())
    ant_common = set(np.asarray(group[0]['antenna_ids'], dtype=np.int32).tolist())
    pol_common = set(str(x) for x in group[0]['stokes_labels'])

    for sol in group[1:]:
        chan_common &= set(np.asarray(sol['chan_indices'], dtype=np.int32).tolist())
        ant_common &= set(np.asarray(sol['antenna_ids'], dtype=np.int32).tolist())
        pol_common &= set(str(x) for x in sol['stokes_labels'])

    pol_common &= set(vis_labels)
    if not chan_common or not ant_common or not pol_common:
        raise ValueError('No common chan/ant/pol coverage between visibility and solution group.')

    chan_ref = np.asarray(group[0]['chan_indices'], dtype=np.int32)
    ant_ref = np.asarray(group[0]['antenna_ids'], dtype=np.int32)
    pol_ref = [str(x) for x in group[0]['stokes_labels']]

    chan_common_arr = np.asarray([int(c) for c in chan_ref.tolist() if int(c) in chan_common], dtype=np.int32)
    ant_common_arr = np.asarray([int(a) for a in ant_ref.tolist() if int(a) in ant_common], dtype=np.int32)
    pol_common_list = [p for p in pol_ref if p in pol_common]

    gains_stack: list[np.ndarray] = []
    valid_stack: list[np.ndarray] = []
    for sol in group:
        chan_sol = np.asarray(sol['chan_indices'], dtype=np.int32)
        ant_sol = np.asarray(sol['antenna_ids'], dtype=np.int32)
        pol_sol = [str(x) for x in sol['stokes_labels']]
        g = np.asarray(sol['gains'], dtype=np.complex128)
        v = np.asarray(sol['valid'], dtype=bool)

        chan_idx = np.asarray([int(np.where(chan_sol == c)[0][0]) for c in chan_common_arr], dtype=np.int32)
        ant_idx = np.asarray([int(np.where(ant_sol == a)[0][0]) for a in ant_common_arr], dtype=np.int32)
        pol_idx = np.asarray([int(pol_sol.index(p)) for p in pol_common_list], dtype=np.int32)

        gains_stack.append(g[np.ix_(chan_idx, ant_idx, pol_idx)])
        valid_stack.append(v[np.ix_(chan_idx, ant_idx, pol_idx)])

    gains_t = np.asarray(gains_stack, dtype=np.complex128)
    valid_t = np.asarray(valid_stack, dtype=bool)
    return chan_common_arr, ant_common_arr, gains_t, valid_t, pol_common_list


def _apply_group_inplace(
    corrected: np.ndarray,
    corrected_flagged: np.ndarray,
    *,
    jd: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    vis_chan_indices: np.ndarray,
    vis_labels: list[str],
    group: list[dict],
    scheme: str,
    extrapolation: str,
    chunk_rows: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    timed_tables: list[dict] = []
    timed_centers: list[float] = []
    untimed_tables: list[dict] = []
    for sol in group:
        has_time, center, _, _ = _table_time_info(sol)
        if has_time and center is not None:
            timed_tables.append(sol)
            timed_centers.append(float(center))
        else:
            untimed_tables.append(sol)

    # For mixed groups, apply untimed tables (all-time constants) first,
    # then time-aware tables. For diagonal/scalar gains this order is equivalent.
    apply_order: list[tuple[str, list[dict], np.ndarray | None]] = []
    if untimed_tables:
        for sol in untimed_tables:
            apply_order.append(('untimed', [sol], None))
    if timed_tables:
        centers_t = np.asarray(timed_centers, dtype=np.float64)
        order = np.argsort(centers_t)
        centers_t = centers_t[order]
        timed_tables = [timed_tables[int(i)] for i in order.tolist()]
        apply_order.append(('timed', timed_tables, centers_t))

    for mode, tables_i, centers_i in apply_order:
        chan_common, ant_common, gains_t, valid_t, pol_common = _stack_group_payload(tables_i, vis_labels)

        ant_to_idx = {int(a): i for i, a in enumerate(ant_common.tolist())}
        ant1_idx = np.asarray([ant_to_idx.get(int(a), -1) for a in ant1], dtype=np.int32)
        ant2_idx = np.asarray([ant_to_idx.get(int(a), -1) for a in ant2], dtype=np.int32)
        missing_row = (ant1_idx < 0) | (ant2_idx < 0)
        safe_ant1 = np.where(ant1_idx >= 0, ant1_idx, 0)
        safe_ant2 = np.where(ant2_idx >= 0, ant2_idx, 0)

        nrows = corrected.shape[0]
        chan_map = {int(c): i for i, c in enumerate(chan_common.tolist())}
        nchan = corrected.shape[1]
        vis_chan = np.asarray(vis_chan_indices, dtype=np.int32)
        if vis_chan.size != nchan:
            raise ValueError(
                f'vis_chan_indices length {vis_chan.size} does not match vis channel axis {nchan}.'
            )
        chan_sel_idx = np.full(nchan, -1, dtype=np.int32)
        for vc in range(nchan):
            chan_sel_idx[vc] = chan_map.get(int(vis_chan[vc]), -1)

        if np.any(chan_sel_idx < 0):
            raise ValueError('Solution group does not cover all selected visibility channels.')

        if mode == 'untimed':
            centers_t = np.asarray([0.0], dtype=np.float64)
            gains_t = gains_t[:1]
            valid_t = valid_t[:1]
            domain = None
        else:
            assert centers_i is not None
            centers_t = np.asarray(centers_i, dtype=np.float64)
            domain = _timed_domain_jd(tables_i, centers_t)

        for start in range(0, nrows, chunk_rows):
            stop = min(start + chunk_rows, nrows)
            sl = slice(start, stop)

            g_chunk, v_chunk = _evaluate_group_gains(
                jd[sl],
                centers_t,
                gains_t,
                valid_t,
                scheme=scheme,
                extrapolation=extrapolation,
                valid_time_domain=domain,
            )
            g_chunk = g_chunk[:, chan_sel_idx, :, :]
            v_chunk = v_chunk[:, chan_sel_idx, :, :]

            local_missing = missing_row[sl]
            local_a1 = safe_ant1[sl]
            local_a2 = safe_ant2[sl]

            for vis_pol_idx, label in enumerate(vis_labels):
                if label not in pol_common:
                    continue
                pol_idx = pol_common.index(label)
                gp = g_chunk[:, :, :, pol_idx]
                vp = v_chunk[:, :, :, pol_idx]

                idx1 = np.broadcast_to(local_a1[:, None, None], (stop - start, nchan, 1))
                idx2 = np.broadcast_to(local_a2[:, None, None], (stop - start, nchan, 1))
                g1 = np.take_along_axis(gp, idx1, axis=2).squeeze(axis=2)
                g2 = np.take_along_axis(gp, idx2, axis=2).squeeze(axis=2)
                v1 = np.take_along_axis(vp, idx1, axis=2).squeeze(axis=2)
                v2 = np.take_along_axis(vp, idx2, axis=2).squeeze(axis=2)

                denom = g1 * np.conj(g2)
                good = v1 & v2 & np.isfinite(np.real(denom)) & np.isfinite(np.imag(denom)) & (np.abs(denom) > 0.0)
                if np.any(local_missing):
                    good[local_missing] = False

                cell = corrected[sl, :, vis_pol_idx]
                cell[good] = cell[good] / denom[good]
                cell[~good] = np.nan + 1j * np.nan
                corrected[sl, :, vis_pol_idx] = cell
                corrected_flagged[sl, :, vis_pol_idx] |= ~good

    corrected[corrected_flagged] = np.nan + 1j * np.nan
    return corrected, corrected_flagged


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
    max_rows = int(args.max_rows) if args.max_rows is not None else int(globals().get('MAX_ROWS_DIAG', 60_000))

    timerange = globals().get('SOLVE_TIMERANGE', None)
    elevation_min = globals().get('SOLVE_ELEVATION_MIN_DEG', None)
    elevation_max = globals().get('SOLVE_ELEVATION_MAX_DEG', None)
    uvrange_m = globals().get('SOLVE_UVRANGE_M', None)
    uvrange_kl = globals().get('SOLVE_UVRANGE_KLAMBDA', None)

    index_cache = _derive_index_cache(fits_path=fits_path, explicit_index_cache=(Path(args.index_cache) if args.index_cache else None))
    index = q.get_or_build_row_index(
        fits_path,
        cache_path=index_cache,
        force_rebuild=False,
        validation_mode=INDEX_VALIDATION_MODE,
        write_cache=True,
    )

    vis = q.load_vis_for_source(
        index,
        source=source,
        stokes=list(STOKES),
        chan_range=chan_range,
        max_rows=max_rows,
        flag_all_corrs_if_any_rawvis_flagged=FLAG_ALL_CORRS_IF_ANY_RAWVIS_FLAGGED,
        timerange=timerange,
        uvrange_m=uvrange_m,
        uvrange_klambda=uvrange_kl,
        elevation_min_deg=elevation_min,
        elevation_max_deg=elevation_max,
    )

    table_paths: list[Path] = []
    for p in args.tables:
        rp = Path(p).expanduser().resolve()
        if not rp.exists():
            sys.exit(f'ERROR: table not found: {rp}')
        table_paths.append(rp)

    solutions: list[dict] = []
    for p in table_paths:
        sol = q.load_bandpass_solution(p)
        sol['_path'] = str(p)
        solutions.append(sol)

    groups = _group_tables_in_order(solutions)
    corrected = np.asarray(vis['vis_complex'], dtype=np.complex128).copy()
    corrected_flagged = np.asarray(vis['flagged'], dtype=bool).copy()
    vis_labels = [str(x) for x in vis['stokes_labels']]

    log.info(
        '[cal-apply] source=%s tables=%d groups=%d time_interp_scheme=%s extrapolation=%s',
        source,
        len(table_paths),
        len(groups),
        args.time_interp_scheme,
        args.time_extrapolation,
    )
    if args.out is not None:
        log.warning(
            '[cal-apply] --out is currently ignored. '
            'TODO: support writing a merged calibration table product; '
            'current mode is in-memory apply/check only.'
        )

    for gi, group in enumerate(groups, 1):
        _check_timed_tables(group, strict=bool(args.strict_time_checks))
        log.info('[cal-apply] applying group %d/%d (n_tables=%d)', gi, len(groups), len(group))
        corrected, corrected_flagged = _apply_group_inplace(
            corrected,
            corrected_flagged,
            jd=np.asarray(vis['jd'], dtype=np.float64),
            ant1=np.asarray(vis['ant1'], dtype=np.int32),
            ant2=np.asarray(vis['ant2'], dtype=np.int32),
            vis_chan_indices=np.asarray(vis['chan_indices'], dtype=np.int32),
            vis_labels=vis_labels,
            group=group,
            scheme=args.time_interp_scheme,
            extrapolation=args.time_extrapolation,
        )

    frac_flagged = float(np.count_nonzero(corrected_flagged)) / float(max(corrected_flagged.size, 1))
    print('[cal-apply] done (in-memory apply only; no vis written)')
    print(
        f'[cal-apply] source={source} rows={vis["nrows"]} tables={len(table_paths)} '
        f'interp={args.time_interp_scheme} extrapolation={args.time_extrapolation} '
        f'flagged_fraction={100.0 * frac_flagged:.2f}%'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
