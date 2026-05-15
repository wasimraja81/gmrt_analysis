#!/usr/bin/env python3
"""
Generalized plotVis utility.

Single multi-panel figure with selectable products and selectors.
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from modules import cal_apply as ca
from modules import ugmrt_query as q
from modules.workflow_common import apply_overrides_to_globals, load_config_into_globals


# Optional config-backed defaults (only used when --config is provided)
CAL_FITS = None
INDEX_CACHE = None
SOURCE = None
CHAN_RANGE = None
SOLVE_TIMERANGE = None
SOLVE_ELEVATION_MIN_DEG = None
SOLVE_ELEVATION_MAX_DEG = None
SOLVE_UVRANGE_M = None
SOLVE_UVRANGE_KLAMBDA = None
EXCLUDE_FOR_PLOTS = []


def _parse_csv_list(text):
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        raw_items = []
        for item in text:
            if item is None:
                continue
            raw_items.extend(str(item).split(','))
        return [x.strip() for x in raw_items if str(x).strip()]
    if str(text).strip() == '':
        return []
    return [x.strip() for x in str(text).split(',') if x.strip()]


def _sanitize_for_filename(text):
    s = str(text).strip().lower()
    s = re.sub(r'[^a-z0-9._-]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or 'source'


def _build_ant_maps(index):
    ant_name_map = {
        int(a['antenna_no']): str(a['name'])
        for a in index.get('antennas', [])
        if a.get('antenna_no') is not None
    }
    name_to_id = {v: k for k, v in ant_name_map.items()}
    return ant_name_map, name_to_id


def _resolve_ant_selector(item, ant_name_map, name_to_id):
    s = str(item).strip()
    if s == '':
        return None
    if s.isdigit():
        return int(s)
    if s in name_to_id:
        return int(name_to_id[s])
    for aid, name in ant_name_map.items():
        if name == s or name.startswith(s + ':') or name.startswith(s):
            return int(aid)
    return None


def _parse_baselines(spec, ant_name_map, name_to_id):
    out = set()
    for token in _parse_csv_list(spec):
        if '-' not in token:
            continue
        a, b = token.split('-', 1)
        a_id = _resolve_ant_selector(a, ant_name_map, name_to_id)
        b_id = _resolve_ant_selector(b, ant_name_map, name_to_id)
        if a_id is None or b_id is None:
            continue
        out.add(tuple(sorted((int(a_id), int(b_id)))))
    return out


def _row_filter_by_baselines(vis, baseline_pairs):
    if not baseline_pairs:
        return np.ones(len(vis['ant1']), dtype=bool)
    a1 = np.asarray(vis['ant1'], dtype=np.int32)
    a2 = np.asarray(vis['ant2'], dtype=np.int32)
    lo = np.minimum(a1, a2)
    hi = np.maximum(a1, a2)
    return np.array([(int(x), int(y)) in baseline_pairs for x, y in zip(lo, hi)], dtype=bool)


def _row_exclude_baselines(vis, exclude_pairs):
    """Return vis with rows belonging to any of the given baseline pairs removed."""
    if not exclude_pairs:
        return vis
    a1 = np.asarray(vis['ant1'], dtype=np.int32)
    a2 = np.asarray(vis['ant2'], dtype=np.int32)
    lo = np.minimum(a1, a2)
    hi = np.maximum(a1, a2)
    keep = np.array([(int(x), int(y)) not in exclude_pairs for x, y in zip(lo, hi)], dtype=bool)
    n_removed = int(np.count_nonzero(~keep))
    if n_removed:
        print(f'[plotVis] --exclude-baselines: removed {n_removed} rows ({len(exclude_pairs)} pair(s))')
    return _slice_rows(vis, keep)


def _row_exclude_antennas(vis, exclude_ant_ids):
    """Return vis with rows involving any of the given antenna IDs (0-based) removed."""
    if not exclude_ant_ids:
        return vis
    exclude_set = set(int(x) for x in exclude_ant_ids)
    a1 = np.asarray(vis['ant1'], dtype=np.int32)
    a2 = np.asarray(vis['ant2'], dtype=np.int32)
    keep = np.array(
        [(int(x) not in exclude_set) and (int(y) not in exclude_set)
         for x, y in zip(a1, a2)], dtype=bool
    )
    n_removed = int(np.count_nonzero(~keep))
    if n_removed:
        print(f'[plotVis] --exclude-antennas: removed {n_removed} rows (excluded ids={sorted(exclude_set)})')
    return _slice_rows(vis, keep)


def _slice_rows(vis, row_keep):
    nrows = len(vis['ant1'])
    out = {}
    for k, v in vis.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == nrows:
            out[k] = v[row_keep]
        else:
            out[k] = v
    return out


def _load_solution(bpcal_path, source_name, index):
    try:
        return q.load_bandpass_solution(bpcal_path)
    except Exception as exc:
        print(f"[plotVis] WARNING: load_bandpass_solution failed ({exc}); fallback raw NPZ.")
        with np.load(bpcal_path, allow_pickle=True) as npz:
            solution = {k: npz[k] for k in npz.files}
        if 'source_name' not in solution:
            solution['source_name'] = str(source_name)
        if 'reference_antenna' not in solution:
            ants = [int(a['antenna_no']) for a in index.get('antennas', []) if a.get('antenna_no') is not None]
            solution['reference_antenna'] = int(min(ants)) if ants else 1
        if 'stokes_labels' in solution:
            solution['stokes_labels'] = [str(x) for x in np.asarray(solution['stokes_labels']).astype(str).tolist()]
        return solution


def _safe_float(value):
    try:
        if value is None:
            return None
        out = float(value)
        return out if np.isfinite(out) else None
    except Exception:
        return None


def _solution_time_info(solution):
    center = _safe_float(solution.get('solution_time_center_jd', None))
    start = _safe_float(solution.get('solution_time_start_jd', None))
    end = _safe_float(solution.get('solution_time_end_jd', None))
    return center, start, end


def _solution_center_or_inf(solution):
    center = _solution_time_info(solution)[0]
    return float(center) if center is not None else float('inf')


def _apply_solutions_by_time(vis, solutions, time_interp_scheme='nearest', time_extrapolation='hold'):
    if not solutions:
        return None
    groups = ca._group_tables_in_order(solutions)

    n_tables = len(solutions)
    n_timed = sum(1 for s in solutions if _solution_time_info(s)[0] is not None)
    n_untimed = n_tables - n_timed
    table_names = [Path(str(s.get('_path', f'table_{i+1}'))).name for i, s in enumerate(solutions)]
    print(
        f'[plotVis] calibration apply: tables={n_tables} '
        f'(untimed={n_untimed}, timed={n_timed}) groups={len(groups)} '
        f'interp={time_interp_scheme} extrap={time_extrapolation}'
    )
    if table_names:
        print(f"[plotVis] calibration tables: {', '.join(table_names)}")

    corrected = np.asarray(vis['vis_complex'], dtype=np.complex128).copy()
    corrected_flagged = np.asarray(vis.get('flagged', np.zeros_like(vis['weight'], dtype=bool)), dtype=bool).copy()
    vis_labels = [str(x) for x in vis['stokes_labels']]

    for gi, group in enumerate(groups, 1):
        g_timed = sum(1 for s in group if _solution_time_info(s)[0] is not None)
        g_untimed = len(group) - g_timed
        print(f'[plotVis] applying group {gi}/{len(groups)}: n_tables={len(group)} (untimed={g_untimed}, timed={g_timed})')
        ca._check_timed_tables(group, strict=False)
        corrected, corrected_flagged = ca._apply_group_inplace(
            corrected,
            corrected_flagged,
            jd=np.asarray(vis['jd'], dtype=np.float64),
            ant1=np.asarray(vis['ant1'], dtype=np.int32),
            ant2=np.asarray(vis['ant2'], dtype=np.int32),
            vis_chan_indices=np.asarray(vis['chan_indices'], dtype=np.int32),
            vis_labels=vis_labels,
            group=group,
            scheme=str(time_interp_scheme),
            extrapolation=str(time_extrapolation),
        )

    raw = np.asarray(vis['vis_complex'], dtype=np.complex128)
    raw_flagged = np.asarray(vis.get('flagged', np.zeros_like(vis['weight'], dtype=bool)), dtype=bool)
    finite_raw = np.isfinite(raw.real) & np.isfinite(raw.imag)
    finite_corr = np.isfinite(np.real(corrected)) & np.isfinite(np.imag(corrected))
    valid = (~raw_flagged) & (~corrected_flagged) & finite_raw & finite_corr
    n_valid = int(np.count_nonzero(valid))
    if n_valid > 0:
        raw_v = raw[valid]
        corr_v = corrected[valid]
        nz = np.abs(raw_v) > 1e-12
        if np.any(nz):
            amp_ratio = np.abs(corr_v[nz]) / np.abs(raw_v[nz])
            phase_shift_deg = np.degrees(np.angle(corr_v[nz] * np.conj(raw_v[nz])))
            print(
                '[plotVis] correction summary: '
                f'valid_cells={n_valid:,} '
                f'median_amp_ratio={float(np.nanmedian(amp_ratio)):.4f} '
                f'median_abs_phase_shift_deg={float(np.nanmedian(np.abs(phase_shift_deg))):.3f}'
            )
        else:
            print(f'[plotVis] correction summary: valid_cells={n_valid:,} (all raw valid cells have near-zero amplitude)')
    else:
        print('[plotVis] correction summary: no valid overlapping raw/corrected cells for delta statistics')

    amp = np.abs(corrected)
    phase = np.degrees(np.angle(corrected)).astype(np.float32)
    amp[corrected_flagged] = np.nan
    phase[corrected_flagged] = np.nan
    corrected[corrected_flagged] = np.nan + 1j * np.nan

    out = dict(vis)
    out['vis_complex_corrected'] = corrected
    out['amp_corrected'] = amp
    out['phase_deg_corrected'] = phase
    out['flagged_corrected'] = corrected_flagged
    out['applied_bandpass_reference_antenna'] = int(solutions[0].get('reference_antenna', -1))
    out['applied_bandpass_source'] = str(solutions[0].get('source_name', 'unknown'))
    return out


def _get_product_data(product, vis, corrected_cache, solution):
    labels = list(vis.get('stokes_labels', []))
    # Normalise case so that e.g. 'rr' matches stored label 'RR'
    if product not in labels and product.upper() in labels:
        product = product.upper()
    if str(product).upper() in ('I', 'Q', 'U', 'V'):
        product = str(product).upper()

    if product in labels:
        idx = labels.index(product)
        if corrected_cache is not None:
            z = np.asarray(corrected_cache['vis_complex_corrected'][:, :, idx], dtype=np.complex128)
            f = np.asarray(corrected_cache.get('flagged_corrected', corrected_cache.get('flagged'))[:, :, idx], dtype=bool)
        else:
            z = np.asarray(vis['vis_complex'][:, :, idx], dtype=np.complex128)
            f = np.asarray(vis.get('flagged', np.zeros_like(vis['weight'], dtype=bool))[:, :, idx], dtype=bool)
        return z, f

    if product in ('I', 'Q', 'U', 'V'):
        if corrected_cache is not None:
            vc = np.asarray(corrected_cache['vis_complex_corrected'], dtype=np.complex128)
            fc = np.asarray(corrected_cache.get('flagged_corrected', corrected_cache.get('flagged')), dtype=bool)
        else:
            # No bpcal supplied — assume the loaded visibilities are already calibrated
            # (e.g. a pre-split calibrated UVFITS).  Only I and V are well-defined
            # without cross-correlations; Q and U will still fail below if RL/LR absent.
            vc = np.asarray(vis['vis_complex'], dtype=np.complex128)
            fc = np.asarray(vis.get('flagged', np.zeros(vc.shape[:2], dtype=bool)), dtype=bool)
            if fc.ndim == 2:
                fc = fc[:, :, np.newaxis] * np.ones((1, 1, vc.shape[2]), dtype=bool)

        def _corr(pol: str):
            if pol not in labels:
                return None, None
            idx = labels.index(pol)
            return vc[:, :, idx], fc[:, :, idx]

        circ_present = {p for p in ('RR', 'LL', 'RL', 'LR') if p in labels}
        lin_present = {p for p in ('XX', 'YY', 'XY', 'YX') if p in labels}

        if circ_present:
            RR, fRR = _corr('RR')
            LL, fLL = _corr('LL')
            RL, fRL = _corr('RL')
            LR, fLR = _corr('LR')

            if product == 'I':
                if RR is None or LL is None:
                    raise ValueError(f'Stokes I (circular basis) requires RR+LL; available correlations: {labels}')
                z = (RR + LL) / 2.0
                f = np.asarray(fRR, dtype=bool) | np.asarray(fLL, dtype=bool)
            elif product == 'V':
                if RR is None or LL is None:
                    raise ValueError(f'Stokes V (circular basis) requires RR+LL; available correlations: {labels}')
                z = (RR - LL) / 2.0
                f = np.asarray(fRR, dtype=bool) | np.asarray(fLL, dtype=bool)
            elif product == 'Q':
                if RL is None or LR is None:
                    raise ValueError(
                        f'Stokes Q (circular basis) requires RL+LR; available correlations: {labels} '
                        '(dual-circ RR+LL only cannot produce Q/U)'
                    )
                z = (RL + LR) / 2.0
                f = np.asarray(fRL, dtype=bool) | np.asarray(fLR, dtype=bool)
            else:  # U
                if RL is None or LR is None:
                    raise ValueError(
                        f'Stokes U (circular basis) requires RL+LR; available correlations: {labels} '
                        '(dual-circ RR+LL only cannot produce Q/U)'
                    )
                z = 1j * (RL - LR) / 2.0
                f = np.asarray(fRL, dtype=bool) | np.asarray(fLR, dtype=bool)

        elif lin_present:
            XX, fXX = _corr('XX')
            YY, fYY = _corr('YY')
            XY, fXY = _corr('XY')
            YX, fYX = _corr('YX')

            if product == 'I':
                if XX is None or YY is None:
                    raise ValueError(f'Stokes I (linear basis) requires XX+YY; available correlations: {labels}')
                z = (XX + YY) / 2.0
                f = np.asarray(fXX, dtype=bool) | np.asarray(fYY, dtype=bool)
            elif product == 'Q':
                if XX is None or YY is None:
                    raise ValueError(f'Stokes Q (linear basis) requires XX+YY; available correlations: {labels}')
                z = (XX - YY) / 2.0
                f = np.asarray(fXX, dtype=bool) | np.asarray(fYY, dtype=bool)
            elif product == 'U':
                if XY is None or YX is None:
                    raise ValueError(
                        f'Stokes U (linear basis) requires XY+YX; available correlations: {labels} '
                        '(dual-lin XX+YY only cannot produce U/V)'
                    )
                z = (XY + YX) / 2.0
                f = np.asarray(fXY, dtype=bool) | np.asarray(fYX, dtype=bool)
            else:  # V
                if XY is None or YX is None:
                    raise ValueError(
                        f'Stokes V (linear basis) requires XY+YX; available correlations: {labels} '
                        '(dual-lin XX+YY only cannot produce U/V)'
                    )
                z = 1j * (YX - XY) / 2.0
                f = np.asarray(fXY, dtype=bool) | np.asarray(fYX, dtype=bool)
        else:
            raise ValueError(
                f'Cannot determine polarization basis from correlations {labels}; '
                'expected circular (RR/LL/RL/LR) or linear (XX/YY/XY/YX).'
            )

        finite = np.isfinite(z.real) & np.isfinite(z.imag)
        f = np.asarray(f, dtype=bool) | (~finite)
        z = np.asarray(z, dtype=np.complex128)
        z[f] = np.nan + 1j * np.nan
        return z, f

    raise ValueError(f'Unsupported product: {product}. Use corr labels in vis or I,Q,U,V.')


def _sample_mask(shape, frac, seed=0):
    if frac >= 1.0:
        return np.ones(shape, dtype=bool)
    rng = np.random.default_rng(seed)
    return rng.random(shape) < max(0.0, frac)


def _parse_panels_per_page(text):
    s = str(text).strip().lower().replace(' ', '')
    if 'x' in s:
        a, b = s.split('x', 1)
        nrows = int(a)
        ncols = int(b)
    elif ',' in s:
        a, b = s.split(',', 1)
        nrows = int(a)
        ncols = int(b)
    else:
        n = int(s)
        if n <= 1:
            nrows, ncols = 1, 1
        elif n <= 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 3, 2
    if nrows <= 0 or ncols <= 0:
        raise ValueError('--panels-per-page must be positive, e.g. 3x2')
    return nrows, ncols, nrows * ncols


def _overlay_flag_fraction_curve(ax, x_values, flag_mask, nbins=80):
    x = np.asarray(x_values, dtype=np.float64).ravel()
    f = np.asarray(flag_mask, dtype=bool).ravel()
    if x.size == 0 or f.size == 0 or x.size != f.size:
        return None
    finite_x = np.isfinite(x)
    if not np.any(finite_x):
        return None
    x = x[finite_x]
    f = f[finite_x]
    if x.size == 0:
        return None
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return None

    bins = max(8, int(nbins))
    edges = np.linspace(xmin, xmax, bins + 1)
    which = np.digitize(x, edges) - 1
    ok = (which >= 0) & (which < bins)
    if not np.any(ok):
        return None
    total = np.bincount(which[ok], minlength=bins).astype(np.float64)
    flagged = np.bincount(which[ok], weights=f[ok].astype(np.float64), minlength=bins)
    frac = np.divide(flagged, total, out=np.zeros_like(flagged), where=total > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ax2 = ax.twinx()
    ax2.step(centers, frac, where='mid', color='red', lw=1.1, alpha=0.9)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel('Flag fraction', color='red')
    ax2.tick_params(axis='y', colors='red', labelsize=7)
    return ax2


def _robust_quantile_bounds(values, lo=2.0, hi=98.0):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    vmin = float(np.nanpercentile(vals, lo))
    vmax = float(np.nanpercentile(vals, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None
    if vmax <= vmin:
        span = max(abs(vmin), 1.0)
        vmin -= 1e-6 * span
        vmax += 1e-6 * span
    return vmin, vmax


def _scale_marker_sizes(values, smin=0.08, smax=1.0):
    vals = np.asarray(values, dtype=np.float64)
    out = np.full(vals.shape, float(smin), dtype=np.float64)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return out
    vmin, vmax = _robust_quantile_bounds(vals[finite], lo=2.0, hi=98.0)
    if vmin is None or vmax is None or vmax <= vmin:
        return out
    normed = (vals[finite] - vmin) / (vmax - vmin)
    normed = np.clip(normed, 0.0, 1.0)
    out[finite] = smin + normed * (smax - smin)
    return out


def _flag_fraction_inputs_3d(vis, corrected_cache, axis, time_axis_row, uvd, freq_mhz):
    flag_cube = None
    if corrected_cache is not None:
        flag_cube = corrected_cache.get('flagged_corrected', corrected_cache.get('flagged'))
    if flag_cube is None:
        flag_cube = vis.get('flagged')
    if flag_cube is None:
        return None, None

    flag_cube = np.asarray(flag_cube, dtype=bool)
    if flag_cube.ndim != 3:
        return None, None
    nrows, nchans, npol = flag_cube.shape

    if axis == 'time':
        x2 = np.repeat(time_axis_row[:nrows], nchans).reshape(nrows, nchans)
    elif axis == 'freq':
        x2 = np.tile(freq_mhz[:nchans], nrows).reshape(nrows, nchans)
    else:
        x2 = np.repeat(uvd[:nrows], nchans).reshape(nrows, nchans)

    x3 = np.repeat(x2[:, :, np.newaxis], npol, axis=2)
    return x3.ravel(), flag_cube.ravel()


def _add_channel_top_axis(ax, freq_mhz, chan_numbers, nbins=7):
    f = np.asarray(freq_mhz, dtype=np.float64)
    c = np.asarray(chan_numbers, dtype=np.float64)
    if f.size < 2 or c.size != f.size:
        return

    if f[0] <= f[-1]:
        f_ref, c_ref = f, c
    else:
        f_ref, c_ref = f[::-1], c[::-1]

    def mhz_to_chan(x):
        x = np.asarray(x, dtype=np.float64)
        return np.interp(x, f_ref, c_ref)

    def chan_to_mhz(x):
        x = np.asarray(x, dtype=np.float64)
        return np.interp(x, c_ref, f_ref)

    top = ax.secondary_xaxis('top', functions=(mhz_to_chan, chan_to_mhz))
    top.set_xlabel('Channel')
    top.xaxis.set_major_locator(MaxNLocator(nbins=max(3, int(nbins)), integer=True, prune='both'))


def _infer_original_chan_numbers(index, vis_freqs_hz):
    vis_freqs_hz = np.asarray(vis_freqs_hz, dtype=np.float64)
    full = np.asarray(index.get('chan_freqs_hz', []), dtype=np.float64)
    if vis_freqs_hz.size == 0:
        return np.array([], dtype=np.int32)
    if full.size == 0:
        return np.arange(vis_freqs_hz.size, dtype=np.int32)

    chan_numbers = np.full(vis_freqs_hz.size, -1, dtype=np.int32)
    for i, vf in enumerate(vis_freqs_hz):
        m = np.where(np.isclose(full, vf, rtol=0.0, atol=1e-3))[0]
        if m.size >= 1:
            chan_numbers[i] = int(m[0])

    if np.any(chan_numbers < 0):
        if vis_freqs_hz.size == full.size and np.allclose(vis_freqs_hz, full, rtol=0.0, atol=1e-6):
            return np.arange(full.size, dtype=np.int32)
        good = chan_numbers >= 0
        if np.any(good):
            x = np.arange(chan_numbers.size)
            chan_numbers[~good] = np.rint(np.interp(x[~good], x[good], chan_numbers[good])).astype(np.int32)
        else:
            chan_numbers = np.arange(vis_freqs_hz.size, dtype=np.int32)
    return chan_numbers


def _compute_row_azel(index, source_name, jd_rows):
    jd_rows = np.asarray(jd_rows, dtype=np.float64)
    finite = np.isfinite(jd_rows)
    if not np.any(finite):
        return None, None

    jd_use = jd_rows[finite]
    if jd_use.size < 2:
        return None, None

    jd_sorted = np.sort(jd_use)
    dt = np.diff(jd_sorted)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size:
        time_step_s = float(np.clip(np.nanmedian(dt) * 86400.0, 1.0, 120.0))
    else:
        time_step_s = 30.0

    try:
        azel = q.compute_source_azel(
            index,
            source=str(source_name),
            timerange=(float(np.nanmin(jd_use)), float(np.nanmax(jd_use))),
            time_step_s=time_step_s,
        )
    except Exception as exc:
        print(f'[plotVis] WARNING: could not compute Az/El track for source={source_name!r}: {exc}')
        return None, None

    jd_track = np.asarray(azel.get('jd', []), dtype=np.float64)
    az_track = np.asarray(azel.get('az_deg', []), dtype=np.float64)
    el_track = np.asarray(azel.get('el_deg', []), dtype=np.float64)
    good = np.isfinite(jd_track) & np.isfinite(az_track) & np.isfinite(el_track)
    if not np.any(good):
        return None, None

    jd_track = jd_track[good]
    az_track = az_track[good]
    el_track = el_track[good]
    order = np.argsort(jd_track)
    jd_track = jd_track[order]
    az_track = az_track[order]
    el_track = el_track[order]

    if jd_track.size < 2:
        return None, None

    az_row = np.full(jd_rows.shape, np.nan, dtype=np.float64)
    el_row = np.full(jd_rows.shape, np.nan, dtype=np.float64)
    az_row[finite] = np.interp(jd_rows[finite], jd_track, az_track)
    el_row[finite] = np.interp(jd_rows[finite], jd_track, el_track)
    return az_row, el_row


def _compute_sampling_stats(products, vis, corrected_cache, solution, sample_frac, shared_sample_mask):
    stats_by_product = {}
    total_good = 0
    total_good_sampled = 0
    total_flag = 0
    total_flag_sampled = 0
    total_flag_sampled_plottable = 0
    total_cells = 0
    total_unflagged_nonfinite = 0

    for ci, prod in enumerate(products):
        z, f = _get_product_data(prod, vis, corrected_cache, solution)
        finite = np.isfinite(z.real) & np.isfinite(z.imag)
        good = (~f) & finite
        flagged = f
        unflagged_nonfinite = (~f) & (~finite)
        if shared_sample_mask is not None and shared_sample_mask.shape == z.shape:
            sm = shared_sample_mask
        else:
            sm = _sample_mask(z.shape, sample_frac, seed=100)
        good_s = good & sm
        flag_s = flagged & sm
        flag_s_plot = flag_s & finite

        g_tot = int(np.count_nonzero(good))
        g_s = int(np.count_nonzero(good_s))
        f_tot = int(np.count_nonzero(flagged))
        f_s = int(np.count_nonzero(flag_s))
        f_s_plot = int(np.count_nonzero(flag_s_plot))
        total_n = int(z.size)
        unflag_nf = int(np.count_nonzero(unflagged_nonfinite))

        stats_by_product[prod] = {
            'good_total': g_tot,
            'good_sampled': g_s,
            'flag_total': f_tot,
            'flag_sampled': f_s,
            'flag_sampled_plottable': f_s_plot,
            'total_cells': total_n,
            'unflagged_nonfinite_total': unflag_nf,
        }

        total_good += g_tot
        total_good_sampled += g_s
        total_flag += f_tot
        total_flag_sampled += f_s
        total_flag_sampled_plottable += f_s_plot
        total_cells += total_n
        total_unflagged_nonfinite += unflag_nf

    totals = {
        'good_total': total_good,
        'good_sampled': total_good_sampled,
        'flag_total': total_flag,
        'flag_sampled': total_flag_sampled,
        'flag_sampled_plottable': total_flag_sampled_plottable,
        'total_cells': total_cells,
        'unflagged_nonfinite_total': total_unflagged_nonfinite,
    }
    return stats_by_product, totals


def _plot_panel(
    ax,
    panel,
    products,
    vis,
    corrected_cache,
    solution,
    args,
    time_axis_row,
    uvd,
    freq_mhz,
    chan_numbers,
    product_colors,
    shared_sample_mask,
    az_row=None,
    el_row=None,
):
    panel = '' if panel is None else str(panel)
    panel_alias = {
        'uvdist': 'amp_uvdist',
        'real_time': 'real_time',
        'imag_freq': 'imag_freq',
        'uv': 'uv_sampling',
        'uvcov': 'uv_sampling',
    }
    panel = panel_alias.get(panel, panel)

    uvcov_metric = None
    uvcov_product = None
    if panel.startswith('uvcov_'):
        parts = panel.split('_')
        if len(parts) == 2:
            uvcov_metric = parts[1].lower()
        elif len(parts) == 3:
            uvcov_product = parts[1].upper()
            uvcov_metric = parts[2].lower()
        if uvcov_metric == 'pha':
            uvcov_metric = 'phase'
        if uvcov_metric in ('amp', 'real', 'imag', 'phase'):
            panel = 'uvcov_metric'
        else:
            uvcov_metric = None
            uvcov_product = None

    supported_scalar = {
        ('amp', 'time'), ('amp', 'freq'), ('amp', 'uvdist'),
        ('real', 'time'), ('real', 'freq'), ('real', 'uvdist'),
        ('imag', 'time'), ('imag', 'freq'), ('imag', 'uvdist'),
        ('phase', 'time'), ('phase', 'freq'), ('phase', 'uvdist'),
    }

    quantity = None
    axis = None
    if '_' in panel:
        qn, axn = panel.split('_', 1)
        if (qn, axn) in supported_scalar:
            quantity = qn
            axis = axn

    if panel in ('az_time', 'el_time'):
        y = az_row if panel == 'az_time' else el_row
        if y is None:
            label = 'azimuth' if panel == 'az_time' else 'elevation'
            ax.text(0.5, 0.5, f'No {label} track available', ha='center', va='center')
            ax.set_title(panel)
            ax.grid(True, alpha=0.25)
            return

        y = np.asarray(y, dtype=np.float64)
        good = np.isfinite(y) & np.isfinite(time_axis_row)
        if not np.any(good):
            ax.text(0.5, 0.5, f'No valid {panel} samples', ha='center', va='center')
            ax.set_title(panel)
            ax.grid(True, alpha=0.25)
            return

        xg = np.asarray(time_axis_row[good], dtype=np.float64)
        yg = y[good]
        order = np.argsort(xg)
        xg = xg[order]
        yg = yg[order]

        ax.plot(xg, yg, lw=1.3, c='tab:cyan' if panel == 'el_time' else 'tab:orange')
        ax.scatter(xg, yg, s=2.0, alpha=0.25, c='black')
        if args.time_format in ('isot_concise', 'isot_full'):
            ax.set_xlabel('UTC Time')
            ax.xaxis_date()
            if args.time_format == 'isot_full':
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%dT%H:%M:%S'))
            else:
                locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.tick_params(axis='x', labelrotation=25)
        else:
            ax.set_xlabel('Time from start (min)')

        if panel == 'az_time':
            ax.set_ylabel('Azimuth (deg)')
            ax.set_ylim(0.0, 360.0)
        else:
            ax.set_ylabel('Elevation (deg)')
            ylo = max(0.0, float(np.nanmin(yg)) - 2.0)
            yhi = min(90.0, float(np.nanmax(yg)) + 2.0)
            if yhi <= ylo:
                ylo, yhi = 0.0, 90.0
            ax.set_ylim(ylo, yhi)

        ax.set_title(panel)
        ax.grid(True, alpha=0.25)
        return

    product_payload = []
    for ci, prod in enumerate(products):
        z, f = _get_product_data(prod, vis, corrected_cache, solution)
        product_payload.append((ci, prod, z, f))

    if panel == 'uvcov_metric':
        uu_sec = np.asarray(vis['uu_sec'], dtype=np.float64)
        vv_sec = np.asarray(vis['vv_sec'], dtype=np.float64)
        freqs_hz = np.asarray(vis['freqs_hz'], dtype=np.float64)
        uu_kl_cell = (uu_sec[:, np.newaxis] * freqs_hz[np.newaxis, :]) / 1e3
        vv_kl_cell = (vv_sec[:, np.newaxis] * freqs_hz[np.newaxis, :]) / 1e3

        if uvcov_product is not None:
            uvcov_products = [uvcov_product]
        else:
            uvcov_products = list(products)

        scatter_last = None
        plotted_any = False
        for pidx, prod in enumerate(uvcov_products):
            try:
                z, f = _get_product_data(prod, vis, corrected_cache, solution)
            except Exception as exc:
                print(f'[plotVis] WARNING: skipping panel {panel} for product={prod}: {exc}')
                continue

            finite = np.isfinite(z.real) & np.isfinite(z.imag)
            good = (~f) & finite
            if shared_sample_mask is not None and shared_sample_mask.shape == z.shape:
                sm = shared_sample_mask
            else:
                sm = _sample_mask(z.shape, args.sample_frac, seed=100 + pidx)
            use = good & sm
            if not np.any(use):
                continue

            u = uu_kl_cell[use]
            v = vv_kl_cell[use]
            if uvcov_metric == 'amp':
                val = np.abs(z[use]).astype(np.float64)
            elif uvcov_metric == 'real':
                val = z.real[use].astype(np.float64)
            elif uvcov_metric == 'imag':
                val = z.imag[use].astype(np.float64)
            else:
                val = np.degrees(np.angle(z[use])).astype(np.float64)

            if args.uvcov_size_by == 'amp':
                sizes = _scale_marker_sizes(np.abs(z[use]).astype(np.float64), smin=0.08, smax=1.0)
            else:
                sizes = 0.12

            um = np.concatenate([u, -u])
            vm = np.concatenate([v, -v])
            cm = np.concatenate([val, val])
            if isinstance(sizes, np.ndarray):
                smark = np.concatenate([sizes, sizes])
            else:
                smark = sizes

            if uvcov_metric == 'phase':
                norm = mpl.colors.Normalize(vmin=-180.0, vmax=180.0)
                scatter_last = ax.scatter(um, vm, c=cm, cmap='twilight', norm=norm, s=smark, alpha=0.42, linewidths=0)
            elif uvcov_metric in ('real', 'imag'):
                vmax = float(np.nanpercentile(np.abs(val[np.isfinite(val)]), 98.0)) if np.any(np.isfinite(val)) else 1.0
                vmax = max(vmax, 1e-6)
                norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
                scatter_last = ax.scatter(um, vm, c=cm, cmap='RdBu_r', norm=norm, s=smark, alpha=0.42, linewidths=0)
            else:
                clog = np.log10(np.maximum(np.abs(cm), 1e-8))
                vmin, vmax = _robust_quantile_bounds(clog, lo=2.0, hi=98.0)
                if vmin is None or vmax is None:
                    vmin, vmax = -8.0, 0.0
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                scatter_last = ax.scatter(um, vm, c=clog, cmap='viridis', norm=norm, s=smark, alpha=0.42, linewidths=0)

            plotted_any = True

        if not plotted_any or scatter_last is None:
            ax.text(0.5, 0.5, 'No sampled cells for selected uvcov panel', ha='center', va='center')
            ax.set_xlabel('u (kλ)')
            ax.set_ylabel('v (kλ)')
            ax.grid(True, alpha=0.25)
            return

        cbar = plt.colorbar(scatter_last, ax=ax, fraction=0.046, pad=0.04)
        if uvcov_metric == 'phase':
            cbar.set_label('Phase (deg)')
        elif uvcov_metric == 'real':
            cbar.set_label('Re(V) (Jy)')
        elif uvcov_metric == 'imag':
            cbar.set_label('Im(V) (Jy)')
        else:
            cbar.set_label('log10 |V| (Jy)')

        ax.set_xlabel('u (kλ)')
        ax.set_ylabel('v (kλ)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.25)
        return

    if panel == 'uv_sampling':
        uv_good_marker_size = 0.075
        uv_flag_marker_size = 0.03
        good_cell_masks = []
        flag_cell_masks = []
        prod_labels = []

        uu_sec = np.asarray(vis['uu_sec'], dtype=np.float64)
        vv_sec = np.asarray(vis['vv_sec'], dtype=np.float64)
        freqs_hz = np.asarray(vis['freqs_hz'], dtype=np.float64)
        uu_kl_cell = (uu_sec[:, np.newaxis] * freqs_hz[np.newaxis, :]) / 1e3
        vv_kl_cell = (vv_sec[:, np.newaxis] * freqs_hz[np.newaxis, :]) / 1e3

        sampled_total = 0
        ntotal_cells = 0
        flagged_total = 0
        flagged_by_prod = {}

        for _, prod, z, f in product_payload:
            good = (~f) & np.isfinite(z.real) & np.isfinite(z.imag)
            if shared_sample_mask is not None and shared_sample_mask.shape == z.shape:
                sm = shared_sample_mask
            else:
                sm = _sample_mask(z.shape, args.sample_frac, seed=100)

            use = good & sm
            bad = f & sm
            good_cell_masks.append(use)
            flag_cell_masks.append(bad)
            prod_labels.append(str(prod))

            n_sampled = int(np.count_nonzero(sm))
            n_flag_sampled = int(np.count_nonzero(bad))
            ntotal_cells += int(z.size)
            flagged_total += int(np.count_nonzero(f))
            sampled_total += n_sampled
            flagged_by_prod[str(prod)] = (n_flag_sampled, n_sampled)

        if good_cell_masks:
            good_stack = np.stack(good_cell_masks, axis=0)
            good_counts = np.sum(good_stack, axis=0)
        else:
            good_counts = np.zeros(uu_kl_cell.shape, dtype=np.int32)

        if flag_cell_masks:
            flag_stack = np.stack(flag_cell_masks, axis=0)
            flag_counts = np.sum(flag_stack, axis=0)
        else:
            flag_counts = np.zeros(uu_kl_cell.shape, dtype=np.int32)

        sampled_good_cells = good_counts > 0
        shared_good_cells = good_counts > 1
        sampled_flag_cells = flag_counts > 0
        shared_flag_cells = flag_counts > 1

        same_good = len(good_cell_masks) <= 1 or all(np.array_equal(good_cell_masks[0], m) for m in good_cell_masks[1:])
        same_flag = len(flag_cell_masks) <= 1 or all(np.array_equal(flag_cell_masks[0], m) for m in flag_cell_masks[1:])

        legend_handles = []

        if same_good:
            ug = uu_kl_cell[sampled_good_cells]
            vg = vv_kl_cell[sampled_good_cells]
            ax.scatter(ug, vg, s=uv_good_marker_size, alpha=0.52, c='tab:blue', linewidths=0)
            ax.scatter(-ug, -vg, s=uv_good_marker_size, alpha=0.28, c='tab:blue', linewidths=0)
        else:
            for prod, cell_mask in zip(prod_labels, good_cell_masks):
                unique_cells = cell_mask & (good_counts == 1)
                if np.any(unique_cells):
                    color = product_colors.get(prod, None)
                    u = uu_kl_cell[unique_cells]
                    v = vv_kl_cell[unique_cells]
                    ax.scatter(u, v, s=uv_good_marker_size, alpha=0.62, c=color, linewidths=0)
                    ax.scatter(-u, -v, s=uv_good_marker_size, alpha=0.34, c=color, linewidths=0)
                    legend_handles.append(
                        Line2D([0], [0], marker='o', linestyle='None', markersize=4,
                               markerfacecolor=color, markeredgecolor=color, label=f'{prod} unique')
                    )

            if np.any(shared_good_cells):
                us = uu_kl_cell[shared_good_cells]
                vs = vv_kl_cell[shared_good_cells]
                ax.scatter(us, vs, s=uv_good_marker_size, alpha=0.42, c='0.35', linewidths=0)
                ax.scatter(-us, -vs, s=uv_good_marker_size, alpha=0.22, c='0.35', linewidths=0)
                legend_handles.append(
                    Line2D([0], [0], marker='o', linestyle='None', markersize=4,
                           markerfacecolor='0.35', markeredgecolor='0.35', label='shared (multi-product)')
                )

        if args.overlay_flags and args.flag_overlay_mode in ('points', 'both'):
            if same_flag:
                if np.any(sampled_flag_cells):
                    uf = uu_kl_cell[sampled_flag_cells]
                    vf = vv_kl_cell[sampled_flag_cells]
                    ax.scatter(uf, vf, s=uv_flag_marker_size, alpha=0.34, c='red', linewidths=0)
                    ax.scatter(-uf, -vf, s=uv_flag_marker_size, alpha=0.18, c='red', linewidths=0)
            else:
                for prod, cell_mask in zip(prod_labels, flag_cell_masks):
                    unique_cells = cell_mask & (flag_counts == 1)
                    if np.any(unique_cells):
                        color = product_colors.get(prod, None)
                        uf = uu_kl_cell[unique_cells]
                        vf = vv_kl_cell[unique_cells]
                        ax.scatter(uf, vf, s=uv_flag_marker_size, alpha=0.26, c=color, linewidths=0)
                        ax.scatter(-uf, -vf, s=uv_flag_marker_size, alpha=0.14, c=color, linewidths=0)
                if np.any(shared_flag_cells):
                    uf = uu_kl_cell[shared_flag_cells]
                    vf = vv_kl_cell[shared_flag_cells]
                    ax.scatter(uf, vf, s=uv_flag_marker_size, alpha=0.34, c='red', linewidths=0)
                    ax.scatter(-uf, -vf, s=uv_flag_marker_size, alpha=0.18, c='red', linewidths=0)

        sampled_points = int(np.count_nonzero(sampled_good_cells))
        total_points = int(good_counts.size)
        stats_lines = [
            f'Plotting {sampled_points:,} out of {total_points:,} sampled uv-points',
        ]
        if prod_labels:
            stats_lines.extend([
                f'% flagged_{p}={((100.0 * nf / nt) if nt > 0 else 0.0):.2f}%'
                for p, (nf, nt) in flagged_by_prod.items()
            ])
        ax.text(
            0.01, 0.99, '\n'.join(stats_lines),
            transform=ax.transAxes, fontsize=7,
            va='top', ha='left', linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
        )

        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=7, frameon=True)

        uv_for_limits = []
        if np.any(sampled_good_cells):
            uv_for_limits.append(np.abs(uu_kl_cell[sampled_good_cells]))
            uv_for_limits.append(np.abs(vv_kl_cell[sampled_good_cells]))
        if np.any(sampled_flag_cells):
            uv_for_limits.append(np.abs(uu_kl_cell[sampled_flag_cells]))
            uv_for_limits.append(np.abs(vv_kl_cell[sampled_flag_cells]))
        if not uv_for_limits:
            uv_for_limits = [np.abs(uu_kl_cell), np.abs(vv_kl_cell)]

        uv_max = max(float(np.nanmax(arr)) for arr in uv_for_limits if arr.size)
        if not np.isfinite(uv_max) or uv_max <= 0:
            uv_max = 1.0
        uv_lim = 1.02 * uv_max

        ax.set_xlabel('u (kλ)')
        ax.set_ylabel('v (kλ)')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-uv_lim, uv_lim)
        ax.set_ylim(-uv_lim, uv_lim)
        ax.grid(True, alpha=0.25)
        ax.set_title('uv sampling (sampled cells)')
        return

    # Fraction overlay is meaningful for 1D-x panels only.
    if args.overlay_flags and quantity is not None and axis in ('time', 'freq', 'uvdist') and args.flag_overlay_mode in ('fraction', 'both'):
        x_all, f_all = _flag_fraction_inputs_3d(vis, corrected_cache, axis, time_axis_row, uvd, freq_mhz)
        if x_all is not None and f_all is not None:
            _overlay_flag_fraction_curve(
                ax,
                x_values=x_all,
                flag_mask=f_all,
                nbins=args.flag_bins,
            )

    for ci, prod, z, f in product_payload:
        good = (~f) & np.isfinite(z.real) & np.isfinite(z.imag)
        if shared_sample_mask is not None and shared_sample_mask.shape == z.shape:
            sm = shared_sample_mask
        else:
            sm = _sample_mask(z.shape, args.sample_frac, seed=100)
        use = good & sm
        bad = f & sm
        color = product_colors.get(prod, None)

        if quantity is not None and axis is not None:
            if axis == 'time':
                x = np.repeat(time_axis_row, z.shape[1])
                if args.time_format in ('isot_concise', 'isot_full'):
                    xlabel = 'UTC Time'
                    ax.xaxis_date()
                    if args.time_format == 'isot_full':
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%dT%H:%M:%S'))
                    else:
                        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
                        ax.xaxis.set_major_locator(locator)
                        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                    ax.tick_params(axis='x', labelrotation=25)
                else:
                    xlabel = 'Time from start (min)'
            elif axis == 'freq':
                x = np.tile(freq_mhz, z.shape[0])
                xlabel = 'Frequency (MHz)'
            else:
                x = np.repeat(uvd, z.shape[1])
                xlabel = 'uvdist (kλ)'

            if quantity == 'amp':
                yall = np.abs(z)
                ylabel = '|V| (Jy)'
            elif quantity == 'real':
                yall = z.real
                ylabel = 'Re(V) (Jy)'
            elif quantity == 'imag':
                yall = z.imag
                ylabel = 'Im(V) (Jy)'
            else:
                yall = np.angle(z, deg=True)
                ylabel = 'Phase(V) (deg)'

            xg = x[use.ravel()]
            yg = yall[use]
            ax.scatter(xg, yg, s=1.2, alpha=0.25, c=color)
            if args.overlay_flags and args.flag_overlay_mode in ('points', 'both'):
                bad_plot = bad & np.isfinite(yall)
                bad_hidden = bad & (~np.isfinite(yall))
                if np.any(bad_plot):
                    xb = x[bad_plot.ravel()]
                    yb = yall[bad_plot]
                    ax.scatter(xb, yb, s=1.0, alpha=0.12, c='red')
                if np.any(bad_hidden):
                    xb2 = x[bad_hidden.ravel()]
                    ylo, yhi = ax.get_ylim()
                    y_rug = ylo + 0.02 * (yhi - ylo)
                    ax.scatter(xb2, np.full(xb2.shape, y_rug, dtype=float), marker='|', s=10, alpha=0.25, c='red')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        elif panel == 'ri_scatter':
            ax.scatter(z.real[use], z.imag[use], s=1.2, alpha=0.25, c=color)
            if args.overlay_flags and args.flag_overlay_mode in ('points', 'both'):
                bad_plot = bad & np.isfinite(z.real) & np.isfinite(z.imag)
                if np.any(bad_plot):
                    ax.scatter(z.real[bad_plot], z.imag[bad_plot], s=1.0, alpha=0.12, c='red')
            ax.set_xlabel('Re(V) (Jy)')
            ax.set_ylabel('Im(V) (Jy)')

        elif panel == 'vector_avg':
            w = np.asarray(vis['weight'])[:, :, 0]
            num = np.nansum(np.where(good, w * z, 0.0), axis=0)
            den = np.nansum(np.where(good, w, 0.0), axis=0)
            vec = np.full(z.shape[1], np.nan + 1j * np.nan, dtype=np.complex128)
            scalar_abs = np.full(z.shape[1], np.nan, dtype=np.float64)
            ok = den > 0
            vec[ok] = num[ok] / den[ok]
            scalar_num = np.nansum(np.where(good, w * np.abs(z), 0.0), axis=0)
            scalar_abs[ok] = scalar_num[ok] / den[ok]
            ax.plot(freq_mhz, vec.real, lw=1.2, c=color)
            ax.plot(freq_mhz, np.abs(vec), lw=1.0, ls='--', alpha=0.7, c=color)
            ax.plot(freq_mhz, scalar_abs, lw=1.0, ls=':', alpha=0.85, c=color)
            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('Vector-avg (Jy)')

        else:
            ax.text(0.5, 0.5, f'Unknown panel: {panel}', ha='center', va='center')

    ax.grid(True, alpha=0.25)
    if quantity is not None and axis == 'freq':
        _add_channel_top_axis(ax, freq_mhz, chan_numbers, nbins=args.channel_ticks)
    if panel == 'vector_avg':
        _add_channel_top_axis(ax, freq_mhz, chan_numbers, nbins=args.channel_ticks)
        style_handles = [
            Line2D([0], [0], color='black', lw=1.4, ls='-', label='solid: Re(<V>)'),
            Line2D([0], [0], color='black', lw=1.2, ls='--', label='dashed: |<V>|'),
            Line2D([0], [0], color='black', lw=1.2, ls=':', label='dotted: <|V|>'),
        ]
        ax.legend(handles=style_handles, loc='upper right', fontsize=7, frameon=True)
    title_map = {
        'vector_avg': 'baseline averaged',
        'uv_sampling': 'uv sampling (sampled)',
        'uvcov_metric': f'uv metric: {uvcov_metric}',
    }
    ax.set_title(title_map.get(panel, panel))


def main():
    parser = argparse.ArgumentParser(description='Generalized uGMRT plotVis utility')
    cfg = parser.add_argument_group('Config and Inputs')
    cfg.add_argument('--config', default=None, help='Optional config file path (no default hardcoded).')
    cfg.add_argument('--fits', default=None)
    cfg.add_argument('--index-cache', type=str, default=None)
    cfg.add_argument('--index-validation', type=str, default='fast')
    cfg.add_argument('--set', action='append', default=[], help='Override any config/global key: --set "KEY=expr"')

    sel = parser.add_argument_group('Selection')
    sel.add_argument('--source', type=str, default=None)
    sel.add_argument('--time-range', nargs=2, default=None, metavar=('START', 'END'))
    sel.add_argument('--chan-range', nargs=2, type=int, metavar=('START', 'END'), default=None,
                        help='Inclusive original FITS channel range. Default: all channels.')
    sel.add_argument('--antennas', type=str, default=None, help='Comma list of antennas to INCLUDE (name/id)')
    sel.add_argument('--baselines', type=str, default=None, help='Comma list of baseline pairs to INCLUDE, e.g. E03-W04,C01-C02')
    sel.add_argument('--exclude-baselines', type=str, default=None,
                     help='Comma list of baseline pairs to EXCLUDE, e.g. "1-25,2-25". '
                          'Antenna IDs are 1-based names or antenna index+1.')
    sel.add_argument('--exclude-antennas', type=str, default=None,
                     help='Comma list of antennas to EXCLUDE by name or 1-based index, e.g. "25,W01:25".')
    sel.add_argument('--elevation-min', type=float, default=None)
    sel.add_argument('--elevation-max', type=float, default=None)
    sel.add_argument('--uvrange-m', nargs=2, type=float, default=None, metavar=('MIN_M', 'MAX_M'))
    sel.add_argument('--uvrange-klambda', nargs=2, type=float, default=None, metavar=('MIN_KL', 'MAX_KL'))

    cal = parser.add_argument_group('Flagging and Gain Application')
    cal.add_argument('--bpcal', nargs='+', default=None, help='One or more gain calibration table paths (time-aware if multiple).')
    cal.add_argument('--flag', default=None)
    cal.add_argument('--time-interp-scheme', choices=ca.TIME_INTERP_SCHEMES, default='nearest',
                     help='Time interpolation for gain-table application only (used when --bpcal is provided).')
    cal.add_argument('--time-extrapolation', choices=['hold', 'nearest', 'none'], default='hold',
                     help='Time extrapolation for gain-table application only (used when --bpcal is provided).')

    plotgrp = parser.add_argument_group('Plot Controls')
    plotgrp.add_argument('--products', type=str, default='RR,LL,V', help='Comma list of products (corr labels and/or I,Q,U,V)')
    plotgrp.add_argument(
        '--panels',
        nargs='+',
        type=str,
        default='amp_uvdist,real_time,imag_freq,ri_scatter,vector_avg',
        help=(
            'Comma list of panels. Scalar families: amp|real|imag|phase with axis time|freq|uvdist '
            '(e.g. amp_time, real_freq, imag_uvdist, phase_time). Also supports ri_scatter, vector_avg, uv_sampling, '
            'and uv-coverage metric panels uvcov_amp|uvcov_pha|uvcov_real|uvcov_imag plus per-product variants '
            'uvcov_<prod>_<metric> (e.g. uvcov_rr_amp, uvcov_ll_real, uvcov_v_pha). '
            'Legacy aliases: uvdist->amp_uvdist, real_time, imag_freq.'
        ),
    )
    plotgrp.add_argument('--uvcov-size-by', choices=['none', 'amp'], default='none',
                         help='Optional marker-size encoding for uvcov_* panels. Default uses color only.')
    plotgrp.add_argument('--panels-per-page', type=str, default='3x2', help='Grid per page, e.g. 3x2 (default), 2x2, 6')
    plotgrp.add_argument('--multipage', choices=['auto', 'pdf', 'png', 'both'], default='auto', help='Output mode when panel count exceeds one page')
    plotgrp.add_argument('--sample-frac', type=float, default=0.03, help='Sparse sample fraction for scatter quicklook')
    plotgrp.add_argument(
        '--time-format',
        choices=['isot_concise', 'isot_full', 'minutes'],
        default='isot_concise',
        help='Time axis format for *_time panels (default: isot_concise)',
    )
    plotgrp.add_argument('--channel-ticks', type=int, default=7, help='Approximate number of top-axis channel tick labels on freq panels')
    plotgrp.add_argument('--overlay-flags', action='store_true', help='Overlay flagged cells in red')
    plotgrp.add_argument('--flag-overlay-mode', choices=['fraction', 'points', 'both'], default='fraction', help='Flag overlay style: binned fraction (recommended), raw points, or both')
    plotgrp.add_argument('--flag-bins', type=int, default=80, help='Number of bins for flag-fraction overlays on 1D-x panels')
    plotgrp.add_argument('--exclude-for-plots', action='append', default=[])

    out = parser.add_argument_group('Output')
    out.add_argument('--outdir', default='./diagnostics_out')
    out.add_argument('--outfile', default=None, help='Output filename. Default: plotvis_<source>.png')

    args = parser.parse_args()

    if args.config:
        load_config_into_globals(args.config, globals())
    apply_overrides_to_globals(args.set, globals())

    fits_path = args.fits if args.fits is not None else globals().get('CAL_FITS', None)
    if fits_path is None:
        sys.exit('ERROR: no input vis provided. Pass --fits or provide CAL_FITS via --config.')

    index_cache = args.index_cache if args.index_cache is not None else globals().get('INDEX_CACHE', None)
    source_name_cli = args.source if args.source is not None else globals().get('SOURCE', None)
    cfg_chan_range = globals().get('CHAN_RANGE', None)
    chan_range = tuple(args.chan_range) if args.chan_range else (
        tuple(cfg_chan_range) if cfg_chan_range is not None else None
    )
    timerange = tuple(args.time_range) if args.time_range else globals().get('SOLVE_TIMERANGE', None)
    elevation_min = args.elevation_min if args.elevation_min is not None else globals().get('SOLVE_ELEVATION_MIN_DEG', None)
    elevation_max = args.elevation_max if args.elevation_max is not None else globals().get('SOLVE_ELEVATION_MAX_DEG', None)
    uvrange_m = tuple(args.uvrange_m) if args.uvrange_m else globals().get('SOLVE_UVRANGE_M', None)
    uvrange_klambda = tuple(args.uvrange_klambda) if args.uvrange_klambda else globals().get('SOLVE_UVRANGE_KLAMBDA', None)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg_exclude = globals().get('EXCLUDE_FOR_PLOTS', [])
    if cfg_exclude is None:
        cfg_exclude_list = []
    elif isinstance(cfg_exclude, (list, tuple)):
        cfg_exclude_list = list(cfg_exclude)
    else:
        cfg_exclude_list = [cfg_exclude]
    exclude_for_plots = cfg_exclude_list + list(args.exclude_for_plots or [])

    index = q.get_or_build_row_index(
        fits_path,
        cache_path=index_cache,
        force_rebuild=False,
        validation_mode=args.index_validation,
        write_cache=True,
        override_dud_names=None,
    )
    ant_name_map, name_to_id = _build_ant_maps(index)

    source_name = source_name_cli
    if not source_name:
        if index.get('id_to_name'):
            source_name = sorted(index['id_to_name'].values())[0]
        else:
            raise RuntimeError('No source found in index and no --source provided.')

    if args.outfile is None or str(args.outfile).strip() == '':
        src_tag = _sanitize_for_filename(source_name)
        args.outfile = f'plotvis_{src_tag}.png'

    ant_list = []
    for token in _parse_csv_list(args.antennas):
        aid = _resolve_ant_selector(token, ant_name_map, name_to_id)
        ant_list.append(aid if aid is not None else token)

    vis = q.load_vis_for_source(
        index,
        source=source_name,
        stokes=('RR', 'LL'),
        ant_list=ant_list if ant_list else None,
        chan_range=chan_range,
        timerange=timerange,
        uvrange_m=uvrange_m,
        uvrange_klambda=uvrange_klambda,
        elevation_min_deg=elevation_min,
        elevation_max_deg=elevation_max,
        flag_all_corrs_if_any_rawvis_flagged=True,
    )

    if args.flag:
        vis, _ = q.apply_flag_tables_to_vis(vis, antenna_name_map=ant_name_map, flag_table_paths=[args.flag])

    if args.baselines:
        pairs = _parse_baselines(args.baselines, ant_name_map, name_to_id)
        vis = _slice_rows(vis, _row_filter_by_baselines(vis, pairs))

    if args.exclude_baselines:
        excl_pairs = _parse_baselines(args.exclude_baselines, ant_name_map, name_to_id)
        vis = _row_exclude_baselines(vis, excl_pairs)

    if args.exclude_antennas:
        excl_ant_ids = []
        for token in _parse_csv_list(args.exclude_antennas):
            aid = _resolve_ant_selector(token, ant_name_map, name_to_id)
            if aid is not None:
                excl_ant_ids.append(aid)
            else:
                print(f'[plotVis] WARNING: --exclude-antennas: could not resolve {token!r}, skipping')
        vis = _row_exclude_antennas(vis, excl_ant_ids)

    solutions = []
    if args.bpcal:
        for bp in args.bpcal:
            sol = _load_solution(bp, source_name, index)
            sol['_path'] = str(bp)
            solutions.append(sol)
    solution = solutions[0] if solutions else None

    if solution is not None and exclude_for_plots:
        vis = q._filter_vis_excluded_antennas(vis, solution, exclude_antennas=exclude_for_plots)

    corrected_cache = _apply_solutions_by_time(
        vis,
        solutions,
        time_interp_scheme=args.time_interp_scheme,
        time_extrapolation=args.time_extrapolation,
    ) if solutions else None

    products = _parse_csv_list(args.products)
    panels = _parse_csv_list(args.panels)
    if not panels:
        panels = ['uvdist']

    color_cycle = plt.rcParams.get('axes.prop_cycle', None)
    cycle_colors = color_cycle.by_key().get('color', []) if color_cycle is not None else []
    if not cycle_colors:
        cycle_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    product_colors = {prod: cycle_colors[i % len(cycle_colors)] for i, prod in enumerate(products)}

    nrows_page, ncols_page, max_panels_page = _parse_panels_per_page(args.panels_per_page)
    panel_pages = [panels[i:i + max_panels_page] for i in range(0, len(panels), max_panels_page)]

    freqs_hz = np.asarray(vis['freqs_hz'], dtype=np.float64)
    freq_mhz = freqs_hz / 1e6
    chan_numbers = _infer_original_chan_numbers(index, freqs_hz)
    if chan_numbers.size > 0:
        eff_c0 = int(np.nanmin(chan_numbers))
        eff_c1 = int(np.nanmax(chan_numbers))
        print(f'[plotVis] channels selected: nchan={chan_numbers.size} file_chan_range=[{eff_c0},{eff_c1}]')
    else:
        print('[plotVis] channels selected: nchan=0')
    if chan_range is not None:
        req_c0, req_c1 = int(chan_range[0]), int(chan_range[1])
        print(f'[plotVis] requested --chan-range=[{req_c0},{req_c1}]')

    jd = np.asarray(vis['jd'], dtype=np.float64)
    if args.time_format in ('isot_concise', 'isot_full'):
        unix_sec = (jd - 2440587.5) * 86400.0
        base = mdates.date2num(datetime(1970, 1, 1))
        time_axis_row = (unix_sec / 86400.0) + base
        dt0 = datetime.fromtimestamp(float(np.nanmin(unix_sec)), tz=timezone.utc)
        time_context = f'UTC date context: {dt0.strftime("%Y-%m-%d")} (from filtered subset)'
    else:
        tmin = jd.min()
        time_axis_row = (jd - tmin) * 24 * 60
        time_context = 'Time origin: filtered subset start'
    uvd = np.asarray(vis.get('uvdist_klambda', np.zeros(len(jd))), dtype=np.float64)

    requested_panels = _parse_csv_list(args.panels)
    needs_azel = any(p in ('az_time', 'el_time') for p in requested_panels)
    az_row = None
    el_row = None
    if needs_azel:
        az_row, el_row = _compute_row_azel(index=index, source_name=source_name, jd_rows=jd)
        if az_row is None or el_row is None:
            print('[plotVis] WARNING: Az/El panels requested but Az/El track could not be computed.')

    shared_sample_mask = None
    if products:
        z0, _ = _get_product_data(products[0], vis, corrected_cache, solution)
        shared_sample_mask = _sample_mask(z0.shape, args.sample_frac, seed=100)

    out_path = outdir / args.outfile
    figures = []
    for page_idx, page_panels in enumerate(panel_pages, start=1):
        fig, axs = plt.subplots(nrows_page, ncols_page, figsize=(7 * ncols_page, 4 * nrows_page), dpi=150)
        axs = np.atleast_1d(axs).ravel()

        for pi, panel in enumerate(page_panels):
            ax = axs[pi]
            _plot_panel(
                ax=ax,
                panel=panel,
                products=products,
                vis=vis,
                corrected_cache=corrected_cache,
                solution=solution,
                args=args,
                time_axis_row=time_axis_row,
                uvd=uvd,
                freq_mhz=freq_mhz,
                chan_numbers=chan_numbers,
                product_colors=product_colors,
                shared_sample_mask=shared_sample_mask,
                az_row=az_row,
                el_row=el_row,
            )

        for j in range(len(page_panels), len(axs)):
            axs[j].set_axis_off()

        summary = (
            f"source={source_name} | chan={tuple(chan_range) if chan_range is not None else 'all'} | el=[{args.elevation_min},{args.elevation_max}] "
            f"| products={products} | page {page_idx}/{len(panel_pages)} | {time_context}"
        )
        fig.suptitle(f'plotVis quicklook\n{summary}', fontsize=11)

        legend_handles = [
            Line2D([0], [0], marker='o', linestyle='None', markersize=5,
                   markerfacecolor=product_colors[p], markeredgecolor=product_colors[p], label=str(p))
            for p in products
        ]
        if args.overlay_flags:
            if args.flag_overlay_mode in ('fraction', 'both'):
                legend_handles.append(
                    Line2D([0], [0], color='red', lw=1.2, label='Flag fraction (binned)')
                )
            if args.flag_overlay_mode in ('points', 'both'):
                legend_handles.append(
                    Line2D([0], [0], marker='o', linestyle='None', markersize=5,
                           markerfacecolor='red', markeredgecolor='red', alpha=0.5, label='Flagged points')
                )
        fig.legend(handles=legend_handles, loc='upper right', fontsize=8, frameon=True)
        fig.tight_layout(rect=(0, 0.0, 0.96, 0.95))
        figures.append(fig)

    mode = args.multipage
    if mode == 'auto':
        mode = 'pdf' if out_path.suffix.lower() == '.pdf' else 'png'

    saved_paths = []
    if len(figures) == 1:
        figures[0].savefig(out_path, dpi=180, bbox_inches='tight')
        saved_paths.append(out_path)
    else:
        stem = out_path.stem
        ext = out_path.suffix.lower()

        if mode in ('pdf', 'both'):
            pdf_path = out_path if ext == '.pdf' else (outdir / f'{stem}.pdf')
            with PdfPages(pdf_path) as pdf:
                for fig in figures:
                    pdf.savefig(fig, dpi=180, bbox_inches='tight')
            saved_paths.append(pdf_path)

        if mode in ('png', 'both'):
            img_ext = ext if ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff') else '.png'
            for i, fig in enumerate(figures, start=1):
                p = outdir / f'{stem}_page{i:02d}{img_ext}'
                fig.savefig(p, dpi=180, bbox_inches='tight')
                saved_paths.append(p)

    for fig in figures:
        plt.close(fig)

    for p in saved_paths:
        print(f'[plotVis] Saved: {p}')


if __name__ == '__main__':
    main()
