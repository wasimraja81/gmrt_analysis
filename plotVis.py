#!/usr/bin/env python3
"""
Generalized plotVis utility.

Single multi-panel figure with selectable products and selectors.
"""

import argparse
import ast
import re
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import ugmrt_query as q


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


def _get_product_data(product, vis, corrected_cache, solution):
    labels = list(vis.get('stokes_labels', []))

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
        if solution is None:
            raise ValueError(f'Product {product} requires --bpcal / calibration solution.')
        st = q.compute_stokes_vis(vis, solution, output_stokes=product, signed=False)
        z = np.asarray(st['vis_complex'][:, :, 0], dtype=np.complex128)
        f = np.asarray(st.get('flagged', np.zeros_like(st['amp'], dtype=bool))[:, :, 0], dtype=bool)
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
    top.xaxis.set_major_locator(MaxNLocator(nbins=max(3, int(nbins)), integer=True))


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


def _plot_panel(ax, panel, products, vis, corrected_cache, solution, args, time_axis_row, uvd, freq_mhz, chan_numbers, product_colors, shared_sample_mask):
    panel_alias = {
        'uvdist': 'amp_uvdist',
        'real_time': 'real_time',
        'imag_freq': 'imag_freq',
    }
    panel = panel_alias.get(panel, panel)

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

    product_payload = []
    for ci, prod in enumerate(products):
        z, f = _get_product_data(prod, vis, corrected_cache, solution)
        product_payload.append((ci, prod, z, f))

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
    }
    ax.set_title(title_map.get(panel, panel))


def main():
    parser = argparse.ArgumentParser(description='Generalized uGMRT plotVis utility')
    parser.add_argument('--fits', required=True)
    parser.add_argument('--bpcal', default=None)
    parser.add_argument('--flag', default=None)
    parser.add_argument('--outdir', default='./diagnostics_out')
    parser.add_argument('--outfile', default=None, help='Output filename. Default: plotvis_<source>.png')

    parser.add_argument('--index-cache', type=str, default=None)
    parser.add_argument('--index-validation', type=str, default='fast')

    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--time-range', nargs=2, default=None, metavar=('START', 'END'))
    parser.add_argument('--chan-range', nargs=2, type=int, metavar=('START', 'END'), default=[64, 191])
    parser.add_argument('--antennas', type=str, default=None, help='Comma list of antennas (name/id)')
    parser.add_argument('--baselines', type=str, default=None, help='Comma list like E03-W04,C01-C02')
    parser.add_argument('--elevation-min', type=float, default=None)
    parser.add_argument('--elevation-max', type=float, default=None)
    parser.add_argument('--uvrange-m', nargs=2, type=float, default=None, metavar=('MIN_M', 'MAX_M'))
    parser.add_argument('--uvrange-klambda', nargs=2, type=float, default=None, metavar=('MIN_KL', 'MAX_KL'))

    parser.add_argument('--products', type=str, default='RR,LL,V', help='Comma list of products (corr labels and/or I,Q,U,V)')
    parser.add_argument(
        '--panels',
        nargs='+',
        type=str,
        default='amp_uvdist,real_time,imag_freq,ri_scatter,vector_avg',
        help=(
            'Comma list of panels. Scalar families: amp|real|imag|phase with axis time|freq|uvdist '
            '(e.g. amp_time, real_freq, imag_uvdist, phase_time). Also supports ri_scatter, vector_avg. '
            'Legacy aliases: uvdist->amp_uvdist, real_time, imag_freq.'
        ),
    )
    parser.add_argument('--panels-per-page', type=str, default='3x2', help='Grid per page, e.g. 3x2 (default), 2x2, 6')
    parser.add_argument('--multipage', choices=['auto', 'pdf', 'png', 'both'], default='auto', help='Output mode when panel count exceeds one page')
    parser.add_argument('--sample-frac', type=float, default=0.03, help='Sparse sample fraction for scatter quicklook')
    parser.add_argument(
        '--time-format',
        choices=['isot_concise', 'isot_full', 'minutes'],
        default='isot_concise',
        help='Time axis format for *_time panels (default: isot_concise)',
    )
    parser.add_argument('--channel-ticks', type=int, default=7, help='Approximate number of top-axis channel tick labels on freq panels')
    parser.add_argument('--overlay-flags', action='store_true', help='Overlay flagged cells in red')
    parser.add_argument('--flag-overlay-mode', choices=['fraction', 'points', 'both'], default='fraction', help='Flag overlay style: binned fraction (recommended), raw points, or both')
    parser.add_argument('--flag-bins', type=int, default=80, help='Number of bins for flag-fraction overlays on 1D-x panels')

    parser.add_argument('--exclude-for-plots', action='append', default=[])
    parser.add_argument('--set', action='append', default=[], help='Backward-compatible, e.g. --set "EXCLUDE_FOR_PLOTS=[\'E03\']"')

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    exclude_for_plots = list(args.exclude_for_plots or [])
    for raw in (args.set or []):
        if '=' not in raw:
            continue
        key, value = raw.split('=', 1)
        if key.strip() == 'EXCLUDE_FOR_PLOTS':
            parsed = ast.literal_eval(value.strip())
            if isinstance(parsed, (list, tuple)):
                exclude_for_plots = list(parsed)
            elif parsed is None:
                exclude_for_plots = []
            else:
                exclude_for_plots = [parsed]

    index = q.get_or_build_row_index(
        args.fits,
        cache_path=args.index_cache,
        force_rebuild=False,
        validation_mode=args.index_validation,
        write_cache=True,
        override_dud_names=None,
    )
    ant_name_map, name_to_id = _build_ant_maps(index)

    source_name = args.source
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
        chan_range=tuple(args.chan_range),
        timerange=tuple(args.time_range) if args.time_range else None,
        uvrange_m=tuple(args.uvrange_m) if args.uvrange_m else None,
        uvrange_klambda=tuple(args.uvrange_klambda) if args.uvrange_klambda else None,
        elevation_min_deg=args.elevation_min,
        elevation_max_deg=args.elevation_max,
        flag_all_corrs_if_any_rawvis_flagged=True,
    )

    if args.flag:
        vis, _ = q.apply_flag_tables_to_vis(vis, antenna_name_map=ant_name_map, flag_table_paths=[args.flag])

    if args.baselines:
        pairs = _parse_baselines(args.baselines, ant_name_map, name_to_id)
        vis = _slice_rows(vis, _row_filter_by_baselines(vis, pairs))

    solution = _load_solution(args.bpcal, source_name, index) if args.bpcal else None

    if solution is not None and exclude_for_plots:
        vis = q._filter_vis_excluded_antennas(vis, solution, exclude_antennas=exclude_for_plots)

    corrected_cache = q.apply_bandpass_solution(vis, solution) if solution is not None else None

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

    jd = np.asarray(vis['jd'], dtype=np.float64)
    if args.time_format in ('isot_concise', 'isot_full'):
        unix_sec = (jd - 2440587.5) * 86400.0
        base = mdates.date2num(datetime(1970, 1, 1))
        time_axis_row = (unix_sec / 86400.0) + base
        dt0 = datetime.utcfromtimestamp(float(np.nanmin(unix_sec)))
        time_context = f'UTC date context: {dt0.strftime("%Y-%m-%d")} (from filtered subset)'
    else:
        tmin = jd.min()
        time_axis_row = (jd - tmin) * 24 * 60
        time_context = 'Time origin: filtered subset start'
    uvd = np.asarray(vis.get('uvdist_klambda', np.zeros(len(jd))), dtype=np.float64)

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
            )

        for j in range(len(page_panels), len(axs)):
            axs[j].set_axis_off()

        summary = (
            f"source={source_name} | chan={tuple(args.chan_range)} | el=[{args.elevation_min},{args.elevation_max}] "
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
