#!/usr/bin/env python3
"""Plot gain-table solutions (primary or secondary; single-time or multi-time).

Features
- Auto mode:
  - single table/time -> 1D per-antenna line plots
  - multi tables/time  -> per-antenna dynamic spectra
- 6x5 grid layout by default (one antenna per panel)
- Two stacked subplots per panel:
  - amp/phase (default) OR real/imag
- For multi-time line mode:
  - average across all times OR pick one time index
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.time import Time

import matplotlib
if not matplotlib.get_backend().lower().startswith('agg'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import ugmrt_query as q

log = logging.getLogger(__name__)


def _expand_table_args(items: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        p = Path(item).expanduser()
        if any(ch in str(item) for ch in ['*', '?', '[']):
            out.extend(sorted(Path().glob(str(p))))
        elif p.exists():
            out.append(p)
    uniq: list[Path] = []
    seen = set()
    for p in out:
        rp = p.resolve()
        if rp.exists() and str(rp) not in seen:
            uniq.append(rp)
            seen.add(str(rp))
    return uniq


def _sort_tables(paths: list[Path], mode: str) -> list[Path]:
    if mode == 'mtime':
        return sorted(paths, key=lambda p: p.stat().st_mtime)
    return sorted(paths, key=lambda p: p.name)


def _resolve_common_axes(solutions: list[dict], pol: str):
    if not solutions:
        raise ValueError('No solutions loaded.')

    first = solutions[0]
    pol_first = [str(x) for x in first['stokes_labels']]
    if pol not in pol_first:
        raise ValueError(f'Polarization {pol!r} not in first table labels={pol_first}')

    common_chan = set(np.asarray(first['chan_indices'], dtype=np.int32).tolist())
    for sol in solutions[1:]:
        pol_labels = [str(x) for x in sol['stokes_labels']]
        if pol not in pol_labels:
            raise ValueError(f'Polarization {pol!r} missing in table {sol.get("_path", "<unknown>")} labels={pol_labels}')
        common_chan &= set(np.asarray(sol['chan_indices'], dtype=np.int32).tolist())

    if not common_chan:
        raise ValueError('No common channels across the provided solution tables.')

    chan_ref = np.asarray(first['chan_indices'], dtype=np.int32)
    chan_common = np.asarray([int(c) for c in chan_ref.tolist() if int(c) in common_chan], dtype=np.int32)
    if chan_common.size == 0:
        raise ValueError('No overlapping channel coverage after intersection.')

    # Keep antenna panel locations stable with a canonical FITS antenna list
    # when available (same behavior as bandpass grid plotting). Fall back to a
    # deterministic table-derived layout otherwise.
    ant_layout: list[int] = []
    ant_name_map: dict[int, str] = {}

    fits_hint = first.get('source_file', None)
    if fits_hint:
        try:
            ants = q.list_antennas(fits_hint)
            ants = [a for a in ants if a.get('antenna_no') is not None]
            ants = sorted(ants, key=lambda a: int(a['antenna_no']))
            ant_layout = [int(a['antenna_no']) for a in ants]
            ant_name_map = {int(a['antenna_no']): str(a.get('name') or f'Ant{int(a["antenna_no"])}') for a in ants}
        except Exception as exc:
            log.warning('Could not build canonical antenna layout from %s (%s); falling back to table-derived order.', fits_hint, exc)

    if not ant_layout:
        ant_layout = [int(a) for a in np.asarray(first['antenna_ids'], dtype=np.int32).tolist()]
        first_names = first.get('antenna_names') or [f'Ant{int(a)}' for a in ant_layout]
        for a, n in zip(ant_layout, first_names):
            ant_name_map[int(a)] = str(n)
        for sol in solutions[1:]:
            ant_sol = [int(a) for a in np.asarray(sol['antenna_ids'], dtype=np.int32).tolist()]
            names_sol = sol.get('antenna_names') or [f'Ant{int(a)}' for a in ant_sol]
            for a, n in zip(ant_sol, names_sol):
                if int(a) not in ant_name_map:
                    ant_name_map[int(a)] = str(n)
            for a in ant_sol:
                if int(a) not in ant_layout:
                    ant_layout.append(int(a))

    # Ensure antennas explicitly excluded during solving are still represented
    # as greyed panels when not present in gains arrays.
    for sol in solutions:
        for a in np.asarray(sol.get('excluded_antenna_ids', []), dtype=np.int32).tolist():
            aa = int(a)
            if aa not in ant_layout:
                ant_layout.append(aa)
            ant_name_map.setdefault(aa, f'Ant{aa}')

    ant_common = np.asarray(ant_layout, dtype=np.int32)
    ant_names_common = [ant_name_map.get(int(a), f'Ant{int(a)}') for a in ant_common]

    return chan_common, ant_common, ant_names_common


def _build_cube(solutions: list[dict], chan_common: np.ndarray, ant_common: np.ndarray, pol: str):
    nt = len(solutions)
    nchan = len(chan_common)
    nant = len(ant_common)

    cube = np.full((nt, nchan, nant), np.nan + 1j * np.nan, dtype=np.complex128)

    chan_target = {int(c): i for i, c in enumerate(chan_common.tolist())}
    ant_target = {int(a): i for i, a in enumerate(ant_common.tolist())}

    freqs_ref = None
    for ti, sol in enumerate(solutions):
        chan_sol = np.asarray(sol['chan_indices'], dtype=np.int32)
        ant_sol = np.asarray(sol['antenna_ids'], dtype=np.int32)
        freqs_sol = np.asarray(sol['freqs_hz'], dtype=np.float64)
        stokes_sol = [str(x) for x in sol['stokes_labels']]
        gains = np.asarray(sol['gains'], dtype=np.complex128)
        valid = np.asarray(sol['valid'], dtype=bool)

        pol_idx = stokes_sol.index(pol)
        chan_map = {int(c): i for i, c in enumerate(chan_sol.tolist())}
        ant_map = {int(a): i for i, a in enumerate(ant_sol.tolist())}

        if freqs_ref is None:
            freqs_ref = np.full(nchan, np.nan, dtype=np.float64)

        for c in chan_common.tolist():
            if int(c) not in chan_map:
                continue
            si = chan_map[int(c)]
            ti_ch = chan_target[int(c)]
            freqs_ref[ti_ch] = freqs_sol[si]
            for a in ant_common.tolist():
                if int(a) not in ant_map:
                    continue
                sa = ant_map[int(a)]
                ta = ant_target[int(a)]
                if valid[si, sa, pol_idx]:
                    cube[ti, ti_ch, ta] = gains[si, sa, pol_idx]

    if freqs_ref is None:
        freqs_ref = np.asarray([], dtype=np.float64)

    return cube, freqs_ref


def _complex_time_average(cube: np.ndarray) -> np.ndarray:
    out = np.full(cube.shape[1:], np.nan + 1j * np.nan, dtype=np.complex128)
    for ch in range(cube.shape[1]):
        for ant in range(cube.shape[2]):
            vals = cube[:, ch, ant]
            vals = vals[np.isfinite(vals.real) & np.isfinite(vals.imag)]
            if vals.size:
                out[ch, ant] = np.mean(vals)
    return out


def _channel_top_ticks(freqs_hz: np.ndarray, chan_indices: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if freqs_hz.size == 0 or chan_indices.size == 0:
        return np.asarray([], dtype=np.float64), []
    n_ticks = min(6, int(freqs_hz.size))
    idx = np.unique(np.linspace(0, int(freqs_hz.size) - 1, n_ticks, dtype=int))
    pos = (np.asarray(freqs_hz, dtype=np.float64) / 1e6)[idx]
    lab = [str(int(np.asarray(chan_indices, dtype=np.int32)[i])) for i in idx.tolist()]
    return pos, lab


def _solution_time_centers_jd(solutions: list[dict]) -> np.ndarray:
    centers = []
    for sol in solutions:
        val = sol.get('solution_time_center_jd', None)
        try:
            x = float(val) if val is not None else np.nan
        except Exception:
            x = np.nan
        centers.append(x)
    return np.asarray(centers, dtype=np.float64)


def _resolve_time_axis_labels(solutions: list[dict], mode: str) -> tuple[str, list[str] | None, str | None]:
    nt = len(solutions)
    if nt <= 1:
        return 'scan', None, None

    centers_jd = _solution_time_centers_jd(solutions)
    has_jd = bool(np.all(np.isfinite(centers_jd)))

    if mode == 'auto':
        mode = 'utc' if has_jd else 'scan'

    if mode == 'scan':
        return 'scan', None, None

    if mode == 'jd':
        if not has_jd:
            log.warning('Requested --time-axis=jd but solution_time_center_jd missing; using scan index.')
            return 'scan', None, None
        return 'jd', [f'{x:.6f}' for x in centers_jd.tolist()], None

    if mode == 'utc':
        if not has_jd:
            log.warning('Requested --time-axis=utc but solution_time_center_jd missing; using scan index.')
            return 'scan', None, None
        try:
            t = Time(centers_jd, format='jd', scale='utc')
            isot_arr = np.atleast_1d(np.asarray(t.isot, dtype=str))
            full_labels = [str(x).replace('T', ' ') for x in isot_arr.tolist()]
            date_parts = [s.split(' ')[0] for s in full_labels]
            time_parts = [s.split(' ')[1] if ' ' in s else s for s in full_labels]
            date_note = None
            if len(set(date_parts)) == 1:
                date_note = f'UTC date {date_parts[0]}'
            else:
                date_note = f'UTC dates {date_parts[0]} → {date_parts[-1]}'
            return 'utc', time_parts, date_note
        except Exception:
            log.warning('Could not convert JD->UTC labels; using scan index.')
            return 'scan', None, None

    return 'scan', None, None


def _panel_dynamic(
    cube: np.ndarray,
    freqs_hz: np.ndarray,
    chan_indices: np.ndarray,
    ant_ids: np.ndarray,
    ant_names: list[str],
    *,
    quantity: str,
    rows: int,
    cols: int,
    title: str,
    save_path: Path,
    time_axis_mode: str,
    time_tick_labels: list[str] | None,
) -> None:
    nt, nchan, nant = cube.shape
    freqs_mhz = freqs_hz / 1e6
    t_axis = np.arange(nt)
    chan_tick_pos, chan_tick_lab = _channel_top_ticks(freqs_hz, chan_indices)

    is_amp_phase = (quantity == 'amp-phase')
    upper = np.abs(cube) if is_amp_phase else np.real(cube)
    lower = np.degrees(np.angle(cube)) if is_amp_phase else np.imag(cube)

    up_vals = upper[np.isfinite(upper)]
    lo_vals = lower[np.isfinite(lower)]

    if is_amp_phase:
        up_vmin, up_vmax = 0.0, float(np.nanpercentile(up_vals, 99)) if up_vals.size else 1.0
        lo_vmin, lo_vmax = -180.0, 180.0
        up_cmap, lo_cmap = 'viridis', 'twilight'
        up_label, lo_label = 'Amp', 'Phase (deg)'
    else:
        q99_up = float(np.nanpercentile(np.abs(up_vals), 99)) if up_vals.size else 1.0
        q99_lo = float(np.nanpercentile(np.abs(lo_vals), 99)) if lo_vals.size else 1.0
        up_vmin, up_vmax = -q99_up, q99_up
        lo_vmin, lo_vmax = -q99_lo, q99_lo
        up_cmap, lo_cmap = 'coolwarm', 'coolwarm'
        up_label, lo_label = 'Real', 'Imag'

    fig = plt.figure(figsize=(32, 40))
    outer = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.55, wspace=0.35, top=0.94, bottom=0.03, left=0.06, right=0.98)

    for panel_idx in range(min(nant, rows * cols)):
        ant_id = int(ant_ids[panel_idx])
        ant_name = ant_names[panel_idx]

        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[panel_idx], hspace=0.08, height_ratios=[1, 1])
        ax_u = fig.add_subplot(inner[0])
        ax_l = fig.add_subplot(inner[1], sharex=ax_u)

        u2 = upper[:, :, panel_idx]
        l2 = lower[:, :, panel_idx]

        panel_has_data = bool(np.any(np.isfinite(u2)) or np.any(np.isfinite(l2)))
        if panel_has_data:
            im_u = ax_u.imshow(u2, aspect='auto', origin='lower', interpolation='nearest',
                               extent=(float(freqs_mhz.min()), float(freqs_mhz.max()), float(t_axis.min()), float(t_axis.max())),
                               cmap=up_cmap, vmin=up_vmin, vmax=up_vmax)
            im_l = ax_l.imshow(l2, aspect='auto', origin='lower', interpolation='nearest',
                               extent=(float(freqs_mhz.min()), float(freqs_mhz.max()), float(t_axis.min()), float(t_axis.max())),
                               cmap=lo_cmap, vmin=lo_vmin, vmax=lo_vmax)
            cbu = fig.colorbar(im_u, ax=ax_u, location='right', fraction=0.046, pad=0.02)
            cbl = fig.colorbar(im_l, ax=ax_l, location='right', fraction=0.046, pad=0.02)
            cbu.ax.tick_params(labelsize=7)
            cbl.ax.tick_params(labelsize=7)
        else:
            ax_u.set_facecolor('#dddddd')
            ax_l.set_facecolor('#dddddd')
            ax_u.text(0.5, 0.5, 'FLAGGED / NO DATA', transform=ax_u.transAxes,
                      ha='center', va='center', fontsize=9, color='black')
            ax_l.text(0.5, 0.5, 'FLAGGED / NO DATA', transform=ax_l.transAxes,
                      ha='center', va='center', fontsize=9, color='black')

        ax_u.set_title(f'{ant_name} | Ant {ant_id}', fontsize=11, pad=8)
        if time_axis_mode == 'scan':
            ylab = 'Scan idx'
        elif time_axis_mode == 'jd':
            ylab = 'Time (JD)'
        else:
            ylab = 'Time (UTC)'
        ax_u.set_ylabel(ylab)
        ax_l.set_ylabel(ylab)
        ax_l.set_xlabel('Freq (MHz)')
        ax_u.tick_params(axis='x', which='both', labelbottom=False)
        top_ax = ax_u.secondary_xaxis('top')
        top_ax.set_xticks(chan_tick_pos)
        top_ax.set_xticklabels(chan_tick_lab, fontsize=8)
        top_ax.set_xlabel('Channel', fontsize=9)

        if time_tick_labels is not None and len(time_tick_labels) == nt:
            n_ticks = min(6, nt)
            tick_idx = np.unique(np.linspace(0, nt - 1, n_ticks, dtype=int))
            tick_pos = [float(t_axis[i]) for i in tick_idx.tolist()]
            tick_lab = [time_tick_labels[i] for i in tick_idx.tolist()]
            ax_u.set_yticks(tick_pos)
            ax_u.set_yticklabels(tick_lab, fontsize=8)
            ax_l.set_yticks(tick_pos)
            ax_l.set_yticklabels(tick_lab, fontsize=8)

        ax_u.text(0.01, 0.95, up_label, transform=ax_u.transAxes, va='top', ha='left', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))
        ax_l.text(0.01, 0.95, lo_label, transform=ax_l.transAxes, va='top', ha='left', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))

    fig.suptitle(title, fontsize=18, y=0.975)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _panel_line(
    arr: np.ndarray,
    freqs_hz: np.ndarray,
    chan_indices: np.ndarray,
    ant_ids: np.ndarray,
    ant_names: list[str],
    *,
    quantity: str,
    rows: int,
    cols: int,
    title: str,
    save_path: Path,
) -> None:
    nchan, nant = arr.shape
    freqs_mhz = freqs_hz / 1e6
    chan_tick_pos, chan_tick_lab = _channel_top_ticks(freqs_hz, chan_indices)

    is_amp_phase = (quantity == 'amp-phase')
    upper = np.abs(arr) if is_amp_phase else np.real(arr)
    lower = np.degrees(np.angle(arr)) if is_amp_phase else np.imag(arr)

    up_vals = upper[np.isfinite(upper)]
    lo_vals = lower[np.isfinite(lower)]

    if is_amp_phase:
        up_ylim = (0.0, float(np.nanpercentile(up_vals, 99) * 1.1) if up_vals.size else 1.0)
        lo_ylim = (-200.0, 200.0)
        up_label, lo_label = 'Amp', 'Phase (deg)'
    else:
        q99_up = float(np.nanpercentile(np.abs(up_vals), 99)) if up_vals.size else 1.0
        q99_lo = float(np.nanpercentile(np.abs(lo_vals), 99)) if lo_vals.size else 1.0
        up_ylim = (-1.1 * q99_up, 1.1 * q99_up)
        lo_ylim = (-1.1 * q99_lo, 1.1 * q99_lo)
        up_label, lo_label = 'Real', 'Imag'

    fig = plt.figure(figsize=(32, 40))
    outer = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.55, wspace=0.35, top=0.94, bottom=0.03, left=0.06, right=0.98)

    for panel_idx in range(min(nant, rows * cols)):
        ant_id = int(ant_ids[panel_idx])
        ant_name = ant_names[panel_idx]

        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[panel_idx], hspace=0.08, height_ratios=[1, 1])
        ax_u = fig.add_subplot(inner[0])
        ax_l = fig.add_subplot(inner[1], sharex=ax_u)

        y_u = upper[:, panel_idx]
        y_l = lower[:, panel_idx]

        panel_has_data = bool(np.any(np.isfinite(y_u)) or np.any(np.isfinite(y_l)))
        if panel_has_data:
            ax_u.plot(freqs_mhz, y_u, lw=1.0, marker='.', ms=2.0, color='C0')
            ax_l.plot(freqs_mhz, y_l, lw=1.0, marker='.', ms=2.0, color='C1')
        else:
            ax_u.set_facecolor('#dddddd')
            ax_l.set_facecolor('#dddddd')
            ax_u.text(0.5, 0.5, 'FLAGGED / NO DATA', transform=ax_u.transAxes,
                      ha='center', va='center', fontsize=9, color='black')
            ax_l.text(0.5, 0.5, 'FLAGGED / NO DATA', transform=ax_l.transAxes,
                      ha='center', va='center', fontsize=9, color='black')

        ax_u.set_title(f'{ant_name} | Ant {ant_id}', fontsize=11, pad=8)
        ax_u.set_ylabel(up_label)
        ax_l.set_ylabel(lo_label)
        ax_l.set_xlabel('Freq (MHz)')
        ax_u.tick_params(axis='x', which='both', labelbottom=False)
        top_ax = ax_u.secondary_xaxis('top')
        top_ax.set_xticks(chan_tick_pos)
        top_ax.set_xticklabels(chan_tick_lab, fontsize=8)
        top_ax.set_xlabel('Channel', fontsize=9)
        ax_u.set_ylim(*up_ylim)
        ax_l.set_ylim(*lo_ylim)
        ax_u.grid(True, alpha=0.25)
        ax_l.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=18, y=0.975)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot gain-table solutions (primary/secondary).')
    p.add_argument('--tables', nargs='+', required=True,
                   help='One or more solution table paths or glob patterns.')
    p.add_argument('--sort', choices=['name', 'mtime'], default='name')
    p.add_argument('--pol', default='RR', help='Polarization label to plot (RR or LL).')
    p.add_argument('--quantity', choices=['amp-phase', 'real-imag'], default='amp-phase')
    p.add_argument('--view', choices=['auto', 'dynamic', 'line'], default='auto')
    p.add_argument('--time-axis', choices=['auto', 'scan', 'jd', 'utc'], default='auto',
                   help='Dynamic-view y-axis labels: scan index (default fallback), JD, or UTC.')
    p.add_argument('--line-from', choices=['average', 'index'], default='average',
                   help='When --view=line and multiple tables are provided.')
    p.add_argument('--time-index', type=int, default=0,
                   help='Time index used when --line-from=index.')
    p.add_argument('--rows', type=int, default=6)
    p.add_argument('--cols', type=int, default=5)
    p.add_argument('--title', default=None)
    p.add_argument('--out', type=Path, default=None,
                   help='Output PNG path. Default derived from first table.')
    p.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s  %(levelname)-8s  %(message)s')

    paths = _expand_table_args(args.tables)
    if not paths:
        raise SystemExit('ERROR: No solution tables resolved from --tables inputs.')
    paths = _sort_tables(paths, args.sort)

    solutions = []
    for p in paths:
        sol = q.load_bandpass_solution(p)
        sol['_path'] = str(p)
        solutions.append(sol)

    chan_common, ant_common, ant_names = _resolve_common_axes(solutions, pol=args.pol)
    cube, freqs_hz = _build_cube(solutions, chan_common, ant_common, pol=args.pol)

    nt = cube.shape[0]
    view = args.view
    if view == 'auto':
        view = 'dynamic' if nt > 1 else 'line'
    if view == 'dynamic' and nt <= 1:
        log.warning('Dynamic mode requested but only one table/time found; falling back to line mode.')
        view = 'line'

    first_stem = Path(paths[0]).stem
    out_path = args.out if args.out is not None else Path(f'{first_stem}_solplot_{view}_{args.quantity}.png')
    out_path = out_path.expanduser().resolve()

    title = args.title
    time_axis_mode, time_tick_labels, time_date_note = _resolve_time_axis_labels(solutions, mode=args.time_axis)
    if title is None:
        src_names = sorted({str(sol.get('source_name', 'unknown')) for sol in solutions})
        src_label = src_names[0] if len(src_names) == 1 else f'{src_names[0]} +{len(src_names)-1} src'
        if view == 'dynamic':
            mode_detail = f'dynamic spectra from {nt} table(s), y={time_axis_mode}'
        else:
            if nt == 1:
                mode_detail = 'line from single table'
            elif args.line_from == 'average':
                mode_detail = f'line from complex mean of {nt} table(s)'
            else:
                idx = int(args.time_index)
                if 0 <= idx < nt and time_tick_labels is not None:
                    mode_detail = f'line from table index {idx} ({time_tick_labels[idx]}) of {nt} table(s)'
                else:
                    mode_detail = f'line from table index {idx} of {nt} table(s)'
        title = (
            f'Gain solutions | source={src_label} | {mode_detail} | '
            f'quantity={args.quantity} | pol={args.pol}'
        )
        if view == 'dynamic' and time_axis_mode == 'utc' and time_date_note:
            title = f'{title} | {time_date_note}'

    if view == 'dynamic':
        _panel_dynamic(
            cube,
            freqs_hz,
            chan_common,
            ant_common,
            ant_names,
            quantity=args.quantity,
            rows=int(args.rows),
            cols=int(args.cols),
            title=title,
            save_path=out_path,
            time_axis_mode=time_axis_mode,
            time_tick_labels=time_tick_labels,
        )
    else:
        if nt == 1:
            arr = cube[0]
        elif args.line_from == 'average':
            arr = _complex_time_average(cube)
        else:
            idx = int(args.time_index)
            if idx < 0 or idx >= nt:
                raise SystemExit(f'ERROR: --time-index {idx} out of range [0, {nt-1}]')
            arr = cube[idx]

        _panel_line(
            arr,
            freqs_hz,
            chan_common,
            ant_common,
            ant_names,
            quantity=args.quantity,
            rows=int(args.rows),
            cols=int(args.cols),
            title=title,
            save_path=out_path,
        )

    print(f'[gainPlots] saved: {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
