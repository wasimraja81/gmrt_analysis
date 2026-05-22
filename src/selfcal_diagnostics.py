#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import PowerNorm
import numpy as np


CYCLE_PRIORITY = {'final': 10_000}


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return float('nan')
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _stage_sort_key(stage: str) -> tuple[int, str]:
    if stage in CYCLE_PRIORITY:
        return (CYCLE_PRIORITY[stage], stage)
    match = re.fullmatch(r'sc(\d+)', stage)
    if match:
        return (int(match.group(1)), stage)
    return (50_000, stage)


def _scan_tag_from_dirname(dirname: str) -> str:
    match = re.search(r'(scan\d+)', dirname)
    return match.group(1) if match else ''


MJD0_UTC = dt.datetime(1858, 11, 17, tzinfo=dt.timezone.utc)


def _ms_time_bounds_seconds(ms_path: Path) -> tuple[float, float] | None:
    try:
        from casatools import table  # type: ignore
        tb = table()
        tb.open(str(ms_path))
        try:
            time_vals = np.asarray(tb.getcol('TIME'), dtype=np.float64)
        finally:
            tb.close()
    except Exception:
        return None

    if time_vals.size == 0:
        return None

    finite = time_vals[np.isfinite(time_vals)]
    if finite.size == 0:
        return None

    t_min = float(np.nanmin(finite))
    t_max = float(np.nanmax(finite))
    return t_min, t_max


def _format_utc_range_from_seconds(t_min: float, t_max: float) -> str:
    dt_start = MJD0_UTC + dt.timedelta(seconds=t_min)
    dt_end = MJD0_UTC + dt.timedelta(seconds=t_max)
    return f'UTC {dt_start:%Y-%m-%d %H:%M:%S} → {dt_end:%H:%M:%S}'


def _format_utc_hms_from_seconds(t_seconds: float) -> str:
    dt_value = MJD0_UTC + dt.timedelta(seconds=float(t_seconds))
    return f'{dt_value:%H:%M:%S}'


def _format_ms_utc_range(ms_path: Path) -> str:
    bounds = _ms_time_bounds_seconds(ms_path)
    if bounds is None:
        return 'UTC N/A'
    return _format_utc_range_from_seconds(bounds[0], bounds[1])


def _discover_debug_jsons(root: str) -> list[str]:
    pattern = os.path.join(root, '**', '*_clean_debug.json')
    return sorted(glob.glob(pattern, recursive=True))


def collect_rows(
    root: str,
    scan_tag: str | None = None,
    require_single_integration: bool = False,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for debug_path in _discover_debug_jsons(root):
        try:
            with open(debug_path, 'r', encoding='utf-8') as handle:
                dbg = json.load(handle)
        except Exception:
            continue

        int_dir = os.path.dirname(debug_path)
        selfcal_dir = os.path.dirname(int_dir)
        selfcal_name = os.path.basename(selfcal_dir)
        resolved_scan_tag = _scan_tag_from_dirname(selfcal_name)
        if scan_tag and resolved_scan_tag != scan_tag:
            continue

        n_integrations = int(dbg.get('n_integrations', 1) or 1)
        if require_single_integration and n_integrations != 1:
            continue

        debug_run_id = str(dbg.get('run_id', '') or '')
        if run_id and debug_run_id != run_id:
            continue

        label = str(dbg.get('label', '') or '')
        start_integration = dbg.get('start_integration', float('nan'))
        try:
            start_integration = int(start_integration)
        except (TypeError, ValueError):
            start_integration = -1

        cycles = dbg.get('cycles', [])
        if not isinstance(cycles, list):
            continue

        for cycle_entry in cycles:
            if not isinstance(cycle_entry, dict):
                continue
            cycle = str(cycle_entry.get('stage', '') or '')
            if not cycle:
                continue

            rows.append({
                'run_id': debug_run_id,
                'selfcal_dir': selfcal_dir,
                'selfcal_name': selfcal_name,
                'label': label,
                'scan_tag': resolved_scan_tag,
                'start_integration': start_integration,
                'n_integrations': n_integrations,
                'cycle': cycle,
                'cycle_index': _to_float(cycle_entry.get('cycle_index', float('nan'))),
                'requested_niter': _to_float(cycle_entry.get('requested_niter', float('nan'))),
                'iterdone': _to_float(cycle_entry.get('iterdone', float('nan'))),
                'nmajordone': _to_float(cycle_entry.get('nmajordone', float('nan'))),
                'stopcode': _to_float(cycle_entry.get('stopcode', float('nan'))),
                'stack_elapsed_s': _to_float(dbg.get('stack_elapsed_s', float('nan'))),
                'cycle_elapsed_s': _to_float(cycle_entry.get('cycle_elapsed_s', float('nan'))),
                'tclean_elapsed_s': _to_float(cycle_entry.get('tclean_elapsed_s', float('nan'))),
                'gaincal_elapsed_s': _to_float(cycle_entry.get('gaincal_elapsed_s', float('nan'))),
                'applycal_elapsed_s': _to_float(cycle_entry.get('applycal_elapsed_s', float('nan'))),
                'export_elapsed_s': _to_float(cycle_entry.get('export_elapsed_s', float('nan'))),
                'peak_jy_per_beam': _to_float(cycle_entry.get('peak_jy_per_beam', float('nan'))),
                'residual_peak_jy_per_beam': _to_float(cycle_entry.get('residual_peak_jy_per_beam', float('nan'))),
                'model_sum_jy': _to_float(cycle_entry.get('model_sum_jy', float('nan'))),
                'dr_peak_over_residual': _to_float(cycle_entry.get('dr_peak_over_residual', float('nan'))),
                'nminor_cycles_total': _to_float(cycle_entry.get('nminor_cycles_total', float('nan'))),
                'nmajor_cycles_used': _to_float(cycle_entry.get('nmajor_cycles_used', float('nan'))),
            })

    rows.sort(key=lambda r: (
        r['scan_tag'],
        r['selfcal_name'],
        r['start_integration'],
        _stage_sort_key(str(r['cycle']))[0],
        str(r['cycle']),
    ))
    return rows


def write_rows_csv(rows: list[dict[str, Any]], out_csv: str) -> None:
    fieldnames = [
        'run_id', 'selfcal_dir', 'selfcal_name', 'label', 'scan_tag', 'start_integration', 'n_integrations',
        'cycle', 'cycle_index', 'requested_niter', 'iterdone', 'nmajordone', 'stopcode', 'stack_elapsed_s',
        'cycle_elapsed_s', 'tclean_elapsed_s', 'gaincal_elapsed_s', 'applycal_elapsed_s', 'export_elapsed_s',
        'peak_jy_per_beam', 'residual_peak_jy_per_beam', 'model_sum_jy', 'dr_peak_over_residual',
        'nminor_cycles_total', 'nmajor_cycles_used',
    ]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows_csv(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            for key in (
                'start_integration', 'n_integrations', 'cycle_index', 'requested_niter', 'iterdone', 'nmajordone',
                'stopcode', 'stack_elapsed_s', 'cycle_elapsed_s', 'tclean_elapsed_s', 'gaincal_elapsed_s',
                'applycal_elapsed_s', 'export_elapsed_s', 'peak_jy_per_beam', 'residual_peak_jy_per_beam',
                'model_sum_jy', 'dr_peak_over_residual', 'nminor_cycles_total', 'nmajor_cycles_used',
            ):
                parsed[key] = _to_float(parsed.get(key))
            rows.append(parsed)
    return rows


def _safe_stats(values: list[float]) -> tuple[float, float, float, float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float('nan'), float('nan'), float('nan'), float('nan')
    return (
        min(finite),
        max(finite),
        statistics.stdev(finite) if len(finite) > 1 else 0.0,
        statistics.median(finite),
    )


def write_summary_stats(rows: list[dict[str, Any]], out_stats_csv: str, cycles_order: list[str]) -> None:
    metrics = ['peak_jy_per_beam', 'residual_peak_jy_per_beam', 'model_sum_jy', 'dr_peak_over_residual']
    stats_rows: list[dict[str, Any]] = []
    for cycle in cycles_order:
        cycle_rows = [r for r in rows if r['cycle'] == cycle]
        if not cycle_rows:
            continue
        stat_row: dict[str, Any] = {'cycle': cycle, 'n': len(cycle_rows)}
        for metric in metrics:
            vmin, vmax, vstd, vmed = _safe_stats([_to_float(r.get(metric)) for r in cycle_rows])
            stat_row[f'{metric}_min'] = vmin
            stat_row[f'{metric}_max'] = vmax
            stat_row[f'{metric}_std'] = vstd
            stat_row[f'{metric}_median'] = vmed
        stats_rows.append(stat_row)

    fieldnames = [
        'cycle', 'n',
        'peak_jy_per_beam_min', 'peak_jy_per_beam_max', 'peak_jy_per_beam_std', 'peak_jy_per_beam_median',
        'residual_peak_jy_per_beam_min', 'residual_peak_jy_per_beam_max', 'residual_peak_jy_per_beam_std', 'residual_peak_jy_per_beam_median',
        'model_sum_jy_min', 'model_sum_jy_max', 'model_sum_jy_std', 'model_sum_jy_median',
        'dr_peak_over_residual_min', 'dr_peak_over_residual_max', 'dr_peak_over_residual_std', 'dr_peak_over_residual_median',
    ]
    Path(out_stats_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_stats_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)


def _read_gcal_per_antenna(gcal_path: Path) -> tuple[dict[int, complex], list[int]]:
    from casatools import table  # type: ignore

    tb = table()
    tb.open(str(gcal_path))
    try:
        cparam = np.asarray(tb.getcol('CPARAM'))
        flag = np.asarray(tb.getcol('FLAG'))
        antenna1 = np.asarray(tb.getcol('ANTENNA1'), dtype=np.int64)
    finally:
        tb.close()

    antenna_names: list[str] = []
    tb.open(str(gcal_path / 'ANTENNA'))
    try:
        if 'NAME' in tb.colnames():
            antenna_names = [str(x) for x in np.asarray(tb.getcol('NAME')).tolist()]
    finally:
        tb.close()

    per_ant: dict[int, complex] = {}
    for row in range(int(antenna1.size)):
        ant_idx = int(antenna1[row])
        if 0 <= ant_idx < len(antenna_names):
            ant_name = antenna_names[ant_idx]
            try:
                ant_id = int(float(ant_name))
            except ValueError:
                ant_id = ant_idx + 1
        else:
            ant_id = ant_idx + 1

        vals = np.asarray(cparam[..., row], dtype=np.complex128).reshape(-1)
        bad = np.asarray(flag[..., row], dtype=bool).reshape(-1)
        good = (~bad) & np.isfinite(vals.real) & np.isfinite(vals.imag)
        if np.any(good):
            per_ant[ant_id] = complex(np.mean(vals[good]))

    ant_layout: list[int] = []
    for idx, ant_name in enumerate(antenna_names):
        try:
            ant_layout.append(int(float(ant_name)))
        except ValueError:
            ant_layout.append(idx + 1)
    ant_layout = sorted(set(ant_layout))
    return per_ant, ant_layout


def plot_selfcal_gain_panel(
    rows: list[dict[str, Any]],
    out_panel_png: str,
    title_prefix: str | None = None,
    rows_n: int = 6,
    cols_n: int = 5,
) -> bool:
    cycle_labels = sorted(
        {str(r.get('cycle', '')) for r in rows if re.fullmatch(r'sc\d+', str(r.get('cycle', '')))},
        key=_stage_sort_key,
    )
    if not cycle_labels:
        return False

    integration_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        label = str(row.get('label', '') or '')
        selfcal_dir = str(row.get('selfcal_dir', '') or '')
        if not label or not selfcal_dir:
            continue
        integration_rows[(selfcal_dir, label)] = {
            'start_integration': int(_to_float(row.get('start_integration'))),
            'selfcal_dir': selfcal_dir,
            'label': label,
        }

    integrations = sorted(integration_rows.values(), key=lambda x: int(x['start_integration']))
    if not integrations:
        return False

    x_vals = [int(item['start_integration']) for item in integrations]
    gain_cube: dict[str, dict[int, dict[int, complex]]] = {c: {} for c in cycle_labels}
    ant_layout: list[int] = []

    for item in integrations:
        label = str(item['label'])
        selfcal_dir = Path(str(item['selfcal_dir']))
        integ_n = int(item['start_integration'])
        for cycle in cycle_labels:
            gcal_path = selfcal_dir / label / f'{label}_{cycle}.gcal'
            if not gcal_path.is_dir():
                continue
            try:
                per_ant, gcal_ant_layout = _read_gcal_per_antenna(gcal_path)
            except Exception:
                continue
            if gcal_ant_layout:
                ant_layout = sorted(set(ant_layout).union(gcal_ant_layout))
            if per_ant:
                gain_cube.setdefault(cycle, {}).setdefault(integ_n, {}).update(per_ant)

    if not ant_layout:
        ant_layout = sorted({ant for cyc in gain_cube.values() for integ in cyc.values() for ant in integ.keys()})
    ant_layout = sorted(ant_layout)
    max_panels = rows_n * cols_n
    ant_layout = ant_layout[:max_panels]

    if not ant_layout:
        return False

    amp_ylim = (0.9, 1.1)
    phase_ylim = (-10.0, 10.0)

    colors = {
        cycle: f'C{idx % 10}'
        for idx, cycle in enumerate(cycle_labels)
    }
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    linestyles = ['-', '--', '-.', ':']
    cycle_styles = {
        cycle: {
            'marker': markers[idx % len(markers)],
            'linestyle': linestyles[idx % len(linestyles)],
            'xoffset': (idx - (len(cycle_labels) - 1) / 2.0) * 0.12,
        }
        for idx, cycle in enumerate(cycle_labels)
    }

    fig = plt.figure(figsize=(32, 40))
    outer = gridspec.GridSpec(
        rows_n,
        cols_n,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
        top=0.94,
        bottom=0.03,
        left=0.06,
        right=0.98,
    )

    for panel_idx in range(max_panels):
        ant_id = ant_layout[panel_idx] if panel_idx < len(ant_layout) else None
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[panel_idx], hspace=0.08, height_ratios=[1, 1])
        ax_u = fig.add_subplot(inner[0])
        ax_l = fig.add_subplot(inner[1], sharex=ax_u)

        has_any = False
        panel_legend_handles: list[Any] = []
        panel_legend_labels: list[str] = []
        if ant_id is not None:
            for cycle in cycle_labels:
                amp_series: list[float] = []
                pha_series: list[float] = []
                for integ_n in x_vals:
                    val = gain_cube.get(cycle, {}).get(integ_n, {}).get(ant_id)
                    if val is None or not (np.isfinite(val.real) and np.isfinite(val.imag)):
                        amp_series.append(float('nan'))
                        pha_series.append(float('nan'))
                    else:
                        has_any = True
                        amp_series.append(float(abs(val)))
                        pha_series.append(float(np.degrees(np.angle(val))))
                finite_count = int(np.count_nonzero(np.isfinite(np.asarray(amp_series, dtype=float))))
                if finite_count == 0:
                    continue

                style = cycle_styles[cycle]
                x_plot = [float(x) + float(style['xoffset']) for x in x_vals]
                line_u, = ax_u.plot(
                    x_plot,
                    amp_series,
                    linestyle=str(style['linestyle']),
                    marker=str(style['marker']),
                    linewidth=1.2,
                    markersize=3.0,
                    color=colors[cycle],
                    label=cycle,
                )
                ax_l.plot(
                    x_plot,
                    pha_series,
                    linestyle=str(style['linestyle']),
                    marker=str(style['marker']),
                    linewidth=1.2,
                    markersize=3.0,
                    color=colors[cycle],
                )
                panel_legend_handles.append(line_u)
                panel_legend_labels.append(cycle)

        if not has_any:
            ax_u.set_facecolor('#dddddd')
            ax_l.set_facecolor('#dddddd')
            ax_u.text(0.5, 0.5, 'FLAGGED / NO DATA', transform=ax_u.transAxes,
                      ha='center', va='center', fontsize=9, color='black')
            ax_l.text(0.5, 0.5, 'FLAGGED / NO DATA', transform=ax_l.transAxes,
                      ha='center', va='center', fontsize=9, color='black')

        panel_name = f'Ant {ant_id}' if ant_id is not None else 'N/A'
        ax_u.set_title(panel_name, fontsize=11, pad=8)
        ax_u.set_ylabel('Amp')
        ax_l.set_ylabel('Phase (deg)')
        ax_l.set_xlabel('Integration N')
        ax_u.tick_params(axis='x', which='both', labelbottom=False)
        ax_u.set_ylim(*amp_ylim)
        ax_l.set_ylim(*phase_ylim)
        ax_u.grid(True, alpha=0.25)
        ax_l.grid(True, alpha=0.25)

        if panel_idx == 0 and panel_legend_handles:
            ax_u.legend(panel_legend_handles, panel_legend_labels, loc='best', fontsize=8)

    title = 'Selfcal gain solutions vs integration (6x5 all antennas)'
    if title_prefix:
        title = f'{title_prefix} {title}'
    fig.suptitle(title, fontsize=18, y=0.975)
    Path(out_panel_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_panel_png, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return True


def _read_ms_flag_summaries(
    ms_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], dict[int, str]]:
    from casatools import table  # type: ignore

    tb = table()
    tb.open(str(ms_path))
    try:
        antenna1 = np.asarray(tb.getcol('ANTENNA1'), dtype=np.int64)
        antenna2 = np.asarray(tb.getcol('ANTENNA2'), dtype=np.int64)
        flags = np.asarray(tb.getcol('FLAG'), dtype=bool)
    finally:
        tb.close()

    flagged_per_row = np.count_nonzero(flags, axis=(0, 1)).astype(np.float64)
    total_per_row = np.full_like(flagged_per_row, fill_value=float(flags.shape[0] * flags.shape[1]), dtype=np.float64)

    flagged_per_chan = np.count_nonzero(flags, axis=(0, 2)).astype(np.float64)
    total_per_chan = np.full_like(flagged_per_chan, fill_value=float(flags.shape[0] * flags.shape[2]), dtype=np.float64)

    ant_layout: list[int] = []
    ant_name_map: dict[int, str] = {}
    ant_table_path = ms_path / 'ANTENNA'
    if ant_table_path.is_dir():
        tb.open(str(ant_table_path))
        try:
            if 'NAME' in tb.colnames():
                ant_names = [str(x) for x in np.asarray(tb.getcol('NAME')).tolist()]
                ant_stations = [str(x) for x in np.asarray(tb.getcol('STATION')).tolist()] if 'STATION' in tb.colnames() else [''] * len(ant_names)
                for idx, ant_name in enumerate(ant_names):
                    try:
                        ant_id = int(float(ant_name))
                    except ValueError:
                        ant_id = idx + 1
                    ant_layout.append(ant_id)
                    station = ant_stations[idx].strip() if idx < len(ant_stations) else ''
                    ant_name_map[ant_id] = station if station else ant_name
        finally:
            tb.close()

    if not ant_layout:
        ant_layout = sorted(set(int(x) + 1 for x in antenna1.tolist() + antenna2.tolist()))
        for ant_id in ant_layout:
            ant_name_map.setdefault(ant_id, str(ant_id))

    return antenna1, antenna2, flagged_per_row, total_per_row, flagged_per_chan, total_per_chan, sorted(set(ant_layout)), ant_name_map


def _read_ms_channel_freq_mhz(ms_path: Path) -> np.ndarray:
    try:
        from casatools import table  # type: ignore
        spw_path = ms_path / 'SPECTRAL_WINDOW'
        if not spw_path.is_dir():
            return np.asarray([], dtype=np.float64)
        tb = table()
        tb.open(str(spw_path))
        try:
            if 'CHAN_FREQ' not in tb.colnames():
                return np.asarray([], dtype=np.float64)
            chan_freq = np.asarray(tb.getcol('CHAN_FREQ'), dtype=np.float64)
        finally:
            tb.close()
    except Exception:
        return np.asarray([], dtype=np.float64)

    if chan_freq.size == 0:
        return np.asarray([], dtype=np.float64)

    if chan_freq.ndim == 1:
        freq_hz = chan_freq
    else:
        freq_hz = chan_freq[:, 0]
    return np.asarray(freq_hz, dtype=np.float64) / 1e6


def plot_flag_baseline_grid(ms_path: Path, out_png: Path, title_prefix: str | None = None) -> bool:
    (
        antenna1,
        antenna2,
        flagged_per_row,
        total_per_row,
        _,
        _,
        ant_layout,
        ant_name_map,
    ) = _read_ms_flag_summaries(ms_path)

    if not ant_layout:
        return False

    ant_to_index = {ant_id: idx for idx, ant_id in enumerate(ant_layout)}
    matrix_size = len(ant_layout)
    flagged_unique = np.zeros((matrix_size, matrix_size), dtype=np.float64)
    total_unique = np.zeros((matrix_size, matrix_size), dtype=np.float64)

    for row_idx in range(int(antenna1.size)):
        ant_i = int(antenna1[row_idx]) + 1
        ant_j = int(antenna2[row_idx]) + 1
        idx_i = ant_to_index.get(ant_i)
        idx_j = ant_to_index.get(ant_j)
        if idx_i is None or idx_j is None:
            continue

        lower_i = max(idx_i, idx_j)
        lower_j = min(idx_i, idx_j)
        flagged_value = float(flagged_per_row[row_idx])
        total_value = float(total_per_row[row_idx])
        flagged_unique[lower_i, lower_j] += flagged_value
        total_unique[lower_i, lower_j] += total_value

    with np.errstate(divide='ignore', invalid='ignore'):
        percent_matrix = 100.0 * flagged_unique / total_unique
    percent_matrix[~np.isfinite(percent_matrix)] = np.nan

    # Build symmetric display matrix from unique-baseline values.
    percent_matrix_display = np.array(percent_matrix, copy=True)
    lower_i, lower_j = np.tril_indices(matrix_size, k=-1)
    percent_matrix_display[lower_j, lower_i] = percent_matrix_display[lower_i, lower_j]

    # With origin='lower' (Cartesian), keep y<=x colored (visual lower triangle)
    # and leave y>x (visual upper triangle) blank for labels.
    blank_mask = np.tril(np.ones_like(percent_matrix_display, dtype=bool), k=-1)
    percent_matrix_display[blank_mask] = np.nan

    finite_vals = percent_matrix[np.isfinite(percent_matrix)]
    n_cells = int(finite_vals.size)
    mean_flag = float(np.nanmean(finite_vals)) if n_cells > 0 else float('nan')
    median_flag = float(np.nanmedian(finite_vals)) if n_cells > 0 else float('nan')
    max_flag = float(np.nanmax(finite_vals)) if n_cells > 0 else float('nan')
    n_gt25 = int(np.count_nonzero(finite_vals > 25.0)) if n_cells > 0 else 0
    n_gt50 = int(np.count_nonzero(finite_vals > 50.0)) if n_cells > 0 else 0
    n_gt90 = int(np.count_nonzero(finite_vals > 90.0)) if n_cells > 0 else 0

    total_3d = float(np.nansum(total_per_row))
    flagged_3d = float(np.nansum(flagged_per_row))
    overall_percent = (100.0 * flagged_3d / total_3d) if total_3d > 0 else float('nan')
    time_range_text = _format_ms_utc_range(ms_path)

    fig, axis = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap('YlGn').copy()
    cmap.set_bad('#d9d9d9')
    image = axis.imshow(
        percent_matrix_display,
        origin='lower',
        cmap=cmap,
        norm=PowerNorm(gamma=0.45, vmin=0.0, vmax=35.0),
        interpolation='nearest',
    )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label('Flagged 3D visibility fraction (%) [YlGn, PowerNorm γ=0.45, capped at 35]')

    tick_positions = np.arange(matrix_size)
    tick_labels = [f'{a} ({ant_name_map.get(a, str(a))})' for a in ant_layout]
    axis.set_xticks(tick_positions)
    axis.set_yticks(tick_positions)
    axis.set_xticklabels(tick_labels, fontsize=6.5, rotation=55, ha='right', rotation_mode='anchor', fontstyle='italic')
    axis.set_yticklabels(tick_labels, fontsize=6.5, rotation=0, va='center', fontstyle='italic')

    # Draw per-cell grid lines.
    axis.set_xticks(np.arange(-0.5, matrix_size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, matrix_size, 1), minor=True)
    axis.grid(which='minor', color='white', linestyle='-', linewidth=0.45, alpha=0.75)
    axis.tick_params(which='minor', bottom=False, left=False)

    axis.set_xlabel('Antenna')
    axis.set_ylabel('Antenna')
    axis.set_title('Baseline flagged fraction (lower triangle; upper triangle reserved for labels)')

    summary_text = (
        'Global FLAG summary\n'
        f'{time_range_text}\n'
        f'Overall 3D flagged: {overall_percent:.2f}%\n'
        f'Visible cells: {n_cells}\n'
        f'Mean / median: {mean_flag:.2f}% / {median_flag:.2f}%\n'
        f'Max cell: {max_flag:.2f}%\n'
        f'Cells >25%: {n_gt25}\n'
        f'Cells >50%: {n_gt50}\n'
        f'Cells >90%: {n_gt90}'
    )
    axis.text(
        0.03,
        0.97,
        summary_text,
        transform=axis.transAxes,
        ha='left',
        va='top',
        fontsize=8.5,
        color='#1f2937',
        bbox={
            'boxstyle': 'round,pad=0.45',
            'facecolor': '#eef2f7',
            'edgecolor': '#64748b',
            'linewidth': 0.9,
            'alpha': 0.86,
        },
    )

    title = f'{ms_path.stem} MS FLAG diagnostics'
    if title_prefix:
        title = f'{title_prefix} {title}'
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return True


def plot_flag_vs_channel(ms_path: Path, out_png: Path, title_prefix: str | None = None) -> bool:
    (
        _,
        _,
        _,
        _,
        flagged_per_chan,
        total_per_chan,
        _,
        _,
    ) = _read_ms_flag_summaries(ms_path)

    if flagged_per_chan.size == 0 or total_per_chan.size == 0:
        return False

    with np.errstate(divide='ignore', invalid='ignore'):
        fraction_percent = 100.0 * flagged_per_chan / total_per_chan
    fraction_percent[~np.isfinite(fraction_percent)] = np.nan

    channel_index = np.arange(fraction_percent.size, dtype=int)
    chan_freq_mhz = _read_ms_channel_freq_mhz(ms_path)
    time_range_text = _format_ms_utc_range(ms_path)
    overall_total = float(np.nansum(total_per_chan))
    overall_flagged = float(np.nansum(flagged_per_chan))
    overall_percent = (100.0 * overall_flagged / overall_total) if overall_total > 0 else float('nan')

    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(channel_index, fraction_percent, color='tab:purple', linewidth=1.3)
    axis.set_xlim(0, max(0, fraction_percent.size - 1))
    axis.set_ylim(0.0, 100.0)
    axis.set_xlabel('Channel index')
    axis.set_ylabel('Flagged 3D visibility fraction (%)')
    axis.set_title('Total flagged fraction vs channel')
    axis.grid(alpha=0.25, linestyle='--')
    if chan_freq_mhz.size == fraction_percent.size and fraction_percent.size > 0:
        if fraction_percent.size <= 6:
            tick_idx = np.arange(fraction_percent.size, dtype=int)
        else:
            tick_idx = np.unique(np.linspace(0, fraction_percent.size - 1, num=6, dtype=int))
        axis_top = axis.secondary_xaxis('top')
        axis_top.set_xticks(tick_idx)
        axis_top.set_xticklabels([f'{chan_freq_mhz[int(idx)]:.2f}' for idx in tick_idx], fontsize=8)
        axis_top.set_xlabel('Frequency (MHz)')
    summary_text = (
        'Global FLAG summary\n'
        f'{time_range_text}\n'
        f'Overall 3D flagged: {overall_percent:.2f}%\n'
        f'Channels: {fraction_percent.size}'
    )
    axis.text(
        0.02,
        0.97,
        summary_text,
        transform=axis.transAxes,
        ha='left',
        va='top',
        fontsize=8.5,
        color='#1f2937',
        bbox={
            'boxstyle': 'round,pad=0.45',
            'facecolor': '#eef2f7',
            'edgecolor': '#64748b',
            'linewidth': 0.9,
            'alpha': 0.86,
        },
    )

    title = f'{ms_path.stem} MS FLAG diagnostics'
    if title_prefix:
        title = f'{title_prefix} {title}'
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return True


def _iter_integration_ms_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    integration_entries: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        selfcal_dir = str(row.get('selfcal_dir', '') or '')
        label = str(row.get('label', '') or '')
        if not selfcal_dir or not label:
            continue
        integration_entries[(selfcal_dir, label)] = {
            'selfcal_dir': selfcal_dir,
            'label': label,
            'start_integration': int(_to_float(row.get('start_integration'))),
            'n_integrations': int(_to_float(row.get('n_integrations'))),
        }
    return sorted(integration_entries.values(), key=lambda item: int(item.get('start_integration', -1)))


def _detect_full_ms_for_rows(rows: list[dict[str, Any]]) -> Path | None:
    candidate_dirs = sorted({str(r.get('selfcal_dir', '') or '') for r in rows if str(r.get('selfcal_dir', '') or '')})
    for selfcal_dir in candidate_dirs:
        base = Path(selfcal_dir)
        if not base.is_dir():
            continue
        full_matches = sorted(p for p in base.glob('*_full.ms') if p.is_dir())
        if full_matches:
            return full_matches[0]
    return None


def plot_flag_baseline_grid_all_integrations(
    rows: list[dict[str, Any]],
    out_png: Path,
    title_prefix: str | None = None,
) -> bool:
    entries = _iter_integration_ms_entries(rows)
    if not entries:
        return False

    ms_summaries: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    ant_layout: list[int] = []
    ant_name_map: dict[int, str] = {}
    global_t_min: float | None = None
    global_t_max: float | None = None

    for entry in entries:
        label = str(entry['label'])
        ms_path = Path(str(entry['selfcal_dir'])) / label / f'{label}.ms'
        if not ms_path.is_dir():
            continue
        try:
            antenna1, antenna2, flagged_per_row, total_per_row, _, _, ant_ids, ant_names = _read_ms_flag_summaries(ms_path)
        except Exception:
            continue

        ms_summaries.append((antenna1, antenna2, flagged_per_row, total_per_row))
        ant_layout = sorted(set(ant_layout).union(ant_ids))
        for ant_id in ant_ids:
            ant_name_map.setdefault(ant_id, ant_names.get(ant_id, str(ant_id)))

        bounds = _ms_time_bounds_seconds(ms_path)
        if bounds is not None:
            b_min, b_max = bounds
            global_t_min = b_min if global_t_min is None else min(global_t_min, b_min)
            global_t_max = b_max if global_t_max is None else max(global_t_max, b_max)

    if not ms_summaries or not ant_layout:
        return False

    ant_to_index = {ant_id: idx for idx, ant_id in enumerate(ant_layout)}
    matrix_size = len(ant_layout)
    flagged_unique = np.zeros((matrix_size, matrix_size), dtype=np.float64)
    total_unique = np.zeros((matrix_size, matrix_size), dtype=np.float64)

    for antenna1, antenna2, flagged_per_row, total_per_row in ms_summaries:
        for row_idx in range(int(antenna1.size)):
            ant_i = int(antenna1[row_idx]) + 1
            ant_j = int(antenna2[row_idx]) + 1
            idx_i = ant_to_index.get(ant_i)
            idx_j = ant_to_index.get(ant_j)
            if idx_i is None or idx_j is None:
                continue

            lower_i = max(idx_i, idx_j)
            lower_j = min(idx_i, idx_j)
            flagged_unique[lower_i, lower_j] += float(flagged_per_row[row_idx])
            total_unique[lower_i, lower_j] += float(total_per_row[row_idx])

    with np.errstate(divide='ignore', invalid='ignore'):
        percent_matrix = 100.0 * flagged_unique / total_unique
    percent_matrix[~np.isfinite(percent_matrix)] = np.nan

    percent_matrix_display = np.array(percent_matrix, copy=True)
    lower_i, lower_j = np.tril_indices(matrix_size, k=-1)
    percent_matrix_display[lower_j, lower_i] = percent_matrix_display[lower_i, lower_j]
    blank_mask = np.tril(np.ones_like(percent_matrix_display, dtype=bool), k=-1)
    percent_matrix_display[blank_mask] = np.nan

    finite_vals = percent_matrix[np.isfinite(percent_matrix)]
    n_cells = int(finite_vals.size)
    mean_flag = float(np.nanmean(finite_vals)) if n_cells > 0 else float('nan')
    median_flag = float(np.nanmedian(finite_vals)) if n_cells > 0 else float('nan')
    max_flag = float(np.nanmax(finite_vals)) if n_cells > 0 else float('nan')
    n_gt25 = int(np.count_nonzero(finite_vals > 25.0)) if n_cells > 0 else 0
    n_gt50 = int(np.count_nonzero(finite_vals > 50.0)) if n_cells > 0 else 0
    n_gt90 = int(np.count_nonzero(finite_vals > 90.0)) if n_cells > 0 else 0

    total_3d = float(np.nansum(total_unique))
    flagged_3d = float(np.nansum(flagged_unique))
    overall_percent = (100.0 * flagged_3d / total_3d) if total_3d > 0 else float('nan')
    time_range_text = (
        _format_utc_range_from_seconds(global_t_min, global_t_max)
        if (global_t_min is not None and global_t_max is not None)
        else 'UTC N/A'
    )

    fig, axis = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap('YlGn').copy()
    cmap.set_bad('#d9d9d9')
    image = axis.imshow(
        percent_matrix_display,
        origin='lower',
        cmap=cmap,
        norm=PowerNorm(gamma=0.45, vmin=0.0, vmax=35.0),
        interpolation='nearest',
    )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label('Flagged 3D visibility fraction (%) [YlGn, PowerNorm γ=0.45, capped at 35]')

    tick_positions = np.arange(matrix_size)
    tick_labels = [f'{a} ({ant_name_map.get(a, str(a))})' for a in ant_layout]
    axis.set_xticks(tick_positions)
    axis.set_yticks(tick_positions)
    axis.set_xticklabels(tick_labels, fontsize=6.5, rotation=55, ha='right', rotation_mode='anchor', fontstyle='italic')
    axis.set_yticklabels(tick_labels, fontsize=6.5, rotation=0, va='center', fontstyle='italic')

    axis.set_xticks(np.arange(-0.5, matrix_size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, matrix_size, 1), minor=True)
    axis.grid(which='minor', color='white', linestyle='-', linewidth=0.45, alpha=0.75)
    axis.tick_params(which='minor', bottom=False, left=False)

    axis.set_xlabel('Antenna')
    axis.set_ylabel('Antenna')
    axis.set_title('Baseline flagged fraction (all selected integrations; lower triangle shown)')

    summary_text = (
        'Global FLAG summary\n'
        f'{time_range_text}\n'
        f'Integrations combined: {len(ms_summaries)}\n'
        f'Overall 3D flagged: {overall_percent:.2f}%\n'
        f'Visible cells: {n_cells}\n'
        f'Mean / median: {mean_flag:.2f}% / {median_flag:.2f}%\n'
        f'Max cell: {max_flag:.2f}%\n'
        f'Cells >25%: {n_gt25}\n'
        f'Cells >50%: {n_gt50}\n'
        f'Cells >90%: {n_gt90}'
    )
    axis.text(
        0.03,
        0.97,
        summary_text,
        transform=axis.transAxes,
        ha='left',
        va='top',
        fontsize=8.5,
        color='#1f2937',
        bbox={
            'boxstyle': 'round,pad=0.45',
            'facecolor': '#eef2f7',
            'edgecolor': '#64748b',
            'linewidth': 0.9,
            'alpha': 0.86,
        },
    )

    title = 'all-integrations MS FLAG diagnostics'
    if title_prefix:
        title = f'{title_prefix} {title}'
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return True


def plot_total_flag_vs_integration(
    rows: list[dict[str, Any]],
    out_png: str,
    title_prefix: str | None = None,
) -> bool:
    entries = _iter_integration_ms_entries(rows)
    x_vals: list[int] = []
    y_vals: list[float] = []
    global_t_min: float | None = None
    global_t_max: float | None = None

    for entry in entries:
        label = str(entry['label'])
        ms_path = Path(str(entry['selfcal_dir'])) / label / f'{label}.ms'
        if not ms_path.is_dir():
            continue
        try:
            _, _, flagged_per_row, total_per_row, _, _, _, _ = _read_ms_flag_summaries(ms_path)
        except Exception:
            continue

        total = float(np.nansum(total_per_row))
        flagged = float(np.nansum(flagged_per_row))
        if total <= 0:
            continue

        x_vals.append(int(entry.get('start_integration', -1)))
        y_vals.append(100.0 * flagged / total)
        bounds = _ms_time_bounds_seconds(ms_path)
        if bounds is not None:
            b_min, b_max = bounds
            global_t_min = b_min if global_t_min is None else min(global_t_min, b_min)
            global_t_max = b_max if global_t_max is None else max(global_t_max, b_max)

    if not x_vals:
        return False

    fig, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(x_vals, y_vals, '-o', color='tab:green', linewidth=1.3, markersize=3.0)
    axis.set_xlabel('Integration N')
    axis.set_ylabel('Total flagged 3D visibility fraction (%)')
    axis.set_ylim(0.0, 100.0)
    axis.grid(alpha=0.25, linestyle='--')
    axis.set_title('Total MS FLAG fraction vs integration')
    time_line = (
        _format_utc_range_from_seconds(global_t_min, global_t_max)
        if (global_t_min is not None and global_t_max is not None)
        else 'UTC N/A'
    )
    summary_text = (
        'Global FLAG summary\n'
        f'{time_line}\n'
        f'Integrations: {len(x_vals)}\n'
        f'Mean / median: {np.nanmean(y_vals):.2f}% / {np.nanmedian(y_vals):.2f}%'
    )
    axis.text(
        0.02,
        0.97,
        summary_text,
        transform=axis.transAxes,
        ha='left',
        va='top',
        fontsize=8.5,
        color='#1f2937',
        bbox={
            'boxstyle': 'round,pad=0.45',
            'facecolor': '#eef2f7',
            'edgecolor': '#64748b',
            'linewidth': 0.9,
            'alpha': 0.86,
        },
    )

    title = 'MS flag diagnostics'
    if title_prefix:
        title = f'{title_prefix} {title}'
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return True


def plot_flag_antenna_panel(
    rows: list[dict[str, Any]],
    out_png: str,
    title_prefix: str | None = None,
    rows_n: int = 6,
    cols_n: int = 5,
) -> bool:
    entries = _iter_integration_ms_entries(rows)
    if not entries:
        return False

    antenna_layout: list[int] = []
    integration_ids: list[int] = []
    per_ant_vs_n: dict[int, list[float]] = {}
    per_ant_chan_flagged: dict[int, np.ndarray] = {}
    per_ant_chan_total: dict[int, np.ndarray] = {}
    n_channels: int | None = None
    channel_freq_mhz: np.ndarray = np.asarray([], dtype=np.float64)
    global_t_min: float | None = None
    global_t_max: float | None = None
    integration_center_seconds: dict[int, float] = {}

    for entry in entries:
        label = str(entry['label'])
        ms_path = Path(str(entry['selfcal_dir'])) / label / f'{label}.ms'
        if not ms_path.is_dir():
            continue
        try:
            antenna1, antenna2, flagged_per_row, total_per_row, flagged_per_chan, _, ant_layout, _ = _read_ms_flag_summaries(ms_path)
        except Exception:
            continue

        if ant_layout:
            antenna_layout = sorted(set(antenna_layout).union(ant_layout))
        if n_channels is None:
            n_channels = int(flagged_per_chan.size)
        if channel_freq_mhz.size == 0:
            channel_freq_mhz = _read_ms_channel_freq_mhz(ms_path)

        bounds = _ms_time_bounds_seconds(ms_path)
        if bounds is not None:
            b_min, b_max = bounds
            global_t_min = b_min if global_t_min is None else min(global_t_min, b_min)
            global_t_max = b_max if global_t_max is None else max(global_t_max, b_max)
            n_id = int(entry.get('start_integration', -1))
            integration_center_seconds[n_id] = 0.5 * (b_min + b_max)

        if n_channels is None or n_channels <= 0:
            continue

        integration_ids.append(int(entry.get('start_integration', -1)))
        from casatools import table  # type: ignore
        tb = table()
        tb.open(str(ms_path))
        try:
            tb_flag = np.asarray(tb.getcol('FLAG'), dtype=bool)
        finally:
            tb.close()

        ant_ids_present = sorted(set(int(x) + 1 for x in np.concatenate([antenna1, antenna2]).tolist()))
        for ant_id in ant_ids_present:
            row_mask = ((antenna1 + 1) == ant_id) | ((antenna2 + 1) == ant_id)
            if not np.any(row_mask):
                continue

            flagged = float(np.nansum(flagged_per_row[row_mask]))
            total = float(np.nansum(total_per_row[row_mask]))
            percent = (100.0 * flagged / total) if total > 0 else float('nan')
            per_ant_vs_n.setdefault(ant_id, []).append(percent)

            sel = tb_flag[..., row_mask]
            flagged_chan = np.count_nonzero(sel, axis=(0, 2)).astype(np.float64)
            total_chan = np.full(flagged_chan.shape, fill_value=float(sel.shape[0] * sel.shape[2]), dtype=np.float64)

            per_ant_chan_flagged[ant_id] = per_ant_chan_flagged.get(ant_id, np.zeros_like(flagged_chan)) + flagged_chan
            per_ant_chan_total[ant_id] = per_ant_chan_total.get(ant_id, np.zeros_like(total_chan)) + total_chan

        for ant_id in antenna_layout:
            if ant_id not in ant_ids_present:
                per_ant_vs_n.setdefault(ant_id, []).append(float('nan'))

    if not integration_ids or not antenna_layout:
        return False

    max_panels = rows_n * cols_n
    antenna_layout = antenna_layout[:max_panels]
    channel_index = np.arange(int(n_channels or 0), dtype=int)

    x_min = min(integration_ids)
    x_max = max(integration_ids)
    if len(integration_ids) <= 4:
        n_tick_positions = sorted(set(integration_ids))
    else:
        idxs = np.linspace(0, len(integration_ids) - 1, num=4, dtype=int)
        n_tick_positions = [integration_ids[int(i)] for i in idxs]
        n_tick_positions = sorted(set(n_tick_positions))
    n_tick_labels = [str(int(x)) for x in n_tick_positions]
    ut_tick_labels = [
        _format_utc_hms_from_seconds(integration_center_seconds.get(int(x), float('nan')))
        if int(x) in integration_center_seconds else 'N/A'
        for x in n_tick_positions
    ]

    fig = plt.figure(figsize=(32, 40))
    outer = gridspec.GridSpec(rows_n, cols_n, figure=fig, hspace=0.55, wspace=0.35, top=0.94, bottom=0.03, left=0.06, right=0.98)

    if channel_index.size <= 6:
        ch_tick_idx = channel_index
    else:
        ch_tick_idx = np.unique(np.linspace(0, channel_index.size - 1, num=6, dtype=int))

    for panel_idx in range(max_panels):
        ant_id = antenna_layout[panel_idx] if panel_idx < len(antenna_layout) else None
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[panel_idx], hspace=0.45, height_ratios=[1, 1])
        ax_u = fig.add_subplot(inner[0])
        ax_l = fig.add_subplot(inner[1])

        has_data = False
        if ant_id is not None:
            y_n = per_ant_vs_n.get(ant_id, [float('nan')] * len(integration_ids))
            if len(y_n) < len(integration_ids):
                y_n = y_n + [float('nan')] * (len(integration_ids) - len(y_n))
            y_n = y_n[:len(integration_ids)]
            if np.any(np.isfinite(np.asarray(y_n, dtype=float))):
                has_data = True
                ax_u.plot(integration_ids, y_n, '-o', color='tab:blue', linewidth=1.2, markersize=2.8)

            fsum = per_ant_chan_flagged.get(ant_id)
            tsum = per_ant_chan_total.get(ant_id)
            if fsum is not None and tsum is not None and fsum.size > 0:
                with np.errstate(divide='ignore', invalid='ignore'):
                    y_f = 100.0 * fsum / tsum
                y_f[~np.isfinite(y_f)] = np.nan
                if np.any(np.isfinite(y_f)):
                    has_data = True
                    ax_l.plot(channel_index, y_f, '-', color='tab:purple', linewidth=1.1)

        if not has_data:
            ax_u.set_facecolor('#dddddd')
            ax_l.set_facecolor('#dddddd')
            ax_u.text(0.5, 0.5, 'FLAGGED / NO DATA', transform=ax_u.transAxes, ha='center', va='center', fontsize=9, color='black')
            ax_l.text(0.5, 0.5, 'FLAGGED / NO DATA', transform=ax_l.transAxes, ha='center', va='center', fontsize=9, color='black')

        panel_name = f'Ant {ant_id}' if ant_id is not None else 'N/A'
        ax_u.set_title(panel_name, fontsize=11, pad=8)
        ax_u.set_ylabel('Flag % vs N')
        ax_l.set_ylabel('Flag % vs Ch')
        ax_l.set_xlabel('Channel')
        ax_u.set_xlim(x_min, x_max)
        ax_u.set_xticks(n_tick_positions)
        ax_u.set_xticklabels(n_tick_labels, fontsize=6.5)
        ax_u.set_xlabel('N', fontsize=8, labelpad=8)

        ax_u_top = ax_u.secondary_xaxis('top')
        ax_u_top.set_xticks(n_tick_positions)
        ax_u_top.set_xticklabels(ut_tick_labels, fontsize=6.0)
        ax_u_top.set_xlabel('UT', fontsize=8, labelpad=3)

        ax_u.set_ylim(0.0, 100.0)
        ax_l.set_ylim(0.0, 100.0)
        if channel_freq_mhz.size == channel_index.size and channel_index.size > 0:
            ax_l_top = ax_l.secondary_xaxis('top')
            ax_l_top.set_xticks(ch_tick_idx)
            ax_l_top.set_xticklabels([f'{channel_freq_mhz[int(idx)]:.2f}' for idx in ch_tick_idx], fontsize=6.0)
            ax_l_top.set_xlabel('Frequency (MHz)', fontsize=7, labelpad=5)
        ax_u.grid(True, alpha=0.25)
        ax_l.grid(True, alpha=0.25)

    time_line = (
        _format_utc_range_from_seconds(global_t_min, global_t_max)
        if (global_t_min is not None and global_t_max is not None)
        else 'UTC N/A'
    )
    fig.text(
        0.012,
        0.988,
        (
            'Global FLAG summary\n'
            f'{time_line}\n'
            f'Integrations: {len(integration_ids)} | Antennas shown: {len(antenna_layout)}'
        ),
        ha='left',
        va='top',
        fontsize=11,
        color='#1f2937',
        bbox={
            'boxstyle': 'round,pad=0.45',
            'facecolor': '#eef2f7',
            'edgecolor': '#64748b',
            'linewidth': 1.0,
            'alpha': 0.90,
        },
    )

    title = 'Per-antenna MS FLAG diagnostics (6x5): flag% vs integration N and channel'
    if title_prefix:
        title = f'{title_prefix} {title}'
    fig.suptitle(title, fontsize=18, y=0.975)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return True


def write_per_integration_flag_plots(rows: list[dict[str, Any]], out_dir: str, title_prefix: str | None = None) -> tuple[int, int]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ordered_entries = _iter_integration_ms_entries(rows)

    generated = 0
    skipped = 0

    for entry in ordered_entries:
        label = str(entry['label'])
        n_integrations = int(entry.get('n_integrations', 1))
        ms_path = Path(str(entry['selfcal_dir'])) / label / f'{label}.ms'
        if not ms_path.is_dir():
            skipped += 1
            continue

        title = label if title_prefix is None else f'{title_prefix} {label}'
        baseline_png = out_path / f'{label}_flag_baseline_grid.png'
        channel_png = out_path / f'{label}_flag_vs_channel.png'

        wrote_baseline = plot_flag_baseline_grid(ms_path, baseline_png, title_prefix=title)
        wrote_channel = plot_flag_vs_channel(ms_path, channel_png, title_prefix=title)

        if wrote_baseline:
            generated += 1
        if wrote_channel:
            generated += 1

        if n_integrations != 1:
            # For stacked integrations, a time-axis flag plot can be added later if requested.
            pass

    return generated, skipped


def build_flag_heatmap_movie(flag_dir: Path, out_mp4: Path, fps: int = 4) -> bool:
    frames = sorted(flag_dir.glob('*_flag_baseline_grid.png'))
    if not frames:
        return False

    ffmpeg_bin = shutil.which('ffmpeg')
    if not ffmpeg_bin:
        return False

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='flag_heatmap_frames_') as temp_dir:
        temp_path = Path(temp_dir)
        for idx, frame in enumerate(frames):
            target = temp_path / f'frame_{idx:04d}.png'
            shutil.copy2(frame, target)

        cmd = [
            ffmpeg_bin,
            '-y',
            '-framerate',
            str(int(fps)),
            '-i',
            str(temp_path / 'frame_%04d.png'),
            '-c:v',
            'libx264',
            '-pix_fmt',
            'yuv420p',
            '-movflags',
            '+faststart',
            str(out_mp4),
        ]
        completed = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return completed.returncode == 0


def _filter_rows_for_micro(
    rows: list[dict[str, Any]],
    scan_tag: str | None,
    selfcal_name: str | None,
    run_id: str | None,
    require_single_integration: bool,
) -> list[dict[str, Any]]:
    filtered = rows
    if run_id:
        filtered = [r for r in filtered if str(r.get('run_id', '') or '') == run_id]
    if scan_tag:
        filtered = [r for r in filtered if r.get('scan_tag') == scan_tag]
    if selfcal_name:
        filtered = [r for r in filtered if r.get('selfcal_name') == selfcal_name]
    if require_single_integration:
        filtered = [r for r in filtered if int(_to_float(r.get('n_integrations'))) == 1]
    return filtered


def plot_micro_scan(
    rows: list[dict[str, Any]],
    out_panel_png: str,
    out_stats_csv: str | None = None,
    title_prefix: str | None = None,
) -> None:
    cycles_order = sorted({str(r['cycle']) for r in rows}, key=_stage_sort_key)
    metrics = [
        ('peak_jy_per_beam', 'Peak (Jy/beam)'),
        ('residual_peak_jy_per_beam', 'Residual Peak (Jy/beam)'),
        ('model_sum_jy', 'Total Cleaned Flux (Jy)'),
        ('dr_peak_over_residual', 'DR (Peak/Residual)'),
        ('clean_cycles', 'Clean Cycles'),
    ]

    if out_stats_csv:
        write_summary_stats(rows, out_stats_csv, cycles_order)

    fig, axes = plt.subplots(nrows=len(cycles_order), ncols=len(metrics), figsize=(18, 14), sharex='col')
    if len(cycles_order) == 1:
        axes = [axes]

    for i, cycle in enumerate(cycles_order):
        cycle_rows = [r for r in rows if r['cycle'] == cycle]
        cycle_rows.sort(key=lambda r: _to_float(r.get('start_integration')))
        x = [int(_to_float(r['start_integration'])) for r in cycle_rows]

        for j, (key, title) in enumerate(metrics):
            ax = axes[i][j]
            if key != 'clean_cycles':
                y = [_to_float(r.get(key)) for r in cycle_rows]
                ax.plot(x, y, '-o', markersize=2.8, linewidth=1.0)
                if i == 0:
                    ax.set_title(title, fontsize=10, fontweight='bold')
                if j == 0:
                    ax.set_ylabel(cycle, fontsize=10, fontweight='bold')
                ax.grid(alpha=0.25, linestyle='--')
            else:
                nminor = [_to_float(r.get('nminor_cycles_total')) for r in cycle_rows]
                nmajor = [_to_float(r.get('nmajor_cycles_used')) for r in cycle_rows]
                ax.plot(x, nminor, '-o', color='tab:blue', markersize=4.0, linewidth=1.0,
                        markerfacecolor='tab:blue', markeredgecolor='tab:blue', label='nMinorCyclesTotal')
                ax.set_ylabel(f'{cycle}\nnMinorCyclesTotal', color='tab:blue', fontsize=9, fontweight='bold')
                ax.tick_params(axis='y', colors='tab:blue')
                ax.spines['left'].set_color('tab:blue')
                ax.grid(alpha=0.25, linestyle='--')
                ax2 = ax.twinx()
                ax2.plot(x, nmajor, '--^', color='tab:red', markersize=4.0, linewidth=1.0,
                         markerfacecolor='none', markeredgecolor='tab:red', markeredgewidth=1.2,
                         label='nMajorCyclesUsed')
                ax2.set_ylabel('nMajorCyclesUsed', color='tab:red', fontsize=9)
                ax2.tick_params(axis='y', colors='tab:red')
                ax2.spines['right'].set_color('tab:red')
                if i == 0:
                    ax.set_title(title, fontsize=10, fontweight='bold')
                    lines_1, labels_1 = ax.get_legend_handles_labels()
                    lines_2, labels_2 = ax2.get_legend_handles_labels()
                    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', fontsize=8)

            if i == len(cycles_order) - 1:
                ax.set_xlabel('Integration N')

    title = 'Selfcal clean-cycle metrics vs integration (5x5)'
    if title_prefix:
        title = f'{title_prefix} {title}'
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    Path(out_panel_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_panel_png, dpi=170)
    plt.close(fig)


def cmd_collect(args: argparse.Namespace) -> int:
    rows = collect_rows(
        root=args.root,
        scan_tag=args.scan_tag,
        require_single_integration=args.require_single_integration,
        run_id=args.run_id,
    )
    write_rows_csv(rows, args.out_csv)
    print(f'wrote {len(rows)} rows to {args.out_csv}')
    return 0


def cmd_plot_micro(args: argparse.Namespace) -> int:
    if args.input_csv:
        rows = read_rows_csv(args.input_csv)
    else:
        rows = collect_rows(
            root=args.root,
            scan_tag=args.scan_tag,
            require_single_integration=args.require_single_integration,
            run_id=args.run_id,
        )

    rows = _filter_rows_for_micro(
        rows,
        scan_tag=args.scan_tag,
        selfcal_name=args.selfcal_name,
        run_id=args.run_id,
        require_single_integration=args.require_single_integration,
    )

    if not rows:
        print('no rows matched requested selection')
        return 2

    plot_micro_scan(
        rows=rows,
        out_panel_png=args.out_panel_png,
        out_stats_csv=args.out_stats_csv,
        title_prefix=args.title_prefix,
    )

    if args.out_csv:
        write_rows_csv(rows, args.out_csv)
        print(f'wrote {len(rows)} filtered rows to {args.out_csv}')

    if args.out_gain_panel_png:
        wrote_gain_panel = plot_selfcal_gain_panel(
            rows=rows,
            out_panel_png=args.out_gain_panel_png,
            title_prefix=args.title_prefix,
        )
        if wrote_gain_panel:
            print(f'wrote selfcal gain panel to {args.out_gain_panel_png}')
        else:
            print('no gain-table data found for requested selection; gain panel not generated')

    if args.out_flag_dir:
        generated_flag_plots, skipped_ms = write_per_integration_flag_plots(
            rows=rows,
            out_dir=args.out_flag_dir,
            title_prefix=args.title_prefix,
        )
        print(f'wrote {generated_flag_plots} per-integration MS flag plots under {args.out_flag_dir}')
        if skipped_ms > 0:
            print(f'skipped {skipped_ms} integrations without readable .ms path')

    if args.out_flag_heatmap_movie_mp4:
        if not args.out_flag_dir:
            print('cannot generate heatmap movie without --out-flag-dir')
        else:
            wrote_movie = build_flag_heatmap_movie(
                flag_dir=Path(args.out_flag_dir),
                out_mp4=Path(args.out_flag_heatmap_movie_mp4),
                fps=4,
            )
            if wrote_movie:
                print(f'wrote per-integration heatmap movie to {args.out_flag_heatmap_movie_mp4}')
            else:
                print('per-integration heatmap movie not generated (no frames or ffmpeg unavailable)')

    if args.out_flag_total_vs_n_png:
        wrote_total = plot_total_flag_vs_integration(
            rows=rows,
            out_png=args.out_flag_total_vs_n_png,
            title_prefix=args.title_prefix,
        )
        if wrote_total:
            print(f'wrote total MS flag% vs integration plot to {args.out_flag_total_vs_n_png}')
        else:
            print('no MS data found for requested selection; total flag% vs integration plot not generated')

    if args.out_flag_antenna_panel_png:
        wrote_ant_panel = plot_flag_antenna_panel(
            rows=rows,
            out_png=args.out_flag_antenna_panel_png,
            title_prefix=args.title_prefix,
        )
        if wrote_ant_panel:
            print(f'wrote per-antenna MS flag panel to {args.out_flag_antenna_panel_png}')
        else:
            print('no MS data found for requested selection; per-antenna MS flag panel not generated')

    if args.out_flag_heatmap_all_png:
        wrote_all_heatmap = plot_flag_baseline_grid_all_integrations(
            rows=rows,
            out_png=Path(args.out_flag_heatmap_all_png),
            title_prefix=args.title_prefix,
        )
        if wrote_all_heatmap:
            print(f'wrote all-integrations baseline heatmap to {args.out_flag_heatmap_all_png}')
        else:
            print('no scan-filtered integration MS data found; all-integrations baseline heatmap not generated')

    if args.out_flag_heatmap_full_png:
        full_ms = _detect_full_ms_for_rows(rows)
        if full_ms is None:
            print('no *_full.ms found for requested selection; full-MS baseline heatmap not generated')
        else:
            wrote_full_heatmap = plot_flag_baseline_grid(
                ms_path=full_ms,
                out_png=Path(args.out_flag_heatmap_full_png),
                title_prefix=args.title_prefix,
            )
            if wrote_full_heatmap:
                print(f'wrote full-MS baseline heatmap to {args.out_flag_heatmap_full_png}')
            else:
                print('full-MS baseline heatmap generation failed')

    print(f'wrote panel plot to {args.out_panel_png}')
    if args.out_stats_csv:
        print(f'wrote summary stats to {args.out_stats_csv}')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generic selfcal diagnostics collector/plotter',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    p_collect = sub.add_parser('collect', help='Collect rows from *_clean_debug.json into CSV')
    p_collect.add_argument('--root', required=True, help='Root directory containing selfcal products')
    p_collect.add_argument('--scan-tag', default=None, help='Filter scan tag, e.g. scan05')
    p_collect.add_argument('--run-id', default=None, help='Filter exact run_id recorded in clean_debug.json')
    p_collect.add_argument('--require-single-integration', action='store_true', help='Keep only n_integrations == 1 rows')
    p_collect.add_argument('--out-csv', required=True, help='Output CSV path')
    p_collect.set_defaults(func=cmd_collect)

    p_plot = sub.add_parser('plot-micro', help='Generate per-integration 5x5 micro-scan diagnostics panel')
    p_plot.add_argument('--input-csv', default=None, help='Input CSV from collect; if omitted rows are collected from --root')
    p_plot.add_argument('--root', default=None, help='Root directory for JSON discovery when --input-csv is omitted')
    p_plot.add_argument('--scan-tag', default=None, help='Filter scan tag, e.g. scan05')
    p_plot.add_argument('--run-id', default=None, help='Filter exact run_id recorded in clean_debug.json')
    p_plot.add_argument('--selfcal-name', default=None, help='Filter selfcal directory basename, e.g. 3c468.1_scan05_stk1')
    p_plot.add_argument('--require-single-integration', action='store_true', help='Keep only n_integrations == 1 rows')
    p_plot.add_argument('--out-panel-png', required=True, help='Output 5x5 panel PNG')
    p_plot.add_argument('--out-gain-panel-png', default=None, help='Optional output 6x5 selfcal gain-panel PNG')
    p_plot.add_argument('--out-flag-dir', default=None, help='Optional output directory for per-integration MS FLAG diagnostics')
    p_plot.add_argument('--out-flag-heatmap-movie-mp4', default=None, help='Optional MP4 movie from per-integration baseline heatmaps')
    p_plot.add_argument('--out-flag-total-vs-n-png', default=None, help='Optional output PNG for total MS flag% vs integration N')
    p_plot.add_argument('--out-flag-antenna-panel-png', default=None, help='Optional output PNG for 6x5 per-antenna MS flag panel')
    p_plot.add_argument('--out-flag-heatmap-all-png', default=None, help='Optional output PNG for scan-filtered all-integrations baseline heatmap')
    p_plot.add_argument('--out-flag-heatmap-full-png', default=None, help='Optional output PNG for full observation baseline heatmap from *_full.ms')
    p_plot.add_argument('--out-stats-csv', default=None, help='Optional summary stats CSV')
    p_plot.add_argument('--out-csv', default=None, help='Optional filtered rows CSV')
    p_plot.add_argument('--title-prefix', default=None, help='Optional title prefix')
    p_plot.set_defaults(func=cmd_plot_micro)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'plot-micro' and not args.input_csv and not args.root:
        parser.error('plot-micro requires either --input-csv or --root')

    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())
