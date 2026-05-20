#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
