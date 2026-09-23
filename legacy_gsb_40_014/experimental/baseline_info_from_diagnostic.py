#!/usr/bin/env python3
"""
Display baseline info from stripe diagnostic output.
Uses standard GMRT antenna coordinates (publicly available from GMRT config).
"""

import json
from pathlib import Path
import numpy as np

# Standard GMRT antenna positions (ITRF, meters, from GMRT array config)
# Reference: antenna.cfg in CASA/GMRT observations
GMRT_ANTENNAS = {
    'C00:01': (1656879.73490, 5953864.47660, 2282113.48850),  # ant 1 (C00, CENTRAL)
    'C00:02': (1656879.56100, 5953864.62750, 2282113.63930),  # ant 2
    'C00:03': (1656879.39710, 5953864.77840, 2282113.79010),  # ant 3
    'C04:04': (1656934.09690, 5953932.92980, 2282010.71870),  # ant 4
    'C04:05': (1656933.93970, 5953933.07040, 2282010.86730),  # ant 5
    'C09:06': (1656980.01730, 5953982.59580, 2281933.52580),  # ant 6
    'C09:07': (1656980.19310, 5953982.45050, 2281933.37380),  # ant 7
    'C09:08': (1656980.36890, 5953982.30520, 2281933.22180),  # ant 8
    'C09:09': (1656980.54470, 5953982.15990, 2281933.07080),  # ant 9
    'C12:10': (1656999.40430, 5954003.85840, 2281907.71800),  # ant 10
    'C12:11': (1656999.58140, 5954003.71300, 2281907.56560),  # ant 11
    'C12:12': (1656999.75850, 5954003.56760, 2281907.41320),  # ant 12
    'E02:13': (1657029.87400, 5953925.26320, 2282068.39150),  # ant 13
    'E02:14': (1657030.06250, 5953925.10920, 2282068.24720),  # ant 14
    'E02:15': (1657030.25100, 5953924.95530, 2282068.10290),  # ant 15
    'E06:16': (1657084.96380, 5953874.67060, 2282167.75170),  # ant 16
    'E06:17': (1657085.15600, 5953874.51460, 2282167.60620),  # ant 17
    'E06:18': (1657085.34810, 5953874.35850, 2282167.46070),  # ant 18
    'W01:19': (1655966.99890, 5954256.96540, 2281750.93790),  # ant 19
    'W01:20': (1655966.81640, 5954257.11980, 2281750.79380),  # ant 20
    'W01:21': (1655966.63390, 5954257.27420, 2281750.64970),  # ant 21
    'W04:22': (1655849.64110, 5954399.88730, 2281504.95950),  # ant 22
    'W04:23': (1655849.45460, 5954400.04310, 2281504.81560),  # ant 23
    'W04:24': (1655849.26800, 5954400.19890, 2281504.67170),  # ant 24
    'W01:25': (1655777.96050, 5954490.77890, 2281377.58970),  # ant 25
    'W01:26': (1655777.77270, 5954490.93500, 2281377.44610),  # ant 26
}


def main():
    import argparse
    p = argparse.ArgumentParser(description='Display baseline info from diagnostic JSON')
    p.add_argument('--report', required=True, help='JSON report from stripe_baseline_diagnostic.py')
    p.add_argument('--freq-mhz', type=float, default=314.398, help='Observation frequency in MHz')
    args = p.parse_args()

    report_path = Path(args.report).expanduser()
    if not report_path.exists():
        raise SystemExit(f'Report not found: {report_path}')

    with open(report_path) as f:
        report = json.load(f)

    freq_hz = args.freq_mhz * 1e6
    c = 299792458.0
    wavelength = c / freq_hz

    print(f'\n📊 Baseline Analysis from Stripe Diagnostic')
    print(f'   Frequency: {args.freq_mhz:.3f} MHz, λ = {wavelength:.4f} m')
    print()

    # Top suspicious baselines
    baselines = report.get('top_suspicious_baselines', [])
    if not baselines:
        print('No baselines in report.')
        return

    print(f'{"Rank":<6} {"Antenna Pair":<25} {"Baseline (m)":<18} {"UV (kλ)":<15} {"Fringe (″)":<14} {"Votes":<8}')
    print('-' * 100)

    for rank, bl_info in enumerate(baselines, 1):
        name_pair = bl_info.get('name_pair', '?')
        station_pair = bl_info.get('station_pair', '?')
        votes = bl_info.get('votes', 0)
        ant_pair = bl_info.get('ant_pair', [])

        # Compute baseline from GMRT config
        if len(ant_pair) == 2:
            # Try to match antenna by position in sorted list
            a_idx, b_idx = ant_pair
            # Map back to standard names (1-indexed in output, 0-indexed in array)
            a_name = list(GMRT_ANTENNAS.keys())[a_idx] if a_idx < len(GMRT_ANTENNAS) else None
            b_name = list(GMRT_ANTENNAS.keys())[b_idx] if b_idx < len(GMRT_ANTENNAS) else None

            if a_name and b_name and a_name in GMRT_ANTENNAS and b_name in GMRT_ANTENNAS:
                pos_a = np.array(GMRT_ANTENNAS[a_name])
                pos_b = np.array(GMRT_ANTENNAS[b_name])
                baseline_vec = pos_b - pos_a
                baseline_len = float(np.linalg.norm(baseline_vec))
                uv_kl = baseline_len / (wavelength * 1e3)
                fringe_arcsec = wavelength / baseline_len * 206265.0
            else:
                baseline_len = None
                uv_kl = None
                fringe_arcsec = None
        else:
            baseline_len = None
            uv_kl = None
            fringe_arcsec = None

        # Format output
        pair_label = f"{name_pair}\n({station_pair})"
        if baseline_len is not None:
            print(
                f'{rank:<6} {pair_label:<25} '
                f'{baseline_len:>15,.1f}   '
                f'{uv_kl:>13.2f}   '
                f'{fringe_arcsec:>12.2f}   '
                f'{votes:>6}'
            )
        else:
            print(f'{rank:<6} {pair_label:<25} (could not compute)')

    print()
    print('💡 Use these to exclude baselines in plotVis:')
    for bl_info in baselines[:5]:  # Top 5
        name_pair = bl_info.get('name_pair', '').replace(' ', '').split('-')
        if len(name_pair) == 2:
            print(f'   --baselines "{name_pair[0]}-{name_pair[1]}"  (to INCLUDE only this pair)')
            print(f'   or antenna="!{name_pair[0]}&{name_pair[1]}"  (to EXCLUDE in CASA)')
        print()


if __name__ == '__main__':
    main()
