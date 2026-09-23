#!/usr/bin/env python3
"""List antenna pairs with baseline lengths in kλ and fringe widths."""

import argparse
import numpy as np
from pathlib import Path

try:
    from casacore.tables import table as casacore_table
except ImportError:
    try:
        from casatools import table as casa_table
        casacore_table = None
    except ImportError:
        casacore_table = None


def _read_ms_table_col(ms_path, subtable, col):
    """Read column from MS subtable using casacore (preferred) or casatools fallback."""
    try:
        import casacore.tables as ct
        with ct.table(str(ms_path / subtable), ack=False, readonly=True) as t:
            return np.asarray(t.getcol(col))
    except Exception:
        try:
            from casatools import table as casa_table
            tb = casa_table()
            tb.open(str(ms_path / subtable))
            try:
                return np.asarray(tb.getcol(col))
            finally:
                tb.close()
        except Exception as exc:
            raise RuntimeError(f"Could not read {subtable}/{col}: {exc}")


def main():
    p = argparse.ArgumentParser(
        description='List antenna pairs with baseline lengths and fringe widths'
    )
    p.add_argument('--ms', required=True, help='Input MeasurementSet path')
    p.add_argument(
        '--pairs',
        nargs='+',
        default=['1-25', '5-25', '5-9'],
        help='Antenna pairs as "N-M" (1-based names)'
    )
    p.add_argument(
        '--include-antenna',
        nargs='+',
        type=int,
        default=[],
        help='Include all pairs touching these antenna IDs (0-based)'
    )
    args = p.parse_args()

    ms_path = Path(args.ms)
    if not ms_path.exists():
        raise SystemExit(f'MS not found: {ms_path}')

    # Read antenna positions and names
    positions = _read_ms_table_col(ms_path, 'ANTENNA', 'POSITION')  # (nant, 3) in meters
    names = _read_ms_table_col(ms_path, 'ANTENNA', 'NAME')  # (nant,)
    stations = _read_ms_table_col(ms_path, 'ANTENNA', 'STATION')  # (nant,)

    # Read frequency
    chan_freq = _read_ms_table_col(ms_path, 'SPECTRAL_WINDOW', 'CHAN_FREQ')
    freq_hz = float(np.mean(chan_freq[0]))  # Use mean frequency
    c = 299792458.0
    wavelength = c / freq_hz

    # Parse antenna pairs from command line
    pairs_to_check = set()
    for pair_str in args.pairs:
        try:
            a, b = pair_str.split('-')
            a_idx = int(a) - 1  # Convert to 0-based
            b_idx = int(b) - 1
            pairs_to_check.add(tuple(sorted((a_idx, b_idx))))
        except ValueError:
            print(f'WARNING: Could not parse pair {pair_str!r}')

    # Add all pairs involving specified antennas
    for ant_id in args.include_antenna:
        for other in range(len(names)):
            if other != ant_id:
                pairs_to_check.add(tuple(sorted((ant_id, other))))

    if not pairs_to_check:
        print('No antenna pairs specified.')
        return

    # Compute baseline lengths and UV coordinates
    print(f'Frequency: {freq_hz / 1e6:.3f} MHz, Wavelength: {wavelength:.4f} m')
    print()
    print(
        f'{"Antennas":<20} '
        f'{"Baseline (m)":<18} '
        f'{"UV (kλ)":<15} '
        f'{"Fringe (arcsec)":<18}'
    )
    print('-' * 80)

    results = []
    for a_idx, b_idx in sorted(pairs_to_check):
        a_name = names[a_idx].strip() if isinstance(names[a_idx], str) else str(names[a_idx])
        b_name = names[b_idx].strip() if isinstance(names[b_idx], str) else str(names[b_idx])
        pair_name = f'{a_idx+1}-{b_idx+1} ({a_name}-{b_name})'

        # Compute baseline vector
        baseline_vec = positions[b_idx] - positions[a_idx]  # meters
        baseline_len = float(np.linalg.norm(baseline_vec))

        # UV coordinate (magnitude)
        uv_m = baseline_len
        uv_kl = uv_m / (wavelength * 1e3)

        # Fringe width (angular):
        # For a baseline of length B meters at frequency f, the coherence scale
        # is λ/B radians, which in arcsec is (λ/B) * 206265 arcsec.
        # But for a "fringe width" we typically use the half-period: λ / (2*B)
        # or simply 1/uv_kl converted to arcsec.
        fringe_width_rad = wavelength / baseline_len
        fringe_width_arcsec = fringe_width_rad * 206265.0

        results.append((pair_name, baseline_len, uv_kl, fringe_width_arcsec))

        print(
            f'{pair_name:<20} '
            f'{baseline_len:>15,.1f}   '
            f'{uv_kl:>13.2f}   '
            f'{fringe_width_arcsec:>16.2f}'
        )

    print()
    print('CASA selectors for exclusion:')
    for pair_str in args.pairs:
        try:
            a, b = pair_str.split('-')
            print(f'  antenna="!{a}&{b}"   (exclude {pair_str})')
        except ValueError:
            pass

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
