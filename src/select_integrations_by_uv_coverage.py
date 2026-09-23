#!/usr/bin/env python3
"""
Select integrations with short-baseline coverage for extended-source imaging.

For moon/extended sources, any integration with at least one short-baseline
visibility (UV < critical_uv) contributes to flux measurement. This tool
identifies which integrations should be included in stacking.

Usage:
    python select_integrations_by_uv_coverage.py --fits <uvfits> --source <NAME> \
        --critical-uv 114.6 [--output integrations.txt]

Example:
    python select_integrations_by_uv_coverage.py \
        --fits ~/DATA/gmrt_40_014/work/split/moon/moon0520_primary_secondary_calibrated_flagged.uvfits \
        --source MOON0520 \
        --critical-uv 114.6
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from modules import ugmrt_query as q
except ImportError:
    print("ERROR: Could not import modules. Make sure you're running from src/ directory", file=sys.stderr)
    sys.exit(1)


def analyze_integration_coverage(
    fits_path,
    source_name,
    critical_uv,
    index_cache=None,
):
    """
    Identify integrations with short-baseline coverage.

    For extended sources, even a single short-baseline visibility helps constrain flux.
    This function returns all integrations that have at least one visibility within
    the critical UV distance.

    Parameters
    ----------
    fits_path : str or Path
        Path to UVFITS file
    source_name : str
        Source name
    critical_uv : float
        Critical UV distance for extended source imaging (in wavelengths)
    index_cache : str or Path, optional
        Path to cached index

    Returns
    -------
    dict : Results with keys:
        - good_integrations: list of integration indices with any short-baseline coverage
        - n_short_baselines_per_integration: array of short-baseline counts per integration
        - n_total_per_integration: array of total visibilities per integration
    """

    fits_path = Path(fits_path)
    if not fits_path.exists():
        raise FileNotFoundError(f"FITS not found: {fits_path}")

    cache_path = Path(index_cache) if index_cache else fits_path.with_suffix('.uvfits.row_index_cache.npz')
    index = q.get_or_build_row_index(str(fits_path), cache_path=str(cache_path), force_rebuild=False)
    vis = q.load_vis_for_source(index, source=source_name, stokes=('RR', 'LL'))

    uu_sec = vis['uu_sec']
    vv_sec = vis['vv_sec']
    freqs_hz = vis['freqs_hz']
    flagged = vis['flagged']
    jd = vis['jd']

    unique_times = np.unique(jd)
    n_integrations = len(unique_times)

    good_integrations = []
    n_short_per_int = []
    n_total_per_int = []

    print(f"Analyzing {n_integrations} integrations...")
    print(f"Critical UV: {critical_uv:.1f}λ (selecting integrations with ANY short baseline)")
    print()

    for int_idx, current_time in enumerate(unique_times):
        time_mask = (jd == current_time)

        n_within_critical = 0
        n_total = 0

        for chan_idx in range(len(freqs_hz)):
            freq = freqs_hz[chan_idx]
            u_lambda = uu_sec * freq
            v_lambda = vv_sec * freq
            uvdist = np.sqrt(u_lambda**2 + v_lambda**2)

            unflagged_mask = ~(flagged[:, chan_idx, 0] | flagged[:, chan_idx, 1])
            mask = unflagged_mask & time_mask

            n_total += np.sum(mask)
            n_within_critical += np.sum(mask & (uvdist <= critical_uv))

        n_short_per_int.append(n_within_critical)
        n_total_per_int.append(n_total)

        if n_within_critical > 0:
            good_integrations.append(int_idx)

        if (int_idx + 1) % max(1, n_integrations // 10) == 0:
            pct = 100 * n_within_critical / n_total if n_total > 0 else 0
            print(f"  {int_idx + 1}/{n_integrations}: "
                  f"{n_within_critical} short baselines ({pct:.1f}% of {n_total} total)")

    return {
        'good_integrations': good_integrations,
        'n_short_baselines_per_integration': np.array(n_short_per_int),
        'n_total_per_integration': np.array(n_total_per_int),
        'n_integrations': n_integrations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--fits', required=True, help='Input UVFITS file')
    parser.add_argument('--source', required=True, help='Source name')
    parser.add_argument(
        '--critical-uv', type=float, default=114.6,
        help='Critical UV distance for extended-source imaging (default: 114.6λ for 0.5° moon)'
    )
    parser.add_argument('--index-cache', help='Path to row index cache')
    parser.add_argument(
        '--min-samples-with-short-baselines', type=int, default=0,
        help='Give me minimum samples with short baselines per integration (default: 0 = include all). '
             'Selects integrations having at least this many visibility samples with UV < critical_uv.'
    )
    parser.add_argument('--output', '-o', help='Write integration list to file')

    args = parser.parse_args()

    try:
        results = analyze_integration_coverage(
            fits_path=args.fits,
            source_name=args.source,
            critical_uv=args.critical_uv,
            index_cache=args.index_cache,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    good_initial = results['good_integrations']
    n_short = results['n_short_baselines_per_integration']
    n_total = results['n_total_per_integration']
    n_ints = results['n_integrations']

    # Apply minimum samples with short baselines threshold
    if args.min_samples_with_short_baselines > 0:
        good = [i for i in good_initial if n_short[i] >= args.min_samples_with_short_baselines]
        print()
        print(f"Applying --min-samples-with-short-baselines {args.min_samples_with_short_baselines}...")
        print(f"  Before filter: {len(good_initial)}/{n_ints} integrations with any samples having short baselines")
        print(f"  After filter:  {len(good)}/{n_ints} integrations with >= {args.min_samples_with_short_baselines} samples having short baselines")
    else:
        good = good_initial
        print()
        print(f"Results:")
        print(f"  Integrations with short-baseline coverage: {len(good)}/{n_ints}")

    print(f"(Counts are summed across all 128 channels per integration)")

    if len(good) > 0:
        short_counts = n_short[good]
        print(f"Short-baseline visibility counts (for selected integrations):")
        print(f"  Min:    {short_counts.min()}")
        print(f"  Max:    {short_counts.max()}")
        print(f"  Mean:   {short_counts.mean():.1f}")
        print(f"  Median: {np.median(short_counts):.1f}")
        print(f"(Each count = total visibilities within {args.critical_uv:.1f}λ across all 128 channels)")
        print()

        print(f"Selected integrations (0-based, {len(good)} total):")
        print(f"  {good}")
        print()

        if args.output:
            with open(args.output, 'w') as f:
                f.write(' '.join(str(i) for i in good))
            print(f"✅ Written to {args.output}")

        # Recommend stacking params
        print()
        print(f"Recommendation for stacking:")
        print(f"  INTEGRATIONS=\"{' '.join(str(i) for i in good)}\" bash bin/run_moon_selfcal_dev.sh")
    else:
        print(f"⚠️  No integrations have short-baseline coverage!")
        print(f"    Check your critical-uv value (current: {args.critical_uv:.1f}λ)")
        print(f"    Total UV distance range: {n_short.min():.1f}λ to {n_short.max():.1f}λ")


if __name__ == '__main__':
    main()
