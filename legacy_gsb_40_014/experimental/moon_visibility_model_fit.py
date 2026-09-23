#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import ugmrt_query as q


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run Moon visibility-domain model fit (disk/ring/annulus/composite).')
    p.add_argument('--fits', required=True, help='Input split Moon UVFITS file')
    p.add_argument('--source', default=None, help='Source name inside UVFITS (default: first indexed source)')
    p.add_argument('--fit', choices=['disk', 'gaussian', 'ring', 'annulus', 'composite'], default='composite')
    p.add_argument('--stokes', nargs='+', default=['RR'], help='Stokes/correlations to load (default: RR)')
    p.add_argument('--chan-range', nargs=2, type=int, default=None, metavar=('START', 'END'))
    p.add_argument('--uvrange-klambda', nargs=2, type=float, default=None, metavar=('MIN_KL', 'MAX_KL'))
    p.add_argument('--timerange', nargs=2, default=None, metavar=('START', 'END'))
    p.add_argument('--elevation-min-deg', type=float, default=None)
    p.add_argument('--elevation-max-deg', type=float, default=None)
    p.add_argument('--out', default='./work/moon_trajectory/moon_visibility_fit.png', help='Output PNG path')
    p.add_argument('--index-validation', default='fast', choices=['none', 'fast', 'strict'])
    return p.parse_args()


def main() -> int:
    args = parse_args()

    fits_path = Path(args.fits).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    index = q.get_or_build_row_index(
        fits_path,
        cache_path=None,
        force_rebuild=False,
        validation_mode=args.index_validation,
        write_cache=False,
        override_dud_names=None,
    )

    source_name = args.source
    if not source_name:
        id_to_name = index.get('id_to_name', {})
        if not id_to_name:
            raise SystemExit('No source found in index; please pass --source explicitly.')
        source_name = sorted(id_to_name.values())[0]

    vis = q.load_vis_for_source(
        index,
        source=source_name,
        stokes=tuple(args.stokes),
        chan_range=tuple(args.chan_range) if args.chan_range else None,
        timerange=tuple(args.timerange) if args.timerange else None,
        uvrange_klambda=tuple(args.uvrange_klambda) if args.uvrange_klambda else None,
        elevation_min_deg=args.elevation_min_deg,
        elevation_max_deg=args.elevation_max_deg,
        flag_all_corrs_if_any_rawvis_flagged=True,
    )

    print(f'[moon-fit] file    : {fits_path}')
    print(f'[moon-fit] source  : {source_name}')
    print(f'[moon-fit] stokes  : {list(args.stokes)}')
    print(f'[moon-fit] fit     : {args.fit}')
    print(f'[moon-fit] out     : {out_path}')

    q.plot_vis_amp_vs_uvdist(
        vis,
        title=f'{source_name} visibility envelope fit ({args.fit})',
        fit=args.fit,
        show_phase=False,
        alpha=0.08,
        save_path=out_path,
    )

    print('[moon-fit] done')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
