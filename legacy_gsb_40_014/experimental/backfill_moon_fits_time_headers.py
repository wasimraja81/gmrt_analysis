#!/usr/bin/env python3
"""Backfill timing + bookkeeping metadata in moon selfcal FITS headers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from fits_time_headers import write_extra_header_to_fits_image
from stack_moon_snapshots import gmrt_location, moon_radec_at_jd


def _parse_stack_label(fits_path: Path) -> tuple[int, int]:
    name = fits_path.name
    m = re.search(r'_stk(\d+)_n(\d+)', name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r'_int(\d+)', name)
    if m:
        return int(m.group(1)), 1
    raise ValueError(f'Could not parse stack start/n from filename: {name}')


def _infer_inttime_sec(jd_start: float, jd_mid: float, nint: int) -> float:
    k = int(nint) // 2
    if k > 0:
        return float((jd_mid - jd_start) * 86400.0 / k)
    return 8.0


def _load_ephem_for_fits(fits_path: Path) -> dict:
    # Prefer exact sibling based on final label naming convention
    sibling = fits_path.with_name(fits_path.stem.replace('_final', '_ephemeris') + '.json')
    if sibling.exists():
        return json.loads(sibling.read_text())

    # Fallback: one ephemeris file per directory in selfcal outputs
    cand = sorted(fits_path.parent.glob('*_ephemeris.json'))
    if not cand:
        raise FileNotFoundError(f'No ephemeris json found for {fits_path}')
    return json.loads(cand[0].read_text())


def _build_uvfits_scan_jds(uvfits_path: Path, scan_name: str, index_path: Path | None) -> np.ndarray:
    src_dir = Path(__file__).resolve().parent.parent / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import modules.ugmrt_query as q  # type: ignore

    index = q.get_or_build_row_index(
        uvfits_path,
        cache_path=index_path,
        write_cache=True,
        force_rebuild=False,
    )

    id_to_name = index.get('id_to_name', {})
    name_to_id = {str(v).upper(): int(k) for k, v in id_to_name.items()}
    scan_upper = scan_name.upper()
    if scan_upper not in name_to_id:
        raise ValueError(f'Scan {scan_upper} not found in UVFITS index. Available={sorted(name_to_id)}')

    sid = name_to_id[scan_upper]
    all_jd = np.asarray(index['jd'], dtype=np.float64)
    all_sid = np.asarray(index['source_id'], dtype=np.int64)
    scan_jds = np.unique(all_jd[all_sid == sid])
    scan_jds.sort()
    if scan_jds.size == 0:
        raise ValueError(f'No JD rows found for scan {scan_upper} in UVFITS index.')
    return scan_jds


def _derive_from_uvfits(
    fits_path: Path,
    scan_jds: np.ndarray,
) -> dict:
    start_idx, nint = _parse_stack_label(fits_path)
    if start_idx < 0 or start_idx + nint > len(scan_jds):
        raise ValueError(
            f'Stack indices out of range for {fits_path.name}: start={start_idx} n={nint} '
            f'len(scan_jds)={len(scan_jds)}'
        )

    jd_stack = np.asarray(scan_jds[start_idx:start_idx + nint], dtype=np.float64)
    jd_start = float(jd_stack[0])
    jd_mid = float(jd_stack[len(jd_stack) // 2])
    jd_mean = float(np.mean(jd_stack))

    if jd_stack.size > 1:
        inttime_sec = float(np.median(np.diff(jd_stack)) * 86400.0)
    else:
        inttime_sec = 8.0
    jd_end = float(jd_start + (nint * inttime_sec) / 86400.0)

    moon_ra, moon_dec = moon_radec_at_jd(jd_mid, gmrt_location())

    return {
        'jd_start': jd_start,
        'jd_mid': jd_mid,
        'jd_mean': jd_mean,
        'jd_end': jd_end,
        'n': int(nint),
        'inttime_sec': inttime_sec,
        'start_integration': int(start_idx),
        'moon_ra_deg': float(moon_ra),
        'moon_dec_deg': float(moon_dec),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Backfill timing metadata into moon selfcal FITS headers')
    ap.add_argument('--selfcal-dir', required=True, help='Root directory containing moon selfcal outputs')
    ap.add_argument('--glob', default='*_final.fits', help='FITS glob under selfcal-dir (default: *_final.fits)')
    ap.add_argument('--include-cycles', action='store_true', help='Also patch *_sc*.fits cycle products')
    ap.add_argument('--uvfits', default='',
                    help='Optional original UVFITS for fallback timing derivation when ephemeris JSON is missing')
    ap.add_argument('--scan', default='MOON0520',
                    help='Scan/source name used with --uvfits fallback (default: MOON0520)')
    ap.add_argument('--index', default='',
                    help='Optional row-index cache path to use with --uvfits fallback')
    ap.add_argument('--prefer-uvfits', action='store_true',
                    help='Derive timing from UVFITS/index even if ephemeris JSON exists (for verification)')
    ap.add_argument('--dry-run', action='store_true', help='Print what would be patched without modifying files')
    args = ap.parse_args()

    root = Path(args.selfcal_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'selfcal-dir not found: {root}')

    fits_paths = sorted(root.rglob(args.glob))
    if args.include_cycles:
        fits_paths.extend(sorted(root.rglob('*_sc*.fits')))
        fits_paths = sorted(set(fits_paths))

    if not fits_paths:
        print(f'[backfill] no FITS matched under {root}')
        return 0

    scan_jds = None
    if args.uvfits.strip():
        uvfits_path = Path(args.uvfits).expanduser().resolve()
        if not uvfits_path.exists():
            raise FileNotFoundError(f'UVFITS not found: {uvfits_path}')
        index_path = Path(args.index).expanduser().resolve() if args.index.strip() else None
        scan_jds = _build_uvfits_scan_jds(uvfits_path, args.scan, index_path)
        print(f'[backfill] UVFITS fallback enabled: scan={args.scan.upper()} n_scan_jds={len(scan_jds)}')

    patched = 0
    skipped = 0

    for fp in fits_paths:
        try:
            if args.prefer_uvfits:
                if scan_jds is None:
                    raise ValueError('--prefer-uvfits requires --uvfits')
                ep = _derive_from_uvfits(fp, scan_jds)
            else:
                try:
                    ep = _load_ephem_for_fits(fp)
                except Exception:
                    if scan_jds is None:
                        raise
                    ep = _derive_from_uvfits(fp, scan_jds)

            jd_start = float(ep['jd_start'])
            jd_mid = float(ep['jd_mid'])
            nint = int(ep.get('n', 1))
            inttime_sec = float(ep.get('inttime_sec', _infer_inttime_sec(jd_start, jd_mid, nint)))
            jd_end = float(ep.get('jd_end', jd_start + (nint * inttime_sec) / 86400.0))
            jd_mean = float(ep.get('jd_mean', jd_mid))

            moon_ra = float(ep['moon_ra_deg']) if 'moon_ra_deg' in ep else None
            moon_dec = float(ep['moon_dec_deg']) if 'moon_dec_deg' in ep else None
            start_int = int(ep['start_integration']) if 'start_integration' in ep else None

            if args.dry_run:
                print(f'[dry-run] {fp}')
                print(f'          jd_start={jd_start:.9f} jd_mid={jd_mid:.9f} jd_end={jd_end:.9f} nint={nint} inttime={inttime_sec:.3f}s')
            else:
                write_extra_header_to_fits_image(
                    fp,
                    jd_start=jd_start,
                    jd_end=jd_end,
                    jd_avg=jd_mid,
                    jd_mid=jd_mid,
                    jd_mean=jd_mean,
                    nint=nint,
                    inttime_sec=inttime_sec,
                    start_integration=start_int,
                    moon_ra_deg=moon_ra,
                    moon_dec_deg=moon_dec,
                    source_tag='backfill_moon_fits_time_headers',
                )
                patched += 1
        except Exception as exc:
            skipped += 1
            print(f'[backfill] skip {fp}: {exc}')

    print(f'[backfill] total={len(fits_paths)} patched={patched} skipped={skipped} dry_run={args.dry_run}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
