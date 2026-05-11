#!/usr/bin/env python3
"""Moon per-integration imaging + phase-only selfcal development/tuning script.

Designed for iterative parameter tuning: operate on a small subset of
integrations from one Moon scan rather than the full scan.

Workflow per integration
------------------------
  For each requested integration index i (0-based within the scan):

  1.  Split that single time-slot from the full-scan MS into a scratch MS.
  2.  Compute Moon apparent RA/Dec via astropy ephemeris at the integration's JD.
  3.  Build a CASA circular mask centred on the Moon.

  Selfcal cycle 1  (shallow, restricted uv):
  4a. tclean(niter=NITER1, uvrange=UVRANGE1, mask=moon_mask)   → initial model
  4b. gaincal(calmode='p', solint='int', uvrange=UVRANGE1)
  4c. applycal()

  Selfcal cycle 2  (deeper, relaxed uv):
  5a. tclean(niter=NITER2, uvrange=UVRANGE2, mask=moon_mask)   → refined model
  5b. gaincal(calmode='p', solint='int', uvrange=UVRANGE2)
  5c. applycal()

  Final image:
  6.  tclean(niter=NITER_FINAL, uvrange=UVRANGE2, mask=moon_mask)
  7.  exportfits → <outdir>/<scan>/<label>_int<i>_final.fits

Usage (normal Python / casatasks venv)
--------------------------------------
  python experimental/moon_selfcal_dev.py \\
      --scan MOON0520 \\
      --integrations 0 5 10 20 50 \\
      --uvfits ~/DATA/gmrt_40_014/work/split/moon/moon0520_primary_secondary_calibrated_flagged.uvfits \\
      --index  ~/DATA/gmrt_40_014/work/40_014_25jul2021_gsb.index.npz \\
      --outdir ~/DATA/gmrt_40_014/work/casa_selfcal/moon0520_dev

All imaging / selfcal parameters are exposed as CLI flags with sensible
defaults so you can tune without editing the script.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, get_body
from astropy.time import Time
from astropy import units as u


# ---------------------------------------------------------------------------
# GMRT array reference position (derived from antenna metadata)
# ---------------------------------------------------------------------------

def _gmrt_location(meta: dict) -> EarthLocation:
    """Return GMRT reference EarthLocation from ITRF antenna offsets in metadata.

    The metadata stores positions as local offsets (x, y, z) in metres
    relative to C02 (the array reference).  We use the standard GMRT ITRF
    geocentric coordinates for C02 and add the offset to get an approximate
    array centre.
    """
    # GMRT reference (C02) ITRF coordinates (from published GMRT documents)
    ref_x = 1656342.30
    ref_y = 5797947.77
    ref_z = 2073243.16  # metres

    active = [a for a in meta['antennas']
              if not any(a['name'].startswith(d) for d in ['C07', 'S05'])]
    offsets = np.array([[a['x_m'], a['y_m'], a['z_m']] for a in active])
    mean_off = offsets.mean(axis=0)
    return EarthLocation.from_geocentric(
        ref_x + mean_off[0],
        ref_y + mean_off[1],
        ref_z + mean_off[2],
        unit='m',
    )


# ---------------------------------------------------------------------------
# Ephemeris
# ---------------------------------------------------------------------------

def moon_radec_at_jd(jd: float, location: EarthLocation) -> tuple[float, float]:
    """Return (ra_deg, dec_deg) of the Moon in ICRS at the given JD (TT~TDB)."""
    t = Time(jd, format='jd', scale='utc')
    moon = get_body('moon', t, location=location)
    c = moon.transform_to('icrs')
    return float(c.ra.deg), float(c.dec.deg)


# ---------------------------------------------------------------------------
# CASA mask string
# ---------------------------------------------------------------------------

def _casa_circle_mask(ra_deg: float, dec_deg: float, radius_arcmin: float) -> str:
    """Return a CASA tclean-compatible circular mask region string."""
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    ra_str = c.ra.to_string(unit=u.hour, sep=':', precision=3, pad=True)
    dec_str = c.dec.to_string(unit=u.deg, sep='.', precision=2, alwayssign=True, pad=True)
    return f"circle[[{ra_str}, {dec_str}], {radius_arcmin:.1f}arcmin]"


# ---------------------------------------------------------------------------
# CASA import helpers
# ---------------------------------------------------------------------------

def _import_casa() -> Any:
    try:
        from casatasks import exportfits, gaincal, applycal, importuvfits, split, tclean  # type: ignore
    except ImportError as exc:
        sys.exit(
            f'casatasks not available: {exc}\n'
            'Activate the CASA-enabled venv (gmrt) before running this script.'
        )
    return importuvfits, split, tclean, gaincal, applycal, exportfits


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Moon per-integration imaging + phase-only selfcal (dev/tuning)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data selection ──
    io = p.add_argument_group('I/O')
    io.add_argument('--scan', required=True,
                    help='Moon scan name as in the index file, e.g. MOON0520')
    io.add_argument('--uvfits', required=True,
                    help='Path to the calibrated split UVFITS for this scan')
    io.add_argument('--index', default=None,
                    help='Path to the row index cache NPZ for the split UVFITS '
                         '(default: <uvfits>.row_index_cache.npz alongside the UVFITS)')
    io.add_argument('--outdir', required=True,
                    help='Output directory for MS, images, and FITS frames')
    io.add_argument('--integrations', nargs='+', type=int, default=[0],
                    help='0-based integration indices within the scan to process')
    io.add_argument('--overwrite', action='store_true',
                    help='Delete existing outputs and re-run')

    # ── Image geometry ──
    img = p.add_argument_group('Image geometry')
    img.add_argument('--cell', default='1.5arcsec',
                     help='CASA cell size')
    img.add_argument('--imsize', type=int, default=2048,
                     help='Image size in pixels (square)')
    img.add_argument('--stokes', default='I',
                     help='Stokes planes to image')
    img.add_argument('--weighting', default='briggs',
                     choices=['natural', 'uniform', 'briggs'])
    img.add_argument('--robust', type=float, default=0.0,
                     help='Briggs robust parameter')
    img.add_argument('--deconvolver', default='multiscale',
                     choices=['multiscale', 'hogbom', 'clark'])
    img.add_argument('--scales', default='0,5,15,45',
                     help='Multiscale clean scales in pixels (comma-separated)')

    # ── Moon mask ──
    msk = p.add_argument_group('Moon mask')
    msk.add_argument('--mask-radius-arcmin', type=float, default=20.0,
                     help='Radius of circular CLEAN mask around Moon centre (arcmin). '
                          'Moon disk ~15 arcmin radius at this freq; default adds margin.')

    # ── Per-cycle parameters (arrays indexed by selfcal cycle) ───────────────
    # Each argument is a comma-separated list with one value per selfcal cycle.
    # To add a 3rd cycle, simply append a value to every list.
    # Scales uses | to separate cycles (comma is used within each cycle's scale list).
    cyc = p.add_argument_group(
        'Per-cycle selfcal parameters',
        'Each value is a comma-separated list — one entry per selfcal cycle.\n'
        'All lists must have the same length (= number of cycles).\n'
        'Example for 2 cycles: --niter-per-cycle 100,300'
    )
    cyc.add_argument('--niter-per-cycle', default='100,300',
                     help='tclean niter per cycle (comma-separated). '
                          'Typically shallow→deeper as model improves.')
    cyc.add_argument('--uvmin-per-cycle', default='0.5,0.12',
                     help='Min uv in kλ per cycle (comma-separated). '
                          'Start restricted, then relax to include short baselines.')
    cyc.add_argument('--uvmax-per-cycle', default=',',
                     help='Max uv in kλ per cycle (comma-separated, empty = no limit).')
    cyc.add_argument('--scales-per-cycle', default='0,5,15|0,5,15,45',
                     help='Multiscale clean scales (pixels) per cycle, '
                          'cycles separated by |, scales within a cycle by comma. '
                          'E.g. "0,5,15|0,5,15,45" for 2 cycles.')

    # ── Final image (after all selfcal cycles) ────────────────────────────────
    fin = p.add_argument_group('Final image (after all selfcal cycles)')
    fin.add_argument('--niter-final', type=int, default=500,
                     help='tclean niter for the final image')
    fin.add_argument('--threshold', default='0mJy',
                     help='CLEAN threshold applied in all rounds')
    fin.add_argument('--cycleniter', type=int, default=100,
                     help='Minor cycles per major cycle (all rounds)')

    # ── Selfcal ──
    sc = p.add_argument_group('Selfcal')
    sc.add_argument('--minsnr', type=float, default=3.0,
                    help='gaincal minsnr parameter (applied to all cycles)')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Parse per-cycle parameter arrays from CLI strings
# ---------------------------------------------------------------------------

def _parse_cycle_params(args: argparse.Namespace) -> list[dict]:
    """Convert per-cycle comma/pipe-separated CLI strings into a list of dicts.

    Returns a list of length N (one dict per cycle) with keys:
      niter, uvmin_kl, uvmax_kl, scales
    Raises ValueError if the lists have inconsistent lengths.
    """
    niters   = [int(v.strip())   for v in args.niter_per_cycle.split(',')]
    uvmins   = [float(v.strip()) if v.strip() else None
                for v in args.uvmin_per_cycle.split(',')]
    uvmaxs   = [float(v.strip()) if v.strip() else None
                for v in args.uvmax_per_cycle.split(',')]
    scales_blocks = args.scales_per_cycle.split('|')

    lengths = {'niter': len(niters), 'uvmin': len(uvmins),
               'uvmax': len(uvmaxs), 'scales': len(scales_blocks)}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f'Per-cycle parameter lists must all have the same length.\n'
            f'Got: {lengths}\n'
            f'  --niter-per-cycle  = {args.niter_per_cycle!r}\n'
            f'  --uvmin-per-cycle  = {args.uvmin_per_cycle!r}\n'
            f'  --uvmax-per-cycle  = {args.uvmax_per_cycle!r}\n'
            f'  --scales-per-cycle = {args.scales_per_cycle!r}'
        )

    cycles = []
    for i, (niter, uvmin, uvmax, sc_str) in enumerate(
            zip(niters, uvmins, uvmaxs, scales_blocks)):
        sc_vals = sorted({int(v.strip()) for v in sc_str.split(',') if v.strip()})
        if 0 not in sc_vals:
            sc_vals.insert(0, 0)
        cycles.append({'niter': niter, 'uvmin_kl': uvmin,
                        'uvmax_kl': uvmax, 'scales': sc_vals})
        print(f'[selfcal-dev] Cycle {i+1} params: niter={niter}, '
              f'uvmin={uvmin}kλ, uvmax={uvmax}kλ, scales={sc_vals}')
    return cycles


# ---------------------------------------------------------------------------
# UV range string helpers
# ---------------------------------------------------------------------------

def _uvrange(lo_kl: float | None, hi_kl: float | None) -> str:
    if lo_kl is None and hi_kl is None:
        return ''
    if hi_kl is None:
        return f'>{lo_kl:.3f}klambda'
    if lo_kl is None:
        return f'<{hi_kl:.3f}klambda'
    return f'{lo_kl:.3f}~{hi_kl:.3f}klambda'


# ---------------------------------------------------------------------------
# Single-integration selfcal loop
# ---------------------------------------------------------------------------

def _process_integration(
    *,
    idx: int,
    jd: float,
    full_ms: Path,
    outdir: Path,
    location: EarthLocation,
    args: argparse.Namespace,
    cycles: list[dict],
    casa: tuple,
) -> Path:
    _, split_task, tclean, gaincal, applycal, exportfits = casa

    label = f'{args.scan.lower()}_int{idx:04d}'
    intdir = outdir / label
    if args.overwrite and intdir.exists():
        shutil.rmtree(intdir)
    intdir.mkdir(parents=True, exist_ok=True)

    # ── 1. Split single integration ──────────────────────────────────────────
    scratch_ms = str(intdir / f'{label}.ms')
    if Path(scratch_ms).exists():
        shutil.rmtree(scratch_ms)

    # Convert JD to CASA time range string  (YYYY/MM/DD/HH:MM:SS±1s)
    t_mid = Time(jd, format='jd', scale='utc')
    dt_half = 4.0  # half-dump + tiny margin, seconds
    t0 = (t_mid - dt_half * u.s).strftime('%Y/%m/%d/%H:%M:%S')
    t1 = (t_mid + dt_half * u.s).strftime('%Y/%m/%d/%H:%M:%S')
    timerange = f'{t0}~{t1}'

    print(f'\n[selfcal-dev] === Integration {idx}  JD={jd:.6f}  timerange={timerange} ===')
    split_task(vis=str(full_ms), outputvis=scratch_ms, timerange=timerange, datacolumn='data')

    # ── 2. Moon ephemeris → mask ──────────────────────────────────────────────
    ra_deg, dec_deg = moon_radec_at_jd(jd, location)
    mask_str = _casa_circle_mask(ra_deg, dec_deg, args.mask_radius_arcmin)
    print(f'[selfcal-dev]   Moon at RA={ra_deg:.4f}°  Dec={dec_deg:.4f}°  mask={mask_str}')

    imname_base = str(intdir / label)

    common_tclean = dict(
        vis=scratch_ms,
        stokes=args.stokes,
        cell=[args.cell],
        imsize=[args.imsize, args.imsize],
        weighting=args.weighting,
        robust=args.robust,
        deconvolver=args.deconvolver,
        threshold=args.threshold,
        cycleniter=args.cycleniter,
        mask=mask_str,
        gridder='standard',
        normtype='flatnoise',
        pbcor=False,
        pblimit=-1.0,
        savemodel='modelcolumn',
        verbose=True,
    )

    # ── 3. Selfcal loop (N cycles, all params from per-cycle arrays) ──────────
    for c_idx, cyc in enumerate(cycles):
        c_num = c_idx + 1
        uvrange_c = _uvrange(cyc['uvmin_kl'], cyc['uvmax_kl'])
        scales_c = cyc['scales'] if args.deconvolver == 'multiscale' else []

        print(f'[selfcal-dev]   Cycle {c_num}/{len(cycles)}: '
              f'tclean niter={cyc["niter"]}  uvrange={uvrange_c!r}  scales={scales_c}')
        imagename_c = imname_base + f'_sc{c_num}'
        _clean_fresh(tclean, imagename=imagename_c, niter=cyc['niter'],
                     uvrange=uvrange_c, scales=scales_c, **common_tclean)

        cal_c = str(intdir / f'{label}_sc{c_num}.gcal')
        print(f'[selfcal-dev]   Cycle {c_num}/{len(cycles)}: gaincal → {cal_c}')
        gaincal(
            vis=scratch_ms,
            caltable=cal_c,
            gaintype='G',
            calmode='p',
            solint='inf',      # one phase solution per antenna for the 8s dump
            uvrange=uvrange_c,
            minsnr=args.minsnr,
            append=False,
        )

        print(f'[selfcal-dev]   Cycle {c_num}/{len(cycles)}: applycal')
        applycal(vis=scratch_ms, gaintable=[cal_c], calwt=False, flagbackup=False)

    # ── 4. Final image (uses last cycle's uv/scale params) ────────────────────
    last = cycles[-1]
    uvrange_final = _uvrange(last['uvmin_kl'], last['uvmax_kl'])
    scales_final = last['scales'] if args.deconvolver == 'multiscale' else []
    print(f'[selfcal-dev]   Final image: tclean niter={args.niter_final}  uvrange={uvrange_final!r}  scales={scales_final}')
    imagename_final = imname_base + '_final'
    _clean_fresh(tclean, imagename=imagename_final, niter=args.niter_final,
                 uvrange=uvrange_final, scales=scales_final, **common_tclean)

    # ── 6. Export FITS ────────────────────────────────────────────────────────
    fits_out = str(intdir / f'{label}_final.fits')
    exportfits(imagename=imagename_final + '.image', fitsimage=fits_out, overwrite=True)
    print(f'[selfcal-dev]   Exported: {fits_out}')

    # Save ephemeris record alongside image
    ephem_file = intdir / f'{label}_ephemeris.json'
    ephem_file.write_text(json.dumps({
        'integration_index': idx,
        'jd': jd,
        'moon_ra_deg': ra_deg,
        'moon_dec_deg': dec_deg,
        'mask': mask_str,
        'fits_out': fits_out,
    }, indent=2))

    return Path(fits_out)


def _clean_fresh(tclean, *, imagename: str, niter: int, uvrange: str, **kwargs) -> None:
    """Remove any existing image products then run tclean from scratch."""
    for suffix in ['.image', '.model', '.psf', '.residual', '.pb', '.sumwt', '.mask']:
        p = Path(imagename + suffix)
        if p.exists():
            shutil.rmtree(str(p)) if p.is_dir() else p.unlink()
    tclean(imagename=imagename, niter=niter, uvrange=uvrange,
           calcpsf=True, calcres=True, restart=False, **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load index ────────────────────────────────────────────────────────────
    # Default: row index cache sitting beside the split UVFITS.
    # This cache reflects the actual split frequencies and only contains rows
    # for this scan — strictly more correct than the raw session index.
    index_path = Path(args.index) if args.index else Path(args.uvfits).with_suffix(
        Path(args.uvfits).suffix + '.row_index_cache.npz'
    )
    if not index_path.exists():
        sys.exit(f'ERROR: index cache not found: {index_path}\n'
                 f'Run visSplit first, or pass --index explicitly.')
    print(f'[selfcal-dev] Using index: {index_path}')
    idx_npz = np.load(index_path, allow_pickle=True)
    meta = json.loads(str(idx_npz['metadata_json']))

    # Map scan name → source id
    name_to_id = {v: int(k) for k, v in meta['id_to_name'].items()}
    scan_upper = args.scan.upper()
    if scan_upper not in name_to_id:
        sys.exit(f'ERROR: scan {args.scan!r} not found in index. '
                 f'Available: {sorted(name_to_id)}')
    src_id = name_to_id[scan_upper]

    # Unique sorted JDs for this scan
    all_jd = idx_npz['jd']
    all_sid = idx_npz['source_id']
    scan_jds = np.unique(all_jd[all_sid == src_id])
    scan_jds.sort()

    print(f'[selfcal-dev] Scan {scan_upper} (id={src_id}): '
          f'{len(scan_jds)} integrations, dump={8.0:.0f}s')

    # Validate requested integration indices
    bad = [i for i in args.integrations if i < 0 or i >= len(scan_jds)]
    if bad:
        sys.exit(f'ERROR: integration indices out of range [0, {len(scan_jds)-1}]: {bad}')

    print(f'[selfcal-dev] Processing integrations: {args.integrations}')

    # ── GMRT location ─────────────────────────────────────────────────────────
    location = _gmrt_location(meta)

    # ── Import UVFITS → full-scan MS (once) ───────────────────────────────────
    casa = _import_casa()
    importuvfits = casa[0]

    full_ms = outdir / f'{scan_upper.lower()}_full.ms'
    if full_ms.exists() and args.overwrite:
        shutil.rmtree(full_ms)
    if not full_ms.exists():
        print(f'[selfcal-dev] Importing UVFITS → {full_ms}')
        importuvfits(fitsfile=str(args.uvfits), vis=str(full_ms))
    else:
        print(f'[selfcal-dev] Reusing existing MS: {full_ms}')

    # ── Per-integration loop ──────────────────────────────────────────────────
    results: list[dict] = []
    cycles = _parse_cycle_params(args)

    for idx_i in sorted(args.integrations):
        jd_i = float(scan_jds[idx_i])
        fits_path = _process_integration(
            idx=idx_i,
            jd=jd_i,
            full_ms=full_ms,
            outdir=outdir,
            location=location,
            args=args,
            cycles=cycles,
            casa=casa,
        )
        results.append({'integration': idx_i, 'jd': jd_i, 'fits': str(fits_path)})

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_file = outdir / 'summary.json'
    summary_file.write_text(json.dumps({'scan': scan_upper, 'results': results}, indent=2))
    print(f'\n[selfcal-dev] Done.  {len(results)} integrations processed.')
    print(f'[selfcal-dev] Summary: {summary_file}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
