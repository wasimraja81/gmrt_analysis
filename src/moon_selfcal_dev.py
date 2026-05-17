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
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import Angle, EarthLocation, SkyCoord, get_body
from astropy.time import Time
from astropy import units as u

from fits_time_headers import write_extra_header_to_fits_image


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
    """Return apparent topocentric (ra_deg, dec_deg) of the Moon at the given JD.

    Uses astropy get_body which returns the GCRS (Geocentric Celestial Reference
    System) frame for the observer location.  For solar system bodies the GCRS
    RA/Dec is the *apparent* topocentric direction — the coordinate system that
    matches what the telescope tracked and what is stored in the MS FIELD table.

    Do NOT transform to ICRS: for the Moon that gives the barycentric direction
    from the solar system barycentre, which is ~25 degrees different.
    """
    t = Time(jd, format='jd', scale='utc')
    moon = get_body('moon', t, location=location)
    # moon is in GCRS frame; .ra/.dec give the apparent topocentric RA/Dec
    ra_deg = float(moon.ra.deg)   # already 0–360
    dec_deg = float(moon.dec.deg)
    return ra_deg, dec_deg


# ---------------------------------------------------------------------------
# CASA mask string
# ---------------------------------------------------------------------------

def _casa_circle_mask(ra_deg: float, dec_deg: float, radius_arcmin: float) -> str:
    """Return a CASA tclean-compatible sky-coordinate circular mask string."""
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    ra_str  = c.ra.to_string(unit=u.hour, sep=':', precision=3, pad=True)
    dec_str = c.dec.to_string(unit=u.deg, sep='.', precision=2,
                              alwayssign=True, pad=True)
    return f"circle[[{ra_str}, {dec_str}], {radius_arcmin:.1f}arcmin]"


def _casa_phasecenter_j2000(ra_deg: float, dec_deg: float) -> str:
    """Return CASA tclean-compatible phasecenter string from input RA/Dec.

    This function only formats the provided sky coordinates; it does not
    transform frames.
    """
    ra_str = Angle(ra_deg * u.deg).to_string(unit=u.hour, sep='hms',
                                             precision=3, pad=True)
    dec_str = Angle(dec_deg * u.deg).to_string(unit=u.deg, sep='dms',
                                               precision=2, alwayssign=True,
                                               pad=True)
    return f'J2000 {ra_str} {dec_str}'


def _ms_field_radec(ms_path: str) -> tuple[float, float]:
    """Return (ra_deg, dec_deg) of the FIELD phase direction actually used.

    CASA split preserves all FIELD rows from the parent MS even when only one
    field is present in the data.  We query FIELD_ID from the MAIN table to
    find which row is actually referenced, then read PHASE_DIR for that row.
    PHASE_DIR is in radians, RA in [-pi, pi]; wrap to [0, 360).
    """
    from casatools import table as tb_tool  # type: ignore
    tb = tb_tool()
    # Find the field ID used in this MS
    tb.open(ms_path)
    try:
        field_ids = tb.getcol('FIELD_ID')
    finally:
        tb.close()
    field_id = int(np.unique(field_ids)[0])
    # Read that row from the FIELD sub-table
    tb.open(os.path.join(ms_path, 'FIELD'))
    try:
        phase_dir = tb.getcol('PHASE_DIR')  # shape (2, n_poly, n_rows)
        ra_rad  = float(phase_dir[0, 0, field_id])
        dec_rad = float(phase_dir[1, 0, field_id])
    finally:
        tb.close()
    ra_deg  = float(np.degrees(ra_rad)) % 360.0   # wrap [-180,180) → [0,360)
    dec_deg = float(np.degrees(dec_rad))
    return ra_deg, dec_deg


def _angular_offset_arcmin(ra1: float, dec1: float,
                           ra2: float, dec2: float) -> float:
    """Great-circle angular separation in arcmin between two ICRS positions."""
    c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg, frame='icrs')
    c2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg, frame='icrs')
    return float(c1.separation(c2).arcmin)


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
# UVFITS pre-processing helpers
# ---------------------------------------------------------------------------

def _patch_uvfits_veldef(uvfits_path: Path, tmp_dir: Path) -> Path:
    """Return a UVFITS path that is safe to pass to CASA importuvfits.

    Two patches are applied if needed:

    1. VELDEF / SPECSYS — casacore requires VELDEF in the primary HDU to
       resolve the frequency reference frame; GMRT visSplit output omits it.

    2. PMRA / PMDEC zeroing in the SU table — non-zero values cause casacore
       to set numPoly=1 and use apparent coordinates (MDirection::APP), which
       requires a full epoch+position Measures frame.  For the Moon, these
       fields hold the angular velocity (not a stellar proper motion) and
       should not be interpreted that way.  Setting them to zero forces
       casacore to use J2000 RAEPO/DECEPO with numPoly=0.

    If no patches are needed the original path is returned unchanged.
    Otherwise a patched copy is written to *tmp_dir* and that path is returned.
    The caller should delete tmp_dir when done.
    """
    from astropy.io import fits  # type: ignore
    import numpy as np

    uvfits_path = Path(uvfits_path)
    with fits.open(str(uvfits_path), memmap=False) as hdul:
        hdr = hdul[0].header
        need_veldef = hdr.get('VELDEF') is None
        need_pmra_zero = False
        for ext_hdu in hdul:
            if hasattr(ext_hdu, 'name') and 'SU' in str(ext_hdu.name):
                if hasattr(ext_hdu, 'data') and ext_hdu.data is not None:
                    if 'PMRA' in ext_hdu.columns.names:
                        if np.any(ext_hdu.data['PMRA'] != 0.0):
                            need_pmra_zero = True
                    if 'PMDEC' in ext_hdu.columns.names:
                        if np.any(ext_hdu.data['PMDEC'] != 0.0):
                            need_pmra_zero = True

        if not need_veldef and not need_pmra_zero:
            return uvfits_path          # already correct — nothing to do

        msgs = []
        if need_veldef:
            msgs.append('VELDEF=RADIO, SPECSYS=TOPOCENT')
        if need_pmra_zero:
            msgs.append('PMRA/PMDEC=0 in SU table')
        print(f'[selfcal-dev] Patching UVFITS for CASA ({", ".join(msgs)})')
        tmp_dir.mkdir(parents=True, exist_ok=True)
        patched = tmp_dir / (uvfits_path.stem + '_casa_patched.uvfits')

        if need_veldef:
            hdul[0].header['VELDEF'] = ('RADIO', 'Radio velocity convention (topocentric)')
            hdul[0].header['SPECSYS'] = ('TOPOCENT', 'Spectral reference frame')
        if need_pmra_zero:
            for i, ext_hdu in enumerate(hdul):
                if hasattr(ext_hdu, 'name') and 'SU' in str(ext_hdu.name):
                    if hasattr(ext_hdu, 'data') and ext_hdu.data is not None:
                        if 'PMRA' in ext_hdu.columns.names:
                            ext_hdu.data['PMRA'][:] = 0.0
                        if 'PMDEC' in ext_hdu.columns.names:
                            ext_hdu.data['PMDEC'][:] = 0.0
        hdul.writeto(str(patched), overwrite=True)
    return patched


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
    io.add_argument('--build-index-if-missing', action='store_true',
                    help='If the row index cache is missing, build it on-the-fly '
                         'from the UVFITS. Useful when imaging a re-flagged UVFITS '
                         'whose index has not yet been generated.')
    io.add_argument('--outdir', required=True,
                    help='Output directory for MS, images, and FITS frames')
    io.add_argument('--integrations', nargs='+', type=int, default=[0],
                    help='0-based integration indices within the scan to process')
    io.add_argument('--stack-size', type=int, default=1,
                    help='Number of consecutive integrations to combine into one MS for '
                         'imaging + selfcal.  Each stack produces one gaincal solution '
                         'and one output image.  Default 1 = per-integration (legacy). '
                         'E.g. 10 = combine 10×8s=80s per stack; Moon smears ~3" '
                         'which is well within the 8" full-resolution beam.')
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
    msk.add_argument('--no-mask', action='store_true', default=False,
                     help='Run all tclean calls with mask="" (no mask). '
                          'Useful for diagnosing whether the Moon disk mask is biasing the selfcal.')
    msk.add_argument('--use-tclean-phasecenter', action='store_true', default=False,
                     help='If set, pass phasecenter to tclean using the same '
                         'Moon ephemeris position used for mask centering. '
                         'This recenters imaging on the Moon without altering visibilities.')

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
                     help='Min uv in kλ per cycle for tclean imaging (comma-separated).')
    cyc.add_argument('--uvmax-per-cycle', default=',',
                     help='Max uv in kλ per cycle for tclean imaging (comma-separated, empty = no limit).')
    cyc.add_argument('--uvmin-cal-per-cycle', default=None,
                     help='Min uv in kλ per cycle for gaincal (comma-separated). '
                          'If omitted, falls back to --uvmin-per-cycle. '
                          'Use to restrict calibration to short baselines where Moon SNR is highest.')
    cyc.add_argument('--uvmax-cal-per-cycle', default=None,
                     help='Max uv in kλ per cycle for gaincal (comma-separated, empty = no limit). '
                          'If omitted, falls back to --uvmax-per-cycle. '
                          'E.g. "2.0,5.0,10.0," for progressive short-to-all-baseline strategy.')
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
    fin.add_argument('--negativethreshold', type=float, default=0.0,
                     help='tclean negativethreshold (Jy/beam): stop minor cycle if a negative '
                          'residual peak exceeds this absolute value. '
                          '0.0 = disabled (default CASA behaviour, clean both +/-). '
                          'Set >0 (e.g. 0.001) to restrict to positive-only cleaning.')

    # ── Selfcal ──
    sc = p.add_argument_group('Selfcal')
    sc.add_argument('--minsnr', type=float, default=3.0,
                    help='gaincal minsnr parameter (applied to all cycles)')
    sc.add_argument('--loop-gain', type=float, default=0.05,
                    help='CLEAN loop gain (tclean gain parameter). '
                         'Default 0.05 is conservative for extended emission like the Moon. '
                         'CASA default is 0.1.')
    sc.add_argument('--solmode', default='',
                    help='gaincal robust solver mode: "" (standard LS), "L1", "R", "L1R". '
                         'Default "" = standard LS, reliable with few antennas (~8 at short BLs). '
                         'L1R needs many baselines to distinguish signal from outliers; '
                         'use only when >~20 antennas contribute to the solve.')
    sc.add_argument('--refant', default='1',
                    help='gaincal reference antenna NAME (as it appears in the MS ANTENNA table). '
                         'Default "1" = physical antenna 1 = C00:01 (central arm, 100%% bandpass valid). '
                         'Pass the CASA name string, not a 0-based integer index.')
    sc.add_argument('--refantmode', default='flex',
                    help='gaincal refantmode: "flex" (switch if refant drops out) or '
                         '"strict" (flag all solutions if refant absent). Default flex.')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Parse per-cycle parameter arrays from CLI strings
# ---------------------------------------------------------------------------

def _parse_cycle_params(args: argparse.Namespace) -> list[dict]:
    """Convert per-cycle comma/pipe-separated CLI strings into a list of dicts.

    Returns a list of length N (one dict per cycle) with keys:
      niter, uvmin_kl, uvmax_kl, cal_uvmin_kl, cal_uvmax_kl, scales
    The cal_* keys hold the uvrange used for gaincal; if --uvmin/max-cal-per-cycle
    are not supplied they fall back to the imaging uvrange values.
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

    # Parse optional per-cycle calibration uvrange; fall back to imaging values if not given.
    n_cycles = len(niters)
    if args.uvmin_cal_per_cycle is not None:
        cal_uvmins = [float(v.strip()) if v.strip() else None
                      for v in args.uvmin_cal_per_cycle.split(',')]
        if len(cal_uvmins) != n_cycles:
            raise ValueError(
                f'--uvmin-cal-per-cycle has {len(cal_uvmins)} entries but '
                f'{n_cycles} cycles were defined by --niter-per-cycle.')
    else:
        cal_uvmins = uvmins  # fall back to imaging uvmin

    if args.uvmax_cal_per_cycle is not None:
        cal_uvmaxs = [float(v.strip()) if v.strip() else None
                      for v in args.uvmax_cal_per_cycle.split(',')]
        if len(cal_uvmaxs) != n_cycles:
            raise ValueError(
                f'--uvmax-cal-per-cycle has {len(cal_uvmaxs)} entries but '
                f'{n_cycles} cycles were defined by --niter-per-cycle.')
    else:
        cal_uvmaxs = uvmaxs  # fall back to imaging uvmax

    cycles = []
    for i, (niter, uvmin, uvmax, cal_uvmin, cal_uvmax, sc_str) in enumerate(
            zip(niters, uvmins, uvmaxs, cal_uvmins, cal_uvmaxs, scales_blocks)):
        sc_vals = sorted({int(v.strip()) for v in sc_str.split(',') if v.strip()})
        if 0 not in sc_vals:
            sc_vals.insert(0, 0)
        cycles.append({'niter': niter, 'uvmin_kl': uvmin, 'uvmax_kl': uvmax,
                        'cal_uvmin_kl': cal_uvmin, 'cal_uvmax_kl': cal_uvmax,
                        'scales': sc_vals})
        print(f'[selfcal-dev] Cycle {i+1} params: niter={niter}, '
              f'img uvmin={uvmin}kλ uvmax={uvmax}kλ, '
              f'cal uvmin={cal_uvmin}kλ uvmax={cal_uvmax}kλ, scales={sc_vals}')
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
# Per-stack selfcal loop  (stack = 1 or more consecutive integrations)
# ---------------------------------------------------------------------------

def _process_stack(
    *,
    start_idx: int,
    jds: list[float],
    full_ms: Path,
    outdir: Path,
    location: EarthLocation,
    args: argparse.Namespace,
    cycles: list[dict],
    casa: tuple,
) -> Path:
    _, split_task, tclean, gaincal, applycal, exportfits = casa

    n = len(jds)
    if n == 1:
        label = f'{args.scan.lower()}_int{start_idx:04d}'
    else:
        label = f'{args.scan.lower()}_stk{start_idx:04d}_n{n:02d}'
    intdir = outdir / label
    if args.overwrite and intdir.exists():
        shutil.rmtree(intdir)
    intdir.mkdir(parents=True, exist_ok=True)

    # ── 1. Split stack time range ─────────────────────────────────────────────
    scratch_ms = str(intdir / f'{label}.ms')
    if Path(scratch_ms).exists():
        shutil.rmtree(scratch_ms)

    # Time range: from 4s before first dump to 4s after last dump
    dt_margin = 4.0  # seconds
    t_first = Time(jds[0],  format='jd', scale='utc')
    t_last  = Time(jds[-1], format='jd', scale='utc')
    t0 = (t_first - dt_margin * u.s).strftime('%Y/%m/%d/%H:%M:%S')
    t1 = (t_last  + dt_margin * u.s).strftime('%Y/%m/%d/%H:%M:%S')
    timerange = f'{t0}~{t1}'

    print(f'\n[selfcal-dev] === Stack start_idx={start_idx}  n={n}  '
          f'JD={jds[0]:.6f}..{jds[-1]:.6f}  timerange={timerange} ===')
    split_task(vis=str(full_ms), outputvis=scratch_ms, timerange=timerange, datacolumn='data')

    if n > 1:
        inttime_sec = float(np.median(np.diff(np.asarray(jds, dtype=np.float64))) * 86400.0)
    else:
        inttime_sec = 8.0

    jd_start = float(jds[0])
    jd_mid = float(jds[len(jds) // 2])
    jd_mean = float(np.mean(np.asarray(jds, dtype=np.float64)))
    jd_end = float(jd_start + (n * inttime_sec) / 86400.0)

    # ── 2. Moon ephemeris at midpoint of stack → sky position → mask ──────────
    ra_deg, dec_deg = moon_radec_at_jd(jd_mid, location)
    phasecenter_str = _casa_phasecenter_j2000(ra_deg, dec_deg)
    if args.no_mask:
        mask_str = ''
    else:
        mask_str = _casa_circle_mask(ra_deg, dec_deg, args.mask_radius_arcmin)

    # Check Moon is within the image before proceeding
    field_ra_deg, field_dec_deg = _ms_field_radec(scratch_ms)
    offset_arcmin = _angular_offset_arcmin(ra_deg, dec_deg,
                                           field_ra_deg, field_dec_deg)
    cell_arcsec   = float(args.cell.replace('arcsec', ''))
    halfwidth_arcmin = (args.imsize * cell_arcsec / 2.0) / 60.0
    margin_arcmin = halfwidth_arcmin - (0.0 if args.no_mask else args.mask_radius_arcmin)
    print(f'[selfcal-dev]   Moon at RA={ra_deg:.4f}°  Dec={dec_deg:.4f}°  '
          f'offset={offset_arcmin:.2f}arcmin from field centre  '
          f'(image half-width={halfwidth_arcmin:.1f}arcmin)')
    if offset_arcmin >= halfwidth_arcmin:
        print(f'[selfcal-dev]   WARNING: Moon centre ({offset_arcmin:.2f} arcmin) '
              f'is outside image half-width ({halfwidth_arcmin:.1f} arcmin). '
              f'Skipping stack starting at integration {start_idx}.')
        shutil.rmtree(str(intdir), ignore_errors=True)
        return None
    if not args.no_mask and offset_arcmin > margin_arcmin:
        print(f'[selfcal-dev]   NOTE: mask circle overhangs image edge '
              f'({offset_arcmin:.2f} + {args.mask_radius_arcmin:.1f} = '
              f'{offset_arcmin + args.mask_radius_arcmin:.2f} > {halfwidth_arcmin:.1f} arcmin) '
              f'-- CASA will clip it, proceeding.')
    print(f'[selfcal-dev]   mask-center (ephem): RA={ra_deg:.6f} deg  Dec={dec_deg:.6f} deg')
    if args.no_mask:
        print('[selfcal-dev]   mask: <disabled by --no-mask>')
    else:
        print(f'[selfcal-dev]   mask-radius: {args.mask_radius_arcmin:.2f} arcmin')
        print(f'[selfcal-dev]   mask-string: {mask_str}')

    if args.use_tclean_phasecenter:
        print(f'[selfcal-dev]   tclean phasecenter: ENABLED ({phasecenter_str})')
    else:
        print('[selfcal-dev]   tclean phasecenter: DISABLED (using MS/native phase center)')

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
        gain=args.loop_gain,
        negativethreshold=args.negativethreshold,
        mask=mask_str,
        gridder='standard',
        normtype='flatnoise',
        pbcor=False,
        pblimit=-1.0,
        savemodel='modelcolumn',
        verbose=True,
    )
    if args.use_tclean_phasecenter:
        common_tclean['phasecenter'] = phasecenter_str

    # ── 3. Selfcal loop (N cycles, all params from per-cycle arrays) ──────────
    for c_idx, cyc in enumerate(cycles):
        c_num = c_idx + 1
        uvrange_c     = _uvrange(cyc['uvmin_kl'],     cyc['uvmax_kl'])
        uvrange_cal_c = _uvrange(cyc['cal_uvmin_kl'], cyc['cal_uvmax_kl'])
        scales_c = cyc['scales'] if args.deconvolver == 'multiscale' else []

        print(f'[selfcal-dev]   Cycle {c_num}/{len(cycles)}: '
              f'tclean niter={cyc["niter"]}  img_uvrange={uvrange_c!r}  '
              f'cal_uvrange={uvrange_cal_c!r}  scales={scales_c}')
        imagename_c = imname_base + f'_sc{c_num}'
        _clean_fresh(tclean, imagename=imagename_c, niter=cyc['niter'],
                     uvrange=uvrange_c, scales=scales_c, **common_tclean)

        cal_c = str(intdir / f'{label}_sc{c_num}.gcal')
        print(f'[selfcal-dev]   Cycle {c_num}/{len(cycles)}: gaincal → {cal_c}')
        print(f'[selfcal-dev]     solmode={args.solmode!r}  refant={args.refant!r}  refantmode={args.refantmode!r}')
        gaincal(
            vis=scratch_ms,
            caltable=cal_c,
            gaintype='G',
            calmode='p',
            solint='inf',      # one phase solution per antenna for the 8s dump
            uvrange=uvrange_cal_c,
            minsnr=args.minsnr,
            solmode=args.solmode,
            rmsthresh=[],      # use CASA defaults for L1R iteration stopping
            refant=args.refant,
            refantmode=args.refantmode,
            append=False,
        )

        if not Path(cal_c).exists():
            fits_out = str(intdir / f'{label}_final.fits')
            print(f'[selfcal-dev]   WARNING: gaincal produced no solution table at cycle {c_num} '
                  f'(insufficient unflagged antennas / low SNR). '
                  f'Writing NaN placeholder FITS: {fits_out}')
            _write_nan_fits(
                fits_out,
                args.imsize,
                label,
                start_idx,
                jd_start=jd_start,
                jd_end=jd_end,
                jd_avg=jd_mid,
                jd_mid=jd_mid,
                jd_mean=jd_mean,
                nint=n,
                inttime_sec=inttime_sec,
                moon_ra_deg=ra_deg,
                moon_dec_deg=dec_deg,
            )
            return Path(fits_out)

        print(f'[selfcal-dev]   Cycle {c_num}/{len(cycles)}: applycal')
        applycal(vis=scratch_ms, gaintable=[cal_c], calwt=False, flagbackup=False,
                 applymode='calonly')  # don't flag data where solution was missing/flagged

        # Export FITS for this cycle so progress can be inspected in DS9
        fits_c = str(intdir / f'{label}_sc{c_num}.fits')
        exportfits(imagename=imagename_c + '.image', fitsimage=fits_c, overwrite=True)
        write_extra_header_to_fits_image(
            fits_c,
            jd_start=jd_start,
            jd_end=jd_end,
            jd_avg=jd_mid,
            jd_mid=jd_mid,
            jd_mean=jd_mean,
            nint=n,
            inttime_sec=inttime_sec,
            start_integration=start_idx,
            moon_ra_deg=ra_deg,
            moon_dec_deg=dec_deg,
            source_tag='moon_selfcal_dev_cycle',
        )
        print(f'[selfcal-dev]   Cycle {c_num}/{len(cycles)}: exported {fits_c}')

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
    write_extra_header_to_fits_image(
        fits_out,
        jd_start=jd_start,
        jd_end=jd_end,
        jd_avg=jd_mid,
        jd_mid=jd_mid,
        jd_mean=jd_mean,
        nint=n,
        inttime_sec=inttime_sec,
        start_integration=start_idx,
        moon_ra_deg=ra_deg,
        moon_dec_deg=dec_deg,
        source_tag='moon_selfcal_dev_final',
    )
    print(f'[selfcal-dev]   Exported: {fits_out}')

    # Save ephemeris record alongside image
    ephem_file = intdir / f'{label}_ephemeris.json'
    ephem_file.write_text(json.dumps({
        'start_integration': start_idx,
        'n': n,
        'jd_start': jd_start,
        'jd_mid': jd_mid,
        'jd_mean': jd_mean,
        'jd_end': jd_end,
        'inttime_sec': inttime_sec,
        'moon_ra_deg': ra_deg,
        'moon_dec_deg': dec_deg,
        'mask': mask_str,
        'fits_out': fits_out,
    }, indent=2))

    return Path(fits_out)


def _write_nan_fits(fits_out: str,
                    imsize: int,
                    label: str,
                    idx: int,
                    *,
                    jd_start: float,
                    jd_end: float,
                    jd_avg: float,
                    jd_mid: float,
                    jd_mean: float,
                    nint: int,
                    inttime_sec: float,
                    moon_ra_deg: float,
                    moon_dec_deg: float) -> None:
    """Write a FITS file filled with NaNs to mark a failed selfcal integration."""
    from astropy.io import fits as _fits
    nan_data = np.full((imsize, imsize), np.nan, dtype=np.float32)
    hdu = _fits.PrimaryHDU(data=nan_data)
    hdu.header['OBJECT']  = label
    hdu.header['COMMENT'] = f'NaN placeholder: selfcal failed at integration {idx} (gaincal no solution)'
    _fits.HDUList([hdu]).writeto(fits_out, overwrite=True)
    write_extra_header_to_fits_image(
        fits_out,
        jd_start=jd_start,
        jd_end=jd_end,
        jd_avg=jd_avg,
        jd_mid=jd_mid,
        jd_mean=jd_mean,
        nint=nint,
        inttime_sec=inttime_sec,
        start_integration=idx,
        moon_ra_deg=moon_ra_deg,
        moon_dec_deg=moon_dec_deg,
        source_tag='moon_selfcal_dev_nan',
    )


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
        if args.build_index_if_missing:
            print(f'[selfcal-dev] Index not found — building from UVFITS: {args.uvfits}')
            _src_dir = str(Path(__file__).resolve().parent.parent / 'src')
            if _src_dir not in sys.path:
                sys.path.insert(0, _src_dir)
            import modules.ugmrt_query as _q_build  # type: ignore
            _q_build.get_or_build_row_index(
                Path(args.uvfits),
                cache_path=index_path,
                force_rebuild=False,
                write_cache=True,
            )
            print(f'[selfcal-dev] Index built: {index_path}')
        else:
            sys.exit(f'ERROR: index cache not found: {index_path}\n'
                     f'Run visSplit first, pass --index explicitly, '
                     f'or add --build-index-if-missing.')
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

    print(f'[selfcal-dev] Processing integrations: {args.integrations} (stack_size={args.stack_size})')

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
        _tmp_dir = outdir / '_tmp_uvfits_patch'
        uvfits_for_casa = _patch_uvfits_veldef(args.uvfits, _tmp_dir)
        importuvfits(fitsfile=str(uvfits_for_casa), vis=str(full_ms))
        if _tmp_dir.exists():
            shutil.rmtree(_tmp_dir)
    else:
        print(f'[selfcal-dev] Reusing existing MS: {full_ms}')

    # ── Per-stack loop ────────────────────────────────────────────────────────
    results: list[dict] = []
    cycles = _parse_cycle_params(args)

    sorted_ints = sorted(args.integrations)
    # Partition into consecutive groups of stack_size.
    # If the final chunk would be smaller than stack_size, merge it into the
    # penultimate chunk rather than leaving a tiny orphan stack.
    stacks = [sorted_ints[i:i + args.stack_size]
              for i in range(0, len(sorted_ints), args.stack_size)]
    if len(stacks) >= 2 and len(stacks[-1]) < args.stack_size:
        stacks[-2] = stacks[-2] + stacks[-1]
        stacks = stacks[:-1]

    for stack in stacks:
        start_idx = stack[0]
        jd_stack = [float(scan_jds[i]) for i in stack]
        fits_path = _process_stack(
            start_idx=start_idx,
            jds=jd_stack,
            full_ms=full_ms,
            outdir=outdir,
            location=location,
            args=args,
            cycles=cycles,
            casa=casa,
        )
        if fits_path is None:
            results.append({'start_integration': start_idx, 'n': len(stack),
                            'jd_start': jd_stack[0], 'fits': None,
                            'skipped': 'moon_outside_image'})
        else:
            results.append({'start_integration': start_idx, 'n': len(stack),
                            'jd_start': jd_stack[0], 'fits': str(fits_path)})

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_file = outdir / 'summary.json'
    summary_file.write_text(json.dumps({'scan': scan_upper, 'results': results}, indent=2))
    print(f'\n[selfcal-dev] Done.  {len(results)} integrations processed.')
    print(f'[selfcal-dev] Summary: {summary_file}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
