#!/usr/bin/env python3
"""Generic GMRT per-stack imaging + phase-only selfcal development/tuning script.

Supports two target modes:
  fixed-target  : phasecenter and mask derived from the MS FIELD table (e.g.
                  calibrators such as 3C468.1, 3C48).  No ephemeris lookup; the
                  source sits at the phase centre for the entire scan.
  moving-target : phasecenter and mask re-derived from an ephemeris body at the
                  midpoint JD of each stack (e.g. the Moon, a planet).

Workflow per stack
------------------
  For each requested stack of integrations:

  1.  Split the time range from the full-scan MS into a scratch MS.
  2.  Determine sky position:
        fixed-target  → read RA/Dec from MS FIELD table (once)
        moving-target → query astropy ephemeris at stack midpoint JD
  3.  Build a CASA circular mask centred on the target.

  Selfcal cycle n (shallow→deep, controlled by --niter/uvmin/uvmax-per-cycle):
  4a. tclean(niter=NITER_n, uvrange=UVRANGE_n, mask=target_mask) → model
  4b. gaincal(calmode='p', solint='inf', uvrange=UVRANGE_CAL_n)
  4c. applycal()

  Final image:
  5.  tclean(niter=NITER_FINAL, uvrange=last-cycle-uvrange, mask=target_mask)
  6.  exportfits → <outdir>/<label>_final.fits

Usage (normal Python / casatasks venv)
--------------------------------------
  # Fixed-target (e.g. 3C468.1):
  python src/gmrt_selfcal_dev.py \\
      --target-mode fixed-target \\
      --scan 3C468.1 \\
      --integrations 0 12 \\
      --uvfits ~/DATA/.../3c468.1_calibrated.uvfits \\
      --outdir ~/DATA/.../casa_selfcal/3c468.1_scan01

  # Moving-target (e.g. Moon):
  python src/gmrt_selfcal_dev.py \\
      --target-mode moving-target \\
      --ephemeris-body moon \\
      --scan MOON0520 \\
      --integrations 0 5 10 \\
      --uvfits ~/DATA/.../moon0520_calibrated.uvfits \\
      --index  ~/DATA/.../40_014_25jul2021_gsb.index.npz \\
      --outdir ~/DATA/.../casa_selfcal/moon0520_dev

All imaging / selfcal parameters are exposed as CLI flags with sensible defaults.

DUPLICATION NOTE [DUPLICATION-001]
-----------------------------------
The core selfcal loop here is structurally identical to src/moon_selfcal_dev.py.
Once this generic engine survives first QC, moon_selfcal_dev.py should be
refactored into a thin moving-target wrapper that imports from this module.
See tools/dev/3c468.1_todo.txt §Code duplication tracking for the full plan.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from astropy.coordinates import Angle, EarthLocation, SkyCoord, get_body
from astropy.time import Time
from astropy import units as u

# fits_time_headers lives in experimental/ — locate it relative to this file.
_EXP_DIR = str(Path(__file__).resolve().parent.parent / 'experimental')
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)
from fits_time_headers import write_extra_header_to_fits_image  # type: ignore  # noqa: E402


def _selfcal_log(message: str) -> None:
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'[{ts}] [selfcal] {message}', flush=True)


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60.0:
        return f'{seconds:.2f}s'
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f'{minutes}m{rem:05.2f}s'


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


# ---------------------------------------------------------------------------
# GMRT array reference position (derived from antenna metadata)
# ---------------------------------------------------------------------------

def _gmrt_location(meta: dict) -> EarthLocation:
    """Return GMRT reference EarthLocation from ITRF antenna offsets in metadata."""
    ref_x = 1656342.30
    ref_y = 5797947.77
    ref_z = 2073243.16  # metres (C02 ITRF geocentric)
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
# Sky-position helpers
# ---------------------------------------------------------------------------

def _ephemeris_radec_at_jd(body: str, jd: float,
                            location: EarthLocation) -> tuple[float, float]:
    """Return apparent topocentric (ra_deg, dec_deg) for an ephemeris body.

    Uses astropy get_body which returns GCRS (apparent topocentric) frame for
    the observer location.  Do NOT transform to ICRS — for the Moon that gives
    the barycentric direction which differs by ~25 degrees.
    """
    t = Time(jd, format='jd', scale='utc')
    body_obj = get_body(body, t, location=location)
    return float(body_obj.ra.deg), float(body_obj.dec.deg)


def _ms_field_radec(ms_path: str) -> tuple[float, float]:
    """Return (ra_deg, dec_deg) of the FIELD phase direction from an MS.

    Queries FIELD_ID from the MAIN table to find the active row, then reads
    PHASE_DIR for that row.  PHASE_DIR is in radians; RA wrapped to [0, 360).
    """
    from casatools import table as tb_tool  # type: ignore
    tb = tb_tool()
    tb.open(ms_path)
    try:
        field_ids = tb.getcol('FIELD_ID')
    finally:
        tb.close()
    field_id = int(np.unique(field_ids)[0])
    tb.open(os.path.join(ms_path, 'FIELD'))
    try:
        phase_dir = tb.getcol('PHASE_DIR')
        ra_rad  = float(phase_dir[0, 0, field_id])
        dec_rad = float(phase_dir[1, 0, field_id])
    finally:
        tb.close()
    return float(np.degrees(ra_rad)) % 360.0, float(np.degrees(dec_rad))


def _angular_offset_arcmin(ra1: float, dec1: float,
                           ra2: float, dec2: float) -> float:
    """Great-circle separation in arcmin between two sky positions."""
    c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg, frame='icrs')
    c2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg, frame='icrs')
    return float(c1.separation(c2).arcmin)


# ---------------------------------------------------------------------------
# CASA mask / phasecenter string helpers
# ---------------------------------------------------------------------------

def _casa_circle_mask(ra_deg: float, dec_deg: float, radius_arcmin: float) -> str:
    """Return a CASA tclean-compatible circular sky-region mask string."""
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    ra_str  = c.ra.to_string(unit=u.hour, sep=':', precision=3, pad=True)
    dec_str = c.dec.to_string(unit=u.deg, sep='.', precision=2,
                              alwayssign=True, pad=True)
    return f"circle[[{ra_str}, {dec_str}], {radius_arcmin:.1f}arcmin]"


def _casa_phasecenter_j2000(ra_deg: float, dec_deg: float) -> str:
    """Return CASA tclean-compatible 'J2000 <RA> <Dec>' phasecenter string."""
    ra_str = Angle(ra_deg * u.deg).to_string(unit=u.hour, sep='hms',
                                             precision=3, pad=True)
    dec_str = Angle(dec_deg * u.deg).to_string(unit=u.deg, sep='dms',
                                               precision=2, alwayssign=True,
                                               pad=True)
    return f'J2000 {ra_str} {dec_str}'


# ---------------------------------------------------------------------------
# CASA import helpers
# ---------------------------------------------------------------------------

def _import_casa() -> Any:
    try:
        from casatasks import exportfits, gaincal, applycal, importuvfits, split, tclean, imstat  # type: ignore
    except ImportError as exc:
        sys.exit(
            f'casatasks not available: {exc}\n'
            'Activate the CASA-enabled venv (gmrt) before running this script.'
        )
    return importuvfits, split, tclean, gaincal, applycal, exportfits, imstat


# ---------------------------------------------------------------------------
# UVFITS pre-processing helpers
# ---------------------------------------------------------------------------

def _patch_uvfits_veldef(uvfits_path: Path, tmp_dir: Path) -> Path:
    """Return a UVFITS path safe to pass to CASA importuvfits.

    Patches applied if needed:
    1. VELDEF/SPECSYS — casacore requires VELDEF in the primary HDU.
    2. PMRA/PMDEC zeroing in the SU table — non-zero values cause casacore to
       use apparent-coordinate (MDirection::APP) mode which requires a full
       Measures frame.  Setting to zero forces J2000 with numPoly=0.

    If no patches are needed the original path is returned unchanged.
    """
    from astropy.io import fits  # type: ignore

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
            return uvfits_path

        msgs = []
        if need_veldef:
            msgs.append('VELDEF=RADIO, SPECSYS=TOPOCENT')
        if need_pmra_zero:
            msgs.append('PMRA/PMDEC=0 in SU table')
        print(f'[selfcal] Patching UVFITS for CASA ({", ".join(msgs)})')
        tmp_dir.mkdir(parents=True, exist_ok=True)
        patched = tmp_dir / (uvfits_path.stem + '_casa_patched.uvfits')

        if need_veldef:
            hdul[0].header['VELDEF'] = ('RADIO', 'Radio velocity convention (topocentric)')
            hdul[0].header['SPECSYS'] = ('TOPOCENT', 'Spectral reference frame')
        if need_pmra_zero:
            for ext_hdu in hdul:
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
        description='Generic GMRT per-stack imaging + phase-only selfcal (dev/tuning)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Target mode ──
    mode = p.add_argument_group('Target mode')
    mode.add_argument(
        '--target-mode',
        required=True,
        choices=['fixed-target', 'moving-target'],
        help=(
            'fixed-target  : mask + phasecenter from MS FIELD table RA/Dec '
            '(e.g. 3C468.1, 3C48).  No ephemeris lookup.\n'
            'moving-target : mask + phasecenter recomputed from an ephemeris body '
            'at the midpoint JD of each stack (e.g. Moon, planets).'
        ),
    )
    mode.add_argument(
        '--ephemeris-body',
        default='moon',
        help='Ephemeris body name passed to astropy get_body().  '
             'Only used when --target-mode moving-target.',
    )

    # ── I/O ──
    io = p.add_argument_group('I/O')
    io.add_argument('--scan', required=True,
                    help='Source/scan name as in the index file, e.g. 3C468.1 or MOON0520')
    io.add_argument('--uvfits', required=True,
                    help='Path to the calibrated split UVFITS for this scan')
    io.add_argument('--index', default=None,
                    help='Path to the row index cache NPZ for the UVFITS '
                         '(default: <uvfits>.row_index_cache.npz alongside the UVFITS)')
    io.add_argument('--build-index-if-missing', action='store_true',
                    help='Build the row index cache on-the-fly if missing.')
    io.add_argument('--outdir', required=True,
                    help='Output directory for MS, images, and FITS frames')
    io.add_argument('--integrations', nargs='+', type=int, default=[0],
                    help='0-based integration indices within the scan to process')
    io.add_argument('--stack-size', type=int, default=1,
                    help='Consecutive integrations to combine into one MS per stack. '
                         '1 = per-integration.  E.g. 10 = 10×8s = 80s per stack.')
    io.add_argument('--run-id', default='',
                    help='Optional provenance run identifier injected by launcher scripts')
    io.add_argument('--launcher-log-path', default='',
                    help='Optional launcher log file path for provenance linkage')
    io.add_argument('--launcher-cmd-path', default='',
                    help='Optional launcher command file path for provenance linkage')
    io.add_argument('--overwrite', action='store_true',
                    help='Delete existing outputs and re-run')

    # ── Image geometry ──
    img = p.add_argument_group('Image geometry')
    img.add_argument('--cell', default='1.5arcsec')
    img.add_argument('--imsize', type=int, default=2048)
    img.add_argument('--stokes', default='I')
    img.add_argument('--weighting', default='briggs',
                     choices=['natural', 'uniform', 'briggs'])
    img.add_argument('--robust', type=float, default=0.0)
    img.add_argument('--deconvolver', default='multiscale',
                     choices=['multiscale', 'hogbom', 'clark'])
    img.add_argument('--scales', default='0,5,15,45',
                     help='Multiscale clean scales in pixels (comma-separated, default cycle)')

    # ── Mask ──
    msk = p.add_argument_group('Mask (circular region centred on target)')
    msk.add_argument('--mask-radius-arcmin', type=float, default=20.0,
                     help='Radius of circular CLEAN mask around target centre. '
                          'For a compact calibrator, a few arcmin is usually sufficient. '
                          'For the Moon disk, default 20 arcmin adds margin around the disk.')
    msk.add_argument('--no-mask', action='store_true', default=False,
                     help='Run all tclean calls without a mask.')
    msk.add_argument('--use-tclean-phasecenter', action='store_true', default=False,
                     help='Pass phasecenter to tclean (recenters imaging on target '
                          'without altering visibilities).  Mainly useful for '
                          'moving-target mode.')

    # ── Per-cycle selfcal parameters ──
    cyc = p.add_argument_group(
        'Per-cycle selfcal parameters',
        'Comma-separated lists — one entry per selfcal cycle (all same length).',
    )
    cyc.add_argument('--niter-per-cycle', default='100,300')
    cyc.add_argument('--uvmin-per-cycle', default='0.5,0.12',
                     help='Min uv in kλ per cycle for tclean imaging.')
    cyc.add_argument('--uvmax-per-cycle', default=',',
                     help='Max uv in kλ per cycle (empty = no limit).')
    cyc.add_argument('--uvmin-cal-per-cycle', default=None,
                     help='Min uv in kλ per cycle for gaincal '
                          '(falls back to --uvmin-per-cycle if omitted).')
    cyc.add_argument('--uvmax-cal-per-cycle', default=None,
                     help='Max uv in kλ per cycle for gaincal '
                          '(falls back to --uvmax-per-cycle if omitted).')
    cyc.add_argument('--scales-per-cycle', default='0,5,15|0,5,15,45',
                     help='Multiscale scales per cycle; cycles separated by |, '
                          'scales within a cycle by comma.')

    # ── Final image ──
    fin = p.add_argument_group('Final image (after all selfcal cycles)')
    fin.add_argument('--niter-final', type=int, default=500)
    fin.add_argument('--threshold', default='0mJy')
    fin.add_argument('--cycleniter', type=int, default=100)
    fin.add_argument('--negativethreshold', type=float, default=0.0,
                     help='tclean negativethreshold (Jy/beam). 0 = disabled.')

    # ── Selfcal ──
    sc = p.add_argument_group('Selfcal')
    sc.add_argument('--minsnr', type=float, default=3.0)
    sc.add_argument('--loop-gain', type=float, default=0.05)
    sc.add_argument('--solmode', default='',
                    help='gaincal solver mode: "" (LS), "L1", "R", "L1R".')
    sc.add_argument('--refant', default='1',
                    help='gaincal reference antenna name (as in MS ANTENNA table).')
    sc.add_argument('--refantmode', default='flex',
                    choices=['flex', 'strict'])

    return p.parse_args()


# ---------------------------------------------------------------------------
# Parse per-cycle parameter arrays from CLI strings
# ---------------------------------------------------------------------------

def _parse_cycle_params(args: argparse.Namespace) -> list[dict]:
    """Convert per-cycle comma/pipe-separated CLI strings into a list of dicts."""
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
            f'Per-cycle parameter lists must all have the same length. Got: {lengths}\n'
            f'  --niter-per-cycle  = {args.niter_per_cycle!r}\n'
            f'  --uvmin-per-cycle  = {args.uvmin_per_cycle!r}\n'
            f'  --uvmax-per-cycle  = {args.uvmax_per_cycle!r}\n'
            f'  --scales-per-cycle = {args.scales_per_cycle!r}'
        )
    n_cycles = len(niters)

    if args.uvmin_cal_per_cycle is not None:
        cal_uvmins = [float(v.strip()) if v.strip() else None
                      for v in args.uvmin_cal_per_cycle.split(',')]
        if len(cal_uvmins) != n_cycles:
            raise ValueError(
                f'--uvmin-cal-per-cycle has {len(cal_uvmins)} entries but '
                f'{n_cycles} cycles defined.')
    else:
        cal_uvmins = uvmins

    if args.uvmax_cal_per_cycle is not None:
        cal_uvmaxs = [float(v.strip()) if v.strip() else None
                      for v in args.uvmax_cal_per_cycle.split(',')]
        if len(cal_uvmaxs) != n_cycles:
            raise ValueError(
                f'--uvmax-cal-per-cycle has {len(cal_uvmaxs)} entries but '
                f'{n_cycles} cycles defined.')
    else:
        cal_uvmaxs = uvmaxs

    cycles = []
    for i, (niter, uvmin, uvmax, cal_uvmin, cal_uvmax, sc_str) in enumerate(
            zip(niters, uvmins, uvmaxs, cal_uvmins, cal_uvmaxs, scales_blocks)):
        sc_vals = sorted({int(v.strip()) for v in sc_str.split(',') if v.strip()})
        if 0 not in sc_vals:
            sc_vals.insert(0, 0)
        cycles.append({'niter': niter, 'uvmin_kl': uvmin, 'uvmax_kl': uvmax,
                        'cal_uvmin_kl': cal_uvmin, 'cal_uvmax_kl': cal_uvmax,
                        'scales': sc_vals})
        print(f'[selfcal] Cycle {i+1} params: niter={niter}, '
              f'img uvmin={uvmin}kλ uvmax={uvmax}kλ, '
              f'cal uvmin={cal_uvmin}kλ uvmax={cal_uvmax}kλ, scales={sc_vals}')
    return cycles


# ---------------------------------------------------------------------------
# UV range string helpers
# ---------------------------------------------------------------------------

def _uvrange(lo_kl: Optional[float], hi_kl: Optional[float]) -> str:
    if lo_kl is None and hi_kl is None:
        return ''
    if hi_kl is None:
        return f'>{lo_kl:.3f}klambda'
    if lo_kl is None:
        return f'<{hi_kl:.3f}klambda'
    return f'{lo_kl:.3f}~{hi_kl:.3f}klambda'


# ---------------------------------------------------------------------------
# Image cleanup helper
# ---------------------------------------------------------------------------

def _as_json_scalar(value: Any) -> Any:
    """Convert CASA/numpy values into JSON-serializable scalars/lists."""
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _as_json_scalar(value.reshape(-1)[0])
        return [_as_json_scalar(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_as_json_scalar(v) for v in value]
    return value


def _to_float(value: Any) -> float:
    """Best-effort scalar float conversion with NaN on failure."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return float('nan')
        value = value.reshape(-1)[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _compute_cycle_counters(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return robust cycle counters with explicit fallback flags."""
    requested_niter = _to_float(metrics.get('requested_niter'))
    requested_cycleniter = _to_float(metrics.get('requested_cycleniter'))
    iterdone = _to_float(metrics.get('iterdone'))
    nmajordone = _to_float(metrics.get('nmajordone'))

    iterdone_is_estimated = False
    nmajor_is_estimated = False

    if not math.isfinite(iterdone):
        if math.isfinite(requested_niter):
            iterdone = requested_niter
            iterdone_is_estimated = True

    if not math.isfinite(nmajordone):
        if math.isfinite(iterdone) and iterdone >= 0 and math.isfinite(requested_cycleniter) and requested_cycleniter > 0:
            nmajordone = float(max(1, int(math.ceil(iterdone / requested_cycleniter))))
            nmajor_is_estimated = True

    out: dict[str, Any] = {
        'nminor_cycles_total': iterdone if math.isfinite(iterdone) else None,
        'nmajor_cycles_used': nmajordone if math.isfinite(nmajordone) else None,
        'nminor_cycles_total_is_estimated': bool(iterdone_is_estimated),
        'nmajor_cycles_used_is_estimated': bool(nmajor_is_estimated),
    }
    return out


def _collect_image_metrics(imstat, *, imagename: str, cycle_metrics: dict[str, Any]) -> dict[str, Any]:
    """Measure image/model/residual diagnostics for one CLEAN stage."""
    image_path = imagename + '.image'
    residual_path = imagename + '.residual'
    model_path = imagename + '.model'

    out: dict[str, Any] = {
        'image_path': image_path,
        'residual_path': residual_path,
        'model_path': model_path,
    }

    if not (Path(image_path).exists() and Path(residual_path).exists() and Path(model_path).exists()):
        out['image_metrics_available'] = False
        return out

    try:
        stats_img = imstat(imagename=image_path)
        stats_res = imstat(imagename=residual_path)
        stats_mod = imstat(imagename=model_path)
    except Exception as exc:
        out['image_metrics_available'] = False
        out['image_metrics_error'] = str(exc)
        return out

    peak = _to_float(stats_img.get('max', [float('nan')]))
    res_max = _to_float(stats_res.get('max', [float('nan')]))
    res_min = _to_float(stats_res.get('min', [float('nan')]))
    model_sum = _to_float(stats_mod.get('sum', [float('nan')]))
    residual_peak = max(abs(res_max), abs(res_min)) if math.isfinite(res_max) and math.isfinite(res_min) else float('nan')
    dr = peak / residual_peak if math.isfinite(peak) and math.isfinite(residual_peak) and residual_peak > 0 else float('nan')

    out.update({
        'image_metrics_available': True,
        'peak_jy_per_beam': peak if math.isfinite(peak) else None,
        'residual_peak_jy_per_beam': residual_peak if math.isfinite(residual_peak) else None,
        'model_sum_jy': model_sum if math.isfinite(model_sum) else None,
        'dr_peak_over_residual': dr if math.isfinite(dr) else None,
    })
    out.update(_compute_cycle_counters(cycle_metrics))
    return out


def _clean_fresh(tclean, *, imagename: str, niter: int, uvrange: str, **kwargs) -> dict:
    """Remove existing image products, run tclean, and return cycle counters."""
    for suffix in ['.image', '.model', '.psf', '.residual', '.pb', '.sumwt', '.mask']:
        p = Path(imagename + suffix)
        if p.exists():
            shutil.rmtree(str(p)) if p.is_dir() else p.unlink()
    out = tclean(imagename=imagename, niter=niter, uvrange=uvrange,
                 calcpsf=True, calcres=True, restart=False, **kwargs)

    counters: dict[str, Any] = {
        'requested_niter': int(niter),
        'requested_uvrange': uvrange,
    }
    cycleniter = kwargs.get('cycleniter', None)
    if cycleniter is not None:
        counters['requested_cycleniter'] = int(cycleniter)

    if isinstance(out, dict):
        for key in ['iterdone', 'nmajordone', 'stopcode', 'stopreason',
                    'cyclethreshold', 'threshold', 'nsigma', 'ncycle']:
            if key in out:
                counters[key] = _as_json_scalar(out[key])
    return counters


# ---------------------------------------------------------------------------
# NaN placeholder FITS writer
# ---------------------------------------------------------------------------

def _write_nan_fits(
    fits_out: str,
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
    source_ra_deg: float,
    source_dec_deg: float,
    target_mode: str,
    ephemeris_body: str,
) -> None:
    """Write a NaN-filled FITS placeholder for a failed selfcal stack."""
    from astropy.io import fits as _fits
    nan_data = np.full((imsize, imsize), np.nan, dtype=np.float32)
    hdu = _fits.PrimaryHDU(data=nan_data)
    hdu.header['OBJECT']  = label
    hdu.header['COMMENT'] = (
        f'NaN placeholder: selfcal failed at integration {idx} (gaincal no solution)'
    )
    _fits.HDUList([hdu]).writeto(fits_out, overwrite=True)

    # For moving-target pass moon_ra_deg/moon_dec_deg for downstream tool compat.
    extra_kwargs: dict = {}
    if target_mode == 'moving-target' and ephemeris_body.lower() == 'moon':
        extra_kwargs['moon_ra_deg']  = source_ra_deg
        extra_kwargs['moon_dec_deg'] = source_dec_deg

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
        source_tag='gmrt_selfcal_dev_nan',
        **extra_kwargs,
    )


# ---------------------------------------------------------------------------
# Per-stack selfcal loop
# ---------------------------------------------------------------------------

def _process_stack(
    *,
    start_idx: int,
    jds: list[float],
    full_ms: Path,
    outdir: Path,
    location: Optional[EarthLocation],
    args: argparse.Namespace,
    cycles: list[dict],
    casa: tuple,
    # fixed-target: pre-computed field RA/Dec (None for moving-target)
    field_ra_deg: Optional[float],
    field_dec_deg: Optional[float],
) -> Optional[Path]:
    _, split_task, tclean, gaincal, applycal, exportfits, imstat = casa
    stack_t0 = time.perf_counter()
    stack_start_utc = _utc_now_iso()

    n = len(jds)
    scan_slug = args.scan.lower().replace('.', 'p')
    if n == 1:
        label = f'{scan_slug}_int{start_idx:04d}'
    else:
        label = f'{scan_slug}_stk{start_idx:04d}_n{n:02d}'
    intdir = outdir / label
    debug_json_path = intdir / f'{label}_clean_debug.json'
    if args.overwrite and intdir.exists():
        shutil.rmtree(intdir)
    intdir.mkdir(parents=True, exist_ok=True)

    # ── 1. Split stack time range ─────────────────────────────────────────────
    scratch_ms = str(intdir / f'{label}.ms')
    if Path(scratch_ms).exists():
        shutil.rmtree(scratch_ms)

    dt_margin = 4.0  # seconds
    t_first = Time(jds[0],  format='jd', scale='utc')
    t_last  = Time(jds[-1], format='jd', scale='utc')
    t0 = (t_first - dt_margin * u.s).strftime('%Y/%m/%d/%H:%M:%S')
    t1 = (t_last  + dt_margin * u.s).strftime('%Y/%m/%d/%H:%M:%S')
    timerange = f'{t0}~{t1}'

    _selfcal_log(f'=== Stack start_idx={start_idx}  n={n}  '
                f'JD={jds[0]:.6f}..{jds[-1]:.6f}  timerange={timerange} ===')
    split_task(vis=str(full_ms), outputvis=scratch_ms, timerange=timerange, datacolumn='data')

    inttime_sec = (float(np.median(np.diff(np.asarray(jds, dtype=np.float64))) * 86400.0)
                   if n > 1 else 8.0)
    jd_start = float(jds[0])
    jd_mid   = float(jds[len(jds) // 2])
    jd_mean  = float(np.mean(np.asarray(jds, dtype=np.float64)))
    jd_end   = float(jd_start + (n * inttime_sec) / 86400.0)

    # ── 2. Determine sky position ─────────────────────────────────────────────
    if args.target_mode == 'fixed-target':
        # For fixed targets the phase centre does not move; use the pre-computed
        # FIELD table values passed in by the caller.
        ra_deg  = field_ra_deg
        dec_deg = field_dec_deg
        target_label = args.scan
    else:
        # moving-target: query ephemeris at stack midpoint
        ra_deg, dec_deg = _ephemeris_radec_at_jd(args.ephemeris_body, jd_mid, location)
        target_label = args.ephemeris_body.capitalize()

    phasecenter_str = _casa_phasecenter_j2000(ra_deg, dec_deg)

    if args.no_mask:
        mask_str = ''
    else:
        mask_str = _casa_circle_mask(ra_deg, dec_deg, args.mask_radius_arcmin)

    # Sanity check: target must lie within the image
    ms_field_ra, ms_field_dec = _ms_field_radec(scratch_ms)
    offset_arcmin   = _angular_offset_arcmin(ra_deg, dec_deg, ms_field_ra, ms_field_dec)
    cell_arcsec     = float(args.cell.replace('arcsec', ''))
    halfwidth_arcmin = (args.imsize * cell_arcsec / 2.0) / 60.0
    margin_arcmin   = halfwidth_arcmin - (0.0 if args.no_mask else args.mask_radius_arcmin)

    _selfcal_log(f'  {target_label} at RA={ra_deg:.4f}°  Dec={dec_deg:.4f}°  '
                f'offset={offset_arcmin:.2f}arcmin from field centre  '
                f'(image half-width={halfwidth_arcmin:.1f}arcmin)')

    if offset_arcmin >= halfwidth_arcmin:
        _selfcal_log(f'  WARNING: {target_label} centre ({offset_arcmin:.2f} arcmin) '
                f'is outside image half-width ({halfwidth_arcmin:.1f} arcmin). '
                f'Skipping stack starting at integration {start_idx}.')
        shutil.rmtree(str(intdir), ignore_errors=True)
        return None

    if not args.no_mask and offset_arcmin > margin_arcmin:
        _selfcal_log(f'  NOTE: mask circle overhangs image edge '
                    f'({offset_arcmin:.2f} + {args.mask_radius_arcmin:.1f} arcmin) '
                    f'-- CASA will clip it, proceeding.')

    _selfcal_log(f'  target-mode: {args.target_mode}')
    if args.no_mask:
        _selfcal_log('  mask: <disabled by --no-mask>')
    else:
        _selfcal_log(f'  mask-radius: {args.mask_radius_arcmin:.2f} arcmin')
        _selfcal_log(f'  mask-string: {mask_str}')
    if args.use_tclean_phasecenter:
        _selfcal_log(f'  tclean phasecenter: ENABLED ({phasecenter_str})')
    else:
        _selfcal_log('  tclean phasecenter: DISABLED (using MS native phase centre)')

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

    # Extra FITS header kwargs: moving-target (Moon) passes moon_ra/dec for
    # backward compatibility with downstream tools expecting those fields.
    extra_hdr: dict = {}
    if args.target_mode == 'moving-target' and args.ephemeris_body.lower() == 'moon':
        extra_hdr['moon_ra_deg']  = ra_deg
        extra_hdr['moon_dec_deg'] = dec_deg

    # ── 3. Selfcal loop ───────────────────────────────────────────────────────
    clean_cycle_metrics: list[dict[str, Any]] = []
    for c_idx, cyc in enumerate(cycles):
        c_num = c_idx + 1
        cycle_t0 = time.perf_counter()
        cycle_start_utc = _utc_now_iso()
        uvrange_c     = _uvrange(cyc['uvmin_kl'],     cyc['uvmax_kl'])
        uvrange_cal_c = _uvrange(cyc['cal_uvmin_kl'], cyc['cal_uvmax_kl'])
        scales_c = cyc['scales'] if args.deconvolver == 'multiscale' else []

        _selfcal_log(f'  Cycle {c_num}/{len(cycles)}: '
                f'tclean niter={cyc["niter"]}  img_uvrange={uvrange_c!r}  '
                f'cal_uvrange={uvrange_cal_c!r}  scales={scales_c}')
        imagename_c = imname_base + f'_sc{c_num}'
        tclean_t0 = time.perf_counter()
        clean_metrics_c = _clean_fresh(
            tclean,
            imagename=imagename_c,
            niter=cyc['niter'],
            uvrange=uvrange_c,
            scales=scales_c,
            **common_tclean,
        )
        tclean_elapsed_s = time.perf_counter() - tclean_t0
        clean_metrics_c.update({
            'stage': f'sc{c_num}',
            'cycle_index': c_num,
            'deconvolver': args.deconvolver,
            'requested_scales': list(scales_c),
            'img_uvrange': uvrange_c,
            'cal_uvrange': uvrange_cal_c,
            'cycle_start_utc': cycle_start_utc,
            'tclean_elapsed_s': tclean_elapsed_s,
        })
        clean_metrics_c.update(_collect_image_metrics(imstat, imagename=imagename_c,
                                                      cycle_metrics=clean_metrics_c))
        clean_cycle_metrics.append(clean_metrics_c)

        cal_c = str(intdir / f'{label}_sc{c_num}.gcal')
        _selfcal_log(f'  Cycle {c_num}/{len(cycles)}: gaincal → {cal_c}')
        gaincal_t0 = time.perf_counter()
        gaincal(
            vis=scratch_ms,
            caltable=cal_c,
            gaintype='G',
            calmode='p',
            solint='inf',
            uvrange=uvrange_cal_c,
            minsnr=args.minsnr,
            solmode=args.solmode,
            rmsthresh=[],
            refant=args.refant,
            refantmode=args.refantmode,
            append=False,
        )
        gaincal_elapsed_s = time.perf_counter() - gaincal_t0
        clean_metrics_c['gaincal_elapsed_s'] = gaincal_elapsed_s

        if not Path(cal_c).exists():
            fits_out = str(intdir / f'{label}_final.fits')
            cycle_elapsed_s = time.perf_counter() - cycle_t0
            cycle_end_utc = _utc_now_iso()
            stack_elapsed_s = time.perf_counter() - stack_t0
            failure_utc = cycle_end_utc
            clean_metrics_c.update({
                'applycal_elapsed_s': float('nan'),
                'export_elapsed_s': float('nan'),
                'cycle_elapsed_s': cycle_elapsed_s,
                'cycle_end_utc': cycle_end_utc,
            })
            _selfcal_log(f'  WARNING: gaincal produced no solution at cycle {c_num}. '
                    f'Writing NaN placeholder: {fits_out}')
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
                source_ra_deg=ra_deg,
                source_dec_deg=dec_deg,
                target_mode=args.target_mode,
                ephemeris_body=args.ephemeris_body,
            )

            fail_record: dict[str, Any] = {
                'run_id': args.run_id,
                'launcher_log_path': args.launcher_log_path,
                'launcher_cmd_path': args.launcher_cmd_path,
                'stack_start_utc': stack_start_utc,
                'stack_end_utc': failure_utc,
                'stack_elapsed_s': stack_elapsed_s,
                'failure_utc': failure_utc,
                'elapsed_until_failure_s': stack_elapsed_s,
                'target_mode': args.target_mode,
                'start_integration': start_idx,
                'n': n,
                'jd_start': jd_start,
                'jd_mid': jd_mid,
                'jd_mean': jd_mean,
                'jd_end': jd_end,
                'inttime_sec': inttime_sec,
                'source_ra_deg': ra_deg,
                'source_dec_deg': dec_deg,
                'mask': mask_str,
                'fits_out': fits_out,
                'selfcal_failed': True,
                'failure_stage': f'sc{c_num}',
                'clean_cycle_metrics': clean_cycle_metrics,
            }
            if args.target_mode == 'moving-target':
                fail_record['ephemeris_body'] = args.ephemeris_body
            (intdir / f'{label}_metadata.json').write_text(json.dumps(fail_record, indent=2))

            debug_artifact = {
                'artifact_type': 'selfcal_clean_debug',
                'version': 2,
                'run_id': args.run_id,
                'launcher_log_path': args.launcher_log_path,
                'launcher_cmd_path': args.launcher_cmd_path,
                'stack_start_utc': stack_start_utc,
                'stack_end_utc': failure_utc,
                'stack_elapsed_s': stack_elapsed_s,
                'failure_utc': failure_utc,
                'elapsed_until_failure_s': stack_elapsed_s,
                'target_mode': args.target_mode,
                'scan': args.scan,
                'label': label,
                'start_integration': start_idx,
                'n_integrations': n,
                'mask_enabled': not args.no_mask,
                'mask': mask_str,
                'deconvolver': args.deconvolver,
                'weighting': args.weighting,
                'robust': args.robust,
                'threshold': args.threshold,
                'cycleniter': args.cycleniter,
                'niter_final': args.niter_final,
                'selfcal_failed': True,
                'failure_stage': f'sc{c_num}',
                'cycles': clean_cycle_metrics,
            }
            debug_json_path.write_text(json.dumps(debug_artifact, indent=2))
            return Path(fits_out)

        _selfcal_log(f'  Cycle {c_num}/{len(cycles)}: applycal')
        applycal_t0 = time.perf_counter()
        applycal(vis=scratch_ms, gaintable=[cal_c], calwt=False, flagbackup=False,
                 applymode='calonly')
        applycal_elapsed_s = time.perf_counter() - applycal_t0

        fits_c = str(intdir / f'{label}_sc{c_num}.fits')
        export_t0 = time.perf_counter()
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
            source_tag='gmrt_selfcal_dev_cycle',
            **extra_hdr,
        )
        export_elapsed_s = time.perf_counter() - export_t0
        cycle_elapsed_s = time.perf_counter() - cycle_t0
        cycle_end_utc = _utc_now_iso()
        clean_metrics_c.update({
            'applycal_elapsed_s': applycal_elapsed_s,
            'export_elapsed_s': export_elapsed_s,
            'cycle_elapsed_s': cycle_elapsed_s,
            'cycle_end_utc': cycle_end_utc,
        })
        _selfcal_log(f'  Cycle {c_num}/{len(cycles)}: exported {fits_c}')
        _selfcal_log(f'  Cycle {c_num}/{len(cycles)}: elapsed {_fmt_elapsed(cycle_elapsed_s)}')

    # ── 4. Final image ────────────────────────────────────────────────────────
    last = cycles[-1]
    uvrange_final = _uvrange(last['uvmin_kl'], last['uvmax_kl'])
    scales_final  = last['scales'] if args.deconvolver == 'multiscale' else []
    final_t0 = time.perf_counter()
    final_start_utc = _utc_now_iso()
    _selfcal_log(f'  Final image: niter={args.niter_final}  uvrange={uvrange_final!r}')
    imagename_final = imname_base + '_final'
    final_tclean_t0 = time.perf_counter()
    clean_metrics_final = _clean_fresh(
        tclean,
        imagename=imagename_final,
        niter=args.niter_final,
        uvrange=uvrange_final,
        scales=scales_final,
        **common_tclean,
    )
    final_tclean_elapsed_s = time.perf_counter() - final_tclean_t0
    clean_metrics_final.update({
        'stage': 'final',
        'cycle_index': len(cycles) + 1,
        'deconvolver': args.deconvolver,
        'requested_scales': list(scales_final),
        'img_uvrange': uvrange_final,
        'cal_uvrange': _uvrange(last['cal_uvmin_kl'], last['cal_uvmax_kl']),
        'cycle_start_utc': final_start_utc,
        'tclean_elapsed_s': final_tclean_elapsed_s,
        'gaincal_elapsed_s': float('nan'),
        'applycal_elapsed_s': float('nan'),
    })
    clean_metrics_final.update(_collect_image_metrics(imstat, imagename=imagename_final,
                                                      cycle_metrics=clean_metrics_final))
    clean_cycle_metrics.append(clean_metrics_final)

    fits_out = str(intdir / f'{label}_final.fits')
    final_export_t0 = time.perf_counter()
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
        source_tag='gmrt_selfcal_dev_final',
        **extra_hdr,
    )
    final_export_elapsed_s = time.perf_counter() - final_export_t0
    final_elapsed_s = time.perf_counter() - final_t0
    final_end_utc = _utc_now_iso()
    clean_metrics_final.update({
        'export_elapsed_s': final_export_elapsed_s,
        'cycle_elapsed_s': final_elapsed_s,
        'cycle_end_utc': final_end_utc,
    })
    _selfcal_log(f'  Exported: {fits_out}')
    _selfcal_log(f'  Final image: elapsed {_fmt_elapsed(final_elapsed_s)}')

    stack_elapsed_s = time.perf_counter() - stack_t0
    stack_end_utc = _utc_now_iso()

    # Save a metadata record alongside the image
    record: dict = {
        'run_id': args.run_id,
        'launcher_log_path': args.launcher_log_path,
        'launcher_cmd_path': args.launcher_cmd_path,
        'stack_start_utc': stack_start_utc,
        'stack_end_utc': stack_end_utc,
        'stack_elapsed_s': stack_elapsed_s,
        'target_mode': args.target_mode,
        'start_integration': start_idx,
        'n': n,
        'jd_start': jd_start,
        'jd_mid': jd_mid,
        'jd_mean': jd_mean,
        'jd_end': jd_end,
        'inttime_sec': inttime_sec,
        'source_ra_deg': ra_deg,
        'source_dec_deg': dec_deg,
        'mask': mask_str,
        'fits_out': fits_out,
        'clean_cycle_metrics': clean_cycle_metrics,
    }
    if args.target_mode == 'moving-target':
        record['ephemeris_body'] = args.ephemeris_body
    (intdir / f'{label}_metadata.json').write_text(json.dumps(record, indent=2))

    debug_artifact = {
        'artifact_type': 'selfcal_clean_debug',
        'version': 2,
        'run_id': args.run_id,
        'launcher_log_path': args.launcher_log_path,
        'launcher_cmd_path': args.launcher_cmd_path,
        'stack_start_utc': stack_start_utc,
        'stack_end_utc': stack_end_utc,
        'stack_elapsed_s': stack_elapsed_s,
        'target_mode': args.target_mode,
        'scan': args.scan,
        'label': label,
        'start_integration': start_idx,
        'n_integrations': n,
        'mask_enabled': not args.no_mask,
        'mask': mask_str,
        'deconvolver': args.deconvolver,
        'weighting': args.weighting,
        'robust': args.robust,
        'threshold': args.threshold,
        'cycleniter': args.cycleniter,
        'niter_final': args.niter_final,
        'selfcal_failed': False,
        'cycles': clean_cycle_metrics,
    }
    debug_json_path.write_text(json.dumps(debug_artifact, indent=2))

    _selfcal_log(f'=== Stack done start_idx={start_idx}  elapsed {_fmt_elapsed(stack_elapsed_s)} ===')

    return Path(fits_out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load row index ────────────────────────────────────────────────────────
    index_path = (Path(args.index) if args.index
                  else Path(args.uvfits).with_suffix(
                      Path(args.uvfits).suffix + '.row_index_cache.npz'))
    if not index_path.exists():
        if args.build_index_if_missing:
            _selfcal_log(f'Index not found — building from UVFITS: {args.uvfits}')
            _src_dir = str(Path(__file__).resolve().parent)
            if _src_dir not in sys.path:
                sys.path.insert(0, _src_dir)
            import modules.ugmrt_query as _q_build  # type: ignore
            _q_build.get_or_build_row_index(
                Path(args.uvfits),
                cache_path=index_path,
                force_rebuild=False,
                write_cache=True,
            )
            _selfcal_log(f'Index built: {index_path}')
        else:
            sys.exit(
                f'ERROR: index cache not found: {index_path}\n'
                f'Run visSplit first, pass --index explicitly, '
                f'or add --build-index-if-missing.'
            )

    _selfcal_log(f'Using index: {index_path}')
    idx_npz = np.load(index_path, allow_pickle=True)
    meta = json.loads(str(idx_npz['metadata_json']))

    # Map scan name → source id
    name_to_id = {v: int(k) for k, v in meta['id_to_name'].items()}
    scan_upper = args.scan.upper()
    if scan_upper not in name_to_id:
        sys.exit(f'ERROR: scan {args.scan!r} not in index. '
                 f'Available: {sorted(name_to_id)}')
    src_id = name_to_id[scan_upper]

    all_jd  = idx_npz['jd']
    all_sid = idx_npz['source_id']
    scan_jds = np.unique(all_jd[all_sid == src_id])
    scan_jds.sort()

    _selfcal_log(f'Scan {scan_upper} (id={src_id}): {len(scan_jds)} integrations')

    bad = [i for i in args.integrations if i < 0 or i >= len(scan_jds)]
    if bad:
        sys.exit(f'ERROR: integration indices out of range [0, {len(scan_jds)-1}]: {bad}')

    _selfcal_log(f'Target mode: {args.target_mode}')
    _selfcal_log(f'Processing integrations: {args.integrations} '
                f'(stack_size={args.stack_size})')

    # ── GMRT location (needed for moving-target ephemeris) ────────────────────
    location: Optional[EarthLocation] = None
    if args.target_mode == 'moving-target':
        location = _gmrt_location(meta)

    # ── For fixed-target: read field RA/Dec once from UVFITS import ───────────
    # We'll read it from the first scratch MS produced inside _process_stack;
    # store as None initially and let _process_stack populate it on first call
    # via _ms_field_radec.  Since the phase centre is constant across stacks we
    # can read it once and pass it to every stack call.
    field_ra_deg:  Optional[float] = None
    field_dec_deg: Optional[float] = None

    # ── Import UVFITS → full-scan MS (once) ───────────────────────────────────
    casa = _import_casa()
    importuvfits = casa[0]

    full_ms = outdir / f'{scan_upper.lower()}_full.ms'
    if full_ms.exists() and args.overwrite:
        shutil.rmtree(full_ms)
    if not full_ms.exists():
        _selfcal_log(f'Importing UVFITS → {full_ms}')
        _tmp_dir = outdir / '_tmp_uvfits_patch'
        uvfits_for_casa = _patch_uvfits_veldef(Path(args.uvfits), _tmp_dir)
        importuvfits(fitsfile=str(uvfits_for_casa), vis=str(full_ms))
        if _tmp_dir.exists():
            shutil.rmtree(_tmp_dir)
    else:
        _selfcal_log(f'Reusing existing MS: {full_ms}')

    # Pre-read field RA/Dec for fixed-target mode (avoids re-opening the full MS
    # per stack; the full MS has the same FIELD table as any split MS).
    if args.target_mode == 'fixed-target':
        field_ra_deg, field_dec_deg = _ms_field_radec(str(full_ms))
        _selfcal_log(f'Fixed-target phase centre from FIELD table: '
                f'RA={field_ra_deg:.6f}°  Dec={field_dec_deg:.6f}°')

    # ── Per-stack loop ────────────────────────────────────────────────────────
    results: list[dict] = []
    cycles = _parse_cycle_params(args)

    sorted_ints = sorted(args.integrations)
    stacks = [sorted_ints[i:i + args.stack_size]
              for i in range(0, len(sorted_ints), args.stack_size)]
    if len(stacks) >= 2 and len(stacks[-1]) < args.stack_size:
        stacks[-2] = stacks[-2] + stacks[-1]
        stacks = stacks[:-1]

    for stack in stacks:
        start_idx = stack[0]
        jd_stack  = [float(scan_jds[i]) for i in stack]
        fits_path = _process_stack(
            start_idx=start_idx,
            jds=jd_stack,
            full_ms=full_ms,
            outdir=outdir,
            location=location,
            args=args,
            cycles=cycles,
            casa=casa,
            field_ra_deg=field_ra_deg,
            field_dec_deg=field_dec_deg,
        )
        if fits_path is None:
            results.append({
                'start_integration': start_idx,
                'n': len(stack),
                'jd_start': jd_stack[0],
                'fits': None,
                'skipped': 'target_outside_image',
            })
        else:
            results.append({
                'start_integration': start_idx,
                'n': len(stack),
                'jd_start': jd_stack[0],
                'fits': str(fits_path),
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_file = outdir / 'summary.json'
    summary_file.write_text(json.dumps({
        'scan': scan_upper,
        'target_mode': args.target_mode,
        'results': results,
    }, indent=2))
    _selfcal_log(f'Done.  {len(results)} stacks processed.')
    _selfcal_log(f'Summary: {summary_file}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
