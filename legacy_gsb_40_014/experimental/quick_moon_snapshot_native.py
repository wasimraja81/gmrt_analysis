#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits
from casatasks import exportfits, importuvfits, tclean
from casatools import table
from typing import cast, Any

DEFAULT_UVFITS = Path(
    '/Users/raj030/DATA/gmrt_40_014/work/split/moon/moon0545_calibrated.uvfits'
)
OUTDIR = Path('casa_out/moon_snapshot_test_ms')


def cleanup_prefix(prefix: Path) -> None:
    for p in prefix.parent.glob(prefix.name + '.*'):
        if p.is_dir():
            import shutil

            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def robust_sigma(values: np.ndarray) -> float:
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if mad > 0:
        return float(1.4826 * mad)
    return float(np.nanstd(values))


def normalize_uvfits_for_casa(fits_path: Path, outdir: Path, label: str) -> Path:
    """Normalize GMRT UVFITS for CASA: add RADESYS/frame keywords, re-write via pyuvdata."""
    from pyuvdata import UVData

    readfix = outdir / f'{label}__casa_readfix.uvfits'
    normalized = outdir / f'{label}__casa_norm.uvfits'
    for p in (readfix, normalized):
        if p.exists():
            p.unlink()

    shutil.copy2(fits_path, readfix)
    with fits.open(readfix, mode='update') as hdul:
        primary = cast(Any, hdul[0])
        primary.header['RADESYS'] = 'fk5'
        if 'AIPS AN' in hdul:
            an = cast(Any, hdul['AIPS AN']).header
            an['FRAME'] = 'ITRF'
            an['XYZHAND'] = 'RIGHT'
        hdul.flush(output_verify='ignore')

    uv = UVData()
    uv.read_uvfits(str(readfix))
    setattr(uv, 'timesys', 'UTC')
    uv.write_uvfits(str(normalized), force_phase=True)
    readfix.unlink(missing_ok=True)
    return normalized


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('uvfits', nargs='?', type=Path, default=DEFAULT_UVFITS,
                    help='UVFITS file to image (default: moon0545)')
    args = ap.parse_args()

    uvfits: Path = args.uvfits
    if not uvfits.exists():
        raise FileNotFoundError(f'UVFITS not found: {uvfits}')

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Import UVFITS → temporary MS (with normalization fallback)
    scan_name = uvfits.stem.replace('_calibrated', '')
    ms_path = OUTDIR / f'{scan_name}.ms'
    if ms_path.exists():
        shutil.rmtree(ms_path)

    print(f'Importing {uvfits.name} → {ms_path.name} ...')
    try:
        importuvfits(fitsfile=str(uvfits), vis=str(ms_path))
    except Exception as exc:
        print(f'Direct import failed ({exc}), normalizing via pyuvdata ...')
        norm_fits = normalize_uvfits_for_casa(uvfits, OUTDIR, scan_name)
        importuvfits(fitsfile=str(norm_fits), vis=str(ms_path))

    MS = ms_path

    tb = table()
    tb.open(str(MS))
    times = np.unique(np.round(np.asarray(tb.getcol('TIME'), dtype=float), 6))
    tb.close()

    if len(times) < 2:
        raise RuntimeError('Not enough unique integrations in MS')

    cadence = float(np.median(np.diff(times)))

    base = OUTDIR / f'{scan_name}_snapshot_native'
    cleanup_prefix(base)

    tclean(
        vis=str(MS),
        imagename=str(base),
        phasecenter='MOON',
        spw='',
        imsize=[2048, 2048],
        cell='2arcsec',
        specmode='mfs',
        deconvolver='multiscale',
        scales=[0, 5, 15, 30],     # pixels: point + compact + Moon-disk
        niter=800,
        threshold='5mJy',
        weighting='briggs',
        robust=0.5,
        stokes='I',
        uvrange='',
        gridder='wproject',
        wprojplanes=-1,
        pblimit=-1,
        pbcor=False,
        interactive=False,
        calcpsf=True,
        calcres=True,
    )

    fits_path = str(base) + '.image.fits'
    exportfits(imagename=str(base) + '.image', fitsimage=fits_path, overwrite=True)

    with fits.open(fits_path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        if data.ndim == 4:
            data = data[0, 0]
        ny, nx = data.shape
        cx = (nx - 1) / 2.0
        cy = (ny - 1) / 2.0
        pixscale_arcsec = abs(float(hdul[0].header['CDELT1'])) * 3600.0

    yy, xx = np.ogrid[:ny, :nx]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    bg = data[(rr >= 200) & (rr <= 450)]
    bg_med = float(np.nanmedian(bg))
    sigma = robust_sigma(bg)

    peak_pos = np.unravel_index(np.nanargmax(data), data.shape)
    peak_y, peak_x = int(peak_pos[0]), int(peak_pos[1])
    peak_val = float(data[peak_y, peak_x])
    peak_snr = (peak_val - bg_med) / (sigma + 1e-12)

    offset_pix = float(np.hypot(peak_x - cx, peak_y - cy))
    offset_arcsec = offset_pix * pixscale_arcsec

    print(f'integrations_used={len(times)} cadence_sec={cadence:.3f}')
    print(f'fits={fits_path}')
    print(f'peak_jy_per_beam={peak_val:.6g} bg={bg_med:.6g} sigma={sigma:.6g} peak_snr={peak_snr:.2f}')
    print(f'peak_pixel=({peak_x},{peak_y}) center=({cx:.1f},{cy:.1f}) offset_arcsec={offset_arcsec:.2f}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
