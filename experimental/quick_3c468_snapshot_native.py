#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from casatasks import exportfits, tclean
from casatools import table

MS = Path('casa_out/3c468.1/3c468.1.ms')
OUTDIR = Path('casa_out/3c468.1_snapshot_test_ms')
SEED = 20260506


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


def main() -> int:
    if not MS.exists():
        raise FileNotFoundError(f'MS not found: {MS}')

    OUTDIR.mkdir(parents=True, exist_ok=True)

    tb = table()
    tb.open(str(MS))
    times = np.unique(np.round(np.asarray(tb.getcol('TIME'), dtype=float), 6))
    tb.close()

    if len(times) < 2:
        raise RuntimeError('Not enough unique integrations in MS')

    cadence = float(np.median(np.diff(times)))
    random.seed(SEED)
    idx = random.randrange(len(times))
    t = float(times[idx])
    half = max(0.1, 0.45 * cadence)

    t0 = Time((t - half) / 86400.0, format='mjd', scale='utc').to_datetime()
    t1 = Time((t + half) / 86400.0, format='mjd', scale='utc').to_datetime()
    timerange = f"{t0.strftime('%Y/%m/%d/%H:%M:%S.%f')[:-3]}~{t1.strftime('%Y/%m/%d/%H:%M:%S.%f')[:-3]}"

    base = OUTDIR / '3c468_snapshot_native'
    cleanup_prefix(base)

    tclean(
        vis=str(MS),
        imagename=str(base),
        timerange=timerange,
        phasecenter='',
        spw='',
        imsize=[1024, 1024],
        cell='2arcsec',
        specmode='mfs',
        deconvolver='hogbom',
        niter=800,
        threshold='5mJy',
        weighting='briggs',
        robust=0.0,
        stokes='I',
        uvrange='',
        gridder='standard',
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

    print(f'random_index={idx+1}/{len(times)} cadence_sec={cadence:.3f}')
    print(f'timerange={timerange}')
    print(f'fits={fits_path}')
    print(f'peak_jy_per_beam={peak_val:.6g} bg={bg_med:.6g} sigma={sigma:.6g} peak_snr={peak_snr:.2f}')
    print(f'peak_pixel=({peak_x},{peak_y}) center=({cx:.1f},{cy:.1f}) offset_arcsec={offset_arcsec:.2f}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
