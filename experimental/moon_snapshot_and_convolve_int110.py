#!/usr/bin/env python3
"""Image the full Moon scan (all integrations) with tclean(phasecenter='MOON') and produce convolved versions."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.stats import mad_std
from casatasks import exportfits, tclean
from casatools import table

MS        = Path('casa_out/moon/moon0520.ms')
OUTDIR    = Path('casa_out/moon_fullscan')
CELL_ARCSEC = 2.0
MOON_FWHMS_ARCMIN = [15, 20, 30]


def cleanup_prefix(prefix: Path) -> None:
    for p in prefix.parent.glob(prefix.name + '.*'):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def robust_sigma(arr: np.ndarray) -> float:
    return float(mad_std(arr, ignore_nan=True))


def fwhm_to_sigma_pix(fwhm_arcmin: float) -> float:
    return (fwhm_arcmin * 60.0 / CELL_ARCSEC) / (2.0 * np.sqrt(2.0 * np.log(2.0)))


# ── 1.  report scan extent ──────────────────────────────────────────────────
tb = table()
tb.open(str(MS))
times = np.unique(np.round(np.asarray(tb.getcol('TIME'), dtype=float), 6))
tb.close()

print(f"Total integrations: {len(times)}")
print(f"Scan duration     : {(times[-1]-times[0])/60:.1f} min")
print("Imaging full scan — no timerange filter")

# ── 2.  tclean ────────────────────────────────────────────────────────────────
OUTDIR.mkdir(parents=True, exist_ok=True)
base = OUTDIR / 'moon_fullscan_native'
cleanup_prefix(base)

tclean(
    vis=str(MS),
    imagename=str(base),
    phasecenter='MOON',
    timerange='',          # all integrations
    spw='',
    imsize=[4096, 4096],
    cell='2arcsec',
    specmode='mfs',
    deconvolver='multiscale',
    scales=[0, 5, 15, 30],     # pixels: point + compact + Moon-disk scales
    niter=800,
    threshold='5mJy',
    weighting='briggs',
    robust=0.5,
    stokes='I',
    uvrange='',
    gridder='wproject',
    wprojplanes=-1,             # auto-calculate w-planes
    pblimit=-1,
    pbcor=False,
    interactive=False,
    calcpsf=True,
    calcres=True,
)

fits_native = str(base) + '.image.fits'
exportfits(imagename=str(base) + '.image', fitsimage=fits_native, overwrite=True)
print(f"native fits → {fits_native}")

# ── 3.  stats on native image ─────────────────────────────────────────────────
with fits.open(fits_native) as hdul:
    header = hdul[0].header.copy()
    data   = hdul[0].data.squeeze().astype(np.float32)

ny, nx = data.shape
cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
yy, xx = np.ogrid[:ny, :nx]
rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)
bg      = data[(rr >= 200) & (rr <= 450)]
bg_med  = float(np.nanmedian(bg))
sigma   = robust_sigma(bg)
py, px  = map(int, np.unravel_index(np.nanargmax(data), data.shape))
peak    = float(data[py, px])
snr     = (peak - bg_med) / (sigma + 1e-12)
offset  = float(np.hypot(px - cx, py - cy)) * CELL_ARCSEC

print(f"native  peak={peak:.4f} Jy/b  rms={sigma:.4f}  SNR={snr:.1f}"
      f"  peak_pixel=({px},{py})  offset={offset:.0f}\" ({offset/60:.1f}')")

# ── 4.  convolutions ──────────────────────────────────────────────────────────
edge = 200
print()
for fwhm_am in MOON_FWHMS_ARCMIN:
    kern = Gaussian2DKernel(x_stddev=fwhm_to_sigma_pix(fwhm_am))
    conv = convolve_fft(data, kern, normalize_kernel=True,
                        allow_huge=True, nan_treatment='fill', fill_value=0.0
                        ).astype(np.float32)

    bg_c   = np.concatenate([conv[:edge,:].ravel(), conv[-edge:,:].ravel(),
                              conv[:,:edge].ravel(), conv[:,-edge:].ravel()])
    sig_c  = robust_sigma(bg_c)
    med_c  = float(np.nanmedian(bg_c))
    py_c, px_c = map(int, np.unravel_index(np.argmax(conv), conv.shape))
    pk_c   = float(conv[py_c, px_c])
    snr_c  = (pk_c - med_c) / (sig_c + 1e-12)
    off_c  = float(np.hypot(px_c - cx, py_c - cy)) * CELL_ARCSEC

    out_fits = OUTDIR / f'moon_fullscan_conv{fwhm_am:02d}arcmin.fits'
    fits.writeto(out_fits,
                 conv[np.newaxis, np.newaxis, :, :],
                 header=header, overwrite=True)

    print(f"FWHM={fwhm_am:2d}'  SNR={snr_c:.1f}  peak=({px_c},{py_c})"
          f"  offset={off_c:.0f}\" ({off_c/60:.1f}')  → {out_fits.name}")

print("\nDone. Load in DS9:")
print(f"  {OUTDIR}/moon_fullscan_native.image.fits")
for fwhm_am in MOON_FWHMS_ARCMIN:
    print(f"  {OUTDIR}/moon_fullscan_conv{fwhm_am:02d}arcmin.fits")
