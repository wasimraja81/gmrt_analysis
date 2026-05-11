"""
Convolve the Moon snapshot dirty image with a Moon-sized circular Gaussian
and report the peak SNR + position.

Moon angular diameter at 150 MHz ≈ 30 arcmin (thermal emission disk).
We try FWHM = 15', 20', 30' to bracket the true extent.
Cell size = 2 arcsec.
"""

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.stats import mad_std

FITS_IN = Path("casa_out/moon_snapshot_test_ms/moon_snapshot_native.image.fits")
OUTDIR  = Path("casa_out/moon_snapshot_test_ms")
CELL_ARCSEC = 2.0

MOON_FWHM_ARCMIN_LIST = [15, 20, 30]   # trial convolution widths

def robust_sigma(arr):
    return mad_std(arr, ignore_nan=True)

def fwhm_to_sigma_pixels(fwhm_arcmin, cell_arcsec):
    fwhm_pix = (fwhm_arcmin * 60.0) / cell_arcsec
    return fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

with fits.open(FITS_IN) as hdul:
    header = hdul[0].header.copy()
    data   = hdul[0].data.squeeze().astype(np.float32)   # (ny, nx)

ny, nx = data.shape
cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
print(f"Image size: {nx}×{ny},  cell={CELL_ARCSEC}\", center=({cx:.1f},{cy:.1f})")
print(f"Input  peak={data.max():.4f}  sigma={robust_sigma(data):.4f} Jy/beam")
print()

best = {}

for fwhm_am in MOON_FWHM_ARCMIN_LIST:
    sigma_pix = fwhm_to_sigma_pixels(fwhm_am, CELL_ARCSEC)
    kernel = Gaussian2DKernel(x_stddev=sigma_pix, y_stddev=sigma_pix)

    conv = convolve_fft(data, kernel, normalize_kernel=True,
                        allow_huge=True, nan_treatment='fill', fill_value=0.0)
    conv = conv.astype(np.float32)

    # noise from corners (avoid Moon disk region near center)
    edge = 200
    bg_pixels = np.concatenate([
        conv[:edge, :].ravel(),
        conv[-edge:, :].ravel(),
        conv[:, :edge].ravel(),
        conv[:, -edge:].ravel(),
    ])
    sigma  = robust_sigma(bg_pixels)
    bg_med = np.nanmedian(bg_pixels)
    peak_val = conv.max()
    py, px   = np.unravel_index(np.argmax(conv), conv.shape)
    snr      = (peak_val - bg_med) / (sigma + 1e-12)

    dx = (px - cx) * CELL_ARCSEC
    dy = (py - cy) * CELL_ARCSEC
    offset_arcsec = np.sqrt(dx**2 + dy**2)

    print(f"FWHM={fwhm_am:2d}'  sigma_pix={sigma_pix:.1f}"
          f"  peak={peak_val:.4f}  rms={sigma:.4f}  SNR={snr:.1f}"
          f"  peak_pixel=({px},{py})  offset={offset_arcsec:.0f}\" ({offset_arcsec/60:.1f}')")

    # save convolved FITS
    out_fits = OUTDIR / f"moon_snapshot_conv{fwhm_am:02d}arcmin.fits"
    hdr_out = header.copy()
    hdr_out["HISTORY"] = f"Convolved with Gaussian FWHM={fwhm_am} arcmin"
    fits.writeto(out_fits, conv[np.newaxis, np.newaxis, :, :], header=hdr_out,
                 overwrite=True)

    best[fwhm_am] = dict(snr=snr, offset=offset_arcsec, px=px, py=py,
                         peak=peak_val, rms=sigma, fits=out_fits)

print()
# Pick the FWHM with highest SNR
best_fwhm = max(best, key=lambda k: best[k]['snr'])
b = best[best_fwhm]
print(f"Best FWHM: {best_fwhm}' → SNR={b['snr']:.1f}, "
      f"peak at ({b['px']},{b['py']}), offset={b['offset']:.0f}\" ({b['offset']/60:.1f}')")
print(f"Saved: {b['fits']}")
