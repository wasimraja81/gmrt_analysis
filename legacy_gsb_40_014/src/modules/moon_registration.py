from __future__ import annotations

from typing import Optional

import numpy as np
from astropy.time import Time


def representative_jd_from_header(header) -> float:
    """Return representative JD using priority: JDMID > JDAVG > MJD-AVG > DATE-AVG > DATE-OBS."""
    if header.get('JDMID') is not None:
        return float(header['JDMID'])
    if header.get('JDAVG') is not None:
        return float(header['JDAVG'])
    if header.get('MJD-AVG') is not None:
        return float(Time(float(header['MJD-AVG']), format='mjd', scale='utc').jd)  # type: ignore[arg-type]
    if header.get('DATE-AVG'):
        return float(Time(header['DATE-AVG'], format='isot', scale='utc').jd)  # type: ignore[arg-type]
    if header.get('DATE-OBS'):
        return float(Time(header['DATE-OBS'], format='isot', scale='utc').jd)  # type: ignore[arg-type]
    raise ValueError('No usable time keyword found (JDMID/JDAVG/MJD-AVG/DATE-AVG/DATE-OBS).')


def choose_reference_index_mean_jd(headers: list) -> tuple[int, np.ndarray, float]:
    """Choose reference frame index nearest mean JD from a list of FITS headers."""
    jd_arr = np.array([representative_jd_from_header(hdr) for hdr in headers], dtype=np.float64)
    jd_mean = float(np.mean(jd_arr))
    ref_idx = int(np.argmin(np.abs(jd_arr - jd_mean)))
    return ref_idx, jd_arr, jd_mean


def phase_cross_correlate(ref: np.ndarray,
                          mov: np.ndarray,
                          upsample: int = 10) -> tuple[float, float, str]:
    """Return (row_shift, col_shift, backend) to align mov onto ref."""
    try:
        from skimage.registration import phase_cross_correlation  # type: ignore

        shift, _, _ = phase_cross_correlation(
            ref,
            mov,
            upsample_factor=upsample,
            normalization=None,
        )
        return float(shift[0]), float(shift[1]), 'skimage.phase_cross_correlation'
    except ImportError:
        f_ref = np.fft.fft2(ref)
        f_mov = np.fft.fft2(mov)
        cross_power = f_ref * np.conj(f_mov)
        cc = np.fft.ifft2(cross_power / (np.abs(cross_power) + 1e-30)).real
        idx = np.unravel_index(np.argmax(cc), cc.shape)
        dr = idx[0] if idx[0] < ref.shape[0] // 2 else idx[0] - ref.shape[0]
        dc = idx[1] if idx[1] < ref.shape[1] // 2 else idx[1] - ref.shape[1]
        return float(dr), float(dc), 'numpy.phase_only_fallback'


def apply_shift_fourier(img: np.ndarray, dr: float, dc: float) -> np.ndarray:
    """Apply sub-pixel shift (dr, dc) via Fourier phase ramp."""
    if dr == 0.0 and dc == 0.0:
        return img
    ny, nx = img.shape
    fr = np.fft.fftfreq(ny)[:, np.newaxis]
    fc = np.fft.fftfreq(nx)[np.newaxis, :]
    phase_ramp = np.exp(-2j * np.pi * (dr * fr + dc * fc))
    shifted = np.fft.ifft2(np.fft.fft2(img) * phase_ramp).real
    return shifted.astype(np.float32)


def align_images(images: list[np.ndarray],
                 ref_idx: int,
                 registration_mode: str,
                 registration_method: str = 'phase-correlation',
                 upsample: int = 10) -> tuple[list[np.ndarray], list[tuple[float, float]], str]:
    """Align images to reference and return (aligned_images, shifts, backend_name)."""
    if ref_idx < 0 or ref_idx >= len(images):
        raise IndexError(f'ref_idx {ref_idx} out of range for {len(images)} images')

    ref_img = images[ref_idx]
    aligned: list[np.ndarray] = []
    shifts: list[tuple[float, float]] = []
    backend = 'none'

    for i, img in enumerate(images):
        if registration_mode == 'none' or i == ref_idx:
            dr, dc = 0.0, 0.0
        else:
            if registration_method != 'phase-correlation':
                raise ValueError(f'Unsupported registration method: {registration_method}')
            dr, dc, backend = phase_cross_correlate(ref_img, img, upsample=upsample)

        shifts.append((dr, dc))
        aligned.append(apply_shift_fourier(img, dr, dc))

    if registration_mode == 'none':
        backend = 'none'

    return aligned, shifts, backend


def off_moon_rms(image: np.ndarray, mask_radius_pix: float,
                 moon_center: Optional[tuple[float, float]] = None) -> float:
    """Robust off-moon RMS estimate using MAD outside circular moon mask.

    Parameters
    ----------
    image : 2-D array
    mask_radius_pix : exclusion radius in pixels
    moon_center : (cy, cx) pixel coords of moon; defaults to image centre.
        For no_phasecenter data the moon is NOT at image centre -- supply the
        centroid of the reference aligned frame so the mask is placed correctly.
    """
    ny, nx = image.shape
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    if moon_center is None:
        cy = (ny - 1) / 2.0
        cx = (nx - 1) / 2.0
    else:
        cy, cx = float(moon_center[0]), float(moon_center[1])
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    vals = image[rr > float(mask_radius_pix)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float('nan')
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    robust_rms = 1.4826 * mad
    if not np.isfinite(robust_rms) or robust_rms <= 0:
        robust_rms = float(np.sqrt(np.mean(vals**2)))
    return robust_rms
